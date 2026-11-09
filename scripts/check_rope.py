"""Re-runnable check of RoPE (rotary position embeddings) and the learned/rope switch (PLAN Day 10).

The unit tests (tests/test_rope.py) check the relative-position property on one random 16-dim pair.
This script checks RoPE exhaustively at the real head size (configs/base.json: head_dim 64, block 256)
and on the trained checkpoint (default ckpt/base, pos = "rope") with real validation text.

  1. tables: rope_cache shapes (T, D/2); angle == pos * base^(-2i/D) (fp64 reference); cos^2 + sin^2 == 1
  2. apply_rope == complex reference: pair (x[2i], x[2i+1]) as x[2i] + j x[2i+1], times e^(j pos theta_i)
  3. rotation: norm preserved per vector, position 0 is the identity, bf16 input stays bf16 and close
  4. relative-position property, every (m, n) pair in the block: <R_m q, R_n k> depends only on m - n
     (all diagonals of the score matrix constant) and equals the closed form sum_i Re(q_i conj(k_i) e^(j(m-n)theta_i))
  5. ... and it is not trivially constant: scores at different offsets differ
  6. trained model, real q/k of layer 0 and the last layer: attention scores with positions shifted by s
     (s = 1, 37, block_size) equal the unshifted scores
  7. trained model, whole forward: shifting every position by s leaves logits/loss unchanged (a RoPE model
     only sees relative positions); a learned-position model (same weights, random pos_emb) does change
  8. causality still holds with RoPE: future tokens don't change past logits, bit-exact, also when shifted
  9. config switch: pos="rope" has rope buffers and no pos_emb, pos="learned" the reverse; rope buffers are
     non-persistent (not in the state_dict), so the checkpoint loads strictly; KV-cache greedy decode == uncached
 10. (info) long-range decay of |<R_m q, R_n q>| for q = all-ones as distance grows

Run: .venv/bin/python scripts/check_rope.py [--config configs/base.json] [--ckpt ckpt/base]
     [--tok-dir data/tok] [--random] [--cpu] [--batch-size 8]
`--random` uses a small, freshly initialised model and random tokens (no files needed; used by pytest).
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from tinyllm.generate import generate  # noqa: E402
from tinyllm.model import GPT, GPTConfig, apply_rope, rope_cache  # noqa: E402


def ref_angles(T: int, D: int, base: float) -> torch.Tensor:
    """fp64 angles (T, D/2): pos * base^(-2i/D)."""
    inv = base ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    return torch.arange(T, dtype=torch.float64)[:, None] * inv[None, :]


def complex_rope(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Reference RoPE in fp64 via complex multiplication. x (..., T, D), ang (T, D/2)."""
    xc = torch.complex(x[..., 0::2].double(), x[..., 1::2].double())
    yc = xc * torch.polar(torch.ones_like(ang), ang)
    return torch.stack((yc.real, yc.imag), dim=-1).flatten(-2)


def score_matrix(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """S[m, n] = <R_m q, R_n k> for single vectors q, k (D,), all positions of the tables."""
    T, D = cos.shape[0], q.shape[0]
    qr = apply_rope(q.expand(T, D).contiguous(), cos, sin).double()
    kr = apply_rope(k.expand(T, D).contiguous(), cos, sin).double()
    return qr @ kr.T


def shifted(m: GPT, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables for positions s .. s + block_size - 1."""
    cos, sin = rope_cache(m.cfg.block_size + s, m.cfg.d_model // m.cfg.n_heads, m.cfg.rope_base,
                          device=m.rope_cos.device)
    return cos[s:].contiguous(), sin[s:].contiguous()


def forward_shifted(m: GPT, x: torch.Tensor, y: torch.Tensor | None, s: int):
    """Whole-model forward with every position moved by s (swaps the rope buffers temporarily)."""
    keep = (m.rope_cos, m.rope_sin)
    m.rope_cos, m.rope_sin = shifted(m, s)
    try:
        return m(x, y)
    finally:
        m.rope_cos, m.rope_sin = keep


def layer_qk(m: GPT, x: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-RoPE q, k (B, H, T, D) of `layer` on input ids x (positions 0..T-1)."""
    B, T = x.shape
    h = m.tok_emb(x)
    rope = (m.rope_cos, m.rope_sin)
    for blk in m.blocks[:layer]:
        h = blk(h, rope)
    a = m.blocks[layer].attn
    q, k, _ = a.qkv(m.blocks[layer].ln1(h)).split(m.cfg.d_model, dim=2)
    q, k = (t.view(B, T, a.n_heads, a.head_dim).transpose(1, 2) for t in (q, k))
    return q, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/base")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--random", action="store_true", help="small fresh model + random tokens")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    torch.manual_seed(args.seed)
    if args.random:
        cfg = GPTConfig(vocab_size=512, block_size=64, n_layers=2, n_heads=4, d_model=64)
        m = GPT(cfg)
        B, T = args.batch_size, cfg.block_size
        xy = torch.randint(0, cfg.vocab_size, (B, T + 1))
        print("model: random", cfg.to_dict())
    else:
        ck = latest_checkpoint(args.ckpt) if args.ckpt.is_dir() else args.ckpt
        if ck is None:
            print(f"[FAIL] no checkpoint in {args.ckpt}\nRESULT FAIL")
            return 1
        st = load_checkpoint(ck)
        cfg = GPTConfig(**st["model_config"])
        m = GPT(cfg)
        res = m.load_state_dict(st["model"], strict=True)
        check(f"checkpoint {ck.name} (step {st['step']}) loads strictly with pos={cfg.pos}",
              not res.missing_keys and not res.unexpected_keys and cfg.pos == "rope")
        cfg_file = GPTConfig(**json.loads(args.config.read_text())["model"])
        check(f"{args.config.name} model config == checkpoint config", cfg_file == cfg)
        B, T = args.batch_size, cfg.block_size
        ids = np.memmap(args.tok_dir / "valid.bin", dtype=np.uint16, mode="r")
        starts = np.random.default_rng(args.seed).integers(0, len(ids) - T - 2, size=B)
        xy = torch.from_numpy(np.stack([ids[s:s + T + 1] for s in starts]).astype(np.int64))
        print(f"model: {ck}, {cfg.to_dict()}; {B} val windows of {T}")
    m = m.to(device).eval()
    x, y = xy[:, :-1].to(device), xy[:, 1:].to(device)
    D, base = cfg.d_model // cfg.n_heads, cfg.rope_base

    # 1. tables
    cos, sin = rope_cache(T, D, base)
    ang = ref_angles(T, D, base)
    check("rope_cache shapes (T, D/2)", tuple(cos.shape) == tuple(sin.shape) == (T, D // 2), f"{tuple(cos.shape)}")
    e_cos = (cos.double() - ang.cos()).abs().max().item()
    e_sin = (sin.double() - ang.sin()).abs().max().item()
    check("cos/sin == fp64 reference angles pos * base^(-2i/D)", max(e_cos, e_sin) < 1e-4,
          f"max err {max(e_cos, e_sin):.1e} (theta_0 = 1, theta_last = {base ** (-(D - 2) / D):.2e})")
    unit = (cos.double() ** 2 + sin.double() ** 2 - 1).abs().max().item()
    check("cos^2 + sin^2 == 1", unit < 1e-6, f"{unit:.1e}")
    check("model rope buffers == rope_cache(block_size)", torch.equal(m.rope_cos.cpu(), cos)
          and torch.equal(m.rope_sin.cpu(), sin))

    # 2. apply_rope == complex multiplication
    g = torch.Generator().manual_seed(args.seed)
    v = torch.randn(3, cfg.n_heads, T, D, generator=g)
    err = (apply_rope(v, cos, sin).double() - complex_rope(v, ang)).abs().max().item()
    check("apply_rope == complex reference (interleaved pairs)", err < 1e-4, f"max err {err:.1e}")

    # 3. rotation properties
    r = apply_rope(v, cos, sin)
    nerr = ((r.norm(dim=-1) - v.norm(dim=-1)).abs() / v.norm(dim=-1)).max().item()
    check("norm preserved per vector", nerr < 1e-5, f"rel err {nerr:.1e}")
    check("position 0 is the identity", torch.equal(r[..., 0, :], v[..., 0, :]))
    rb = apply_rope(v.bfloat16(), cos, sin)
    berr = (rb.float() - r).abs().max().item() / r.abs().max().item()
    check("bf16 input -> bf16 output, close to fp32", rb.dtype == torch.bfloat16 and berr < 1e-2, f"rel {berr:.1e}")

    # 4./5. relative-position property over every (m, n) pair
    q, k = torch.randn(D, generator=g), torch.randn(D, generator=g)
    S = score_matrix(q, k, cos, sin)
    scale = q.norm().item() * k.norm().item()
    worst, closed = 0.0, 0.0
    qc = torch.complex(q[0::2].double(), q[1::2].double())
    kc = torch.complex(k[0::2].double(), k[1::2].double())
    theta = ang[1]  # base^(-2i/D)
    diag_means = []
    for off in range(-(T - 1), T):
        d = torch.diagonal(S, offset=-off)  # entries with m - n == off
        worst = max(worst, (d - d.mean()).abs().max().item())
        cf = (qc * kc.conj() * torch.polar(torch.ones_like(theta), off * theta)).real.sum().item()
        closed = max(closed, (d - cf).abs().max().item())
        diag_means.append(d.mean().item())
    check(f"<R_m q, R_n k> depends only on m - n (all {T}x{T} pairs, head_dim {D})", worst / scale < 1e-5,
          f"max spread {worst:.1e} (|q||k| = {scale:.1f})")
    check("score == closed form sum Re(q_i conj(k_i) e^(j(m-n)theta_i))", closed / scale < 1e-5,
          f"max err {closed:.1e}")
    spread = float(np.std(diag_means))
    check("scores vary with the offset (RoPE is not a no-op)", spread / scale > 1e-2,
          f"std over offsets {spread:.3f}, at offset 0/1/10: {diag_means[T - 1]:.3f} "
          f"{diag_means[T]:.3f} {diag_means[T + 9]:.3f}")

    with torch.no_grad():
        # 6. real q/k: scores invariant to a common position shift
        for layer in sorted({0, cfg.n_layers - 1}):
            qq, kk = layer_qk(m, x, layer)
            c0, s0 = m.rope_cos[:T], m.rope_sin[:T]
            att0 = apply_rope(qq, c0, s0).float() @ apply_rope(kk, c0, s0).float().transpose(-2, -1)
            raw = qq.float() @ kk.float().transpose(-2, -1)
            sc = att0.abs().max().item()
            errs = []
            for s in (1, 37, cfg.block_size):
                cs, ss = shifted(m, s)
                att = apply_rope(qq, cs[:T], ss[:T]).float() @ apply_rope(kk, cs[:T], ss[:T]).float().transpose(-2, -1)
                errs.append((att - att0).abs().max().item() / sc)
            diff_raw = (raw - att0).abs().max().item() / sc
            check(f"layer {layer}: real q.k scores unchanged by position shift s=1/37/{cfg.block_size}",
                  max(errs) < 1e-4 and diff_raw > 1e-2,
                  "rel err " + " ".join(f"{e:.1e}" for e in errs) + f"; vs no rope {diff_raw:.2f} (max |score| {sc:.1f})")

        # 7. whole model translation invariance
        logits, loss = m(x, y)
        lscale = logits.abs().max().item()
        for s in (1, 37, cfg.block_size):
            ls, lo = forward_shifted(m, x, y, s)
            e = (ls - logits).abs().max().item()
            check(f"rope model: positions shifted by {s} -> same logits/loss", e / lscale < 1e-4
                  and abs(lo.item() - loss.item()) < 1e-4,
                  f"max logit diff {e:.1e} (max |logit| {lscale:.1f}); loss {loss.item():.5f} vs {lo.item():.5f}")
        lcfg = GPTConfig(**{**cfg.to_dict(), "pos": "learned"})
        ml = GPT(lcfg).to(device).eval()
        ml.load_state_dict({**m.state_dict(), "pos_emb.weight": ml.pos_emb.weight.detach().clone()}, strict=True)
        la, _ = ml(x)
        keep = ml.pos_emb.weight.detach().clone()
        ml.pos_emb.weight.copy_(torch.roll(keep, shifts=-1, dims=0))  # table row t now holds position t+1
        lb, _ = ml(x)
        ml.pos_emb.weight.copy_(keep)
        ld = (la - lb).abs().max().item()
        check("learned-position model (same weights): shifting positions by 1 changes logits", ld > 1e-2,
              f"max logit diff {ld:.3f}")

        # 8. causality with rope
        t = T // 2
        x2 = x.clone()
        x2[:, t:] = torch.randint(0, cfg.vocab_size, x2[:, t:].shape, device=device)
        l2, _ = m(x2)
        past = (l2[:, :t] - logits[:, :t]).abs().max().item()
        fut = (l2[:, t:] - logits[:, t:]).abs().max().item()
        a37, _ = forward_shifted(m, x, None, 37)
        b37, _ = forward_shifted(m, x2, None, 37)
        past37 = (a37[:, :t] - b37[:, :t]).abs().max().item()
        check("rope causality: future tokens don't change past logits (also with shift 37)",
              past == 0.0 and past37 == 0.0 and fut > 0, f"past {past:.1e} / {past37:.1e}, future {fut:.3f}")

        # 9. config switch, buffers, cache
        mr = GPT(GPTConfig(**{**cfg.to_dict(), "pos": "rope"}))
        check("pos switch: rope -> no pos_emb + rope buffers; learned -> pos_emb + no rope buffers",
              mr.pos_emb is None and hasattr(mr, "rope_cos") and ml.pos_emb is not None
              and not hasattr(ml, "rope_cos"))
        check("rope buffers are non-persistent (not in state_dict)",
              not any("rope" in k_ for k_ in m.state_dict()))
        prompt = x[:1, :8]
        g1 = generate(m, prompt, max_new_tokens=32, temperature=0.0, use_cache=True)
        g2 = generate(m, prompt, max_new_tokens=32, temperature=0.0, use_cache=False)
        check("rope + KV cache: greedy decode == uncached (32 tokens)", torch.equal(g1, g2))

    # 10. info: long-range decay
    ones = torch.ones(D)
    So = score_matrix(ones, ones, *rope_cache(cfg.block_size * 4, D, base))[0]
    pts = [d_ for d_ in (0, 1, 4, 16, 64, 256, 1023) if d_ < So.numel()]
    print("[INFO] <R_0 1, R_d 1> / D for d = " + ", ".join(f"{d_}: {So[d_].item() / D:+.3f}" for d_ in pts))

    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
