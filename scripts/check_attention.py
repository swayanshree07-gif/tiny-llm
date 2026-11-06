"""Re-runnable check of causal self-attention on the trained model (PLAN Day 8).

The unit tests (tests/test_attention.py) use small random modules. This script repeats the checks
on the real checkpoint (default ckpt/base) and real validation text, where activations are
large and peaked. That matters: agreement on random weights does not prove agreement on trained ones.

  1. manual vs sdpa, one attention module (layer 0 and the last layer) on real hidden states: fp32
  2. manual vs sdpa, whole model: fp32 logits close, val loss equal; bf16 autocast (CUDA) loss close
  3. causality, whole model, both paths: changing tokens at positions >= t leaves logits < t unchanged
  4. causality by gradient: d(logit at t) / d(input embedding at s) == 0 for every s > t, != 0 for s <= t
  5. masked multi-token chunk after position 0 (the KV-cache path, start > 0 and T > 1):
     manual == sdpa == the uncached full forward
  6. (CUDA) timing and peak memory of manual vs sdpa for a training-sized batch (info only)

Run: .venv/bin/python scripts/check_attention.py [--ckpt ckpt/base] [--random] [--cpu]
     [--tok-dir data/tok] [--batch-size 8]
`--random` uses a freshly initialised small model and random tokens (no files needed; used by pytest).
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402


def build(cfg: dict, state: dict | None, impl: str, device) -> GPT:
    m = GPT(GPTConfig(**{**cfg, "attn_impl": impl}))
    if state is not None:
        m.load_state_dict(state)
    return m.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/base")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--random", action="store_true", help="random small model + random tokens")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 checks really are fp32
    torch.backends.cudnn.allow_tf32 = False
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    torch.manual_seed(args.seed)
    if args.random:
        cfg = GPTConfig(vocab_size=512, block_size=64, n_layers=2, n_heads=4, d_model=64).to_dict()
        state = GPT(GPTConfig(**cfg)).state_dict()
        T, B = cfg["block_size"], args.batch_size
        x = torch.randint(0, cfg["vocab_size"], (B, T + 1))
        print("model: random", cfg)
    else:
        ck = latest_checkpoint(args.ckpt) if args.ckpt.is_dir() else args.ckpt
        if ck is None:
            print(f"no checkpoint in {args.ckpt}")
            return 1
        st = load_checkpoint(ck)
        cfg, state = st["model_config"], st["model"]
        T, B = cfg["block_size"], args.batch_size
        ids = np.memmap(args.tok_dir / "valid.bin", dtype=np.uint16, mode="r")
        rng = np.random.default_rng(args.seed)
        starts = rng.integers(0, len(ids) - T - 2, size=B)
        x = torch.from_numpy(np.stack([ids[s:s + T + 1] for s in starts]).astype(np.int64))
        print(f"model: {ck} (step {st['step']}), {B} val windows of {T}")
    x, y = x[:, :-1].to(device), x[:, 1:].to(device)
    man, sdp = build(cfg, state, "manual", device), build(cfg, state, "sdpa", device)
    L = cfg["n_layers"]

    with torch.no_grad():
        # 1. single module on real hidden states (input of layer 0 and of the last layer)
        hs = []
        h = man.tok_emb(x)
        if man.pos_emb is not None:
            h = h + man.pos_emb(torch.arange(T, device=device))
        rope = None if man.pos_emb is not None else (man.rope_cos, man.rope_sin)
        for i, blk in enumerate(man.blocks):
            hs.append(blk.ln1(h))
            h = blk(h, rope)
        for i in (0, L - 1):
            a = man.blocks[i].attn(hs[i], rope)
            b = sdp.blocks[i].attn(hs[i], rope)
            d = (a - b).abs().max().item()
            check(f"layer {i} attention manual == sdpa (fp32)", d < 1e-4 * max(1.0, a.abs().max().item()),
                  f"max|diff| {d:.2e}, max|y| {a.abs().max().item():.2f}")

        # 2. whole model
        lm, loss_m = man(x, y)
        ls, loss_s = sdp(x, y)
        d = (lm - ls).abs().max().item()
        check("model logits manual == sdpa (fp32)", d < 1e-3, f"max|diff| {d:.2e}")
        check("model loss manual == sdpa (fp32)", abs(loss_m.item() - loss_s.item()) < 1e-5,
              f"{loss_m.item():.6f} vs {loss_s.item():.6f}")
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, bm = man(x, y)
                _, bs = sdp(x, y)
            check("model loss manual ~ sdpa (bf16 autocast)", abs(bm.item() - bs.item()) < 1e-2,
                  f"{bm.item():.4f} vs {bs.item():.4f} (fp32 {loss_s.item():.4f})")

        # 3. causality by perturbation
        t = T // 2
        x2 = x.clone()
        x2[:, t:] = torch.randint(0, cfg["vocab_size"], x2[:, t:].shape, device=device)
        for name, m, l1 in (("manual", man, lm), ("sdpa", sdp, ls)):
            l2, _ = m(x2)
            past = (l1[:, :t] - l2[:, :t]).abs().max().item()
            fut = (l1[:, t:] - l2[:, t:]).abs().max().item()
            check(f"{name}: future tokens don't change past logits", past == 0.0 and fut > 0,
                  f"past max|diff| {past:.1e}, future {fut:.2f}")

    # 4. causality by gradient: Jacobian row of one logit w.r.t. all input embeddings
    for name, m in (("manual", man), ("sdpa", sdp)):
        emb = m.tok_emb(x[:1]).detach().requires_grad_(True)
        captured = {}
        hook = m.tok_emb.register_forward_hook(lambda mod, inp, out: emb)
        logits, _ = m(x[:1])
        hook.remove()
        tq = T // 3
        logits[0, tq].logsumexp(-1).backward()
        g = emb.grad[0].abs().sum(-1)  # (T,)
        captured["after"] = g[tq + 1:].max().item()
        captured["before"] = int((g[:tq + 1] > 0).sum().item())
        check(f"{name}: grad of logit {tq} is zero for all later inputs, nonzero for earlier",
              captured["after"] == 0.0 and captured["before"] == tq + 1,
              f"max later {captured['after']:.1e}, nonzero earlier {captured['before']}/{tq + 1}")

    # 5. chunk after position 0 through the KV cache (the explicit-mask path)
    with torch.no_grad():
        p = T // 4
        outs = {}
        for name, m in (("manual", man), ("sdpa", sdp)):
            cache = m.new_kv_cache(B)
            m(x[:, :p], kv_cache=cache)
            outs[name], _ = m(x[:, p:], kv_cache=cache)
        full = ls[:, p:]
        d1 = (outs["manual"] - outs["sdpa"]).abs().max().item()
        d2 = (outs["sdpa"] - full).abs().max().item()
        check("cached chunk (start>0, T>1): manual == sdpa", d1 < 1e-3, f"max|diff| {d1:.2e}")
        check("cached chunk == uncached full forward", d2 < 1e-3, f"max|diff| {d2:.2e}")

    # 6. speed / memory (info only)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        xb = x.repeat(max(1, 32 // B), 1)
        for name, m in (("manual", man), ("sdpa", sdp)):
            m.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(6):
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = m(xb)
                    loss = F.cross_entropy(lg.float().flatten(0, 1), xb.flatten())
                loss.backward()
                m.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            tok_s = xb.numel() / float(np.median(times[1:]))
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[INFO] {name}: fwd+bwd bf16 batch {tuple(xb.shape)}: {tok_s/1e3:.1f}k tok/s, peak {mem:.2f} GB")
            m.eval()

    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
