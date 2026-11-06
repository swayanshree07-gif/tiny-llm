"""Re-runnable check of the full GPT model: MLP, Block, LayerNorm, init, param count (PLAN Day 9).

The unit tests (tests/test_model.py) use a tiny d=64 model with random targets. This script repeats the
checks at the real size (configs/base.json, d=512, 8 layers) on real validation text, where effects that
scale with d (e.g. the tied-embedding "self token" logit) would show up. Both position variants
(learned, rope) are built from the same config.

  1. param count == closed-form formula (per block 12 d^2 + 4 d, embeddings, final LN, learned positions)
  2. init: head tied to tok_emb (same storage); weight stds 0.02 / 0.02 / sqrt(2 L) for residual projections;
     LayerNorm weight 1, bias 0
  3. forward: logits (B, T, V) fp32 and finite; bf16 autocast finite (CUDA)
  4. loss at init on real val windows ~ ln(V) (|diff| < 0.15); near-uniform prediction (entropy ~ ln V)
  5. residual stream std per layer at init stays bounded (no blow-up / collapse)
  6. Block structure: with both residual projections zeroed a Block is the identity; MLP hidden = 4 d
  7. whole-model causality: changing tokens >= t leaves logits < t bit-identical (eval and train mode)
  8. one backward: every parameter gets a finite, non-zero gradient
  9. a few AdamW steps on one real batch lower the loss (the model can learn)
 10. trained checkpoint (unless --random): strict state_dict load, same param count, val loss far below ln V

Run: .venv/bin/python scripts/check_model.py [--config configs/base.json] [--ckpt ckpt/base]
     [--tok-dir data/tok] [--random] [--cpu] [--batch-size 8]
`--random` uses a small config and random tokens (no files needed; used by pytest).
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from tinyllm.model import GPT, Block, GPTConfig  # noqa: E402


def expected_params(c: GPTConfig) -> int:
    d, h = c.d_model, c.mlp_ratio * c.d_model
    per_block = 4 * d * d + 2 * d * h + 4 * d  # qkv + attn proj, fc + mlp proj, 2 LayerNorms (w + b)
    n = c.vocab_size * d + c.n_layers * per_block + 2 * d  # tied embedding/head, final LN
    if c.pos == "learned":
        n += c.block_size * d
    return n


def residual_stds(m: GPT, x: torch.Tensor) -> list[float]:
    """Std of the residual stream after the embedding and after each block (no KV cache)."""
    T = x.size(1)
    h = m.tok_emb(x)
    rope = None
    if m.pos_emb is not None:
        h = h + m.pos_emb(torch.arange(T, device=x.device))
    else:
        rope = (m.rope_cos, m.rope_sin)
    out = [h.std().item()]
    for blk in m.blocks:
        h = blk(h, rope)
        out.append(h.std().item())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "ckpt/base")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--random", action="store_true", help="small config + random tokens")
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
        base = GPTConfig(vocab_size=512, block_size=64, n_layers=2, n_heads=4, d_model=64)
        B, T = args.batch_size, base.block_size
        x = torch.randint(0, base.vocab_size, (B, T + 1))
        print("config: random", base.to_dict())
    else:
        base = GPTConfig(**json.loads(args.config.read_text())["model"])
        B, T = args.batch_size, base.block_size
        ids = np.memmap(args.tok_dir / "valid.bin", dtype=np.uint16, mode="r")
        starts = np.random.default_rng(args.seed).integers(0, len(ids) - T - 2, size=B)
        x = torch.from_numpy(np.stack([ids[s:s + T + 1] for s in starts]).astype(np.int64))
        print(f"config: {args.config.name} {base.to_dict()}; {B} val windows of {T}")
    x, y = x[:, :-1].to(device), x[:, 1:].to(device)
    V, L, d = base.vocab_size, base.n_layers, base.d_model
    lnV = math.log(V)

    for pos in ("learned", "rope"):
        cfg = GPTConfig(**{**base.to_dict(), "pos": pos, "dropout": 0.0})
        torch.manual_seed(args.seed)
        m = GPT(cfg).to(device).eval()
        tag = f"[{pos}]"

        # 1. param count
        n, exp = m.num_params(non_embedding=False), expected_params(cfg)
        check(f"{tag} param count == formula", n == exp, f"{n:,} vs {exp:,}")

        # 2. init statistics
        check(f"{tag} head tied to tok_emb", m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr())
        bad = []
        for name, p in m.named_parameters():
            if p.dim() == 2:
                target = 0.02 / math.sqrt(2 * L) if name.endswith("proj.weight") else 0.02
                if abs(p.std().item() / target - 1) > 0.05:
                    bad.append(f"{name} std {p.std().item():.4f} (want {target:.4f})")
            elif "ln" in name:
                want = 1.0 if name.endswith("weight") else 0.0
                if not torch.all(p == want):
                    bad.append(f"{name} != {want}")
        check(f"{tag} init stds / LayerNorm params as designed", not bad, "; ".join(bad[:3]))

        with torch.no_grad():
            # 3. forward
            logits, loss = m(x, y)
            check(f"{tag} logits shape/dtype/finite",
                  tuple(logits.shape) == (B, T, V) and logits.dtype == torch.float32
                  and bool(torch.isfinite(logits).all()), f"{tuple(logits.shape)} {logits.dtype}")
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lb, lossb = m(x, y)
                check(f"{tag} bf16 autocast forward finite", bool(torch.isfinite(lb.float()).all())
                      and math.isfinite(lossb.item()), f"loss {lossb.item():.4f}")

            # 4. init loss ~ ln V, near-uniform predictions
            lp = logits.log_softmax(-1)
            ent = -(lp.exp() * lp).sum(-1).mean().item()
            self_logit = (logits.gather(-1, x[..., None]).squeeze(-1) - logits.mean(-1)).mean().item()
            check(f"{tag} init loss ~ ln(V) on real tokens", abs(loss.item() - lnV) < 0.15,
                  f"{loss.item():.4f} vs ln({V}) = {lnV:.4f}; entropy {ent:.4f}; "
                  f"self-token logit +{self_logit:.3f}; logit std {logits.std().item():.3f}")
            check(f"{tag} init predictions near-uniform", abs(ent - lnV) < 0.15, f"entropy {ent:.4f}")

            # 5. residual stream at init
            stds = residual_stds(m, x)
            growth = stds[-1] / stds[0]
            check(f"{tag} residual stream std bounded at init", all(math.isfinite(s) for s in stds)
                  and 1.0 <= growth < 50, "per layer " + " ".join(f"{s:.3f}" for s in stds)
                  + f" (x{growth:.1f})")

            # 7. causality (eval mode; train mode below with dropout 0 must match too)
            t = T // 2
            x2 = x.clone()
            x2[:, t:] = torch.randint(0, V, x2[:, t:].shape, device=device)
            l2, _ = m(x2)
            past = (logits[:, :t] - l2[:, :t]).abs().max().item()
            fut = (logits[:, t:] - l2[:, t:]).abs().max().item()
            m.train()
            l3, _ = m(x2)
            m.eval()
            tr = (l3[:, :t] - logits[:, :t]).abs().max().item()
            check(f"{tag} whole-model causality (eval + train mode)", past == 0.0 and tr == 0.0 and fut > 0,
                  f"past diff {past:.1e} / train {tr:.1e}, future {fut:.3f}")

        # 8. every parameter gets a gradient
        m.train()
        _, loss = m(x, y)
        loss.backward()
        missing = [nm for nm, p in m.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0]
        check(f"{tag} all {len(list(m.parameters()))} params get finite non-zero grads", not missing,
              ", ".join(missing[:3]))
        m.zero_grad(set_to_none=True)

        # 9. it can learn: a few AdamW steps on one batch
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
        l0 = None
        for _ in range(20):
            _, loss = m(x, y)
            l0 = loss.item() if l0 is None else l0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            l1 = m(x, y)[1].item()
        check(f"{tag} 20 AdamW steps on one batch lower the loss", l1 < l0 - 1.0, f"{l0:.3f} -> {l1:.3f}")
        del m, opt

    # 6. Block structure (independent of position type)
    torch.manual_seed(args.seed)
    blk = Block(base).to(device).eval()
    with torch.no_grad():
        h = torch.randn(2, 16, d, device=device)
        hid = blk.mlp.fc(blk.ln2(h))
        check("MLP hidden width == mlp_ratio * d and output (B, T, d)",
              hid.shape[-1] == base.mlp_ratio * d and blk.mlp(h).shape == h.shape, f"hidden {hid.shape[-1]}")
        blk.attn.proj.weight.zero_()
        blk.mlp.proj.weight.zero_()
        out = blk(h, None)
        check("Block with zeroed residual projections is the identity (pre-LN residual form)",
              torch.equal(out, h))
        ln = blk.ln1(h * 3.0 + 5.0)
        check("LayerNorm output has per-token mean 0, std 1",
              ln.mean(-1).abs().max().item() < 1e-4 and (ln.std(-1, unbiased=False) - 1).abs().max().item() < 1e-3)

    # 10. trained checkpoint
    if not args.random:
        ck = latest_checkpoint(args.ckpt) if args.ckpt.is_dir() else args.ckpt
        if ck is None:
            check(f"trained checkpoint found in {args.ckpt}", False)
        else:
            st = load_checkpoint(ck)
            tc = GPTConfig(**st["model_config"])
            m = GPT(tc)
            res = m.load_state_dict(st["model"], strict=True)
            m = m.to(device).eval()
            check(f"checkpoint {ck.name} (step {st['step']}) loads strictly",
                  not res.missing_keys and not res.unexpected_keys)
            check("checkpoint param count == formula", m.num_params(False) == expected_params(tc),
                  f"{m.num_params(False):,} ({m.num_params():,} non-embedding)")
            with torch.no_grad():
                _, lt = m(x, y)
            check("trained val loss far below ln(V)", lt.item() < lnV - 6.0,
                  f"{lt.item():.4f} (ppl {math.exp(lt.item()):.2f}) vs init ~{lnV:.2f}")

    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
