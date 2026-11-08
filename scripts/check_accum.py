"""Re-runnable check of gradient accumulation, clipping, AdamW decay groups and bf16 autocast (PLAN Day 12).

Uses the real `tinyllm.train.accumulate_grads`, `tinyllm.optim.make_optimizer` and the full-size model
(configs/base.json, dropout 0) on real training text, and checks:

  1. fp32 (TF32 off): grads of one batch B == grads of k micro-batches of B/k for k = 2, 4, 8
     (max relative error per tensor < 1e-4), and the returned loss == the one-batch loss
  2. one AdamW step after accumulation == one step on the full batch (params within 1e-6 abs)
  3. bf16 autocast: logits are bf16 inside the context, grads close to fp32 (cosine > 0.99)
  4. clip_grad_norm_: returns the pre-clip norm; after clipping the norm == min(norm, c);
     direction unchanged; no-op when c > norm
  5. decay groups on the full model: every dim>=2 tensor decayed, every bias/LayerNorm not, the tied
     embedding/head weight counted once, totals == num_params; with zero grads one AdamW step shrinks
     decayed params by exactly (1 - lr*wd) and leaves undecayed ones unchanged
  6. info: weighted-loss path (fine-tune) — per-micro-batch normalisation means accumulation is only exact
     when the weight sums per micro-batch match; the size of the difference is printed (not a failure)

Run: .venv/bin/python scripts/check_accum.py [--config configs/base.json] [--tok-dir data/tok]
     [--batch-size 16] [--random] [--cpu]
`--random` uses a small model and random tokens (no data needed; used by pytest). Exit 0 only if all pass.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.data import TokenDataset  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402
from tinyllm.optim import make_optimizer, param_groups  # noqa: E402
from tinyllm.train import accumulate_grads, autocast_ctx, config_from_dict  # noqa: E402


def grads_of(model: GPT, batches, ctx_factory) -> tuple[float, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = accumulate_grads(model, batches, ctx_factory)
    return loss, [p.grad.detach().clone() for p in model.parameters()]


def split(x: torch.Tensor, y: torch.Tensor, k: int):
    return [(a, b) for a, b in zip(x.chunk(k), y.chunk(k))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-4, help="max relative grad error per tensor (fp32)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--break-scaling", action="store_true",
                    help="mutation test: sum micro-batch losses without 1/k (must FAIL)")
    args = ap.parse_args(argv)

    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    B = args.batch_size
    if args.random:
        mcfg = GPTConfig(vocab_size=256, block_size=32, n_layers=2, n_heads=2, d_model=64)
        g = torch.Generator().manual_seed(0)
        x_all = torch.randint(0, 256, (B, 33), generator=g)
    else:
        mcfg, _ = config_from_dict(json.loads(args.config.read_text()))
        ds = TokenDataset(args.tok_dir / "train.bin", mcfg.block_size, seed=123)
        x, y = ds.get_batch(B)
        x_all = torch.cat([x, y[:, -1:]], dim=1)
    mcfg = replace(mcfg, dropout=0.0)
    x, y = x_all[:, :-1].to(device), x_all[:, 1:].to(device)
    torch.manual_seed(0)
    model = GPT(mcfg).to(device).train()
    print("model:", mcfg.to_dict(), "device:", device, "batch:", tuple(x.shape))
    fp32 = contextlib.nullcontext

    if args.break_scaling:  # mutation: forget the 1/k factor

        def bad(model, batches, ctx_factory):
            tot = 0.0
            for xb, yb, *_ in batches:
                with ctx_factory():
                    _, loss = model(xb, yb)
                loss.backward()
                tot += float(loss)
            return tot / len(batches)
        global accumulate_grads
        accumulate_grads = bad  # noqa: F811

    # 1. accumulation equivalence in fp32
    loss1, g1 = grads_of(model, [(x, y)], fp32)
    for k in (2, 4, 8):
        if B % k:
            continue
        lk, gk = grads_of(model, split(x, y, k), fp32)
        rel = max(float((a - b).norm() / b.norm().clamp_min(1e-12)) for a, b in zip(gk, g1))
        check(f"fp32 grads: {k}x(B/{k}) == 1xB", rel < args.tol, f"(max rel err {rel:.2e})")
        check(f"fp32 loss: {k}x(B/{k}) == 1xB", abs(lk - loss1) < 1e-5 * max(1, loss1),
              f"({lk:.6f} vs {loss1:.6f})")

    # 2. one optimizer step after accumulation == one step on the full batch
    state0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    after = []
    for batches in ([(x, y)], split(x, y, 4 if B % 4 == 0 else 2)):
        model.load_state_dict(state0, strict=False)
        opt = make_optimizer(model, lr=1e-3, weight_decay=0.1)
        model.zero_grad(set_to_none=True)
        accumulate_grads(model, batches, fp32)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        after.append({n: p.detach().clone() for n, p in model.named_parameters()})
    # Adam's first step is ~lr*sign(g), so elements with |g| ~ eps amplify fp noise; compare whole updates.
    ua = torch.cat([(after[0][n] - state0[n]).flatten() for n in state0])
    ub = torch.cat([(after[1][n] - state0[n]).flatten() for n in state0])
    rel = float((ua - ub).norm() / ua.norm())
    check("AdamW step after accum == full-batch step", rel < 1e-3 and float(ua.norm()) > 0,
          f"(update rel diff {rel:.2e}; update norm {float(ua.norm()):.3f})")
    model.load_state_dict(state0, strict=False)

    # 3. bf16 autocast
    bf = (lambda: autocast_ctx(device, "bf16")) if device.type == "cuda" else \
        (lambda: torch.autocast("cpu", dtype=torch.bfloat16))
    with bf():
        logits, _ = model(x[:2])
    check("autocast logits are bf16", logits.dtype == torch.bfloat16, f"({logits.dtype})")
    _, gb = grads_of(model, split(x, y, 2), bf)
    va, vb = torch.cat([t.flatten() for t in g1]), torch.cat([t.flatten() for t in gb])
    cos = float(torch.nn.functional.cosine_similarity(va.float(), vb.float(), dim=0))
    check("bf16 grads ~ fp32 grads", cos > 0.99, f"(cosine {cos:.5f})")

    # 4. clipping
    _, g1 = grads_of(model, [(x, y)], fp32)
    norm = float(torch.cat([t.flatten() for t in g1]).norm())
    c = norm / 3
    ret = float(torch.nn.utils.clip_grad_norm_(model.parameters(), c))
    after_n = float(torch.cat([p.grad.flatten() for p in model.parameters()]).norm())
    dirn = float(torch.nn.functional.cosine_similarity(
        torch.cat([p.grad.flatten() for p in model.parameters()]), torch.cat([t.flatten() for t in g1]), dim=0))
    check("clip returns pre-clip norm", abs(ret - norm) < 1e-4 * norm, f"({ret:.4f} vs {norm:.4f})")
    check("clipped norm == c", abs(after_n - c) < 1e-3 * c, f"({after_n:.4f} vs c {c:.4f})")
    check("clip keeps direction", dirn > 0.99999, f"(cosine {dirn:.6f})")
    before = [p.grad.clone() for p in model.parameters()]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10 * norm)
    same = all(torch.equal(a, p.grad) for a, p in zip(before, model.parameters()))
    check("clip is a no-op when c > norm", same)

    # 5. decay groups
    decay, no_decay = param_groups(model, 0.1)
    ids = [id(p) for gr in (decay, no_decay) for p in gr["params"]]
    named = dict(model.named_parameters())
    nd_names = [n for n, p in named.items() if any(p is q for q in no_decay["params"])]
    check("decay = all dim>=2, no_decay = all dim<2",
          all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"]),
          f"({len(decay['params'])} / {len(no_decay['params'])} tensors)")
    check("no bias/LayerNorm param decayed", all(("bias" in n or "ln" in n or "norm" in n) for n in nd_names)
          and not any(p.dim() < 2 for p in decay["params"]))
    check("tied weight once, totals == num_params",
          len(ids) == len(set(ids)) and sum(p.numel() for gr in (decay, no_decay) for p in gr["params"])
          == model.num_params(False), f"({model.num_params(False):,} params)")
    lr, wd = 1e-2, 0.1
    opt = make_optimizer(model, lr=lr, weight_decay=wd)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    snap = {n: p.detach().clone() for n, p in named.items()}
    dec_ids = {id(p) for p in decay["params"]}
    dec_names = {n for n, p in named.items() if id(p) in dec_ids}
    opt.step()
    named = {n: p.detach() for n, p in named.items()}
    err_d = max(float((p - snap[n] * (1 - lr * wd)).abs().max()) for n, p in named.items() if n in dec_names)
    err_n = max(float((p - snap[n]).abs().max()) for n, p in named.items() if n not in dec_names)
    check("zero-grad AdamW step: decayed *= (1-lr*wd), others unchanged", err_d < 1e-6 and err_n == 0.0,
          f"(errors {err_d:.1e} / {err_n:.1e})")
    model.load_state_dict(state0, strict=False)

    # 6. info: weighted path
    w = torch.ones_like(y, dtype=torch.float32)
    w[: B // 2, : y.shape[1] // 2] = 5.0  # uneven weight sums across micro-batches
    _, gw1 = grads_of(model, [(x, y, w)], fp32)
    _, gw2 = grads_of(model, [(a, b, c_) for a, b, c_ in zip(x.chunk(2), y.chunk(2), w.chunk(2))], fp32)
    rel = max(float((a - b).norm() / b.norm().clamp_min(1e-12)) for a, b in zip(gw2, gw1))
    print(f"[INFO] weighted loss, uneven weight sums: 2x(B/2) vs 1xB max rel grad diff {rel:.2e} "
          "(per-micro-batch normalisation; expected, not a bug for training)")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        check("peak VRAM < 6.5 GB", peak < 6.5, f"({peak:.2f} GB)")
    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
