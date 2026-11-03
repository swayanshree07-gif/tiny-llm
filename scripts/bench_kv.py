"""Measure generation speed with and without the KV cache (PLAN Day 24).

  .venv/bin/python scripts/bench_kv.py --ckpt ckpt/base --out docs/kv_cache.md
Also checks that greedy outputs are identical in both modes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import generate, load_model  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402


def timed(fn, device) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="checkpoint dir/file; default: random 29.4M model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--new", type=int, default=240)
    ap.add_argument("--out", default=None, help="write a markdown table here")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    if args.ckpt:
        model, st = load_model(args.ckpt, device)
        src = f"checkpoint step {st['step']}"
    else:
        model, src = GPT(GPTConfig()).to(device).eval(), "random init"
    n_new = min(args.new, model.cfg.block_size - args.prompt_len)
    rows = []
    for B in args.batch:
        idx = torch.randint(0, model.cfg.vocab_size, (B, args.prompt_len), device=device,
                            generator=torch.Generator(device=device).manual_seed(0))
        generate(model, idx, 8, temperature=0, use_cache=True)  # warm-up
        generate(model, idx, 8, temperature=0)
        a, ta = timed(lambda: generate(model, idx, n_new, temperature=0), device)
        b, tb = timed(lambda: generate(model, idx, n_new, temperature=0, use_cache=True), device)
        same = torch.equal(a, b)
        agree = (a == b).float().mean().item()
        rows.append((B, n_new, B * n_new / ta, B * n_new / tb, ta / tb, same, agree))
        print(f"B={B}: no cache {B * n_new / ta:.0f} tok/s, cache {B * n_new / tb:.0f} tok/s, "
              f"speedup {ta / tb:.2f}x, identical={same} (token agreement {agree:.4f})")
    if args.out:
        lines = [f"# KV cache benchmark\n", f"Model: {src}, device {device}, fp32, greedy, prompt "
                 f"{args.prompt_len} tokens + {n_new} new (stays within block_size).\n",
                 "| batch | new tokens | no cache tok/s | cache tok/s | speedup | identical |",
                 "|---|---|---|---|---|---|"]
        lines += [f"| {B} | {n} | {x:.0f} | {y:.0f} | {s:.2f}x | {same} ({ag:.4f}) |"
                  for B, n, x, y, s, same, ag in rows]
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
