"""Throughput / peak-VRAM benchmark for training steps (bf16 autocast, AdamW, random tokens).

  .venv/bin/python scripts/bench.py --out docs/bench.md
Each config: a few warmup steps, then timed steps; peak = torch.cuda.max_memory_allocated.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tinyllm.device import VRAM_BUDGET_BYTES  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402
from tinyllm.optim import make_optimizer  # noqa: E402

CASES = [  # (name, model overrides, micro-batch, compile)
    ("29M d512 L8 T256", {}, 16, False),
    ("29M d512 L8 T256", {}, 32, False),
    ("29M d512 L8 T256", {}, 64, False),
    ("29M d512 L8 T256", {}, 32, True),
    ("29M d512 L8 T512", {"block_size": 512}, 16, False),
    ("29M d512 L8 T512", {"block_size": 512}, 32, False),
    ("51M d640 L10 T256", {"d_model": 640, "n_layers": 10, "n_heads": 10}, 32, False),
]


def bench(over: dict, B: int, compile_: bool, steps: int, warmup: int) -> dict:
    torch.manual_seed(0)
    cfg = GPTConfig(**over)
    model = GPT(cfg).cuda()
    opt = make_optimizer(model, 3e-4)
    fwd = torch.compile(model) if compile_ else model
    x = torch.randint(0, cfg.vocab_size, (B, cfg.block_size), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    res = {"params_m": model.num_params() / 1e6, "tok_s": B * cfg.block_size * steps / dt,
           "peak_gb": torch.cuda.max_memory_allocated() / 1024**3}
    del model, opt, fwd
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = ["| model | micro-batch | compile | params (M) | tokens/s | peak VRAM (GB) | fits < 6.5 GB |",
            "|---|---|---|---|---|---|---|"]
    for name, over, B, comp in CASES:
        try:
            r = bench(over, B, comp, args.steps, args.warmup + (10 if comp else 0))
            fits = "yes" if r["peak_gb"] * 1024**3 < VRAM_BUDGET_BYTES else "no"
            row = f"| {name} | {B} | {comp} | {r['params_m']:.1f} | {r['tok_s']/1e3:.1f}k | {r['peak_gb']:.2f} | {fits} |"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = f"| {name} | {B} | {comp} | - | OOM | - | no |"
        except Exception as e:  # e.g. compile backend missing
            row = f"| {name} | {B} | {comp} | - | error: {type(e).__name__} | - | - |"
        print(row, flush=True)
        rows.append(row)
    if args.out:
        Path(args.out).write_text("# Training throughput benchmark (RTX 5050 Laptop, bf16)\n\n"
                                  + "\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
