"""Size the "larger model" (PLAN Day 15): throughput + peak VRAM of bigger candidates, bf16, compiled.

  timeout 20m .venv/bin/python scripts/bench_large.py --out docs/bench_large.md
Uses `bench()` from scripts/bench.py (random tokens, AdamW step, peak = max_memory_allocated).
The est. column = minutes of training for 300M tokens at the measured speed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from bench import bench  # noqa: E402
from tinyllm.device import VRAM_BUDGET_BYTES  # noqa: E402

L12 = {"d_model": 768, "n_layers": 12, "n_heads": 12}
CASES = [  # (name, model overrides, micro-batch)
    ("29M d512 L8 (base)", {}, 32),
    ("54M d640 L10", {"d_model": 640, "n_layers": 10, "n_heads": 10}, 32),
    ("91M d768 L12", L12, 16),
    ("91M d768 L12", L12, 32),
    ("91M d768 L12 +ckpt", {**L12, "grad_checkpoint": True}, 32),
    ("91M d768 L12 +ckpt", {**L12, "grad_checkpoint": True}, 64),
    ("120M d768 L16", {**L12, "n_layers": 16}, 32),
    ("120M d768 L16 +ckpt", {**L12, "n_layers": 16, "grad_checkpoint": True}, 32),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = ["| model | micro-batch | params (M) | tokens/s | peak VRAM (GB) | fits < 6.5 GB | min / 300M tok |",
            "|---|---|---|---|---|---|---|"]
    for name, over, B in CASES:
        try:
            r = bench(over, B, True, args.steps, 15)
            fits = "yes" if r["peak_gb"] * 1024**3 < VRAM_BUDGET_BYTES else "no"
            row = (f"| {name} | {B} | {r['params_m']:.1f} | {r['tok_s']/1e3:.1f}k | {r['peak_gb']:.2f} | {fits} "
                   f"| {300e6 / r['tok_s'] / 60:.0f} |")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = f"| {name} | {B} | - | OOM | - | no | - |"
        print(row, flush=True)
        rows.append(row)
    if args.out:
        Path(args.out).write_text("# Larger-model benchmark (RTX 5050 Laptop, bf16, torch.compile, T=256)\n\n"
                                  + "\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
