"""Re-runnable check of the minimal training loop: overfit a single batch (PLAN Day 11).

Drives the real `tinyllm.train.train()` (the same function `scripts/train.py` uses) with
`overfit_one_batch=True` at full model size on real training text, and checks:

  1. step-1 loss ~ ln(V) (fresh init, |diff| < 0.3)
  2. the loss on the fixed batch drops below --target (0.1) within --steps (default 300) steps;
     the first step where it does is reported
  3. the loss curve really goes down (min over the second half < 5% of the step-1 loss)
  4. control: the same loop on fresh batches (no overfit) for --control-steps stays far above the
     target, so the drop in (2) is memorisation of the one batch, not a broken loss
  5. no checkpoint written (ckpt_interval=0), temp out_dir removed afterwards
  6. peak VRAM < 6.5 GB (CUDA only)

Run: .venv/bin/python scripts/check_overfit.py [--config configs/base.json] [--tok-dir data/tok]
     [--steps 300] [--batch-size 8] [--random] [--cpu]
`--random` uses a small model and a random token file (no data needed; used by pytest).
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.model import GPTConfig  # noqa: E402
from tinyllm.train import TrainConfig, config_from_dict, train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--control-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target", type=float, default=0.1)
    ap.add_argument("--random", action="store_true", help="small model + random token file")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    tmp = Path(tempfile.mkdtemp(prefix="overfit_", dir=ROOT / "ckpt" if (ROOT / "ckpt").is_dir() else None))
    try:
        if args.random:
            mcfg = GPTConfig(vocab_size=256, block_size=32, n_layers=2, n_heads=2, d_model=64)
            rng = np.random.default_rng(0)
            for name in ("train.bin", "valid.bin"):
                rng.integers(0, 256, 50_000, dtype=np.uint16).tofile(tmp / name)
            data_dir = tmp
        else:
            mcfg, _ = config_from_dict(json.loads(args.config.read_text()))
            data_dir = args.tok_dir
        print("model:", mcfg.to_dict())
        base = TrainConfig(
            data_dir=str(data_dir), out_dir=str(tmp / "run"), batch_size=args.batch_size, grad_accum=1,
            max_steps=args.steps, lr=args.lr, min_lr=args.lr, warmup=10, weight_decay=0.0,
            eval_interval=0, log_interval=1, ckpt_interval=0, resume=False, compile=False,
            device="cpu" if args.cpu else "cuda", dtype="auto", overfit_one_batch=True, seed=0)

        torch.manual_seed(0)
        res = train(mcfg, base, verbose=False)
        losses = [h["loss"] for h in res["history"] if h["type"] == "train"]
        lnv = math.log(mcfg.vocab_size)
        check("ran all steps", res["step"] == args.steps and len(losses) == args.steps,
              f"({res['step']} steps)")
        check("step-1 loss ~ ln(V)", abs(losses[0] - lnv) < 0.3, f"({losses[0]:.3f} vs {lnv:.3f})")
        first = next((i + 1 for i, l in enumerate(losses) if l < args.target), None)
        check(f"loss < {args.target} within {args.steps} steps", first is not None,
              f"(first at step {first}; final {losses[-1]:.2e})")
        for s in (25, 50, 100, 200, args.steps):
            if s <= len(losses):
                print(f"  step {s}: loss {losses[s - 1]:.4g}")
        tail = min(losses[len(losses) // 2:])
        check("loss curve goes down", tail < 0.05 * losses[0], f"(min 2nd half {tail:.2e})")
        peak = max((h.get("peak_mem_gb", 0.0) for h in res["history"]), default=0.0)
        if not args.cpu and torch.cuda.is_available():
            check("peak VRAM < 6.5 GB", peak < 6.5, f"({peak:.2f} GB)")
        written = list((tmp / "run").glob("*.pt"))
        check("no checkpoint files written", not written, f"({len(written)})")

        torch.manual_seed(0)
        ctrl = train(mcfg, replace(base, out_dir=str(tmp / "ctrl"), overfit_one_batch=False,
                                   max_steps=args.control_steps), verbose=False)
        cl = [h["loss"] for h in ctrl["history"] if h["type"] == "train"]
        at = min(args.control_steps, len(losses))
        check("control (fresh batches) stays far above target", cl[-1] > 10 * args.target
              and cl[-1] > 5 * losses[at - 1],
              f"(control {cl[-1]:.3f} vs overfit {losses[at - 1]:.3g} at step {at})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("temp dir removed", not tmp.exists())
    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
