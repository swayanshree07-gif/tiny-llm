"""Score checkpoints on the same fixed validation windows -> markdown table (PLAN Day 20/26).

  .venv/bin/python scripts/eval_ppl.py ckpt/base ckpt/large --labels base-29M large-91M \
      --iters 200 --out docs/large_vs_base.md

Windows come from TokenDataset(valid.bin, seed=--seed), so every model sees exactly the same tokens
(requires equal block_size). Loss is mean token cross-entropy under bf16 autocast (as in training).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tinyllm.data import TokenDataset  # noqa: E402
from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import load_model  # noqa: E402
from tinyllm.train import autocast_ctx, evaluate  # noqa: E402


def score(ckpt: str, data: Path, iters: int, bs: int, seed: int, device) -> dict:
    model, st = load_model(ckpt, device)
    T = model.cfg.block_size
    ds = TokenDataset(data, T, seed=seed)
    t0 = time.time()
    with torch.no_grad():
        loss = evaluate(model, ds, iters, bs, device, lambda: autocast_ctx(device, "bf16"), seed)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"step": st["step"], "loss": loss, "ppl": math.exp(loss), "params": n_params,
            "tokens": iters * bs * T, "block": T, "secs": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", help="checkpoint files or run dirs (latest checkpoint used)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--data", default=str(ROOT / "data/tok/valid.bin"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the markdown table here")
    a = ap.parse_args(argv)
    labels = a.labels or [Path(c).name for c in a.ckpts]
    if len(labels) != len(a.ckpts):
        raise SystemExit("--labels must match the number of checkpoints")
    device = get_device(a.device)
    res = [score(c, Path(a.data), a.iters, a.batch_size, a.seed, device) for c in a.ckpts]
    if len({r["block"] for r in res}) > 1:
        raise SystemExit("models have different block_size; windows would not match")
    lines = [f"Validation: {a.iters} × {a.batch_size} windows of {res[0]['block']} tokens "
             f"({res[0]['tokens']:,} tokens, seed {a.seed}), same windows for every model.\n",
             "| model | params | step | val loss | ppl |", "|---|---|---|---|---|"]
    for lab, r in zip(labels, res):
        lines.append(f"| {lab} | {r['params'] / 1e6:.1f}M | {r['step']} | {r['loss']:.4f} | {r['ppl']:.3f} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
