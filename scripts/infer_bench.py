"""Inference speed benchmark across models, precisions, batch sizes, cache on/off (PLAN Day 24).

  .venv/bin/python scripts/infer_bench.py ckpt/base ckpt/large --labels base-29M large-91M \
      --out docs/infer_bench.md

Greedy decoding from a fixed random prompt; tokens/s = batch * new tokens / wall time (CUDA-synced,
median of --reps). Also reports whether cached and uncached greedy outputs are identical.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from bench_kv import timed  # noqa: E402
from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import generate, load_model  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402


def amp(device, prec: str):
    if prec == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def bench_model(model, label, device, batches, precs, prompt_len, n_new, reps) -> list[dict]:
    n_new = min(n_new, model.cfg.block_size - prompt_len)
    rows = []
    for prec in precs:
        for B in batches:
            idx = torch.randint(0, model.cfg.vocab_size, (B, prompt_len), device=device,
                                generator=torch.Generator(device=device).manual_seed(0))
            res = {}
            with torch.no_grad(), amp(device, prec):
                for cache in (False, True):
                    generate(model, idx, 4, temperature=0, use_cache=cache)  # warm-up
                    ts, out = [], None
                    for _ in range(reps):
                        out, t = timed(lambda: generate(model, idx, n_new, temperature=0, use_cache=cache), device)
                        ts.append(t)
                    res[cache] = (out, statistics.median(ts))
            (a, ta), (b, tb) = res[False], res[True]
            r = {"model": label, "prec": prec, "B": B, "new": n_new, "nocache": B * n_new / ta,
                 "cache": B * n_new / tb, "speedup": ta / tb, "same": torch.equal(a, b),
                 "agree": (a == b).float().mean().item()}
            rows.append(r)
            print(f"{label} {prec} B={B}: no cache {r['nocache']:.0f} tok/s, cache {r['cache']:.0f} tok/s, "
                  f"x{r['speedup']:.2f}, identical={r['same']} ({r['agree']:.4f})", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="*", help="checkpoint files/dirs; none = tiny random model")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--prec", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--new", type=int, default=240)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    device = get_device(a.device)
    rows = []
    if not a.ckpts:
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=256, block_size=64, n_layers=2, d_model=64, n_heads=4)).to(device).eval()
        rows += bench_model(m, "random-tiny", device, a.batch, a.prec, a.prompt_len, a.new, a.reps)
    labels = a.labels or [Path(c).name for c in a.ckpts]
    if len(labels) != len(a.ckpts):
        raise SystemExit("--labels must match the number of checkpoints")
    for c, lab in zip(a.ckpts, labels):
        m, st = load_model(c, device)
        rows += bench_model(m, f"{lab} (step {st['step']})", device, a.batch, a.prec, a.prompt_len, a.new, a.reps)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if a.out:
        lines = ["# Inference speed benchmark\n",
                 f"Device {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else "")
                 + f"; greedy; prompt {a.prompt_len} random tokens + up to {a.new} new; median of {a.reps} runs. "
                 "bf16 = autocast (weights stay fp32). Regenerate with `scripts/infer_bench.py`.\n",
                 "| model | precision | batch | new tokens | no cache tok/s | cache tok/s | speedup | greedy identical |",
                 "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['prec']} | {r['B']} | {r['new']} | {r['nocache']:.0f} | {r['cache']:.0f} | "
                  f"{r['speedup']:.2f}x | {r['same']} ({r['agree']:.4f}) |" for r in rows]
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
