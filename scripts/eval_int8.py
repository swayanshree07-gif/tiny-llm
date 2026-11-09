"""Int8 weight-only quantization: val loss, size and speed vs the float model (PLAN Day 25).

  .venv/bin/python scripts/eval_int8.py ckpt/base ckpt/large --labels base-29M large-91M \
      --out docs/infer_bench_int8.md

Val loss uses the same fixed windows as eval_ppl.py (bf16 autocast). Speed = cached greedy decoding
(bf16 autocast), 8-token random prompt + 240 new tokens, median of 3.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from bench_kv import timed  # noqa: E402
from tinyllm.data import TokenDataset  # noqa: E402
from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import generate, load_model  # noqa: E402
from tinyllm.quant import model_bytes, quantize_model  # noqa: E402
from tinyllm.train import autocast_ctx, evaluate  # noqa: E402


def speed(model, device, B: int, n_new: int = 240) -> float:
    idx = torch.randint(0, model.cfg.vocab_size, (B, 8), device=device,
                        generator=torch.Generator(device=device).manual_seed(0))
    with torch.no_grad(), autocast_ctx(device, "bf16"):
        generate(model, idx, 4, temperature=0, use_cache=True)
        ts = [timed(lambda: generate(model, idx, n_new, temperature=0, use_cache=True), device)[1] for _ in range(3)]
    return B * n_new / statistics.median(ts)


def measure(model, ds, a, device) -> dict:
    with torch.no_grad():
        loss = evaluate(model, ds, a.iters, a.batch_size, device, lambda: autocast_ctx(device, "bf16"), a.seed)
    return {"loss": loss, "ppl": math.exp(loss), "mb": model_bytes(model) / 2**20,
            "tps1": speed(model, device, 1), "tps8": speed(model, device, 8)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--data", default=str(ROOT / "data/tok/valid.bin"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-delta", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    labels = a.labels or [Path(c).name for c in a.ckpts]
    device = get_device(a.device)
    rows, ok = [], True
    for c, lab in zip(a.ckpts, labels):
        model, st = load_model(c, device)
        ds = TokenDataset(Path(a.data), model.cfg.block_size, seed=a.seed)
        f = measure(model, ds, a, device)
        q = measure(quantize_model(model), ds, a, device)
        d = q["loss"] - f["loss"]
        ok &= d <= a.max_delta
        rows.append((lab, st["step"], f, q, d))
        print(f"{lab}: fp32 {f['loss']:.4f} ({f['mb']:.0f} MB) -> int8 {q['loss']:.4f} ({q['mb']:.0f} MB), "
              f"delta {d:+.4f}; tok/s B1 {f['tps1']:.0f}->{q['tps1']:.0f}, B8 {f['tps8']:.0f}->{q['tps8']:.0f}",
              flush=True)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    if a.out:
        lines = ["# Int8 weight-only quantization\n",
                 f"Per-output-channel symmetric int8 for every Linear (incl. LM head; embeddings/LayerNorm stay fp32), "
                 f"`tinyllm/quant.py`. Val loss on {a.iters}×{a.batch_size} fixed windows (seed {a.seed}), bf16 autocast. "
                 "Size = persistent weights in memory. tok/s = cached greedy, bf16 autocast, 240 new tokens. "
                 "Regenerate with `scripts/eval_int8.py`.\n",
                 "| model | step | weights | val loss | ppl | Δ loss | size MB | tok/s B=1 | tok/s B=8 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for lab, step, f, q, d in rows:
            lines.append(f"| {lab} | {step} | fp32 | {f['loss']:.4f} | {f['ppl']:.3f} | – | {f['mb']:.0f} | "
                         f"{f['tps1']:.0f} | {f['tps8']:.0f} |")
            lines.append(f"| {lab} | {step} | int8 | {q['loss']:.4f} | {q['ppl']:.3f} | {d:+.4f} | {q['mb']:.0f} | "
                         f"{q['tps1']:.0f} | {q['tps8']:.0f} |")
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"wrote {a.out}")
    print("PASS" if ok else f"FAIL: some delta > {a.max_delta}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
