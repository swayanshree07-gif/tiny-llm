"""Re-runnable check of the training data loaders on the real files (PLAN Day 7).

Token windows (data/tok/{train,valid}.bin via TokenDataset), for each split:
  1. x, y shape (B, T), dtype int64, contiguous
  2. y == x shifted by one, and x/y equal the memmap slice at the window start (found by search)
  3. every id in [0, vocab_size); window starts within [0, max_start]
  4. same seed -> same batches; state_dict/load_state_dict resumes the exact stream
  5. on CUDA: batch lands on the GPU and equals the CPU batch; loader throughput (tokens/s)
     compared with the training throughput (~88k tok/s), so the loader is not the bottleneck
Instruct rows (data/instruct/*.npz via InstructDataset), if present:
  6. shapes/dtypes; ids in range; where y != -100, y[t] == x[t+1] (loss targets are the next token)

Run: .venv/bin/python scripts/check_loader.py [--tok-dir data/tok] [--instruct-dir data/instruct]
     [--batches 50] [--batch-size 32] [--block 256] [--cpu]
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.data import TokenDataset  # noqa: E402
from tinyllm.instruct import InstructDataset  # noqa: E402

TRAIN_TOK_S = 88_000  # main-run throughput (docs/bench.md, Day 5 log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--instruct-dir", type=Path, default=ROOT / "data/instruct")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="skip the CUDA checks")
    args = ap.parse_args()

    meta = json.loads((args.tok_dir / "meta.json").read_text())
    vocab = meta["vocab_size"]
    B, T = args.batch_size, args.block
    cuda = torch.cuda.is_available() and not args.cpu
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    for split in meta["splits"]:
        path = args.tok_dir / f"{split}.bin"
        ds = TokenDataset(path, T, seed=args.seed)
        raw = ds.data
        xs, ys = [], []
        for _ in range(args.batches):
            x, y = ds.get_batch(B)
            xs.append(x)
            ys.append(y)
        x, y = torch.cat(xs), torch.cat(ys)
        check(f"{split}: shape/dtype", x.shape == y.shape == (B * args.batches, T)
              and x.dtype == y.dtype == torch.int64 and xs[0].is_contiguous() and ys[0].is_contiguous(),
              f"({tuple(xs[0].shape)} {xs[0].dtype})")
        check(f"{split}: y == x shifted by one", torch.equal(x[:, 1:], y[:, :-1]))
        check(f"{split}: ids in [0, {vocab})", int(x.min()) >= 0 and int(max(x.max(), y.max())) < vocab,
              f"(min {int(x.min())}, max {int(max(x.max(), y.max()))})")

        # Re-draw the starts with the same seed and compare against the memmap directly.
        rng = np.random.default_rng(args.seed)
        starts = rng.integers(0, ds.max_start + 1, size=B)
        first = xs[0].numpy()
        match = all(np.array_equal(first[i], raw[s:s + T]) and np.array_equal(ys[0][i].numpy(), raw[s + 1:s + T + 1])
                    for i, s in enumerate(starts))
        check(f"{split}: windows == memmap slices", match,
              f"(starts {int(starts.min())}..{int(starts.max())} of max {ds.max_start:,})")
        check(f"{split}: starts in range", 0 <= starts.min() and starts.max() <= ds.max_start
              and ds.max_start + T + 1 < len(raw))

        a, b = TokenDataset(path, T, seed=7), TokenDataset(path, T, seed=7)
        same = all(torch.equal(a.get_batch(4)[0], b.get_batch(4)[0]) for _ in range(3))
        st = a.state_dict()
        nxt = a.get_batch(4)[0]
        c = TokenDataset(path, T, seed=123)
        c.load_state_dict(st)
        check(f"{split}: seeded + resumable", same and torch.equal(c.get_batch(4)[0], nxt))

        if cuda:
            g = TokenDataset(path, T, seed=11)
            h = TokenDataset(path, T, seed=11)
            xg, yg = g.get_batch(B, "cuda")
            xc, yc = h.get_batch(B)
            check(f"{split}: cuda batch == cpu batch", xg.is_cuda and torch.equal(xg.cpu(), xc)
                  and torch.equal(yg.cpu(), yc))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.batches):
                xg, yg = g.get_batch(B, "cuda")
            torch.cuda.synchronize()
            tps = args.batches * B * T / (time.perf_counter() - t0)
            check(f"{split}: loader faster than training", tps > 5 * TRAIN_TOK_S,
                  f"({tps / 1e6:.2f}M tok/s to GPU, {tps / TRAIN_TOK_S:.0f}x training)")

    for name in ("train.npz", "valid.npz"):
        p = args.instruct_dir / name
        if not p.exists():
            print(f"[SKIP] instruct {name} not found")
            continue
        z = np.load(p)
        ds = InstructDataset(z["x"].astype(np.int64), z["y"].astype(np.int64), seed=args.seed)
        x, y = ds.get_batch(4 * B)
        check(f"instruct {name}: shape/dtype", x.shape == y.shape == (4 * B, ds.block_size)
              and x.dtype == y.dtype == torch.int64, f"({len(ds):,} rows x {ds.block_size})")
        m = y[:, :-1] != -100
        check(f"instruct {name}: targets == next token", torch.equal(y[:, :-1][m], x[:, 1:][m]),
              f"({m.float().mean():.1%} target tokens)")
        yv = y[y != -100]
        check(f"instruct {name}: ids in range", int(x.min()) >= 0 and int(x.max()) < vocab
              and int(yv.min()) >= 0 and int(yv.max()) < vocab)

    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
