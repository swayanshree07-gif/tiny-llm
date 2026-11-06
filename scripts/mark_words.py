"""Fine-tune v4: mark the target tokens that spell a requested word, for loss up-weighting.

Reads data_dir/{split}.npz + {split}_prompts.jsonl (row i <-> line i, as written by build_instruct.py)
and writes data_dir/{split}_wordmask.npz with bool "m" (rows x block_size). "begins" rows get no marks.
Used by train.train_mask_file / train.mask_loss_weight (configs/instruct_v4.json).

Run: .venv/bin/python scripts/mark_words.py --data data/instruct_v3 [--split train]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.instruct import word_target_mask  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/tok/tokenizer.json")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    vocab = Tokenizer.load(args.tokenizer).vocab
    y = np.load(args.data / f"{args.split}.npz")["y"].astype(np.int64)
    specs = [json.loads(line) for line in open(args.data / f"{args.split}_prompts.jsonl")]
    assert len(specs) == len(y), f"{len(specs)} prompts vs {len(y)} rows"
    m = np.zeros(y.shape, dtype=bool)
    words_rows = missing = 0
    for i, spec in enumerate(specs):
        if spec["kind"] != "words":
            continue
        words_rows += 1
        m[i] = word_target_mask(vocab, y[i], spec["words"])
        missing += not m[i].any()
    out = args.data / f"{args.split}_wordmask.npz"
    np.savez_compressed(out, m=m)
    tgt = int((y != -100).sum())
    print(f"{out}: {words_rows} words rows, {int(m.sum())} marked of {tgt} target tokens "
          f"({m.sum() / max(tgt, 1):.2%}), rows with no mark {missing}, {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
