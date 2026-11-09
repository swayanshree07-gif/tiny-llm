"""Build the chat fine-tune set (PLAN Day 21) from our own stories (no download).

-> data/chat/{train,valid}.npz with x (uint16) and y (int16, -100 = no loss: user turns, padding)
   and data/chat/{split}_convs.jsonl (messages + checkable specs, for evals and chat tests).

Run: .venv/bin/python scripts/build_chat.py [--train 60000] [--valid 1000] [--block 256]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.chat import build_chat_rows  # noqa: E402
from tinyllm.instruct import stories_from_tokens, word_target_mask  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--out", type=Path, default=ROOT / "data/chat")
    ap.add_argument("--train", type=int, default=60000, help="number of source stories")
    ap.add_argument("--valid", type=int, default=1000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--mode", choices=["mixed", "three"], default="mixed",
                    help="'three': always-3-word requests, mostly salient words (chat v3)")
    args = ap.parse_args()

    tok = Tokenizer.load(args.tok_dir / "tokenizer.json")
    args.out.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", args.train, 11), ("valid", args.valid, 12)):
        if n == 0:
            continue
        t0 = time.perf_counter()
        ids = np.memmap(args.tok_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        stories = stories_from_tokens(tok, ids, n, seed=seed)
        x, y, convs = build_chat_rows(tok, stories, args.block, seed=seed, mode=args.mode)
        np.savez(args.out / f"{split}.npz", x=x.astype(np.uint16), y=y.astype(np.int16))
        # tokens of requested words (all "words" turns in the row), up-weighted by train.mask_loss_weight
        m = np.stack([word_target_mask(tok.vocab, row, [w for sp in specs if sp["kind"] == "words"
                                                         for w in sp["words"]])
                      for row, (_, specs) in zip(y, convs)])
        np.savez(args.out / f"{split}_wordmask.npz", m=m)
        with open(args.out / f"{split}_convs.jsonl", "w") as f:
            for msgs, specs in convs:
                f.write(json.dumps({"messages": msgs, "specs": specs}) + "\n")
        turns = Counter(len(s) for _, s in convs)
        kinds = Counter(sp["kind"] for _, s in convs for sp in s)
        print(f"{split}: {len(stories)} stories -> {len(convs)} rows, turns/row {dict(sorted(turns.items()))}, "
              f"kinds {dict(kinds)}, {float((y != -100).mean()):.1%} target tokens, "
              f"{float(m.mean()):.2%} word-mask tokens, "
              f"{time.perf_counter() - t0:.1f}s")
    print("example row:", json.dumps(convs[0][0])[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
