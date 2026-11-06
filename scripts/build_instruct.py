"""Build the Day 27 instruction fine-tune set from the existing token files (no download).

-> data/instruct/{train,valid}.npz with x (uint16) and y (int16, -100 = no loss)
   and data/instruct/{split}_prompts.jsonl (prompt + spec, for prompt-following evals).

Run: .venv/bin/python scripts/build_instruct.py [--train 40000] [--valid 1000] [--block 256] [--word-mode any|salient]
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

from tinyllm.instruct import build_rows, stories_from_tokens  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--out", type=Path, default=ROOT / "data/instruct")
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--valid", type=int, default=1000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--word-mode", choices=["any", "salient"], default="any",
                    help="'salient': prompt words occur >= 2x in the story (fine-tune v3)")
    args = ap.parse_args()

    tok = Tokenizer.load(args.tok_dir / "tokenizer.json")
    args.out.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", args.train, 1), ("valid", args.valid, 2)):
        t0 = time.perf_counter()
        ids = np.memmap(args.tok_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        stories = stories_from_tokens(tok, ids, n, seed=seed)
        x, y, used = build_rows(tok, stories, args.block, seed=seed, word_mode=args.word_mode)
        np.savez(args.out / f"{split}.npz", x=x.astype(np.uint16), y=y.astype(np.int16))
        with open(args.out / f"{split}_prompts.jsonl", "w") as f:
            for prompt, spec in used:
                f.write(json.dumps({"prompt": prompt, **spec}) + "\n")
        kinds = {k: sum(s["kind"] == k for _, s in used) for k in ("words", "begins")}
        frac = float((y != -100).mean())
        print(f"{split}: {len(stories)} stories -> {len(used)} rows {kinds}, "
              f"{frac:.1%} target tokens, {time.perf_counter() - t0:.1f}s")
    print("example prompt:", used[0][0].replace("\n", "\\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
