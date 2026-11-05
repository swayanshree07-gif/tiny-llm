"""Re-runnable check of the token files data/tok/*.bin + meta.json (PLAN Day 6).

Checks, for every split in meta.json:
  1. file size == tokens * 2 (uint16) and == meta "tokens"
  2. all ids < vocab_size; EOT id present; first story starts at index 0 (no leading EOT)
  3. N random slices (seeded) decode to readable text: valid UTF-8 for whole stories,
     >= 95% printable/whitespace characters, mostly ASCII letters
  4. meta "bytes" / tokens gives a sane compression ratio (3.5..5 bytes/token)
Prints token counts, stories per split, and one sample slice per split.

Run: .venv/bin/python scripts/check_tokens.py [--slices 20] [--len 64] [--seed 0] [--tok-dir data/tok]
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.tokenizer import Tokenizer  # noqa: E402


def readable(text: str) -> float:
    """Fraction of characters that are printable or ordinary whitespace."""
    if not text:
        return 0.0
    return sum(c.isprintable() or c in "\n\t" for c in text) / len(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--len", type=int, default=64, help="tokens per random slice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.tok_dir / "meta.json").read_text())
    tok = Tokenizer.load(args.tok_dir / "tokenizer.json")
    vocab, eot = meta["vocab_size"], meta["eot_id"]
    rng = np.random.default_rng(args.seed)
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    check("meta matches tokenizer", vocab == tok.vocab_size and eot == vocab - 1
          and meta["dtype"] == "uint16", f"(vocab {vocab}, eot {eot})")
    for split, info in meta["splits"].items():
        path = args.tok_dir / f"{split}.bin"
        ids = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(ids)
        print(f"--- {split}: {n:,} tokens ({path.stat().st_size / 2**20:.1f} MiB)")
        check(f"{split}: size == meta tokens", path.stat().st_size == 2 * n == 2 * info["tokens"],
              f"(meta {info['tokens']:,})")
        # chunked max / EOT count so the 350 MB train file isn't loaded at once
        mx, n_eot = 0, 0
        for s in range(0, n, 1 << 24):
            chunk = np.asarray(ids[s:s + (1 << 24)])
            mx = max(mx, int(chunk.max()))
            n_eot += int((chunk == eot).sum())
        check(f"{split}: ids < vocab", mx < vocab, f"(max {mx})")
        check(f"{split}: stories end with EOT", n_eot > 0 and int(ids[-1]) == eot and int(ids[0]) != eot,
              f"({n_eot:,} stories, {n / max(n_eot, 1):.0f} tokens/story)")
        bpt = info["bytes"] / n
        check(f"{split}: compression sane", 3.5 < bpt < 5.0, f"({bpt:.3f} bytes/token)")

        worst, sample = 1.0, ""
        for i in range(args.slices):
            s = int(rng.integers(0, n - args.len))
            text = tok.decode(ids[s:s + args.len].tolist())
            r = readable(text)
            letters = sum(c.isascii() and c.isalpha() for c in text) / max(len(text), 1)
            worst = min(worst, r if letters > 0.5 else 0.0)
            if i == 0:
                sample = text
        check(f"{split}: {args.slices} random slices readable", worst >= 0.95, f"(worst {worst:.3f})")
        # whole stories (between two EOTs) must be valid UTF-8 — slices may cut a multibyte char
        s = int(rng.integers(0, n - 2048))
        window = np.asarray(ids[s:s + 2048])
        cuts = np.flatnonzero(window == eot)
        if len(cuts) >= 2:
            story = window[cuts[0] + 1:cuts[1]].tolist()
            text = tok.decode(story)  # decode uses errors="replace": bad UTF-8 shows up as U+FFFD
            check(f"{split}: whole story decodes", "�" not in text and readable(text) >= 0.99,
                  f"({len(story)} tokens)")
        print(f"sample slice: {sample!r}")
    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
