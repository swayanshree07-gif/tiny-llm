"""Re-runnable check of the final tokenizer and the token files (PLAN Day 5/6).

For a split (default: valid, the full file):
  1. encode(normalised raw) -> decode == normalised raw     (round-trip)
  2. fresh encoding == data/tok/<split>.bin                  (token file is current)
  3. compression ratio (UTF-8 bytes / token) and encode speed
Also checks: ids < vocab_size, EOT count == story count, unicode round-trip.

Run: .venv/bin/python scripts/check_tokenizer.py [--split valid] [--mb N]
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tokenize_data import iter_blocks, normalize  # noqa: E402
from tinyllm.tokenizer import EOT, Tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--mb", type=float, default=0, help="only check the first N MiB (0 = all)")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/tok/tokenizer.json")
    args = ap.parse_args()

    tok = Tokenizer.load(args.tokenizer)
    eot_id = tok.encode(EOT)
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    uni = "Héllo wörld — 你好 🙂 tabs\tand\nnewlines  "
    check("unicode round-trip", tok.decode(tok.encode(uni)) == uni)
    check("EOT is one special id", eot_id == [tok.vocab_size - 1], str(eot_id))

    raw = ROOT / "data/raw" / f"{args.split}.txt"
    bin_ids = np.memmap(ROOT / "data/tok" / f"{args.split}.bin", dtype=np.uint16, mode="r")
    limit = int(args.mb * (1 << 20)) if args.mb else None

    n_tok = n_bytes = n_stories = 0
    rt_ok = bin_ok = True
    t_enc = 0.0
    for text in iter_blocks(raw):
        text = normalize(text)
        t0 = time.perf_counter()
        ids = tok.encode(text)
        t_enc += time.perf_counter() - t0
        rt_ok &= tok.decode(ids) == text
        bin_ok &= np.array_equal(np.asarray(bin_ids[n_tok:n_tok + len(ids)]), np.asarray(ids, np.uint16))
        n_tok += len(ids)
        n_bytes += len(text.encode())
        n_stories += text.count(EOT)
        if limit and n_bytes >= limit:
            break

    full = limit is None
    check("decode(encode(x)) == x", rt_ok, f"({n_bytes / 2**20:.1f} MiB)")
    check("encoding matches token file", bin_ok and (not full or n_tok == len(bin_ids)),
          f"({n_tok:,} tokens, file {len(bin_ids):,})")
    check("ids in range", int(bin_ids[:n_tok].max()) < tok.vocab_size)
    check("EOT count == stories", int((bin_ids[:n_tok] == tok.vocab_size - 1).sum()) == n_stories,
          f"({n_stories:,} stories)")
    print(f"compression: {n_bytes / n_tok:.3f} bytes/token; encode {n_bytes / 2**20 / t_enc:.1f} MiB/s")
    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
