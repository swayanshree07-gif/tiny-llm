"""Train the byte-level BPE tokenizer on the first --mb MiB of a text file.

Run: .venv/bin/python scripts/train_tokenizer.py --mb 50 --vocab 8192 --out data/tok/tokenizer.json
Prints timing and compression (bytes/token) on the validation file sample.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.bpe import pretokenize, train_bpe  # noqa: E402
from tinyllm.tokenizer import EOT, Tokenizer  # noqa: E402


def read_prefix(path: Path, mb: float) -> str:
    with open(path, "rb") as f:
        raw = f.read(int(mb * 2**20))
    cut = raw.rfind(EOT.encode())  # end on a story boundary
    return raw[: cut + len(EOT)].decode("utf-8", errors="ignore") if cut > 0 else raw.decode("utf-8", "ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/train.txt")
    ap.add_argument("--eval", type=Path, default=ROOT / "data/raw/valid.txt")
    ap.add_argument("--mb", type=float, default=50)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=ROOT / "data/tok/tokenizer.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    text = read_prefix(args.input, args.mb)
    counts = Counter()
    for part in text.split(EOT):
        counts.update(pretokenize(part))
    t1 = time.perf_counter()
    print(f"pretokenize : {len(text) / 2**20:.1f} MiB, {len(counts):,} unique chunks, {t1 - t0:.1f}s")
    tok = Tokenizer(train_bpe(counts, args.vocab - 1))
    t2 = time.perf_counter()
    print(f"bpe train   : {len(tok.merges):,} merges -> vocab {tok.vocab_size}, {t2 - t1:.1f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved       : {args.out}")

    sample = read_prefix(args.eval, 2)
    t3 = time.perf_counter()
    ids = tok.encode(sample)
    t4 = time.perf_counter()
    assert tok.decode(ids) == sample, "round-trip failed"
    print(f"eval        : {len(sample.encode()) / len(ids):.2f} bytes/token on 2 MiB val, "
          f"{len(sample.encode()) / 2**20 / (t4 - t3):.2f} MiB/s encode, round-trip ok")
    longest = sorted((tok.vocab[i] for i in range(256, tok.eot_id)), key=len)[-8:]
    print("longest     :", [b.decode("utf-8", "replace") for b in longest])
    return 0


if __name__ == "__main__":
    sys.exit(main())
