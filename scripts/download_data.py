"""Download TinyStories (V2, GPT-4 generated) into data/raw/.

- validation file: downloaded in full (~22 MB)
- train file: only the first --train-mb megabytes (HTTP Range request), then
  truncated at the last complete story so no story is cut in half.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories (CDLA-Sharing-1.0)
Stories are separated by the line "<|endoftext|>".

Run: .venv/bin/python scripts/download_data.py [--train-mb 700]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"valid": "TinyStoriesV2-GPT4-valid.txt", "train": "TinyStoriesV2-GPT4-train.txt"}
SEP = b"<|endoftext|>"
ROOT = Path(__file__).resolve().parents[1]
MAX_DATA_BYTES = int(1.5 * 1024**3)  # project rule: data/ stays under 1.5 GB


def fetch(url: str, dest: Path, max_bytes: int | None = None, chunk: int = 1 << 20) -> int:
    """Stream url to dest (optionally only the first max_bytes). Returns bytes written."""
    req = urllib.request.Request(url)
    if max_bytes is not None:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            if max_bytes is not None and n + len(buf) > max_bytes:
                buf = buf[: max_bytes - n]
            f.write(buf)
            n += len(buf)
            if n and n % (50 << 20) < chunk:
                print(f"  {dest.name}: {n / 2**20:.0f} MiB", flush=True)
            if max_bytes is not None and n >= max_bytes:
                break
    tmp.rename(dest)
    return n


def truncate_to_last_story(path: Path) -> None:
    """Cut the file right after the last complete separator line."""
    with open(path, "rb+") as f:
        f.seek(0, 2)
        size = f.tell()
        tail = min(size, 1 << 20)
        f.seek(size - tail)
        buf = f.read()
        i = buf.rfind(SEP)
        if i < 0:
            raise RuntimeError("no story separator found in last MiB")
        end = size - tail + i + len(SEP)
        f.truncate(end)
        f.seek(end)
        f.write(b"\n")


def count_stories(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip() == SEP:
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mb", type=int, default=700, help="MiB of the train file to keep")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    valid = args.out / "valid.txt"
    train = args.out / "train.txt"
    if not valid.exists():
        print("downloading validation set ...")
        fetch(BASE + FILES["valid"], valid)
    if not train.exists():
        print(f"downloading first {args.train_mb} MiB of train set ...")
        fetch(BASE + FILES["train"], train, max_bytes=args.train_mb << 20)
        truncate_to_last_story(train)

    total = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    for p in (valid, train):
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MiB, {count_stories(p):,} stories")
    print(f"data/raw total: {total / 2**20:.1f} MiB")
    if total > MAX_DATA_BYTES:
        print("FAIL: raw data exceeds 1.5 GB budget")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
