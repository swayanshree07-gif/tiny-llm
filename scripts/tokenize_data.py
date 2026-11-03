"""Tokenise data/raw/{train,valid}.txt into data/tok/{train,valid}.bin (uint16).

Stories are separated by the <|endoftext|> id. The file is read in blocks cut
at story boundaries so memory stays small. Writes data/tok/meta.json.

Run: .venv/bin/python scripts/tokenize_data.py
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

from tinyllm.tokenizer import EOT, Tokenizer  # noqa: E402

SEP = EOT.encode()


def iter_blocks(path: Path, block: int = 16 << 20):
    """Yield text blocks that end on a story separator (last one may not)."""
    rest = b""
    with open(path, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                break
            buf = rest + buf
            cut = buf.rfind(SEP)
            if cut < 0:
                rest = buf
                continue
            cut += len(SEP)
            if cut == len(buf):  # need to see whether a newline follows
                rest = buf
                continue
            if buf[cut:cut + 1] == b"\n":
                cut += 1
            yield buf[:cut].decode("utf-8", errors="ignore")
            rest = buf[cut:]
    if rest.strip():
        yield rest.decode("utf-8", errors="ignore")


def normalize(text: str) -> str:
    """Each story -> stripped story text + EOT; empty stories dropped."""
    return "".join(s.strip() + EOT for s in text.split(EOT) if s.strip())


def encode_file(tok: Tokenizer, src: Path, dst: Path) -> dict:
    t0 = time.perf_counter()
    n_tokens = n_bytes = 0
    tmp = dst.with_suffix(".bin.part")
    with open(tmp, "wb") as f:
        for text in iter_blocks(src):
            text = normalize(text)
            ids = np.asarray(tok.encode(text), dtype=np.uint16)
            f.write(ids.tobytes())
            n_tokens += len(ids)
            n_bytes += len(text.encode())
    tmp.rename(dst)
    dt = time.perf_counter() - t0
    print(f"{src.name}: {n_tokens:,} tokens, {n_bytes / n_tokens:.2f} bytes/token, {dt:.0f}s")
    return {"tokens": n_tokens, "bytes": n_bytes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/tok/tokenizer.json")
    ap.add_argument("--splits", nargs="+", default=["valid", "train"])
    args = ap.parse_args()
    tok = Tokenizer.load(args.tokenizer)
    assert tok.vocab_size <= 65536
    meta = {"vocab_size": tok.vocab_size, "eot_id": tok.eot_id, "dtype": "uint16", "splits": {}}
    meta_path = args.out / "meta.json"
    if meta_path.exists():
        meta["splits"] = json.loads(meta_path.read_text()).get("splits", {})
    for split in args.splits:
        meta["splits"][split] = encode_file(tok, args.raw / f"{split}.txt", args.out / f"{split}.bin")
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
