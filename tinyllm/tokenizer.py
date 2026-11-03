"""Byte-level BPE tokenizer: training wrapper, encode/decode, save/load.

Id layout (vocab_size V):
  0..255           raw bytes
  256..V-2         merges, in the order they were learned
  V-1              the special token <|endoftext|> (story separator)

Text is split on the special token first, then pre-tokenised with the GPT-2
regex; each chunk is BPE-encoded independently (results cached per chunk).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import regex

from tinyllm.bpe import GPT2_PATTERN, pretokenize, train_bpe

EOT = "<|endoftext|>"


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], pattern: str = GPT2_PATTERN) -> None:
        self.merges = [tuple(m) for m in merges]
        self.pattern = pattern
        self._re = regex.compile(pattern)
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self.vocab[self.eot_id] = EOT.encode()
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.eot_id + 1

    # ---- training -------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int) -> "Tokenizer":
        """Train on text; the special token is never merged across."""
        counts: Counter = Counter()
        for part in text.split(EOT):
            counts.update(pretokenize(part))
        return cls(train_bpe(counts, vocab_size - 1))

    # ---- encode / decode -----------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            new_id, out, i, n = 256 + best_rank, [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating '<|endoftext|>' as plain characters."""
        out: list[int] = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str) -> list[int]:
        """Encode text; occurrences of '<|endoftext|>' become the special id."""
        parts = text.split(EOT)
        out: list[int] = []
        for i, part in enumerate(parts):
            if i:
                out.append(self.eot_id)
            out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"version": 1, "pattern": self.pattern, "special": EOT,
                        "merges": [list(m) for m in self.merges]})
        )

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in d["merges"]], d["pattern"])
