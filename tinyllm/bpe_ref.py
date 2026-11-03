"""Reference byte-level BPE: deliberately simple and slow, used as ground truth.

Training repeatedly counts all adjacent pairs and merges the most frequent one.
Ties are broken by the smallest pair (lexicographic on token ids) so results are
deterministic; the fast trainer (Day 4) must reproduce the same merges.
Token ids 0..255 are raw bytes; merge i creates id 256 + i.
"""
from __future__ import annotations

from collections import Counter


def get_pair_counts(ids: list[int], counts: Counter | None = None) -> Counter:
    counts = Counter() if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair (left to right) by new_id."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def best_pair(counts: Counter) -> tuple[int, int]:
    # highest count, ties -> smallest pair
    return min(counts, key=lambda p: (-counts[p], p))


class RefBPE:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            counts = get_pair_counts(ids)
            if not counts:
                break
            pair = best_pair(counts)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge present
            pair = min(get_pair_counts(ids), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
