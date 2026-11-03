"""Fast byte-level BPE trainer (incremental pair counts).

Text is first split into chunks by a GPT-2 style regex so merges never cross
word boundaries; identical chunks are counted once with a weight. Pair counts
are then updated incrementally: only words containing the merged pair are
touched. Tie-breaking matches `bpe_ref` (max count, then smallest pair), so on
an un-split text the merges are identical to the reference implementation.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

# GPT-2 pre-tokenisation pattern (needs the `regex` module for \p{..}).
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, pattern: str = GPT2_PATTERN) -> Counter:
    import regex

    return Counter(m.encode("utf-8") for m in regex.findall(pattern, text))


def train_bpe(word_counts: dict[bytes, int], vocab_size: int) -> list[tuple[int, int]]:
    """Return the ordered merge list; merge i creates token id 256 + i."""
    words: list[list[int]] = [list(w) for w in word_counts]
    weights: list[int] = list(word_counts.values())
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, ids in enumerate(words):
        w = weights[wi]
        for p in zip(ids, ids[1:]):
            pair_counts[p] += w
            where[p].add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < vocab_size - 256 and heap:
        negc, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -negc or negc == 0:
            continue  # stale heap entry
        new_id = 256 + len(merges)
        merges.append(pair)
        changed: set[tuple[int, int]] = set()
        for wi in list(where[pair]):
            ids, w = words[wi], weights[wi]
            for p in zip(ids, ids[1:]):  # remove old pairs of this word
                pair_counts[p] -= w
                changed.add(p)
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            words[wi] = out
            old_pairs = set(zip(ids, ids[1:]))
            new_pairs = set(zip(out, out[1:]))
            for p in zip(out, out[1:]):
                pair_counts[p] += w
                changed.add(p)
            for p in old_pairs - new_pairs:
                where[p].discard(wi)
            for p in new_pairs:
                where[p].add(wi)
        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)
        pair_counts.pop(pair, None)
        where.pop(pair, None)
    return merges
