"""Cheap, automatic text-quality signals for generated stories (PLAN Day 26).

These don't measure "coherence", but catch the usual failure modes of small LMs:
degenerate loops (high repeated-n-gram fraction, low distinct-n) and stories that never end.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(seq: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(text: str, n: int = 2) -> float:
    """Unique n-grams / total n-grams (1.0 = no repetition). 1.0 for texts shorter than n."""
    g = ngrams(words(text), n)
    return len(set(g)) / len(g) if g else 1.0


def repeated_ngram_frac(text: str, n: int = 4) -> float:
    """Fraction of n-gram positions whose n-gram already appeared earlier in the text."""
    g = ngrams(words(text), n)
    seen, rep = set(), 0
    for x in g:
        rep += x in seen
        seen.add(x)
    return rep / len(g) if g else 0.0


def story_stats(text: str, ended: bool) -> dict:
    w = words(text)
    return {"words": len(w), "ended": ended, "distinct2": round(distinct_n(text, 2), 3),
            "rep4": round(repeated_ngram_frac(text, 4), 3)}
