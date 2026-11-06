"""Instruction-style fine-tune data (PLAN Day 27), built from the stories we already have.

No extra download: each story is turned into a (prompt, story) pair whose prompt can be
checked automatically afterwards.

    Prompt: Write a short story that uses the words "ball", "park" and "happy".
    Story: <the story><|endoftext|>        (prompt ends at "Story:", story tokens start " Once")

or

    Prompt: Write a short story that begins with: "<first sentence>"
    Story: <the story><|endoftext|>

Prompts are plain text (no new special tokens, the vocab stays 8192). Each example is one row
of block_size + 1 tokens: prompt + story + EOT, then EOT padding. Targets for the prompt and the
padding are -100 (F.cross_entropy's default ignore_index), so only the story and its first EOT
are learned. Stories that don't fit in one row are skipped.
"""
from __future__ import annotations

import re

import numpy as np
import torch

PROMPT_PREFIX = "Prompt: "
STORY_PREFIX = "\nStory:"  # the story is encoded with a leading space (" Once")
IGNORE = -100

STOPWORDS = set("""
a about after again all also an and any are as at away back be because been before being big
but by came can could day did do does down each even every for from get go going good got had
has have he her here him his how i if in into is it its just like little look looked made make
many me more much my new no not now of off oh on once one only or other our out over said saw
say see she so some than that the their them then there they thing things this time to too
two up upon us very was way we went were what when where which while who why will with would
yes you your
""".split())
# whole lowercase words only. Before Day 8 this was r"[a-z]{4,}", which also matched the tail of
# capitalised words ("Suddenly" -> "uddenly", "Tommy" -> "ommy"): ~9% of v1/v2 "words" prompts asked
# for a fragment that no story can contain. Capitalised words (names, sentence starts) are skipped.
_WORD = re.compile(r"(?<![A-Za-z])[a-z]{4,}(?![A-Za-z])")


def candidate_words(story: str) -> list[str]:
    """Lowercase content words (>= 4 letters, not stop words), in first-seen order, unique."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(story):
        if w not in STOPWORDS:
            seen.setdefault(w, None)
    return list(seen)


def salient_words(story: str, min_count: int = 2) -> list[str]:
    """Content words (as in candidate_words) that occur at least min_count times, most frequent first.

    These carry the story's topic (characters, objects, places), so a prompt made of them says
    much more about the story than random content words like "decided" or "suddenly" (fine-tune v3).
    """
    counts: dict[str, int] = {}
    for w in _WORD.findall(story):
        if w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts, key=lambda w: -counts[w])  # stable: ties keep first-seen order
    return [w for w in ranked if counts[w] >= min_count]


def first_sentence(story: str) -> str | None:
    m = re.match(r"\s*(.{20,200}?[.!?])(\s|$)", story, flags=re.S)
    if not m or "\n" in m.group(1) or '"' in m.group(1):
        return None
    return m.group(1)


def words_prompt(words: list[str]) -> str:
    q = [f'"{w}"' for w in words]
    listed = ", ".join(q[:-1]) + " and " + q[-1] if len(q) > 1 else q[0]
    return f"{PROMPT_PREFIX}Write a short story that uses the words {listed}.{STORY_PREFIX}"


def begins_prompt(sentence: str) -> str:
    return f'{PROMPT_PREFIX}Write a short story that begins with: "{sentence}"{STORY_PREFIX}'


def make_example(story: str, rng: np.random.Generator, n_words: int = 3,
                 word_mode: str = "any") -> tuple[str, dict] | None:
    """Return (prompt, spec) for a story, or None if no usable prompt. spec says what to check.

    word_mode "any": the words are drawn from all content words (v1/v2);
    "salient": from the words that occur >= 2 times in the story (v3).
    """
    if rng.random() < 0.75:
        if word_mode == "salient":
            cands = salient_words(story)
        elif word_mode == "any":
            cands = candidate_words(story)
        else:
            raise ValueError(f"unknown word_mode {word_mode!r}")
        if len(cands) < n_words:
            return None
        words = sorted(rng.choice(cands, size=n_words, replace=False).tolist())
        return words_prompt(words), {"kind": "words", "words": words}
    s = first_sentence(story)
    if s is None:
        return None
    return begins_prompt(s), {"kind": "begins", "sentence": s}


def follows(spec: dict, story: str) -> bool:
    """Does a generated story satisfy the prompt? (all words present / starts with the sentence)."""
    if spec["kind"] == "words":
        found = set(re.findall(r"[a-z]+", story.lower()))
        return all(w in found for w in spec["words"])
    return story.lstrip().startswith(spec["sentence"])


def words_found(spec: dict, story: str) -> float:
    """Fraction of the requested words that appear in the story (partial credit for "words" prompts)."""
    found = set(re.findall(r"[a-z]+", story.lower()))
    return sum(w in found for w in spec["words"]) / len(spec["words"])


def build_rows(tok, stories: list[str], block_size: int, seed: int = 0, word_mode: str = "any"):
    """Encode examples into (x, y) int arrays of shape (N, block_size); y = -100 outside the story.

    Returns x, y (np.int64) and the list of (prompt, spec) actually used.
    """
    rng = np.random.default_rng(seed)
    eot = tok.vocab_size - 1
    xs, ys, used = [], [], []
    for story in stories:
        ex = make_example(story, rng, word_mode=word_mode)
        if ex is None:
            continue
        prompt, spec = ex
        p = tok.encode_ordinary(prompt)
        s = tok.encode_ordinary(" " + story) + [eot]
        if len(p) + len(s) > block_size + 1:
            continue
        row = p + s + [eot] * (block_size + 1 - len(p) - len(s))
        tgt = [IGNORE] * (len(p) - 1) + s + [IGNORE] * (block_size + 1 - len(p) - len(s))
        xs.append(row[:-1])
        ys.append(tgt)
        used.append((prompt, spec))
    x = np.asarray(xs, dtype=np.int64).reshape(-1, block_size)
    y = np.asarray(ys, dtype=np.int64).reshape(-1, block_size)
    return x, y, used


def stories_from_tokens(tok, ids: np.ndarray, n: int, seed: int = 0, max_tokens: int = 400) -> list[str]:
    """Decode n random whole stories (between two EOTs) from a token memmap."""
    eot = tok.vocab_size - 1
    rng = np.random.default_rng(seed)
    out: list[str] = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        s = int(rng.integers(0, len(ids) - 2 * max_tokens))
        w = np.asarray(ids[s:s + 2 * max_tokens])
        cuts = np.flatnonzero(w == eot)
        if len(cuts) < 2:
            continue
        out.append(tok.decode(w[cuts[0] + 1:cuts[1]].tolist()))
    return out


def word_target_mask(vocab: list[bytes], y: np.ndarray, words: list[str]) -> np.ndarray:
    """Bool mask over target positions of one row: True where the target token is (part of) a requested word.

    The story is rebuilt from the target tokens (y != -100) as bytes; every whole-word,
    case-insensitive occurrence of a requested word marks all tokens overlapping it
    (fine-tune v4 up-weights the loss on these tokens). Prompt positions are never marked.
    """
    pos = np.flatnonzero(y != IGNORE)
    pieces = [vocab[int(t)] for t in y[pos]]
    text = b"".join(pieces)
    lens = np.array([len(b) for b in pieces], dtype=np.int64)
    ends = np.cumsum(lens)
    starts = ends - lens
    mask = np.zeros(y.shape, dtype=bool)
    for w in words:
        for m in re.finditer(rb"(?<![A-Za-z])" + re.escape(w.encode()) + rb"(?![A-Za-z])", text, flags=re.I):
            hit = (starts < m.end()) & (ends > m.start())
            mask[pos[hit]] = True
    return mask


class InstructDataset:
    """Random rows of a prebuilt (x, y) set; same get_batch/state_dict API as TokenDataset.

    With per-position loss weights `w` (same shape, float), get_batch returns (x, y, w).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int = 0, w: np.ndarray | None = None) -> None:
        assert x.shape == y.shape and x.ndim == 2
        self.x, self.y = torch.from_numpy(x), torch.from_numpy(y)
        self.w = None
        if w is not None:
            assert w.shape == x.shape
            self.w = torch.from_numpy(np.asarray(w, dtype=np.float32))
        self.block_size = x.shape[1]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def get_batch(self, batch_size: int, device: torch.device | str = "cpu"):
        idx = torch.from_numpy(self.rng.integers(0, len(self.x), size=batch_size))
        out = [self.x[idx], self.y[idx]] + ([self.w[idx]] if self.w is not None else [])
        device = torch.device(device)
        if device.type == "cuda":
            out = [t.pin_memory().to(device, non_blocking=True) for t in out]
        return tuple(out)

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
