"""Chat template and chat fine-tune data (PLAN Day 21).

Template (plain text on the existing 8192 vocab, no new special tokens):

    User: <message>
    Assistant: <reply><|endoftext|>
    User: <message>
    Assistant: <reply><|endoftext|>

Each assistant reply ends with the EOT token, so generation stops there and the next turn starts
with "\\nUser:". Replies are encoded with a leading space (" Once"), like stories in pre-training.
Only assistant reply tokens (and their closing EOT) are trained on; every other target is -100.

The data is built from our own stories (no download), reusing the instruct prompts:
  - "words"/"begins" requests -> a story that satisfies them (checkable afterwards),
  - "Tell me a story." -> a story,
  - a few fixed small-talk turns (greeting / what can you do / thanks) with fixed replies.
Conversations have 1-3 turns; a conversation is packed into one row of block_size + 1 tokens
(EOT padding), and turns that don't fit are dropped from the end (at least one assistant turn kept).
"""
from __future__ import annotations

import numpy as np

from tinyllm.instruct import IGNORE, first_sentence, candidate_words, salient_words

USER = "User: "
ASSISTANT = "\nAssistant:"  # reply text follows with a leading space
TURN_SEP = "\n"             # after the reply's EOT, before the next "User: "

SMALL_TALK = [
    ("Hello!", "Hello! I can write short stories for you. Give me some words or a first sentence."),
    ("Hi, who are you?", "I am a small story bot. Ask me for a short story and I will write one."),
    ("What can you do?", "I can write short stories. Tell me words to use or how the story should begin."),
    ("Thank you!", "You are welcome! Do you want another story?"),
]


def encode_chat(tok, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[bool]]:
    """Encode [{"role": "user"|"assistant", "content": str}, ...] -> (ids, train_mask).

    train_mask[i] is True when ids[i] is an assistant reply token or the EOT closing a reply.
    With add_generation_prompt the ids end with "\\nAssistant:" after the last user message.
    """
    eot = tok.vocab_size - 1
    ids: list[int] = []
    mask: list[bool] = []
    for i, m in enumerate(messages):
        if m["role"] == "user":
            t = tok.encode_ordinary((TURN_SEP if i else "") + USER + m["content"] + ASSISTANT)
            ids += t
            mask += [False] * len(t)
        elif m["role"] == "assistant":
            if i == 0 or messages[i - 1]["role"] != "user":
                raise ValueError("assistant message must follow a user message")
            t = tok.encode_ordinary(" " + m["content"]) + [eot]
            ids += t
            mask += [True] * len(t)
        else:
            raise ValueError(f"unknown role {m['role']!r}")
    if add_generation_prompt and (not messages or messages[-1]["role"] != "user"):
        raise ValueError("generation prompt needs a final user message")
    return ids, mask


def decode_chat(tok, ids: list[int]) -> list[dict]:
    """Inverse of encode_chat (for complete conversations): text -> list of messages."""
    eot = tok.vocab_size - 1
    out: list[dict] = []
    # split on EOT: each piece is "[\n]User: msg\nAssistant: reply"
    pieces, cur = [], []
    for t in ids:
        if t == eot:
            pieces.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        pieces.append(cur)
    for p in pieces:
        text = tok.decode(p)
        if text.startswith(TURN_SEP):
            text = text[len(TURN_SEP):]
        if not text.startswith(USER) or ASSISTANT not in text:
            raise ValueError(f"not a chat turn: {text!r}")
        user, reply = text[len(USER):].split(ASSISTANT, 1)
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": reply[1:] if reply.startswith(" ") else reply})
    return out


def story_request(story: str, rng: np.random.Generator) -> tuple[str, dict] | None:
    """User message asking for this story, plus a checkable spec (kind words/begins/free)."""
    r = rng.random()
    if r < 0.55:
        cands = salient_words(story) if rng.random() < 0.5 else candidate_words(story)
        n = int(rng.integers(1, 4))
        if len(cands) < n:
            return None
        words = sorted(rng.choice(cands, size=n, replace=False).tolist())
        q = [f'"{w}"' for w in words]
        listed = ", ".join(q[:-1]) + " and " + q[-1] if n > 1 else q[0]
        noun = "words" if n > 1 else "word"
        msg = rng.choice([f"Write a short story that uses the {noun} {listed}.",
                          f"Can you tell me a story with the {noun} {listed}?",
                          f"Please write a story about {listed}."]) if n > 1 else \
            rng.choice([f"Write a short story that uses the word {listed}.",
                        f"Can you tell me a story with the word {listed}?",
                        f"Tell me a story about {q[0]}."])
        return str(msg), {"kind": "words", "words": words}
    if r < 0.85:
        s = first_sentence(story)
        if s is None:
            return None
        msg = rng.choice([f'Write a short story that begins with: "{s}"',
                          f'Continue this story: "{s}"'])
        return str(msg), {"kind": "begins", "sentence": s}
    msg = rng.choice(["Tell me a story.", "Write a short story.", "Can you tell me a story?",
                      "Another story, please."])
    return str(msg), {"kind": "free"}


def make_conversation(stories: list[str], rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """A 1-3 turn conversation from up to 3 stories (consumed from the front). Returns (messages, specs)."""
    msgs: list[dict] = []
    specs: list[dict] = []
    n_turns = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
    if rng.random() < 0.1:  # small talk opener
        u, a = SMALL_TALK[int(rng.integers(0, 2 if rng.random() < 0.7 else 3))]
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        specs.append({"kind": "smalltalk"})
    while stories and len(specs) < n_turns + (1 if specs and specs[0]["kind"] == "smalltalk" else 0):
        story = stories.pop(0)
        req = story_request(story, rng)
        if req is None:
            continue
        msgs += [{"role": "user", "content": req[0]}, {"role": "assistant", "content": story}]
        specs.append(req[1])
    if len(specs) > 1 and rng.random() < 0.15:
        msgs += [{"role": "user", "content": SMALL_TALK[3][0]}, {"role": "assistant", "content": SMALL_TALK[3][1]}]
        specs.append({"kind": "smalltalk"})
    return msgs, specs


def fit_row(tok, messages: list[dict], block_size: int) -> tuple[list[int], list[int], int] | None:
    """Encode, dropping whole (user, assistant) turns from the end until it fits in block_size + 1.

    Returns (x, y, n_turns_kept) with y = -100 except assistant tokens, or None if one turn doesn't fit.
    """
    eot = tok.vocab_size - 1
    for keep in range(len(messages) // 2, 0, -1):
        ids, mask = encode_chat(tok, messages[:2 * keep])
        if len(ids) <= block_size + 1:
            pad = block_size + 1 - len(ids)
            row = ids + [eot] * pad
            tgt = [t if m else IGNORE for t, m in zip(ids[1:], mask[1:])] + [IGNORE] * pad
            return row[:-1], tgt, keep
    return None


def build_chat_rows(tok, stories: list[str], block_size: int, seed: int = 0):
    """Stories -> (x, y, convs). convs = list of (messages kept, specs kept) for each row."""
    rng = np.random.default_rng(seed)
    pool = list(stories)
    xs, ys, convs = [], [], []
    while pool:
        msgs, specs = make_conversation(pool, rng)
        if not specs:
            continue
        r = fit_row(tok, msgs, block_size)
        if r is None:
            continue
        x, y, keep = r
        xs.append(x)
        ys.append(y)
        convs.append((msgs[:2 * keep], specs[:keep]))
    x = np.asarray(xs, dtype=np.int64).reshape(-1, block_size)
    y = np.asarray(ys, dtype=np.int64).reshape(-1, block_size)
    return x, y, convs
