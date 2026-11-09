from pathlib import Path

import numpy as np
import pytest

from tinyllm import chat as C
from tinyllm.instruct import IGNORE
from tinyllm.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
STORY = ("Once upon a time, there was a little dog named Max. Max loved his red ball. "
         "One day, Max took the ball to the park and played with a happy girl. The ball was red.")
MSGS = [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": 'Tell me a story about "ball".'}, {"role": "assistant", "content": STORY}]


@pytest.fixture(scope="module")
def tok():
    p = ROOT / "data/tok/tokenizer.json"
    if p.exists():
        return Tokenizer.load(p)
    return Tokenizer.train((STORY + " User: Assistant: Hello Tell me a story ") * 30, 400)


def test_template_text(tok):
    ids, _ = C.encode_chat(tok, MSGS)
    eot = tok.vocab_size - 1
    text = tok.decode([t for t in ids if t != eot])
    assert text.startswith("User: Hello!\nAssistant: Hi there.\nUser: Tell me")
    assert ids.count(eot) == 2 and ids[-1] == eot


def test_round_trip(tok):
    ids, _ = C.encode_chat(tok, MSGS)
    assert C.decode_chat(tok, ids) == MSGS


def test_mask_only_assistant(tok):
    ids, mask = C.encode_chat(tok, MSGS)
    eot = tok.vocab_size - 1
    masked = [t for t, m in zip(ids, mask) if m]
    # masked tokens are exactly " reply" + EOT for each assistant turn
    want = tok.encode_ordinary(" Hi there.") + [eot] + tok.encode_ordinary(" " + STORY) + [eot]
    assert masked == want
    unmasked = tok.decode([t for t, m in zip(ids, mask) if not m])
    assert "Hello!" in unmasked and "Assistant:" in unmasked and "Max" not in unmasked


def test_generation_prompt_ends_with_assistant(tok):
    ids, _ = C.encode_chat(tok, MSGS[:3], add_generation_prompt=True)
    assert tok.decode(ids).endswith("\nAssistant:")
    with pytest.raises(ValueError):
        C.encode_chat(tok, MSGS, add_generation_prompt=True)
    with pytest.raises(ValueError):
        C.encode_chat(tok, [{"role": "assistant", "content": "x"}])


def test_fit_row_targets_and_truncation(tok):
    x, y, keep = C.fit_row(tok, MSGS, 256)
    ids, mask = C.encode_chat(tok, MSGS[:2 * keep])
    assert len(x) == len(y) == 256
    y = np.asarray(y)
    tgt = np.flatnonzero(y != IGNORE)
    # every target is the next input token, and only assistant positions are targets
    for i in tgt:
        assert mask[i + 1] and y[i] == ids[i + 1]
    assert len(tgt) == sum(mask[1:])
    # a tiny block drops the long story turn and keeps the small-talk turn
    short = len(C.encode_chat(tok, MSGS[:2])[0])
    x2, y2, keep2 = C.fit_row(tok, MSGS, short - 1)
    assert keep2 == 1
    assert C.fit_row(tok, MSGS, 3) is None


def test_build_chat_rows(tok):
    stories = [STORY] * 30
    x, y, convs = C.build_chat_rows(tok, stories, 256, seed=0)
    assert x.shape == y.shape and x.shape[0] == len(convs) > 0
    assert ((y == IGNORE) | (y >= 0)).all()
    for (msgs, specs), row in zip(convs, y):
        assert len(msgs) == 2 * len(specs)
        assert (row != IGNORE).sum() > 0
        for m, s in zip(msgs[1::2], specs):
            if s["kind"] == "words":
                assert all(w in m["content"].lower() for w in s["words"])
