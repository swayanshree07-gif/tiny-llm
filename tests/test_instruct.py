from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tinyllm import instruct as I
from tinyllm.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
STORY = ("Once upon a time, there was a little dog named Max. Max loved his red ball. "
         "One day, Max took the ball to the park and played with a happy girl.")


@pytest.fixture(scope="module")
def tok():
    p = ROOT / "data/tok/tokenizer.json"
    if p.exists():
        return Tokenizer.load(p)
    sample = (STORY + " ") * 50 + "Prompt: Write a short story that uses the words begins with Story:"
    return Tokenizer.train(sample, 400)


def test_candidate_words_skip_stopwords_and_short():
    w = I.candidate_words(STORY)
    assert "ball" in w and "park" in w and "happy" in w and "loved" in w
    assert "there" not in w and "the" not in w and "max" not in w  # names are capitalised -> skipped
    assert len(w) == len(set(w))


def test_prompts_and_follows():
    p = I.words_prompt(["ball", "happy", "park"])
    assert p == 'Prompt: Write a short story that uses the words "ball", "happy" and "park".\nStory:'
    assert I.follows({"kind": "words", "words": ["ball", "park"]}, "The Ball was in the park!")
    assert not I.follows({"kind": "words", "words": ["ball", "zebra"]}, "The ball was in the park.")
    s = I.first_sentence(STORY)
    assert s == "Once upon a time, there was a little dog named Max."
    assert I.follows({"kind": "begins", "sentence": s}, " " + STORY)
    assert not I.follows({"kind": "begins", "sentence": s}, "Max was a dog.")


def test_make_example_words_come_from_story():
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(50):
        prompt, spec = I.make_example(STORY, rng)
        kinds.add(spec["kind"])
        assert prompt.startswith(I.PROMPT_PREFIX) and prompt.endswith(I.STORY_PREFIX)
        assert I.follows(spec, STORY)  # the real story always satisfies its own prompt
    assert kinds == {"words", "begins"}


def test_build_rows_masks_prompt_and_padding(tok):
    T = 96
    x, y, used = I.build_rows(tok, [STORY, STORY + " " + "x" * 2000], T, seed=1)
    assert x.shape == y.shape == (1, T)  # the long story doesn't fit and is skipped
    prompt, _ = used[0]
    p = tok.encode_ordinary(prompt)
    s = tok.encode_ordinary(" " + STORY) + [tok.vocab_size - 1]
    # x is the input row; y[t] is the target for x[t]
    assert x[0, :len(p)].tolist() == p
    assert (y[0, :len(p) - 1] == I.IGNORE).all()
    assert y[0, len(p) - 1:len(p) - 1 + len(s)].tolist() == s
    assert (y[0, len(p) - 1 + len(s):] == I.IGNORE).all()
    # where y is a real target it equals the next input token
    real = np.flatnonzero(y[0, :-1] != I.IGNORE)
    assert (y[0, real] == x[0, real + 1]).all()
    assert tok.decode(s[:-1]).strip() == STORY


def test_ignored_targets_do_not_change_loss():
    logits = torch.randn(1, 5, 11)
    y = torch.tensor([[I.IGNORE, 3, 4, I.IGNORE, I.IGNORE]])
    full = F.cross_entropy(logits.reshape(-1, 11), y.reshape(-1))
    ref = F.cross_entropy(logits[0, 1:3], y[0, 1:3])
    assert torch.allclose(full, ref)


def test_dataset_batches_and_state():
    x = np.arange(40, dtype=np.int64).reshape(10, 4)
    y = x + 1
    ds = I.InstructDataset(x, y, seed=3)
    st = ds.state_dict()
    a = ds.get_batch(6)
    ds.load_state_dict(st)
    b = ds.get_batch(6)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], a[0] + 1)
    assert a[0].shape == (6, 4) and a[0].dtype == torch.int64


@pytest.mark.skipif(not (ROOT / "data/tok/valid.bin").exists(), reason="token files not present")
def test_stories_from_tokens(tok):
    ids = np.memmap(ROOT / "data/tok/valid.bin", dtype=np.uint16, mode="r")
    st = I.stories_from_tokens(tok, ids, 5, seed=0)
    assert len(st) == 5 and all(len(s) > 50 and "<|endoftext|>" not in s for s in st)


def test_words_found_partial():
    spec = {"kind": "words", "words": ["ball", "park", "zebra"]}
    assert I.words_found(spec, "The ball is in the Park.") == pytest.approx(2 / 3)


def test_candidate_words_no_fragments_of_capitalised_words():
    w = I.candidate_words("Suddenly, Timmy saw a big dragon. Finally Lily smiled at the dragon.")
    assert w == ["dragon", "smiled"]


def test_requested_words_always_in_source_story():
    """Every words prompt must be satisfiable by its own story (regression for the fragment bug)."""
    stories = [STORY, "Suddenly, Tommy found a shiny shell. Tommy loved the shell and the sea. Everyone cheered.",
               "Finally the brave knight saw the castle. The knight ran to the castle, happy and tired."]
    for mode in ("any", "salient"):
        rng = np.random.default_rng(0)
        for _ in range(30):
            for s in stories:
                ex = I.make_example(s, rng, word_mode=mode)
                if ex is not None:
                    assert I.follows(ex[1], s), (mode, ex[1])


def test_salient_words():
    s = "The knight saw the castle. The knight ran to the castle. The dragon slept. Then the knight smiled."
    assert I.salient_words(s) == ["knight", "castle"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        ex = I.make_example(s + " The dragon woke up.", rng, word_mode="salient")
        if ex is not None and ex[1]["kind"] == "words":
            assert ex[1]["words"] == ["castle", "dragon", "knight"]
    with pytest.raises(ValueError):
        I.make_example(s, np.random.default_rng(0), word_mode="nope")
