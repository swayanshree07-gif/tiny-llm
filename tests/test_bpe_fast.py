import random

import pytest

from tinyllm.bpe import pretokenize, train_bpe
from tinyllm.bpe_ref import RefBPE


def _ref_merges(text, vocab_size):
    t = RefBPE()
    t.train(text, vocab_size)
    return list(t.merges)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_matches_reference_on_unsplit_text(seed):
    rng = random.Random(seed)
    words = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "é", "🐱", " "]
    text = "".join(rng.choice(words) for _ in range(400))
    assert train_bpe({text.encode(): 1}, 330) == _ref_merges(text, 330)


def test_hand_example():
    assert train_bpe({b"aaabdaaabac": 1}, 259) == [(97, 97), (97, 98), (256, 257)]


def test_word_counts_equal_repeated_words():
    # weights must behave like repeating separate words
    wc = {b"low": 5, b"lower": 2, b"newest": 6, b"widest": 3}
    merges = train_bpe(wc, 256 + 10)
    assert merges[0] in {(ord("e"), ord("s")), (ord("s"), ord("t"))}
    assert len(merges) == 10


def test_pretokenize_splits_words():
    wc = pretokenize("Once upon a time, a cat's hat.")
    assert wc[b" a"] == 2 and b"'s" in wc and b"," in wc
