from tinyllm.quality import distinct_n, repeated_ngram_frac, story_stats, words


def test_words():
    assert words("Hi, Tom's dog!") == ["hi", ",", "tom's", "dog", "!"]


def test_repetition_metrics():
    fresh = "the cat sat on a warm mat and then slept"
    loop = "the cat sat. " * 6
    assert repeated_ngram_frac(fresh) == 0.0 and distinct_n(fresh) == 1.0
    assert repeated_ngram_frac(loop) > 0.7 and distinct_n(loop) < 0.3
    assert repeated_ngram_frac("a b") == 0.0 and distinct_n("a") == 1.0


def test_story_stats():
    s = story_stats("one two three", ended=True)
    assert s == {"words": 3, "ended": True, "distinct2": 1.0, "rep4": 0.0}


def test_sample_suite_run_tiny():
    import sys
    from pathlib import Path

    import pytest
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import sample_suite
    from tinyllm.model import GPT, GPTConfig
    from tinyllm.tokenizer import Tokenizer

    tok = Tokenizer.train("Once upon a time there was a cat. " * 20, 270)
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layers=1, n_heads=2, d_model=16)).eval()
    res = list(sample_suite.run(m, tok, torch.device("cpu"), ["Once", "A cat"],
                                sample_suite.SETTINGS[:2], max_new=10, seed=0))
    assert len(res) == 4
    assert all(text.startswith(p) and set(s) == {"words", "ended", "distinct2", "rep4"}
               for p, _, text, s in res)
    again = list(sample_suite.run(m, tok, torch.device("cpu"), ["Once", "A cat"],
                                  sample_suite.SETTINGS[:2], max_new=10, seed=0))
    assert [r[2] for r in res] == [r[2] for r in again]  # seeded -> reproducible
