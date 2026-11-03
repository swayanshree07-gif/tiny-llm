from tinyllm.bpe_ref import RefBPE, merge


def test_merge():
    assert merge([1, 2, 2, 2, 1, 2], (1, 2), 9) == [9, 2, 2, 9]
    assert merge([2, 2, 2], (2, 2), 9) == [9, 2]  # non-overlapping, left to right


def test_hand_worked_example():
    # "aaabdaaabac" (Wikipedia BPE example): first merge is (a,a), then (aa,a)->"aaa"?
    # counts: aa=4, ab=2, ... -> merge (97,97)=256 ; ids: 256 a b d 256 a b a c
    # then (256,97)=2 ties with (97,98)=2 -> smaller pair (97,98) wins = 257
    t = RefBPE()
    t.train("aaabdaaabac", 256 + 3)
    assert list(t.merges) == [(97, 97), (97, 98), (256, 257)]
    assert t.vocab[258] == b"aaab"
    assert t.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_roundtrip_unicode():
    text = "Once upon a time, a little cat 🐱 met a café owner. Ünïcödé — 日本語!"
    t = RefBPE()
    t.train(text * 3, 300)
    for s in [text, "", "a", "brand new text 🐶 not in training"]:
        assert t.decode(t.encode(s)) == s
    assert len(t.encode(text)) < len(text.encode("utf-8"))
