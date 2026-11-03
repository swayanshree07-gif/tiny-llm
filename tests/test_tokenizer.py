from tinyllm.tokenizer import EOT, Tokenizer

TEXT = (
    "Once upon a time, there was a little cat named Tom. Tom liked to play.\n"
    f"{EOT}\n"
    "One day, Lily saw a big red ball. She said, \"Wow!\" Ünïcödé café 🐱.\n"
    f"{EOT}\n"
) * 20


def test_roundtrip_and_special():
    tok = Tokenizer.train(TEXT, 300)
    assert tok.vocab_size == 300 and tok.eot_id == 299
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert ids.count(tok.eot_id) == 40
    assert len(ids) < 0.8 * len(TEXT.encode())
    for s in ["", "unseen words 日本語 🐶", "x" * 50]:
        assert tok.decode(tok.encode(s)) == s


def test_no_merge_contains_special_text():
    tok = Tokenizer.train(TEXT, 400)
    assert all(b"<|endoftext|>" not in tok.vocab[i] for i in range(tok.eot_id))


def test_encode_ordinary_keeps_special_as_text():
    tok = Tokenizer.train(TEXT, 300)
    assert tok.eot_id not in tok.encode_ordinary(EOT)
    assert tok.decode(tok.encode_ordinary(EOT)) == EOT


def test_save_load(tmp_path):
    tok = Tokenizer.train(TEXT, 320)
    tok.save(tmp_path / "tok.json")
    tok2 = Tokenizer.load(tmp_path / "tok.json")
    assert tok2.merges == tok.merges
    assert tok2.encode(TEXT) == tok.encode(TEXT)
