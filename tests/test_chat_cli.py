"""scripts/chat.py driven through stdin with a tiny random model (PLAN Day 23)."""
import importlib.util
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tinyllm.chat import encode_chat  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKP = ROOT / "data/tok/tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKP.exists(), reason="needs data/tok/tokenizer.json")

spec = importlib.util.spec_from_file_location("chat_cli", ROOT / "scripts/chat.py")
chat_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_cli)


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.load(TOKP)


@pytest.fixture
def ckpt(tmp_path, tok):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layers=2, n_heads=2, d_model=32)
    m = GPT(cfg)
    p = tmp_path / "ckpt_0000001.pt"
    torch.save({"model": m.state_dict(), "model_config": cfg.to_dict(), "step": 1}, p)
    return tmp_path


def run(ckpt, lines, monkeypatch, capsys, *extra):
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(l + "\n" for l in lines)))
    rc = chat_cli.main(["--ckpt", str(ckpt), "--device", "cpu", "--seed", "0", "--max-new", "20", *extra])
    return rc, capsys.readouterr()


def test_scripted_session(ckpt, monkeypatch, capsys):
    rc, out = run(ckpt, ["Hello!", "Tell me a story.", "Thanks", "never reached"], monkeypatch, capsys,
                  "--max-turns", "3")
    assert rc == 0
    assert out.out.count("Assistant:") == 3
    assert "never reached" not in out.out


def test_seed_is_deterministic_and_reset(ckpt, monkeypatch, capsys):
    _, a = run(ckpt, ["Hi", "/reset", "Hi"], monkeypatch, capsys)
    _, b = run(ckpt, ["Hi", "/reset", "Hi"], monkeypatch, capsys)
    assert a.out == b.out and "[history cleared]" in a.out


def test_history_truncation(ckpt, monkeypatch, capsys):
    # block 64, max_new 20 -> 44 prompt tokens: long messages force old turns out
    msgs = ["Write a short story that uses the words cat, dog and tree please."] * 4
    rc, out = run(ckpt, msgs, monkeypatch, capsys)
    assert rc == 0 and out.out.count("Assistant:") == 4
    assert "dropped" in out.err


def test_build_prompt_fits_and_ends_with_generation_prompt(tok):
    hist = []
    for i in range(5):
        hist += [{"role": "user", "content": f"story {i}"}, {"role": "assistant", "content": "Once upon a time." * 5}]
    ids, dropped = chat_cli.build_prompt(tok, hist, "one more", 128, 60)
    assert len(ids) <= 68 and dropped >= 1
    full, _ = encode_chat(tok, hist[2 * dropped:] + [{"role": "user", "content": "one more"}],
                          add_generation_prompt=True)
    assert ids == full
    long_ids, _ = chat_cli.build_prompt(tok, [], "word " * 500, 128, 60)
    assert len(long_ids) == 68 and tok.decode(long_ids).endswith("Assistant:")
    with pytest.raises(ValueError):
        chat_cli.build_prompt(tok, [], "x", 64, 60)


def test_min_new_keeps_history_and_fits_context(ckpt, tok):
    from tinyllm.generate import load_model
    model, _ = load_model(str(ckpt), "cpu")
    hist = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}]
    g = torch.Generator().manual_seed(0)
    # Reserving the full max_new (old behaviour) drops the history ...
    _, d_old = chat_cli.reply(model, tok, hist, "Tell me more.", 50, 0.8, None, 0.95, g)
    # ... a small min_new keeps it, and generation stays within block_size (no error).
    text, d_new = chat_cli.reply(model, tok, hist, "Tell me more.", 50, 0.8, None, 0.95, g, min_new=10)
    assert d_old == 1 and d_new == 0 and isinstance(text, str)
