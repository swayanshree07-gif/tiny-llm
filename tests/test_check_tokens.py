import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_tokens.py"
TOKJ = ROOT / "data/tok/tokenizer.json"


def _make_tok_dir(tmp_path: Path, corrupt: bool = False) -> Path:
    sys.path.insert(0, str(ROOT))
    from tinyllm.tokenizer import EOT, Tokenizer

    tok = Tokenizer.load(TOKJ)
    stories = [f"Once upon a time, a cat named Tom {i} found a red ball in the park." for i in range(200)]
    text = "".join(s + EOT for s in stories)
    ids = np.asarray(tok.encode(text), dtype=np.uint16)
    ids.tofile(tmp_path / "valid.bin")
    n_tokens = len(ids) + (7 if corrupt else 0)
    meta = {"vocab_size": tok.vocab_size, "eot_id": tok.vocab_size - 1, "dtype": "uint16",
            "splits": {"valid": {"tokens": n_tokens, "bytes": len(text.encode())}}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "tokenizer.json").write_bytes(TOKJ.read_bytes())
    return tmp_path


def _run(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--tok-dir", str(d), "--len", "32"],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(not TOKJ.exists(), reason="tokenizer not present")
def test_check_tokens_ok(tmp_path):
    r = _run(_make_tok_dir(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout and "200 stories" in r.stdout


@pytest.mark.skipif(not TOKJ.exists(), reason="tokenizer not present")
def test_check_tokens_detects_meta_mismatch(tmp_path):
    r = _run(_make_tok_dir(tmp_path, corrupt=True))
    assert r.returncode == 1
    assert "[FAIL] valid: size == meta tokens" in r.stdout
