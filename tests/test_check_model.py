import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_model.py"
ARGS = ["--random", "--cpu", "--batch-size", "2"]


def _load():
    spec = importlib.util.spec_from_file_location("check_model", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_model_random_ok():
    r = subprocess.run([sys.executable, str(SCRIPT), *ARGS], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout
    assert "[PASS] [rope] whole-model causality" in r.stdout


def test_expected_params_matches_default_model():
    from tinyllm.model import GPT, GPTConfig

    mod = _load()
    for pos in ("rope", "learned"):
        c = GPTConfig(pos=pos)
        assert GPT(c).num_params(False) == mod.expected_params(c)


def test_check_model_catches_bad_init(monkeypatch, capsys):
    """A too-large init (std 1.0) must fail the init-loss and init-std checks."""
    import tinyllm.model as model

    def bad_init(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(m.weight, std=1.0)

    monkeypatch.setattr(model.GPT, "_init_weights", staticmethod(bad_init))
    monkeypatch.setattr(sys, "argv", ["check_model.py", *ARGS])
    assert _load().main() == 1
    out = capsys.readouterr().out
    assert "[FAIL] [rope] init loss ~ ln(V) on real tokens" in out
    assert "[FAIL] [rope] init stds / LayerNorm params as designed" in out
    assert "[PASS] [rope] whole-model causality" in out
