import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_overfit.py"
ARGS = ["--random", "--cpu", "--batch-size", "4"]


def _load():
    spec = importlib.util.spec_from_file_location("check_overfit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_overfit_random_ok():
    r = subprocess.run([sys.executable, str(SCRIPT), *ARGS], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout


def test_check_overfit_catches_broken_optimizer(monkeypatch, capsys):
    """If the optimizer never updates weights (lr 0), the overfit check must fail."""
    mod = _load()
    assert mod.main([*ARGS, "--lr", "0", "--steps", "40", "--control-steps", "10"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] loss < 0.1 within 40 steps" in out
    assert "[PASS] step-1 loss ~ ln(V)" in out
    assert "[PASS] temp dir removed" in out
