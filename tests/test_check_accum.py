import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_accum.py"
ARGS = ["--random", "--cpu"]


def _load():
    spec = importlib.util.spec_from_file_location("check_accum", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_accum_random_ok():
    r = subprocess.run([sys.executable, str(SCRIPT), *ARGS], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout


def test_check_accum_catches_missing_loss_scaling(capsys):
    """Summing micro-batch losses without 1/k must make the equivalence checks fail."""
    mod = _load()
    assert mod.main([*ARGS, "--break-scaling"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] fp32 grads: 2x(B/2) == 1xB" in out
    assert "[PASS] clip returns pre-clip norm" in out
