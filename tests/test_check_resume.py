import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_resume.py"
ARGS = ["--random", "--cpu", "--batch-size", "4"]


def _load():
    spec = importlib.util.spec_from_file_location("check_resume", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_resume_random_ok():
    r = subprocess.run([sys.executable, str(SCRIPT), *ARGS], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout


def test_check_resume_catches_lost_data_state(capsys):
    """If the sampler state is not restored on resume, the check must fail."""
    mod = _load()
    assert mod.main([*ARGS, "--break-data-state"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] train losses resumed == straight" in out
    assert "[PASS] temp dir removed" in out
