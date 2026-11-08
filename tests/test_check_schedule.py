import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_schedule.py"


def _args(tmp_path):
    return ["--random", "--cpu", "--batch-size", "4", "--steps", "40", "--warmup", "8",
            "--fig", str(tmp_path / "s.png"), "--run-fig", str(tmp_path / "r.png")]


def test_check_schedule_random_ok(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), *_args(tmp_path)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout
    assert (tmp_path / "s.png").exists() and (tmp_path / "r.png").exists()


def test_check_schedule_catches_wrong_warmup(tmp_path):
    """If the run uses a different warmup than the checked schedule, the check must fail."""
    r = subprocess.run([sys.executable, str(SCRIPT), *_args(tmp_path), "--run-warmup", "2"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[FAIL] logged LR == schedule at every step" in r.stdout
