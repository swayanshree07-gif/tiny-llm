"""scripts/demo.sh: bad args are rejected; --quick runs end-to-end when checkpoints exist."""
import glob
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo.sh"


def test_demo_rejects_unknown_arg():
    r = subprocess.run(["bash", str(DEMO), "--bogus"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "unknown arg" in r.stderr


@pytest.mark.skipif(not (glob.glob(str(ROOT / "ckpt/base/ckpt_*.pt"))
                         and glob.glob(str(ROOT / "ckpt/chat_large_v3/ckpt_*.pt"))),
                    reason="checkpoints not present")
def test_demo_quick_runs():
    r = subprocess.run(["bash", str(DEMO), "--quick"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    assert "--- sample 1 ---" in out and "--- sample 2 ---" in out
    assert out.count("Assistant:") == 2
    assert "[demo] DONE" in out
