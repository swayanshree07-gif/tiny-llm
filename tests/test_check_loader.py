import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_loader.py"


def _tok_dir(tmp_path: Path, bad_id: bool = False) -> Path:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 8191, size=20_000).astype(np.uint16)
    if bad_id:
        ids[::50] = 9000  # >= vocab_size, must be caught
    ids.tofile(tmp_path / "valid.bin")
    meta = {"vocab_size": 8192, "eot_id": 8191, "dtype": "uint16",
            "splits": {"valid": {"tokens": len(ids), "bytes": 4 * len(ids)}}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def _run(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--tok-dir", str(d), "--instruct-dir", str(d / "none"),
                           "--batches", "5", "--batch-size", "4", "--block", "32", "--cpu"],
                          capture_output=True, text=True, timeout=120)


def test_check_loader_ok(tmp_path):
    r = _run(_tok_dir(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout and "[PASS] valid: y == x shifted by one" in r.stdout


def test_check_loader_catches_bad_ids(tmp_path):
    r = _run(_tok_dir(tmp_path, bad_id=True))
    assert r.returncode == 1
    assert "[FAIL] valid: ids in [0, 8192)" in r.stdout
