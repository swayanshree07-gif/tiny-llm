import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / "data/tok/tokenizer.json", ROOT / "data/tok/valid.bin", ROOT / "data/raw/valid.txt"]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="tokenized data not present")
def test_check_tokenizer_on_val_slice():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/check_tokenizer.py"), "--mb", "1"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout
    ratio = float(r.stdout.split("compression: ")[1].split()[0])
    assert 3.5 < ratio < 5.0
