import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_rope.py"
ARGS = ["--random", "--cpu", "--batch-size", "2"]


def _load():
    spec = importlib.util.spec_from_file_location("check_rope", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_rope_random_ok():
    r = subprocess.run([sys.executable, str(SCRIPT), *ARGS], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout
    assert "[PASS] <R_m q, R_n k> depends only on m - n" in r.stdout
    assert "[PASS] rope causality" in r.stdout


def test_complex_reference_matches_apply_rope():
    from tinyllm.model import apply_rope, rope_cache

    mod = _load()
    cos, sin = rope_cache(32, 8)
    x = torch.randn(2, 32, 8)
    torch.testing.assert_close(apply_rope(x, cos, sin).double(), mod.complex_rope(x, mod.ref_angles(32, 8, 1e4)),
                               atol=1e-5, rtol=1e-5)


def test_check_rope_catches_absolute_angles(monkeypatch, capsys):
    """Angles growing with pos^2 (not linear in pos) break the relative property: the script must fail."""
    import tinyllm.model as model

    mod = _load()
    real = model.rope_cache

    def bad_cache(T, head_dim, base=10000.0, device=None):
        cos, sin = real(T, head_dim, base, device)
        ang = torch.atan2(sin, cos) * torch.arange(T, device=device, dtype=torch.float32)[:, None] * 0.01
        return ang.cos(), ang.sin()

    monkeypatch.setattr(model, "rope_cache", bad_cache)
    monkeypatch.setattr(mod, "rope_cache", bad_cache)
    monkeypatch.setattr(sys, "argv", ["check_rope.py", *ARGS])
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "[FAIL] <R_m q, R_n k> depends only on m - n" in out
    assert "[FAIL] rope model: positions shifted by 1 -> same logits/loss" in out
    assert "[PASS] rope causality" in out
