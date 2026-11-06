import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_attention.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_attention", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_attention_random_ok():
    r = subprocess.run([sys.executable, str(SCRIPT), "--random", "--cpu", "--batch-size", "2"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT OK" in r.stdout
    assert "[PASS] sdpa: future tokens don't change past logits" in r.stdout


def test_check_attention_catches_noncausal_sdpa(monkeypatch, capsys):
    """If the sdpa path loses its causal mask, the script must fail."""
    import tinyllm.model as model

    real = model.F.scaled_dot_product_attention

    def leaky(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        return real(q, k, v, dropout_p=dropout_p)  # drops both the mask and is_causal

    class FakeF:
        def __getattr__(self, name):
            return leaky if name == "scaled_dot_product_attention" else getattr(torch.nn.functional, name)

    monkeypatch.setattr(model, "F", FakeF())
    monkeypatch.setattr(sys, "argv", ["check_attention.py", "--random", "--cpu", "--batch-size", "2"])
    assert _load().main() == 1
    out = capsys.readouterr().out
    assert "[FAIL] sdpa: future tokens don't change past logits" in out
    assert "[FAIL] model logits manual == sdpa (fp32)" in out
    assert "[PASS] manual: future tokens don't change past logits" in out
