import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tinyllm.checkpoint import save_checkpoint  # noqa: E402
from tinyllm.model import GPT, GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_ppl", ROOT / "scripts/eval_ppl.py")
eval_ppl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_ppl)


def _ckpt(d: Path, seed: int, block: int = 16) -> Path:
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=50, block_size=block, n_layers=2, n_heads=2, d_model=32)
    save_checkpoint({"step": 7, "model": GPT(cfg).state_dict(), "model_config": cfg.to_dict()}, d, 7)
    return d


def test_eval_ppl_table_and_determinism(tmp_path, capsys):
    data = tmp_path / "val.bin"
    np.random.default_rng(0).integers(0, 50, 5000, dtype=np.uint16).tofile(data)
    a, b = _ckpt(tmp_path / "a", 0), _ckpt(tmp_path / "b", 1)
    out = tmp_path / "t.md"
    args = [str(a), str(b), "--labels", "A", "B", "--data", str(data), "--iters", "3",
            "--batch-size", "4", "--device", "cpu", "--out", str(out)]
    assert eval_ppl.main(args) == 0
    text = out.read_text()
    assert "| A |" in text and "| B |" in text and "3 × 4 windows of 16 tokens" in text
    # same checkpoint scored twice -> identical loss; random init loss ~ ln(50)
    r1 = eval_ppl.score(str(a), data, 3, 4, 1234, torch.device("cpu"))
    r2 = eval_ppl.score(str(a), data, 3, 4, 1234, torch.device("cpu"))
    assert r1["loss"] == r2["loss"] and abs(r1["loss"] - math.log(50)) < 0.5


def test_eval_ppl_rejects_block_mismatch(tmp_path):
    data = tmp_path / "val.bin"
    np.random.default_rng(0).integers(0, 50, 5000, dtype=np.uint16).tofile(data)
    a, b = _ckpt(tmp_path / "a", 0, 16), _ckpt(tmp_path / "b", 0, 8)
    with pytest.raises(SystemExit):
        eval_ppl.main([str(a), str(b), "--data", str(data), "--iters", "1", "--batch-size", "2", "--device", "cpu"])
