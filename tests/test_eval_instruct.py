import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKJ = ROOT / "data/tok/tokenizer.json"


@pytest.mark.skipif(not TOKJ.exists(), reason="tokenizer not present")
def test_eval_instruct_runs_on_tiny_checkpoint(tmp_path):
    import torch
    from tinyllm.checkpoint import save_checkpoint
    from tinyllm.instruct import words_prompt
    from tinyllm.model import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=8192, block_size=64, n_layers=1, n_heads=2, d_model=32)
    torch.manual_seed(0)
    save_checkpoint({"step": 7, "model": GPT(cfg).state_dict(), "model_config": cfg.to_dict()},
                    tmp_path / "ck", 7)
    prompts = tmp_path / "p.jsonl"
    with open(prompts, "w") as f:
        for ws in (["ball", "park", "tree"], ["dog", "red", "sun"], ["cat", "hat", "mat"]):
            f.write(json.dumps({"prompt": words_prompt(ws), "kind": "words", "words": ws}) + "\n")
    out = tmp_path / "r.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_instruct.py"), "--ckpt", str(tmp_path / "ck"),
                        "--prompts", str(prompts), "--max-new", "8", "--out", str(out)],
                       capture_output=True, text=True, timeout=120, env={"CUDA_VISIBLE_DEVICES": "", "PATH": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT follows 0.000" in r.stdout  # a random model doesn't follow prompts
    assert "| words | 3 |" in out.read_text() and "(step 7)" in out.read_text()
