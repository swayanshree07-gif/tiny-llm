import importlib.util
from pathlib import Path

import numpy as np

from tinyllm.tokenizer import EOT, Tokenizer

spec = importlib.util.spec_from_file_location(
    "tokenize_data", Path(__file__).resolve().parents[1] / "scripts" / "tokenize_data.py"
)
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)


def test_blocks_and_bin_roundtrip(tmp_path):
    stories = [f"Story {i}: a cat 🐱 ran {'far ' * (i % 7)}away." for i in range(300)]
    raw = "".join(s + "\n" + EOT + ("\n\n" if i % 5 == 0 else "\n") for i, s in enumerate(stories))
    src = tmp_path / "x.txt"
    src.write_text(raw, encoding="utf-8")
    # tiny blocks force many boundaries, including ones right after the separator
    blocks = list(td.iter_blocks(src, block=97))
    assert "".join(blocks) == raw
    tok = Tokenizer.train(raw, 300)
    dst = tmp_path / "x.bin"
    td.encode_file(tok, src, dst)
    ids = np.fromfile(dst, dtype=np.uint16)
    assert int((ids == tok.eot_id).sum()) == 300
    assert tok.decode(ids.tolist()) == "".join(s + EOT for s in stories)


def test_normalize():
    assert td.normalize(f"a\n{EOT}\n\n b \n{EOT}\n\n") == f"a{EOT}b{EOT}"
