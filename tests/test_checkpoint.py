import pytest

torch = pytest.importorskip("torch")
from tinyllm.checkpoint import (latest_checkpoint, list_checkpoints, load_checkpoint,  # noqa: E402
                                save_checkpoint)


def test_keep_latest_two(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for s in (10, 20, 30, 40):
        save_checkpoint({"step": s, "w": torch.ones(3) * s}, tmp_path, s, keep=2)
    cks = list_checkpoints(tmp_path)
    assert [p.name for p in cks] == ["ckpt_0000030.pt", "ckpt_0000040.pt"]
    assert latest_checkpoint(tmp_path).name == "ckpt_0000040.pt"
    st = load_checkpoint(latest_checkpoint(tmp_path))
    assert st["step"] == 40 and torch.equal(st["w"], torch.full((3,), 40.0))
    assert not list(tmp_path.glob("*.part"))  # no temp files left behind


def test_sorted_numerically_and_ignores_others(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".ckpt_0000099.pt.part").write_bytes(b"partial")  # crashed write
    save_checkpoint({"step": 9}, tmp_path, 9, keep=5)
    save_checkpoint({"step": 100}, tmp_path, 100, keep=5)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["ckpt_0000009.pt", "ckpt_0000100.pt"]
