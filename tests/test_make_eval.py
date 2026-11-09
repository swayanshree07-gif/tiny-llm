import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_eval", ROOT / "scripts/make_eval.py")
me = importlib.util.module_from_spec(spec)
spec.loader.exec_module(me)


def test_follow_rates_parse_from_docs():
    got = {lab: me.read_follow(ROOT / doc, rx) for lab, doc, rx in me.FOLLOW}
    assert got["instruct v4 (word tokens ×5)"] == 25.0
    assert got["chat v3 (large, always 3 words)"] == 20.5


def test_link_checker(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    md = tmp_path / "r.md"
    md.write_text("![a](a.png) [b](missing.md) [w](https://x.org) [h](#top)")
    assert me.check_links(md) == ["missing.md"]


def test_render_with_fake_rows():
    rows = [{"label": "base-29M", "ckpt": "ckpt/base", "step": 1, "params": 2e6, "loss": 1.0,
             "ppl": 2.718, "mb": 1.0, "note": "n"}]
    text = me.render(rows, 2, 3, 4)
    assert "| base-29M |" in text and "+0.0000" in text


def test_existing_report_links_ok():
    if (ROOT / "docs/EVAL.md").exists():
        assert me.check_links(ROOT / "docs/EVAL.md") == []
