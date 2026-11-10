import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_report", ROOT / "scripts/check_report.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_real_report_ok():
    assert cr.check(ROOT / "docs/REPORT.md", ROOT / "docs/EVAL.md") == []


def test_eval_losses_parsed():
    assert "1.2192" in cr.eval_losses(ROOT / "docs/EVAL.md")


def test_mutations_fail():
    text = (ROOT / "docs/REPORT.md").read_text()
    tmp = ROOT / "docs" / "_tmp_report_check.md"  # in docs/ so relative links resolve
    try:
        for bad in [text.replace("1.2192", "1.2199"), text.replace("## 8. Limitations", "## 8. Notes"),
                    text + "\n![x](figures/nope.png)\n"]:
            tmp.write_text(bad)
            assert cr.check(tmp, ROOT / "docs/EVAL.md") != []
    finally:
        tmp.unlink(missing_ok=True)
