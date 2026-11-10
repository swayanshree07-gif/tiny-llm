"""Check docs/REPORT.md (PLAN Day 29): required sections, every relative link/image exists, at least the main loss
PNGs are embedded, and every val loss in the EVAL.md perplexity table also appears in the report (so the two agree).

  .venv/bin/python scripts/check_report.py [--report docs/REPORT.md] [--eval docs/EVAL.md]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_eval", ROOT / "scripts/make_eval.py")
_me = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_me)

SECTIONS = ["Data and tokenizer", "Model", "Training", "Samples", "Fine-tunes", "Inference", "What changed results",
            "Limitations"]
IMAGES = ["figures/loss.png", "figures/large_loss.png", "figures/large_vs_base.png"]


def eval_losses(eval_md: Path) -> list[str]:
    """Val losses (4 decimals) and ppl of the EVAL.md perplexity table."""
    out = []
    for line in eval_md.read_text().splitlines():
        if line.startswith("| ") and "`ckpt/" in line:
            cells = [c.strip() for c in line.split("|")]
            out += [cells[6], cells[7]]  # val loss, ppl
    return out


def check(report: Path, eval_md: Path) -> list[str]:
    text = report.read_text()
    errs = [f"missing link target: {m}" for m in _me.check_links(report)]
    heads = re.findall(r"^## \d+\. (.+)$", text, re.M)
    errs += [f"missing section: {s}" for s in SECTIONS if not any(h.startswith(s) for h in heads)]
    errs += [f"image not embedded: {i}" for i in IMAGES if f"]({i})" not in text]
    losses = eval_losses(eval_md)
    if not losses:
        errs.append("no losses parsed from EVAL.md")
    errs += [f"EVAL.md loss {v} not in report" for v in losses if v not in text]
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(ROOT / "docs/REPORT.md"))
    ap.add_argument("--eval", default=str(ROOT / "docs/EVAL.md"))
    a = ap.parse_args(argv)
    errs = check(Path(a.report), Path(a.eval))
    for e in errs:
        print("FAIL", e)
    print("RESULT", "OK" if not errs else f"FAIL ({len(errs)})")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
