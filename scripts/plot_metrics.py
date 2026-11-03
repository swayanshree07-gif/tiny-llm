"""Plot train/val loss and the LR schedule from a run's metrics.jsonl to a PNG.

  .venv/bin/python scripts/plot_metrics.py ckpt/base/metrics.jsonl --out docs/figures/loss.png
Two stacked panels sharing the step axis (loss on top, LR below) - no dual y-axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAIN, VAL = "#2a78d6", "#eb6834"  # categorical slots 1, 2


def load(path: Path):
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    tr = [r for r in recs if r["type"] == "train"]
    va = [r for r in recs if r["type"] == "val"]
    return tr, va


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(metrics: Path, out: Path, title: str | None = None) -> Path:
    tr, va = load(metrics)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    for ax in (a1, a2):
        style(ax)
    a1.plot([r["step"] for r in tr], [r["loss"] for r in tr], color=TRAIN, lw=2, label="train")
    if va:
        a1.plot([r["step"] for r in va], [r["loss"] for r in va], color=VAL, lw=2, marker="o",
                ms=5, mec=SURFACE, mew=1.5, label="validation")
        last = va[-1]
        a1.annotate(f"val {last['loss']:.3f} (ppl {last['ppl']:.1f})", (last["step"], last["loss"]),
                    xytext=(-8, 12), textcoords="offset points", ha="right", color=INK, fontsize=9)
    a1.set_ylabel("cross-entropy loss (nats)", color=INK2)
    a1.legend(frameon=False, labelcolor=INK)
    a1.set_title(title or f"Training curves: {metrics.parent.name}", color=INK, loc="left", fontsize=12)
    lo = min([r["loss"] for r in tr + va] or [0])
    a1.set_ylim(bottom=max(0, lo - 0.2))
    a2.plot([r["step"] for r in tr], [r["lr"] for r in tr], color=TRAIN, lw=2)
    a2.set_ylabel("learning rate", color=INK2)
    a2.set_xlabel("optimizer step", color=INK2)
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out", default="docs/figures/loss.png")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    print(plot(Path(a.metrics), Path(a.out), a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
