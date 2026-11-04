"""Compare several runs: validation loss curves on one PNG + a markdown table of final values.

  .venv/bin/python scripts/compare_runs.py ckpt/abl_pos_rope ckpt/abl_pos_learned \
      --labels RoPE learned --out docs/figures/abl_pos.png --table docs/abl_pos.md

Each argument is a run directory containing metrics.jsonl (or the metrics file itself).
If a run was resumed, records for a repeated step are de-duplicated (the later one wins).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1f9e6e", "#8a5cd1", "#c9a227"]


def load_run(path: Path) -> dict:
    """Return {'val': [(step, loss)], 'train': [(step, loss)], 'tok_per_s': float|None}."""
    f = path / "metrics.jsonl" if path.is_dir() else path
    val, train, tps = {}, {}, []
    for ln in f.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("type") == "val":
            val[r["step"]] = r["loss"]
        elif r.get("type") == "train":
            train[r["step"]] = r["loss"]
            if r.get("tok_per_s") and r["step"] > 100:  # skip compile warm-up
                tps.append(r["tok_per_s"])
    return {"val": sorted(val.items()), "train": sorted(train.items()),
            "tok_per_s": sum(tps) / len(tps) if tps else None}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def tokens_per_step(run: Path) -> int:
    """batch_size * grad_accum * block_size from the run's config.json."""
    c = json.loads(((run if run.is_dir() else run.parent) / "config.json").read_text())
    return c["train"]["batch_size"] * c["train"].get("grad_accum", 1) * c["model"]["block_size"]


def compare(runs: list[Path], labels: list[str], out: Path, title: str, skip_first: int = 1,
            x: str = "step"):
    data = [load_run(r) for r in runs]
    scale = [tokens_per_step(r) / 1e6 if x == "tokens" else 1 for r in runs]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    lows = []
    for i, (d, lab, k) in enumerate(zip(data, labels, scale)):
        pts = d["val"][skip_first:] or d["val"]  # step-0 loss (~9) would squash the y-axis
        if not pts:
            continue
        xs, ys = zip(*pts)
        xs = [v * k for v in xs]
        lows.append(min(ys))
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, mec=SURFACE, mew=1.2, label=lab)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=c, fontsize=9)
    ax.set_xlabel("training tokens (millions)" if x == "tokens" else "optimizer step", color=INK2)
    ax.set_ylabel("validation loss (nats)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)

    rows = ["| run | steps | tokens (M) | final val loss | val ppl | best val loss | tok/s |",
            "|-----|------:|-----------:|---------------:|--------:|--------------:|------:|"]
    for d, lab, r in zip(data, labels, runs):
        if not d["val"]:
            rows.append(f"| {lab} | – | – | – | – | – | – |")
            continue
        step, vl = d["val"][-1]
        best = min(v for _, v in d["val"])
        tps = f"{d['tok_per_s'] / 1000:.1f}k" if d["tok_per_s"] else "–"
        try:
            toks = f"{step * tokens_per_step(r) / 1e6:.1f}"
        except (OSError, KeyError, ValueError):
            toks = "–"
        rows.append(f"| {lab} | {step} | {toks} | {vl:.4f} | {math.exp(vl):.2f} | {best:.4f} | {tps} |")
    return out, "\n".join(rows) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="docs/figures/compare.png")
    ap.add_argument("--table", default=None, help="also write the markdown table to this file")
    ap.add_argument("--title", default="Validation loss by run")
    ap.add_argument("--x", choices=["step", "tokens"], default="step",
                    help="x-axis; 'tokens' needs config.json in each run dir")
    a = ap.parse_args(argv)
    runs = [Path(r) for r in a.runs]
    labels = a.labels or [r.name for r in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must match the number of runs")
    out, table = compare(runs, labels, Path(a.out), a.title, x=a.x)
    print(out)
    print(table, end="")
    if a.table:
        Path(a.table).write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
