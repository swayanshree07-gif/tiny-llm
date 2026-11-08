"""Re-runnable check of the warmup + cosine LR schedule and the JSON config + CLI (PLAN Day 13).

  1. schedule values (`tinyllm.optim.lr_at`) for the config's own lr/min_lr/warmup/max_steps vs an
     independent closed form at every step: linear warmup to lr, lr exactly at the end of warmup,
     the midpoint (lr+min_lr)/2, min_lr at max_steps and after, monotone decay, no jump at the boundary
  2. a short real run through the CLI (`scripts/train.py --config ... --set ...`, subprocess) with a
     shortened schedule (--steps, --warmup), log_interval 1: config.json written with the overrides
     applied, every logged LR == lr_at(step-1) of the *expected* schedule, the last update is within 1% of min_lr (it uses step max_steps-1),
     loss finite and lower at the end than at the start, no checkpoints, temp dir removed
  3. PNGs: the full config schedule (--fig, default docs/figures/lr_schedule.png) and the short
     run's loss + LR curve from its metrics.jsonl (--run-fig, default docs/figures/lr_run.png)

Run: .venv/bin/python scripts/check_schedule.py [--config configs/base.json] [--steps 120] [--warmup 20]
     [--random] [--cpu]   (`--random`: small model + random token file, no data needed; used by pytest)
`--run-warmup N` runs the CLI with a different warmup than the one checked (mutation test: must FAIL).
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tinyllm.optim import lr_at  # noqa: E402


def reference(step: int, lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Independent closed form (written differently from lr_at on purpose)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + (lr - min_lr) * (math.cos(math.pi * t) + 1) / 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--run-warmup", type=int, default=None, help="mutation: warmup passed to the run")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fig", type=Path, default=ROOT / "docs/figures/lr_schedule.png")
    ap.add_argument("--run-fig", type=Path, default=ROOT / "docs/figures/lr_run.png")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    cfg = json.loads(args.config.read_text())
    t = cfg["train"]
    lr, mn, wu, total = t["lr"], t["min_lr"], t["warmup"], t["max_steps"]
    print(f"config schedule: lr {lr} -> {mn}, warmup {wu}, max_steps {total}")

    # 1. schedule values over the whole config schedule
    vals = np.array([lr_at(s, lr, mn, wu, total) for s in range(total + 50)])
    ref = np.array([reference(s, lr, mn, wu, total) for s in range(total + 50)])
    err = float(np.max(np.abs(vals - ref)) / lr)
    check("lr_at == closed form at every step", err < 1e-12, f"(max rel err {err:.1e})")
    check("warmup starts at lr/warmup", math.isclose(vals[0], lr / wu, rel_tol=1e-12), f"({vals[0]:.3e})")
    check("lr reached at end of warmup", math.isclose(vals[wu - 1], lr, rel_tol=1e-12)
          and math.isclose(vals[wu], lr, rel_tol=1e-12))
    check("warmup is linear", np.allclose(np.diff(vals[:wu]), lr / wu, rtol=1e-9))
    mid = wu + (total - wu) // 2
    check("midpoint == (lr+min_lr)/2", math.isclose(vals[mid], (lr + mn) / 2, rel_tol=1e-3), f"({vals[mid]:.3e})")
    check("min_lr at max_steps and after", np.all(vals[total:] == mn))
    check("monotone non-increasing after warmup", bool(np.all(np.diff(vals[wu:]) <= 1e-15)))
    jump = float(np.max(np.abs(np.diff(vals))))
    check("no step jumps more than lr/warmup", jump <= lr / wu * (1 + 1e-9), f"({jump:.2e})")
    check("lr never below min_lr after warmup", float(vals[wu:].min()) >= mn - 1e-15)

    # 2. short real run through the CLI with a shortened schedule
    tmp = Path(tempfile.mkdtemp(prefix="sched_", dir=ROOT / "ckpt" if (ROOT / "ckpt").is_dir() else None))
    hist: list[dict] = []
    try:
        sets = [f"train.out_dir={tmp / 'run'}", f"train.max_steps={args.steps}",
                f"train.warmup={args.run_warmup if args.run_warmup is not None else args.warmup}",
                f"train.batch_size={args.batch_size}", "train.grad_accum=1", "train.log_interval=1",
                "train.eval_interval=0", "train.ckpt_interval=0", "train.resume=false",
                "train.compile=false", "train.max_minutes=5",
                f"train.device={'cpu' if args.cpu else 'cuda'}"]
        if args.random:
            rng = np.random.default_rng(0)
            for name in ("train.bin", "valid.bin"):
                rng.integers(0, 256, 50_000, dtype=np.uint16).tofile(tmp / name)
            sets += [f"train.data_dir={tmp}", "model.vocab_size=256", "model.block_size=32",
                     "model.n_layers=2", "model.n_heads=2", "model.d_model=64"]
        else:
            sets.append(f"train.data_dir={args.tok_dir}")
        cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(args.config), "--set", *sets]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
        check("CLI run exit 0", r.returncode == 0, "" if r.returncode == 0 else r.stderr[-2000:])
        written = json.loads((tmp / "run/config.json").read_text()) if (tmp / "run/config.json").exists() else {}
        wt = written.get("train", {})
        check("config.json has the overrides", wt.get("max_steps") == args.steps and wt.get("log_interval") == 1
              and wt.get("lr") == lr, f"(max_steps {wt.get('max_steps')}, lr {wt.get('lr')})")
        mpath = tmp / "run/metrics.jsonl"
        if mpath.exists():
            hist = [json.loads(ln) for ln in mpath.read_text().splitlines() if ln.strip()]
            hist = [h for h in hist if h["type"] == "train"]
        steps = [h["step"] for h in hist]
        check("one log record per step", steps == list(range(1, args.steps + 1)), f"({len(steps)} records)")
        exp = [lr_at(s - 1, lr, mn, args.warmup, args.steps) for s in steps]
        got = [h["lr"] for h in hist]
        lerr = max((abs(a - b) / lr for a, b in zip(got, exp)), default=1.0)
        check("logged LR == schedule at every step", bool(hist) and lerr < 1e-12, f"(max rel err {lerr:.1e})")
        if hist:
            print(f"  lr: step 1 {got[0]:.3e}, step {args.warmup} {got[args.warmup - 1]:.3e}, "
                  f"step {args.steps} {got[-1]:.3e}")
            check("last update ~ min_lr (within 1% of lr-min_lr)", 0 <= got[-1] - mn < 0.01 * (lr - mn), f"({got[-1]:.4e})")
            losses = [h["loss"] for h in hist]
            check("losses finite", all(math.isfinite(x) for x in losses))
            first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
            check("loss went down", last < first, f"({first:.3f} -> {last:.3f})")
            if "peak_mem_gb" in hist[-1]:
                check("peak VRAM < 6.5 GB", hist[-1]["peak_mem_gb"] < 6.5, f"({hist[-1]['peak_mem_gb']:.2f} GB)")
        check("no checkpoints written", not list((tmp / "run").glob("*.pt")))

        # 3. figures
        if args.fig and hist:
            import plot_metrics as pm
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 3), facecolor=pm.SURFACE)
            pm.style(ax)
            ax.plot(np.arange(1, total + 1), vals[:total], color=pm.TRAIN, linewidth=1.8)
            ax.axvline(wu, color=pm.INK2, linewidth=0.8, linestyle=":")
            ax.set_xlabel("step", color=pm.INK2)
            ax.set_ylabel("learning rate", color=pm.INK2)
            ax.set_title(f"{args.config.name}: warmup {wu} + cosine {lr:g} → {mn:g} over {total} steps",
                         color=pm.INK, fontsize=10, loc="left")
            args.fig.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(args.fig, dpi=120, facecolor=pm.SURFACE)
            plt.close(fig)
            pm.plot(mpath, args.run_fig, title=f"check_schedule short run ({args.steps} steps, warmup {args.warmup})")
            check("figures written", args.fig.exists() and args.run_fig.exists(), f"({args.fig.name}, {args.run_fig.name})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("temp dir removed", not tmp.exists())
    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
