"""Re-runnable check of checkpointing + exact resume (PLAN Day 14).

Drives the real `tinyllm.train.train()` at full model size on real training text:

  A. straight run: --steps (20) steps, checkpoint every --ckpt-every (5) steps
  B. split run:    same config with stop_at_step = steps/2 (session 1), then a second call that
                   resumes from the latest checkpoint and finishes (session 2)
  C. control:      run A repeated from scratch (CUDA only) to measure the GPU's own run-to-run noise

Checks:
  1. session 1 stops at steps/2, session 2 resumes there and ends at --steps
  2. per-step train losses of B == A (bit-exact if A == C bit-exact, else within the A-vs-C noise x 2 + 1e-6)
  3. val losses (logged at every checkpoint step) of B == A, same tolerance rule
  4. final weights and AdamW state (exp_avg, exp_avg_sq, step) of B == A (same rule)
  5. data-sampler state and CPU/CUDA RNG states in the final checkpoints are identical
  6. checkpoint contents: step, model, optimizer, model/train config, data state, RNG states
  7. keep latest 2: only the 2 newest ckpt_*.pt remain, no leftover .part files
  8. metrics.jsonl of B has train steps 1..N exactly once (records after the ckpt are dropped on resume)
  9. resume with a different model config is rejected (ValueError)
 10. peak VRAM < 6.5 GB (CUDA), temp dir removed

Dropout is set to --dropout (default 0.1) so that the torch RNG restore is actually exercised.

Run: .venv/bin/python scripts/check_resume.py [--config configs/base.json] [--tok-dir data/tok]
     [--steps 20] [--batch-size 8] [--random] [--cpu] [--break-data-state]
`--random` uses a small model and a random token file (pytest). `--break-data-state` is a mutation:
the sampler state is NOT restored on resume, so the check must FAIL.
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm import data as data_mod  # noqa: E402
from tinyllm.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint  # noqa: E402
from tinyllm.model import GPTConfig  # noqa: E402
from tinyllm.train import TrainConfig, config_from_dict, train  # noqa: E402


def _losses(hist: list[dict], kind: str) -> dict[int, float]:
    return {h["step"]: h["loss"] for h in hist if h["type"] == kind}


def _max_diff(a: dict[int, float], b: dict[int, float]) -> float:
    if a.keys() != b.keys():
        return float("inf")
    return max((abs(a[k] - b[k]) for k in a), default=0.0)


def _tensor_diff(sa, sb) -> float:
    """Max abs diff over two (nested) state dicts of tensors/numbers; inf on structural mismatch."""
    if isinstance(sa, torch.Tensor):
        if not isinstance(sb, torch.Tensor) or sa.shape != sb.shape:
            return float("inf")
        return float((sa.float() - sb.float()).abs().max()) if sa.numel() else 0.0
    if isinstance(sa, dict):
        if not isinstance(sb, dict) or sa.keys() != sb.keys():
            return float("inf")
        return max((_tensor_diff(sa[k], sb[k]) for k in sa), default=0.0)
    if isinstance(sa, (list, tuple)):
        if len(sa) != len(sb):
            return float("inf")
        return max((_tensor_diff(x, y) for x, y in zip(sa, sb)), default=0.0)
    if isinstance(sa, (int, float)):
        return abs(sa - sb)
    return 0.0 if sa == sb else float("inf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    ap.add_argument("--tok-dir", type=Path, default=ROOT / "data/tok")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--random", action="store_true", help="small model + random token file")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--break-data-state", action="store_true", help="mutation: skip sampler restore")
    args = ap.parse_args(argv)
    assert args.steps % 2 == 0 and (args.steps // 2) % args.ckpt_every == 0, "half must be a ckpt step"

    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    cuda = not args.cpu and torch.cuda.is_available()
    tmp = Path(tempfile.mkdtemp(prefix="resume_", dir=ROOT / "ckpt" if (ROOT / "ckpt").is_dir() else None))
    orig_load = data_mod.TokenDataset.load_state_dict
    try:
        if args.random:
            mcfg = GPTConfig(vocab_size=256, block_size=32, n_layers=2, n_heads=2, d_model=64)
            rng = np.random.default_rng(0)
            for name in ("train.bin", "valid.bin"):
                rng.integers(0, 256, 50_000, dtype=np.uint16).tofile(tmp / name)
            data_dir = tmp
        else:
            mcfg, _ = config_from_dict(json.loads(args.config.read_text()))
            data_dir = args.tok_dir
        mcfg = replace(mcfg, dropout=args.dropout)
        print("model:", mcfg.to_dict())
        half = args.steps // 2
        base = TrainConfig(
            data_dir=str(data_dir), out_dir="", batch_size=args.batch_size, grad_accum=2,
            max_steps=args.steps, lr=6e-4, min_lr=6e-5, warmup=5, weight_decay=0.1,
            eval_interval=args.ckpt_every, eval_iters=4, log_interval=1, ckpt_interval=args.ckpt_every,
            keep_ckpts=2, resume=True, compile=False, device="cuda" if cuda else "cpu",
            dtype="auto", seed=1337)

        a = train(mcfg, replace(base, out_dir=str(tmp / "A")), verbose=False)
        s1 = train(mcfg, replace(base, out_dir=str(tmp / "B"), stop_at_step=half), verbose=False)
        b_mid = [p.name for p in list_checkpoints(tmp / "B")]
        if args.break_data_state:
            data_mod.TokenDataset.load_state_dict = lambda self, st: None
        s2 = train(mcfg, replace(base, out_dir=str(tmp / "B")), verbose=False)
        data_mod.TokenDataset.load_state_dict = orig_load
        check("session 1 stops at half", s1["step"] == half, f"(step {s1['step']}, ckpts {b_mid})")
        check("session 2 resumes and finishes", s2["step"] == args.steps, f"(step {s2['step']})")

        if cuda:
            c = train(mcfg, replace(base, out_dir=str(tmp / "C")), verbose=False)
            noise = {"train": _max_diff(_losses(a["history"], "train"), _losses(c["history"], "train")),
                     "val": _max_diff(_losses(a["history"], "val"), _losses(c["history"], "val"))}
            ca = load_checkpoint(latest_checkpoint(tmp / "C"))
        else:
            noise, ca = {"train": 0.0, "val": 0.0}, None
        print(f"  run-to-run noise A vs C: train {noise['train']:.3g}, val {noise['val']:.3g}")

        def tol(n: float) -> float:
            return 0.0 if n == 0.0 else 2 * n + 1e-6

        la, lb = _losses(a["history"], "train"), _losses(s1["history"] + s2["history"], "train")
        d = _max_diff(la, lb)
        check("train losses resumed == straight", d <= tol(noise["train"]),
              f"(max diff {d:.3g}, tol {tol(noise['train']):.3g}; step {half + 1}: "
              f"{la.get(half + 1, float('nan')):.6f} vs {lb.get(half + 1, float('nan')):.6f})")
        va = _losses(a["history"], "val")
        vb = {**_losses(s1["history"], "val"), **_losses(s2["history"], "val")}
        d = _max_diff(va, vb)
        check("val losses resumed == straight", d <= tol(noise["val"]),
              f"(steps {sorted(va)}, max diff {d:.3g})")

        ka, kb = load_checkpoint(latest_checkpoint(tmp / "A")), load_checkpoint(latest_checkpoint(tmp / "B"))
        wn = _tensor_diff(ka["model"], ca["model"]) if ca else 0.0
        on = _tensor_diff(ka["optimizer"], ca["optimizer"]) if ca else 0.0
        dw, do = _tensor_diff(ka["model"], kb["model"]), _tensor_diff(ka["optimizer"], kb["optimizer"])
        check("final weights equal", dw <= tol(wn), f"(max diff {dw:.3g}, noise {wn:.3g})")
        check("final AdamW state equal", do <= tol(on), f"(max diff {do:.3g}, noise {on:.3g})")
        check("data sampler state equal", ka["train_data"] == kb["train_data"])
        rng_eq = torch.equal(ka["rng_cpu"], kb["rng_cpu"]) and (
            not cuda or torch.equal(ka["rng_cuda"], kb["rng_cuda"]))
        check("CPU/CUDA RNG states equal", rng_eq)

        need = {"step", "model", "optimizer", "model_config", "train_config", "train_data", "rng_cpu", "rng_cuda"}
        check("checkpoint has all state", need <= kb.keys() and kb["step"] == args.steps
              and kb["model_config"] == mcfg.to_dict(), f"(keys {sorted(kb.keys())})")
        names = [p.name for p in list_checkpoints(tmp / "B")]
        want = [f"ckpt_{s:07d}.pt" for s in (args.steps - args.ckpt_every, args.steps)]
        parts = list((tmp / "B").glob("*.part"))
        check("keep latest 2, no .part files", names == want and not parts, f"({names})")

        recs = [json.loads(x) for x in (tmp / "B" / "metrics.jsonl").read_text().splitlines() if x.strip()]
        steps = [r["step"] for r in recs if r["type"] == "train"]
        vsteps = [r["step"] for r in recs if r["type"] == "val"]
        check("metrics.jsonl continuous, no duplicates", steps == list(range(1, args.steps + 1))
              and vsteps == list(range(0, args.steps + 1, args.ckpt_every)), f"(val steps {vsteps})")

        other = replace(mcfg, n_layers=mcfg.n_layers + 1)
        try:
            train(other, replace(base, out_dir=str(tmp / "B"), max_steps=args.steps + 2), verbose=False)
            rejected = False
        except ValueError:
            rejected = True
        check("resume with different model config rejected", rejected)

        if cuda:
            peak = max(h.get("peak_mem_gb", 0.0) for h in a["history"] + s2["history"])
            check("peak VRAM < 6.5 GB", peak < 6.5, f"({peak:.2f} GB)")
    finally:
        data_mod.TokenDataset.load_state_dict = orig_load
        shutil.rmtree(tmp, ignore_errors=True)
    check("temp dir removed", not tmp.exists())
    print("RESULT", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
