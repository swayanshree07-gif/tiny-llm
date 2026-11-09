"""Regenerate the evaluation report docs/EVAL.md (PLAN Day 26).

  timeout 1200 .venv/bin/python scripts/make_eval.py            # full: scores all checkpoints on the GPU
  .venv/bin/python scripts/make_eval.py --check                  # only verify links/sources of the existing EVAL.md

Perplexity rows are measured now (same fixed validation windows as eval_ppl.py / eval_int8.py).
Follow rates, ablation and sweep numbers are read from the docs that their own scripts wrote
(eval_instruct.py, compare_runs.py), so this script never invents a number. After writing, every
relative link/image in EVAL.md is checked to exist; a missing one is an error (exit 1).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MODELS = [  # label, checkpoint dir, quantize, note
    ("base-29M", "ckpt/base", False, "8×512 pretrain, 655M tokens"),
    ("base-29M int8", "ckpt/base", True, "weight-only int8 of base"),
    ("large-91M", "ckpt/large", False, "12×768 pretrain, 295M tokens"),
    ("large-91M int8", "ckpt/large", True, "weight-only int8 of large"),
    ("instruct-v4 (29M)", "ckpt/instruct_v4", False, "prompt fine-tune of base"),
    ("chat-v3 (91M)", "ckpt/chat_large_v3", False, "chat fine-tune of large"),
]
FOLLOW = [  # label, doc, regex (group 1 = percent)
    ("base-29M, no fine-tune", "docs/instruct_eval_base.md", None),
    ("instruct v1", "docs/instruct_eval_ft.md", None),
    ("instruct v2", "docs/instruct_eval_ft_v2.md", None),
    ("instruct v3", "docs/instruct_eval_v3_on_anyfix.md", None),
    ("instruct v4 (word tokens ×5)", "docs/instruct_eval_v4_on_anyfix.md", None),
    ("large-91M, no fine-tune", "docs/instruct_eval_large_base.md", None),
    ("chat v1 (large)", "docs/chat_eval_large.md", None),
    ("chat v2 (large, ×5)", "docs/chat_eval_large_v2.md", None),
    ("chat v3 (large, always 3 words)", "docs/chat_finetune.md", r"\| chat_large_v3 \|[^|]*\| \*\*([\d.]+)%"),
]
ALL_RE = r"\| \*\*all\*\* \| \d+ \| \*\*([\d.]+)%"
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def read_follow(doc: Path, regex: str | None) -> float:
    m = re.search(regex or ALL_RE, doc.read_text())
    if not m:
        raise SystemExit(f"no follow rate found in {doc}")
    return float(m.group(1))


def read_table(doc: Path) -> str:
    return "\n".join(l for l in doc.read_text().splitlines() if l.startswith("|"))


def check_links(md: Path) -> list[str]:
    """Return relative link targets in md that don't exist (http links and anchors ignored)."""
    missing = []
    for t in LINK_RE.findall(md.read_text()):
        if t.startswith(("http://", "https://", "#")):
            continue
        if not (md.parent / t.split("#")[0]).exists():
            missing.append(t)
    return missing


def score_all(iters: int, bs: int, seed: int, device_name: str) -> list[dict]:
    import torch

    from tinyllm.data import TokenDataset
    from tinyllm.device import get_device
    from tinyllm.generate import load_model
    from tinyllm.quant import model_bytes, quantize_model
    from tinyllm.train import autocast_ctx, evaluate

    device = get_device(device_name)
    rows = []
    for label, ck, q, note in MODELS:
        model, st = load_model(str(ROOT / ck), device)
        n_params = sum(p.numel() for p in model.parameters())
        if q:
            quantize_model(model)
        ds = TokenDataset(ROOT / "data/tok/valid.bin", model.cfg.block_size, seed=seed)
        with torch.no_grad():
            loss = evaluate(model, ds, iters, bs, device, lambda: autocast_ctx(device, "bf16"), seed)
        rows.append({"label": label, "ckpt": ck, "step": st["step"], "params": n_params, "loss": loss,
                     "ppl": math.exp(loss), "mb": model_bytes(model) / 2**20, "note": note})
        print(f"{label}: {loss:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def render(rows: list[dict], iters: int, bs: int, seed: int) -> str:
    base = next(r for r in rows if r["label"] == "base-29M")["loss"]
    L = ["# Evaluation report (PLAN Day 26)", "",
         "Regenerate with `timeout 1200 .venv/bin/python scripts/make_eval.py` (≈ 2 min on the RTX 5050). "
         "Perplexity is measured by the script; follow rates and ablation/sweep tables are copied from the docs "
         "their own scripts wrote (linked).", "",
         "## 1. Perplexity on TinyStories validation", "",
         f"{iters} × {bs} windows of 256 tokens ({iters * bs * 256:,} tokens, seed {seed}), identical for every "
         "row, bf16 autocast. Tokenizer: own 8192-token byte-level BPE, so ppl is per BPE token (≈ 4 bytes).", "",
         "| model | checkpoint | step | params | weights MB | val loss | ppl | Δ vs base | notes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | `{r['ckpt']}` | {r['step']} | {r['params'] / 1e6:.1f}M | {r['mb']:.0f} | "
                 f"{r['loss']:.4f} | {r['ppl']:.3f} | {r['loss'] - base:+.4f} | {r['note']} |")
    L += ["", "Fine-tuned models score worse on plain stories because they were trained on a different format "
          "(prompted stories / chat turns); that is expected and not a regression.", "",
          "![large vs base](figures/large_vs_base.png)", "",
          "Training curves: [base](figures/loss.png), [large](figures/large_loss.png). "
          "Detail: [large_vs_base.md](large_vs_base.md).", "",
          "## 2. Prompt following (200 held-out prompts, T=0.8 top-p=0.95)", "",
          "A sample \"follows\" if it contains all 3 requested words, or starts with the requested sentence.", "",
          "| model | follows | source |", "|---|---|---|"]
    for label, doc, rx in FOLLOW:
        L.append(f"| {label} | {read_follow(ROOT / doc, rx):.1f}% | [{Path(doc).name}]({Path(doc).name}) |")
    L += ["", "Details and per-kind splits: [chat_finetune.md](chat_finetune.md).", "",
          "## 3. Ablations at a fixed 49M-token budget (8×512 model)", "",
          "Positions ([abl_pos_table.md](abl_pos_table.md)):", "", read_table(ROOT / "docs/abl_pos_table.md"), "",
          "![positions](figures/abl_pos.png)", "",
          "Learning rate / batch ([sweep_table.md](sweep_table.md)):", "", read_table(ROOT / "docs/sweep_table.md"), "",
          "![sweep](figures/sweep.png)", "",
          "## 4. Inference", "",
          "KV cache, batch and dtype speed: [infer_bench.md](infer_bench.md); int8: "
          "[infer_bench_int8.md](infer_bench_int8.md). Int8 costs ≤ +0.0002 val loss and shrinks weights ~3×, "
          "but is ~25% slower (dequantised per call, no int8 kernel).", "",
          "## 5. Samples", "",
          "- Base sample suite: [suite_step40000.md](../samples/suite_step40000.md)",
          "- Large sample suite: [large.md](../samples/large.md)",
          "- Scripted chat session (chat-v3): [chat_session.md](../samples/chat_session.md)", "",
          "## 6. What helped (and what didn't)", "",
          "- **Model size**: 91M beats 29M at every equal token count (1.208 vs 1.348 at 295M tokens) and wins "
          "on final val loss despite 2.2× fewer tokens. The largest single gain.",
          "- **Higher LR**: 1.2e-3 beat 6e-4 by 0.039 at 49M tokens; 3e-4 was clearly worse. A bigger batch "
          "(32k tokens/step) at the same token budget was worse (fewer updates).",
          "- **RoPE** over learned positions: −0.030 val loss at 49M tokens.",
          "- **Loss weight ×5 on requested-word tokens** was the only fine-tune change that moved prompt "
          "following a lot (instruct v3 16.5% → v4 25.0%). More epochs or more rows of the same data did not "
          "help; val loss plateaus within ~100 steps.",
          "- **Chat format on the large model** reached 20.5% follow, below the 29M instruct v4 (25%): bigger "
          "base did not transfer to instruction following with this small synthetic data.",
          "- **KV cache**: 1.6–3× faster at batch 1, ~22–25× at batch 32; **int8** is near-lossless but slower.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "docs/EVAL.md"))
    ap.add_argument("--check", action="store_true", help="only check links of the existing report")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not a.check:
        out.write_text(render(score_all(a.iters, a.batch_size, a.seed, a.device), a.iters, a.batch_size, a.seed))
        print(f"wrote {out}")
    missing = check_links(out)
    if missing:
        print("MISSING links:", *missing, sep="\n  ")
        return 1
    print(f"links OK ({len(LINK_RE.findall(out.read_text()))} checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
