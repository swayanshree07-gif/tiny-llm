"""Does a model use the prompt? Story loss with the row's own prompt vs another row's prompt.

For the first n held-out "words" rows of data/instruct/valid.npz, score the same story twice:
after its own prompt and after the prompt of the next "words" row (wrong words). If the model
reads the word list, the loss with its own prompt is lower. The script prints mean story loss for both
and the gap (swapped - own), in nats/token. It also prints the same split for "begins" rows.

Run: .venv/bin/python scripts/prompt_sensitivity.py --ckpt ckpt/instruct_v2 [-n 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import load_model  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


@torch.no_grad()
def story_loss(model, prompt_ids: list[int], story_ids: list[int], device) -> float:
    ids = (prompt_ids + story_ids)[: model.cfg.block_size + 1]
    x = torch.tensor(ids[:-1], device=device)[None]
    logits, _ = model(x)  # GPT.forward returns (logits, loss)
    logits = logits[0, len(prompt_ids) - 1:].float()
    tgt = torch.tensor(ids[len(prompt_ids):], device=device)
    return F.cross_entropy(logits[: len(tgt)], tgt).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/instruct")
    ap.add_argument("-n", type=int, default=200)
    args = ap.parse_args()

    device = get_device()
    tok = Tokenizer.load(ROOT / "data/tok/tokenizer.json")
    model, _ = load_model(args.ckpt, device)
    z = np.load(args.data / "valid.npz")
    specs = [json.loads(l) for l in (args.data / "valid_prompts.jsonl").read_text().splitlines()]
    y = z["y"].astype(np.int64)
    assert len(specs) == len(y)
    for kind in ("words", "begins"):
        rows = [i for i, s in enumerate(specs) if s["kind"] == kind][: args.n]
        own, swp = [], []
        for j, i in enumerate(rows):
            story = y[i][y[i] != -100].tolist()
            other = specs[rows[(j + 1) % len(rows)]]["prompt"]
            own.append(story_loss(model, tok.encode_ordinary(specs[i]["prompt"]), story, device))
            swp.append(story_loss(model, tok.encode_ordinary(other), story, device))
        own_m, swp_m = float(np.mean(own)), float(np.mean(swp))
        print(f"{kind}: n={len(rows)} own-prompt loss {own_m:.4f}, swapped-prompt loss {swp_m:.4f}, "
              f"gap {swp_m - own_m:+.4f} nats/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
