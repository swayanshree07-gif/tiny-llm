"""Sample stories from a checkpoint.

  .venv/bin/python scripts/generate.py --ckpt ckpt/base --prompt "Once upon a time" -n 3
  .venv/bin/python scripts/generate.py --ckpt ckpt/base --temperature 0   # greedy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import generate, load_model  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint file or directory (latest is used)")
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tok/tokenizer.json"))
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("-n", "--num-samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    tok = Tokenizer.load(args.tokenizer)
    model, st = load_model(args.ckpt, device)
    ids = tok.encode(args.prompt) or [tok.eot_id]
    idx = torch.tensor([ids] * args.num_samples, dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(args.seed)
    out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
                   eot_id=tok.eot_id, generator=g)
    print(f"# checkpoint step {st['step']}, T={args.temperature} top_k={args.top_k} top_p={args.top_p}")
    for i, row in enumerate(out.tolist()):
        row = row[: row.index(tok.eot_id, len(ids))] if tok.eot_id in row[len(ids):] else row
        print(f"--- sample {i + 1} ---\n{tok.decode(row)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
