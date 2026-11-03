"""Fixed prompts x sampling settings -> markdown file with samples and quality stats (PLAN Day 26).

  .venv/bin/python scripts/sample_suite.py --ckpt ckpt/base --out samples/final.md
Seeds are fixed, so rerunning on the same checkpoint gives the same file.
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
from tinyllm.quality import story_stats  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402

PROMPTS = [
    "Once upon a time",
    "Lily and her dog went to the park.",
    "Tom was sad because",
    "One day, a little bird",
    "The big red ball",
    "Mom said, \"",
]
SETTINGS = [  # name, temperature, top_k, top_p
    ("greedy", 0.0, None, None),
    ("T=0.7 top-k=40", 0.7, 40, None),
    ("T=0.8 top-p=0.95", 0.8, None, 0.95),
    ("T=1.0 top-p=0.9", 1.0, None, 0.9),
]


def run(model, tok, device, prompts, settings, max_new: int, seed: int):
    """Yield (prompt, setting_name, text, stats) for each prompt x setting (1 sample each)."""
    for pi, prompt in enumerate(prompts):
        ids = tok.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        for si, (name, t, k, p) in enumerate(settings):
            g = torch.Generator(device=device).manual_seed(seed + 100 * pi + si)
            out = generate(model, idx, max_new, t, k, p, eot_id=tok.eot_id, generator=g,
                           use_cache=True)[0].tolist()
            new = out[len(ids):]
            ended = tok.eot_id in new
            if ended:
                new = new[: new.index(tok.eot_id)]
            text = tok.decode(ids + new)
            yield prompt, name, text, story_stats(text, ended)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/tok/tokenizer.json"))
    ap.add_argument("--out", default=str(ROOT / "samples/suite.md"))
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    tok = Tokenizer.load(args.tokenizer)
    model, st = load_model(args.ckpt, device)
    res = list(run(model, tok, device, PROMPTS, SETTINGS, args.max_new_tokens, args.seed))

    lines = [f"# Sample suite — checkpoint step {st['step']}\n",
             f"{len(PROMPTS)} prompts x {len(SETTINGS)} settings, max {args.max_new_tokens} new tokens, "
             f"seed {args.seed}. `rep4` = fraction of repeated 4-grams (lower is better), "
             "`distinct2` = unique bigrams / bigrams, `ended` = model emitted `<|endoftext|>`.\n",
             "## Summary by setting\n", "| setting | mean words | ended | mean distinct2 | mean rep4 |",
             "|---|---|---|---|---|"]
    for name, *_ in SETTINGS:
        r = [s for _, n, _, s in res if n == name]
        lines.append(f"| {name} | {sum(s['words'] for s in r) / len(r):.0f} | "
                     f"{sum(s['ended'] for s in r)}/{len(r)} | "
                     f"{sum(s['distinct2'] for s in r) / len(r):.3f} | {sum(s['rep4'] for s in r) / len(r):.3f} |")
    lines.append("\n## Samples\n")
    for prompt, name, text, s in res:
        lines += [f"### `{prompt}` — {name}", f"*{s}*\n", text.strip().replace("\n", "\n\n"), ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[3:5 + len(SETTINGS)]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
