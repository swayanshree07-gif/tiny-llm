"""Prompt-following eval for the Day 27 fine-tune: how often does a model do what the prompt asks?

For n held-out prompts from data/instruct/valid_prompts.jsonl, generate a story (seeded sampling,
T=0.8 top-p=0.95 by default, or greedy) and check it with tinyllm.instruct.follows:
  words  -> all requested words appear;  begins -> the story starts with the given sentence.
Also reports how many stories end with <|endoftext|> and repeated-4-gram fraction.
Run it on the base model (before) and the fine-tuned model (after).

Run: .venv/bin/python scripts/eval_instruct.py --ckpt ckpt/base [-n 200] [--out docs/instruct_eval_base.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.chat import encode_chat  # noqa: E402
from tinyllm.device import get_device  # noqa: E402
from tinyllm.generate import generate, load_model  # noqa: E402
from tinyllm.instruct import PROMPT_PREFIX, STORY_PREFIX, follows, words_found  # noqa: E402
from tinyllm.quality import repeated_ngram_frac  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


def chat_message(prompt: str) -> str:
    """'Prompt: <request>\\nStory:' (instruct format) -> '<request>' (chat user message)."""
    return prompt.removeprefix(PROMPT_PREFIX).removesuffix(STORY_PREFIX)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", type=Path, default=ROOT / "data/instruct/valid_prompts.jsonl")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/tok/tokenizer.json")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=230)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chat", action="store_true",
                    help="wrap each request in the chat template (User:/Assistant:) for chat fine-tunes")
    args = ap.parse_args(argv)

    device = get_device()
    tok = Tokenizer.load(args.tokenizer)
    eot = tok.vocab_size - 1
    model, st = load_model(args.ckpt, device)
    specs = [json.loads(l) for l in args.prompts.read_text().splitlines()[:args.n]]

    # batch prompts of equal token length together (no padding needed)
    groups: dict[int, list[int]] = defaultdict(list)
    if args.chat:
        enc = [encode_chat(tok, [{"role": "user", "content": chat_message(s["prompt"])}],
                           add_generation_prompt=True)[0] for s in specs]
    else:
        enc = [tok.encode_ordinary(s["prompt"]) for s in specs]
    for i, e in enumerate(enc):
        groups[len(e)].append(i)
    outs: list[str] = [""] * len(specs)
    g = torch.Generator(device=device).manual_seed(args.seed)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for L in sorted(groups):
            idxs = groups[L]
            for b in range(0, len(idxs), args.batch):
                chunk = idxs[b:b + args.batch]
                x = torch.tensor([enc[i] for i in chunk], device=device)
                y = generate(model, x, args.max_new, temperature=args.temperature, top_p=args.top_p,
                             eot_id=eot, generator=g, use_cache=True)
                for i, row in zip(chunk, y[:, L:].tolist()):
                    ended = eot in row
                    text = tok.decode(row[:row.index(eot)] if ended else row)
                    outs[i] = text + ("<|endoftext|>" if ended else "")

    by_kind: dict[str, list[bool]] = defaultdict(list)
    ended, rep4, partial = 0, [], []
    for s, o in zip(specs, outs):
        text = o.removesuffix("<|endoftext|>")
        by_kind[s["kind"]].append(follows(s, text))
        if s["kind"] == "words":
            partial.append(words_found(s, text))
        ended += o.endswith("<|endoftext|>")
        rep4.append(repeated_ngram_frac(text, 4))
    allf = [v for vs in by_kind.values() for v in vs]
    lines = [f"# Prompt-following eval — `{args.ckpt}` (step {st.get('step')})", "",
             f"{len(specs)} held-out prompts{" (chat template)" if args.chat else ""}, T={args.temperature} top-p={args.top_p}, seed {args.seed}.", "",
             "| prompt kind | n | follows |", "|---|---|---|"]
    for k, vs in sorted(by_kind.items()):
        lines.append(f"| {k} | {len(vs)} | {sum(vs) / len(vs):.1%} |")
    lines += [f"| **all** | {len(allf)} | **{sum(allf) / len(allf):.1%}** |", "",
              f"ended with EOT: {ended}/{len(specs)}; mean rep4 {sum(rep4) / len(rep4):.3f}; "
              f"requested words present (words prompts): {sum(partial) / max(len(partial), 1):.1%}", "",
              "## First examples", ""]
    for s, o in list(zip(specs, outs))[:4]:
        lines += [f"**{s['prompt'].strip()}** — follows: {follows(s, o.removesuffix('<|endoftext|>'))}", "",
                  "> " + o.strip().replace("\n", "\n> "), ""]
    report = "\n".join(lines)
    print("\n".join(lines[:12 + len(by_kind)]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
    print("RESULT follows", f"{sum(allf) / len(allf):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
