"""Interactive multi-turn chat with a chat fine-tuned checkpoint (PLAN Day 23).

Reads one user message per line from stdin, prints the assistant reply. Uses the chat template from
tinyllm/chat.py, the KV cache (generate(..., use_cache=True)) and stops at the reply's EOT.
History truncation: the oldest (user, assistant) pairs are dropped until the prompt plus
--max-new tokens fits in the model context (so the KV cache never has to be rebuilt); a single
over-long user message is cut to its last tokens.

Commands: /reset (forget the history), /quit (exit).

Run: .venv/bin/python scripts/chat.py --ckpt ckpt/chat_large_v3 [--temperature 0.8 --top-p 0.95]
     echo 'Tell me a story about "cat", "tree" and "rain".' | .venv/bin/python scripts/chat.py --ckpt ... --max-turns 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm.chat import encode_chat  # noqa: E402
from tinyllm.generate import generate, load_model  # noqa: E402
from tinyllm.tokenizer import Tokenizer  # noqa: E402


def build_prompt(tok, history: list[dict], user_msg: str, block_size: int,
                 max_new: int) -> tuple[list[int], int]:
    """Prompt ids for history + new user message, dropping old turns to leave max_new tokens.

    Returns (ids, n_pairs_dropped)."""
    budget = block_size - max_new
    if budget < 8:
        raise ValueError(f"max_new {max_new} leaves no room for a prompt in block_size {block_size}")
    msgs = history + [{"role": "user", "content": user_msg}]
    dropped = 0
    while True:
        ids, _ = encode_chat(tok, msgs, add_generation_prompt=True)
        if len(ids) <= budget or len(msgs) == 1:
            break
        msgs = msgs[2:]
        dropped += 1
    if len(ids) > budget:  # one very long user message: keep its end (still ends with "\nAssistant:")
        ids = ids[-budget:]
    return ids, dropped


def reply(model, tok, history: list[dict], user_msg: str, max_new: int, temperature: float,
          top_k: int | None, top_p: float | None, generator: torch.Generator | None,
          min_new: int | None = None) -> tuple[str, int]:
    """Generate one assistant reply. Returns (text, n_pairs_dropped).

    History is kept while at least `min_new` reply tokens fit; the reply budget is then
    min(max_new, room left). min_new=None means reserve the full max_new (old behaviour)."""
    T = model.cfg.block_size
    max_new = min(max_new, T - 8)
    reserve = max_new if min_new is None else min(min_new, max_new)
    ids, dropped = build_prompt(tok, history, user_msg, T, reserve)
    max_new = min(max_new, T - len(ids))
    dev = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    eot = tok.vocab_size - 1
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = generate(model, x, max_new, temperature=temperature, top_k=top_k, top_p=top_p,
                       eot_id=eot, generator=generator, use_cache=True)
    new = out[0, len(ids):].tolist()
    if eot in new:
        new = new[:new.index(eot)]
    return tok.decode(new).strip(), dropped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint file or directory (latest is used)")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data/tok/tokenizer.json")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--min-new", type=int, default=120,
                    help="keep chat history while at least this many reply tokens fit")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None, help="exit after this many replies")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    tok = Tokenizer.load(args.tokenizer)
    model, _ = load_model(args.ckpt, args.device)
    if model.cfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"model vocab {model.cfg.vocab_size} != tokenizer vocab {tok.vocab_size}")
    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
    interactive = sys.stdin.isatty()
    if interactive:
        print("Chat with the story bot. /reset clears history, /quit exits.")
    history: list[dict] = []
    turns = 0
    while args.max_turns is None or turns < args.max_turns:
        if interactive:
            print("User: ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            break
        msg = line.strip()
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            history = []
            print("[history cleared]", flush=True)
            continue
        if not interactive:
            print(f"User: {msg}")
        text, dropped = reply(model, tok, history, msg, args.max_new, args.temperature,
                              args.top_k, args.top_p, gen, args.min_new)
        if dropped:
            history = history[2 * dropped:]
            print(f"[dropped {dropped} old turn(s) to fit the context]", file=sys.stderr)
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": text}]
        print(f"Assistant: {text}\n", flush=True)
        turns += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
