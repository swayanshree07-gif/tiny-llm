"""Autoregressive sampling: temperature, top-k, top-p (nucleus), optional KV cache.

With use_cache=True the prompt is run once (prefill) and then each step feeds only the new token.
While the sequence fits in block_size this gives exactly the same logits as the uncached path.
When the cache is full, it is rebuilt from the last block_size // 2 tokens (one prefill), so from
then on the model sees a shorter, jumping window than the uncached sliding window: outputs past
block_size can differ between the two modes.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import latest_checkpoint, load_checkpoint
from .model import GPT, GPTConfig


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Return logits (..., V) with tokens outside top-k / the top-p nucleus set to -inf.

    top-p keeps the smallest set of most-likely tokens whose probability mass is >= top_p
    (the token that crosses the threshold is kept; the most likely token is always kept).
    """
    logits = logits.clone()
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits[logits < kth] = float("-inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs  # mass of strictly more likely tokens
        drop_sorted = mass_before >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, idx, drop_sorted)
        logits[drop] = float("-inf")
    return logits


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None, eot_id: int | None = None,
             generator: torch.Generator | None = None, use_cache: bool = False) -> torch.Tensor:
    """Extend idx (B, T) by up to max_new_tokens. temperature <= 0 means greedy (argmax).

    If eot_id is given and every row has produced it, generation stops early.
    use_cache=True uses a KV cache (see module docstring for behaviour past block_size).
    """
    model.eval()
    T = model.cfg.block_size
    done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    cache = None
    for _ in range(max_new_tokens):
        if not use_cache:
            logits, _ = model(idx[:, -T:])
        elif cache is None or cache[0].pos >= T:
            # (re)fill: whole prompt if it fits with room to grow, else the last half-window
            ctx = idx if idx.size(1) < T else idx[:, -max(1, T // 2):]
            cache = model.new_kv_cache(idx.size(0), idx.device)
            logits, _ = model(ctx, kv_cache=cache)
        else:
            logits, _ = model(idx[:, -1:], kv_cache=cache)
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = filter_logits(logits / temperature, top_k, top_p)
            nxt = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)
        if eot_id is not None:
            nxt[done] = eot_id  # finished rows just pad with eot
            done |= nxt.squeeze(1) == eot_id
        idx = torch.cat([idx, nxt], dim=1)
        if eot_id is not None and bool(done.all()):
            break
    return idx


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[GPT, dict]:
    """Load a GPT from a checkpoint file, or from the latest checkpoint in a directory."""
    p = Path(path)
    if p.is_dir():
        ck = latest_checkpoint(p)
        if ck is None:
            raise FileNotFoundError(f"no checkpoints in {p}")
        p = ck
    st = load_checkpoint(p)
    model = GPT(GPTConfig(**st["model_config"]))
    model.load_state_dict(st["model"])
    return model.to(device).eval(), st
