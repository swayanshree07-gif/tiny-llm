"""Autoregressive sampling: temperature, top-k, top-p (nucleus). No KV cache yet (PLAN Day 24)."""
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
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Extend idx (B, T) by up to max_new_tokens. temperature <= 0 means greedy (argmax).

    If eot_id is given and every row has produced it, generation stops early.
    """
    model.eval()
    T = model.cfg.block_size
    done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -T:])
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
