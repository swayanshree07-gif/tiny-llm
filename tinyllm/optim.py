"""AdamW with weight-decay parameter groups, and the warmup + cosine LR schedule."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrices (Linear/Embedding weights, dim >= 2); never decay biases or LayerNorm.

    `named_parameters()` yields shared (tied) tensors only once, so the tied
    embedding/head weight lands in exactly one group.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1,
                   betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8) -> torch.optim.AdamW:
    use_fused = all(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps,
                             fused=use_fused)


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warmup from max_lr/warmup to max_lr over `warmup` steps, then cosine to min_lr at `total`.

    `step` is 0-based (the LR used for the update that produces step+1).
    Past `total` the LR stays at min_lr.
    """
    if warmup > 0 and step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
