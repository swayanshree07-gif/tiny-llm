"""Int8 weight-only quantization, per output channel, symmetric (PLAN Day 25). Own code, no libraries.

Each nn.Linear weight W (out, in) becomes int8 Q and an fp32 scale s (out,) with s = max|W_row| / 127,
Q = round(W / s) clamped to [-127, 127]. At run time W ≈ Q * s is dequantized to the activation dtype
on the fly (weight-only: activations, LayerNorm and embeddings stay in float). The tied LM head is
quantized as its own Linear, so the token embedding keeps full precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.detach().float()
    scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


class Int8Linear(nn.Module):
    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        q, s = quantize_per_channel(lin.weight)
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.register_buffer("qweight", q)
        self.register_buffer("scale", s)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    def dequant(self, dtype=torch.float32) -> torch.Tensor:
        return self.qweight.to(dtype) * self.scale.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = torch.get_autocast_dtype("cuda") if (x.is_cuda and torch.is_autocast_enabled("cuda")) else x.dtype
        b = None if self.bias is None else self.bias.to(dt)
        return F.linear(x.to(dt), self.dequant(dt), b)


def quantize_model(model: nn.Module) -> nn.Module:
    """Replace every nn.Linear in place (recursively) with Int8Linear. Returns the model."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, Int8Linear(child))
        else:
            quantize_model(child)
    return model


def model_bytes(model: nn.Module) -> int:
    """Bytes of all unique parameters and persistent buffers (tied weights counted once)."""
    seen, n = set(), 0
    for t in model.state_dict().values():  # persistent only
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        n += t.numel() * t.element_size()
    return n
