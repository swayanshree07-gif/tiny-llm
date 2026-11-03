"""GPT decoder components, written from scratch.

Pre-LayerNorm decoder: x + attn(ln(x)), x + mlp(ln(x)); tied input/output embeddings.
Positions: cfg.pos = "learned" (absolute embeddings) or "rope" (rotary, applied to q and k).

Causal self-attention has two interchangeable paths:
  - "manual": explicit QK^T / sqrt(d), causal mask, softmax, @V (readable reference)
  - "sdpa":   torch.nn.functional.scaled_dot_product_attention(is_causal=True) (fast)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    mlp_ratio: int = 4
    dropout: float = 0.0
    attn_impl: str = "sdpa"
    pos: str = "rope"  # "rope" or "learned"
    rope_base: float = 10000.0

    def to_dict(self) -> dict:
        return asdict(self)


def rope_cache(T: int, head_dim: int, base: float = 10000.0, device=None):
    """cos/sin tables of shape (T, head_dim/2) for rotary embeddings."""
    assert head_dim % 2 == 0
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) of x (..., T, D) by position-dependent angles."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, impl: str = "sdpa") -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert impl in ("sdpa", "manual")
        self.n_heads, self.head_dim, self.impl = n_heads, d_model // n_heads, impl
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope: tuple | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, D)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            cos, sin = rope[0][:T], rope[1][:T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        p = self.dropout if self.training else 0.0
        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            att = F.dropout(att, p=p, training=self.training)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, ratio * d_model, bias=False)
        self.proj = nn.Linear(ratio * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.attn_impl)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor, rope: tuple | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        assert cfg.pos in ("learned", "rope")
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos == "learned" else None
        if cfg.pos == "rope":
            cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_heads, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        # GPT-2 style: scale residual output projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        rope = None
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        else:
            rope = (self.rope_cos, self.rope_sin)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, rope)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
