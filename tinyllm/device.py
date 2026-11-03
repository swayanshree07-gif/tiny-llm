"""Device helpers shared by training, evaluation and generation."""
from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the CUDA device if available (and requested), else CPU."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def best_autocast_dtype(device: torch.device) -> torch.dtype:
    """bf16 on GPUs that support it (no GradScaler needed), else fp16 on CUDA, fp32 on CPU."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# Project rule: keep VRAM use under ~6.5 GB on the 8 GB RTX 5050.
VRAM_BUDGET_BYTES = int(6.5 * 1024**3)
