"""Training batches: random contiguous windows from a uint16 token memmap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: str | Path, block_size: int, seed: int = 0) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        assert len(self.data) > block_size + 1, "token file shorter than one block"
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, device: torch.device | str = "cpu"):
        """x, y of shape (B, T) int64; y is x shifted left by one token."""
        T = self.block_size
        starts = self.rng.integers(0, len(self.data) - T - 1, size=batch_size)
        buf = np.stack([self.data[s : s + T + 1] for s in starts]).astype(np.int64)
        xy = torch.from_numpy(buf)
        x, y = xy[:, :-1].contiguous(), xy[:, 1:].contiguous()
        device = torch.device(device)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y

    # RNG state is part of a checkpoint so resumed runs see the same batches.
    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]


def load_meta(tok_dir: str | Path) -> dict:
    return json.loads((Path(tok_dir) / "meta.json").read_text())
