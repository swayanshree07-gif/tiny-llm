"""Checkpoints: atomic save, keep only the latest N (project rule: 2), find latest, load."""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch

_PAT = re.compile(r"^ckpt_(\d+)\.pt$")


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    """Checkpoints in `ckpt_dir` sorted by step (oldest first)."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in d.iterdir() if (m := _PAT.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    cks = list_checkpoints(ckpt_dir)
    return cks[-1] if cks else None


def save_checkpoint(state: dict, ckpt_dir: str | Path, step: int, keep: int = 2) -> Path:
    """Write ckpt_<step>.pt atomically (tmp file + rename), then delete all but the newest `keep`."""
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_{step:07d}.pt"
    tmp = d / f".{path.name}.part"
    torch.save(state, tmp)
    os.replace(tmp, path)
    for old in list_checkpoints(d)[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    # weights_only=False: the state includes RNG states and plain-python config dicts we wrote ourselves.
    return torch.load(path, map_location=map_location, weights_only=False)
