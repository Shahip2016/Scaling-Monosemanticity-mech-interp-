"""Shared utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_activations(x: torch.Tensor, d_model: int) -> torch.Tensor:
    """Scale activations so average squared L2 norm equals d_model (paper Section 1.1)."""
    current_norm = (x**2).sum(dim=-1).mean()
    scale = (d_model / current_norm.clamp(min=1e-8)).sqrt()
    return x * scale


def load_activations(path: str) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data
    return data["activations"]
