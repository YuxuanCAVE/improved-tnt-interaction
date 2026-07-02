from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compatible_vectornet_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("subgraph.") for key in state):
        return state
    converted = {key: value for key, value in state.items() if not key.startswith("subgraph.")}
    for key, value in state.items():
        if not key.startswith("subgraph."):
            continue
        suffix = key[len("subgraph.") :]
        converted[f"traj_subgraph.{suffix}"] = value
        converted[f"map_subgraph.{suffix}"] = value
    return converted
