from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = torch.load(path, map_location="cpu")
    return loaded
