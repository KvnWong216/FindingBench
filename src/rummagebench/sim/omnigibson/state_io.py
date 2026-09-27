"""Deterministic state snapshot I/O for the OmniGibson backend.

The scenario build captures one snapshot; every episode reset restores it, so
the same environment deterministically reproduces the same episode start.
Snapshots are serialized torch states saved as .pt (large; gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch as th

        th.manual_seed(seed)
    except Exception:
        pass


def dump_state(sim) -> Any:
    """Capture the full simulator state as a serialized (flat tensor) snapshot."""
    return sim.dump_state(serialized=True)


def load_state(sim, state: Any, settle_steps: int = 5) -> None:
    """Restore a snapshot. Several steps let controllers/rendering re-converge
    to the restored state (relative states are inaccurate until stepped)."""
    sim.load_state(state, serialized=True)
    for _ in range(settle_steps):
        sim.step()


def save_snapshot(path: str | Path, state: Any, meta: dict | None = None) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "meta": meta or {}}, path)
    return path


def load_snapshot(path: str | Path) -> tuple[Any, dict]:
    import torch

    payload = torch.load(path, weights_only=False)
    return payload["state"], payload.get("meta", {})


def save_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
