"""§31/§45: immutable environment snapshots, atomic writes, hashing.

Accepted environments are immutable: writes go through tmp+rename, and an
existing accepted environment is never overwritten unless explicitly forced.
The canonical state hash binds the environment identity across the five
robot instantiations.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> str:
    """Stable JSON: sorted keys, fixed separators, no inf/nan."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """§45: atomic candidate output — tmp file + rename, never partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(canonical_json(payload))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_environment_snapshot(env_dir: Path, artifacts: dict[str, Any]) -> None:
    """§31: write the immutable environment directory atomically.

    Existing accepted snapshots are never overwritten unless `force` was
    requested by the operator (handled by the caller).
    """
    env_dir = Path(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        if name.endswith(".png"):
            import base64
            target = env_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(payload) if isinstance(payload, str)
                        else payload)
            os.replace(tmp, target)
        else:
            atomic_write_json(env_dir / name, payload)


def environment_manifest(environment_id: str, plan_dict: dict,
                         candidate: dict, appearance: dict) -> dict:
    """§31: canonical manifest WITHOUT robot-dependent perturbation."""
    manifest = {
        "environment_id": environment_id,
        "paradigm": plan_dict["paradigm"],
        "support_model": plan_dict["support_model"],
        "container_model": plan_dict.get("container_model"),
        "objects": candidate["objects"],
        "placements": candidate["placements"],
        "relations": candidate.get("relations", []),
        "occupancy_spec": candidate["occupancy_spec"],
        "surface": candidate["surface"],
        "appearance": appearance,
    }
    manifest["manifest_hash"] = canonical_hash(
        {k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest
