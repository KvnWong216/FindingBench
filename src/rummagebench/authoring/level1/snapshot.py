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
import shutil
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
    """Publish a whole immutable bundle; identical reruns are idempotent.

    Reject conflicting/partial existing bundles before any mutation. The
    sibling lock serializes writers, and rename publishes a complete bundle.
    Force is deliberately not available in this low-level freeze operation.
    """
    import base64

    env_dir = Path(env_dir)
    encoded = {}
    for name, payload in artifacts.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"invalid snapshot artifact path: {name}")
        if name.endswith(".png"):
            encoded[name] = base64.b64decode(payload, validate=True) if isinstance(payload, str) else bytes(payload)
        else:
            encoded[name] = canonical_json(payload).encode("utf-8")
    if not encoded:
        raise ValueError("empty snapshot bundle")
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = env_dir.parent / f".{env_dir.name}.freeze.lock"
    # An interrupted writer can leave a lock, requiring explicit inspection;
    # it cannot cause a later writer to overwrite accepted state.
    with lock.open("x"):
        stage = None
        try:
            if env_dir.exists():
                if any(not (env_dir / n).is_file() or (env_dir / n).read_bytes() != data
                       for n, data in encoded.items()):
                    raise FileExistsError(f"conflicting or incomplete frozen snapshot: {env_dir}")
                return
            stage = Path(tempfile.mkdtemp(dir=env_dir.parent, prefix=f".{env_dir.name}."))
            for name, data in encoded.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            stage.rename(env_dir)
        finally:
            if stage is not None and stage.exists():
                shutil.rmtree(stage)
            lock.unlink()


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
