"""§18/rev§7: voxel proxy cache.

Cache identity binds asset hash, proxy source, scale, orientation bin,
voxel resolution and preprocessing version, so an unrotated mask can never
be reused after an orientation change. Entries are content-checked (crc).
"""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

PROXY_VERSION = 1


def cache_key(category: str, model_id: str, proxy_source: str,
              scale: float, orientation_bin: int, voxel_size: float) -> str:
    payload = json.dumps({
        "category": category, "model": model_id, "source": proxy_source,
        "scale": round(float(scale), 6), "orientation_bin": int(orientation_bin),
        "voxel": float(voxel_size), "version": PROXY_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def proxy_cache_path(cache_root: Path, category: str, model_id: str) -> Path:
    return Path(cache_root) / "voxels" / f"{category}__{model_id}.npz"


def save_proxy(path: Path, grid: np.ndarray, voxel_size: float,
               proxy_source: str, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"grid": grid.astype(np.bool_), "voxel_size": float(voxel_size),
               "source": proxy_source, "version": PROXY_VERSION,
               "crc": zlib.crc32(grid.tobytes())}
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def load_proxy(path: Path) -> dict:
    data = np.load(Path(path), allow_pickle=False)
    grid = data["grid"]
    if zlib.crc32(grid.tobytes()) != int(data["crc"]):
        raise ValueError(f"voxel proxy cache corrupted: {path}")
    return {"grid": grid, "voxel_size": float(data["voxel_size"]),
            "source": str(data["source"]), "version": int(data["version"])}
