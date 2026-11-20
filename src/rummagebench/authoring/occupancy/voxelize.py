"""§18: conservative voxelization for ordinary objects.

CPU-available path: an axis-aligned solid proxy built from catalog
dimensions (`proxy_source="aabb"`), with every non-zero physical dimension
guaranteed at least one occupied voxel (thin-object rule). The GPU
verification stage may replace it with a collision-geometry voxelization
(`proxy_source="collision"`) or a visual-mesh voxelization when the collider
is unusable (`proxy_source="visual"`, recorded explicitly). Final scene
certification always uses real collision geometry inside the simulator.
"""
from __future__ import annotations

import numpy as np


def object_dims(native_aabb_xyz, collision_aabb_xyz=None) -> np.ndarray:
    dims = np.asarray(collision_aabb_xyz or native_aabb_xyz, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError(f"invalid object dimensions: {dims}")
    return dims


def voxelize_dims(dims_xyz, voxel_size: float = 0.02,
                  min_cells_per_axis: int = 1) -> np.ndarray:
    dims = object_dims(dims_xyz)
    cells = np.maximum(
        np.ceil(dims / float(voxel_size)).astype(int),
        min_cells_per_axis)
    return np.ones(tuple(int(v) for v in cells), dtype=bool)


def aabb_proxy(category: str, model_id: str, native_aabb_xyz,
               voxel_size: float = 0.02) -> dict:
    """Build a CPU conservative proxy record (PROPOSAL stage only)."""
    dims = object_dims(native_aabb_xyz)
    grid = voxelize_dims(dims, voxel_size)
    lo, mid, hi = np.sort(dims)
    return {
        "category": category,
        "model_id": model_id,
        "proxy_source": "aabb",
        "voxel_size_m": float(voxel_size),
        "grid_dims": [int(v) for v in grid.shape],
        "physical_dims_m": [float(v) for v in dims],
        "world_volume_m3": float(np.prod(dims)),
        # §8 shape descriptors
        "flatness": float(lo / hi),
        "elongation": float(hi / mid),
        "compactness": float(mid / lo),
        "note": "proposal-only conservative proxy; final certification uses "
                "real collision geometry",
    }
