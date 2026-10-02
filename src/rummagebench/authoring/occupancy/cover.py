"""§22/rev§6: cover relation construction and coarse validation."""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.occupancy.voxelize import voxelize_dims


def cavity_cover_feasible(cover_dims_xyz, target_dims_xyz,
                          voxel_size: float) -> bool:
    """Coarse cavity test: the target's conservative voxel proposal must fit
    inside the cover's voxel volume (an inverted bowl must actually contain
    the target). The GPU stage re-checks against the REAL collider cavity —
    AABB inclusion alone never certifies a cover."""
    cover = np.asarray(cover_dims_xyz, dtype=float)
    target = np.asarray(target_dims_xyz, dtype=float)
    cover_grid = voxelize_dims(cover, voxel_size)
    target_grid = voxelize_dims(target, voxel_size)
    return bool(np.all(np.asarray(target_grid.shape) <=
                       np.asarray(cover_grid.shape)))


def footprint_cover_feasible(cover_footprint_xy, target_footprint_xy) -> bool:
    """Footprint-cover test: the cover's projected footprint must contain the
    target's projected footprint."""
    return bool(np.all(np.asarray(cover_footprint_xy) >=
                       np.asarray(target_footprint_xy)))


def select_cover_relation(cover_dims_xyz, target_dims_xyz,
                          voxel_size: float) -> str | None:
    """Pick the strongest feasible cover relation for a (cover, target) pair."""
    if cavity_cover_feasible(cover_dims_xyz, target_dims_xyz, voxel_size):
        return "cavity_cover"
    cover_fp = np.asarray(cover_dims_xyz, dtype=float)[:2]
    target_fp = np.asarray(target_dims_xyz, dtype=float)[:2]
    if footprint_cover_feasible(cover_fp, target_fp):
        return "footprint_cover"
    return None
