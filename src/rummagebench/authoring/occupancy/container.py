"""§12/§14/rev§6: container interior, fit tests, stability notes."""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.occupancy.grid import ContainerMask
from rummagebench.authoring.occupancy.voxelize import voxelize_dims


def build_container_mask(interior_origin, interior_size_xyz,
                         voxel_size: float) -> ContainerMask:
    """Interior mask from the container's ACTUAL usable interior (filled by
    the GPU collider-cavity stage). Never derived from a convex hull that
    would close the cavity."""
    dims = np.maximum(
        np.ceil(np.asarray(interior_size_xyz, dtype=float) / voxel_size).astype(int),
        1)
    return ContainerMask(origin=np.asarray(interior_origin, dtype=float),
                         dims=tuple(int(v) for v in dims),
                         voxel_size=voxel_size)


def object_fits(proxy_grid: np.ndarray, container: ContainerMask) -> bool:
    """Container-conditioned fit test (§12): the object's voxel proposal must
    fit inside the interior volume before it is sampled (no discover-too-late)."""
    if container.volume_cells() == 0:
        return False
    return bool(np.all(np.asarray(proxy_grid.shape) <=
                       np.asarray(container.dims)))


def interior_accepts_cells(container: ContainerMask, cells) -> bool:
    inside = container.free_interior_cells([])
    return all(tuple(int(v) for v in c) in inside for c in cells)
