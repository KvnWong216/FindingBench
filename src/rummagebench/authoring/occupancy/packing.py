"""§19-§22: non-overlapping placement proposals against the occupancy grid.

Packing answers WHERE a proposed object may go:
- pack_on_support: tabletop clutter with a local high-density clutter region
  (never uniform over the table);
- pack_in_container: interior cells of an open-top container (container-
  conditioned sampling, §12 — only objects whose voxel proposal fits);
- propose_cover: a cover relation whose projected footprint covers the
  target (cavity-aware; a bowl merely nearby is NOT covering).

Every proposal is a PROPOSAL: PhysX relaxation + exact validation happen
later in the simulator.
"""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.occupancy.grid import (
    ContainerMask,
    OccupancyGrid,
    SupportHeightGrid,
    box_cells,
)


def _proxy_cells(grid_dims, rng) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Randomly oriented proxy footprint: permute the proxy grid axes to
    approximate an axis-aligned orientation bin (fine collision checks at
    the proposed transform happen later, revision §7)."""
    perm = tuple(rng.permutation(3))
    return np.transpose(grid_dims), perm


def pack_on_support(grid: OccupancyGrid, support: SupportHeightGrid,
                    proxy_grid: np.ndarray, voxel_size: float,
                    rng, cluster_centers_xy: list[np.ndarray],
                    edge_margin_m: float = 0.08,
                    max_attempts: int = 400) -> dict | None:
    """Propose a non-overlapping placement of `proxy_grid` on the support.

    Strategy: pick one of the clutter cluster centers, scan outward (seeded
    jittered spiral) for a free column whose stacked height accepts the
    proxy; register the proposal into the grid and the height map.
    Returns {"origin_world", "cells", "support_z"} or None (pack failed).
    """
    dims = np.asarray(proxy_grid.shape)
    footprint_xy = dims[:2] * voxel_size
    for _ in range(max_attempts):
        center = cluster_centers_xy[rng.integers(0, len(cluster_centers_xy))]
        angle = rng.random() * 2.0 * np.pi
        radius = rng.random() * 0.25
        xy = center + np.array([np.cos(angle), np.sin(angle)]) * radius
        if not support.in_support(xy):
            continue
        half = footprint_xy / 2.0
        lo = xy - half
        hi = xy + half
        margin_ok = all(support.in_support(c) for c in (lo, hi))
        if not margin_ok:
            continue
        # columns of the footprint must be inside the support with margin
        steps = max(2, int(max(footprint_xy) / support.cell_size) + 1)
        cols = [lo + (hi - lo) * (i / steps) for i in range(steps + 1)]
        cols += [np.array([lo[0], hi[1]]), np.array([hi[0], lo[1]])]
        if not all(support.in_support(c) for c in cols):
            continue
        support_z = max(support.height_at(c) for c in cols)
        base_cell = grid.world_to_cell(
            [lo[0], lo[1], support_z + 1e-4])
        # the proxy stands on the support: only its bottom layer occupies the
        # first row of cells; upper layers stack through the height map
        cells = box_cells(base_cell, dims)
        if not grid.is_free(cells):
            continue
        grid.fill(cells)
        for c in cols:
            support.raise_column(c, support_z + dims[2] * voxel_size)
        return {
            "origin_world": [float(v) for v in
                             grid.cell_center(base_cell)
                             - 0.5 * dims * voxel_size],
            "support_z": support_z,
            "cells": cells,
            "dims": [int(v) for v in dims],
        }
    return None


def pack_on_support_clusters(grid: OccupancyGrid, support: SupportHeightGrid,
                             rng, n_objects: int, proxy_grids: list[np.ndarray],
                             voxel_size: float, surface_size_xy,
                             surface_top_z: float) -> list[dict] | None:
    """§20: 1-2 clutter cluster centers + coverage-ordered packing; requires
    at least one local high-density region (not uniform spreading)."""
    support.origin = np.asarray(surface_center_xy(surface_size_xy), dtype=float) \
        - np.asarray(surface_size_xy) / 2.0
    support.top_z = np.full(support.size_xy, float(surface_top_z), dtype=float)
    margin = 0.08
    usable = (np.asarray(surface_size_xy) - 2 * margin)
    n_clusters = int(rng.integers(1, 3))
    centers = [np.array([margin + usable[0] * rng.random(),
                         margin + usable[1] * rng.random()])
               + support.origin for _ in range(n_clusters)]
    placed = []
    for proxy in proxy_grids[:n_objects]:
        proposal = pack_on_support(grid, support, proxy, voxel_size, rng, centers)
        if proposal is None:
            return None
        placed.append(proposal)
    return placed


def surface_center_xy(surface_size_xy) -> np.ndarray:
    return np.asarray(surface_size_xy, dtype=float) / 2.0


def pack_in_container(grid: OccupancyGrid, container: ContainerMask,
                      proxy_grid: np.ndarray, rng,
                      max_attempts: int = 300) -> dict | None:
    """§21: pack one object into the container interior mask. Objects may
    stack; the target's fit is a hard precondition (§12)."""
    free = container.free_interior_cells([])
    if not free:
        return None
    dims = np.asarray(proxy_grid.shape)
    for _ in range(max_attempts):
        anchor = np.asarray(list(free)[rng.integers(0, len(free))])
        base = anchor - dims // 2 + np.array([0, 0, dims[2] // 2])
        cells = [(int(base[0] + i), int(base[1] + j), int(base[2] + k))
                 for i in range(dims[0]) for j in range(dims[1])
                 for k in range(dims[2])]
        inside_ok = all(tuple(c) in free or container.interior[
            tuple(np.clip(c, 0, np.asarray(container.dims) - 1))] for c in cells)
        if not inside_ok:
            continue
        if not grid.is_free(cells):
            continue
        grid.fill(cells)
        return {"cells_local": cells,
                "origin_world": [float(v) for v in
                                 container.world_center(base)
                                 - 0.5 * dims * container.voxel_size],
                "dims": [int(v) for v in dims]}
    return None


def propose_cover(proxy_grid_cover: np.ndarray,
                  proxy_grid_target: np.ndarray,
                  voxel_size: float, rng) -> dict | None:
    """§22/rev§6: build a cavity-cover relation in occupancy space.

    Valid only when the target's voxel proposal fits INSIDE the cover's
    voxel volume with the cover resting inverted above the support plane —
    the cover cavity must genuinely contain the target (AABB inclusion alone
    is insufficient at GPU stage; here it is the coarse precondition).
    """
    cover_dims = np.asarray(proxy_grid_cover.shape)
    target_dims = np.asarray(proxy_grid_target.shape)
    if np.any(target_dims > cover_dims):
        return None  # target does not fit the cover cavity
    margin_cells = 0  # conservative proxy already dilates thin axes
    inner = target_dims + 2 * margin_cells
    if np.any(inner > cover_dims):
        return None
    offset = ((cover_dims - target_dims) // 2)
    return {
        "relation": "cavity_cover",
        "offset_cells": [int(v) for v in offset],
        "cover_dims": [int(v) for v in cover_dims],
        "target_dims": [int(v) for v in target_dims],
        "note": "coarse proposal; GPU stage re-checks the real collider cavity",
    }
