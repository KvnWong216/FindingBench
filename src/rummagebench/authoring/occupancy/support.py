"""§13: verified support / work surface helpers."""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.occupancy.grid import SupportHeightGrid


def build_support_grid(surface_center_xy, surface_size_xy, surface_top_z: float,
                       cell_size: float = 0.05) -> SupportHeightGrid:
    """Height grid over one verified support: `surface_top_z` must come from
    the loaded support asset (never assumed equal to a nominal height)."""
    return SupportHeightGrid.from_support_box(
        surface_center_xy, surface_size_xy, surface_top_z, cell_size)


def sample_cluster_centers(support_grid: SupportHeightGrid,
                           margin_m: float, rng, n_clusters: int = 2,
                           ) -> list[np.ndarray]:
    """§20: 1-2 clutter cluster centers inside the usable surface area."""
    size = np.asarray(support_grid.size_xy, dtype=float) * support_grid.cell_size
    usable = np.maximum(size - 2.0 * margin_m, 1e-3)
    centers = []
    for _ in range(n_clusters):
        local = margin_m + usable * rng.random(2)
        centers.append(support_grid.origin + local)
    return centers
