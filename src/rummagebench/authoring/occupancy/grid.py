"""§19: occupancy grid, support/height grid, container interior mask."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OccupancyGrid:
    """Voxel-solid occupancy over the local workspace volume.

    origin = world-space corner of cell (0,0,0); every cell is
    voxel_size metres per side. `solid` counts overlapping proposals.
    """
    origin: np.ndarray            # (3,) float, world metres
    dims: tuple[int, int, int]    # cells along x/y/z
    voxel_size: float
    solid: np.ndarray = field(init=False)  # (nx,ny,nz) int32 overlap counts

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float)
        self.solid = np.zeros(self.dims, dtype=np.int32)

    def world_to_cell(self, xyz) -> np.ndarray:
        return np.floor((np.asarray(xyz, dtype=float) - self.origin)
                        / self.voxel_size).astype(int)

    def cell_center(self, cell) -> np.ndarray:
        return self.origin + (np.asarray(cell, dtype=float) + 0.5) * self.voxel_size

    def in_bounds(self, cell) -> bool:
        return bool(np.all(0 <= np.asarray(cell)) and
                    np.all(np.asarray(cell) < np.asarray(self.dims)))

    def occupies(self, cells) -> bool:
        """True if ANY cell in the iterable is inside the grid."""
        return any(self.in_bounds(c) for c in cells)

    def carve(self, cells) -> None:
        for c in cells:
            if self.in_bounds(c):
                self.solid[tuple(c)] = 0

    def fill(self, cells, amount: int = 1) -> None:
        for c in cells:
            if self.in_bounds(c):
                self.solid[tuple(c)] += amount

    def is_free(self, cells) -> bool:
        """A placement is free when every cell is in-bounds and unoccupied."""
        for c in cells:
            if not self.in_bounds(c) or self.solid[tuple(c)] > 0:
                return False
        return True

    def occupied_fraction(self) -> float:
        return float((self.solid > 0).mean())


def box_cells(origin_cell, dims) -> list[tuple[int, int, int]]:
    """All cells of an axis-aligned box proposal in cell coordinates."""
    ox, oy, oz = (int(v) for v in origin_cell)
    dx, dy, dz = (max(1, int(v)) for v in dims)
    return [(ox + i, oy + j, oz + k)
            for i in range(dx) for j in range(dy) for k in range(dz)]


@dataclass
class SupportHeightGrid:
    """§19: the highest supported surface per (x, y) column, in metres.
    Built from the support top AABB; objects add to it when stacked."""
    origin: np.ndarray
    size_xy: tuple[int, int]
    cell_size: float
    top_z: np.ndarray = field(init=False)   # (nx, ny) float

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float)

    @classmethod
    def from_support_box(cls, support_center_xy, support_size_xy,
                         support_top_z: float, cell_size: float = 0.05,
                         ) -> "SupportHeightGrid":
        size = (int(round(support_size_xy[0] / cell_size)),
                int(round(support_size_xy[1] / cell_size)))
        g = cls(origin=np.asarray(support_center_xy, dtype=float)
                - np.asarray([size[0] * cell_size / 2, size[1] * cell_size / 2]),
                size_xy=size, cell_size=cell_size)
        g.top_z = np.full(size, float(support_top_z), dtype=float)
        return g

    def in_support(self, xy) -> bool:
        idx = np.floor((np.asarray(xy, dtype=float) - self.origin)
                       / self.cell_size).astype(int)
        return bool(np.all(0 <= idx) and np.all(idx < np.asarray(self.size_xy)))

    def height_at(self, xy) -> float:
        idx = np.floor((np.asarray(xy, dtype=float) - self.origin)
                       / self.cell_size).astype(int)
        if not self.in_support(xy):
            raise ValueError("column outside support")
        return float(self.top_z[tuple(idx)])

    def raise_column(self, xy, new_top: float) -> None:
        idx = np.floor((np.asarray(xy, dtype=float) - self.origin)
                       / self.cell_size).astype(int)
        self.top_z[tuple(idx)] = max(self.top_z[tuple(idx)], float(new_top))


@dataclass
class ContainerMask:
    """§19: interior cells of an open-top container (local frame), plus the
    opening plane. Built from the ACTUAL loaded collider cavity at GPU stage;
    the proposal stage uses conservative interior estimates only."""
    origin: np.ndarray            # container interior origin (world)
    dims: tuple[int, int, int]
    voxel_size: float
    interior: np.ndarray = field(init=False)  # bool

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float)
        self.interior = np.ones(self.dims, dtype=bool)

    def volume_cells(self) -> int:
        return int(self.interior.sum())

    def free_interior_cells(self, excluded_cells) -> set[tuple[int, int, int]]:
        excluded = {tuple(int(v) for v in c) for c in excluded_cells}
        cells = set()
        it = np.argwhere(self.interior)
        for c in it:
            t = tuple(int(v) for v in c)
            if t not in excluded:
                cells.add(t)
        return cells

    def world_center(self, cell) -> np.ndarray:
        return self.origin + (np.asarray(cell, dtype=float) + 0.5) * self.voxel_size
