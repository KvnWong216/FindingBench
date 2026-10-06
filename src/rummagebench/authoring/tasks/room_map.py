"""CPU replica of OmniGibson's room segmentation + traversability maps.

Reads the BEHAVIOR scene layout PNGs (layout/floor_insseg_0.png,
floor_semseg_0.png, floor_trav_0.png) with exactly OmniGibson's conventions
(omnigibson/maps/segmentation_map.py + map_base.py, v3.9.3):

  * images are 0.01 m/pixel, resized (nearest) to ``resolution``;
  * room instance names are "<room_category>_<i>", i = rank of the instance
    id among that category's instance ids (ascending);
  * map_to_world(row, col) = ((col - size/2) * res, (row - size/2) * res).

Used by the embodiment probe to propose robot start / interaction poses without
a simulator query; the simulator still has the last word (footprint
collision check, IK, rendering).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RoomMap:
    resolution: float
    size: int
    room_ins: np.ndarray  # (size, size) instance id, 0 = none
    trav: np.ndarray  # (size, size) bool, traversable with objects
    names: dict[int, str]  # instance id -> room name (e.g. kitchen_0)

    @classmethod
    def load(cls, scene_dir: str | Path, room_categories: str | Path,
             resolution: float = 0.05) -> "RoomMap":
        import cv2

        layout = Path(scene_dir) / "layout"
        ins = cv2.imread(str(layout / "floor_insseg_0.png"), cv2.IMREAD_GRAYSCALE)
        sem = cv2.imread(str(layout / "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
        trav = cv2.imread(str(layout / "floor_trav_0.png"), cv2.IMREAD_GRAYSCALE)
        if ins is None or sem is None or trav is None:
            raise FileNotFoundError(f"layout maps missing under {layout}")
        cats = [ln.rstrip() for ln in Path(room_categories).read_text().splitlines()]
        names: dict[int, str] = {}
        per_sem: dict[int, list[int]] = {}
        for ins_id in sorted(int(v) for v in np.unique(ins) if v != 0):
            r, c = np.argwhere(ins == ins_id)[0]
            per_sem.setdefault(int(sem[r, c]), []).append(ins_id)
        for sem_id, ids in per_sem.items():
            for i, ins_id in enumerate(ids):
                names[ins_id] = f"{cats[sem_id - 1]}_{i}"
        size = int(ins.shape[0] * 0.01 / resolution)
        ins_r = cv2.resize(ins, (size, size), interpolation=cv2.INTER_NEAREST)
        trav_r = cv2.resize(trav, (size, size), interpolation=cv2.INTER_NEAREST)
        return cls(resolution, size, ins_r.astype(np.int32), trav_r > 0, names)

    # ------------------------------------------------------------ frames

    def to_world(self, row: float, col: float) -> tuple[float, float]:
        return (float((col - self.size / 2.0) * self.resolution),
                float((row - self.size / 2.0) * self.resolution))

    def to_map(self, x: float, y: float) -> tuple[int, int]:
        return (int(y / self.resolution + self.size / 2.0),
                int(x / self.resolution + self.size / 2.0))

    def room_at(self, x: float, y: float) -> Optional[str]:
        r, c = self.to_map(x, y)
        if not (0 <= r < self.size and 0 <= c < self.size):
            return None
        return self.names.get(int(self.room_ins[r, c]))

    def traversable(self, x: float, y: float) -> bool:
        r, c = self.to_map(x, y)
        return bool(0 <= r < self.size and 0 <= c < self.size and self.trav[r, c])

    # ------------------------------------------------------------ sampling

    def free_points(self, room: str, clearance_m: float,
                    spacing_m: float) -> list[tuple[float, float]]:
        """Grid points of ``room`` whose disc of radius clearance_m is
        traversable and inside the room (deterministic, row-major)."""
        import cv2

        ids = [i for i, n in self.names.items() if n == room]
        if not ids:
            return []
        mask = (np.isin(self.room_ins, ids) & self.trav).astype(np.uint8)
        k = max(1, int(math.ceil(clearance_m / self.resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask = cv2.erode(mask, kernel)
        step = max(1, int(round(spacing_m / self.resolution)))
        out = []
        for r in range(0, self.size, step):
            for c in range(0, self.size, step):
                if mask[r, c]:
                    out.append(self.to_world(r, c))
        return out

    def clearest_points(self, k: int = 10,
                        room: Optional[str] = None) -> list[tuple[float, tuple[float, float]]]:
        """The k traversable points farthest from any non-traversable cell
        (and, with ``room``, from the room boundary), as (clearance_m, (x, y)),
        best first (deterministic)."""
        import cv2

        mask = self.trav
        if room is not None:
            ids = [i for i, n in self.names.items() if n == room]
            mask = mask & np.isin(self.room_ins, ids)
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        flat = np.argsort(-dist, axis=None, kind="stable")[:k]
        out = []
        for idx in flat:
            r, c = np.unravel_index(int(idx), dist.shape)
            out.append((float(dist[r, c]) * self.resolution, self.to_world(r, c)))
        return out
