"""Deterministic layout generator (§24, §25, §28).

Deterministic model selection from the curated pool via layout_seed; pose
sampling from configured regions with structured rejection (overlap,
clearance, openable free space, robot base free space); upright alignment.
Same repo + config + layout_seed => identical placements and manifest hash.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from rummagebench.authoring.layout.spec import LayoutConfig, Placement


def _rects_overlap(aabb1, aabb2, clearance: float) -> bool:
    """Axis-aligned XY overlap test with clearance margin."""
    (x0, y0), (x1, y1) = aabb1
    (u0, v0), (u1, v1) = aabb2
    return not (
        x1 + clearance <= u0 or x0 - clearance >= u1
        or y1 + clearance <= v0 or y0 - clearance >= v1
    )


def footprint(p: Placement) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy, _ = p.aabb_size
    return (p.position[0] - sx / 2, p.position[1] - sy / 2), \
           (p.position[0] + sx / 2, p.position[1] + sy / 2)


@dataclass
class GeneratedLayout:
    config: LayoutConfig
    layout_seed: int
    placements: list[Placement] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


class LayoutGenerator:
    def __init__(self, catalog, config: LayoutConfig):
        self.catalog = catalog
        self.config = config
        self._counter = itertools.count()

    def generate(self, layout_seed: int) -> GeneratedLayout:
        rng = np.random.default_rng(layout_seed)  # local generator only (§24)
        self._counter = itertools.count()  # names are seed-deterministic
        layout = GeneratedLayout(config=self.config, layout_seed=layout_seed)
        roles = (
            [("storage", self.config.storage_units),
             ("surface", self.config.work_surfaces)]
        )
        for role, count in roles:
            models = list(self.catalog.models_for(role).items())
            for i in range(count):
                model_id, entry = models[int(rng.integers(len(models)))]
                placement = self._place(layout, rng, role, model_id, entry.aabb_size)
                if placement is None:
                    layout.rejected.append({
                        "role": role, "model_id": model_id,
                        "reason": "NO_VALID_PLACEMENT_AFTER_MAX_ATTEMPTS",
                    })
                    continue
                layout.placements.append(placement)
        # clutter on top of accepted surfaces
        surfaces = [p for p in layout.placements if p.role == "surface"]
        clutter = list(self.catalog.models_for("clutter").items())
        for surface in surfaces:
            for _ in range(self.config.clutter_per_surface):
                model_id, entry = clutter[int(rng.integers(len(clutter)))]
                placement = self._place_on_surface(layout, rng, model_id, entry.aabb_size, surface)
                if placement is not None:
                    layout.placements.append(placement)
        return layout

    # -------------------------------------------------------------- sampling

    def _place(self, layout, rng, role: str, model_id: str, size: list[float]) -> Placement | None:
        regions = self.config.placement_regions
        if not regions:
            raise ValueError("layout config: placement_regions is empty")
        sx, sy, _sz = size
        for _attempt in range(self.config.max_sampling_attempts):
            region = regions[int(rng.integers(len(regions)))]
            x = float(rng.uniform(region[0] + sx / 2, region[2] - sx / 2))
            y = float(rng.uniform(region[1] + sy / 2, region[3] - sy / 2))
            yaw = 0.0  # upright, axis-aligned; the access front faces -y
            placement = Placement(
                name=f"{role}_{next(self._counter):03d}",
                category=role, model_id=model_id,
                position=[x, y, 0.0], orientation_deg=yaw,
                aabb_size=list(size), role=role,
            )
            if self._admissible(layout, placement):
                return placement
        return None

    def _place_on_surface(self, layout, rng, model_id: str, size: list[float],
                          surface: Placement) -> Placement | None:
        sx, sy, _ = size
        ssx, ssy, ssz = surface.aabb_size
        for _attempt in range(self.config.max_sampling_attempts):
            x = float(rng.uniform(
                surface.position[0] - ssx / 2 + sx / 2,
                surface.position[0] + ssx / 2 - sx / 2))
            y = float(rng.uniform(
                surface.position[1] - ssy / 2 + sy / 2,
                surface.position[1] + ssy / 2 - sy / 2))
            placement = Placement(
                name=f"clutter_{next(self._counter):03d}",
                category=model_id, model_id=model_id,
                position=[x, y, surface.position[2] + ssz / 2 + size[2] / 2],
                orientation_deg=0.0, aabb_size=list(size), role="clutter",
            )
            if self._admissible(layout, placement):
                return placement
        return None

    # ------------------------------------------------------------ validation

    def _admissible(self, layout, placement: Placement) -> bool:
        cfg = self.config
        for other in layout.placements:
            if _rects_overlap(footprint(placement), footprint(other),
                              cfg.min_furniture_clearance_m):
                return False
        # robot base free space around the object (§28.9)
        corners = footprint(placement)
        rx, ry = cfg.robot_spawn
        if (_point_near_rect(rx, ry, corners, cfg.min_robot_clearance_m)
                and placement.role != "clutter"):
            return False
        # storage needs a collision-free base strip in FRONT (§28.8/28.9,
        # §29.4): otherwise the unit can never be approached
        if placement.role == "storage":
            front = _front_strip(placement, cfg.min_robot_clearance_m + 0.3)
            for other in layout.placements:
                if other.role == "clutter":
                    continue
                if _rects_overlap(front, footprint(other), 0.0):
                    return False
        return True


def _front_strip(placement: Placement, depth: float):
    """Strip of base free space in front of a yaw-0 storage unit."""
    (x0, y0), (x1, y1) = footprint(placement)
    return ((x0, y0 - depth), (x1, y0))


def _point_near_rect(px: float, py: float, rect, margin: float) -> bool:
    (x0, y0), (x1, y1) = rect
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return math_hypot(dx, dy) < margin


def math_hypot(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5
