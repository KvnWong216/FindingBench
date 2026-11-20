"""Deterministic layout generator (§24, §25, §28).

Deterministic model selection from the curated pool via layout_seed; pose
sampling from configured regions with structured rejection (overlap,
clearance, openable free space, robot base free space); upright alignment.
Same repo + config + layout_seed => identical placements and manifest hash.
"""

from __future__ import annotations

import itertools
import math
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
    angle = math.radians(p.orientation_deg)
    sx, sy = abs(math.cos(angle)) * sx + abs(math.sin(angle)) * sy, abs(math.sin(angle)) * sx + abs(math.cos(angle)) * sy
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
            if count == 0:
                continue
            models = list(self.catalog.models_for(role).items())
            for i in range(count):
                model_id, entry = models[int(rng.integers(len(models)))]
                placement = self._place(layout, rng, role, model_id, entry.aabb_size, entry.category)
                if placement is None:
                    layout.rejected.append({
                        "role": role, "model_id": model_id,
                        "reason": "NO_VALID_PLACEMENT_AFTER_MAX_ATTEMPTS",
                    })
                    continue
                layout.placements.append(placement)
        # clutter on top of accepted surfaces
        surfaces = [p for p in layout.placements if p.role == "surface"]
        clutter = (list(self.catalog.models_for("clutter").items())
                   if surfaces and self.config.clutter_per_surface else [])
        for surface in surfaces:
            for _ in range(self.config.clutter_per_surface):
                model_id, entry = clutter[int(rng.integers(len(clutter)))]
                placement = self._place_on_surface(layout, rng, model_id, entry.aabb_size, surface, entry.category)
                if placement is not None:
                    layout.placements.append(placement)
                else:
                    layout.rejected.append({"role": "clutter", "model_id": model_id,
                                            "support_name": surface.name,
                                            "reason": "NO_VALID_PLACEMENT_AFTER_MAX_ATTEMPTS"})
        return layout

    # -------------------------------------------------------------- sampling

    def _place(self, layout, rng, role: str, model_id: str, size: list[float], category: str) -> Placement | None:
        regions = self.config.placement_regions
        if not regions:
            raise ValueError("layout config: placement_regions is empty")
        sx, sy, sz = size
        regions = [r for r in regions if r[2] - r[0] >= sx and r[3] - r[1] >= sy]
        if not regions:
            return None
        for _attempt in range(self.config.max_sampling_attempts):
            region = regions[int(rng.integers(len(regions)))]
            x = float(rng.uniform(region[0] + sx / 2, region[2] - sx / 2))
            y = float(rng.uniform(region[1] + sy / 2, region[3] - sy / 2))
            yaw = 0.0  # upright, axis-aligned; the access front faces -y
            placement = Placement(
                name=f"{role}_{next(self._counter):03d}",
                category=category, model_id=model_id,
                position=[x, y, sz / 2], orientation_deg=yaw,
                aabb_size=list(size), role=role,
            )
            if self._admissible(layout, placement):
                return placement
        return None

    def _place_on_surface(self, layout, rng, model_id: str, size: list[float],
                          surface: Placement, category: str) -> Placement | None:
        sx, sy, _ = size
        ssx, ssy, ssz = surface.aabb_size
        if sx > ssx or sy > ssy:
            return None
        for _attempt in range(self.config.max_sampling_attempts):
            x = float(rng.uniform(
                surface.position[0] - ssx / 2 + sx / 2,
                surface.position[0] + ssx / 2 - sx / 2))
            y = float(rng.uniform(
                surface.position[1] - ssy / 2 + sy / 2,
                surface.position[1] + ssy / 2 - sy / 2))
            placement = Placement(
                name=f"clutter_{next(self._counter):03d}",
                category=category, model_id=model_id,
                position=[x, y, surface.position[2] + ssz / 2 + size[2] / 2],
                orientation_deg=0.0, aabb_size=list(size), role="clutter",
                support_name=surface.name,
            )
            if self._admissible(layout, placement):
                return placement
        return None

    # ------------------------------------------------------------ validation

    def _admissible(self, layout, placement: Placement) -> bool:
        cfg = self.config
        for other in layout.placements:
            if placements_overlap(placement, other, cfg.min_furniture_clearance_m):
                return False
        corners = footprint(placement)
        rx, ry = cfg.robot_spawn
        if (_point_near_rect(rx, ry, corners, cfg.min_robot_clearance_m)
                and placement.role != "clutter"):
            return False
        # Protect every cabinet's full access strip, regardless of insertion order.
        if not access_strips_free(layout.placements + [placement], cfg):
            return False
        return True


def _front_strip(placement: Placement, depth: float):
    """Conservative world AABB of the local -Y access strip."""
    sx, sy, _ = placement.aabb_size
    theta = math.radians(placement.orientation_deg)
    c, s = math.cos(theta), math.sin(theta)
    pts = [(placement.position[0] + c*x - s*y,
            placement.position[1] + s*x + c*y)
           for x in (-sx/2, sx/2) for y in (-sy/2-depth, -sy/2)]
    return ((min(x for x,y in pts), min(y for x,y in pts)),
            (max(x for x,y in pts), max(y for x,y in pts)))


def placements_overlap(a: Placement, b: Placement, clearance: float) -> bool:
    """3D overlap; supporting surface contact is not volumetric overlap."""
    az, bz = a.position[2], b.position[2]
    ah, bh = a.aabb_size[2]/2, b.aabb_size[2]/2
    if az + ah <= bz - bh + 1e-9 or bz + bh <= az - ah + 1e-9:
        return False
    return _rects_overlap(footprint(a), footprint(b), clearance)


def access_strips_free(placements, config) -> bool:
    for storage in placements:
        if storage.role != "storage":
            continue
        front = _front_strip(storage, config.min_robot_clearance_m + config.front_strip_extra_m)
        for other in placements:
            if other is storage or other.role == "clutter":
                continue
            if _rects_overlap(front, footprint(other), 0):
                return False
    return True


def _point_near_rect(px: float, py: float, rect, margin: float) -> bool:
    (x0, y0), (x1, y1) = rect
    if x0 < px < x1 and y0 < py < y1:
        return True  # zero margin still forbids an occupied spawn
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return math_hypot(dx, dy) < margin


def math_hypot(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5
