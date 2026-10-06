"""Landmark search regions.

Pure 2-D geometry on oriented footprints: no simulator, no robot data.
"""
from __future__ import annotations

import math
from typing import Sequence

Rect = tuple[float, float, float, float, float]  # cx, cy, sx, sy, yaw


def corners(r: Rect) -> list[tuple[float, float]]:
    cx, cy, sx, sy, yaw = r
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2)):
        out.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return out


def _overlap(a: list, b: list) -> bool:
    """Separating-axis test for two convex quads."""
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = y1 - y2, x2 - x1
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def _point_seg(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def rect_distance(a: Rect, b: Rect) -> float:
    """Minimum distance between two oriented rectangles (0 if they overlap)."""
    ca, cb = corners(a), corners(b)
    if _overlap(ca, cb):
        return 0.0
    best = math.inf
    for P, Q in ((ca, cb), (cb, ca)):
        for p in P:
            for i in range(4):
                best = min(best, _point_seg(p, Q[i], Q[(i + 1) % 4]))
    return best


def footprint(position: Sequence[float], size: Sequence[float], yaw: float) -> Rect:
    return (float(position[0]), float(position[1]), float(size[0]), float(size[1]), float(yaw))
