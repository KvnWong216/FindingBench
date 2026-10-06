"""Agent-protocol navigation witness: start pose -> interaction pose using
ONLY the public MOVE / TURN actions (no NAV).

The footprint test is a vectorised re-implementation of
skills/move.py::_aabb_overlaps_footprint over a frozen obstacle set (same
z band, same margin, touching = collision), and the sweep sampling mirrors
execute_move (<= 5 cm) / execute_turn (<= 5 deg). A planned route is only a
CANDIDATE: the certifier replays it through the real VisualProtocolSession,
whose own swept checks are authoritative.

Routes are polylines start -> [<= 2 waypoints] -> goal. Each leg is one TURN
(skipped when < min_turn_deg; the heading error is then simulated, not
ignored) plus ceil(len / max_move) equal MOVEs, driven forward or backward,
whichever needs the smaller turn. The route with the fewest actions wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Z_LOW, Z_HIGH = 0.05, 0.5  # obstacle must intersect the base height band


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class FootprintChecker:
    def __init__(self, aabbs: Sequence[tuple[Sequence[float], Sequence[float]]],
                 half_extent: float, margin: float):
        boxes = [(lo, hi) for lo, hi in aabbs
                 if not (lo[2] > Z_HIGH or hi[2] < Z_LOW)]
        self.half = float(half_extent)
        self.margin = float(margin)
        if boxes:
            lo = np.array([b[0][:2] for b in boxes], float) - margin
            hi = np.array([b[1][:2] for b in boxes], float) + margin
        else:
            lo = hi = np.zeros((0, 2))
        self.lo, self.hi = lo, hi
        # obstacle rectangle corners (N, 4, 2), same order as move.py
        self.corners = np.stack([np.stack([lo[:, 0], lo[:, 1]], 1),
                                 np.stack([hi[:, 0], lo[:, 1]], 1),
                                 np.stack([hi[:, 0], hi[:, 1]], 1),
                                 np.stack([lo[:, 0], hi[:, 1]], 1)], 1)

    @classmethod
    def from_backend(cls, backend, half_extent: float, margin: float,
                     skip: frozenset[str] = frozenset()) -> "FootprintChecker":
        """Obstacle set exactly as skills/move.py::base_pose_collision_free
        builds it (all non-robot, non-held entities with an AABB)."""
        controlled = backend.robot_entity_names()
        boxes = []
        for name in backend.entity_names():
            if name in skip or name in controlled:
                continue
            if backend.describe_entity(name) is None or backend.is_holding(name):
                continue
            aabb = backend.entity_aabb(name)
            if aabb is not None:
                boxes.append(aabb)
        return cls(boxes, half_extent, margin)

    def free(self, x: float, y: float, yaw: float) -> bool:
        return bool(self.free_many(np.array([x]), np.array([y]), np.array([yaw]))[0])

    def free_many(self, xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """Vectorised footprint test for P poses -> (P,) bool (same SAT as
        skills/move.py::_aabb_overlaps_footprint; touching = collision)."""
        xs, ys, yaws = (np.asarray(v, dtype=float).reshape(-1) for v in (xs, ys, yaws))
        if not len(self.lo):
            return np.ones(len(xs), bool)
        h = self.half
        offs = np.array([(-h, -h), (h, -h), (h, h), (-h, h)])            # (4, 2)
        c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        bx = xs[:, None] + c * offs[None, :, 0] - s * offs[None, :, 1]   # (P, 4)
        by = ys[:, None] + s * offs[None, :, 0] + c * offs[None, :, 1]
        eps = 1e-12
        lo, hi = self.lo[None], self.hi[None]                              # (1, N, 2)
        sep = (bx.max(1)[:, None] < lo[..., 0] - eps) | (hi[..., 0] < bx.min(1)[:, None] - eps)
        sep |= (by.max(1)[:, None] < lo[..., 1] - eps) | (hi[..., 1] < by.min(1)[:, None] - eps)
        for i in (0, 1):
            ax = -(by[:, i + 1] - by[:, i])                                # (P,)
            ay = bx[:, i + 1] - bx[:, i]
            bp = bx * ax[:, None] + by * ay[:, None]                       # (P, 4)
            op = (self.corners[None, :, :, 0] * ax[:, None, None]
                  + self.corners[None, :, :, 1] * ay[:, None, None])       # (P, N, 4)
            sep |= (bp.max(1)[:, None] < op.min(2) - eps) | (op.max(2) < bp.min(1)[:, None] - eps)
        return sep.all(1)

    def disc_free(self, x: float, y: float) -> bool:
        """Sufficient condition for every heading at (x, y) being free: the
        footprint's circumscribed disc (+ margin) misses every obstacle."""
        if not len(self.lo):
            return True
        r = self.half * math.sqrt(2.0)
        dx = np.maximum(np.maximum(self.lo[:, 0] - x, 0.0), x - self.hi[:, 0])
        dy = np.maximum(np.maximum(self.lo[:, 1] - y, 0.0), y - self.hi[:, 1])
        return bool(np.all(np.hypot(dx, dy) > r + 1e-9))

    def move_free(self, x: float, y: float, yaw: float, d: float,
                  sample_m: float = 0.05) -> bool:
        steps = max(1, int(abs(d) / max(sample_m, 1e-6)) + 1)
        f = np.arange(1, steps + 1) / steps
        return bool(self.free_many(x + d * math.cos(yaw) * f, y + d * math.sin(yaw) * f,
                                   np.full(steps, yaw)).all())

    def turn_free(self, x: float, y: float, yaw: float, angle_deg: float,
                  sample_deg: float = 5.0) -> bool:
        if self.disc_free(x, y):
            return True
        steps = max(1, int(abs(angle_deg) / max(sample_deg, 1e-6)) + 1)
        f = np.arange(1, steps + 1) / steps
        return bool(self.free_many(np.full(steps, x), np.full(steps, y),
                                   yaw + math.radians(angle_deg) * f).all())


@dataclass
class Route:
    actions: list[dict]
    waypoints: list[tuple[float, float]]
    final: tuple[float, float, float]  # simulated x, y, yaw (rad)

    @property
    def steps(self) -> int:
        return len(self.actions)


@dataclass
class ProtocolLimits:
    min_move_m: float = 0.05
    max_move_m: float = 1.0
    min_turn_deg: float = 5.0
    max_turn_deg: float = 180.0


def _leg(chk: FootprintChecker, pose: list[float], q: tuple[float, float],
         lim: ProtocolLimits, actions: list[dict]) -> bool:
    x, y, yaw = pose
    dx, dy = q[0] - x, q[1] - y
    dist = math.hypot(dx, dy)
    if dist < lim.min_move_m:
        return True
    travel = math.degrees(math.atan2(dy, dx))
    fwd = wrap_deg(travel - math.degrees(yaw))
    bwd = wrap_deg(travel + 180.0 - math.degrees(yaw))
    turn, sign = (fwd, 1.0) if abs(fwd) <= abs(bwd) else (bwd, -1.0)
    if abs(turn) >= lim.min_turn_deg:
        if not chk.turn_free(x, y, yaw, turn):
            return False
        actions.append({"skill": "TURN", "angle_deg": round(turn, 2)})
        yaw = math.radians(math.degrees(yaw) + turn)
    # drive along the ACTUAL heading (a skipped sub-5-degree turn leaves a
    # small error that is simulated, never assumed away)
    n = max(1, math.ceil(dist / lim.max_move_m - 1e-9))
    step = sign * dist / n
    for _ in range(n):
        if not chk.move_free(x, y, yaw, step):
            return False
        actions.append({"skill": "MOVE", "distance_cm": round(step * 100.0, 1)})
        x, y = x + step * math.cos(yaw), y + step * math.sin(yaw)
    pose[:] = [x, y, yaw]
    return True


def route_through(chk: FootprintChecker, start: tuple[float, float, float],
                  points: Sequence[tuple[float, float]], goal_yaw: float,
                  lim: ProtocolLimits = ProtocolLimits()) -> Optional[Route]:
    pose = list(start)
    actions: list[dict] = []
    for q in points:
        if not _leg(chk, pose, q, lim, actions):
            return None
    turn = wrap_deg(math.degrees(goal_yaw - pose[2]))
    if abs(turn) >= lim.min_turn_deg:
        if not chk.turn_free(pose[0], pose[1], pose[2], turn):
            return None
        actions.append({"skill": "TURN", "angle_deg": round(turn, 2)})
        pose[2] = math.radians(math.degrees(pose[2]) + turn)
    return Route(actions, list(points[:-1]), (pose[0], pose[1], pose[2]))


def _turn_actions(a: float, lim: ProtocolLimits) -> list[dict]:
    """Public TURNs realising a heading change of ``a`` degrees in place
    (|a| < min_turn needs two turns: overshoot, then come back)."""
    a = wrap_deg(a)
    if abs(a) < 0.5:
        return []
    if abs(a) >= lim.min_turn_deg:
        return [{"skill": "TURN", "angle_deg": round(a, 2)}]
    over = math.copysign(lim.min_turn_deg * 2.0, a)
    return [{"skill": "TURN", "angle_deg": round(over, 2)},
            {"skill": "TURN", "angle_deg": round(a - over, 2)}]


def plan_route(chk: FootprintChecker, start: tuple[float, float, float],
               goal: tuple[float, float, float],
               waypoints: Sequence[tuple[float, float]],
               lim: ProtocolLimits = ProtocolLimits(),
               max_points: int = 200, max_edge_m: float = 4.0) -> Optional[Route]:
    """MOVE / TURN route start -> goal pose (Dijkstra on a visibility graph).

    The robot only turns where it can turn freely: at "pivot" points whose
    circumscribed footprint disc misses every obstacle (any rotation there
    is collision-free), and at the start pose with an exact swept check.
    Straight drives (forward or backward, <= max_move chunks) connect
    pivots; every drive is swept-checked. Tight interaction anchors are
    reached by driving straight in along their own heading from a pivot on
    the line behind them, so no turn is needed at the anchor itself. The
    returned actions are exact (turns happen only at pivots / the start, so
    there is no unmodelled drift); the action count is minimised over the
    graph (edge cost = turns + ceil(d / max_move)).
    """
    import heapq

    sx, sy, syaw = start
    gx, gy, gyaw = goal

    def detour(p):
        return math.hypot(p[0] - sx, p[1] - sy) + math.hypot(p[0] - gx, p[1] - gy)

    cand = sorted(waypoints, key=lambda p: (detour(p), p))[:max_points]
    for k in range(1, 17):  # pivots on the start line and the goal approach line
        d = 0.25 * k
        cand.append((gx - d * math.cos(gyaw), gy - d * math.sin(gyaw)))
        cand.append((sx + d * math.cos(syaw), sy + d * math.sin(syaw)))
        cand.append((sx - d * math.cos(syaw), sy - d * math.sin(syaw)))
    pivots = [p for p in dict.fromkeys(cand) if chk.disc_free(p[0], p[1])]
    goal_pivot = chk.disc_free(gx, gy)

    def n_moves(d):
        return max(1, math.ceil(d / lim.max_move_m - 1e-9))

    seg_cache: dict[tuple, bool] = {}

    def drive_ok(x0, y0, x1, y1):
        """Straight swept drive with the footprint along the travel line."""
        key = (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
        if key not in seg_cache:
            d = math.hypot(x1 - x0, y1 - y0)
            h = math.atan2(y1 - y0, x1 - x0)
            ok = True
            n = n_moves(d)
            for i in range(n):
                if not chk.move_free(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n,
                                     h, d / n):
                    ok = False
                    break
            seg_cache[key] = ok
        return seg_cache[key]

    def heading_change(h_from, travel):
        """Smallest turn (deg) to drive along ``travel`` forward or backward."""
        fwd = wrap_deg(travel - h_from)
        bwd = wrap_deg(travel + 180.0 - h_from)
        return fwd if abs(fwd) <= abs(bwd) else bwd

    # nodes: 0 = start, 1 = goal, 2.. = pivots
    nodes = [(sx, sy), (gx, gy)] + pivots
    INF = float("inf")
    dist = [INF] * len(nodes)
    head = [None] * len(nodes)  # heading on arrival (deg)
    prev: list = [None] * len(nodes)
    dist[0], head[0] = 0.0, math.degrees(syaw)
    heap = [(0.0, 0)]
    while heap:
        c, i = heapq.heappop(heap)
        if c > dist[i] or i == 1:
            continue
        x0, y0 = nodes[i]
        for j in range(1, len(nodes)):
            if j == i:
                continue
            x1, y1 = nodes[j]
            d = math.hypot(x1 - x0, y1 - y0)
            if d < lim.min_move_m or d > max_edge_m:
                continue
            travel = math.degrees(math.atan2(y1 - y0, x1 - x0))
            a = heading_change(head[i], travel)
            if abs(a) >= 0.5:
                if i == 0:  # the start is not necessarily a pivot: exact sweep
                    if abs(a) < lim.min_turn_deg or not chk.turn_free(
                            x0, y0, math.radians(head[i]), a):
                        continue
            turns = len(_turn_actions(a, lim))
            arrive = wrap_deg(head[i] + a)
            extra = 0
            if j == 1:  # final heading at the goal
                fa = wrap_deg(math.degrees(gyaw) - arrive)
                if abs(fa) >= 0.5:
                    if not (goal_pivot or (abs(fa) >= lim.min_turn_deg and chk.turn_free(
                            gx, gy, math.radians(arrive), fa))):
                        continue
                    extra = len(_turn_actions(fa, lim))
            if not drive_ok(x0, y0, x1, y1):
                continue
            nc = c + turns + n_moves(d) + extra
            if nc < dist[j]:
                dist[j], head[j], prev[j] = nc, arrive, i
                if j != 1:
                    heapq.heappush(heap, (nc, j))
    if dist[1] == INF:
        return None
    # rebuild exact actions along the node path
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    path.reverse()
    actions: list[dict] = []
    h = math.degrees(syaw)
    x, y = sx, sy
    for i, j in zip(path, path[1:]):
        (x0, y0), (x1, y1) = nodes[i], nodes[j]
        travel = math.degrees(math.atan2(y1 - y0, x1 - x0))
        a = heading_change(h, travel)
        actions += _turn_actions(a, lim)
        h = wrap_deg(h + a)
        sign = 1.0 if abs(wrap_deg(travel - h)) < 90.0 else -1.0
        d = math.hypot(x1 - x0, y1 - y0)
        n = n_moves(d)
        actions += [{"skill": "MOVE", "distance_cm": round(sign * d / n * 100.0, 1)}] * n
        x, y = x1, y1
    actions += _turn_actions(wrap_deg(math.degrees(gyaw) - h), lim)
    return Route(actions, [nodes[i] for i in path[1:-1]], (x, y, gyaw))
