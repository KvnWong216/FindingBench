"""MOVE: signed relative base translation (protocol §12.1, §12.3).

Deterministic, controller-free base motion along the CURRENT heading.
Before execution the swept path is sampled at <= translation_sample_cm and
every sample's base footprint is checked against world collision geometry:
any collision rejects the whole motion (UNSAFE) with zero partial execution.
"""

from __future__ import annotations

import math

import numpy as np

from rummagebench.core.scenario import AnchorSpec


def yaw_from_quat(q: list[float]) -> float:
    """Z-Y-X intrinsic yaw for a quaternion [x, y, z, w]."""
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def footprint_corners(x: float, y: float, yaw: float, half_extent: float) -> list[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in (
            (-half_extent, -half_extent), (half_extent, -half_extent),
            (half_extent, half_extent), (-half_extent, half_extent),
        )
    ]


def _aabb_overlaps_footprint(aabb, corners, margin: float) -> bool:
    (lo, hi) = aabb
    # obstacle must matter at base height (z overlap with [0, 0.5])
    if lo[2] > 0.5 or hi[2] < 0.05:
        return False
    # Keep the actual rotated base rectangle. Its enclosing world AABB
    # contains empty corners and is only a broad-phase bound.
    obstacle = [
        (lo[0] - margin, lo[1] - margin),
        (hi[0] + margin, lo[1] - margin),
        (hi[0] + margin, hi[1] + margin),
        (lo[0] - margin, hi[1] + margin),
    ]
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for i in (0, 1):
        a, b = corners[i], corners[i + 1]
        axes.append((-(b[1] - a[1]), b[0] - a[0]))
    for ax, ay in axes:
        base_projection = [x * ax + y * ay for x, y in corners]
        obstacle_projection = [x * ax + y * ay for x, y in obstacle]
        # Touching remains a collision, including the full safety margin.
        if (max(base_projection) < min(obstacle_projection) - 1e-12
                or max(obstacle_projection) < min(base_projection) - 1e-12):
            return False
    return True


def base_pose_collision_free(backend, x: float, y: float, yaw: float,
                             half_extent: float, margin: float,
                             skip_entities: set[str] = frozenset()) -> bool:
    """Conservative footprint check against world obstacle AABBs."""
    corners = footprint_corners(x, y, yaw, half_extent)
    controlled = getattr(backend, "robot_entity_names", lambda: set())()
    for name in backend.entity_names():
        if name in skip_entities or name in controlled:
            continue
        info = backend.describe_entity(name)
        if info is None:
            continue
        # Movable furniture still obstructs a teleporting base. Only bodies
        # carried by this robot are excluded from the external obstacle set.
        if getattr(backend, "is_holding", lambda _: False)(name):
            continue
        aabb = backend.entity_aabb(name)
        if aabb is None:
            continue
        if _aabb_overlaps_footprint(aabb, corners, margin):
            return False
    return True


def move_target_pose(robot_pose, distance_m: float) -> tuple[list[float], list[float]]:
    """Final pose after signed translation along the current heading."""
    (x, y, z), q = robot_pose
    yaw = yaw_from_quat(q)
    nx = x + distance_m * math.cos(yaw)
    ny = y + distance_m * math.sin(yaw)
    return [nx, ny, z], list(q)


def turn_target_pose(robot_pose, angle_deg: float) -> tuple[list[float], list[float]]:
    (x, y, z), q = robot_pose
    yaw = yaw_from_quat(q) + math.radians(angle_deg)
    return [x, y, z], quat_from_yaw(yaw)


def execute_move(backend, robot_pose, distance_m: float, half_extent: float,
                 sample_m: float, margin: float) -> tuple[bool, list[float] | None]:
    """Swept-collision MOVE. Returns (safe, final_pose_or_None)."""
    (x0, y0, z0), q = robot_pose
    yaw = yaw_from_quat(q)
    steps = max(1, int(abs(distance_m) / max(sample_m, 1e-6)) + 1)
    for i in range(1, steps + 1):
        f = i / steps
        xi = x0 + distance_m * math.cos(yaw) * f
        yi = y0 + distance_m * math.sin(yaw) * f
        if not base_pose_collision_free(backend, xi, yi, yaw, half_extent, margin):
            return False, None
    target, tq = move_target_pose(robot_pose, distance_m)
    backend.teleport_robot(AnchorSpec(position=target, orientation=tq))
    return True, target


def execute_turn(backend, robot_pose, angle_deg: float, half_extent: float,
                 sample_deg: float, margin: float) -> tuple[bool, list[float] | None]:
    """Swept-collision TURN (position unchanged, yaw interpolated)."""
    (x, y, z), q = robot_pose
    yaw0 = yaw_from_quat(q)
    steps = max(1, int(abs(angle_deg) / max(sample_deg, 1e-6)) + 1)
    for i in range(1, steps + 1):
        yi = yaw0 + math.radians(angle_deg) * i / steps
        if not base_pose_collision_free(backend, x, y, yi, half_extent, margin):
            return False, None
    target, tq = turn_target_pose(robot_pose, angle_deg)
    backend.teleport_robot(AnchorSpec(position=target, orientation=tq))
    return True, target
