"""OBSERVE: object-centric active perception (protocol §13).

The clicked point is resolved through the same visual bridge (no semantic
label supplied by the agent). Deterministic candidate viewpoints are sampled
on an azimuth ring around the object's AABB center at a clamped radius;
viewpoints are accepted only when base-collision-free and the object is
actually visible (private instance segmentation pixel count). The robot's
exact base pose is restored afterwards; OBSERVE never permanently changes
robot pose, object states or held-object state, and consumes exactly ONE
planning step.
"""

from __future__ import annotations

import math

from rummagebench.core.scenario import AnchorSpec
from rummagebench.skills.move import base_pose_collision_free, quat_from_yaw


class ObserveConfig:
    def __init__(
        self,
        num_views: int = 8,
        min_radius_m: float = 0.8,
        max_radius_m: float = 1.5,
        surface_margin_m: float = 0.4,
        min_visible_pixels: int = 100,
        restore_robot_pose: bool = True,
    ):
        if num_views not in (8, 16):
            raise ValueError("observe.num_views must be 8 or 16")
        self.num_views = num_views
        self.min_radius_m = min_radius_m
        self.max_radius_m = max_radius_m
        self.surface_margin_m = surface_margin_m
        self.min_visible_pixels = min_visible_pixels
        self.restore_robot_pose = restore_robot_pose


def object_aabb_xy(backend, entity: str) -> tuple[tuple[float, float], float]:
    """((cx, cy), xy_radius): center and half-diagonal of the AABB's XY box."""
    aabb = backend.entity_aabb(entity)
    if aabb is None:
        pose = backend.entity_pose(entity)
        return (pose[0], pose[1]), 0.15
    (lo, hi) = aabb
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    radius = 0.5 * math.hypot(hi[0] - lo[0], hi[1] - lo[1])
    return (cx, cy), radius


def candidate_viewpoints(center_xy, xy_radius: float, cfg: ObserveConfig):
    """Deterministic azimuth ring facing the object center (§13.2)."""
    obs_radius = min(max(xy_radius + cfg.surface_margin_m, cfg.min_radius_m),
                     cfg.max_radius_m)
    out = []
    for i in range(cfg.num_views):
        az = 2.0 * math.pi * i / cfg.num_views
        x = center_xy[0] + obs_radius * math.cos(az)
        y = center_xy[1] + obs_radius * math.sin(az)
        yaw = math.atan2(center_xy[1] - y, center_xy[0] - x)
        out.append((x, y, yaw))
    return out


def run_observe(backend, entity: str, cfg: ObserveConfig,
                base_half_extent: float, collision_margin: float,
                render_private_frame, visible_pixel_count,
                settle=None, log=None) -> list[dict]:
    """Sample, filter, render and RESTORE. `render_private_frame()` and
    `visible_pixel_count(frame, entity)` are injected by the session so this
    module stays simulator-agnostic. Returns accepted view records
    (dicts with private diagnostics + the RGB payload for the caller)."""
    original_pose = backend.robot_pose()
    center_xy, xy_radius = object_aabb_xy(backend, entity)
    accepted: list[dict] = []
    for (x, y, yaw) in candidate_viewpoints(center_xy, xy_radius, cfg):
        record = {"base_pose": [x, y, yaw], "accepted": False}
        try:
            if not base_pose_collision_free(
                backend, x, y, yaw, base_half_extent, collision_margin,
                skip_entities={entity},
            ):
                record["reason"] = "BASE_COLLISION"
            else:
                backend.teleport_robot(
                    AnchorSpec(position=[x, y, original_pose[0][2]],
                               orientation=quat_from_yaw(yaw))
                )
                if settle is not None:
                    settle(3)
                frame = render_private_frame()
                pixels = visible_pixel_count(frame, entity)
                record["visible_pixels"] = pixels
                if pixels >= cfg.min_visible_pixels:
                    record["accepted"] = True
                    record["rgb"] = frame.rgb
                    accepted.append(record)
                else:
                    record["reason"] = "NOT_VISIBLE_ENOUGH"
        except Exception as e:  # one bad viewpoint never kills OBSERVE
            record["reason"] = f"RENDER_ERROR:{type(e).__name__}"
        if log is not None:
            log(record)
    if cfg.restore_robot_pose:
        backend.teleport_robot(
            AnchorSpec(position=list(original_pose[0]),
                       orientation=list(original_pose[1]))
        )
        if settle is not None:
            settle(3)
    return accepted
