"""§8.6 integration on the real simulator.

The §39-ideal path is instance segmentation + depth unprojection. The
current Isaac Sim 5.1 host build segfaults in the syntheticdata graph when
those annotators are enabled (reproduced on idle GPUs), so production runs
rgb-only and resolves points through the PhysX raytest fallback
(backend.pixel_ray_hit, §8.7 approximation recorded per frame). The tests
below verify the ACTIVE bridge:

    - segmentation available -> pixel -> entity identity + <=1 cm unprojection
    - rgb-only -> projected entity pose -> raycast pixel -> same entity, and
      the hit position lies on that entity's surface (AABB containment)
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sim


def _frame(backend):
    frame = backend.capture_visual_frame()
    frame.frame_id = "frame_simtest"
    return frame


def _project(frame, world_point):
    """World point -> normalized (x, y), or None when behind the camera."""
    T_cam = np.linalg.inv(np.asarray(frame.camera_extrinsics))
    p = T_cam @ np.append(np.asarray(world_point, dtype=float), 1.0)
    if p[2] <= 0.05:
        return None
    K = np.asarray(frame.camera_intrinsics)
    u = K[0, 0] * p[0] / p[2] + K[0, 2]
    v = K[1, 1] * p[1] / p[2] + K[1, 2]
    if not (0 <= u <= frame.image_width - 1 and 0 <= v <= frame.image_height - 1):
        return None
    return u / (frame.image_width - 1), v / (frame.image_height - 1)


def _scenario_of(backend):
    return backend._scenario


def _point_aabb_distance(aabb, point) -> float:
    (lo, hi) = aabb
    d = [max(lo[i] - point[i], 0.0, point[i] - hi[i]) for i in range(3)]
    return float(np.linalg.norm(d))


def _unprojection_error(frame, world_point):
    T_cam = np.linalg.inv(np.asarray(frame.camera_extrinsics))
    p = T_cam @ np.append(world_point, 1.0)
    if p[2] <= 0.05:
        return None
    K = np.asarray(frame.camera_intrinsics)
    u = int(round(K[0, 0] * p[0] / p[2] + K[0, 2]))
    v = int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))
    if not (0 <= u < frame.image_width and 0 <= v < frame.image_height):
        return None
    d = float(np.asarray(frame.depth)[v, u])
    if not np.isfinite(d) or d <= 0:
        return None
    from rummagebench.perception.camera_geometry import (
        to_world, unproject_range, unproject_z_depth,
    )

    pc = (unproject_z_depth(u, v, d, K)
          if frame.depth_convention == "z_depth"
          else unproject_range(u, v, d, K))
    return float(np.linalg.norm(to_world(pc, frame.camera_extrinsics) - world_point))


def _at_anchor(backend, scenario, anchor_name):
    from rummagebench.core.scenario import AnchorSpec

    a = scenario.anchors[anchor_name]
    backend.teleport_robot(
        AnchorSpec(position=list(a.position), orientation=list(a.orientation))
    )
    backend.settle(5)
    return _frame(backend)


def test_visual_bridge_roundtrip_under_1cm(bridge_env):
    backend = bridge_env
    frame = _frame(backend)
    seg = np.asarray(frame.instance_segmentation)

    if seg.any():
        # §8.6 ideal path: depth-based unprojection error <= 1 cm
        for entity in backend.visible_entities():
            info = backend.describe_entity(entity)
            if info is None or info.fixed_base:
                continue
            pose = backend.entity_pose(entity)
            err = _unprojection_error(frame, np.asarray(pose, dtype=float))
            if err is not None:
                assert err <= 0.01, f"roundtrip error {err:.4f} m > 1 cm"
                return
        pytest.skip("no rigid entity projectable in the current view")

    # rgb-only host: raycast identity + surface geometry. Occluded entities
    # legitimately resolve to their occluder — keep scanning until one clean
    # roundtrip (entity identity + surface containment) is found.
    for entity in backend.visible_entities():
        info = backend.describe_entity(entity)
        if info is None or info.fixed_base:
            continue
        aabb = backend.entity_aabb(entity)
        if aabb is None:
            continue
        xy = _project(frame, backend.entity_pose(entity))
        if xy is None:
            continue
        entity_hit, hit_pos = backend.pixel_ray_hit(frame, xy[0], xy[1])
        if entity_hit != entity or hit_pos is None:
            continue  # occluded or background: try the next entity
        dist = _point_aabb_distance(aabb, hit_pos)
        assert dist <= 0.01, f"raycast hit {dist:.3f} m from {entity}'s AABB"
        return  # one clean hit is a full roundtrip
    pytest.skip("no unoccluded raycast roundtrip available in the current view")


def test_instance_resolution_to_owning_entity(bridge_env):
    backend = bridge_env
    scenario = _scenario_of(backend)
    known = set(backend.entity_names())

    # rgb-only: rays cast through each anchor view must resolve to KNOWN
    # owning entities (mapping correctness). Exact per-projected-entity
    # matching is the roundtrip test's job — scene-history occlusion is
    # legitimate here, so ANY named hit counts.
    named = tried = 0
    for anchor_name in scenario.anchors:
        frame = _at_anchor(backend, scenario, anchor_name)
        seg = np.asarray(frame.instance_segmentation)
        if seg.any():
            continue  # segmentation hosts take the <=1 cm unprojection path
        H, W = frame.image_height, frame.image_width
        for fy in (0.3, 0.5, 0.7):
            for fx in (0.2, 0.35, 0.5, 0.65, 0.8):
                tried += 1
                entity, _pos = backend.pixel_ray_hit(frame, fx, fy)
                if entity is not None and entity in known:
                    named += 1
    assert tried > 0, "no anchor view available for ray probing"
    assert named > 0, f"none of {tried} rays resolved to a known owning entity"


@pytest.fixture
def bridge_env(sim_session):
    """The shared ORACLE session's backend hosts the visual bridge too (the
    bridge is session-mode-agnostic). A reset restores the compiled initial
    scene: earlier episodes in the same pytest process displace objects, and
    the bridge assertions need the deterministic build state."""
    sim_session.reset()
    return sim_session._backend
