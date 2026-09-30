"""§39 relative base motion: signed conventions, swept collision, no partial."""

import numpy as np
import pytest

from rummagebench.skills.move import (
    execute_move,
    move_target_pose,
    turn_target_pose,
    yaw_from_quat,
)


def test_move_forward_positive():
    pose = ([1.0, 2.0, 0.05], [0.0, 0.0, 0.0, 1.0])  # yaw = 0 -> +x forward
    (x, y, _), _ = move_target_pose(pose, 0.4)
    assert abs(x - 1.4) < 1e-9 and abs(y - 2.0) < 1e-9


def test_move_backward_negative():
    pose = ([1.0, 2.0, 0.05], [0.0, 0.0, 0.0, 1.0])
    (x, y, _), _ = move_target_pose(pose, -0.4)
    assert abs(x - 0.6) < 1e-9


def test_move_follows_current_heading():
    yaw = np.pi / 2  # facing +y
    pose = ([1.0, 2.0, 0.05], [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    (x, y, _), _ = move_target_pose(pose, 0.5)
    assert abs(x - 1.0) < 1e-9 and abs(y - 2.5) < 1e-9


def test_turn_left_positive_ccw():
    pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    _, q = turn_target_pose(pose, 90.0)
    assert abs(yaw_from_quat(q) - np.pi / 2) < 1e-9
    (x, y, _), _ = turn_target_pose(pose, 90.0)
    assert (x, y) == (0.0, 0.0)  # position unchanged


def test_turn_right_negative_cw():
    pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    _, q = turn_target_pose(pose, -45.0)
    assert abs(yaw_from_quat(q) + np.pi / 4) < 1e-9


def test_blocked_move_is_unsafe_with_zero_partial_motion(visual_session, visual_backend):
    # wall right in front of the spawn (mini.yaml living_room = origin)
    from rummagebench.sim.base import EntityInfo

    visual_backend.entities["wall"] = EntityInfo(name="wall", category="wall", fixed_base=True)
    visual_backend.aabbs["wall"] = ([0.45, -0.4, 0.0], [0.75, 0.4, 0.9])
    visual_backend.poses["wall"] = [0.6, 0.0, 0.4]
    r = visual_session.step({"skill": "MOVE", "distance_cm": 40.0})
    assert r.feedback.code == "UNSAFE"
    x, y, _ = visual_backend.robot_pose()[0]
    assert abs(x) < 1e-6 and abs(y) < 1e-6  # spawn pose intact


def test_blocked_turn_is_unsafe_with_zero_partial_rotation(visual_session, visual_backend):
    # wall on the left: clear of the yaw-0 footprint, hit during the sweep
    from rummagebench.sim.base import EntityInfo

    visual_backend.entities["wall"] = EntityInfo(name="wall", category="wall", fixed_base=True)
    visual_backend.aabbs["wall"] = ([-0.1, 0.34, 0.0], [0.2, 0.6, 0.9])
    visual_backend.poses["wall"] = [0.05, 0.47, 0.4]
    r = visual_session.step({"skill": "TURN", "angle_deg": 90.0})
    assert r.feedback.code == "UNSAFE"
    assert yaw_from_quat(visual_backend.robot_pose()[1]) == 0.0


def test_free_move_executes_and_advances(visual_session, visual_backend):
    r = visual_session.step({"skill": "MOVE", "distance_cm": 40.0})
    assert r.feedback.code == "EXECUTED"
    x, y, _ = visual_backend.robot_pose()[0]
    assert abs(x - 0.4) < 1e-6 and abs(y) < 1e-6  # +0.4 m along +x (spawn yaw 0)


def test_out_of_range_move_is_invalid_not_clamped(visual_session):
    r = visual_session.step({"skill": "MOVE", "distance_cm": 500.0})
    assert r.feedback.code == "INVALID_ACTION"
    r = visual_session.step({"skill": "MOVE", "distance_cm": 0.0})
    assert r.feedback.code == "INVALID_ACTION"

def test_rotated_base_empty_aabb_corner_is_not_collision():
    from rummagebench.skills.move import footprint_corners, _aabb_overlaps_footprint
    corners = footprint_corners(0, 0, np.pi / 4, .4)
    # Above the diamond's sloping edge, inside its broad-phase AABB.
    assert not _aabb_overlaps_footprint(([.44,.44,0],[.5,.5,.4]), corners, .03)

def test_rotated_base_true_overlap_and_margin_still_collide():
    from rummagebench.skills.move import footprint_corners, _aabb_overlaps_footprint
    corners = footprint_corners(0, 0, np.pi / 4, .4)
    assert _aabb_overlaps_footprint(([.2,.2,0],[.3,.3,.4]), corners, .03)
    assert _aabb_overlaps_footprint(([.58,-.01,0],[.60,.01,.4]), corners, .03)
    assert not _aabb_overlaps_footprint(([.61,-.01,0],[.63,.01,.4]), corners, .03)

def test_reported_breakfast_table_false_positive():
    from rummagebench.skills.move import _aabb_overlaps_footprint
    corners = [(4.0669419634,-6.4685886459),(3.3201895698,-6.1816028227),
               (3.0332037466,-6.9283552163),(3.7799561402,-7.2153410395)]
    obstacle = ([2.3257446289,-7.70688199997,0],[3.04167461395,-7.17469644546,.708])
    assert not _aabb_overlaps_footprint(obstacle, corners, .03)


# ---------------------------------------------------------------------------
# Static reproduction of the audited pillar rejection (raw_agent_resume_1
# collision_audit_12.json): MOVE 50 cm along yaw~180° from [3.0102,-6.5167]
# is blocked by the square column walls_tjfjwe_0. The pillar is visible in
# the public frame_000007 RGB; these tests pin the gate's exact behavior.
# ---------------------------------------------------------------------------

_PILLAR_POSE = ([3.01020884513855, -6.516660690307617, 0.005284354090690613],
                [-7.766857743263245e-05, -7.723308954155073e-06,
                 0.9999998211860657, 0.0007758901920169592])
_PILLAR_AABB = ([2.038912057876587, -7.061361312866211, -2.384185791015625e-07],
                [2.2485122680664062, -6.8517608642578125, 2.3999998569488525])


class _PillarBackend:
    """Single-obstacle backend double pinned to the audited pillar AABB."""

    def __init__(self):
        self.teleported = None

    def entity_names(self):
        return ["walls_tjfjwe_0"]

    def robot_entity_names(self):
        return {"robot_0"}

    def describe_entity(self, name):
        return object()

    def is_holding(self, name):
        return False

    def entity_aabb(self, name):
        return _PILLAR_AABB

    def teleport_robot(self, anchor):
        self.teleported = anchor


def _first_blocked_distance(pose, distance_m, margin):
    from rummagebench.skills.move import base_pose_collision_free, yaw_from_quat
    (x0, y0, _), q = pose
    yaw = yaw_from_quat(q)
    steps = 11  # 50 cm at sample_m=0.05
    for i in range(1, steps + 1):
        f = i / steps
        if not base_pose_collision_free(_PillarBackend(), x0 + distance_m * f * np.cos(yaw),
                                        y0 + distance_m * f * np.sin(yaw), yaw,
                                        0.40, margin):
            return distance_m * f
    return None


def test_audited_pillar_blocks_move50_at_audited_sample():
    backend = _PillarBackend()
    safe, final = execute_move(backend, _PILLAR_POSE, 0.5, half_extent=0.40,
                               sample_m=0.05, margin=0.03)
    assert safe is False and final is None
    assert backend.teleported is None  # zero partial execution
    blocked = _first_blocked_distance(_PILLAR_POSE, 0.5, 0.03)
    assert blocked == pytest.approx(0.36363636363636365)  # audit_12 first_blocks[0]


def test_audited_pillar_blocks_even_with_zero_margin():
    # The front footprint corner penetrates the pillar AABB by ~1.3 mm, so
    # the rejection is real geometry, not an artifact of the 3 cm margin.
    blocked = _first_blocked_distance(_PILLAR_POSE, 0.5, 0.0)
    assert blocked == pytest.approx(0.36363636363636365)


def test_lateral_shift_clears_the_pillar():
    # Shifting the base center 11 cm toward +y clears the audited column
    # with the full 3 cm margin: a legal approach past the pillar exists.
    (x, y, z), q = _PILLAR_POSE
    shifted = ([x, y + 0.11, z], q)
    backend = _PillarBackend()
    safe, final = execute_move(backend, shifted, 0.5, half_extent=0.40,
                               sample_m=0.05, margin=0.03)
    assert safe is True
    # yaw is 179.911°: cos gives ~-0.5 m in x, sin drifts ~0.8 mm in y
    assert final[0] == pytest.approx(x - 0.5, abs=1e-5)
    assert final[1] == pytest.approx(y + 0.11, abs=2e-3)
    assert backend.teleported is not None
