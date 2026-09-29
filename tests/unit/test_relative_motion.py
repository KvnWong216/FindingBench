"""§39 relative base motion: signed conventions, swept collision, no partial."""

import numpy as np

from rummagebench.skills.move import (
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
