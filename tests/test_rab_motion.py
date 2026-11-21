"""Unit tests for scripts/rab_motion.py (pure numpy; no simulator)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rab_motion as M  # noqa: E402


def test_min_jerk_boundaries_and_monotonic():
    s = np.linspace(0, 1, 101)
    v = M.min_jerk(s)
    assert v[0] == 0 and abs(v[-1] - 1) < 1e-12 and np.all(np.diff(v) >= 0)
    # zero velocity at both ends
    assert M.min_jerk(1e-4) < 1e-9 and 1 - M.min_jerk(1 - 1e-4) < 1e-9


def test_base_frames_follow_astar_waypoints():
    # start facing +x, go to (1,0), then (1,1), end facing -x
    poses = [(0, 0, 0, 0.0), (1, 0, 0, 0.0), (1, 1, 0, math.pi / 2), (1, 1, 0, math.pi)]
    fr = M.base_frames(poses, fps=20)
    assert len(fr) > 10
    end = fr[-1]
    assert abs(end[0] - 1) < 1e-9 and abs(end[1] - 1) < 1e-9 and abs(M.wrap_angle(end[3] - math.pi)) < 1e-9
    # every waypoint is actually visited
    for wp in poses[1:3]:
        assert min(math.hypot(f[0] - wp[0], f[1] - wp[1]) for f in fr) < 1e-9
    # never moves faster than the limit (+ discretization slack)
    lim = M.BaseLimits()
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) * 20 for a, b in zip(fr, fr[1:])]
    assert max(steps) <= lim.lin_vmax * 1.05


def test_base_frames_pure_rotation_takes_short_way():
    fr = M.base_frames([(0, 0, 0, math.radians(170)), (0, 0, 0, math.radians(-170))], fps=20)
    yaws = [M.wrap_angle(f[3]) for f in fr]
    # passes through 180 deg, not through 0
    assert all(abs(abs(y) - math.pi) < math.radians(11) for y in yaws)


def test_retime_path_ends_on_ik_solution_and_is_smooth():
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.normal(0, .05, (30, 9)), axis=0)
    out = M.retime_path(path, fps=20, vmax=.9, tmin=.5)
    assert np.allclose(out[-1], path[-1])
    assert len(out) >= M.frame_count(.5, 20)
    # all frames lie on the polyline's bounding box (pure interpolation)
    assert np.all(out >= path.min(0) - 1e-9) and np.all(out <= path.max(0) + 1e-9)


def test_retime_weights_ignore_masked_dofs():
    path = np.array([[0, 0], [0, 5.0]])  # only DOF 1 moves
    out = M.retime_path(path, fps=20, weights=[1, 0])  # masked -> zero length
    assert len(out) == 1 and np.allclose(out[-1], path[-1])


def test_rigid_transform_roundtrip():
    q = np.array([0.1, -0.2, 0.3, 0.9])
    q /= np.linalg.norm(q)
    a = np.r_[1, 2, 3, q]
    b = np.r_[-.5, .1, .7, 0, 0, math.sin(.3), math.cos(.3)]
    rel = M.relative(a, b)
    back = M.compose(a, rel)
    assert np.allclose(back[:3], b[:3]) and abs(abs(np.dot(back[3:], b[3:])) - 1) < 1e-9


def test_recorder_png_fallback(tmp_path):
    rec = M.DemoRecorder(tmp_path / "demo", fps=10, cameras=("chase",), dof_names=["a", "b"])
    rec._png = True  # force fallback regardless of imageio availability
    rec.begin(1, "MOVE_TO", {"skill": "MOVE_TO"}, "EXECUTED")
    img = np.zeros((8, 8, 3), np.uint8)
    for _ in range(3):
        rec.add({"chase": img}, dict(phase="base", q=np.zeros(2)))
    rec.end(animated=True)
    rec.close()
    rows = [json.loads(l) for l in (tmp_path / "demo/trajectory.jsonl").read_text().splitlines()]
    seg = json.loads((tmp_path / "demo/segments.jsonl").read_text())
    assert len(rows) == 3 and rows[0]["skill"] == "MOVE_TO" and seg["frames"] == 3
    assert len(list((tmp_path / "demo/frames/chase").glob("*.png"))) == 3


def test_rot_error_vec_exact_on_known_rotations():
    from scipy.spatial.transform import Rotation
    r0 = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    for axis, angle_rad in [([1, 0, 0], 0.5), ([0, 1, 0], 1e-7), ([0, 0, 1], math.pi - 1e-6), ([1, 2, -1], 1.2)]:
        u = np.asarray(axis, float) / np.linalg.norm(axis)
        r1 = Rotation.from_rotvec(u * angle_rad).as_matrix() @ r0
        err = M.rot_error_vec(r1, r0)
        assert abs(np.linalg.norm(err) - angle_rad) < 1e-4
        # Applying exp([err]_x) @ r0 recovers r1
        rec = Rotation.from_rotvec(err).as_matrix() @ r0
        assert np.allclose(rec, r1, atol=1e-4)


def test_gripper_frame_orthonormal_and_avoids_180_flip():
    R1 = M.gripper_frame([0, 0, -1], [1, 0, 0])
    assert abs(np.linalg.det(R1) - 1.0) < 1e-9
    assert np.allclose(R1.T @ R1, np.eye(3), atol=1e-9)
    assert np.allclose(R1[:, 2], [0, 0, -1], atol=1e-9)  # local +Z is approach
    # Flip pinch by 180 deg around Z; passing R1 as cur_R must keep +X branch (no flip)
    R_keep = M.gripper_frame([0, 0, -1], [-1, 0, 0], cur_R=R1)
    assert np.allclose(R_keep, R1, atol=1e-9)
    # Passing flipped cur_R must pick the flipped branch
    R_flipped = np.column_stack((-R1[:, 0], -R1[:, 1], R1[:, 2]))
    assert np.allclose(M.gripper_frame([0, 0, -1], [1, 0, 0], cur_R=R_flipped), R_flipped, atol=1e-9)


def test_plan_grasp_top_down_drawer_and_flat_items():
    # Elongated remote (4 cm wide along X, 16 cm long along Y, 2 cm tall) inside a drawer
    box = [[0.98, -0.08, 0.50], [1.02, 0.08, 0.52]]
    gp = M.plan_grasp_pose(box, [1.0, 0.0, 0.51, 0, 0, 0, 1], [0.4, 0.0, 0.0], 0.0, in_open_drawer=True)
    assert gp.mode == "top_down"
    assert np.allclose(gp.rotmat[:, 2], [0, 0, -1], atol=1e-6)  # straight vertical down
    assert abs(abs(gp.rotmat[0, 0]) - 1.0) < 1e-6  # pinches along narrow 4 cm X axis
    assert abs(gp.pinch_width_m - 0.04) < 1e-4
    assert gp.pre_pos[2] > gp.grasp_pos[2] + 0.12
    assert gp.q_open > gp.q_close >= M.GRIP_JOINT_MIN


def test_plan_grasp_side_tall_bottle_and_calibrated_jaw():
    # Tall soda can (6.6 cm diameter, 14 cm tall) on open counter in front of robot
    box = [[0.967, -0.033, 0.75], [1.033, 0.033, 0.89]]
    gp = M.plan_grasp_pose(box, [1.0, 0.0, 0.82, 0, 0, 0, 1], [0.4, 0.0, 0.0], 0.0, in_open_drawer=False)
    assert gp.mode == "side"
    assert gp.rotmat[0, 2] > 0.9  # approaches forward along +X
    assert gp.rotmat[2, 2] < -0.1  # slight downward pitch
    assert abs(abs(gp.rotmat[1, 0]) - 1.0) < 1e-6  # fingers pinch horizontally across ±Y
    assert gp.pre_pos[0] < gp.grasp_pos[0] - 0.08  # pre-grasp backed off toward robot
    assert gp.lift_pos[2] > gp.grasp_pos[2] + 0.08  # vertical lift
    # Adaptive jaw width monotonic: thin marker < soda can < wide box
    q_thin = M.gripper_close_angle(0.02)
    q_can = M.gripper_close_angle(0.066)
    q_wide = M.gripper_close_angle(0.11)
    assert M.GRIP_JOINT_MIN <= q_thin < q_can < q_wide <= M.GRIP_JOINT_MAX_CLOSE


def test_plan_place_pose_vertical_descent_and_clearance():
    box = [[0.0, 0.0, 0.0], [0.07, 0.07, 0.14]]
    pp = M.plan_place_pose(box, [1.0, 0.0, 0.75], [0.4, 0.0, 0.0], 0.0)
    assert pp.mode == "side"
    assert abs(pp.pre_pos[0] - pp.place_pos[0]) < 1e-9  # straight vertical descent
    assert pp.pre_pos[2] > pp.place_pos[2] + 0.08
    assert pp.lift_pos[2] > pp.place_pos[2] and pp.lift_pos[0] < pp.place_pos[0]
    assert pp.q_open > pp.q_close

