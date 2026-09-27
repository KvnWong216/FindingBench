"""Real IK tests (§21.1): URDF-backed numerical IK on a planar arm fixture.

reachable target -> success; unreachable -> NO_IK_SOLUTION;
joint-limit-blocked -> JOINT_LIMIT; determinism -> identical results.
"""

from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from rummagebench.feasibility.ik_solver import IKFailureReason, Pose
from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics

FIXTURES = Path(__file__).parent / "fixtures" / "robots"


@pytest.fixture
def short_arm():
    return PinocchioKinematics(
        urdf_path=str(FIXTURES / "short_arm.urdf"),
        base_link="base_link",
        eef_link="gripper",
        pos_tol=0.005,
        rot_tol=0.05,
    )


def _target(x: float, y: float, z: float, direction=(0.0, 0.0, 1.0)) -> Pose:
    from rummagebench.feasibility.ik_solver import look_at_quaternion

    return Pose(np.array([x, y, z]), look_at_quaternion(np.array(direction)))


def test_reachable_target_solves(short_arm):
    # (0.4, 0.3) from the shoulder at (0, 0, 0.3): |d| = 0.5 <= 0.55 reach
    result = short_arm.solve_ik(_target(0.4, 0.0, 0.6), seed_q=None)
    assert result.success, result
    assert result.q is not None
    assert np.all(result.q >= short_arm.lower) and np.all(result.q <= short_arm.upper)
    pose = short_arm.eef_pose(result.q)
    assert np.linalg.norm(pose.position - np.array([0.4, 0.0, 0.6])) < 0.01


def test_unreachable_target_fails_with_no_ik_solution(short_arm):
    # 1.2 m from the shoulder; max reach is 0.55 m
    result = short_arm.solve_ik(_target(0.0, 0.0, 1.5), seed_q=None)
    assert not result.success
    assert result.reason == IKFailureReason.NO_IK_SOLUTION
    assert result.q is None


def test_joint_limit_target_reports_joint_limit(short_arm):
    # 0.06 m from the shoulder: solvable only with |elbow| > 2.9 rad limit
    result = short_arm.solve_ik(_target(0.0, 0.0, 0.36), seed_q=None)
    assert not result.success
    assert result.reason == IKFailureReason.JOINT_LIMIT


def test_ik_is_deterministic(short_arm):
    a = short_arm.solve_ik(_target(0.4, 0.0, 0.6))
    b = short_arm.solve_ik(_target(0.4, 0.0, 0.6))
    assert a.success and b.success
    assert np.allclose(a.q, b.q)


def test_seed_q_is_respected(short_arm):
    seed = short_arm.q_seed_neutral()
    a = short_arm.solve_ik(_target(0.3, 0.0, 0.7), seed_q=seed)
    b = short_arm.solve_ik(_target(0.3, 0.0, 0.7), seed_q=seed)
    assert a.success and b.success
    assert np.allclose(a.q, b.q)


def test_forward_kinematics_identity(short_arm):
    poses = short_arm.forward_kinematics(short_arm.q_seed_neutral())
    assert "gripper" in poses and "link1" in poses
    # neutral pose ~ straight up: gripper above the base
    assert poses["gripper"].position[2] > 0.7
    assert abs(poses["gripper"].position[0]) < 1e-6


def test_link_classes_for_acm(short_arm):
    classes = short_arm.link_classes
    assert classes["gripper"] == "gripper"
    assert classes["link1"] == "arm"
    assert classes["link2"] == "arm"
