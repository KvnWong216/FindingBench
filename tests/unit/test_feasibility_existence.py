"""Feasibility = existence of ONE collision-free configuration (README).

The engine must not stop at the first converging IK seed: a redundant arm
reaches the same interaction pose in several configurations, and only one of
them needs to be collision-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rummagebench.feasibility.collision import CollisionResult
from rummagebench.feasibility.ik_solver import IKResult
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import EntityInfo, ResolvedTarget
from rummagebench.core.types import TargetKind
from rummagebench.state.benchmark_state import BenchmarkWorldState
from rummagebench.validation.feasibility import FeasibilityValidator

from conftest import FakeBackend

FIXTURES = Path(__file__).parent / "fixtures"
Q_BAD, Q_GOOD = np.array([0.0, 0.0]), np.array([1.0, -1.0])


class TwoSolutionKinematics:
    nq = 2
    controlled_joint_names = ["a", "b"]

    def q_seed_neutral(self):
        return np.zeros(2)

    def solve_ik(self, target, seed_q=None):
        return IKResult(success=True, q=Q_BAD.copy(), position_error=0.0, orientation_error=0.0)

    def iter_ik_solutions(self, target, seed_q=None, max_solutions=8):
        sols = [Q_BAD, Q_GOOD][:max_solutions]
        for q in sols:
            yield IKResult(success=True, q=q.copy(), position_error=0.0, orientation_error=0.0)


class CollidesAtQBad:
    def refresh_world(self, backend):
        pass

    def check_configuration(self, q, ctx):
        bad = np.allclose(q, Q_BAD)
        return CollisionResult(collision_free=not bad, self_collision=False,
                               world_collision=bad)


def _check(max_solutions: int):
    backend = FakeBackend(
        entities={"cup": EntityInfo(name="cup", category="cup")}, anchors=set(),
        poses={"cup": [0.5, 0.0, 0.5]}, aabbs={"cup": ([0.45, -0.05, 0.45], [0.55, 0.05, 0.55])})
    v = FeasibilityValidator(backend, kinematics=TwoSolutionKinematics(),
                             config_collision=CollidesAtQBad())
    v.MAX_IK_SOLUTIONS = max_solutions
    resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity="cup",
                              info=backend.describe_entity("cup"))
    robot = RobotEmbodiment(name="r", reach_radius=2.0, z_min=0.0, z_max=2.0, hand_capacity=1)
    return v.check("GRASP", resolved, robot, BenchmarkWorldState())


def test_second_ik_solution_makes_the_candidate_feasible():
    verdict = _check(max_solutions=8)
    assert verdict.feasible
    assert np.allclose(verdict.details["q"], Q_GOOD)
    assert verdict.details["ik_solutions_checked"] == 2


def test_single_solution_engine_would_have_reported_collision():
    verdict = _check(max_solutions=1)
    assert not verdict.feasible and verdict.reason == "COLLISION"


def test_iter_ik_solutions_distinct_and_deterministic():
    pytest.importorskip("pinocchio")
    from rummagebench.feasibility.ik_solver import Pose, look_at_quaternion
    from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics

    kin = PinocchioKinematics(urdf_path=str(FIXTURES / "robots" / "long_arm.urdf"),
                              base_link="base_link", eef_link="gripper")
    target = Pose(np.array([0.5, 0.0, 0.6]), look_at_quaternion(np.array([1.0, 0.0, 0.0])))
    a = [r.q for r in kin.iter_ik_solutions(target, max_solutions=4)]
    b = [r.q for r in kin.iter_ik_solutions(target, max_solutions=4)]
    assert len(a) >= 2  # elbow-up and elbow-down
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    for i in range(len(a)):
        for k in range(i + 1, len(a)):
            assert np.max(np.abs(a[i] - a[k])) >= 0.05
    first = kin.solve_ik(target)
    assert np.allclose(first.q, a[0])  # same seed order as solve_ik


# --- GRASP approach segment ------------------------------------------------

class PositionKinematics:
    """q == EEF target position (identity base): every pose is reachable."""
    nq = 3
    controlled_joint_names = ["x", "y", "z"]

    def q_seed_neutral(self):
        return np.zeros(3)

    def solve_ik(self, target, seed_q=None):
        return IKResult(success=True, q=np.asarray(target.position, float).copy(),
                        position_error=0.0, orientation_error=0.0)

    def iter_ik_solutions(self, target, seed_q=None, max_solutions=8):
        yield self.solve_ik(target)


class SlabWorld:
    """Gripper occupies z in [q_z - 0.02, q_z + 0.10] (fingertips 2 cm past the
    EEF point). World: a table top below the cup and, optionally, a 2 cm slab
    between the cup's top face (0.55) and the 6 cm standoff (0.61)."""

    def __init__(self, slab: bool):
        self.bands = [(0.0, 0.45)] + ([(0.565, 0.585)] if slab else [])

    def refresh_world(self, backend):
        pass

    def check_configuration(self, q, ctx):
        lo, hi = q[2] - 0.02, q[2] + 0.10
        hit = any(lo < b_hi and b_lo < hi for b_lo, b_hi in self.bands)
        return CollisionResult(collision_free=not hit, self_collision=False, world_collision=hit)


def _grasp_with(slab: bool):
    backend = FakeBackend(
        entities={"cup": EntityInfo(name="cup", category="cup")}, anchors=set(),
        poses={"cup": [0.5, 0.0, 0.5]}, aabbs={"cup": ([0.45, -0.05, 0.45], [0.55, 0.05, 0.55])})
    v = FeasibilityValidator(backend, kinematics=PositionKinematics(),
                             config_collision=SlabWorld(slab))
    resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity="cup",
                              info=backend.describe_entity("cup"))
    robot = RobotEmbodiment(name="r", reach_radius=2.0, z_min=0.0, z_max=2.0, hand_capacity=1)
    return v.check("GRASP", resolved, robot, BenchmarkWorldState())


def test_standoff_beyond_a_thin_barrier_is_not_a_grasp():
    verdict = _grasp_with(slab=True)
    assert not verdict.feasible and verdict.reason == "COLLISION"
    top = [c for c in verdict.details["candidates"]
           if c.get("approach") and c["approach"].get("modelled")]
    assert top, "the top-face standoff is collision-free: only its approach fails"
    assert all(not c["approach"]["clear"] for c in top)
    assert top[0]["approach"]["blocked_at_m"] < 0.06


def test_clear_approach_keeps_the_grasp_feasible():
    verdict = _grasp_with(slab=False)
    assert verdict.feasible
    ap = verdict.details["approach"]
    assert ap["clear"] and ap["modelled"] and ap["waypoints"] == 4


# --- OPEN must not swing the furniture into the robot base ----------------

class OpeningBackend(FakeBackend):
    """Robot at the origin facing +x (default FakeBackend pose); a cabinet
    whose drawer, fully opened, reaches back to ``open_front_x``."""

    def __init__(self, open_front_x: float):
        super().__init__(entities={"cab": EntityInfo(name="cab", category="cabinet",
                                                    openable=True, fixed_base=True)},
                         anchors=set(), poses={"cab": [1.2, 0.0, 0.4]},
                         aabbs={"cab": ([0.9, -0.4, 0.0], [1.5, 0.4, 0.85])})
        self._open_front_x = open_front_x

    def opened_entity_aabb(self, entity):
        lo, hi = self.entity_aabb(entity)
        return ([self._open_front_x, lo[1], lo[2]], list(hi))


def _open_check(open_front_x: float):
    from rummagebench.core.types import FeasibilityVerdict

    backend = OpeningBackend(open_front_x)
    v = FeasibilityValidator(backend, kinematics=PositionKinematics(),
                             config_collision=SlabWorld(slab=False))
    v._check_configuration_space = lambda *a, **k: FeasibilityVerdict(feasible=True)
    resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity="cab",
                              info=backend.describe_entity("cab"))
    robot = RobotEmbodiment(name="r", reach_radius=2.0, z_min=0.0, z_max=2.0, hand_capacity=1)
    return v, v.check("OPEN", resolved, robot, BenchmarkWorldState())


def test_open_that_swings_into_the_base_is_a_collision():
    # base front edge at x = 0.40 (+ 0.03 margin); the drawer opens to 0.38
    v, verdict = _open_check(open_front_x=0.38)
    assert not verdict.feasible and verdict.reason == "COLLISION"
    assert "intrude" in verdict.details["note"]
    # a smaller base (as configured by the session) clears it
    v.base_half_extent = 0.30
    assert v._opened_intrudes_base("cab") is None


def test_open_with_room_to_spare_stays_feasible():
    _, verdict = _open_check(open_front_x=0.50)
    assert verdict.feasible
