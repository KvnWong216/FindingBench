"""R02 fault injection: solver/backend exceptions are infrastructure faults.

The configuration-space check must distinguish normal "no solution found"
(structured rejection) from numerical/data errors and solver/collision
backend crashes (FeasibilityBackendError). Evaluation exceptions must never
be laundered into an UNREACHABLE/COLLISION proof.
"""

from __future__ import annotations

import numpy as np
import pytest

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.types import TargetKind
from rummagebench.feasibility.collision import CollisionResult
from rummagebench.feasibility.ik_solver import IKResult
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import EntityInfo, ResolvedTarget
from rummagebench.state.benchmark_state import BenchmarkWorldState
from rummagebench.validation.feasibility import FeasibilityValidator

from conftest import FakeBackend


class _PassiveCollision:
    def refresh_world(self, backend):
        pass

    def check_configuration(self, q, ctx):
        return CollisionResult(collision_free=True, self_collision=False,
                               world_collision=False)


class _CollidingCollision(_PassiveCollision):
    def check_configuration(self, q, ctx):
        return CollisionResult(collision_free=False, self_collision=False,
                               world_collision=True)


class _ThrowingIK:
    """Every candidate IK raises: a solver/backend crash, not a measurement."""

    nq = 2
    controlled_joint_names = ["a", "b"]

    def q_seed_neutral(self):
        return np.zeros(2)

    def solve_ik(self, target, seed_q=None):
        raise RuntimeError("IK backend exploded")


class _NonConvergingIK:
    """Clean structured IK failure: a normal, measured rejection."""

    nq = 2
    controlled_joint_names = ["a", "b"]

    def q_seed_neutral(self):
        return np.zeros(2)

    def solve_ik(self, target, seed_q=None):
        return IKResult(success=False, q=None, reason="NO_IK_SOLUTION",
                        position_error=1e9, orientation_error=1e9)


class _MixedIK:
    """Deterministic two-candidate behaviour by call order: the first
    candidate completes a valid evaluation, later candidates crash."""

    nq = 2
    controlled_joint_names = ["a", "b"]

    def __init__(self, first: str = "fail"):
        self._first = first
        self.calls = 0

    def q_seed_neutral(self):
        return np.zeros(2)

    def solve_ik(self, target, seed_q=None):
        self.calls += 1
        if self.calls == 1:
            if self._first == "ok":
                return IKResult(success=True, q=np.zeros(2), position_error=0.0,
                                orientation_error=0.0)
            return IKResult(success=False, q=None, reason="NO_IK_SOLUTION",
                            position_error=1e9, orientation_error=1e9)
        raise RuntimeError("collision engine exploded")


class _TwoCandidateInterface:
    """Stand-in ObjectInterface exposing exactly two interaction targets."""

    def __init__(self, info, backend):
        self.info, self.backend = info, backend

    def interaction_targets(self, skill_name, world):
        from rummagebench.feasibility.ik_solver import Pose

        return [
            type("T", (), {
                "pose": Pose.from_lists([0.5, 0.0, 0.5]),
                "link": None, "metadata": {},
                "to_dict": lambda self: {"link": None},
            })(),
            type("T", (), {
                "pose": Pose.from_lists([0.5, 0.0, 0.6]),
                "link": None, "metadata": {},
                "to_dict": lambda self: {"link": None},
            })(),
        ]


def _validator(kinematics, collision=None, backend=None):
    backend = backend or _small_backend()
    return FeasibilityValidator(
        backend, kinematics=kinematics,
        config_collision=collision or _PassiveCollision(),
    )


def _small_backend(nan_pose: bool = False) -> FakeBackend:
    backend = FakeBackend(
        entities={"cup": EntityInfo(name="cup", category="cup")},
        anchors=set(),
        poses={"cup": [0.5, 0.0, 0.5]},
        aabbs={"cup": ([0.45, -0.05, 0.45], [0.55, 0.05, 0.55])},
    )
    if nan_pose:
        backend._last_anchor_position = [float("nan"), 0.0, 0.0]
    else:
        backend._last_anchor_position = [0.0, 0.0, 0.0]
    return backend


def _check_grasp(validator, backend=None):
    backend = backend or _small_backend()
    resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity="cup",
                              info=backend.describe_entity("cup"))
    robot = RobotEmbodiment(name="r", reach_radius=2.0, z_min=0.0, z_max=2.0,
                            hand_capacity=1)
    return validator.check("GRASP", resolved, robot, BenchmarkWorldState())


def test_all_ik_candidates_throwing_is_infrastructure_fault():
    # fault injection from the review (V02): all candidates raise ->
    # previously UNREACHABLE with evaluation_errors=N; now a loud fault
    v = _validator(_ThrowingIK())
    with pytest.raises(FeasibilityBackendError, match="valid evaluation"):
        _check_grasp(v)


def test_nonfinite_base_pose_is_infrastructure_fault():
    backend = _small_backend(nan_pose=True)
    v = _validator(_NonConvergingIK(), backend=backend)
    with pytest.raises(FeasibilityBackendError, match="non-finite"):
        _check_grasp(v, backend)


def test_clean_ik_nonconvergence_is_a_normal_rejection():
    v = _validator(_NonConvergingIK())
    verdict = _check_grasp(v)
    assert not verdict.feasible
    assert verdict.reason == "UNREACHABLE"
    assert all("evaluation_error" not in c for c in verdict.details["candidates"])


def test_mixed_collision_and_errors_is_not_a_proof(monkeypatch):
    """One measured collision + one crashed candidate: the rejection stands
    but must be flagged as resting on incomplete evaluation."""
    import rummagebench.validation.feasibility as feas_mod

    monkeypatch.setattr(
        feas_mod, "build_object_interface",
        lambda info, backend: _TwoCandidateInterface(info, backend),
    )
    v = _validator(_MixedIK(first="ok"), collision=_CollidingCollision())
    verdict = _check_grasp(v)
    assert not verdict.feasible
    assert verdict.reason == "COLLISION"
    assert verdict.details.get("incomplete_evaluation") is True
    assert verdict.details.get("evaluation_errors") == 1


def test_mixed_clean_failure_and_errors_cannot_claim_unreachable(monkeypatch):
    """Clean IK failure on one candidate + crash on another: the exceptions
    prove nothing about reachability — infrastructure fault, not UNREACHABLE."""
    import rummagebench.validation.feasibility as feas_mod

    monkeypatch.setattr(
        feas_mod, "build_object_interface",
        lambda info, backend: _TwoCandidateInterface(info, backend),
    )
    v = _validator(_MixedIK(first="fail"), collision=_PassiveCollision())
    with pytest.raises(FeasibilityBackendError, match="valid evaluation"):
        _check_grasp(v)
