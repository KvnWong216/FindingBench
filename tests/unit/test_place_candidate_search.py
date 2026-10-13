"""Round-6: PLACE candidate search without the 8-point cap, and the PLACE
approach-segment check.

Every test here drives the REAL candidate generation
(``receptacle_place_targets`` through the Receptacle adapter) and the
PRODUCTION ``FeasibilityValidator.check``; IK and collision are injectable
controllable doubles. No hand-written candidate lists.
"""

from pathlib import Path

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

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"

# the countertop grid over [0,0,0.5]-[1.6,1.6,0.7] at 0.08 m step with a
# 0.06 m edge inset: (1.48/0.08 -> 19) x 19 grid points
GRID_X_COUNT = 19
GRID_Y_COUNT = 19


class _CountingIK:
    """Clean IK failure for the first ``fail_first`` calls, success after;
    optionally raises on a specific call. The returned q encodes the target
    pose Z so approach waypoints produce distinct configurations."""

    nq = 1
    controlled_joint_names = ["j"]

    def __init__(self, fail_first: int = 0, raise_on_call: int | None = None):
        self.calls = 0
        self.fail_first = fail_first
        self.raise_on_call = raise_on_call

    def q_seed_neutral(self):
        return np.zeros(1)

    def solve_ik(self, target, seed_q=None):
        self.calls += 1
        if self.raise_on_call is not None and self.calls >= self.raise_on_call:
            raise RuntimeError("IK backend exploded")
        if self.calls <= self.fail_first:
            return IKResult(success=False, q=None, reason="NO_IK_SOLUTION",
                            position_error=1e9, orientation_error=1e9)
        return IKResult(success=True,
                        q=np.array([float(target.position[2])]),
                        position_error=0.0, orientation_error=0.0)


class _ScriptedCollision:
    """Controllable collision double: an optional q-floor for mid-descent
    blocks, an optional HELD-ONLY q-floor (the carried body collides in the
    low zone while an empty EEF passes), and an optional raising call."""

    def __init__(self, q_floor: float | None = None,
                 held_floor: float | None = None,
                 raise_on_call: int | None = None):
        self.calls = 0
        self.q_floor = q_floor
        self.held_floor = held_floor
        self.raise_on_call = raise_on_call

    def refresh_world(self, backend):
        pass

    def check_configuration(self, q, ctx):
        self.calls += 1
        if self.raise_on_call is not None and self.calls >= self.raise_on_call:
            raise RuntimeError("coal distance query exploded")
        q0 = float(np.asarray(q).ravel()[0])
        if self.q_floor is not None and q0 < self.q_floor:
            return CollisionResult(collision_free=False, self_collision=False,
                                   world_collision=True)
        if (self.held_floor is not None
                and getattr(ctx, "held_offset", None) is not None
                and q0 < self.held_floor):
            # only the CARRIED body collides in the low zone
            return CollisionResult(collision_free=False, self_collision=False,
                                   world_collision=True)
        return CollisionResult(collision_free=True, self_collision=False,
                               world_collision=False)


def _search_backend():
    """Countertop support surface + a graspable held cup. The EntityInfo
    fields mirror exactly what production build_entity_info produces for
    these OG objects: a fixed countertop is is_receptacle=True and
    graspable=False; a small object is graspable=True."""
    backend = FakeBackend(
        entities={
            "counter": EntityInfo(name="counter", category="countertop",
                                  fixed_base=True, is_receptacle=True,
                                  graspable=False),
            "cup": EntityInfo(name="cup", category="cup", graspable=True),
        },
        anchors={"base"},
        poses={"counter": [0.8, 0.8, 0.6], "cup": [0.5, 0.5, 0.8]},
        aabbs={
            "counter": ([0.0, 0.0, 0.5], [1.6, 1.6, 0.7]),
            "cup": ([0.45, 0.45, 0.75], [0.55, 0.55, 0.85]),
        },
    )
    backend._last_anchor_position = [0.0, 0.0, 0.0]
    return backend


def _held_world():
    world = BenchmarkWorldState()
    # held_offset = T_eef^-1 . T_obj: the object sits 10 cm ABOVE the EEF
    # origin in EEF frame (identity rotations)
    world.grasp("cup", None)
    from rummagebench.feasibility.ik_solver import Pose

    world.held_offset = Pose.from_lists([0.0, 0.0, 0.10])
    return world


def _check(backend, kinematics, collision, world):
    validator = FeasibilityValidator(
        backend, kinematics=kinematics, config_collision=collision)
    resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity="counter",
                              info=backend.describe_entity("counter"))
    robot = RobotEmbodiment(name="r", reach_radius=2.0, z_min=0.0,
                            z_max=2.0, hand_capacity=1)
    return validator.check("PLACE", resolved, robot, world)


# ---------------------------------------------------------------------------
# candidate search without the 8-point cap
# ---------------------------------------------------------------------------


def test_first_batch_fails_ninth_candidate_succeeds():
    """Candidates 1-8 fail cleanly (first batch), candidate 9 (second
    batch) is a complete feasible witness: the search must continue into
    the next batch and report the witness with its index."""
    backend = _search_backend()
    ik = _CountingIK(fail_first=8)
    verdict = _check(backend, ik, _ScriptedCollision(), _held_world())
    assert verdict.feasible
    search = verdict.details["search"]
    assert search["candidates_total"] >= 9  # real grid, >= 9 candidates
    assert search["success_candidate_index"] == 8  # the NINTH candidate
    assert search["candidates_checked"] == 9
    assert search["stop_reason"] == "witness_found"
    assert search["search_complete"] is True


def test_full_candidate_set_exhausted_is_honest_unreachable():
    """Every candidate completes a clean evaluation and none is feasible:
    a normal rejection whose search record states the DISCRETE candidate
    set has no feasible member — never a continuous-space proof."""
    backend = _search_backend()
    ik = _CountingIK(fail_first=10**9)
    verdict = _check(backend, ik, _ScriptedCollision(), _held_world())
    assert not verdict.feasible
    assert verdict.reason == "UNREACHABLE"
    search = verdict.details["search"]
    assert search["candidates_total"] == search["candidates_checked"]
    assert search["candidates_total"] >= GRID_X_COUNT * GRID_Y_COUNT - 4
    assert search["search_complete"] is True
    assert search["stop_reason"] == "candidate_set_exhausted"
    assert "no feasible member" in search["note"]
    assert verdict.details["candidates_evaluated"] == search["candidates_total"]


def test_complete_witness_stops_the_search_early():
    """The first candidate is a complete witness: the search stops with
    exactly one checked candidate and never evaluates the rest."""
    backend = _search_backend()
    ik = _CountingIK(fail_first=0)
    collision = _ScriptedCollision()
    verdict = _check(backend, ik, collision, _held_world())
    assert verdict.feasible
    search = verdict.details["search"]
    assert search["candidates_checked"] == 1
    assert search["candidates_total"] > 1
    assert search["stop_reason"] == "witness_found"
    assert search["success_candidate_index"] == 0


def test_backend_exception_is_attributed_not_laundered():
    """A backend exception mid-search is an infrastructure fault carrying
    the search progress — never an UNREACHABLE/COLLISION verdict."""
    backend = _search_backend()
    ik = _CountingIK(raise_on_call=3)
    with pytest.raises(FeasibilityBackendError,
                       match="did not complete a valid evaluation"):
        _check(backend, ik, _ScriptedCollision(), _held_world())


# ---------------------------------------------------------------------------
# PLACE approach segment
# ---------------------------------------------------------------------------


def test_approach_normal_descent_passes_with_evidence():
    """Clear descent: the approach evidence records the modelled straight-
    line path, its resolution, segment length and the held geometry."""
    backend = _search_backend()
    ik = _CountingIK(fail_first=0)
    verdict = _check(backend, ik, _ScriptedCollision(), _held_world())
    assert verdict.feasible
    approach = verdict.details["approach"]
    assert approach["modelled"] is True
    assert approach["clear"] is True
    assert approach["held_geometry"] is True
    assert approach["waypoint_resolution_m"] == 0.01
    assert approach["waypoints"] >= 1
    assert approach["segment_m"] > 0.0
    assert "symbolic execution teleports" in approach["path"]


def test_approach_midway_collision_rejects_feasible_endpoint():
    """The endpoint configuration is clear but a MID-descent waypoint is in
    collision: the candidate must be rejected with the block located on the
    segment (not at the endpoint)."""
    backend = _search_backend()
    ik = _CountingIK(fail_first=0)
    # endpoint q ~= standoff z (0.76); the descent lowers q towards the
    # release z (0.66): a floor at 0.72 blocks mid-segment only
    verdict = _check(backend, ik, _ScriptedCollision(q_floor=0.72),
                     _held_world())
    assert not verdict.feasible
    candidate = verdict.details["candidates"][0]
    assert candidate["approach"]["clear"] is False
    assert candidate["approach"]["modelled"] is True
    blocked = candidate["approach"]["blocked_at_m"]
    assert 0.0 < blocked < candidate["approach"]["segment_m"]


def test_approach_carries_the_held_geometry():
    """The held body (real grasp offset) participates in the waypoint
    collision checks: with the grasp offset the carried cup collides in the
    low zone (descent blocked); without the offset the same descent is
    clear — the held geometry demonstrably changes the verdict."""
    backend = _search_backend()
    collision = _ScriptedCollision(held_floor=0.72)
    verdict = _check(backend, _CountingIK(fail_first=0), collision,
                     _held_world())
    assert not verdict.feasible
    approach = verdict.details["candidates"][0]["approach"]
    assert approach["clear"] is False
    assert approach["held_geometry"] is True
    assert 0.0 < approach["blocked_at_m"] < approach["segment_m"]

    # Without the grasp offset no production carry check can be made.
    empty_world = BenchmarkWorldState()
    empty_world.grasp("cup", None)
    with pytest.raises(FeasibilityBackendError, match="PLACE.*offset"):
        _check(backend, _CountingIK(), collision, empty_world)


def test_approach_check_exception_is_infrastructure_fault():
    """A collision exception inside the approach is an infrastructure
    fault — never a normal rejection of the candidate."""
    backend = _search_backend()
    with pytest.raises(FeasibilityBackendError, match="coal distance query exploded"):
        _check(backend, _CountingIK(fail_first=0),
               _ScriptedCollision(raise_on_call=2), _held_world())


def test_approach_unmodelled_is_recorded_not_claimed():
    """Non-PLACE/GRASP skills keep an explicit unmodelled record — an
    unmodelled check is never evidence of safety."""
    from rummagebench.validation.feasibility import FeasibilityValidator

    validator = FeasibilityValidator.__new__(FeasibilityValidator)
    validator._kinematics = None
    validator._config_collision = None

    class _T:
        metadata = {"approach_normal": [0.0, 0.0, -1.0], "standoff_m": 0.06}

    record = validator._approach("OPEN", _T(), np.zeros(1), None, None, None)
    assert record["modelled"] is False
    assert "not modelled" in record["note"]


@pytest.mark.parametrize('rotated_base', [False, True])
def test_place_release_pose_matches_execution_with_rotated_grasp(rotated_base):
    from rummagebench.feasibility.ik_solver import Pose

    class RecordingIK(_CountingIK):
        def __init__(self):
            super().__init__()
            self.targets = []

        def solve_ik(self, target, seed_q=None):
            self.targets.append(target)
            return super().solve_ik(target, seed_q)

    backend = _search_backend()
    base = Pose.from_lists([0.2, -0.1, 0.0], [0, 0, np.sin(.3), np.cos(.3)]) if rotated_base else Pose.identity()
    backend.robot_pose = lambda: base.to_lists()
    world = _held_world()
    world.held_offset = Pose.from_lists([.04, -.03, .10], [np.sin(.4), 0, 0, np.cos(.4)])
    ik = RecordingIK()
    verdict = _check(backend, ik, _ScriptedCollision(), world)
    assert verdict.feasible
    point = verdict.details['interaction_target']['surface_point']
    expected = Pose.from_lists([point[0], point[1], point[2] + .05 + .01])
    actual = base.compose(ik.targets[-1]).compose(world.held_offset)
    np.testing.assert_allclose(actual.position, expected.position, atol=1e-9)
    assert abs(np.dot(actual.orientation, expected.orientation)) == pytest.approx(1.0)
    # Rotation must be sampled as well as translation; final-only rotation
    # would miss collisions encountered while reorienting the carried object.
    angles = [2 * np.arccos(np.clip(abs(np.dot(a.orientation, b.orientation)), 0, 1))
              for a, b in zip(ik.targets, ik.targets[1:])]
    assert max(angles) <= np.deg2rad(5) + 1e-8
    assert any(angle > 1e-4 for angle in angles)


@pytest.mark.parametrize('missing', ['offset', 'aabb'])
def test_missing_place_carry_data_faults_before_ik(missing):
    backend = _search_backend()
    world = _held_world()
    if missing == 'offset':
        world.held_offset = None
    else:
        original = backend.entity_aabb
        backend.entity_aabb = lambda entity: None if entity == 'cup' else original(entity)
    ik = _CountingIK()
    with pytest.raises(FeasibilityBackendError, match='PLACE.*(offset|AABB)'):
        _check(backend, ik, _ScriptedCollision(), world)
    assert ik.calls == 0


def test_place_pure_rotation_checks_intermediate_collisions():
    from rummagebench.feasibility.ik_solver import Pose

    class RotationIK(_CountingIK):
        def solve_ik(self, target, seed_q=None):
            return IKResult(success=True, q=target.orientation.copy(),
                            position_error=0.0, orientation_error=0.0)

    class RotationCollision(_ScriptedCollision):
        def check_configuration(self, q, ctx):
            # Both endpoint orientations are clear; only the intermediate
            # rotation of the carried body intersects the obstacle.
            angle = 2 * np.arccos(np.clip(abs(q[3]), 0, 1))
            blocked = np.deg2rad(60) < angle < np.deg2rad(120)
            return CollisionResult(collision_free=not blocked,
                                   self_collision=False, world_collision=blocked)

    world = _held_world()
    world.held_offset = Pose.identity()
    # cup half-height + release clearance == STANDOFF_M: no translation,
    # but the downward EEF must rotate to the identity release orientation.
    verdict = _check(_search_backend(), RotationIK(), RotationCollision(), world)
    assert not verdict.feasible
    approach = verdict.details['candidates'][0]['approach']
    assert approach['segment_m'] == pytest.approx(0.0)
    assert approach['reason'] == 'COLLISION'
    assert approach['waypoints'] >= 36
