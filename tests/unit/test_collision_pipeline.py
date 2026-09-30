"""Collision pipeline tests (§21.3 / §21.4): real configurations, real gates.

- self collision: the EEF target is reachable but ANY reaching configuration
  puts the gripper inside the robot's own base volume -> skill unavailable,
  failure = COLLISION with kind='self' evidence.
- world collision: the target sits beyond a wall slab; every configuration
  reaching it crosses the slab -> skill unavailable, failure = COLLISION
  with kind='world' evidence.

Both run the full session pipeline on URDF kinematics (short_arm fixture).
"""

from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.feasibility.ik_solver import Pose
from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics
from rummagebench.sim.base import EntityInfo, WorldCollisionObject

from conftest import FakeBackend

FIXTURES = Path(__file__).parent / "fixtures"


def _coal():
    from rummagebench.feasibility.fcl_compat import collision_backend
    coal = collision_backend()

    return coal


def _world_body(name: str, lo, hi, category: str = "") -> WorldCollisionObject:
    coal = _coal()
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    return WorldCollisionObject(
        entity=name,
        link=None,
        geometry=coal.Box(float(half[0]), float(half[1]), float(half[2])),
        pose=Pose.from_lists([float(v) for v in center]),
        category=category,
        approximation="aabb_primitive",
        aabb=([float(v) for v in lo], [float(v) for v in hi]),
    )


class GeometryBackend(FakeBackend):
    """FakeBackend whose world collision set is provided explicitly."""

    def __init__(self, world_bodies, **kwargs):
        super().__init__(**kwargs)
        self._world_bodies = world_bodies

    def collision_geometries(self):
        return list(self._world_bodies)


def _short_arm_backend(world_bodies, knife_aabb):
    return GeometryBackend(
        world_bodies,
        entities={
            "target_knife": EntityInfo(name="target_knife", category="knife"),
        },
        anchors={"living_room"},
        poses={"target_knife": [(a + b) / 2 for a, b in zip(*knife_aabb)]},
        aabbs={"target_knife": knife_aabb},
    )


def _session(backend) -> BenchmarkSession:
    scenario = load_scenario(FIXTURES / "morph.yaml")
    scenario.robot.kinematics.urdf_path = "tests/unit/fixtures/robots/short_arm.urdf"
    return BenchmarkSession(backend, scenario)


def _grasp(knife: str = "target_knife") -> Action:
    return Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value=knife))


def test_reachable_but_self_colliding_target_is_unavailable():
    """§21.3: the knife floats inside the robot's own base volume.

    Every IK solution reaching into the base box makes a gripper/link body
    overlap the base collision box -> self collision -> GRASP unavailable
    with COLLISION attribution.
    """
    knife_aabb = ([-0.03, -0.03, 0.12], [0.03, 0.03, 0.18])
    backend = _short_arm_backend(world_bodies=[], knife_aabb=knife_aabb)
    session = _session(backend)
    session.reset()

    assert "GRASP(target_knife)" not in session.available_skills()
    result = session.act(_grasp())
    assert result.executed is False
    assert result.failure_reason.value == "COLLISION"
    details = result.events[-1]["details"]
    collision_kinds = {
        (p["kind"], p["robot_link"]) for c in details["candidates"]
        for p in c.get("collision", {}).get("pairs", [])
    }
    assert ("self", "base_link") in collision_kinds


def test_reachable_but_world_blocked_target_is_unavailable():
    """§21.4: the knife sits beyond a wall slab; any reaching arm crosses it.

    Slab x in [0.15, 0.23], z in [0, 0.8]: the short arm's shoulder is at
    z=0.30 and its elbow radius is 0.30, so every crossing happens inside the
    slab's z-range -> world collision on every candidate.
    """
    knife_aabb = ([0.35, -0.03, 0.47], [0.41, 0.03, 0.53])
    slab = _world_body("wall_slab", [0.15, -0.5, 0.0], [0.23, 0.5, 0.8], "wall")
    backend = _short_arm_backend(world_bodies=[slab], knife_aabb=knife_aabb)
    session = _session(backend)
    session.reset()

    assert "GRASP(target_knife)" not in session.available_skills()
    result = session.act(_grasp())
    assert result.executed is False
    assert result.failure_reason.value == "COLLISION"
    details = result.events[-1]["details"]
    world_pairs = {
        p["other"] for c in details["candidates"]
        for p in c.get("collision", {}).get("pairs", [])
        if p["kind"] == "world"
    }
    assert "wall_slab" in world_pairs


def test_clear_target_is_graspable_with_real_ik():
    """Same engine, same arm, no obstacle: the knife is graspable."""
    knife_aabb = ([0.42, -0.03, 0.47], [0.48, 0.03, 0.53])
    backend = _short_arm_backend(world_bodies=[], knife_aabb=knife_aabb)
    session = _session(backend)
    session.reset()
    # candidate gripper frames are <= 0.54 m out at z in [0.44, 0.56]:
    # within the 0.55 m planar reach for the near-side candidates
    assert "GRASP(target_knife)" in session.available_skills()


def test_self_collision_checker_direct():
    """Checker-level (§21.3): the IK configuration reaching into the base
    volume self-collides."""
    kin = PinocchioKinematics(
        urdf_path=str(FIXTURES / "robots" / "short_arm.urdf"),
        base_link="base_link",
        eef_link="gripper",
    )
    from rummagebench.feasibility.hpp_fcl_checker import PinocchioCollisionChecker
    from rummagebench.feasibility.collision import InteractionCollisionContext

    checker = PinocchioCollisionChecker(kin, compute_min_distance=False)
    checker.refresh_world(_short_arm_backend([], ([-0.03, -0.03, 0.12], [0.03, 0.03, 0.18])))

    # a reaching configuration for a target inside the base volume
    ik = kin.solve_ik(_target_pose(0.09, 0.0, 0.15, (1.0, 0.0, 0.0)))
    assert ik.success
    ctx = InteractionCollisionContext(
        skill="GRASP", target_entity="target_knife", base_pose=Pose.identity()
    )
    result = checker.check_configuration(ik.q, ctx)
    assert not result.collision_free
    assert result.self_collision
    assert not result.world_collision


def _target_pose(x, y, z, direction):
    from rummagebench.feasibility.ik_solver import look_at_quaternion

    return Pose(np.array([x, y, z]), look_at_quaternion(np.array(direction)))
