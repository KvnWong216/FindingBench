"""R1Pro collision model of the exported URDF.

The exported URDF carries real collider boxes; the q-based checker must
(1) be self-collision free at the neutral configuration (box-approximation
contacts disabled), (2) ignore the mobile base resting on the floor, and
(3) still detect the arm entering an obstacle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

URDF = Path(__file__).resolve().parents[2] / "build" / "robots" / "r1pro.urdf"
pytestmark = pytest.mark.skipif(not URDF.exists(), reason="exported r1pro URDF not present")


@pytest.fixture(scope="module")
def checker():
    from rummagebench.feasibility.hpp_fcl_checker import PinocchioCollisionChecker
    from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics

    k = PinocchioKinematics(urdf_path=str(URDF), base_link="base_link",
                            eef_link="right_eef_link", controlled_joints="auto")
    return PinocchioCollisionChecker(k, compute_min_distance=False)


def _box_world(center, half, entity):
    from rummagebench.feasibility.fcl_compat import collision_backend
    from rummagebench.feasibility.ik_solver import Pose
    from rummagebench.sim.base import WorldCollisionObject

    coal = collision_backend()
    c, h = np.asarray(center, float), np.asarray(half, float)
    w = WorldCollisionObject(entity=entity, link=None, geometry=coal.Box(*(2 * h)),
                             pose=Pose.from_lists(list(c), [0, 0, 0, 1]),
                             approximation="physics_collider_box",
                             aabb=(list(c - h), list(c + h)))
    return (w, w.geometry)


def _ctx():
    from rummagebench.feasibility.collision import InteractionCollisionContext
    from rummagebench.feasibility.ik_solver import Pose

    return InteractionCollisionContext(skill="GRASP", target_entity="t",
                                       base_pose=Pose.from_lists([0, 0, 0], [0, 0, 0, 1]))


def test_robot_has_collision_geometry(checker):
    assert len(checker._robot_geoms) > 100


def test_neutral_configuration_is_self_collision_free(checker):
    checker._world = []
    q = checker._kin.q_seed_neutral()
    assert checker.check_configuration(q, _ctx()).collision_free


def test_floor_under_the_base_is_not_an_interaction_collision(checker):
    checker._world = [_box_world([0, 0, -0.05], [5, 5, 0.05], "floors_0")]
    q = checker._kin.q_seed_neutral()
    assert checker.check_configuration(q, _ctx()).collision_free


def test_arm_inside_an_obstacle_collides(checker):
    k = checker._kin
    q = k.q_seed_neutral()
    bodies = k.collision_bodies(q)
    names = [n for n, _ in bodies]
    i = names.index("right_arm_link5")
    t = np.asarray(bodies[i][1].getTranslation())
    checker._world = [_box_world(t, [0.05, 0.05, 0.05], "cabinet_0")]
    res = checker.check_configuration(q, _ctx())
    assert not res.collision_free and res.world_collision


def test_aabb_prefilter_is_exact(checker):
    """Prefiltered world loop == unfiltered loop (pairs + verdict)."""
    rng = np.random.default_rng(3)
    k = checker._kin
    lo, hi = k.model.lowerPositionLimit, k.model.upperPositionLimit
    world = [_box_world(rng.uniform([-1, -1, 0], [1, 1, 1.6]), rng.uniform(0.02, 0.3, 3),
                        f"obj_{i}") for i in range(40)]
    checker._world = world
    for _ in range(40):
        q = rng.uniform(lo, hi)
        a = checker.check_configuration(q, _ctx())
        checker._compute_min_distance = True  # forces the unfiltered path
        try:
            b = checker.check_configuration(q, _ctx())
        finally:
            checker._compute_min_distance = False
        world_pairs = lambda r: sorted((p.robot_link, p.other) for p in r.pairs if p.kind == "world")
        assert a.collision_free == b.collision_free
        assert world_pairs(a) == world_pairs(b)
