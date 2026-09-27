"""Allowed collision matrix tests (§21.5 / §9).

finger <-> target allowed, forearm <-> target forbidden, robot <-> foreign
forbidden, OPEN interaction-link semantics, PLACE receptacle semantics —
both at the matrix level and through the configuration-space checker.
"""

from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from rummagebench.feasibility.collision import (
    AllowedCollisionMatrix,
    InteractionCollisionContext,
)
from rummagebench.feasibility.hpp_fcl_checker import PinocchioCollisionChecker
from rummagebench.feasibility.ik_solver import Pose, look_at_quaternion
from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics
from rummagebench.sim.base import EntityInfo, WorldCollisionObject

from conftest import FakeBackend

FIXTURES = Path(__file__).parent / "fixtures"


def _matrix():
    return AllowedCollisionMatrix({"gripper": "gripper", "link1": "arm", "link2": "arm"})


def test_grasp_finger_vs_target_allowed():
    assert _matrix().is_allowed("gripper", "knife", "GRASP", "knife")


def test_grasp_arm_vs_target_forbidden():
    assert not _matrix().is_allowed("link2", "knife", "GRASP", "knife")
    assert not _matrix().is_allowed("link1", "knife", "GRASP", "knife")


def test_robot_vs_foreign_forbidden():
    assert not _matrix().is_allowed("gripper", "table", "GRASP", "knife")
    assert not _matrix().is_allowed("link2", "table", "GRASP", "knife")


def test_open_interaction_link_semantics():
    m = _matrix()
    # gripper may touch the interaction interface only
    assert m.is_allowed("gripper", "cabinet", "OPEN", "cabinet",
                        interaction_link="cabinet_B_door", world_link="cabinet_B_door")
    assert not m.is_allowed("gripper", "cabinet", "OPEN", "cabinet",
                            interaction_link="cabinet_B_door", world_link="cabinet_B_body")
    # arm never touches the cabinet
    assert not m.is_allowed("link2", "cabinet", "OPEN", "cabinet",
                            interaction_link="cabinet_B_door", world_link="cabinet_B_door")


def test_place_receptacle_semantics():
    m = _matrix()
    # held object and gripper may contact the receptacle
    assert m.is_allowed("held:cup", "table", "PLACE", "table", held_entity="cup")
    assert m.is_allowed("gripper", "table", "PLACE", "table")
    # the arm may not
    assert not m.is_allowed("link2", "table", "PLACE", "table")
    # held object against a foreign body is still forbidden
    assert not m.is_allowed("held:cup", "shelf", "PLACE", "table", held_entity="cup")


def _kin():
    return PinocchioKinematics(
        urdf_path=str(FIXTURES / "robots" / "short_arm.urdf"),
        base_link="base_link",
        eef_link="gripper",
    )


def _box_body(name, lo, hi, category=""):
    import coal

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    return WorldCollisionObject(
        entity=name, link=None,
        geometry=coal.Box(float(half[0]), float(half[1]), float(half[2])),
        pose=Pose.from_lists([float(v) for v in center]),
        category=category, approximation="aabb_primitive",
        aabb=([float(v) for v in lo], [float(v) for v in hi]),
    )


class _Backend(FakeBackend):
    def __init__(self, bodies):
        super().__init__(
            entities={"knife": EntityInfo(name="knife", category="knife")},
            anchors={"living_room"},
            poses={"knife": [0.45, 0.0, 0.5]},
            aabbs={"knife": ([0.42, -0.03, 0.47], [0.48, 0.03, 0.53])},
        )
        self._bodies = bodies

    def collision_geometries(self):
        return list(self._bodies)


def test_checker_finger_contact_with_target_is_collision_free():
    kin = _kin()
    checker = PinocchioCollisionChecker(kin, compute_min_distance=False)
    knife = _box_body("knife", [0.42, -0.03, 0.47], [0.48, 0.03, 0.53])
    checker.refresh_world(_Backend([knife]))

    # IK for a grasp candidate in front of the knife's -x face
    target = Pose(np.array([0.36, 0.0, 0.5]), look_at_quaternion(np.array([-1.0, 0, 0])))
    ik = kin.solve_ik(target)
    assert ik.success, ik

    ctx = InteractionCollisionContext(
        skill="GRASP", target_entity="knife", base_pose=Pose.identity()
    )
    result = checker.check_configuration(ik.q, ctx)
    assert result.collision_free, result.to_dict()


def test_checker_arm_contact_with_target_is_forbidden():
    kin = _kin()
    checker = PinocchioCollisionChecker(kin, compute_min_distance=False)
    # "knife" volumes flanking the folded-elbow corridor: the forearm
    # (link2) inevitably overlaps the near-side volume, the gripper does not
    right = _box_body("knife", [0.15, -0.2, 0.25], [0.26, 0.2, 0.48])
    right.link = "l_right"
    left = _box_body("knife", [-0.26, -0.2, 0.25], [-0.15, 0.2, 0.48])
    left.link = "l_left"
    checker.refresh_world(_Backend([right, left]))

    # a directly constructed folded configuration (deterministic): the elbow
    # sits inside the right volume's corridor and link2 sweeps back through it
    q_folded = np.array([0.981, 2.625, -2.036])
    ctx = InteractionCollisionContext(
        skill="GRASP", target_entity="knife", base_pose=Pose.identity()
    )
    result = checker.check_configuration(q_folded, ctx)
    assert not result.collision_free
    assert result.world_collision
    assert not result.self_collision
    # the offending robot links are arm-class, never the gripper
    assert all(p.robot_link != "gripper" for p in result.pairs)
