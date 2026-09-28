"""§21.6 / acceptance G: OPEN is grounded at the interaction interface.

Regression guard against degrading back to object-center reachability: the
cabinet ROOT origin is within proxy-reach, but the auto-derived door anchor
(handle fallback_link_anchor) is beyond the arm's URDF reach -> OPEN must be
unavailable, with candidate evidence showing the link anchor (not the root).
"""

from pathlib import Path

import pytest

pin = pytest.importorskip("pinocchio")

from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.sim.base import ArticulationInfo, ArticulationJointInfo, EntityInfo

from conftest import FakeBackend

FIXTURES = Path(__file__).parent / "fixtures"


def _backend() -> FakeBackend:
    """Cabinet root at 0.55 m (inside nominal proxy reach), door face at
    0.73 m -> gripper frame at 0.67 m, 0.37 m above the shoulder:
    distance 0.766 m > short_arm reach 0.55 m."""
    return FakeBackend(
        entities={
            "cabinet_B": EntityInfo(
                name="cabinet_B", category="cabinet", fixed_base=True,
                openable=True, is_receptacle=True,
            ),
        },
        anchors={"living_room"},
        poses={"cabinet_B": [0.55, 0.0, 0.45]},
        aabbs={"cabinet_B": ([0.73, -0.05, 0.0], [0.77, 0.05, 0.9])},
        articulations={
            "cabinet_B": ArticulationInfo(
                entity="cabinet_B",
                joints=[
                    ArticulationJointInfo(
                        name="cabinet_B_door_joint", joint_type="revolute",
                        parent_link="cabinet_B_body", child_link="cabinet_B_door",
                        axis=[0.0, 0.0, 1.0], limits=(0.0, 1.6),
                    )
                ],
                links=["cabinet_B_body", "cabinet_B_door"],
            ),
        },
        link_poses={
            ("cabinet_B", "cabinet_B_body"): [0.90, 0.0, 0.45],
            ("cabinet_B", "cabinet_B_door"): [0.75, 0.0, 0.45],
        },
        link_aabbs={
            ("cabinet_B", "cabinet_B_door"): ([0.73, -0.05, 0.0], [0.77, 0.05, 0.9]),
        },
    )


def _session() -> BenchmarkSession:
    from rummagebench.core.scenario import load_scenario

    scenario = load_scenario(FIXTURES / "morph.yaml")
    scenario.robot.kinematics.urdf_path = "tests/unit/fixtures/robots/short_arm.urdf"
    return BenchmarkSession(_backend(), scenario)


def _open() -> Action:
    return Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B"))


def test_open_unavailable_when_only_handle_is_out_of_reach():
    session = _session()
    session.reset()
    # the regression: object-center proxy reach would admit this skill
    assert "OPEN(cabinet_B)" not in session.available_skills()
    result = session.act(_open())
    assert result.executed is False
    assert result.failure_reason.value == "INVALID_ACTION"
    assert result.events[-1].get("not_in_manipulation_space") is True
    details = result.events[-1]["details"]
    assert details["candidates"], "candidate evidence missing"
    for candidate in details["candidates"]:
        # every evaluated candidate is the door-link interface, never the root
        assert candidate["target"]["link"] == "cabinet_B_door"
        assert candidate["target"]["source"] == "fallback_link_anchor"


def test_handle_anchor_is_auto_derived_not_the_root_pose():
    """The generated anchor sits on the door face with standoff, not at the
    cabinet root pose."""
    from rummagebench.feasibility.interaction_target import (
        articulated_interaction_targets,
    )

    backend = _backend()
    targets = articulated_interaction_targets("cabinet_B", backend)
    assert targets, "no interaction interface derived"
    target = targets[0]
    pos = target.pose.position
    # door face x=0.73 minus 0.06 standoff; NOT the root pose (0.55, 0, 0.45)
    assert abs(pos[0] - 0.67) < 1e-6
    assert abs(pos[2] - 0.45) < 1e-6
    assert target.source == "fallback_link_anchor"
    assert target.link == "cabinet_B_door"
