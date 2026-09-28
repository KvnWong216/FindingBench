"""Headline morphology test (§21.2 / acceptance A): same world, same task,
DIFFERENT robot morphology -> different admissible action sets.

Both robots share one topology (3-DOF planar arm, fixtures/robots/*.urdf) and
differ ONLY in link lengths: short_arm reaches 0.55 m, long_arm 0.85 m. The
cabinet interaction interface (auto-derived door anchor) sits at 0.59 m
shoulder distance: real URDF IK + configuration-space collision must admit
OPEN for the long arm and refuse it for the short one. No reach_radius is
involved anywhere (the pinocchio engine never reads it).
"""

from pathlib import Path

import pytest

pin = pytest.importorskip("pinocchio")

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.sim.base import (
    ArticulationInfo,
    ArticulationJointInfo,
    EntityInfo,
)

from conftest import FakeBackend

FIXTURES = Path(__file__).parent / "fixtures"


def _morphology_backend() -> FakeBackend:
    """Cabinet whose interaction interface is at 0.59 m shoulder distance.

    Door anchor: face x=0.63 minus 0.06 m standoff -> gripper frame at
    (0.57, 0, 0.45); shoulder at (0, 0, 0.30) -> 0.589 m. short_arm (0.55 m
    reach) cannot; long_arm (0.85 m) can.
    """
    return FakeBackend(
        entities={
            "cabinet_B": EntityInfo(
                name="cabinet_B", category="cabinet", fixed_base=True,
                openable=True, is_receptacle=True,
            ),
            "target_knife": EntityInfo(name="target_knife", category="knife"),
        },
        anchors={"living_room"},
        poses={
            "cabinet_B": [0.65, 0.0, 0.45],
            "target_knife": [0.30, 0.40, 0.40],  # off the arm's working plane
        },
        aabbs={
            "cabinet_B": ([0.63, -0.05, 0.0], [0.67, 0.05, 0.9]),
            "target_knife": ([0.27, 0.37, 0.37], [0.33, 0.43, 0.43]),
        },
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
            ("cabinet_B", "cabinet_B_body"): [0.80, 0.0, 0.45],
            ("cabinet_B", "cabinet_B_door"): [0.65, 0.0, 0.45],
        },
        link_aabbs={
            ("cabinet_B", "cabinet_B_door"): ([0.63, -0.05, 0.0], [0.67, 0.05, 0.9]),
        },
    )


def _scenario(urdf: str):
    scenario = load_scenario(FIXTURES / "morph.yaml")
    scenario.robot.kinematics.urdf_path = f"tests/unit/fixtures/robots/{urdf}"
    return scenario


def _session(urdf: str) -> BenchmarkSession:
    return BenchmarkSession(_morphology_backend(), _scenario(urdf))


def test_long_arm_reaches_short_arm_does_not():
    s_long = _session("long_arm.urdf")
    s_short = _session("short_arm.urdf")
    s_long.reset()
    s_short.reset()

    long_actions = s_long.available_skills()
    short_actions = s_short.available_skills()

    # THE headline property: morphology, not a capability scalar, decides
    assert "OPEN(cabinet_B)" in long_actions
    assert "OPEN(cabinet_B)" not in short_actions


def test_short_arm_structured_failure_is_unreachable():
    s_short = _session("short_arm.urdf")
    s_short.reset()
    result = s_short.act(
        Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B"))
    )
    assert result.executed is False
    # out of the manipulation space -> not a listed skill, attempt is invalid
    assert result.failure_reason.value == "INVALID_ACTION"
    event = result.events[-1]
    assert event.get("not_in_manipulation_space") is True
    # fine-grained internal attribution is preserved in the event details
    ik_reasons = {
        c["ik"].get("reason") for c in event["details"]["candidates"]
    }
    assert "NO_IK_SOLUTION" in ik_reasons


def test_nav_then_open_chain_differs_by_morphology():
    """Reach enables the whole action chain, not just one skill label."""
    s_long = _session("long_arm.urdf")
    s_short = _session("short_arm.urdf")
    s_long.reset()
    s_short.reset()

    nav = Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="living_room"))
    s_long.act(nav)
    s_short.act(nav)
    open_action = Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B"))
    r_long = s_long.act(open_action)
    assert r_long.executed
    r_short = s_short.act(open_action)
    assert not r_short.executed
