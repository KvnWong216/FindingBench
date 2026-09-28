"""Dynamic skill grounding + embodiment feasibility tests.

Verifies the core claim of the embodiment-aware benchmark: the available
action space depends on robot capability, geometry and world state.
"""

from pathlib import Path

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef, WorldState
from rummagebench.feasibility.collision import CollisionChecker
from rummagebench.feasibility.ik_solver import default_solver
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.validation.feasibility import FeasibilityValidator

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(fake_backend):
    return BenchmarkSession(fake_backend, load_scenario(FIXTURE))


def test_nav_always_available(fake_backend):
    s = _session(fake_backend)
    s.reset()
    av = s.available_skills()
    assert "NAV(kitchen)" in av
    assert "NAV(living_room)" in av


def test_grasp_blocked_while_container_closed(fake_backend):
    s = _session(fake_backend)
    s.reset()
    # the knife sits inside CLOSED cabinet_B: interaction point is buried
    assert "GRASP(target_knife)" not in s.available_skills()


def test_open_unreachable_from_far_anchor(fake_backend):
    s = _session(fake_backend)
    s.reset()  # robot at living_room; cabinet_B is ~3.3 m away
    assert "OPEN(cabinet_B)" not in s.available_skills()


def test_skills_appear_after_nav_and_open(fake_backend):
    s = _session(fake_backend)
    s.reset()
    s.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    av = s.available_skills()
    assert "OPEN(cabinet_B)" in av
    s.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    av = s.available_skills()
    assert "GRASP(target_knife)" in av
    assert "CLOSE(cabinet_B)" in av


def test_out_of_space_attempt_is_invalid_action_not_listed(fake_backend):
    """Reachability is list membership: attempting an action outside the
    manipulation space is INVALID_ACTION (never an exposed UNREACHABLE)."""
    s = _session(fake_backend)
    s.reset()
    assert "OPEN(cabinet_B)" not in s.available_skills()
    r = s.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    assert r.executed is False
    assert r.failure_reason.value == "INVALID_ACTION"
    flag = any(
        ev.get("not_in_manipulation_space") for ev in r.events
    )
    assert flag
    assert r.observation.previous_action_result["failure_reason"] == "INVALID_ACTION"
    assert s.status().value == "RUNNING"


def test_hand_full_is_invalid_state(fake_backend):
    s = _session(fake_backend)
    # keep the episode running through a grasp: disable both grasp terminations
    scenario = load_scenario(FIXTURE)
    scenario.termination.fail_on_wrong_grasp = False
    scenario.termination.succeed_when_holding_target = False
    s = BenchmarkSession(fake_backend, scenario)
    s.reset()
    s.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    s.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    s.act(Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value="target_knife")))
    r = s.act(Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value="distractor_spoon")))
    assert r.failure_reason.value == "INVALID_STATE"
    assert s.status().value == "RUNNING"


def test_close_round_trip(fake_backend):
    s = _session(fake_backend)
    s.reset()
    s.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    s.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    r = s.act(Action(skill="CLOSE", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    assert r.executed and r.observation.previous_action_result["postcondition_satisfied"]
    assert not fake_backend.is_open("cabinet_B")


def test_place_feasibility_directly(fake_backend):
    v = FeasibilityValidator(
        fake_backend,
        ik_solver=default_solver(),
        collision_checker=CollisionChecker(fake_backend),
    )
    fake_backend._last_anchor_position = [3.0, 1.0, 0.0]  # stand at kitchen
    robot = RobotEmbodiment(name="test", hand_capacity=1)
    resolved = fake_backend.resolve_entity("cabinet_B")

    empty = WorldState(held_count=0)
    verdict = v.check("PLACE", resolved, robot, empty)
    assert not verdict.feasible and verdict.reason == "INVALID_STATE"

    fake_backend.symbolic_grasp("distractor_spoon")
    holding = WorldState(held_count=1)
    verdict = v.check("PLACE", resolved, robot, holding)
    assert verdict.feasible
