"""Same task, different embodiment -> different admissible action space.

This is the benchmark's headline claim (difficulty Level 3 / capability-aware
planning): two robots face the IDENTICAL world, and only their embodiment
capability differs; the grounded action graph must differ accordingly.
"""

from pathlib import Path

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.skill_grounder import (
    EmbodiedActionGroundingEngine,
    SkillGrounder,
)
from rummagebench.core.types import Action, TargetKind, TargetRef

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _with_reach(reach_radius: float):
    scenario = load_scenario(FIXTURE)
    scenario.robot.reach_radius = reach_radius
    return scenario


def test_same_world_different_embodiment_different_actions(fake_backend):
    nav_kitchen = Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen"))

    s_mobile = BenchmarkSession(fake_backend, _with_reach(reach_radius=1.0))  # mobile manipulator
    s_compact = BenchmarkSession(fake_backend, _with_reach(reach_radius=0.3))  # compact arm

    s_mobile.reset()
    s_compact.reset()
    # identical worlds, identical navigation step (perfect executor)
    s_mobile.act(nav_kitchen)
    s_compact.act(nav_kitchen)

    mobile_actions = s_mobile.available_skills()
    compact_actions = s_compact.available_skills()

    # the mobile manipulator can open the cabinet; the compact arm cannot
    assert "OPEN(cabinet_B)" in mobile_actions
    assert "OPEN(cabinet_B)" not in compact_actions
    # navigation is embodiment-independent (perfect navigation executor)
    assert "NAV(kitchen)" in mobile_actions
    assert "NAV(kitchen)" in compact_actions
    # GRASP semantics are embodiment-independent; reach is what gates them
    assert not any(a.startswith("GRASP(") for a in compact_actions if a not in mobile_actions)


def test_grounding_engine_alias():
    assert EmbodiedActionGroundingEngine is SkillGrounder


def test_capability_change_flips_skill_availability(fake_backend):
    """One robot, two capability configs: the action space must track the
    embodiment parameters, not the world."""
    nav = Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen"))

    s = BenchmarkSession(fake_backend, _with_reach(reach_radius=1.0))
    s.reset()
    s.act(nav)
    assert "OPEN(drawer_A)" in s.available_skills()

    # same world, degraded arm: drawer_A is ~0.77 m from the kitchen anchor
    scenario = _with_reach(reach_radius=0.4)
    s_short = BenchmarkSession(fake_backend, scenario)
    s_short.reset()
    s_short.act(nav)
    assert "OPEN(drawer_A)" not in s_short.available_skills()
