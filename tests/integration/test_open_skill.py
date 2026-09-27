"""OPEN skill integration: closed cabinet -> OPEN -> open."""

import pytest

pytestmark = pytest.mark.sim

from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.core.scenario import load_scenario

from pathlib import Path

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios" / "knife_search_001" / "scenario.yaml"


def _target_container():
    scenario = load_scenario(SCENARIO)
    # first closed container listed in initial_states
    return next(
        name for name, st in scenario.initial_states.items() if st.open is False
    )


def test_open_sets_open_state(sim_session):
    sim_session.reset()
    container = _target_container()
    assert not sim_session._backend.is_open(container)

    # embodiment-aware: the interaction anchor id equals the container name;
    # OPEN is only feasible once the robot stands at the container
    sim_session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value=container)))
    result = sim_session.act(
        Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value=container))
    )

    assert result.executed
    assert result.observation.previous_action_result["postcondition_satisfied"]
    assert sim_session._backend.is_open(container)


def test_reset_restores_closed_state(sim_session):
    sim_session.reset()
    container = _target_container()
    sim_session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value=container)))
    sim_session.act(
        Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value=container))
    )
    assert sim_session._backend.is_open(container)
    sim_session.reset()
    assert not sim_session._backend.is_open(container)
