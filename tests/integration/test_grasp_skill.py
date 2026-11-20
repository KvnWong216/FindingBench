"""GRASP skill integration: validated grasp -> robot holds the object.

The grasp test opens the knife's receptacle first: acquiring an object
through a closed door is not a valid benchmark action, and the assisted
grasp joint cannot survive being torn through furniture.
"""

import pytest

pytestmark = pytest.mark.sim

from rummagebench.core.types import Action, TargetKind, TargetRef


def test_symbolic_grasp_holds_target(sim_session):
    sim_session.reset()
    scenario = sim_session.scenario
    target = scenario.target.entity

    # open the receptacle that holds the target (data-driven from placements)
    placement = next(
        p for p in scenario.placements if p.entity == target and p.relation == "inside"
    )
    receptacle = placement.receptacle
    sim_session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value=receptacle)))
    sim_session.act(
        Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value=receptacle))
    )

    result = sim_session.act(
        Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value=target))
    )

    assert result.executed
    assert sim_session._backend.is_holding(target)
    assert sim_session.status().value == "SUCCESS"
