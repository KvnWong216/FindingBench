"""NAV skill integration: living_room -> kitchen teleport changes the view."""

import pytest

pytestmark = pytest.mark.sim

from rummagebench.core.types import Action, TargetKind, TargetRef


def test_nav_moves_robot(sim_session):
    sim_session.reset()
    pos0, _ = sim_session._backend.robot_pose()
    rgb0 = sim_session._backend.get_observation().copy()

    result = sim_session.act(
        Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen"))
    )

    pos1, _ = sim_session._backend.robot_pose()
    import numpy as np

    assert result.executed
    assert result.observation.previous_action_result["postcondition_satisfied"]
    assert np.linalg.norm(np.asarray(pos1) - np.asarray(pos0)) > 0.5
    assert not np.array_equal(rgb0, sim_session._backend.get_observation())


def test_nav_unknown_place_is_invalid_not_crash(sim_session):
    sim_session.reset()
    result = sim_session.act(
        Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="atlantis"))
    )
    assert result.executed is False
    assert result.observation.previous_action_result["failure_reason"] == "INVALID_ACTION"
    assert sim_session.status().value == "RUNNING"
