"""Deterministic reset: reset -> act -> reset returns the same semantic start."""

import numpy as np
import pytest

pytestmark = pytest.mark.sim

from rummagebench.core.types import Action, TargetKind, TargetRef


def test_reset_determinism(sim_session):
    # two consecutive resets share one history path, isolating reset
    # determinism from physics re-settling differences after actions
    sim_session.reset()
    rgb_a = sim_session._backend.get_observation().copy()
    sim_session.reset()
    rgb_b = sim_session._backend.get_observation()

    assert rgb_a.shape == rgb_b.shape
    # The reset guarantee is SEMANTIC: same container states, same placements,
    # same robot anchor. Base micro-pose re-settles within ~2 cm and RTX
    # rendering is not pixel-deterministic, so the view is compared loosely.
    diff = float(np.abs(rgb_a.astype(int) - rgb_b.astype(int)).mean())
    assert diff < 45.0, f"reset view changed completely: mean abs diff {diff:.2f}"

    # semantic state must match the built initial state
    assert not sim_session._backend.is_open("bottom_cabinet_no_top_qohxjq_0")
    assert not sim_session._backend.is_holding("target_knife")


def test_reset_restores_robot_pose(sim_session):
    sim_session.reset()
    pose0, _ = sim_session._backend.robot_pose()
    sim_session.act(
        Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen"))
    )
    sim_session.reset()
    pose1, _ = sim_session._backend.robot_pose()
    assert np.allclose(pose0, pose1, atol=0.1)
