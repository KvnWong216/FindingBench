"""§21.7 / acceptance D: GRASP must not move the robot base.

symbolic_grasp is a realization step only: before/after base poses must be
identical (no implicit NAV, ever).
"""

from pathlib import Path

import numpy as np

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(fake_backend) -> BenchmarkSession:
    return BenchmarkSession(fake_backend, load_scenario(FIXTURE))


def test_grasp_does_not_teleport_base(fake_backend):
    session = _session(fake_backend)
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))

    before_pos, before_quat = fake_backend.robot_pose()
    result = session.act(
        Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value="target_knife"))
    )
    assert result.executed

    after_pos, after_quat = fake_backend.robot_pose()
    # the realization records the base pose it saw; it must be unchanged
    assert fake_backend.grasp_base_pose is not None
    assert np.allclose(before_pos, fake_backend.grasp_base_pose[0])
    assert np.allclose(after_pos, before_pos)
    assert np.allclose(after_quat, before_quat)


def test_grasp_event_declares_no_base_motion(fake_backend):
    session = _session(fake_backend)
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    result = session.act(
        Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value="target_knife"))
    )
    event = next(e for e in result.events if e["event"] == "grasp_executed")
    assert event["base_moved"] is False
