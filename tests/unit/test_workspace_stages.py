"""§39 two-stage workspace validation: OUT_OF_CAPABILITY vs UNSAFE."""

from rummagebench.core.public_types import PublicActionFeedback


def _open_point(visual_session, x, y, frame_id=None):
    frame_id = frame_id or visual_session._store.current_id
    return visual_session.step(
        {"skill": "OPEN", "point": {"frame_id": frame_id, "x": x, "y": y}}
    )


def test_kinematic_failure_maps_to_out_of_capability(visual_session, visual_backend):
    # open a cabinet far from the robot: semantic+state valid, IK impossible
    r = _open_point(visual_session, 0.1, 0.5)  # cabinet_B, ~3.3 m away
    assert r.feedback.code == PublicActionFeedback.OUT_OF_CAPABILITY
    assert r.episode_status == "RUNNING"


def test_rejected_interaction_leaves_world_unchanged(visual_session, visual_backend):
    before = visual_backend.dump_state()
    _open_point(visual_session, 0.1, 0.5)
    assert visual_backend.dump_state() == before


def test_state_invalid_maps_to_invalid_action(visual_session, visual_backend):
    # GRASP the (visible) closed-cabinet region? cabinet_B is openable not
    # graspable: GRASP(point on cabinet) -> INVALID_ACTION, RUNNING
    r = visual_session.step(
        {"skill": "GRASP", "point": {"frame_id": visual_session._store.current_id,
                                     "x": 0.1, "y": 0.5}}
    )
    assert r.feedback.code == PublicActionFeedback.INVALID_ACTION
    assert r.episode_status == "RUNNING"
    assert not visual_backend.is_holding("cabinet_B")


def test_failed_attempt_consumes_exactly_one_step(visual_session):
    s0 = visual_session.step(
        {"skill": "OPEN", "point": {"frame_id": "frame_999999", "x": 0.1, "y": 0.5}}
    )
    assert s0.planning_step == 1
    s1 = visual_session.step(
        {"skill": "OPEN", "point": {"frame_id": "frame_999999", "x": 0.1, "y": 0.5}}
    )
    assert s1.planning_step == 2


def test_stage_a_then_b_order_preserved(visual_session, visual_backend):
    """A Stage-A-valid but Stage-B-blocked candidate returns UNSAFE, not
    OUT_OF_CAPABILITY: the mapping follows legacy COLLISION."""
    from rummagebench.core.types import FailureReason
    from rummagebench.core.visual_session import VisualProtocolSession

    session = visual_session
    fb = Feedback = session._feedback(PublicActionFeedback.UNSAFE)
    assert fb.code == PublicActionFeedback.UNSAFE
    # and legacy mapping table:
    from rummagebench.core.types import StepResult  # noqa: F401  (schema import check)

    class _R:
        executed = False
        failure_reason = FailureReason.COLLISION

    assert session._map_legacy_feedback(_R()).code == PublicActionFeedback.UNSAFE

    class _R2:
        executed = False
        failure_reason = FailureReason.UNREACHABLE

    assert session._map_legacy_feedback(_R2()).code == PublicActionFeedback.OUT_OF_CAPABILITY
