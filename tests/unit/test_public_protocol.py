"""§39 public protocol tests: exact 8-skill library, strict schemas."""

from rummagebench.core.public_types import (
    PUBLIC_SKILLS,
    Feedback,
    PublicActionFeedback,
    PublicObservation,
)


def test_skill_library_includes_move_to():
    assert PUBLIC_SKILLS == [
        "MOVE_TO", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE", "VIEW", "CHECK", "REPORT_DONE",
    ]


def test_no_entity_bound_skills_in_library():
    assert not any("(" in s or ")" in s for s in PUBLIC_SKILLS)


def _validate(raw):
    from pydantic import TypeAdapter
    from rummagebench.core.public_types import PublicAction

    return TypeAdapter(PublicAction).validate_python(raw)


def test_removed_navigation_actions_rejected(visual_session):
    import pytest
    from pydantic import ValidationError
    for raw in ({"skill":"MOVE","distance_cm":40}, {"skill":"TURN","angle_deg":90}):
        with pytest.raises(ValidationError):
            _validate(raw)
        before=visual_session._backend.robot_pose()
        result=visual_session.step(raw)
        assert result.feedback.code == PublicActionFeedback.INVALID_ACTION
        assert visual_session._backend.robot_pose() == before


def test_point_schema_bounds_and_frame():
    ok = _validate({"skill": "OPEN",
                    "point": {"frame_id": "frame_000001", "x": 0.43, "y": 0.61}})
    assert ok.point.x == 0.43
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _validate({"skill": "GRASP",
                   "point": {"frame_id": "frame_000001", "x": 1.5, "y": 0.1}})


def test_report_done_no_parameters():
    import pytest
    from pydantic import ValidationError

    _validate({"skill": "REPORT_DONE"})
    with pytest.raises(ValidationError):
        _validate({"skill": "REPORT_DONE", "point": None})


def test_feedback_is_exactly_four_classes():
    assert {c.value for c in PublicActionFeedback} == {
        "EXECUTED", "INVALID_ACTION", "OUT_OF_CAPABILITY", "UNSAFE",
    }


def test_public_observation_has_no_privileged_fields():
    fields = set(PublicObservation.model_fields.keys())
    forbidden = {
        "available_skills", "candidate_skills", "robot_state", "object_states",
        "state_update", "holding", "resolved_target", "events", "target",
        "segmentation", "depth", "camera_extrinsics", "pose",
    }
    assert not (fields & forbidden)


def test_stale_frame_is_invalid_action(visual_session):
    r = visual_session.step(
        {"skill": "OPEN", "point": {"frame_id": "frame_999999", "x": 0.1, "y": 0.5}}
    )
    assert r.feedback.code == PublicActionFeedback.INVALID_ACTION
    assert r.episode_status == "RUNNING"


def test_discrete_turn_contract_and_execution(visual_session, monkeypatch):
    import math
    import pytest
    from pydantic import ValidationError
    from rummagebench.skills.move import yaw_from_quat
    backend=visual_session._backend
    original_pose=backend.robot_pose
    monkeypatch.setattr(backend, "robot_pose", lambda:
        (list(backend._teleports[-1][0]), list(backend._teleports[-1][1]))
        if backend._teleports else original_pose())
    for direction,angle in [("left",90),("right",-90),("back",180)]:
        before=visual_session._backend.robot_pose()
        result=visual_session.step({"skill":"TURN","direction":direction})
        assert result.feedback.code == PublicActionFeedback.EXECUTED
        after=visual_session._backend.robot_pose()
        assert after[0] == before[0]
        diff=yaw_from_quat(after[1])-yaw_from_quat(before[1])-math.radians(angle)
        assert math.atan2(math.sin(diff),math.cos(diff)) == pytest.approx(0,abs=1e-6)
        assert direction in result.feedback.message
    for raw in ({"skill":"TURN","direction":"forward"},
                {"skill":"TURN","angle_deg":90},
                {"skill":"TURN","direction":"left","angle_deg":90}):
        with pytest.raises(ValidationError): _validate(raw)
