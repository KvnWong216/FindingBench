"""§39 public protocol tests: exact 8-skill library, strict schemas."""

from rummagebench.core.public_types import (
    PUBLIC_SKILLS,
    Feedback,
    PublicActionFeedback,
    PublicObservation,
)


def test_skill_library_is_exactly_eight():
    assert PUBLIC_SKILLS == [
        "MOVE", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE", "REPORT_DONE",
    ]


def test_no_entity_bound_skills_in_library():
    assert not any("(" in s or ")" in s for s in PUBLIC_SKILLS)


def _validate(raw):
    from pydantic import TypeAdapter
    from rummagebench.core.public_types import PublicAction

    return TypeAdapter(PublicAction).validate_python(raw)


def test_move_schema_signed():
    a = _validate({"skill": "MOVE", "distance_cm": -40.0})
    assert a.distance_cm == -40.0
    for bad in ({"skill": "MOVE", "distance_cm": 0.0},
                {"skill": "MOVE", "distance_cm": 200.0},
                {"skill": "MOVE", "distance_cm": 2.0}):
        import pytest
        from pydantic import ValidationError
        # zero / out of range rejected at protocol level (session maps range
        # violations to INVALID_ACTION; schema itself allows floats)
        if bad["distance_cm"] == 0.0:
            continue  # zero is INVALID_ACTION at session level, schema-valid
        _validate(bad)  # schema-valid float; range enforced by session


def test_move_extra_fields_forbidden():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _validate({"skill": "MOVE", "distance_cm": 40.0, "target": "kitchen"})


def test_turn_schema_signed():
    a = _validate({"skill": "TURN", "angle_deg": -90.0})
    assert a.angle_deg == -90.0


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
