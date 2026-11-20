"""Meaningful information-boundary and no-intervention checks for opt-in feedback."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/analysis"))

from failure_explanations import explain, sweep
from reason_feedback_session import ReasonFeedbackSession


def test_unverified_geometry_never_names_object():
    e = dict(raw_action={"skill": "MOVE"}, private_reason="MOVE_BLOCKED", feedback="UNSAFE")
    boxes = [dict(entity="secret_123", category="cabinet")]
    assert "cabinet" not in explain(e, boxes)
    assert "cabinet" in explain(e, boxes, verified=True)
    assert "secret_123" not in explain(e, boxes, verified=True)
    assert "back" not in explain(e, boxes, verified=True)


def test_reach_failure_does_not_invent_distance_or_fix():
    s = explain(dict(raw_action={"skill": "GRASP"}, legacy_failure_reason="UNREACHABLE"))
    assert "valid grasp pose" in s
    assert not any(word in s for word in ("meters", "closer", "backward"))


def test_grasp_causes_distinguish_reach_safety_and_object_type():
    event=dict(raw_action={"skill":"GRASP"}, feedback="INVALID_ACTION",
               private_reason="SKILL_NOT_APPLICABLE_TO_ENTITY",
               selected_entity="countertop_private_123")
    message=explain(event,selected_object=dict(category="countertop",cannot_pick_up=True))
    assert "countertop" in message and "cannot be picked up" in message
    assert "private_123" not in message and "not attempted" in message
    state=explain(event,selected_object=dict(category="cup",cannot_pick_up=False))
    assert "current object or holding state" in state
    assert "cannot be picked up" not in state
    collision=explain(dict(raw_action={"skill":"GRASP"},legacy_failure_reason="COLLISION"))
    assert "unsafe" in collision and "tested candidates" in collision
    assert "could not reach" not in collision
    assert "countertop" not in explain(event)


def test_object_state_and_capability_feedback_are_distinct():
    def message(skill, **evidence):
        return explain(dict(raw_action={"skill":skill},
                            private_reason="SKILL_NOT_APPLICABLE_TO_ENTITY"),
                       selected_object=evidence)
    assert "already open" in message("OPEN",openable=True,is_open=True)
    assert "already closed" in message("CLOSE",openable=True,is_open=False)
    assert "does not support opening" in message("OPEN",openable=False,category="cup")
    assert "not holding" in message("PLACE",place_supported=True,holding_empty=True)
    assert "does not support placement" in message("PLACE",place_supported=False,holding_empty=False)
    assert "capacity is already full" in message("GRASP",cannot_pick_up=False,holding_full=True)
    assert "already open" not in message("OPEN")


def test_collision_policy_and_geometry_failures_are_not_conflated():
    e=dict(raw_action={"skill":"OPEN"},legacy_failure_reason="COLLISION",
           manipulation_diagnostic_private=[{"details":{"note":"opened links intrude the robot base footprint",
                                                        "entity":"SECRET"}}])
    assert "opened object would overlap" in explain(e)
    assert "SECRET" not in explain(e)
    e["legacy_failure_reason"]="UNSAFE_ACTION"
    assert "task safety rule" in explain(e) and "collision" not in explain(e)
    for reason, phrase in [("NO_VALID_DEPTH","depth measurement"),
                           ("NO_OWNING_ENTITY","interactive object"),
                           ("NO_VISUAL_TARGET","no object"),
                           ("VIEW_UNAVAILABLE","no panoramic camera")]:
        assert phrase in explain(dict(private_reason=reason))
    e.update(legacy_failure_reason="UNREACHABLE",
             manipulation_diagnostic_private=[{"details":{"note":"no interaction interface"}}])
    assert "Reachability was not established" in explain(e)


def test_success_feedback_describes_effects_without_next_action():
    assert "right" in explain(dict(feedback="EXECUTED",raw_action={"skill":"TURN","angle_deg":-30}))
    assert "backward" in explain(dict(feedback="EXECUTED",raw_action={"skill":"MOVE","distance_cm":-10}))
    assert "heading did not change" in explain(dict(feedback="EXECUTED",raw_action={"skill":"VIEW"}))
    assert "episode has ended" in explain(dict(private_reason="GOAL_UNSATISFIED"))


def test_schema_feedback_uses_validation_evidence_only(visual_session):
    visual_session.__class__=ReasonFeedbackSession
    result=visual_session.step({"skill":"TURN"})
    assert "angle_deg is missing" in result.feedback.message
    assert result.planning_step==1 and result.feedback.code=="INVALID_ACTION"
    assert visual_session._trace[-1]["raw_action"]=={"skill":"TURN"}
    e=dict(private_reason="SCHEMA",raw_action=["SECRET"],
           schema_errors_private=[{"type":"extra_forbidden","loc":["SECRET"]}])
    assert "SECRET" not in explain(e)


def test_sweep_identifies_first_obstacle():
    e = dict(raw_action={"skill": "MOVE", "distance_cm": 100}, robot_pose_before=[0, 0, 0])
    boxes = [dict(entity="near", lo=[.6, -.2, .1], hi=[.8, .2, .5]),
             dict(entity="far", lo=[1.3, -.2, .1], hi=[1.5, .2, .5])]
    d = sweep(e, 0, boxes)
    assert not d["predicted_safe"]
    assert [b["entity"] for b in d["blockers"]] == ["near"]


def test_opt_in_format_failure_consumes_one_step_and_preserves_action(visual_session):
    visual_session.__class__ = ReasonFeedbackSession
    raw = {"skill": "MOVE", "distance_cm": 9999}
    r = visual_session.step(raw)
    assert r.planning_step == 1
    assert r.feedback.code == "INVALID_ACTION"
    assert "outside the allowed range" in r.feedback.message
    assert r.observation.feedback == r.feedback
    assert visual_session.observe().feedback == r.feedback
    event = visual_session._trace[-1]
    assert event["raw_action"] == raw
    assert event["feedback_protocol"] == "reason-feedback-v1"
    assert event["robot_pose_before"] == event["robot_pose_after"]


def test_opt_in_invalid_shape_is_not_repaired(visual_session):
    visual_session.__class__ = ReasonFeedbackSession
    r = visual_session.step({"skill": "TURN"})
    assert r.feedback.code == "INVALID_ACTION"
    assert r.planning_step == 1
    assert visual_session._trace[-1]["raw_action"] == {"skill": "TURN"}


def test_live_blocker_reason_preserves_rejection_and_hides_instance_id(visual_session, visual_backend):
    visual_session.__class__ = ReasonFeedbackSession
    x, y, z = visual_backend.robot_pose()[0]
    visual_backend.aabbs = {"cabinet_A": ([x+.6, y-.2, .1], [x+.8, y+.2, .5])}
    before = visual_backend.robot_pose()
    r = visual_session.step({"skill": "MOVE", "distance_cm": 50})
    assert r.feedback.code == "UNSAFE"
    assert "cabinet" in r.feedback.message
    assert "cabinet_A" not in r.feedback.message
    assert visual_backend.robot_pose() == before
    assert r.planning_step == 1
    event = visual_session._trace[-1]
    assert event["failure_diagnostic_private"]["verified_live"] is True
    assert event["failure_diagnostic_private"]["blockers"][0]["entity"] == "cabinet_A"


def test_diagnostic_error_does_not_change_executor_result(visual_session, visual_backend, monkeypatch):
    visual_session.__class__ = ReasonFeedbackSession
    x, y, z = visual_backend.robot_pose()[0]
    visual_backend.aabbs = {"cabinet_A": ([x+.6, y-.2, .1], [x+.8, y+.2, .5])}
    def fail(*a, **kw):
        raise ValueError("diagnostic unavailable")
    monkeypatch.setattr("rummagebench.core.reason_feedback_session.sweep", fail)
    r = visual_session.step({"skill": "MOVE", "distance_cm": 50})
    assert r.feedback.code == "UNSAFE"
    assert "an obstacle" in r.feedback.message
    assert "cabinet" not in r.feedback.message
    assert r.planning_step == 1
