"""R03: versioned evaluation events + the visual batch entry.

The visual session's trace is a first-class evaluation log: reset/metadata
never count as steps, action rows carry the schema fields, terminal and
infrastructure failures are explicit event types, metrics consume agent-mode
trajectories, and oracle/agent logs can never silently mix.
"""

from pathlib import Path

import json

import pytest

from rummagebench.core.events import (
    EVENT_TYPE_ACTION,
    EVENT_TYPE_INFRASTRUCTURE_ERROR,
    EVENT_TYPE_RESET,
    EVENT_TYPE_TERMINAL,
    EVALUATION_EVENTS_SCHEMA_VERSION,
    SESSION_MODE_AGENT,
    assert_single_session_mode,
)
from rummagebench.core.scenario import AnchorSpec, load_scenario
from rummagebench.core.visual_session import VisualProtocolSession
from rummagebench.evaluation.metrics import compute_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _teleport_kitchen(backend):
    backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                      orientation=[0.0, 0.0, 0.0, 1.0]))


def _visual_session(fake_backend) -> VisualProtocolSession:
    scenario = load_scenario(FIXTURE)
    session = VisualProtocolSession(fake_backend, scenario)
    session.reset()
    return session


def _actions(session):
    return [e for e in session._trace if e.get("event_type") == EVENT_TYPE_ACTION]


def test_trace_has_versioned_schema_fields(visual_session):
    frame_id = visual_session.observe().frame_id
    visual_session.step({"skill": "MOVE", "distance_cm": 10})
    actions = _actions(visual_session)
    assert len(actions) == 1
    e = actions[0]
    assert e["schema_version"] == EVALUATION_EVENTS_SCHEMA_VERSION
    assert e["session_mode"] == SESSION_MODE_AGENT
    assert e["action_protocol"] and e["feedback_protocol"]
    assert e["parsed_action"]["skill"] == "MOVE"
    assert e["executed"] is True
    assert e["binding"]["frame_id"] == frame_id
    assert "state_fingerprint_before" in e and "state_fingerprint_after" in e
    assert e["frame_id_after"]


def test_invalid_input_and_report_are_recorded(visual_session, visual_backend):
    visual_session.step({"skill": "TELEPORT"})  # not a public skill
    _teleport_kitchen(visual_backend)
    point = {"frame_id": visual_session.observe().frame_id, "x": 0.1, "y": 0.5}
    visual_session.step({"skill": "OPEN", "point": point})  # valid interaction
    visual_session.step({"skill": "REPORT_DONE"})  # report (goal not held)
    actions = _actions(visual_session)
    assert [e["feedback"] for e in actions] == [
        "INVALID_ACTION", "EXECUTED", "INVALID_ACTION",
    ]
    assert actions[0]["parsed_action"] is None
    assert actions[0]["private_reason"] == "SCHEMA"
    assert actions[1]["postcondition_satisfied"] is True
    assert actions[1]["skill_events"]
    # the failed report terminates: explicit terminal event, never an action
    terminal = [e for e in visual_session._trace
                if e.get("event_type") == EVENT_TYPE_TERMINAL]
    assert len(terminal) == 1
    assert terminal[0]["episode_status"] == "FAIL_FALSE_COMPLETION"


def test_reset_and_metadata_are_not_skill_steps(visual_session):
    visual_session.step({"skill": "REPORT_DONE"})
    types = [e.get("event_type") for e in visual_session._trace]
    assert types[0] == "run_metadata"
    assert types[1] == EVENT_TYPE_RESET
    # two non-action rows, one action row: reset is not a step
    assert len(_actions(visual_session)) == 1
    metrics = compute_metrics(list(visual_session._trace))
    assert metrics["steps"] == 1
    assert metrics["session_mode"] == SESSION_MODE_AGENT
    # the report failed (nothing held): FAIL_FALSE_COMPLETION
    assert metrics["final_status"] == "FAIL_FALSE_COMPLETION"
    assert metrics["task_success"] is False


def test_metrics_compute_on_visual_trajectory(visual_session, visual_backend):
    """V02 repro from the review: a visual session trace fed to
    compute_metrics used to raise KeyError('action')."""
    _teleport_kitchen(visual_backend)
    visual_backend.set_open("cabinet_B", True)  # spoon reachable (acceptance recipe)
    visual_session.step({"skill": "MOVE", "distance_cm": 10})
    visual_session.step({"skill": "TELEPORT"})
    point = {"frame_id": visual_session.observe().frame_id, "x": 0.79, "y": 0.5}
    visual_session.step({"skill": "GRASP", "point": point})
    visual_session.step({"skill": "REPORT_DONE"})  # holds spoon, not target
    metrics = compute_metrics(list(visual_session._trace))
    assert metrics["steps"] == 4
    # the schema-invalid TELEPORT still consumed a submitted step
    assert metrics["skills_used"] == {
        "MOVE": 1, "TELEPORT": 1, "GRASP": 1, "REPORT_DONE": 1,
    }
    assert metrics["interaction_attempts"] == 1
    # the grasp landed (executed, postcondition held): a valid interaction
    assert metrics["interaction_efficiency"] == 1.0
    assert metrics["final_status"] == "FAIL_FALSE_COMPLETION"


def test_oracle_and_agent_logs_cannot_mix(visual_session):
    visual_session.step({"skill": "MOVE", "distance_cm": 10})
    agent_events = [
        {**e, "session_mode": SESSION_MODE_AGENT} for e in _actions(visual_session)
    ]
    oracle_event = {
        "event_type": EVENT_TYPE_ACTION,
        "session_mode": "oracle",
        "action": {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}},
        "validation": {"semantic_valid": True},
        "execution": {"executed": True, "failure_reason": "NONE"},
        "status": "RUNNING",
    }
    with pytest.raises(ValueError, match="mixes session modes"):
        compute_metrics(agent_events + [oracle_event])
    # a homogeneous oracle log stays scoreable
    metrics = compute_metrics([oracle_event])
    assert metrics["session_mode"] == "oracle"


def test_assert_single_session_mode_helper():
    assert assert_single_session_mode([]) == "oracle"  # legacy default
    assert assert_single_session_mode(
        [{"event_type": EVENT_TYPE_ACTION, "session_mode": SESSION_MODE_AGENT}]
    ) == SESSION_MODE_AGENT


def test_infrastructure_error_event(visual_session, visual_backend, monkeypatch):
    """A realization fault during a visual step is an explicit
    infrastructure_error event and invalidates the run."""
    monkeypatch.setattr(type(visual_backend), "symbolic_grasp",
                        lambda self, entity: False)
    monkeypatch.setattr(type(visual_backend), "is_holding",
                        lambda self, entity: False)
    _teleport_kitchen(visual_backend)
    visual_backend.set_open("cabinet_B", True)  # spoon reachable (acceptance recipe)
    point = {"frame_id": visual_session.observe().frame_id, "x": 0.79, "y": 0.5}
    with pytest.raises(RuntimeError, match="invalidated"):
        visual_session.step({"skill": "GRASP", "point": point})
    infra = [e for e in visual_session._trace
             if e.get("event_type") == EVENT_TYPE_INFRASTRUCTURE_ERROR]
    assert len(infra) == 1
    assert infra[0]["run_invalidated"] is True


def test_run_visual_episode_batch_entry(visual_session, visual_backend, tmp_path):
    """R03: the visual batch loop drives .step() with raw public actions,
    writes the versioned event log, and scores from it."""
    from rummagebench.core.scenario import AnchorSpec
    from rummagebench.evaluation.episode_log import (
        ScriptedPixelAgent,
        run_visual_episode,
    )

    def make_agent(session):
        def setup(s):
            # run AFTER run_visual_episode's internal session.reset()
            s._backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                                 orientation=[0.0, 0.0, 0.0, 1.0]))
            s._backend.set_open("cabinet_B", True)
            s._capture_current()

        return ScriptedPixelAgent(
            [{"skill": "GRASP", "target": {"value": "distractor_spoon"}},
             {"skill": "REPORT_DONE"}],
            session, setup=setup,
        )

    summary = run_visual_episode(
        visual_session, make_agent(visual_session), tmp_path,
        model_version="unit-test-model",
    )
    assert summary["session_mode"] == "agent"
    assert summary["run_valid"] is True
    assert summary["status"] == "FAIL_FALSE_COMPLETION"
    assert summary["model_version"] == "unit-test-model"
    records = [json.loads(line)
               for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert records[0]["event_type"] == "run_metadata"
    assert records[0]["model_version"] == "unit-test-model"
    actions = [r for r in records if r.get("event_type") == EVENT_TYPE_ACTION]
    assert [a["raw_action"]["skill"] for a in actions] == ["GRASP", "REPORT_DONE"]
    assert actions[0]["binding"]["selected_entity"] == "distractor_spoon"
    metrics = compute_metrics(records)
    assert metrics["session_mode"] == SESSION_MODE_AGENT
    assert metrics["steps"] == 2
    assert metrics["final_status"] == "FAIL_FALSE_COMPLETION"


def test_run_visual_episode_infrastructure_invalidates(tmp_path, visual_backend,
                                                       monkeypatch):
    """A realization fault in the loop marks run_valid=False instead of a
    model failure."""
    from rummagebench.core.scenario import AnchorSpec
    from rummagebench.core.visual_session import VisualProtocolSession
    from rummagebench.evaluation.episode_log import (
        ScriptedPixelAgent,
        run_visual_episode,
    )

    monkeypatch.setattr(type(visual_backend), "symbolic_grasp",
                        lambda self, entity: False)
    monkeypatch.setattr(type(visual_backend), "is_holding",
                        lambda self, entity: False)
    scenario = load_scenario(FIXTURE)
    session = VisualProtocolSession(visual_backend, scenario)

    def setup(s):
        s._backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                             orientation=[0.0, 0.0, 0.0, 1.0]))
        s._backend.set_open("cabinet_B", True)
        s._capture_current()

    agent = ScriptedPixelAgent(
        [{"skill": "GRASP", "target": {"value": "distractor_spoon"}}],
        session, setup=setup)
    summary = run_visual_episode(session, agent, tmp_path)
    assert summary["run_valid"] is False
    # the run is INVALID (infrastructure), not a budget exhaustion: the
    # summary carries the session's terminal state honestly
    assert summary["status"] == "ENGINE_ERROR"
    records = [json.loads(line)
               for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(r.get("event_type") == EVENT_TYPE_INFRASTRUCTURE_ERROR
               for r in records)
