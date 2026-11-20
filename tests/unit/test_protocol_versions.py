"""R05: protocol/feedback versioning, the fine feedback code whitelist and
the REPORT_COMPLETE alias normalization.

The four-class public boundary is unchanged and replayable
(four-class-v1); the canonical fine codes are evaluator-private, one per
failure branch with a fixed priority, and never leak hidden answers.
"""

from pathlib import Path

import pytest

from rummagebench.core.public_types import (
    FINE_FEEDBACK_CODES,
    FINE_REASON_MAP,
    PROTOCOL_VERSION,
    FEEDBACK_VERSION,
    AgentProtocolConfig,
    fine_reason_for,
)
from rummagebench.core.scenario import AnchorSpec, load_scenario

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _teleport_kitchen(backend):
    backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                      orientation=[0.0, 0.0, 0.0, 1.0]))


def test_versions_are_pinned_and_consistent():
    assert PROTOCOL_VERSION == "agent-visual-v2"
    assert FEEDBACK_VERSION == "four-class-v1"
    cfg = AgentProtocolConfig()
    assert cfg.protocol_version == PROTOCOL_VERSION
    assert cfg.feedback_version == FEEDBACK_VERSION
    # every mapped private reason lands in the fixed whitelist
    assert set(FINE_REASON_MAP.values()) <= FINE_FEEDBACK_CODES


def test_fine_reason_priority_is_fixed():
    # 1. mapped private reason wins
    assert fine_reason_for("STALE_FRAME", "INVALID_ACTION",
                           "RUNNING") == "STALE_FRAME"
    # 2. terminal statuses
    assert fine_reason_for(None, "INVALID_ACTION",
                           "FAIL_MAX_STEPS") == "MAX_STEPS"
    assert fine_reason_for(None, "INVALID_ACTION",
                           "FAIL_FALSE_COMPLETION") == "FALSE_COMPLETION"
    # 3. the successful report
    assert fine_reason_for(None, "EXECUTED", "SUCCESS",
                           skill="REPORT_DONE") == "TASK_COMPLETED"
    # 4. feedback-class-derived
    assert fine_reason_for(None, "OUT_OF_CAPABILITY", "RUNNING") == \
        "OUT_OF_CAPABILITY"
    assert fine_reason_for(None, "UNSAFE", "RUNNING") == "ACTION_UNSAFE"
    assert fine_reason_for(None, "EXECUTED", "RUNNING",
                           skill="GRASP") is None


def test_report_complete_alias_normalized(visual_session, visual_backend):
    """The optional public alias REPORT_COMPLETE behaves exactly like
    REPORT_DONE: same success rule, parsed action normalized, the alias
    normalization recorded, the raw action preserved."""
    _teleport_kitchen(visual_backend)
    visual_backend.set_open("cabinet_B", True)
    visual_backend.frame_entities["target_knife"] = (24, 28, 10, 10)
    visual_session._capture_current()
    r = visual_session.step({"skill": "GRASP",
                             "point": {"frame_id": visual_session.observe().frame_id,
                                       "x": 0.46, "y": 0.52}})
    assert r.feedback.code.value == "EXECUTED"
    r = visual_session.step({"skill": "REPORT_COMPLETE"})
    assert r.episode_status == "SUCCESS"
    actions = [e for e in visual_session._trace
               if e.get("event_type") == "action"]
    assert actions[-1]["raw_action"]["skill"] == "REPORT_COMPLETE"
    assert actions[-1]["parsed_action"]["skill"] == "REPORT_DONE"
    assert actions[-1]["alias_normalized"] == "REPORT_COMPLETE->REPORT_DONE"
    assert actions[-1]["fine_reason"] == "TASK_COMPLETED"
    terminal = [e for e in visual_session._trace
                if e.get("event_type") == "terminal"]
    assert terminal[-1]["episode_status"] == "SUCCESS"


def test_report_complete_alias_false_completion(visual_session):
    """The alias on an unsatisfied goal is the same terminal false
    completion, with the FALSE_COMPLETION fine code."""
    r = visual_session.step({"skill": "REPORT_COMPLETE"})
    assert r.episode_status == "FAIL_FALSE_COMPLETION"
    actions = [e for e in visual_session._trace
               if e.get("event_type") == "action"]
    assert actions[-1]["fine_reason"] == "FALSE_COMPLETION"


def test_fine_reasons_on_failure_branches(visual_session):
    """One canonical fine code per failure branch."""
    # malformed input -> INVALID_PARAMETER
    visual_session.step({"skill": "TELEPORT"})
    # capability rejection -> OUT_OF_CAPABILITY (far entity, proxy reach)
    point = {"frame_id": visual_session.observe().frame_id, "x": 0.1, "y": 0.5}
    r = visual_session.step({"skill": "OPEN", "point": point})
    fine_by_step = {e["planning_step"]: e.get("fine_reason")
                    for e in visual_session._trace
                    if e.get("event_type") == "action"}
    assert fine_by_step[1] == "INVALID_PARAMETER"
    assert fine_by_step[2] == "OUT_OF_CAPABILITY"
    assert r.feedback.code.value == "OUT_OF_CAPABILITY"
