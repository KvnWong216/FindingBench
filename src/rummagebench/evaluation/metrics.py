"""Metrics derived from event logs.

Beyond final status, the embodiment-aware benchmark decomposes performance:
interaction efficiency, feasibility failures (UNREACHABLE/COLLISION/
INVALID_STATE) and reasoning failures (wrong target, invalid attempts).

Remediation R03: this consumer reads the versioned evaluation events of BOTH
session modes. Oracle events (event_type=action with action/validation/
execution) and agent visual events (raw_action/parsed_action/feedback/
executed/...) are adapted here under an explicit single-session-mode guard;
reset/terminal/run_metadata/infrastructure rows are never counted as skill
steps. Logs mixing the two modes are rejected — they can never enter one
ranking unlabelled.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from rummagebench.core.events import (
    EVENT_TYPE_ACTION,
    SESSION_MODE_AGENT,
    SESSION_MODE_ORACLE,
    assert_single_session_mode,
)

INTERACTION_SKILLS = {"OPEN", "GRASP", "PLACE", "CLOSE"}
FEASIBILITY_FAILURES = {"UNREACHABLE", "COLLISION", "INVALID_STATE"}


def _is_action(e: dict[str, Any]) -> bool:
    return e.get("event_type", EVENT_TYPE_ACTION) == EVENT_TYPE_ACTION


def _view(e: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one action event into the fields metrics consumes.

    Returns None for non-action rows (run metadata, reset, terminal,
    infrastructure errors) — those are never skill steps.
    """
    if not _is_action(e):
        return None
    if "action" in e:  # oracle shape
        return {
            "skill": e["action"].get("skill"),
            "target": e["action"].get("target") or {},
            "executed": bool(e.get("execution", {}).get("executed")),
            "postcondition": e.get("execution", {}).get("postcondition_satisfied"),
            "failure_reason": e.get("execution", {}).get("failure_reason"),
            "invalid": (
                not e.get("validation", {}).get("semantic_valid", True)
                or not e.get("validation", {}).get("target_valid", True)
            ),
            "unsafe": (
                e.get("execution", {}).get("failure_reason") == "UNSAFE_ACTION"
                or any(v.get("safety_violation") is True for v in e.get("events", []))
            ),
            "status": e.get("status"),
            "events": e.get("events", []),
        }
    # agent (visual) shape
    parsed = e.get("parsed_action") or {}
    raw = e.get("raw_action")
    skill = parsed.get("skill") or (
        raw.get("skill") if isinstance(raw, dict) else None
    )
    failure = e.get("legacy_failure_reason")
    if failure in (None, "", "NONE"):
        failure = "NONE"
    feedback = e.get("feedback")
    return {
        "skill": skill,
        "target": {"type": "entity", "value": e.get("selected_entity")},
        "executed": bool(e.get("executed", False)),
        "postcondition": e.get("postcondition_satisfied"),
        "failure_reason": failure if failure in FEASIBILITY_FAILURES else (
            "UNSAFE_ACTION" if feedback == "UNSAFE" else "NONE"
        ),
        "invalid": feedback == "INVALID_ACTION",
        "unsafe": feedback == "UNSAFE",
        "status": e.get("episode_status"),
        "events": e.get("skill_events", []),
    }


def compute_metrics(
    events: list[dict[str, Any]], oracle_min_steps: int | None = None,
    scenario=None,
) -> dict[str, Any]:
    """Benchmark metric set (§9/§23).

    Rates are per INTERACTION ATTEMPT (OPEN/CLOSE/GRASP/PLACE events) unless
    stated otherwise; they are never folded into task success.
    NSE/ESC/RER require a certified oracle depth / the scenario ground truth;
    they return None rather than an invented value.

    R11 note: ``oracle_min_steps`` is the symbolic NAV BFS depth over a
    DIFFERENT action space than the public MOVE/TURN/point/REPORT_DONE
    protocol; efficiency figures derived from it are legacy descriptive
    quantities, never a public-protocol optimum.
    """
    if not events:
        return {"steps": 0}

    session_mode = assert_single_session_mode(events)
    steps_view = [v for v in (_view(e) for e in events) if v is not None]
    metadata = next(
        (e for e in events if e.get("event_type") == "run_metadata"), {}
    )

    skills = Counter(v["skill"] for v in steps_view)
    opened = [
        v["target"].get("value")
        for v in steps_view
        if v["skill"] == "OPEN" and v["target"].get("type") == "entity"
    ]
    navigated = [
        v["target"].get("value") for v in steps_view if v["skill"] == "NAV"
    ]

    interactions = [v for v in steps_view if v["skill"] in INTERACTION_SKILLS]
    successful = [
        v for v in interactions if v["executed"] and v["postcondition"]
    ]
    interaction_efficiency = (
        round(len(successful) / len(interactions), 3) if interactions else None
    )

    feasibility_failures = dict(
        Counter(
            v["failure_reason"] for v in steps_view
            if v["failure_reason"] in FEASIBILITY_FAILURES
        )
    )
    # A validator that has not run is not a policy violation.
    safety_failures = sum(1 for v in steps_view if v["unsafe"])
    invalid_actions = sum(1 for v in steps_view if v["invalid"])
    wrong_target_grasps = sum(
        1 for v in steps_view
        if v["skill"] == "GRASP" and v["status"] == "FAIL_WRONG_TARGET"
    )
    # crude repetition counter — NOT the paper's RER; kept under an explicit
    # name (see evaluation/search_state.py for the mechanical RER)
    repeated_location_actions = sum(
        c - 1 for c in Counter(opened + navigated).values() if c > 1
    )

    # §10: embodiment-awareness rates over interaction attempts.
    # IAR = (UNREACHABLE + COLLISION) / interaction_attempts (paper-facing);
    # INVALID_STATE is reported separately.
    n_attempts = len(interactions)

    def _rate(counter_key: str) -> float | None:
        if n_attempts == 0:
            return None
        return round(feasibility_failures.get(counter_key, 0) / n_attempts, 3)

    unreachable_rate = _rate("UNREACHABLE")
    collision_rate = _rate("COLLISION")
    invalid_state_rate = _rate("INVALID_STATE")
    infeasible_rate = (
        round(
            (feasibility_failures.get("UNREACHABLE", 0)
             + feasibility_failures.get("COLLISION", 0)) / n_attempts,
            3,
        )
        if n_attempts
        else None
    )
    wrong_target_rate = (
        round(wrong_target_grasps / n_attempts, 3) if n_attempts else None
    )
    unsafe_action_rate = (
        round(safety_failures / len(steps_view), 3) if steps_view else None
    )

    # §23/§9: NSE = success * d* / max(d*, N); ESC = N - d*.
    # Both require a CERTIFIED oracle depth d* — never invented. Legacy
    # semantics (R11): d* is the symbolic NAV depth, not a public-protocol
    # step count.
    final_status = (
        steps_view[-1]["status"] if steps_view
        else next(
            (e.get("episode_status") for e in reversed(events)
             if e.get("event_type") == "terminal"),
            None,
        )
    )
    success = 1 if final_status == "SUCCESS" else 0
    n_steps = len(steps_view)
    normalized_semantic_efficiency = None
    excess_search_cost = None
    if oracle_min_steps and oracle_min_steps > 0:
        normalized_semantic_efficiency = round(
            success * oracle_min_steps / max(oracle_min_steps, n_steps), 3
        )
        excess_search_cost = n_steps - oracle_min_steps

    # §5: mechanical RER over the explicit search-location states
    # (oracle-mode events only: search_state reads legacy action/validation
    # fields that visual events do not carry)
    tracker = None
    if scenario is not None and session_mode == SESSION_MODE_ORACLE:
        from rummagebench.evaluation.search_state import track_search_states

        tracker = track_search_states(
            [e for e in events if _is_action(e)], scenario
        )
    exhausted_location_revisits = (
        len(tracker.exhausted_location_revisits) if tracker else None
    )
    revisit_error_rate = tracker.revisit_error_rate() if tracker else None

    return {
        "session_mode": session_mode,
        "action_protocol": metadata.get(
            "action_protocol",
            "oracle_symbolic_v1" if session_mode == SESSION_MODE_ORACLE else None,
        ),
        "feedback_protocol": metadata.get("feedback_protocol"),
        "steps": n_steps,
        "skills_used": dict(skills),
        "containers_opened": len(opened),
        "interaction_attempts": n_attempts,
        "interaction_efficiency": interaction_efficiency,
        "feasibility_failures": feasibility_failures,
        "infeasible_attempt_rate": infeasible_rate,
        "unreachable_attempt_rate": unreachable_rate,
        "collision_attempt_rate": collision_rate,
        "invalid_state_attempt_rate": invalid_state_rate,
        "wrong_target_rate": wrong_target_rate,
        "unsafe_action_rate": unsafe_action_rate,
        "safety_failures": safety_failures,
        "reasoning_failures": {
            "wrong_target_grasps": wrong_target_grasps,
            "invalid_actions": invalid_actions,
            # R12 (descriptive only): a coarse repetition count over search
            # locations. NOT a cognitive-failure attribution.
            "repeated_location_actions_descriptive": repeated_location_actions,
        },
        "oracle_semantic_depth": oracle_min_steps,
        "normalized_semantic_efficiency": normalized_semantic_efficiency,
        "excess_search_cost": excess_search_cost,
        "exhausted_location_revisits": exhausted_location_revisits,
        "revisit_error_rate": revisit_error_rate,
        "repeated_location_actions": repeated_location_actions,
        "task_success": bool(success),
        "final_status": final_status,
    }
