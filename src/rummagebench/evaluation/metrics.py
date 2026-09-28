"""Metrics derived from event logs.

Beyond final status, the embodiment-aware benchmark decomposes performance:
interaction efficiency, feasibility failures (UNREACHABLE/COLLISION/
INVALID_STATE) and reasoning failures (wrong target, invalid attempts,
unnecessary exploration).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

INTERACTION_SKILLS = {"OPEN", "GRASP", "PLACE", "CLOSE"}
FEASIBILITY_FAILURES = {"UNREACHABLE", "COLLISION", "INVALID_STATE"}


def compute_metrics(
    events: list[dict[str, Any]], oracle_min_steps: int | None = None,
    scenario=None,
) -> dict[str, Any]:
    """Benchmark metric set (§9/§23).

    Rates are per INTERACTION ATTEMPT (OPEN/CLOSE/GRASP/PLACE events) unless
    stated otherwise; they are never folded into task success.
    NSE/ESC/RER require a certified oracle depth / the scenario ground truth;
    they return None rather than an invented value.
    """
    if not events:
        return {"steps": 0}

    skills = Counter(e["action"]["skill"] for e in events)
    opened = [
        e["action"]["target"].get("value")
        for e in events
        if e["action"]["skill"] == "OPEN"
        and e["action"]["target"].get("type") == "entity"
    ]
    navigated = [
        e["action"]["target"].get("value")
        for e in events
        if e["action"]["skill"] == "NAV"
    ]

    interactions = [e for e in events if e["action"]["skill"] in INTERACTION_SKILLS]
    successful = [
        e
        for e in interactions
        if e["execution"]["executed"]
        and e["execution"].get("postcondition_satisfied")
    ]
    interaction_efficiency = (
        round(len(successful) / len(interactions), 3) if interactions else None
    )

    feasibility_failures = dict(
        Counter(
            e["execution"]["failure_reason"]
            for e in events
            if e["execution"]["failure_reason"] in FEASIBILITY_FAILURES
        )
    )
    safety_failures = sum(1 for e in events if not e["validation"].get("safe", True))
    invalid_actions = sum(
        1
        for e in events
        if not e["validation"].get("semantic_valid", True)
        or not e["validation"].get("target_valid", True)
    )
    wrong_target_grasps = sum(
        1 for e in events if e["action"]["skill"] == "GRASP" and e["status"] == "FAIL_WRONG_TARGET"
    )
    # crude repetition counter — NOT the paper's RER; kept under an explicit
    # name (see evaluation/search_state.py for the mechanical RER)
    repeated_location_actions = sum(
        c - 1 for c in Counter(opened + navigated).values() if c > 1
    )

    # §10: embodiment-awareness rates over interaction attempts.
    # IAR = (UNREACHABLE + COLLISION) / interaction_attempts (paper-facing);
    # INVALID_STATE is reported separately.
    attempts = [e for e in events if e["action"]["skill"] in INTERACTION_SKILLS]
    n_attempts = len(attempts)

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
    unsafe_action_rate = round(safety_failures / len(events), 3) if events else None

    # §23/§9: NSE = success * d* / max(d*, N); ESC = N - d*.
    # Both require a CERTIFIED oracle depth d* — never invented.
    success = 1 if events[-1]["status"] == "SUCCESS" else 0
    n_steps = len(events)
    normalized_semantic_efficiency = None
    excess_search_cost = None
    if oracle_min_steps and oracle_min_steps > 0:
        normalized_semantic_efficiency = round(
            success * oracle_min_steps / max(oracle_min_steps, n_steps), 3
        )
        excess_search_cost = n_steps - oracle_min_steps

    # §5: mechanical RER over the explicit search-location states
    from rummagebench.evaluation.search_state import track_search_states

    tracker = track_search_states(events, scenario) if scenario is not None else None
    exhausted_location_revisits = (
        len(tracker.exhausted_location_revisits) if tracker else None
    )
    revisit_error_rate = tracker.revisit_error_rate() if tracker else None

    return {
        "steps": len(events),
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
            "unnecessary_exploration": repeated_location_actions,
        },
        "oracle_semantic_depth": oracle_min_steps,
        "normalized_semantic_efficiency": normalized_semantic_efficiency,
        "excess_search_cost": excess_search_cost,
        "exhausted_location_revisits": exhausted_location_revisits,
        "revisit_error_rate": revisit_error_rate,
        "repeated_location_actions": repeated_location_actions,
        "task_success": bool(success),
        "final_status": events[-1]["status"],
    }
