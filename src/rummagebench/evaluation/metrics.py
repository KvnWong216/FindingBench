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
    events: list[dict[str, Any]], oracle_min_steps: int | None = None
) -> dict[str, Any]:
    """Benchmark v0 metric set (§23).

    Rates are per INTERACTION ATTEMPT (OPEN/CLOSE/GRASP/PLACE events) unless
    stated otherwise; they are never folded into task success.
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
        if not e["validation"]["semantic_valid"]
        or not e["validation"].get("target_valid", True)
    )
    wrong_target_grasps = sum(
        1 for e in events if e["action"]["skill"] == "GRASP" and e["status"] == "FAIL_WRONG_TARGET"
    )
    revisits = sum(
        c - 1 for c in Counter(opened + navigated).values() if c > 1
    )

    # §10: embodiment-awareness rates over interaction attempts.
    # UNREACHABLE is not an exposed reason: reachability is list membership.
    # Attempts outside the manipulation space show up as INVALID_ACTION with
    # the not_in_manipulation_space event flag.
    attempts = [e for e in events if e["action"]["skill"] in INTERACTION_SKILLS]
    n_attempts = len(attempts)

    def _rate(counter_key: str) -> float | None:
        if n_attempts == 0:
            return None
        return round(feasibility_failures.get(counter_key, 0) / n_attempts, 3)

    not_in_space_attempts = sum(
        1 for e in events
        for flag in [any(
            ev.get("not_in_manipulation_space") for ev in e.get("events", [])
        )]
        if flag
    )
    not_in_space_rate = (
        round(not_in_space_attempts / n_attempts, 3) if n_attempts else None
    )
    collision_rate = _rate("COLLISION")
    invalid_state_rate = _rate("INVALID_STATE")
    infeasible_rate = (
        round(
            (not_in_space_attempts
             + feasibility_failures.get("COLLISION", 0)
             + feasibility_failures.get("INVALID_STATE", 0)) / n_attempts,
            3,
        )
        if n_attempts
        else None
    )
    wrong_target_rate = (
        round(wrong_target_grasps / n_attempts, 3) if n_attempts else None
    )
    unsafe_action_rate = round(safety_failures / len(events), 3) if events else None

    # §23: Search Efficiency = success * N*/N (clairvoyant oracle minimum);
    # without an oracle minimum only raw steps are reported — never invented.
    success = 1 if events[-1]["status"] == "SUCCESS" else 0
    search_efficiency = None
    if oracle_min_steps:
        search_efficiency = round(success * oracle_min_steps / len(events), 3)

    return {
        "steps": len(events),
        "skills_used": dict(skills),
        "containers_opened": len(opened),
        "interaction_attempts": n_attempts,
        "interaction_efficiency": interaction_efficiency,
        "feasibility_failures": feasibility_failures,
        "infeasible_attempt_rate": infeasible_rate,
        "not_in_space_attempt_rate": not_in_space_rate,
        "collision_attempt_rate": collision_rate,
        "invalid_state_attempt_rate": invalid_state_rate,
        "wrong_target_rate": wrong_target_rate,
        "unsafe_action_rate": unsafe_action_rate,
        "safety_failures": safety_failures,
        "reasoning_failures": {
            "wrong_target_grasps": wrong_target_grasps,
            "invalid_actions": invalid_actions,
            "unnecessary_exploration": revisits,
        },
        "search_efficiency": search_efficiency,
        "oracle_min_steps": oracle_min_steps,
        "task_success": bool(success),
        "final_status": events[-1]["status"],
    }
