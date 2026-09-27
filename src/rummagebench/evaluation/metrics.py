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


def compute_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
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

    return {
        "steps": len(events),
        "skills_used": dict(skills),
        "containers_opened": len(opened),
        "interaction_efficiency": interaction_efficiency,
        "feasibility_failures": feasibility_failures,
        "safety_failures": safety_failures,
        "reasoning_failures": {
            "wrong_target_grasps": wrong_target_grasps,
            "invalid_actions": invalid_actions,
            "unnecessary_exploration": revisits,
        },
        "final_status": events[-1]["status"],
    }
