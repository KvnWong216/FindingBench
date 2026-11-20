"""Post-build predicate verification helpers.

The simulator backend performs placement and reports what it did; this module
turns the backend build report plus the scenario into a final pass/fail
summary for the build report. Build must fail if any required predicate is
not satisfied.
"""

from __future__ import annotations

from typing import Any


def verify_predicates(scenario, build_report: dict[str, Any]) -> dict[str, Any]:
    """Summarize backend build results against scenario requirements."""
    placements_ok = all(p["verified"] for p in build_report.get("placements", []))
    anchors_ok = all(
        a["distance_to_anchor"] <= 0.5 for a in build_report.get("anchors", [])
    )
    spawned_ok = len(build_report.get("spawned", [])) == len(scenario.objects)
    states_ok = len(build_report.get("initial_states", [])) == len(
        [s for s in scenario.initial_states.values() if s.open is not None]
    )
    passed = placements_ok and anchors_ok and spawned_ok and states_ok
    return {
        "passed": passed,
        "checks": {
            "spawned_all_objects": spawned_ok,
            "placements_verified": placements_ok,
            "initial_states_applied": states_ok,
            "anchors_verified": anchors_ok,
        },
    }
