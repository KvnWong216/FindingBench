"""REPORT_DONE: the ONLY normal success trigger (protocol §15).

Privately evaluates the scenario goal predicate. For the knife-search task:

    goal = benchmark_world_state.held_object == scenario.target.entity

True  -> feedback EXECUTED, episode SUCCESS.
False -> terminal FAIL_FALSE_COMPLETION (the agent is never told which
predicate failed).

Remediation R04: scenarios MAY declare a data-driven ``goal`` (terminal-state
predicates) and ``temporal`` (committed-event ordering) spec; those are
evaluated by evaluation/goal_checker.py — the SAME implementation metrics
consume. A scenario without a goal spec keeps the legacy holding(target)
rule, so existing published episodes and oracle certification are unchanged.
"""

from __future__ import annotations

from rummagebench.core.errors import FeasibilityBackendError


def goal_satisfied(world_state, scenario, backend=None, events=None) -> bool:
    if getattr(scenario, "behavior_task", None) is not None:
        if backend is None:
            raise ValueError("native BEHAVIOR goal requires the official task evaluator")
        return backend.behavior_goal_satisfied()
    goal = list(getattr(scenario, "goal", None) or [])
    temporal = list(getattr(scenario, "temporal", None) or [])
    if not goal and not temporal:
        target = getattr(scenario.target, "entity", None)
        return target is not None and world_state.held_object == target

    # data-driven contract: geometry/open-state predicates need backend
    # queries; temporal conditions need the committed event stream
    from rummagebench.evaluation.goal_checker import check_goal, check_temporal

    if backend is None:
        raise ValueError(
            "data-driven goal/temporal spec requires backend access for "
            "evaluation"
        )
    if goal:
        check = check_goal(world_state, scenario, backend)
        if check.fault is not None:
            # round-6: a placement-region/backend fault must NOT surface as a
            # plain false completion (model error) — invalidate the run
            # instead
            raise FeasibilityBackendError(
                f"goal evaluation cannot establish a placement relation: "
                f"{check.fault}"
            )
        if not check.satisfied:
            return False
    if temporal:
        return check_temporal(list(events or []), scenario).satisfied
    return True
