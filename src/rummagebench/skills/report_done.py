"""REPORT_DONE: the ONLY normal success trigger (protocol §15).

Privately evaluates the scenario goal predicate. For the knife-search task:

    goal = benchmark_world_state.held_object == scenario.target.entity

True  -> feedback EXECUTED, episode SUCCESS.
False -> terminal FAIL_FALSE_COMPLETION (the agent is never told which
predicate failed).
"""

from __future__ import annotations


def goal_satisfied(world_state, scenario) -> bool:
    target = getattr(scenario.target, "entity", None)
    return target is not None and world_state.held_object == target
