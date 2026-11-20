"""Safety validator: deterministic benchmark-level interaction rules (MVP).

This is deliberately simple and explicitly extensible: later versions may
add collision queries, support-relationship reasoning, fragile-object
constraints, motion feasibility, grasp-region safety and disturbance
prediction. The MVP does NOT claim trajectory-level collision safety —
symbolic GRASP never performs it.

MVP rules:
- grasping fixed-base furniture/appliances is unsafe (they cannot be acquired)
- grasping objects whose category is listed in scenario.safety.forbidden_categories
  is unsafe (e.g. hot pots, glassware if annotated)

Unsafe actions terminate the episode with FAIL_UNSAFE_ACTION and are NOT
executed.
"""

from __future__ import annotations

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import SafetyVerdict
from rummagebench.sim.base import ResolvedTarget


class SafetyValidator:
    def __init__(self, scenario: ScenarioSpec):
        self._scenario = scenario

    def check(self, skill_name: str, resolved: ResolvedTarget) -> SafetyVerdict:
        safety = self._scenario.safety
        if skill_name == "GRASP":
            info = resolved.info
            if info is not None:
                if safety.grasping_fixed_base_unsafe and info.fixed_base:
                    return SafetyVerdict(
                        safe=False,
                        reason="grasp_fixed_base",
                        details={
                            "entity": resolved.entity,
                            "category": info.category,
                        },
                    )
                if info.category in safety.forbidden_categories:
                    return SafetyVerdict(
                        safe=False,
                        reason="grasp_forbidden_category",
                        details={
                            "entity": resolved.entity,
                            "category": info.category,
                        },
                    )
        return SafetyVerdict(safe=True)
