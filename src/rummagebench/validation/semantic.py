"""Semantic validator: is the requested skill legal in this benchmark at all?

Checks the skill is enabled by the scenario and that the target kind matches
the skill's signature (NAV wants a place; OPEN/GRASP want an entity).
"""

from __future__ import annotations

from dataclasses import dataclass

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import Action, TargetKind
from rummagebench.skills.registry import SkillRegistry


@dataclass
class SemanticVerdict:
    valid: bool
    reason: str = "none"


class SemanticValidator:
    def __init__(self, scenario: ScenarioSpec, registry: SkillRegistry):
        self._scenario = scenario
        self._registry = registry

    def check(self, action: Action) -> SemanticVerdict:
        skill = self._registry.get(action.skill)
        if skill is None:
            return SemanticVerdict(
                valid=False, reason=f"unknown skill {action.skill!r}"
            )
        if action.skill not in self._scenario.skills:
            return SemanticVerdict(
                valid=False, reason=f"skill {action.skill!r} not enabled for scenario"
            )
        expected = skill.target_kind
        if action.target.type != expected:
            return SemanticVerdict(
                valid=False,
                reason=f"skill {action.skill!r} expects target type "
                f"{expected.value!r}, got {action.target.type.value!r}",
            )
        if action.target.type == TargetKind.PIXEL:
            return SemanticVerdict(
                valid=False, reason="pixel targets are not accepted in the MVP"
            )
        return SemanticVerdict(valid=True)
