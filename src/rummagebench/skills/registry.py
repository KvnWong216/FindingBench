"""Skill registry and protocol.

All skills share one interface: a name, the target kind they accept, and an
execute() that performs the semantic transition and returns a structured
SkillResult. Skills never decide benchmark success/failure/horizon (Rule 4) —
that is BenchmarkSession's job.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


@runtime_checkable
class Skill(Protocol):
    name: str
    target_kind: TargetKind

    def execute(
        self, backend: SimBackend, resolved: ResolvedTarget, state
    ) -> SkillResult: ...


class SkillRegistry:
    """Maps skill names to implementations for one benchmark run."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills.keys())


def default_registry() -> SkillRegistry:
    from rummagebench.skills.close import CloseSkill
    from rummagebench.skills.grasp import GraspSkill
    from rummagebench.skills.nav import NavSkill
    from rummagebench.skills.open import OpenSkill
    from rummagebench.skills.place import PlaceSkill

    registry = SkillRegistry()
    for skill in (NavSkill(), OpenSkill(), GraspSkill(), PlaceSkill(), CloseSkill()):
        registry.register(skill)
    return registry
