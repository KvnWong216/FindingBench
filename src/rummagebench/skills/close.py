"""CLOSE(container): symbolically set Open(container) == False.

Mirror of OPEN. Feasibility (openable, currently open, reachable) is
checked by the benchmark core before this skill runs.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


class CloseSkill:
    name = "CLOSE"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget) -> SkillResult:
        assert resolved.entity is not None
        achieved = backend.set_open(resolved.entity, False)
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": resolved.entity},
            executed=True,
            postcondition_satisfied=bool(achieved),
            events=[
                {
                    "event": "close_executed",
                    "entity": resolved.entity,
                    "open": bool(achieved),
                }
            ],
            details={"open": bool(achieved)},
        )
