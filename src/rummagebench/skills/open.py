"""OPEN(container): symbolically set Open(container) == True.

The agent never controls handle grasp, joint trajectories, forces or
velocities. Opening the wrong container is NOT a failure — wrong search
hypotheses are allowed and simply consume planning steps.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


class OpenSkill:
    name = "OPEN"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget) -> SkillResult:
        assert resolved.entity is not None
        achieved = backend.set_open(resolved.entity, True)
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": resolved.entity},
            executed=True,
            postcondition_satisfied=bool(achieved),
            events=[
                {
                    "event": "open_executed",
                    "entity": resolved.entity,
                    "open": bool(achieved),
                }
            ],
            details={"open": bool(achieved)},
        )
