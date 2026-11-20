"""OPEN(container): symbolically set Open(container) == True.

The agent never controls handle grasp, joint trajectories, forces or
velocities. Opening the wrong container is NOT a failure — wrong search
hypotheses are allowed and simply consume planning steps.

Commit discipline (skills/realization.py): the semantic open is committed
only when the backend joint state actually took it; otherwise the backend is
rolled back and an infrastructure fault is raised — the step is never
reported as EXECUTED with an unsatisfied postcondition.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.skills.realization import commit_realization


class OpenSkill:
    name = "OPEN"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        entity = resolved.entity
        commit_realization(
            backend,
            realize=lambda: backend.set_open(entity, True),
            verify=lambda: backend.is_open(entity),
            what=f"OPEN({entity})",
        )
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": entity},
            executed=True,
            postcondition_satisfied=True,
            events=[
                {
                    "event": "open_executed",
                    "entity": entity,
                    "open": True,
                }
            ],
            details={"open": True},
        )
