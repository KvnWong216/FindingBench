"""CLOSE(container): symbolically set Open(container) == False.

Mirror of OPEN. Feasibility (openable, currently open, reachable) is
checked by the benchmark core before this skill runs.

Commit discipline (skills/realization.py): the semantic close is committed
only when the backend joint state actually took it; otherwise the backend is
rolled back and an infrastructure fault is raised — the step is never
reported as EXECUTED with an unsatisfied postcondition.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.skills.realization import commit_realization


class CloseSkill:
    name = "CLOSE"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        entity = resolved.entity
        commit_realization(
            backend,
            realize=lambda: backend.set_open(entity, False),
            verify=lambda: not backend.is_open(entity),
            what=f"CLOSE({entity})",
        )
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": entity},
            executed=True,
            postcondition_satisfied=True,
            events=[
                {
                    "event": "close_executed",
                    "entity": entity,
                    "open": False,
                }
            ],
            details={"open": False},
        )
