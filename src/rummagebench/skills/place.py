"""PLACE(target): release the held object onto/into a receptacle.

The held object is implicit world state (hand capacity is 1 in the MVP);
the action names only the destination receptacle. Execution is an instant
symbolic transition; feasibility (hand not empty, receptacle reachable)
is checked by the benchmark core before this skill runs.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


class PlaceSkill:
    name = "PLACE"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget) -> SkillResult:
        assert resolved.entity is not None
        placed = backend.symbolic_place_held(resolved.entity)
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": resolved.entity},
            executed=True,
            postcondition_satisfied=bool(placed),
            events=[
                {
                    "event": "place_executed",
                    "receptacle": resolved.entity,
                    "placed": bool(placed),
                }
            ],
            details={"placed": bool(placed)},
        )
