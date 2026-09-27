"""PLACE(target): release the held object onto/into a receptacle.

The held object comes from the BENCHMARK-owned state, never from the
backend's internal grasp dict:

    held = state.held_object          (feasibility validated: not None)
    backend.symbolic_place(held, receptacle)   # explicit entity, no re-lookup
    state.held_object = None

Execution is an instant symbolic transition; feasibility (hand not empty,
receptacle region reachable, IK, collision) is checked by the benchmark core
before this skill runs.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


class PlaceSkill:
    name = "PLACE"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        receptacle = resolved.entity

        held = state.held_object
        assert held is not None, "PLACE executed without benchmark-held object"

        # realization: release the held entity's grasp joint and move the
        # object to the receptacle (explicit entity — no re-derivation)
        realized = backend.symbolic_place(held, receptacle)

        # benchmark-owned state transition
        state.release()

        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": receptacle},
            executed=True,
            postcondition_satisfied=True,
            events=[
                {
                    "event": "place_executed",
                    "entity": held,
                    "receptacle": receptacle,
                    "realized": bool(realized),
                }
            ],
            details={"held": held, "receptacle": receptacle, "realized": bool(realized)},
        )
