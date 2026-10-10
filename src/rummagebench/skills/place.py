"""PLACE(target): release the held object onto/into a receptacle.

The held object comes from the BENCHMARK-owned state, never from the
backend's internal grasp dict:

    held = state.held_object          (feasibility validated: not None)
    backend.symbolic_place(held, receptacle)   # explicit entity, no re-lookup
    state.held_object = None

Commit discipline (skills/realization.py): the semantic release is committed
only after the standardized realization established the released postcondition;
a failed realization rolls the backend back and raises an infrastructure
fault instead of reporting EXECUTED. The validated placement point is the one
the feasibility stage selected (``resolved.place_point``) — no alternative
candidates are sought after a failed execution.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.skills.realization import commit_realization


class PlaceSkill:
    name = "PLACE"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        receptacle = resolved.entity

        held = state.held_object
        assert held is not None, "PLACE executed without benchmark-held object"

        # standardized realization: release the held entity's grasp joint and
        # move the object to the validated point (explicit entity — no
        # re-derivation). Raises FeasibilityBackendError (with backend
        # rollback) when the released postcondition cannot be established.
        commit_realization(
            backend,
            realize=lambda: backend.symbolic_place(
                held, receptacle, at=resolved.place_point),
            verify=lambda: not backend.is_holding(held),
            what=f"PLACE({held} -> {receptacle})",
        )

        # benchmark-owned state transition (semantic truth, committed AFTER
        # the realization landed)
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
                    "realized": True,
                }
            ],
            details={"held": held, "receptacle": receptacle, "realized": True},
        )
