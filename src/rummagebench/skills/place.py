"""PLACE(target): release the held object onto/into a receptacle.

The held object comes from the BENCHMARK-owned state, never from the
backend's internal grasp dict:

    held = state.held_object          (feasibility validated: not None)
    backend.symbolic_place(held, receptacle)   # explicit entity, no re-lookup
    state.held_object = None

Commit discipline (skills/realization.py) with a placement-aware
postcondition (round-2 hardening): the semantic release commits only after
the ACTUAL backend state confirms BOTH that the object is released AND that
it rests at the placement the feasibility stage validated —
``resolved.place_point`` (xy within settle-drift tolerance; z is
support-managed by the backend) or, when no validated point exists, the
explicit target relation (object footprint within the receptacle's plan
footprint, not below its floor). A backend that returns True but leaves the
object anywhere else is rolled back and reported as an infrastructure fault;
``state.release()`` and ``postcondition_satisfied=True`` never happen before
verification completes.
"""

from __future__ import annotations

import math

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.skills.realization import commit_realization

# resting drift around the validated support point (backend places the
# object centred on it; physics settle may move it slightly)
_PLACE_POINT_XY_TOL = 0.05  # m
# AABB slack for the target-relation fallback (no validated point)
_RELATION_SLACK = 0.02  # m


class PlaceSkill:
    name = "PLACE"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        receptacle = resolved.entity

        held = state.held_object
        assert held is not None, "PLACE executed without benchmark-held object"

        def placement_established() -> bool:
            """Actual-backend postcondition: RELEASED and AT the validated
            placement (or within the receptacle's target footprint when no
            validated point exists)."""
            if backend.is_holding(held):
                return False
            point = getattr(resolved, "place_point", None)
            if point is not None:
                pose = backend.entity_pose6d(held)
                if pose is None:
                    return False
                dx = float(pose.position[0]) - float(point[0])
                dy = float(pose.position[1]) - float(point[1])
                return math.hypot(dx, dy) <= _PLACE_POINT_XY_TOL
            obj = backend.entity_aabb(held)
            box = backend.entity_aabb(receptacle)
            if obj is None or box is None:
                return False  # placement unverifiable: fail loudly, not silently
            (alo, ahi), (blo, bhi) = obj, box
            return bool(
                alo[0] < bhi[0] and ahi[0] > blo[0]
                and alo[1] < bhi[1] and ahi[1] > blo[1]
                and alo[2] >= blo[2] - _RELATION_SLACK
            )

        # standardized realization: release the held entity's grasp joint and
        # move the object to the validated point (explicit entity — no
        # re-derivation). Raises FeasibilityBackendError (with backend
        # rollback) when the verified placement postcondition cannot be
        # established.
        commit_realization(
            backend,
            realize=lambda: backend.symbolic_place(
                held, receptacle, at=resolved.place_point),
            verify=placement_established,
            what=f"PLACE({held} -> {receptacle})",
        )

        # benchmark-owned state transition (semantic truth, committed AFTER
        # the verified realization landed)
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
