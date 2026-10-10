"""PLACE(target): release the held object onto/into a receptacle.

The held object comes from the BENCHMARK-owned state, never from the
backend's internal grasp dict:

    held = state.held_object          (feasibility validated: not None)
    backend.symbolic_place(held, receptacle)   # explicit entity, no re-lookup
    state.held_object = None

Commit discipline (skills/realization.py) with a placement-aware
postcondition (round-3): the semantic release commits only after the ACTUAL
backend state confirms the object is RELEASED and sits in the placement
relation the validated point implies.

Coordinate semantics (verified against sim/omnigibson/backend.py
``symbolic_place``): ``resolved.place_point`` is the SUPPORT-SURFACE point
(``interaction_target.py`` pins top-surface candidates to the region's top
face); the backend sets the object ORIGIN at ``support_z + half_height +
0.01``, so the object's post-move AABB BOTTOM lands ~1 cm above the support
plane. All vertical comparisons therefore use the post-move
``entity_aabb`` bottom against ``place_point[2]`` — never
``entity_pose6d.position[2]`` (the object origin), whose Z is not
commensurate with the support point.

With a validated point: XY centred on it (settle-drift bound) AND the
vertical/resting check of the receptacle's relation — on_top: bottom
resting on the support plane within a bidirectional band (same XY with an
arbitrary Z never commits); inside: full AABB containment. Without a
validated point: the relation is established from the receptacle's own
support region (explicit on_top/inside bounds); when the relation cannot
be established the step is rolled back and reported as an infrastructure
fault.
"""

from __future__ import annotations

import math

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.skills.realization import commit_realization

# resting drift around the validated support point (backend places the
# object centred on it; physics settle may move it slightly)
_PLACE_POINT_XY_TOL = 0.05  # m
# bidirectional resting band around the support plane (the backend leaves
# ~1 cm clearance between the object bottom and the support point)
_PLACE_Z_TOL = 0.05  # m
# AABB slack for the inside relation
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
            """Actual-backend postcondition: RELEASED and in the placement
            relation the validated point implies (see module docstring for
            the support-point coordinate semantics)."""
            if backend.is_holding(held):
                return False
            obj = backend.entity_aabb(held)
            if obj is None:
                return False
            alo, ahi = obj
            region = None
            get_region = getattr(backend, "receptacle_region", None)
            if callable(get_region):
                region = get_region(receptacle)
            kind = getattr(region, "kind", None)
            point = getattr(resolved, "place_point", None)

            if point is not None:
                # XY: object centred on the validated support point
                cx = (float(alo[0]) + float(ahi[0])) / 2.0
                cy = (float(alo[1]) + float(ahi[1])) / 2.0
                if math.hypot(cx - float(point[0]), cy - float(point[1])) \
                        > _PLACE_POINT_XY_TOL:
                    return False
                if kind == "inside_volume":
                    box = backend.entity_aabb(receptacle)
                    return box is not None and _contained(obj, box)
                # on_top relation (top_surface; also when the backend cannot
                # name a region but the feasibility stage validated a point):
                # the bottom RESTS on the support plane — same XY with an
                # arbitrary Z never commits
                return abs(float(alo[2]) - float(point[2])) <= _PLACE_Z_TOL + 1e-9

            # no validated point: establish the relation from the receptacle's
            # own support region, with explicit bounds per relation
            box = backend.entity_aabb(receptacle)
            if kind not in ("top_surface", "inside_volume") or box is None:
                return False  # relation not establishable: infrastructure fault
            if kind == "inside_volume":
                return _contained(obj, box)
            blo, bhi = box
            xy_overlap = (
                float(alo[0]) < float(bhi[0]) and float(ahi[0]) > float(blo[0])
                and float(alo[1]) < float(bhi[1]) and float(ahi[1]) > float(blo[1])
            )
            # on_top bounds: over the footprint AND bottom resting on the top
            # plane (not sunk into it, not floating above it)
            return bool(
                xy_overlap
                and abs(float(alo[2]) - float(bhi[2])) <= _PLACE_Z_TOL + 1e-9
            )

        def _contained(obj, box):
            """Inside relation: object AABB within the receptacle AABB
            (slack-tolerant on every axis — explicit upper AND lower
            bounds)."""
            (alo, ahi), (blo, bhi) = obj, box
            return bool(
                float(alo[0]) >= float(blo[0]) - _RELATION_SLACK
                and float(ahi[0]) <= float(bhi[0]) + _RELATION_SLACK
                and float(alo[1]) >= float(blo[1]) - _RELATION_SLACK
                and float(ahi[1]) <= float(bhi[1]) + _RELATION_SLACK
                and float(alo[2]) >= float(blo[2]) - _RELATION_SLACK
                and float(ahi[2]) <= float(bhi[2]) + _RELATION_SLACK
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
