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
``entity_aabb`` bottom — never ``entity_pose6d.position[2]`` (the object
origin), whose Z is not commensurate with the support point.

The receptacle's own ``receptacle_region`` is the SINGLE source of truth
for the placement relation and its bounds (round-4): both the with-point
and no-point paths require ``region.kind`` to be ``top_surface`` or
``inside_volume`` — a missing, unknown or non-place region (e.g.
``link_surface``) never commits, even with a validated point. Region
bounds are world-frame AABBs (the OG backend builds them from world-frame
link/entity boxes, the same frame as ``entity_aabb``), so no coordinate
conversion applies — but the entity's OUTER AABB is never used as a proxy:
``inside_volume`` containment and ``top_surface`` support-plane/XY bounds
all read the region, whose volume can be strictly smaller than the outer
bounding box (root-link body vs. tall doors, interior vs. walls).

With a validated point: the point must lie within its region (consistency),
the object XY centred on it (settle-drift bound), and the region's relation
vertical rule satisfied — on_top: bottom resting on the region's support
plane within a bidirectional band; inside: full containment in the region
volume. Without a validated point: the region's relation is verified
directly. Any failure rolls the backend back and is reported as an
infrastructure fault.
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
# AABB slack for the inside relation and point/region consistency
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
            relation the receptacle's own support region defines (see module
            docstring for coordinate semantics)."""
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
            # round-4: BOTH paths require a known place relation from the
            # receptacle's own region — missing/unknown/non-place kinds
            # (None, 'link_surface', ...) never commit, even with a
            # validated point. Region bounds are world-frame AABBs, the same
            # frame as entity_aabb; the entity's outer box is never used as
            # a proxy for the region's valid volume.
            if kind not in ("top_surface", "inside_volume"):
                return False
            try:
                rlo = [float(v) for v in region.lo]
                rhi = [float(v) for v in region.hi]
            except (TypeError, ValueError):
                return False

            point = getattr(resolved, "place_point", None)
            if point is not None:
                # point consistency: the validated support point must lie in
                # the region it claims to come from
                p = [float(v) for v in point]
                if not all(
                    rlo[i] - _RELATION_SLACK <= p[i] <= rhi[i] + _RELATION_SLACK
                    for i in range(3)
                ):
                    return False
                # XY: object centred on the validated support point
                cx = (float(alo[0]) + float(ahi[0])) / 2.0
                cy = (float(alo[1]) + float(ahi[1])) / 2.0
                if math.hypot(cx - p[0], cy - p[1]) > _PLACE_POINT_XY_TOL:
                    return False
                if kind == "inside_volume":
                    return _contained(obj, (rlo, rhi))
                # top_surface: bottom rests on the REGION's support plane —
                # same XY with an arbitrary Z never commits
                return abs(float(alo[2]) - rhi[2]) <= _PLACE_Z_TOL + 1e-9

            # no validated point: verify the region's relation directly,
            # with bounds taken from the region (not the outer entity AABB)
            if kind == "inside_volume":
                return _contained(obj, (rlo, rhi))
            xy_overlap = (
                float(alo[0]) < rhi[0] and float(ahi[0]) > rlo[0]
                and float(alo[1]) < rhi[1] and float(ahi[1]) > rlo[1]
            )
            # on_top bounds: over the region footprint AND bottom resting on
            # the region's support plane (not sunk into it, not floating)
            return bool(
                xy_overlap
                and abs(float(alo[2]) - rhi[2]) <= _PLACE_Z_TOL + 1e-9
            )

        def _contained(obj, box):
            """Inside relation: object AABB within the REGION's valid volume
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
