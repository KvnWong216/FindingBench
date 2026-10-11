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
from typing import Any

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.evaluation.goal_checker import check_placement_relation
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

        relation: dict[str, Any] = {}

        def placement_established() -> bool:
            """Actual-backend postcondition: RELEASED and in the placement
            relation the receptacle's own support region defines.

            Round-6: the relation verdict comes from the SHARED
            evaluation/goal_checker.check_placement_relation (single
            implementation, region-authoritative, fault-aware). This closure
            keeps only the PLACE-specific extras: validated-point
            consistency and settle-drift bounds. A region fault raises
            FeasibilityBackendError (distinguishable from a normal relation
            miss, which just fails verification); commit_realization rolls
            the backend back in both cases.
            """
            rel = check_placement_relation(
                backend, held, receptacle,
                # PLACE verifies the BACKEND release; the semantic release is
                # what happens after verification — consulting it here would
                # be circular
                backend_held=bool(backend.is_holding(held)))
            relation.update(rel.to_dict())
            if rel.fault is not None:
                # region data / backend fault — never dressed up as a normal
                # relation miss
                raise FeasibilityBackendError(
                    f"PLACE({held} -> {receptacle}): placement relation "
                    f"cannot be evaluated: {rel.fault}"
                )
            if not rel.satisfied:
                return False
            if rel.kind != "top_surface" and rel.kind != "inside_volume":
                return False  # defensive: shared check already rejects

            point = getattr(resolved, "place_point", None)
            if point is not None and rel.region_lo is not None:
                # PLACE-specific: the validated support point must lie in the
                # region it claims to come from (consistency)
                p = [float(v) for v in point]
                if not all(
                    rel.region_lo[i] - _RELATION_SLACK
                    <= p[i] <= rel.region_hi[i] + _RELATION_SLACK
                    for i in range(3)
                ):
                    relation["point_consistent"] = False
                    return False
                # PLACE-specific: settle-drift bound around the validated
                # point (object xy centre)
                obj = backend.entity_aabb(held)
                if obj is None:
                    return False
                cx = (float(obj[0][0]) + float(obj[1][0])) / 2.0
                cy = (float(obj[0][1]) + float(obj[1][1])) / 2.0
                drift = math.hypot(cx - p[0], cy - p[1])
                relation["xy_drift_m"] = round(drift, 4)
                if drift > _PLACE_POINT_XY_TOL:
                    return False
            relation["point_consistent"] = True
            return True

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
            details={"held": held, "receptacle": receptacle, "realized": True,
                     "relation": dict(relation)},
        )
