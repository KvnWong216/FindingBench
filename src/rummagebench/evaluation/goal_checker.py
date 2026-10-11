"""Shared goal / temporal condition evaluation (remediation R04).

ONE implementation consumed by REPORT_DONE (skills/report_done.py), the
session termination path and the metrics export — report, goal checking and
scoring can no longer diverge.

Terminal-state predicates are evaluated against the benchmark-owned state
plus the backend's open-state/geometry queries:
    holding:  world_state.held_object == entity
    open:     backend.is_open(entity)
    closed:   not backend.is_open(entity)
    on_top:   the entity rests ON the receptacle's support plane — xy overlap,
              bottom within a bidirectional tolerance band of the support top
              (floating above or penetrating the surface is not support), and
              the entity released (a held object rests nowhere)
    inside:   the entity AABB is contained in the receptacle AABB

Temporal conditions are evaluated over COMMITTED skill events only (the
events the session recorded for successfully executed skills); a missing
required event never satisfies a constraint — absence is not truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# z tolerance when deciding "resting on" (m) and AABB containment slack
_ON_TOP_Z_TOL = 0.06
_CONTAIN_SLACK = 0.02

# region kinds the placement relation supports; anything else (missing
# region, link_surface, ...) is a relation-evaluation fault, never silently
# replaced by the outer entity AABB
SUPPORTED_REGION_KINDS = ("top_surface", "inside_volume")


@dataclass
class PlacementRelationResult:
    """Verdict of the SHARED placement-relation check (round-6).

    Used by BOTH the PLACE postcondition (skills/place.py) and the goal
    predicates below, so one object geometry can never be judged by two
    different rules. ``fault`` distinguishes normal "relation not
    established" from region-data/backend faults: a fault must surface as an
    infrastructure error (PLACE rolls back and raises; REPORT_DONE
    invalidates the run), never as a plain model-facing rejection or a false
    completion.
    """

    kind: str | None          # "top_surface" | "inside_volume" | raw unknown
    satisfied: bool
    fault: str | None         # None | region_missing | region_kind_unsupported | no_object_aabb
    reason: str               # ok | still_held | outside_volume | not_on_plane | no_xy_overlap | kind_mismatch | <fault>
    region_lo: list[float] | None = None
    region_hi: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "satisfied": self.satisfied,
            "fault": self.fault,
            "reason": self.reason,
            "region_lo": self.region_lo,
            "region_hi": self.region_hi,
        }


def check_placement_relation(backend, entity: str, receptacle: str, *,
                             semantic_held: bool | None = None,
                             backend_held: bool | None = None) -> PlacementRelationResult:
    """SHARED placement-relation check (round-6, single implementation).

    Relation and bounds come from ``backend.receptacle_region(receptacle)``
    — the world-frame interior volume (inside) or support plane + valid XY
    footprint (on_top). The outer ``entity_aabb`` of the receptacle is never
    used as a proxy.

    Release state is caller-supplied because the two consumers hold truth in
    different places: the goal check passes BOTH the semantic state
    (``world_state.held_object``) and the backend joint state; the PLACE
    verification passes ONLY the backend state — the semantic release is
    exactly what PLACE verifies afterwards, so consulting it there would be
    circular. ``None`` disables a signal; any provided True counts as
    still-held (both relations unmet, reason ``still_held``).

    Location choice: this lives in evaluation/goal_checker.py (the
    goal/terminal-state evaluator) because the goal predicates are its
    primary consumer; skills/place.py imports it — a one-way
    skills -> evaluation dependency with no import cycle (goal_checker
    imports nothing from skills).
    """
    obj = backend.entity_aabb(entity)
    if obj is None:
        return PlacementRelationResult(None, False, "no_object_aabb",
                                       "no_object_aabb")
    alo, ahi = obj
    region = None
    get_region = getattr(backend, "receptacle_region", None)
    if callable(get_region):
        region = get_region(receptacle)
    kind = getattr(region, "kind", None)
    if kind not in SUPPORTED_REGION_KINDS:
        fault = "region_missing" if region is None else "region_kind_unsupported"
        return PlacementRelationResult(
            kind if isinstance(kind, str) else None, False, fault, fault)
    rlo = [float(v) for v in region.lo]
    rhi = [float(v) for v in region.hi]
    if (semantic_held is True or backend_held is True):
        return PlacementRelationResult(kind, False, None, "still_held", rlo, rhi)
    if kind == "inside_volume":
        contained = _aabb_contained(obj, (rlo, rhi))
        return PlacementRelationResult(kind, contained, None,
                                       "ok" if contained else "outside_volume",
                                       rlo, rhi)
    xy_overlap = (
        float(alo[0]) < rhi[0] and float(ahi[0]) > rlo[0]
        and float(alo[1]) < rhi[1] and float(ahi[1]) > rlo[1]
    )
    on_plane = abs(float(alo[2]) - rhi[2]) <= _ON_TOP_Z_TOL + 1e-9
    ok = bool(xy_overlap and on_plane)
    reason = "ok" if ok else ("no_xy_overlap" if not xy_overlap else "not_on_plane")
    return PlacementRelationResult(kind, ok, None, reason, rlo, rhi)


def _aabb_contained(obj, box) -> bool:
    (alo, ahi), (blo, bhi) = obj, box
    return bool(
        float(alo[0]) >= float(blo[0]) - _CONTAIN_SLACK
        and float(ahi[0]) <= float(bhi[0]) + _CONTAIN_SLACK
        and float(alo[1]) >= float(blo[1]) - _CONTAIN_SLACK
        and float(ahi[1]) <= float(bhi[1]) + _CONTAIN_SLACK
        and float(alo[2]) >= float(blo[2]) - _CONTAIN_SLACK
        and float(ahi[2]) <= float(bhi[2]) + _CONTAIN_SLACK
    )


@dataclass
class GoalCheckResult:
    satisfied: bool
    predicates: list[dict[str, Any]] = field(default_factory=list)
    fault: str | None = None  # placement-region/backend fault (run-invalid), R-统一


@dataclass
class TemporalCheckResult:
    satisfied: bool
    constraints: list[dict[str, Any]] = field(default_factory=list)


def check_goal(world_state, scenario, backend) -> GoalCheckResult:
    """Evaluate scenario.goal (R04 first-batch predicate kinds). Requires a
    data-driven goal spec; the legacy holding(target) default stays in
    skills/report_done.goal_satisfied."""
    goal = list(getattr(scenario, "goal", None) or [])
    results: list[dict[str, Any]] = []
    ok = bool(goal)
    fault: str | None = None
    for g in goal:
        entity = g.entity
        if g.kind == "holding":
            met = world_state.held_object == entity
            results.append({"kind": g.kind, "entity": entity, "satisfied": met})
        elif g.kind == "open":
            met = bool(backend.is_open(entity))
            results.append({"kind": g.kind, "entity": entity, "satisfied": met})
        elif g.kind == "closed":
            met = not backend.is_open(entity)
            results.append({"kind": g.kind, "entity": entity, "satisfied": met})
        elif g.kind in ("on_top", "inside"):
            # round-6: the SHARED relation check (region-authoritative) —
            # identical to the PLACE postcondition's relation verdict. The
            # goal caller supplies both held signals (semantic + backend).
            rel = check_placement_relation(
                backend, entity, g.receptacle,
                semantic_held=(world_state.held_object == entity),
                backend_held=bool(backend.is_holding(entity)))
            requested = "top_surface" if g.kind == "on_top" else "inside_volume"
            if rel.fault is not None:
                met, reason, fault = False, rel.fault, rel.fault
            elif rel.kind != requested:
                # e.g. an inside goal against a closed container whose
                # region is currently top_surface: the relation is not
                # verifiable/established — unmet, NOT a fault
                met, reason = False, "kind_mismatch"
            else:
                met, reason = rel.satisfied, rel.reason
            results.append({
                "kind": g.kind, "entity": entity, "receptacle": g.receptacle,
                "satisfied": met, "reason": reason, "fault": fault,
                "region": {"kind": rel.kind,
                           "lo": rel.region_lo, "hi": rel.region_hi},
            })
            fault = fault or (rel.fault if rel.fault is not None else None)
        else:  # pragma: no cover - schema restricts kinds
            results.append({"kind": g.kind, "entity": entity, "satisfied": False})
            fault = fault or "unsupported_predicate"
        ok = ok and results[-1]["satisfied"]
    return GoalCheckResult(satisfied=ok and fault is None, predicates=results,
                           fault=fault)


def check_temporal(events: list[dict[str, Any]], scenario) -> TemporalCheckResult:
    """Evaluate scenario.temporal over the episode's committed skill events.

    ``events`` are the per-step dicts the session recorded (each carrying
    ``event`` and ``entity`` keys); only committed executions are ever
    appended there.
    """
    constraints_spec = list(getattr(scenario, "temporal", None) or [])
    results: list[dict[str, Any]] = []
    ok = True  # no temporal constraints -> vacuously satisfied
    timeline = [
        (str(e.get("event")), e.get("entity"))
        for e in events or []
        if isinstance(e, dict) and e.get("event")
    ]

    def first_index(name: str, entity: str | None) -> int | None:
        for i, (ev, ent) in enumerate(timeline):
            if ev == name and (entity is None or ent == entity):
                return i
        return None

    for spec in constraints_spec:
        if spec.kind == "required_event":
            idx = first_index(spec.event, spec.entity)
            met = idx is not None
            results.append({
                "kind": spec.kind, "event": spec.event, "entity": spec.entity,
                "satisfied": met, "event_index": idx,
            })
        else:  # before
            first = first_index(spec.event, spec.entity)
            then = first_index(spec.before_event, spec.before_entity)
            # missing required event fails; missing reference event cannot
            # order the required one before it
            met = first is not None and then is not None and first < then
            results.append({
                "kind": spec.kind, "event": spec.event, "entity": spec.entity,
                "before_event": spec.before_event,
                "before_entity": spec.before_entity,
                "satisfied": met,
                "event_index": first, "before_index": then,
            })
        ok = ok and results[-1]["satisfied"]
    return TemporalCheckResult(satisfied=ok, constraints=results)
