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


@dataclass
class GoalCheckResult:
    satisfied: bool
    predicates: list[dict[str, Any]] = field(default_factory=list)


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
            met = _check_placement(backend, world_state, entity, g.receptacle, g.kind)
            results.append({
                "kind": g.kind, "entity": entity, "receptacle": g.receptacle,
                "satisfied": met,
            })
        else:  # pragma: no cover - schema restricts kinds
            results.append({"kind": g.kind, "entity": entity, "satisfied": False})
        ok = ok and results[-1]["satisfied"]
    return GoalCheckResult(satisfied=ok, predicates=results)


def _check_placement(backend, world_state, entity: str, receptacle: str,
                     kind: str) -> bool:
    aabb = backend.entity_aabb(entity)
    box = backend.entity_aabb(receptacle)
    if aabb is None or box is None:
        return False
    (alo, ahi), (blo, bhi) = aabb, box
    if kind == "on_top":
        # Explicit support judgement (round-2 hardening): xy overlap AND the
        # object's bottom RESTING on the support plane within a bidirectional
        # tolerance band. Merely being above the surface (floating) or sunk
        # into it (penetrating) is not support, and a still-held object
        # cannot rest anywhere.
        xy_overlap = (
            alo[0] < bhi[0] - _CONTAIN_SLACK and ahi[0] > blo[0] + _CONTAIN_SLACK
            and alo[1] < bhi[1] - _CONTAIN_SLACK and ahi[1] > blo[1] + _CONTAIN_SLACK
        )
        resting = abs(alo[2] - bhi[2]) <= _ON_TOP_Z_TOL + 1e-9
        released = getattr(world_state, "held_object", None) != entity
        return bool(xy_overlap and resting and released)
    # inside: entity AABB within the receptacle AABB (slack-tolerant)
    return bool(
        alo[0] >= blo[0] - _CONTAIN_SLACK and ahi[0] <= bhi[0] + _CONTAIN_SLACK
        and alo[1] >= blo[1] - _CONTAIN_SLACK and ahi[1] <= bhi[1] + _CONTAIN_SLACK
        and alo[2] >= blo[2] - _CONTAIN_SLACK and ahi[2] <= bhi[2] + _CONTAIN_SLACK
    )


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
