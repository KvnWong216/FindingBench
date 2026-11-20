"""Oracle semantic planner (§1): exact shortest semantic plan under
full information.

    C(s)        semantic candidate actions
    Phi_E(s,a)  embodiment-conditioned physical feasibility predicate
    A_E(s)      = {a in C(s) : Phi_E(s,a)=1}
    d*_E(s)     = min # admissible semantic actions to reach the goal

The planner operates on the SAME semantic transition semantics as
BenchmarkSession — it is NOT an independent approximate benchmark:

- candidate enumeration reuses the production object adapters;
- physical feasibility reuses the production FeasibilityValidator through an
  OverlayBackend that projects the hypothetical oracle state onto the real
  backend (only the semantic variables — open states, held object, robot
  anchor — are overlaid; all geometry/articulation data is the production
  data, so admissibility is bit-identical to the production engine);
- the transition function mirrors the skill executors (NAV teleport +
  anchor, OPEN/CLOSE set_open, GRASP/PLACE benchmark-held-object updates);
- the goal predicate mirrors BenchmarkSession's success rule
  (succeed_when_holding_target: held_object == target.entity).

Feasibility results are cached per (anchor, opens, held, skill, target):
for the fixed-anchor benchmark, admissibility depends only on those
variables, so no simulator step is ever launched per BFS node.

The oracle is FULL-INFORMATION: it deliberately ignores agent-visibility
filtering (the agent-visible action graph is a separate, documented view).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import TargetKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OracleWorldState:
    """Canonical, hashable semantic state.

    Contains ONLY the variables that affect transitions: robot anchor,
    open-state set, benchmark-held object, and object location relations.
    Render/physics noise is structurally excluded.
    """

    robot_anchor: str
    open_entities: frozenset[str]
    held_object: str | None
    object_relations: frozenset[tuple[str, str, str]]  # (entity, relation, receptacle)

    def holding(self) -> int:
        return 1 if self.held_object is not None else 0

    def location_of(self, entity: str) -> tuple[str, str] | None:
        """(relation, receptacle) for an entity, if it has one."""
        for entity_name, relation, receptacle in self.object_relations:
            if entity_name == entity:
                return (relation, receptacle)
        return None


@dataclass
class OraclePlanResult:
    solvable: bool
    depth: int | None
    actions: list[dict[str, Any]]
    visited_states: int
    expanded_states: int
    reason: str | None = None


class OverlayBackend:
    """Projects a hypothetical OracleWorldState onto a real SimBackend.

    Delegates everything to the wrapped production backend EXCEPT the
    semantic variables (open states, held object, robot pose) and the
    open-state-dependent geometry, which come from per-open-set snapshots
    realized on the real backend (see OracleFeasibilityModel._realize).
    This guarantees oracle admissibility is exactly production admissibility
    evaluated at the hypothetical state.
    """

    def __init__(self, inner: Any, scenario: ScenarioSpec):
        self._inner = inner
        self._scenario = scenario
        self.current_state: OracleWorldState | None = None
        # per-realization snapshots (open set, displaced objects), realized
        # on the real backend by the model
        self.snapshots: dict[tuple[frozenset[str], frozenset[str]], dict[str, Any]] = {}
        self._initial_relations = frozenset(
            (p.entity, p.relation, p.receptacle)
            for p in getattr(scenario, "placements", None) or [])
        self._anchor_cache = {
            name: anchor for name, anchor in scenario.anchors.items()
        }

    # ---- overlaid semantic variables -------------------------------------

    def is_anchor_validated(self, name, anchor) -> bool:
        return self._inner.is_anchor_validated(name, anchor)

    def is_open(self, entity: str) -> bool:
        if self.current_state is None:
            return self._inner.is_open(entity)
        return entity in self.current_state.open_entities

    def held_count(self) -> int:
        if self.current_state is None:
            return self._inner.held_count()
        return self.current_state.holding()

    def is_holding(self, entity: str) -> bool:
        if self.current_state is None:
            return self._inner.is_holding(entity)
        return self.current_state.held_object == entity

    def robot_pose(self):
        if self.current_state is None:
            return self._inner.robot_pose()
        anchor = self._anchor_cache[self.current_state.robot_anchor]
        return list(anchor.position), list(anchor.orientation)

    # full-information view for the oracle: no visibility filtering
    def visible_entities(self) -> list[str]:
        return self.entity_names()

    # ---- open-state-dependent data (from the active snapshot) ------------

    def displaced(self, state: OracleWorldState) -> frozenset[str]:
        """Objects no longer where the scenario put them: held, or PLACEd
        elsewhere. Empty for every state an easy-tier search reaches (only
        the target is ever grasped, and that is the goal)."""
        moved = {r[0] for r in state.object_relations if r not in self._initial_relations}
        moved |= {r[0] for r in self._initial_relations if r not in state.object_relations}
        if state.held_object is not None:
            moved.add(state.held_object)
        return frozenset(moved)

    def realization_key(self, state: OracleWorldState):
        return (state.open_entities, self.displaced(state))

    def _snapshot(self) -> dict[str, Any] | None:
        if self.current_state is None:
            return None
        return self.snapshots.get(self.realization_key(self.current_state))

    def entity_aabb(self, name: str):
        snap = self._snapshot()
        if snap is not None and name in snap["entity_aabbs"]:
            return snap["entity_aabbs"][name]
        return self._inner.entity_aabb(name)

    def opened_entity_aabb(self, entity: str):
        """Fully-opened AABB for the OPEN base-intrusion check (A.8 F8).
        Delegated to the live counterfactual: without it the feasibility
        engine silently skips the check and the oracle certifies OPENs the
        session rejects (pilot 3, fb_ae154bf9a4ac5128)."""
        fn = getattr(self._inner, "opened_entity_aabb", None)
        return fn(entity) if fn is not None else None

    def collision_geometries(self):
        snap = self._snapshot()
        if snap is not None:
            return snap["collision_geometries"]
        return self._inner.collision_geometries()

    def link_pose(self, entity: str, link: str):
        snap = self._snapshot()
        if snap is not None:
            key = (entity, link)
            if key in snap["link_poses"]:
                return snap["link_poses"][key]
        return self._inner.link_pose(entity, link)

    def link_aabb(self, entity: str, link: str):
        snap = self._snapshot()
        if snap is not None:
            key = (entity, link)
            if key in snap["link_aabbs"]:
                return snap["link_aabbs"][key]
        return self._inner.link_aabb(entity, link)

    def receptacle_region(self, entity: str):
        snap = self._snapshot()
        if snap is not None and entity in snap["receptacle_regions"]:
            return snap["receptacle_regions"][entity]
        return self._inner.receptacle_region(entity)

    # ---- static production data (delegated) ------------------------------

    def entity_names(self) -> list[str]:
        return self._inner.entity_names()

    def describe_entity(self, name: str):
        return self._inner.describe_entity(name)

    def resolve_entity(self, name: str):
        return self._inner.resolve_entity(name)

    def entity_pose(self, name: str):
        return self._inner.entity_pose(name)

    def entity_pose6d(self, name: str):
        return self._inner.entity_pose6d(name)

    def articulation_info(self, entity: str):
        return self._inner.articulation_info(entity)

    def eef_pose(self):
        return self._inner.eef_pose()

    def robot_joint_positions(self):
        return self._inner.robot_joint_positions()

    def settle(self, steps: int = 10) -> None:
        pass  # oracle never steps physics


class OracleFeasibilityModel:
    """Cached Phi_E over (anchor, opens, held, skill, target).

    Evaluates through the production FeasibilityValidator on the
    OverlayBackend. Before the first query at a given open-set, the model
    REALIZES that door configuration on the real backend (set_open + settle)
    and snapshots every open-state-dependent quantity (world collision
    geometry, entity AABBs, openable-link poses, receptacle regions). All
    later queries at that open-set read the snapshot — production admissibility,
    no simulator steps per BFS node.

    Documented approximation: PLACE feasibility is evaluated with the held
    body's grasp offset unknown at plan time (held_offset=None, held body
    excluded from collision). In easy tasks PLACE edges never lie on a
    shortest path to the hold-the-target goal; in rearrangement tiers
    (medium) they do — putting a cover down — and the protocol witness
    executes that PLACE for real, with the held body checked.
    """

    def __init__(self, backend: Any, scenario: ScenarioSpec, feasibility):
        self._overlay = OverlayBackend(backend, scenario)
        self._inner = backend
        self._scenario = scenario
        # clone the production validator onto the overlay: hypothetical-state
        # queries must read the OVERLAY's semantic variables (robot pose,
        # open states, held object), not the live simulator's
        self._feasibility = feasibility.with_backend(self._overlay)
        self._production_feasibility = feasibility
        self._robot = None  # set by caller (RobotEmbodiment)
        self._cache: dict[tuple, bool] = {}
        self._realizing = None

    @property
    def overlay(self) -> OverlayBackend:
        return self._overlay

    def bind_robot(self, robot) -> None:
        self._robot = robot

    def _realize(self, key) -> None:
        """Realize a door configuration (+ displaced objects) on the real
        backend and snapshot all state-dependent quantities.

        Displaced objects (held, or PLACEd elsewhere — only reachable when
        grasping non-targets is not terminal, i.e. rearrangement tiers) are
        moved out of the world for the snapshot: they no longer obstruct
        their old place. Documented approximation: a PLACEd object does not
        obstruct its new place either (optimistic for d*; the protocol
        witness executes the real PLACE)."""
        opens, displaced = key
        if key in self._overlay.snapshots or self._realizing is not None:
            return
        self._realizing = key
        parked: list[tuple[Any, Any]] = []
        try:
            from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

            is_real = isinstance(self._inner, OmniGibsonBackend)
            previous = {
                e: self._inner.is_open(e) for e in self._scenario.initial_states
            }
            changed = any((e in opens) != prev for e, prev in previous.items())
            if is_real and changed:
                for e, prev in previous.items():
                    self._inner.set_open(e, e in opens)
                self._inner.settle(15)
            if is_real and displaced:
                parked = self._inner.park_entities(sorted(displaced))
            snap: dict[str, Any] = {
                "collision_geometries": self._inner.collision_geometries(),
                "entity_aabbs": {
                    e: self._inner.entity_aabb(e) for e in self._inner.entity_names()
                },
                "link_poses": {},
                "link_aabbs": {},
                "receptacle_regions": {},
            }
            for e in self._scenario.initial_states:
                info = self._inner.describe_entity(e)
                if info is None:
                    continue
                art = self._inner.articulation_info(e)
                if art is None:
                    continue
                for link in art.links:
                    try:
                        snap["link_poses"][(e, link)] = self._inner.link_pose(e, link)
                        snap["link_aabbs"][(e, link)] = self._inner.link_aabb(e, link)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("snapshot link %s/%s failed: %s", e, link, exc)
                try:
                    snap["receptacle_regions"][e] = self._inner.receptacle_region(e)
                except Exception as exc:  # pragma: no cover
                    logger.warning("snapshot region %s failed: %s", e, exc)
            self._overlay.snapshots[key] = snap
            # restore the pre-realization door state so the live session is
            # untouched (the next realization toggles again as needed)
            if is_real and parked:
                self._inner.unpark_entities(parked)
                parked = []
            if is_real and changed:
                for e, prev in previous.items():
                    self._inner.set_open(e, prev)
                self._inner.settle(5)
        finally:
            if parked:
                self._inner.unpark_entities(parked)
            self._realizing = None

    def is_feasible(self, state: OracleWorldState, skill: str, target: str) -> bool:
        rkey = self._overlay.realization_key(state)
        key = (state.robot_anchor, tuple(sorted(state.open_entities)),
               state.held_object, tuple(sorted(rkey[1])), skill, target)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if skill != "NAV":
            self._realize(rkey)
        self._overlay.current_state = state
        if skill == "NAV":
            self._cache[key] = target in self._scenario.anchors
            return self._cache[key]
        info = self._overlay.describe_entity(target)
        from rummagebench.sim.base import ResolvedTarget

        resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity=target, info=info)
        verdict = self._feasibility.check(skill, resolved, self._robot, self._overlay_world(state))
        self._cache[key] = bool(verdict.feasible)
        return self._cache[key]

    def _overlay_world(self, state: OracleWorldState):
        from rummagebench.state.benchmark_state import BenchmarkWorldState

        world = BenchmarkWorldState()
        if state.held_object is not None:
            world.grasp(state.held_object, None)
        return world


@dataclass
class _OracleTransition:
    """Semantic transition + candidate enumeration mirroring the production
    benchmark (SkillGrounder adapters + skill executors + session rules)."""

    scenario: ScenarioSpec
    model: OracleFeasibilityModel

    def candidates(self, state: OracleWorldState) -> list[tuple[str, str]]:
        """C(s) = semantic candidates, full-information (visibility-agnostic)."""
        out: list[tuple[str, str]] = []
        for anchor in self.scenario.anchors:
            out.append(("NAV", anchor))
        backend = self.model.overlay
        for name in backend.entity_names():
            info = backend.describe_entity(name)
            if info is None:
                continue
            open_now = name in state.open_entities
            if info.openable:
                out.append(("OPEN" if not open_now else "CLOSE", name))
            if not info.fixed_base and info.graspable and state.held_object is None:
                out.append(("GRASP", name))
            if state.held_object is not None and (
                getattr(info, "is_receptacle", False) or info.openable
            ):
                out.append(("PLACE", name))
        return out

    def feasible(self, state: OracleWorldState, skill: str, target: str) -> bool:
        return self.model.is_feasible(state, skill, target)

    def unsafe(self, state: OracleWorldState, skill: str, target: str) -> bool:
        """Task-level safety policy (mirrors SafetyValidator)."""
        if skill != "GRASP":
            return False
        info = self.model.overlay.describe_entity(target)
        if info is None:
            return True
        safety = self.scenario.safety
        if safety.grasping_fixed_base_unsafe and info.fixed_base:
            return True
        return info.category in safety.forbidden_categories

    def transition(self, state: OracleWorldState, skill: str, target: str) -> OracleWorldState:
        """Deterministic symbolic transition (mirrors the skill executors)."""
        if skill == "NAV":
            return OracleWorldState(
                robot_anchor=target,
                open_entities=state.open_entities,
                held_object=state.held_object,
                object_relations=state.object_relations,
            )
        if skill == "OPEN":
            return OracleWorldState(
                robot_anchor=state.robot_anchor,
                open_entities=state.open_entities | {target},
                held_object=state.held_object,
                object_relations=state.object_relations,
            )
        if skill == "CLOSE":
            return OracleWorldState(
                robot_anchor=state.robot_anchor,
                open_entities=frozenset(e for e in state.open_entities if e != target),
                held_object=state.held_object,
                object_relations=state.object_relations,
            )
        if skill == "GRASP":
            # the held object leaves its location relation (travels with the hand)
            relations = frozenset(
                r for r in state.object_relations if r[0] != target
            )
            return OracleWorldState(
                robot_anchor=state.robot_anchor,
                open_entities=state.open_entities,
                held_object=target,
                object_relations=relations,
            )
        if skill == "PLACE":
            # PLACE(target receptacle): held object moves into/onto receptacle
            held = state.held_object
            relations = frozenset(r for r in state.object_relations if r[0] != held)
            relations = relations | {(held, "placed", target)}
            return OracleWorldState(
                robot_anchor=state.robot_anchor,
                open_entities=state.open_entities,
                held_object=None,
                object_relations=relations,
            )
        raise ValueError(f"unknown skill {skill!r}")


def initial_oracle_state(scenario: ScenarioSpec) -> OracleWorldState:
    """Full-information initial state from the scenario spec."""
    opens = frozenset(
        e for e, st in scenario.initial_states.items() if st.open
    )
    relations = frozenset(
        (p.entity, p.relation, p.receptacle) for p in scenario.placements
    )
    return OracleWorldState(
        robot_anchor=scenario.robot.init_anchor,
        open_entities=opens,
        held_object=None,
        object_relations=relations,
    )


def solve_full_information(
    scenario: ScenarioSpec,
    initial_state: OracleWorldState,
    model: OracleFeasibilityModel,
    max_depth: int = 24,
) -> OraclePlanResult:
    """Exact BFS shortest semantic plan to the task goal under embodiment E.

    Goal (mirrors BenchmarkSession): hold the target entity when
    succeed_when_holding_target is set. GRASP of a non-target with
    fail_on_wrong_grasp terminates the episode as FAIL_WRONG_TARGET and is
    therefore pruned from the search (it cannot lead to SUCCESS).

    No-plan outcomes are distinguished (remediation R11):
    SEARCH_LIMIT_REACHED (the max_depth bound cut off unexplored states —
    NOT evidence of unsolvability) vs EXHAUSTED_ABSTRACT_GRAPH (the finite
    abstract graph was fully explored within the depth bound: the actual
    embodiment-unsolvability proof). A depth-limited run must never claim
    unsolvability.
    """
    if scenario.robot.kinematics and model._robot is None:
        raise ValueError("oracle model has no bound robot embodiment")

    term = scenario.termination
    transition = _OracleTransition(scenario, model)

    def goal(state: OracleWorldState) -> bool:
        if term.succeed_when_holding_target:
            return state.held_object == scenario.target.entity
        return False  # no other success rule defined for search episodes

    start = initial_state
    if goal(start):
        return OraclePlanResult(solvable=True, depth=0, actions=[],
                                visited_states=1, expanded_states=0,
                                reason="goal already satisfied")

    queue: deque[tuple[OracleWorldState, list[dict[str, Any]]]] = deque()
    queue.append((start, []))
    visited = {start}
    visited_count, expanded_count = 1, 0
    # remediation R11: depth truncation is a SEARCH RESOURCE limit, never
    # evidence that the embodiment cannot solve the task
    depth_truncated = False

    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            depth_truncated = True
            continue
        expanded_count += 1
        for skill, target in transition.candidates(state):
            if skill == "GRASP" and target != scenario.target.entity \
                    and term.fail_on_wrong_grasp:
                continue  # terminal FAIL_WRONG_TARGET: cannot lead to SUCCESS
            if transition.unsafe(state, skill, target):
                continue  # terminal FAIL_UNSAFE_ACTION: pruned
            if not transition.feasible(state, skill, target):
                continue
            next_state = transition.transition(state, skill, target)
            if next_state in visited:
                continue
            visited.add(next_state)
            visited_count += 1
            new_path = path + [{"skill": skill, "target": {
                "type": "place" if skill == "NAV" else "entity",
                "value": target,
            }}]
            if goal(next_state):
                return OraclePlanResult(
                    solvable=True,
                    depth=len(new_path),
                    actions=new_path,
                    visited_states=visited_count,
                    expanded_states=expanded_count,
                )
            queue.append((next_state, new_path))

    # remediation R11: distinguish the three no-plan outcomes. A depth
    # truncation means the search ran out of resources (SEARCH_LIMIT_REACHED);
    # only a full exploration of the finite abstract graph within the depth
    # bound may claim UNSOLVABLE_FOR_EMBODIMENT.
    if depth_truncated:
        reason = "SEARCH_LIMIT_REACHED"
    else:
        reason = "EXHAUSTED_ABSTRACT_GRAPH"
    return OraclePlanResult(
        solvable=False, depth=None, actions=[],
        visited_states=visited_count, expanded_states=expanded_count,
        reason=reason,
    )


def build_oracle_model(session) -> tuple[OracleFeasibilityModel, OracleWorldState]:
    """Build a cached oracle model bound to a LIVE production session
    (its backend + feasibility validator are the production ones)."""
    model = OracleFeasibilityModel(session._backend, session.scenario,
                                   session._feasibility)
    model.bind_robot(session.robot)
    return model, initial_oracle_state(session.scenario)
