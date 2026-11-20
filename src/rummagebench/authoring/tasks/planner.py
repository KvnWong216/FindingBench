"""Deterministic planning: seed + TierSpec + SceneSlots + Priors -> TaskPlan.

Tier-independent machinery: candidate-slot derivation, geometry fit,
lookalike counting, SearchStructureCertificate re-derivation and the seeded
retry loop. Tier rules live in tiers/<tier>.py and only propose plans;
certificate_for() recomputes the search structure from the assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from rummagebench.authoring.level1.rng import stage_seed
from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.eligibility import ObjectEligibility
from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
from rummagebench.authoring.tasks.plan import TaskPlan
from rummagebench.authoring.tasks.priors import PlacementPriors
from rummagebench.authoring.tasks.search_certificate import (
    SearchStructureCertificate, minimum_required_interactions, navigation_lower_bound)
from rummagebench.authoring.tasks.slots import SceneSlots, Slot
from rummagebench.authoring.tasks.tiers.base import TierSpec
from rummagebench.core.public_types import AgentProtocolConfig


class PlanReject(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class PlannerContext:
    tier: TierSpec
    scenes: dict[str, SceneSlots]
    priors: PlacementPriors
    eligibility: ObjectEligibility
    overlays: dict[str, SlotEmbodimentOverlay]  # scene -> overlay (one robot)
    provenance_base: dict[str, Any] = field(default_factory=dict)
    protocol: AgentProtocolConfig = field(default_factory=AgentProtocolConfig)
    # per-context memo of tier-rule enumerations (pure functions of the
    # inputs above); never copied by dataclasses.replace
    cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # scene -> RoomMap (route waypoints, as the certifier uses); optional
    room_maps: dict[str, Any] = field(default_factory=dict)
    # batch steering (e.g. a non-kitchen top-up batch): room types never drawn
    exclude_room_types: frozenset[str] = frozenset()

    @property
    def max_move_m(self) -> float:
        return self.protocol.max_move_cm / 100.0

    @property
    def max_turn_deg(self) -> float:
        return self.protocol.max_turn_deg

    def overlay(self, scene: str) -> Optional[SlotEmbodimentOverlay]:
        ov = self.overlays.get(scene)
        if ov is None or not ov.probed:
            return None
        return ov


# ---------------------------------------------------------------------------
# seeded streams
# ---------------------------------------------------------------------------

def stream(seed: int, index: int, stage: str, retry: int) -> np.random.Generator:
    """Independent generator per (seed, episode, stage, retry) — never
    shared across stages, so one stage's draws cannot perturb another."""
    return np.random.default_rng(stage_seed(seed, index, stage, retry))


def choice(rng: np.random.Generator, items: Sequence[Any]) -> Any:
    if not items:
        raise ValueError("choice from empty sequence")
    return items[int(rng.integers(len(items)))]


def weighted_choice(rng: np.random.Generator, items: Sequence[Any],
                    weights: Sequence[float]) -> Any:
    w = np.asarray(weights, dtype=float)
    if len(items) == 0 or w.sum() <= 0:
        raise ValueError("weighted choice from empty / zero-weight set")
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def randint_bound(rng: np.random.Generator, lo: float, hi: float) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


# ---------------------------------------------------------------------------
# geometry: CPU approximation from slot AABBs
# ---------------------------------------------------------------------------

MAX_UPRIGHT_ASPECT_ON_TOP = 2.5


def fits(bbox: Sequence[float], slot: Slot) -> bool:
    """Object fits slot with a free yaw: horizontal dims sorted against the
    slot opening, height against the usable height (inside only)."""
    ox, oy = sorted(bbox[:2], reverse=True)
    sx, sy = sorted(slot.geometry.opening_size, reverse=True)
    if ox > sx or oy > sy:
        return False
    # tall, narrow items topple off surfaces: Stage 5 on_top placements with
    # height / shorter footprint side > 2.5 failed verification 47% of the
    # time (spray / detergent bottles), <= 2.5 under 1% (A.8.4)
    if slot.relation == "on_top" and bbox[2] > MAX_UPRIGHT_ASPECT_ON_TOP * oy:
        return False
    if slot.relation == "inside" and bbox[2] > slot.geometry.usable_extent[2]:
        return False
    return True


def occupancy(bboxes: Iterable[Sequence[float]], slot: Slot) -> float:
    """Fill ratio (inside: volume) or area ratio (on_top: footprint)."""
    if slot.relation == "inside":
        cap = slot.geometry.usable_volume
        used = sum(b[0] * b[1] * b[2] for b in bboxes)
    else:
        cap = slot.geometry.opening_size[0] * slot.geometry.opening_size[1]
        used = sum(b[0] * b[1] for b in bboxes)
    return float("inf") if cap <= 0 else used / cap


# ---------------------------------------------------------------------------
# candidate slots
# ---------------------------------------------------------------------------

def candidate_slots(ctx: PlannerContext, scene: str, rooms: Iterable[str],
                    category: str, bbox: Sequence[float],
                    region: Optional[str] = None) -> list[Slot]:
    """A slot is a candidate for the target iff (1) an APPROVED placement prior
    covers (category, slot_type, room_type), (2) it lies in the searched area
    — the rooms, narrowed to the landmark region when the instruction names
    one, (3) the embodiment can serve it (overlay), (4) it is not a priori
    impossible — the target geometrically fits."""
    ss = ctx.scenes[scene]
    ov = ctx.overlay(scene)
    rooms = set(rooms)
    members = None if region is None else set(ss.region(region).furniture)
    out = []
    for slot in ss.slots:
        if slot.room not in rooms:
            continue
        if members is not None and slot.parent_entity not in members:
            continue
        if not ctx.priors.is_approved(category, slot.slot_type, slot.room_type):
            continue
        if not fits(bbox, slot):
            continue
        if not slot_usable(ctx, ov, slot):
            continue
        out.append(slot)
    return sorted(out, key=lambda s: s.slot_id)


# MOVE/TURN reachability (CPU, static obstacles) --------------------------------

# the certifier's witness and skills/move.py use a 0.40 m half extent and a
# 0.03 m margin; routes are planned with extra slack so the physics settle
# after every MOVE / TURN (a few cm of drift) cannot turn a planned-feasible
# leg into an UNSAFE one
ROUTE_HALF_EXTENT_M = 0.40
ROUTE_MARGIN_M = 0.03
ROUTE_SLACK_M = 0.02


def route_steps(ctx: PlannerContext, scene: str, room: str, start, goal) -> Optional[int]:
    """MOVE/TURN actions from start Pose2 to goal Pose2 among the scene's
    measured static obstacles (every slot furniture closed); None when no
    route exists. Raises KeyError when the scene has no measured obstacles."""
    from rummagebench.authoring.tasks.navigation import FootprintChecker, ProtocolLimits, plan_route

    ss = ctx.scenes[scene]
    if not ss.footprint_obstacles:
        raise KeyError(scene)
    key = ("route", scene, room, tuple(start.position), tuple(start.orientation),
           tuple(goal.position), tuple(goal.orientation))
    if key in ctx.cache:
        return ctx.cache[key]
    ck = ("route_checker", scene)
    if ck not in ctx.cache:
        ctx.cache[ck] = FootprintChecker([tuple(b) for b in ss.footprint_obstacles.values()],
                                         ROUTE_HALF_EXTENT_M, ROUTE_MARGIN_M + ROUTE_SLACK_M)
    wk = ("route_waypoints", scene, room)
    if wk not in ctx.cache:
        rm = ctx.room_maps.get(scene)
        ctx.cache[wk] = [] if rm is None else rm.free_points(room, 0.4, 0.25)
    p = ctx.protocol
    lim = ProtocolLimits(p.min_move_cm / 100.0, p.max_move_cm / 100.0,
                         p.min_turn_deg, p.max_turn_deg)
    r = plan_route(ctx.cache[ck], (*start.xy, start.yaw), (*goal.xy, goal.yaw),
                   ctx.cache[wk], lim)
    ctx.cache[key] = None if r is None else r.steps
    return ctx.cache[key]


def slot_usable(ctx: PlannerContext, ov: Optional[SlotEmbodimentOverlay],
                slot: Slot) -> bool:
    if ov is None:
        return False
    inter = ov.slots.get(slot.slot_id)
    return inter is not None and inter.usable(slot.requires_open)


def slot_placeable(ctx: PlannerContext, ov: Optional[SlotEmbodimentOverlay],
                   slot: Slot) -> bool:
    """The probe could physically place an object in this slot (builder
    placement + relation/link verification)."""
    inter = None if ov is None else ov.slots.get(slot.slot_id)
    if inter is None or not inter.probed:
        return False
    return bool((inter.evidence or {}).get("placed"))


def target_slot_ok(ctx: PlannerContext, ov: Optional[SlotEmbodimentOverlay],
                   slot: Slot, mode: str) -> bool:
    """Extra condition on the TARGET slot (not on candidates): the revealed
    probe object was visible from the interaction pose; for in_container,
    the probe must also have shown OPEN to be a causal reveal
    (invisible and ungraspable while closed). A no-top cabinet whose
    interior is reachable from above stays a search candidate but can never
    hold an in_container target."""
    inter = None if ov is None else ov.slots.get(slot.slot_id)
    # the agent must SEE the revealed target from a pose it can GRASP from
    # (probe object, benchmark threshold; the real target is certified on GPU)
    gate = ctx.tier.gates.modes.get(mode)
    seen = None if inter is None else inter.reveal_visibility
    if inter is not None and inter.probed and inter.reveal_anchor is None:
        return False  # no pose where the revealed target is seen AND graspable
    if seen is None:
        return False
    if gate is not None and seen < gate.revealed_visibility_min:
        return False
    if mode != "in_container" or not slot.requires_open:
        return True
    causal = None if inter is None else inter.open_causal
    return causal is True


def lookalike_count(ctx: PlannerContext, plan: TaskPlan) -> int:
    el = ctx.eligibility
    tgt = plan.target.category
    n = sum(1 for o in plan.distractors if el.lookalike(tgt, o.category))
    ss = ctx.scenes[plan.scene]
    for room in plan.search_rooms:
        for cat in ss.native_objects.get(room, []):
            if cat == tgt or (cat in el.categories and el.lookalike(tgt, cat)):
                n += 1
    return n


def certificate_for(ctx: PlannerContext, plan: TaskPlan) -> SearchStructureCertificate:
    """Re-derive the search structure from the assets (not from plan claims)."""
    ss = ctx.scenes[plan.scene]
    slots = ss.by_id()
    tslot = slots.get(plan.target.slot_id)
    if tslot is None:
        raise PlanReject(gates.INVALID_SLOT, f"unknown target slot {plan.target.slot_id}")
    bbox = ctx.eligibility.categories[plan.target.category].model(plan.target.model).bbox
    if plan.search_region is not None:
        try:
            region = ss.region(plan.search_region)
        except KeyError as e:
            raise PlanReject(gates.INVALID_SLOT, str(e))
        if not region.usable or region.room not in plan.search_rooms:
            raise PlanReject(gates.INVALID_SLOT,
                             f"region {region.region_id} is not usable (ambiguous)")
    cands = candidate_slots(ctx, plan.scene, plan.search_rooms, plan.target.category,
                            bbox, plan.search_region)
    rooms_with = sorted({s.room for s in cands})
    nav = navigation_lower_bound_for(ctx, plan, tslot)
    open_depth = 1 if tslot.requires_open else 0
    rearr = int(plan.expected.get("rearrangement_depth", 0))
    return SearchStructureCertificate(
        tier_proposed=plan.tier, mode=plan.mode, room_count=len(rooms_with),
        candidate_slot_count=len(cands),
        containment_depth=1 if tslot.relation == "inside" else 0,
        open_depth=open_depth, rearrangement_depth=rearr,
        lookalike_count=lookalike_count(ctx, plan),
        target_slot_type=tslot.slot_type, navigation_lower_bound=nav,
        minimum_required_interactions=minimum_required_interactions(open_depth, rearr, nav),
        candidate_slots=[s.slot_id for s in cands], rooms=rooms_with)


def navigation_lower_bound_for(ctx: PlannerContext, plan: TaskPlan, tslot: Slot) -> int:
    ov = ctx.overlays.get(plan.scene)
    inter = None if ov is None else ov.slots.get(tslot.slot_id)
    if inter is None:
        raise PlanReject(gates.INVALID_SLOT, f"target slot {tslot.slot_id} has no "
                                             "navigation anchor in the overlay")
    from rummagebench.authoring.tasks.embodiment_slots import Pose2
    start = Pose2.from_dict(plan.start)
    goal = inter.navigation_anchor
    return navigation_lower_bound(start.xy, start.yaw, goal.xy, goal.yaw,
                                  ctx.max_move_m, ctx.max_turn_deg)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

@dataclass
class PlanOutcome:
    plan: Optional[TaskPlan]
    certificate: Optional[SearchStructureCertificate]
    attempts: list[dict[str, Any]]  # one record per rejected attempt


ProposeFn = Callable[[PlannerContext, int, int, int], TaskPlan]


def tier_rules(tier: str) -> ProposeFn:
    if tier == "easy":
        from rummagebench.authoring.tasks.tiers.easy import propose
        return propose
    if tier == "medium":
        from rummagebench.authoring.tasks.tiers.medium import propose
        return propose
    raise ValueError(f"unknown tier {tier!r}")


def plan_episode(ctx: PlannerContext, seed: int, index: int,
                 max_attempts: int = 8) -> PlanOutcome:
    """Propose -> CPU gates; retry with a new retry-indexed stream on reject.
    Deterministic in (seed, index, inputs)."""
    if not ctx.tier.planner_enabled:
        raise RuntimeError(f"{ctx.tier.name}: planner disabled in the tier spec")
    propose = tier_rules(ctx.tier.tier)
    attempts: list[dict[str, Any]] = []
    for retry in range(max_attempts):
        try:
            plan = propose(ctx, seed, index, retry)
            cert = certificate_for(ctx, plan)
            rejections = gates.cpu_plan_gates(ctx, plan, cert)
        except PlanReject as r:
            attempts.append({"retry": retry, "codes": [r.code], "detail": [r.detail]})
            continue
        if rejections:
            attempts.append({"retry": retry, "codes": [r.code for r in rejections],
                             "detail": [r.detail for r in rejections]})
            continue
        return PlanOutcome(plan, cert, attempts)
    return PlanOutcome(None, None, attempts)


# ---------------------------------------------------------------------------
# asset loading
# ---------------------------------------------------------------------------

def load_context(tier_name: str, scenes: Sequence[str], robot_id: str,
                 assets_dir, generated_dir,
                 behavior_assets_root=None) -> PlannerContext:
    """Load hand-written / probed inputs from assets_dir and the regenerable
    ones (scene slots, priors, eligibility) from generated_dir; record every
    input's hash."""
    import importlib.metadata as md
    from pathlib import Path

    from rummagebench.authoring.tasks.embodiment_slots import overlay_path
    from rummagebench.authoring.tasks.plan import git_state, sha256_path
    from rummagebench.authoring.tasks.tiers import tier_spec_path
    from rummagebench.authoring.tasks.tiers.base import load_tier_spec

    assets_dir, generated_dir = Path(assets_dir), Path(generated_dir)
    tier_path = tier_spec_path(tier_name)
    priors_path = generated_dir / "priors" / "placement_v1.yaml"
    elig_path = generated_dir / "priors" / "eligibility_v1.yaml"
    scene_slots, overlays, ss_hash, ov_hash = {}, {}, {}, {}
    for scene in scenes:
        p = generated_dir / "scenes" / f"{scene}.yaml"
        scene_slots[scene] = SceneSlots.load(p)
        ss_hash[scene] = sha256_path(p)
        op = overlay_path(assets_dir, scene, robot_id)
        if op.exists():
            overlays[scene] = SlotEmbodimentOverlay.load(op)
            ov_hash[scene] = sha256_path(op)
    try:
        og_version = md.version("omnigibson")
    except md.PackageNotFoundError:
        og_version = None
    behavior_version = None
    if behavior_assets_root is not None:
        vf = Path(behavior_assets_root) / "VERSION"
        behavior_version = vf.read_text().strip() if vf.exists() else None
    repo = Path(__file__).resolve().parents[4]
    base = {
        "generator": git_state(repo),
        "inputs": {"tier_spec_hash": sha256_path(tier_path),
                   "placement_priors_hash": sha256_path(priors_path),
                   "eligibility_hash": sha256_path(elig_path)},
        "scene_slots_hashes": ss_hash,
        "overlay_hashes": ov_hash,
        "environment": {"behavior_assets_version": behavior_version,
                        "omnigibson_version": og_version},
    }
    room_maps = {}
    if behavior_assets_root is not None:
        from rummagebench.authoring.tasks.room_map import RoomMap
        root = Path(behavior_assets_root)
        for scene in scenes:
            if scene_slots[scene].footprint_obstacles and (root / "scenes" / scene).exists():
                room_maps[scene] = RoomMap.load(root / "scenes" / scene,
                                                root / "metadata" / "room_categories.txt")
    tier = load_tier_spec(tier_path)
    eligibility = ObjectEligibility.load(elig_path)
    if tier.lookalike_groups:
        eligibility = eligibility.with_lookalike_groups(tier.lookalike_groups)
    return PlannerContext(tier=tier, scenes=scene_slots,
                          priors=PlacementPriors.load(priors_path),
                          eligibility=eligibility,
                          overlays=overlays, provenance_base=base,
                          room_maps=room_maps)
