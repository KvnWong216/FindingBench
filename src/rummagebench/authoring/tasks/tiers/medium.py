"""Medium tier rules (TASK_TIERS_PLAN §9-§11, §29): find the location, then
uncover the target.

The target is a FLAT object hidden under 1-2 flat COVER objects with a
larger footprint (the builder stacks each cover pose-level on the previous
stack item, backend._stack_on). The agent must reach the right slot,
recognise the target under the covers, move the covers away (GRASP + PLACE
on any reachable support surface), then GRASP the target. At least one
lookalike of the target is in the room (a cover may itself be one). The
rearrangement depth is a CLAIM: the GPU certifier measures it (target not
graspable from any reachable viewpoint before, graspable after removing the
covers), §42.1.

Modes:
  buried   the stack is in a drawer / cabinet the probe proved to be a causal
           OPEN reveal (containment depth 1, as easy in_container)
  covered  the stack is on a counter / table / desk (containment depth 0,
           the start pose is out of reach as easy on_surface)

The search scope is the room or, when the room alone is too wide, a landmark
region the instruction names (as easy).

GPU findings that shaped this (runs/medium_probe/, A.10): flat-on-flat
stacks survive settling (folder on notebook: target visibility 0.34, GRASP
COLLISION; covers removed: 1.0 / feasible); covers on small or round items
slide off; 5 random items in a drawer neither cover a small target nor fit a
kitchen drawer (it pushes the neighbour open).
"""
from __future__ import annotations

from typing import Any

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.plan import PlacedObject, TaskPlan
from rummagebench.authoring.tasks.planner import (
    PlanReject, PlannerContext, candidate_slots, choice, fits, navigation_lower_bound_for,
    randint_bound, route_steps, slot_placeable, slot_usable, stream, target_slot_ok,
    weighted_choice)
from rummagebench.authoring.tasks.search_certificate import minimum_required_interactions
from rummagebench.authoring.tasks.slots import Slot

# flat = lies stably under / on another flat object
FLAT_MAX_HEIGHT_M = 0.05
FLAT_MIN_SIDE_M = 0.12
# the TARGET must be thin: pilot 1 (A.10) bowls (rim 0.03-0.06 m) stayed
# visible / side-graspable under a plate; thin boards / books were hidden.
# It need not be wide (keys, a spoon, a calculator under a plate / tray /
# folder): only covers must be flat and wide (A.12)
TARGET_MAX_HEIGHT_M = 0.035
TARGET_MIN_SIDE_M = 0.03
# a cover must overhang the target by this much on every side (sorted
# footprints; the builder aligns the long axes): GRASP's side candidates put
# the tool 0.06 m (STANDOFF_M) out from a target face, so a shorter overhang
# leaves a side grasp under the cover open — the full-information oracle
# then grasps the target without moving anything (A.12 pilot: plate under a
# 1.3x platter, 2/3 side-graspable). 0.08 = standoff + gripper margin
COVER_MIN_OVERHANG_M = 0.08
# concave targets never stay covered: pilot 1 (A.10) 4/4 bowls under a plate
# were fully visible and graspable after settling (the cover slides off the rim)
CONCAVE_TARGET_CATEGORIES = frozenset({"bowl", "mixing_bowl", "casserole", "saucepot",
                                       "stockpot", "frying_pan"})


def is_flat(bbox) -> bool:
    return bbox[2] <= FLAT_MAX_HEIGHT_M and min(bbox[0], bbox[1]) >= FLAT_MIN_SIDE_M


def is_thin_target(bbox) -> bool:
    return bbox[2] <= TARGET_MAX_HEIGHT_M and min(bbox[0], bbox[1]) >= TARGET_MIN_SIDE_M


def covers_target(cover_bbox, target_bbox) -> bool:
    c = sorted(cover_bbox[:2], reverse=True)
    t = sorted(target_bbox[:2], reverse=True)
    return all(ci - ti >= 2 * COVER_MIN_OVERHANG_M for ci, ti in zip(c, t))


MODES = ("buried", "covered")


def _rooms(ctx: PlannerContext) -> list[tuple[str, str]]:
    return [(scene, room) for scene in sorted(ctx.scenes)
            if ctx.overlay(scene) is not None
            for room in sorted(ctx.overlay(scene).start_poses)
            if ctx.overlay(scene).start_poses[room]
            and ctx.scenes[scene].rooms.get(room) not in ctx.exclude_room_types]


def _pick_start(ctx: PlannerContext, scene: str, room: str, mode: str, tslot: Slot, rng):
    """A start pose with a MOVE/TURN route to the target furniture; covered:
    also out of arm's reach of it (the uncovering is not done from the
    start, as easy on_surface). Starts are tried in a seeded random order and
    routed lazily (a failed route search is the planner's dominant cost)."""
    from rummagebench.authoring.tasks.tiers.easy import ON_SURFACE_START_CLEARANCE_M, rect_distance

    ss = ctx.scenes[scene]
    ov = ctx.overlay(scene)
    starts = list(ov.start_poses[room])
    if mode == "covered":
        box = ss.footprint_obstacles.get(tslot.parent_entity)
        if box is not None:
            starts = [p for p in starts
                      if rect_distance(p.pose.xy, box) >= ON_SURFACE_START_CLEARANCE_M]
    goal = ov.slots[tslot.slot_id].navigation_anchor
    for i in rng.permutation(len(starts)).tolist():
        p = starts[i]
        if not ss.footprint_obstacles or route_steps(ctx, scene, room, p.pose, goal) is not None:
            return p
    return None


def propose(ctx: PlannerContext, seed: int, index: int, retry: int) -> TaskPlan:
    spec = ctx.tier
    unknown = sorted(set(spec.modes) - set(MODES))
    if unknown:
        raise PlanReject(gates.TIER_GATE_FAILED, f"{spec.name}: unknown modes {unknown}")
    # draw only among modes / rooms that can host a task at all
    viable: dict[str, dict[str, list[str]]] = {}
    for mode in sorted(spec.modes):
        if not spec.modes[mode]:
            continue
        for scene, room in _rooms(ctx):
            if _target_options(ctx, scene, room, mode):
                viable.setdefault(mode, {}).setdefault(scene, []).append(room)
    if not viable:
        raise PlanReject(gates.INVALID_PRIOR,
                         "[medium]: no room has a flat target + cover + lookalike option")
    modes = sorted(viable)
    mode = weighted_choice(stream(seed, index, "mode", retry), modes,
                           [spec.modes[m] for m in modes])
    rng = stream(seed, index, "room", retry)
    scene = choice(rng, sorted(viable[mode]))
    room = choice(rng, viable[mode][scene])
    ss = ctx.scenes[scene]
    ov = ctx.overlay(scene)
    room_type = ss.rooms[room]
    options = _target_options(ctx, scene, room, mode)

    def w(c, s):
        return ctx.priors.get(c, s.slot_type, room_type).generation_weight
    syn = ctx.priors.synset_of
    synsets = sorted({syn(o["category"]) for o in options})
    syn_w = [max(w(o["category"], o["slot"]) for o in options if syn(o["category"]) == y)
             for y in synsets]
    rng = stream(seed, index, "target", retry)
    chosen = weighted_choice(rng, synsets, syn_w)
    cat = choice(rng, sorted({o["category"] for o in options if syn(o["category"]) == chosen}))
    mine = [o for o in options if o["category"] == cat]
    opt = weighted_choice(rng, mine, [w(cat, o["slot"]) for o in mine])
    tslot: Slot = opt["slot"]
    scope = opt["scope"]
    region_id = None if scope is None else scope.region_id
    model = opt["model"]
    tbox = ctx.eligibility.categories[cat].model(model).bbox
    target = PlacedObject(entity=f"target_{cat}", category=cat, model=model,
                          slot_id=tslot.slot_id, role="target")

    # covers: stacked on the target in plan order (compile: each cover
    # on_top the previous stack item); inside a container the stack height
    # must fit the slot
    rng = stream(seed, index, "covers", retry)
    cb = spec.covers
    n_cov = randint_bound(rng, cb.min, cb.max)
    covers: list[PlacedObject] = []
    height = tbox[2]
    for _ in range(n_cov):
        pool = [(c, m) for c, m in opt["covers"]
                if tslot.relation != "inside"
                or height + ctx.eligibility.categories[c].model(m).bbox[2]
                <= tslot.geometry.usable_extent[2]]
        if not pool:
            break
        c, m = choice(rng, pool)
        height += ctx.eligibility.categories[c].model(m).bbox[2]
        covers.append(PlacedObject(entity="", category=c, model=m,
                                   slot_id=tslot.slot_id, role="cover"))
    if len(covers) < max(1, int(cb.min)):
        raise PlanReject(gates.GEOMETRY_OVERFLOW,
                         f"no cover stack fits {tslot.slot_id} over {cat}")

    from rummagebench.authoring.tasks.tiers.easy import _distractors
    background = [o for o in _distractors(ctx, seed, index, retry, scene, room, tslot, target)
                  if o.slot_id != tslot.slot_id]  # the target slot holds only the stack

    # lookalikes: covers may already be; else one in another room slot
    el = ctx.eligibility
    n_look = sum(1 for o in covers if el.lookalike(cat, o.category))
    extra: list[PlacedObject] = []
    if n_look < spec.search.lookalike_count.min:
        rng = stream(seed, index, "lookalike", retry)
        used = {o.slot_id for o in background}
        pool = []
        for s in ss.slots_in_room(room):
            if s.slot_id == tslot.slot_id or not slot_placeable(ctx, ov, s):
                continue
            for c in opt["lookalikes"]:
                if not ctx.priors.is_approved(c, s.slot_type, room_type):
                    continue
                for m in el.categories[c].models:
                    if fits(m.bbox, s):
                        pool.append((s.slot_id not in used, s.slot_id, c, m.model_id))
        if not pool:
            raise PlanReject(gates.TIER_GATE_FAILED,
                             f"no lookalike of {cat} can be placed in {room}")
        free = [p for p in pool if p[0]] or pool
        _, sid, c, m = choice(rng, sorted(free))
        extra.append(PlacedObject(entity="", category=c, model=m, slot_id=sid,
                                  role="lookalike"))
    distractors = covers + extra + background
    if spec.max_distractors is not None:
        distractors = distractors[:spec.max_distractors]
    for i, o in enumerate(distractors):
        o.entity = f"obj_{i:02d}_{o.category}"

    start = _pick_start(ctx, scene, room, mode, tslot, stream(seed, index, "start", retry))
    if start is None:
        raise PlanReject(gates.START_UNREACHABLE,
                         f"no start pose in {room} has a MOVE/TURN route to "
                         f"{tslot.parent_entity} (out of reach for {mode})")
    if scope is None:
        templates = spec.instructions.templates if spec.instructions else [
            "Find the {object} in the {room}."]
    else:
        templates = (spec.instructions.region_templates if spec.instructions else []) or [
            "Find the {object}. It is right next to the {landmark} in the {room}."]
    instruction = choice(stream(seed, index, "instruction", retry), templates).format(
        object=cat.replace("_", " "), room=room_type.replace("_", " "),
        landmark=None if scope is None else scope.name)
    rearr = len(covers)
    open_depth = 1 if tslot.requires_open else 0
    plan = TaskPlan(
        tier=spec.tier, tier_version=spec.version, mode=mode, seed=int(seed),
        episode_index=int(index), scene=scene, room=room, room_type=room_type,
        robot_id=ov.robot_id, target=target, distractors=distractors,
        start_pose=start.name, start=start.pose.to_anchor(), instruction=instruction,
        search_rooms=[room], search_region=region_id,
        candidate_slots=[s.slot_id for s in candidate_slots(ctx, scene, [room], cat, tbox,
                                                            region_id)],
        expected={"rearrangement_depth": rearr, "open_depth": open_depth,
                  "containment_depth": 1 if tslot.relation == "inside" else 0,
                  "lookalike_count": n_look + len(extra)},
        provenance={})
    nav = navigation_lower_bound_for(ctx, plan, tslot)
    plan.expected["navigation_lower_bound"] = nav
    plan.expected["minimum_required_interactions"] = minimum_required_interactions(
        open_depth, rearr, nav)
    from rummagebench.authoring.tasks.tiers.easy import _provenance
    plan.provenance = _provenance(ctx, plan)
    return plan


def _target_options(ctx: PlannerContext, scene: str, room: str, mode: str
                    ) -> list[dict[str, Any]]:
    """Flat target (category, model, slot, scope) options of one room for
    ``mode``: a causal-reveal container slot (buried) or a surface slot
    (covered), candidate width in range for the scope, >= 1 fitting cover
    and a lookalike approved somewhere in the room. Memoized on the context."""
    key = ("medium_options", scene, room, mode)
    if key in ctx.cache:
        return ctx.cache[key]
    ss = ctx.scenes[scene]
    ov = ctx.overlay(scene)
    room_type = ss.rooms[room]
    el = ctx.eligibility
    width = ctx.tier.search.candidate_slot_count
    room_slots = ss.slots_in_room(room)
    scopes: list = [None] + [r for r in ss.regions if r.room == room and r.usable]
    want_container = mode == "buried"
    reveal_mode = "in_container" if want_container else "on_surface"
    out: list[dict[str, Any]] = []
    widths: dict[tuple, int] = {}

    def width_of(cat, bbox, scope) -> int:
        k = (cat, tuple(bbox), None if scope is None else scope.region_id)
        if k not in widths:
            widths[k] = len(candidate_slots(ctx, scene, [room], cat, bbox, k[2]))
        return widths[k]

    for slot in room_slots:
        if slot.is_container != want_container or slot.slot_id not in ov.slots:
            continue
        if (not slot_usable(ctx, ov, slot) or not target_slot_ok(ctx, ov, slot, reveal_mode)
                or not slot_placeable(ctx, ov, slot)):
            continue
        approved = [c for c in sorted(el.categories)
                    if ctx.priors.is_approved(c, slot.slot_type, room_type)]
        for cat in approved:
            if cat in CONCAVE_TARGET_CATEGORIES:
                continue
            looks = [c for c in sorted(el.categories) if c != cat and el.lookalike(cat, c)
                     and any(ctx.priors.is_approved(c, s.slot_type, room_type)
                             for s in room_slots)]
            for m in el.categories[cat].models:
                if not is_thin_target(m.bbox) or not fits(m.bbox, slot):
                    continue
                covers = [(c, cm.model_id) for c in approved
                          if c != cat and c not in CONCAVE_TARGET_CATEGORIES
                          for cm in el.categories[c].models
                          if is_flat(cm.bbox) and fits(cm.bbox, slot)
                          and covers_target(cm.bbox, m.bbox)
                          and (slot.relation != "inside"
                               or m.bbox[2] + cm.bbox[2] <= slot.geometry.usable_extent[2])]
                if not covers:
                    continue
                if not looks and not any(el.lookalike(cat, c) for c, _ in covers):
                    continue
                for scope in scopes:
                    if scope is not None and slot.parent_entity not in set(scope.furniture):
                        continue
                    if not width.contains(width_of(cat, m.bbox, scope)):
                        continue
                    out.append({"category": cat, "model": m.model_id, "slot": slot,
                                "scope": scope, "covers": covers, "lookalikes": looks})
                    break  # the widest in-range scope (room first)
    ctx.cache[key] = out
    return out
