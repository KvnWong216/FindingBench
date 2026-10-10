"""Easy tier rules: find the location.

Single room (the instruction names it), 2-4 candidate slots, one direct
reveal (OPEN for in_container, viewpoint change for on_surface), zero
rearrangement, zero lookalikes. Every draw uses its own seeded stream.
"""
from __future__ import annotations

from typing import Any

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.plan import PlacedObject, TaskPlan, provenance
from rummagebench.authoring.tasks.planner import (
    PlanReject, PlannerContext, candidate_slots, choice, fits, navigation_lower_bound_for,
    occupancy, randint_bound, route_steps, slot_placeable, slot_usable, stream,
    target_slot_ok, weighted_choice)
from rummagebench.authoring.tasks.search_certificate import minimum_required_interactions
from rummagebench.authoring.tasks.slots import Slot

# Stage 5 + top-up (A.9): every on_surface start found GRASP-feasible on GPU
# (TARGET_PREMATURELY_GRASPABLE, 15 episodes) stood <= 0.97 m from the target
# furniture's footprint; certified ones started up to that close too, so the
# CPU filter keeps a margin over the measured reach
ON_SURFACE_START_CLEARANCE_M = 1.0


def rect_distance(xy, box) -> float:
    """Horizontal distance from a point to an AABB's xy rectangle (0 inside)."""
    lo, hi = box[0], box[1]
    dx = max(lo[0] - xy[0], 0.0, xy[0] - hi[0])
    dy = max(lo[1] - xy[1], 0.0, xy[1] - hi[1])
    return (dx * dx + dy * dy) ** 0.5


def propose(ctx: PlannerContext, seed: int, index: int, retry: int) -> TaskPlan:
    spec = ctx.tier
    modes = sorted(spec.modes)
    mode = weighted_choice(stream(seed, index, "mode", retry), modes,
                           [spec.modes[m] for m in modes])

    rooms = [(scene, room) for scene in sorted(ctx.scenes)
             if ctx.overlay(scene) is not None
             for room in sorted(ctx.overlay(scene).start_poses)
             if ctx.overlay(scene).start_poses[room]
             and ctx.scenes[scene].rooms.get(room) not in ctx.exclude_room_types]
    if not rooms:
        raise PlanReject(gates.INVALID_SLOT, "no room with a robot start pose "
                                             "(probe the embodiment overlay first)")
    # draw only among rooms that can host this mode at all (no wasted
    # INVALID_PRIOR retries), scene first then room, so a scene's share does
    # not scale with how many rooms it has
    viable: dict[str, list[str]] = {}
    widths_all: dict[int, int] = {}
    for scene, room in rooms:
        opts, widths = _target_options(ctx, scene, room, mode)
        for n, k in widths.items():
            widths_all[n] = widths_all.get(n, 0) + k
        if opts:
            viable.setdefault(scene, []).append(room)
    if not viable:
        if not widths_all:
            raise PlanReject(gates.INVALID_PRIOR,
                             f"[{mode}]: no room has an approved, fitting, "
                             "embodiment-usable target slot")
        raise PlanReject(gates.SEARCH_WIDTH_MISMATCH,
                         f"[{mode}]: candidate widths {dict(sorted(widths_all.items()))} "
                         f"outside {spec.search.candidate_slot_count.describe()} in every room")
    rng = stream(seed, index, "room", retry)
    scene = choice(rng, sorted(viable))
    room = choice(rng, viable[scene])
    ss = ctx.scenes[scene]
    ov = ctx.overlay(scene)
    room_type = ss.rooms[room]
    options, _ = _target_options(ctx, scene, room, mode)
    # three-stage draw so a semantic object's share does not scale with how
    # many asset aliases (chopping_board / cutting_board), models (bowl: 44)
    # or slots / scopes it expands into:
    #   synset by its prior generation_weight -> alias category uniformly ->
    #   option weighted by the prior covering its slot
    def w(c, s):
        return ctx.priors.get(c, s.slot_type, room_type).generation_weight
    syn = ctx.priors.synset_of
    synsets = sorted({syn(c) for c, _, _, _ in options})
    syn_w = [max(w(c, s) for c, _, s, _ in options if syn(c) == y) for y in synsets]
    rng = stream(seed, index, "target", retry)
    chosen = weighted_choice(rng, synsets, syn_w)
    cat = choice(rng, sorted({c for c, _, _, _ in options if syn(c) == chosen}))
    mine = [o for o in options if o[0] == cat]
    _, model, tslot, scope = weighted_choice(rng, mine, [w(cat, s) for _, _, s, _ in mine])
    region_id = None if scope is None else scope.region_id
    target = PlacedObject(entity=f"target_{cat}", category=cat, model=model,
                          slot_id=tslot.slot_id, role="target")

    distractors = _distractors(ctx, seed, index, retry, scene, room, tslot, target)

    # start visibility: an in_container target is hidden by the closed
    # container (verified on GPU), so seeing the cabinet from the start is
    # natural; an on_surface target must be outside the start view, which
    # the probe records conservatively as "slot furniture not seen at all"
    starts = ov.start_poses[room]
    hidden = [p for p in starts
              if mode == "in_container" or p.visible_slots is None
              or tslot.slot_id not in p.visible_slots]
    if not hidden:
        raise PlanReject(gates.START_VISIBLE,
                         f"every start pose in {room} sees {tslot.slot_id}")
    # an on_surface target must not be within arm's reach of the start (the
    # reveal is a viewpoint change, not a GRASP from where the robot stands)
    if mode == "on_surface":
        box = ss.footprint_obstacles.get(tslot.parent_entity)
        if box is not None:
            hidden = [p for p in hidden
                      if rect_distance(p.pose.xy, box) >= ON_SURFACE_START_CLEARANCE_M]
            if not hidden:
                raise PlanReject(gates.TARGET_PREMATURELY_GRASPABLE,
                                 f"every start pose in {room} is within "
                                 f"{ON_SURFACE_START_CLEARANCE_M} m of {tslot.parent_entity}")
    # the witness drives from the start to the target furniture's interaction
    # pose by MOVE/TURN only: keep the starts that have a route there (the
    # probe only proved reachability from SOME start position of the room)
    if ctx.scenes[scene].footprint_obstacles:
        goal = ov.slots[tslot.slot_id].navigation_anchor
        hidden = [p for p in hidden
                  if route_steps(ctx, scene, room, p.pose, goal) is not None]
        if not hidden:
            raise PlanReject(gates.START_UNREACHABLE,
                             f"no start pose in {room} has a MOVE/TURN route to "
                             f"{tslot.parent_entity}")
    start = choice(stream(seed, index, "start", retry), hidden)

    if scope is None:
        templates = spec.instructions.templates if spec.instructions else [
            "Find the {object} in the {room}."]
    else:
        templates = (spec.instructions.region_templates if spec.instructions else []) or [
            "Find the {object}. It is right next to the {landmark} in the {room}."]
    instruction = choice(stream(seed, index, "instruction", retry), templates).format(
        object=cat.replace("_", " "), room=room_type.replace("_", " "),
        landmark=None if scope is None else scope.name)

    plan = TaskPlan(
        tier=spec.tier, tier_version=spec.version, mode=mode, seed=int(seed),
        episode_index=int(index), scene=scene, room=room, room_type=room_type,
        robot_id=ov.robot_id, target=target, distractors=distractors,
        start_pose=start.name, start=start.pose.to_anchor(), instruction=instruction,
        search_rooms=[room], search_region=region_id,
        candidate_slots=[s.slot_id for s in candidate_slots(
            ctx, scene, [room], cat,
            ctx.eligibility.categories[cat].model(model).bbox, region_id)],
        expected={"rearrangement_depth": 0,
                  "open_depth": 1 if tslot.requires_open else 0,
                  "containment_depth": 1 if tslot.relation == "inside" else 0,
                  "lookalike_count": 0},
        provenance={})
    nav = navigation_lower_bound_for(ctx, plan, tslot)
    plan.expected["navigation_lower_bound"] = nav
    plan.expected["minimum_required_interactions"] = minimum_required_interactions(
        plan.expected["open_depth"], 0, nav)
    plan.provenance = _provenance(ctx, plan)
    return plan


def _target_options(ctx: PlannerContext, scene: str, room: str, mode: str
                    ) -> tuple[list[tuple[str, str, Slot, Any]], dict[int, int]]:
    """(category, model, target slot, scope) options of one room whose
    candidate width is in range, plus the histogram of every width seen.
    Pure in the context's inputs, memoized on it."""
    key = ("easy_options", scene, room, mode)
    if key in ctx.cache:
        return ctx.cache[key]
    ss = ctx.scenes[scene]
    ov = ctx.overlay(scene)
    room_type = ss.rooms[room]
    width = ctx.tier.search.candidate_slot_count
    # search scopes: the whole room, or a landmark region the instruction
    # names; only unambiguous regions are offered
    scopes: list = [None] + [r for r in ss.regions if r.room == room and r.usable]
    options: list[tuple[str, str, Slot, Any]] = []
    widths_seen: dict[int, int] = {}
    want_container = mode == "in_container"
    for scope in scopes:
        members = None if scope is None else set(scope.furniture)
        for slot in ss.slots_in_room(room):
            if slot.is_container != want_container or slot.slot_id not in ov.slots:
                continue
            if members is not None and slot.parent_entity not in members:
                continue
            if (not slot_usable(ctx, ov, slot) or not target_slot_ok(ctx, ov, slot, mode)
                    or not slot_placeable(ctx, ov, slot)):
                continue
            for cat in sorted(ctx.eligibility.categories):
                if not ctx.priors.is_approved(cat, slot.slot_type, room_type):
                    continue
                for m in ctx.eligibility.categories[cat].models:
                    if not fits(m.bbox, slot):
                        continue
                    n = len(candidate_slots(ctx, scene, [room], cat, m.bbox,
                                            None if scope is None else scope.region_id))
                    widths_seen[n] = widths_seen.get(n, 0) + 1
                    if width.contains(n):
                        options.append((cat, m.model_id, slot, scope))
    ctx.cache[key] = (options, widths_seen)
    return ctx.cache[key]


def _distractors(ctx: PlannerContext, seed: int, index: int, retry: int, scene: str,
                 room: str, tslot: Slot, target: PlacedObject) -> list[PlacedObject]:
    spec = ctx.tier
    ss = ctx.scenes[scene]
    el = ctx.eligibility
    room_type = ss.rooms[room]
    tbox = el.categories[target.category].model(target.model).bbox
    out: list[PlacedObject] = []
    rng = stream(seed, index, "distractors", retry)

    def pick_for(slot: Slot, used: list) -> PlacedObject | None:
        """One approved, fitting, non-lookalike object for slot, or None."""
        pool = []
        for cat in sorted(el.categories):
            if el.lookalike(target.category, cat):
                continue  # easy: lookalike_count == 0
            if not ctx.priors.is_approved(cat, slot.slot_type, room_type):
                continue
            for m in el.categories[cat].models:
                if fits(m.bbox, slot):
                    pool.append((cat, m))
        if not pool:
            return None
        cat, m = choice(rng, pool)
        cap = (spec.target_slot.max_fill_ratio if slot.slot_id == tslot.slot_id else
               spec.other_containers.max_fill_ratio if slot.is_container else
               spec.other_surfaces.max_area_ratio)
        if occupancy(used + [m.bbox], slot) > cap:
            return None
        used.append(m.bbox)
        return PlacedObject(entity="", category=cat, model=m.model_id,
                            slot_id=slot.slot_id, role="")

    def place(slot: Slot, k: int, role: str, used: list) -> None:
        for _ in range(k):
            if spec.max_distractors is not None and len(out) >= spec.max_distractors:
                return
            o = pick_for(slot, used)
            if o is None:
                return
            o.entity = f"obj_{len(out):02d}_{o.category}"
            o.role = role
            out.append(o)

    b = spec.target_slot.extra_items
    place(tslot, randint_bound(rng, b.min, b.max), "target_slot", [tbox])

    ov = ctx.overlay(scene)
    # only slots the probe could physically fill (a failed distractor
    # placement aborts the whole episode build)
    room_slots = [s for s in ss.slots_in_room(room)
                  if s.slot_id != tslot.slot_id and slot_placeable(ctx, ov, s)]
    containers = [s for s in room_slots if s.is_container]
    surfaces = [s for s in room_slots if not s.is_container]
    oc = spec.other_containers
    if containers and oc.occupied_fraction.max:
        frac = float(rng.uniform(oc.occupied_fraction.min, oc.occupied_fraction.max))
        n_occ = min(len(containers), max(1, round(frac * len(containers))))
        chosen = sorted(rng.choice(len(containers), size=n_occ, replace=False).tolist())
        for i in chosen:
            place(containers[i], randint_bound(rng, oc.items.min, oc.items.max),
                  "other_container", [])
    si = spec.other_surfaces.items
    for s in surfaces:
        place(s, randint_bound(rng, si.min, si.max), "other_surface", [])
    return out


def _provenance(ctx: PlannerContext, plan: TaskPlan) -> dict[str, Any]:
    base = ctx.provenance_base
    ov = ctx.overlays[plan.scene]
    inputs = dict(base.get("inputs", {}))
    inputs["scene_slots_hash"] = base.get("scene_slots_hashes", {}).get(plan.scene)
    inputs["overlay_hash"] = base.get("overlay_hashes", {}).get(plan.scene)
    env = dict(base.get("environment", {}))
    env["scene"] = plan.scene
    return provenance(
        generator=dict(base.get("generator", {})), inputs=inputs, environment=env,
        robot={"id": ov.robot_id, "urdf_hash": ov.urdf_hash,
               "overlay_status": ov.status},
        seed=plan.seed, episode_index=plan.episode_index)
