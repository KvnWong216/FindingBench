"""Deterministic easy planning,
compiler, CPU gates — on a synthetic scene so no simulator/probe is needed."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.compile import compile_plan, scenario_hash
from rummagebench.authoring.tasks.eligibility import (
    EligibleCategory, ModelRecord, ObjectEligibility)
from rummagebench.authoring.tasks.embodiment_slots import (
    Pose2, SlotEmbodimentOverlay, SlotInteraction, StartPose)
from rummagebench.authoring.tasks.plan import TaskPlan
from rummagebench.authoring.tasks.planner import (
    PlannerContext, certificate_for, plan_episode)
from rummagebench.authoring.tasks.priors import PlacementPriors, PriorEntry
from rummagebench.authoring.tasks.slots import (
    Furniture, Region, SceneSlots, Slot, SlotGeometry)
from rummagebench.authoring.tasks.tiers import load_all_tiers, load_tier
from rummagebench.core.scenario import ScenarioSpec

SCENE = "Synth_0_int"
YAW90 = (0.0, 0.0, 0.7071068, 0.7071068)


def _slot(furn, link, room, slot_type, x, requires_open=True):
    inside = slot_type in ("drawer", "cabinet_interior")
    geom = (SlotGeometry((0.4, 0.35, 0.15), 0.021, (0.4, 0.35), 0.6) if inside else
            SlotGeometry((1.2, 0.6, 0.0), 0.0, (1.2, 0.6), 0.9))
    return Slot(slot_id=f"{SCENE}/{furn}/{link or ('interior' if inside else 'on_top')}",
                scene=SCENE, room=room, room_type="kitchen", parent_entity=furn,
                parent_category="bottom_cabinet" if inside else "countertop",
                parent_model="m", relation="inside" if inside else "on_top",
                slot_type=slot_type, link=link, requires_open=inside and requires_open,
                articulated=inside, geometry=geom, parent_position=(x, 0.0, 0.4),
                parent_yaw=0.0)


def build_ctx(start_xy=(1.0, -1.0), probed=True,
              wide_room=True, with_regions=True) -> PlannerContext:
    slots = [_slot("cab_a", "link_0", "kitchen_0", "drawer", 0.0),
             _slot("cab_a", "link_1", "kitchen_0", "drawer", 0.0),
             _slot("cab_b", "link_0", "kitchen_0", "drawer", 1.0),
             _slot("counter_a", None, "kitchen_0", "countertop", 2.0)]
    furniture = [Furniture("cab_a", "bottom_cabinet", "m", "kitchen_0", "kitchen",
                           (0.0, 0.0, 0.4), 0.0, True),
                 Furniture("cab_b", "bottom_cabinet", "m", "kitchen_0", "kitchen",
                           (1.0, 0.0, 0.4), 0.0, True),
                 Furniture("counter_a", "countertop", "m", "kitchen_0", "kitchen",
                           (2.0, 0.0, 0.4), 0.0, False)]
    rooms = {"kitchen_0": "kitchen"}
    if wide_room:  # 6 drawers -> candidate width 6 > easy max 4
        rooms["kitchen_1"] = "kitchen"
        for i in range(3):
            furniture.append(Furniture(f"wide_{i}", "bottom_cabinet", "m", "kitchen_1",
                                       "kitchen", (10.0 + i, 0.0, 0.4), 0.0, True))
            for l in ("link_0", "link_1"):
                slots.append(_slot(f"wide_{i}", l, "kitchen_1", "drawer", 10.0 + i))
    regions = []
    if wide_room and with_regions:
        regions = [Region(f"{SCENE}/kitchen_1/near_sink_0", "kitchen_1", "sink_0", "sink",
                          ("wide_0",), 0.9, True, True),
                   # ambiguous boundary -> must never be offered
                   Region(f"{SCENE}/kitchen_1/near_fridge_0", "kitchen_1", "fridge_0",
                          "fridge", ("wide_2",), 0.2, False, True)]
    ss = SceneSlots(scene=SCENE, version=1, source={}, rooms=rooms, slots=slots,
                    furniture=furniture,
                    native_objects={"kitchen_0": ["door", "electric_switch"]},
                    regions=regions)

    def anchor(x):
        return Pose2((x, -0.8, 0.0), YAW90)

    # probe evidence: OPEN is a causal reveal for every container slot
    seen = {"placed": True,
            "open_visibility": {"visible_px": 400, "reference_px": 500, "ratio": 0.8}}
    causal = {**seen,
              "closed_visibility": {"visible_px": 0, "reference_px": 500, "ratio": 0.0},
              "closed_graspable": False}
    inter = {s.slot_id: SlotInteraction(anchor(s.parent_position[0]), True, True, probed,
                                        evidence=dict(causal if s.requires_open else seen),
                                        reveal_anchor=anchor(s.parent_position[0]))
             for s in slots}
    starts = {"kitchen_0": [StartPose("k0_start", Pose2((*start_xy, 0.0), YAW90), [])]}
    if wide_room:
        starts["kitchen_1"] = [StartPose("k1_start", Pose2((11.0, -2.0, 0.0), YAW90), [])]
    ov = SlotEmbodimentOverlay(
        scene=SCENE, robot_id="r1pro", urdf_path="build/robots/r1pro.urdf",
        urdf_hash="u" * 64, status="probed" if probed else "unverified", source="test",
        robot_config={"model": "r1pro", "name": "robot_0", "obs_modalities": ["rgb"],
                      "kinematics": {"urdf_path": "build/robots/r1pro.urdf",
                                     "end_effector_link": "right_eef_link"}},
        feasibility={"backend": "pinocchio", "mode": "endpoint"},
        start_poses=starts, slots=inter)

    def cat(name, keys, bbox):
        return EligibleCategory(name, f"{name}.n.01", 0.2, keys,
                                [ModelRecord(f"{name}_m0", bbox),
                                 ModelRecord(f"{name}_m1", bbox)])
    el = ObjectEligibility(1, {}, {
        "bowl": cat("bowl", ["category:bowl", "head:bowl"], (0.15, 0.15, 0.08)),
        "tablefork": cat("tablefork", ["category:tablefork", "parent:cutlery"],
                         (0.2, 0.03, 0.02)),
        "tablespoon": cat("tablespoon", ["category:tablespoon", "parent:cutlery"],
                          (0.18, 0.04, 0.02)),
        "apple": cat("apple", ["category:apple"], (0.08, 0.08, 0.08)),
    }, {})

    classes = {"drawer": "cabinet_storage", "cabinet_interior": "cabinet_storage"}

    def pe(c, st, w=1.0, approved=True):
        cls = classes.get(st, st)
        covers = ["cabinet_interior", "drawer"] if cls == "cabinet_storage" else [st]
        cats = ["chopping_board", "cutting_board"] if c == "chopping_board" else [c]
        return PriorEntry(f"{c}.n.01", cls, "kitchen",
                          "inside" if st == "drawer" else "on_top",
                          10, 5, object_categories=cats, covers_slot_types=covers,
                          approved=approved,
                          compatibility="high", generation_weight=w if approved else 0.0,
                          review={"reviewer": "test"} if approved else None)
    pri = PlacementPriors(1, {}, [pe("bowl", "drawer"), pe("bowl", "countertop"),
                                  pe("tablefork", "drawer"), pe("tablespoon", "drawer"),
                                  pe("apple", "countertop"),
                                  pe("apple", "drawer", approved=False)],
                          slot_classes=classes)
    base = {"generator": {"git_commit": "c0", "git_dirty": False},
            "inputs": {"tier_spec_hash": "t", "placement_priors_hash": "p",
                       "eligibility_hash": "e"},
            "scene_slots_hashes": {SCENE: "s"}, "overlay_hashes": {SCENE: "o"},
            "environment": {"behavior_assets_version": "3.9.0",
                            "omnigibson_version": "3.9.3"}}
    return PlannerContext(tier=load_tier("easy_v1"), scenes={SCENE: ss}, priors=pri,
                          eligibility=el, overlays={SCENE: ov}, provenance_base=base)


@pytest.fixture(scope="module")
def ctx():
    return build_ctx()


@pytest.fixture(scope="module")
def outcomes(ctx):
    return [plan_episode(ctx, seed=7, index=i) for i in range(24)]


def test_same_seed_same_bytes(ctx):
    a = plan_episode(ctx, seed=7, index=3)
    b = plan_episode(ctx, seed=7, index=3)
    assert a.plan is not None
    assert a.plan.canonical_bytes() == b.plan.canonical_bytes()
    assert a.attempts == b.attempts


def test_input_hash_change_changes_identity(ctx):
    a = plan_episode(ctx, seed=7, index=3).plan
    c2 = build_ctx()
    c2.provenance_base["inputs"]["placement_priors_hash"] = "p2"
    b = plan_episode(c2, seed=7, index=3).plan
    assert a.task_id != b.task_id  # same seed, other inputs != same episode
    c3 = build_ctx()
    c3.provenance_base["generator"]["git_commit"] = "c1"
    assert plan_episode(c3, seed=7, index=3).plan.task_id == a.task_id


def test_plans_are_certified_easy_structurally(ctx, outcomes):
    tiers = load_all_tiers()
    plans = [o.plan for o in outcomes if o.plan]
    assert len(plans) >= 20
    modes = {p.mode for p in plans}
    assert modes == {"in_container", "on_surface"}
    for p in plans:
        cert = certificate_for(ctx, p)
        assert gates.classify(cert, tiers) == ["easy"]
        assert cert.room_count == 1 and 2 <= cert.candidate_slot_count <= 4
        assert cert.lookalike_count == 0 and cert.rearrangement_depth == 0
        if p.search_rooms == ["kitchen_1"]:
            # 6-drawer room only fits easy through the landmark region
            assert p.search_region == f"{SCENE}/kitchen_1/near_sink_0"
            assert "next to the sink" in p.instruction or "by the sink" in p.instruction \
                or "around the sink" in p.instruction
            assert all("/wide_0/" in c for c in cert.candidate_slots)
        else:
            assert p.search_rooms == ["kitchen_0"] and p.search_region is None
        if p.mode == "in_container":
            assert cert.open_depth == 1 and cert.target_slot_type == "drawer"
        else:
            assert cert.open_depth == 0 and cert.target_slot_type == "countertop"
        cats = {o.category for o in p.distractors}
        if p.target.category in ("tablefork", "tablespoon"):
            assert not cats & {"tablefork", "tablespoon"}  # no lookalikes


def test_wide_room_is_never_drawn_and_rejected_with_reason():
    ctx = build_ctx(with_regions=False)  # no landmark region to narrow it
    outs = [plan_episode(ctx, seed=1, index=i) for i in range(30)]
    assert all(o.plan.room == "kitchen_0" for o in outs if o.plan)
    assert not [a for o in outs for a in o.attempts
                if gates.INVALID_PRIOR in a["codes"]]  # rooms drawn only if viable
    # the wide room alone: the reason is still reported
    ctx = build_ctx(with_regions=False)
    del ctx.overlays[SCENE].start_poses["kitchen_0"]
    codes = [c for i in range(10) for a in plan_episode(ctx, seed=1, index=i).attempts
             for c in a["codes"]]
    assert gates.SEARCH_WIDTH_MISMATCH in codes


def test_scene_share_does_not_scale_with_room_count():
    """Scene first, then room: a scene with many viable rooms gets the same
    share as a scene with one."""
    import copy
    from collections import Counter

    base = build_ctx(wide_room=True)  # Synth_0: kitchen_0 + region-scoped kitchen_1
    other = copy.deepcopy(base.scenes[SCENE])
    other.scene = "Synth_1_int"
    other.rooms = {"kitchen_0": "kitchen", "kitchen_1": "kitchen"}
    ov1 = copy.deepcopy(base.overlays[SCENE])
    del ov1.start_poses["kitchen_1"]  # Synth_1 has one viable room
    prov = copy.deepcopy(base.provenance_base)
    prov["scene_slots_hashes"]["Synth_1_int"] = "s1"
    prov["overlay_hashes"]["Synth_1_int"] = "o1"
    ctx = PlannerContext(tier=base.tier, scenes={SCENE: base.scenes[SCENE], "Synth_1_int": other},
                         priors=base.priors, eligibility=base.eligibility,
                         overlays={SCENE: base.overlays[SCENE], "Synth_1_int": ov1},
                         provenance_base=prov)
    c = Counter(plan_episode(ctx, seed=2, index=i).plan.scene for i in range(200))
    assert abs(c[SCENE] - c["Synth_1_int"]) < 50, c


def test_horizon_lower_bound_rejects_far_start():
    ctx = build_ctx(start_xy=(0.0, -20.0), wide_room=False)
    out = plan_episode(ctx, seed=7, index=0)
    assert out.plan is None
    assert all(gates.HORIZON_LOWER_BOUND in a["codes"] for a in out.attempts)


def test_unprobed_overlay_is_rejected():
    out = plan_episode(build_ctx(probed=False, wide_room=False), seed=7, index=0)
    assert out.plan is None
    assert out.attempts[0]["codes"] == [gates.INVALID_SLOT]


def test_provenance_is_complete(outcomes):
    p = next(o.plan for o in outcomes if o.plan)
    prov = p.provenance
    for k in ("tier_spec_hash", "scene_slots_hash", "placement_priors_hash",
              "eligibility_hash", "overlay_hash"):
        assert prov["inputs"][k]
    assert prov["robot"]["urdf_hash"] and prov["episode"]["seed"] == 7
    assert prov["generator"]["version"]


def test_compiler_emits_frozen_schema_without_tier_metadata(ctx, outcomes):
    tier = load_tier("easy_v1")
    for o in outcomes:
        if not o.plan:
            continue
        p = o.plan
        doc = compile_plan(p, ctx.scenes[SCENE], ctx.overlays[SCENE], tier)
        spec = ScenarioSpec.model_validate(doc)
        assert spec.id == p.task_id and spec.id.startswith("fb_")
        assert spec.termination.max_planning_steps == 16
        text = json.dumps(doc)
        for leak in ("easy", "in_container", "on_surface", "seed", "tier"):
            assert leak not in text, leak
        # all openable furniture of the searched room start closed
        expected = ({"cab_a", "cab_b"} if p.search_rooms == ["kitchen_0"]
                    else {"wide_0", "wide_1", "wide_2"})  # whole room, not just region
        assert set(spec.initial_states) == expected
        assert all(s.open is False for s in spec.initial_states.values())
        tp = next(pl for pl in spec.placements if pl.entity == p.target.entity)
        if p.mode == "in_container":
            assert tp.relation == "inside" and tp.link in ("link_0", "link_1")
            assert [a["skill"] for a in doc["agent"]["scripted_success"]] == [
                "NAV", "OPEN", "GRASP"]
        else:
            assert tp.relation == "on_top" and tp.link is None
        assert scenario_hash(doc) == scenario_hash(deepcopy(doc))


def test_cpu_gates_catch_tampered_plans(ctx, outcomes):
    p = next(o.plan for o in outcomes if o.plan and o.plan.mode == "in_container")
    bad = TaskPlan.from_dict(p.to_dict())
    bad.target.model = "not_whitelisted"
    codes = {r.code for r in gates.cpu_plan_gates(ctx, bad, certificate_for(ctx, p))}
    assert gates.INVALID_OBJECT_CATEGORY in codes

    bad = TaskPlan.from_dict(p.to_dict())
    bad.distractors.append(type(p.target)("obj_99_apple", "apple", "apple_m0",
                                           p.target.slot_id, "target_slot"))
    codes = {r.code for r in gates.cpu_plan_gates(ctx, bad, certificate_for(ctx, bad))}
    assert gates.INVALID_PRIOR in codes  # apple/drawer evidence exists but unapproved

    bad = TaskPlan.from_dict(p.to_dict())
    bad.candidate_slots = bad.candidate_slots[:1]
    codes = {r.code for r in gates.cpu_plan_gates(ctx, bad, certificate_for(ctx, bad))}
    assert gates.SEARCH_WIDTH_MISMATCH in codes


def test_disabled_tiers_refuse_to_plan(ctx):
    c = build_ctx()
    c.tier = load_tier("medium_v1").model_copy(update={"planner_enabled": False})
    with pytest.raises(RuntimeError):
        plan_episode(c, seed=0, index=0)


def test_wide_room_yields_region_scoped_plans(outcomes):
    regional = [o.plan for o in outcomes if o.plan and o.plan.search_region]
    assert regional
    assert all(p.mode == "in_container" for p in regional)  # region has no surface
    assert all("fridge" not in p.search_region for p in regional)


def test_tampered_region_is_rejected(ctx, outcomes):
    p = next(o.plan for o in outcomes if o.plan and o.plan.search_region)
    bad = TaskPlan.from_dict(p.to_dict())
    bad.search_region = f"{SCENE}/kitchen_1/near_fridge_0"
    with pytest.raises(Exception) as e:
        certificate_for(ctx, bad)
    assert "not usable" in str(e.value)
    bad.search_region = None  # whole 6-drawer room -> width out of range
    cert = certificate_for(ctx, bad)
    codes = {r.code for r in gates.cpu_plan_gates(ctx, bad, cert)}
    assert gates.SEARCH_WIDTH_MISMATCH in codes


def _records():
    from rummagebench.authoring.tasks.split import EpisodeRecord
    out = []
    for i, (scene, fam) in enumerate([("A", "A/k/near_sink"), ("A", "A/k/near_fridge"),
                                      ("B", "B/k/near_sink"), ("C", "C/k/near_oven"),
                                      ("C", "C/k/near_oven")]):
        out.append(EpisodeRecord(f"fb_{i}", scene, "kitchen_0", "kitchen", fam, "bowl",
                                 "bowl/m0", ["plate/p0"], "drawer", "in_container"))
    return out


def test_split_holds_out_scenes_and_checks_leakage():
    from rummagebench.authoring.tasks.split import leakage_violations, make_split
    recs = _records()
    s = make_split(recs, ["C"])
    assert s.policy == "held_out_scene" and sorted(s.test) == ["fb_3", "fb_4"]
    assert leakage_violations(s, recs) == []
    s.dev.append("fb_3")
    assert leakage_violations(s, recs)
    with pytest.raises(ValueError):
        make_split(recs, ["A", "B", "C"])  # cannot hold out everything


def test_split_family_fallback_is_explicit():
    from rummagebench.authoring.tasks.split import leakage_violations, make_split
    recs = [r for r in _records() if r.scene == "A"]
    with pytest.raises(ValueError):
        make_split(recs, [])  # fallback needs explicit families
    s = make_split(recs, [], test_families=["A/k/near_fridge"])
    assert s.policy == "held_out_family" and s.test == ["fb_1"] and s.notes
    assert leakage_violations(s, recs) == []


def test_category_share_does_not_scale_with_model_count():
    """bowl gets 40 models, tablefork keeps 2: the two-stage draw must still
    pick tablefork targets regularly (one-stage expansion would drown it)."""
    from rummagebench.authoring.tasks.eligibility import ModelRecord
    ctx = build_ctx(wide_room=False)
    ctx.eligibility.categories["bowl"].models = [
        ModelRecord(f"bowl_m{i}", (0.15, 0.15, 0.08)) for i in range(40)]
    from collections import Counter
    c = Counter(plan_episode(ctx, seed=3, index=i).plan.target.category
                for i in range(120))
    in_container = c["tablefork"] + c["tablespoon"]
    assert in_container >= 25, c  # one-stage would give ~2/42 of drawer picks


def test_alias_categories_share_one_synset_share():
    """chopping_board + cutting_board are ONE synset: together they must get
    one synset's share, not two (two-stage-by-category would double them)."""
    from collections import Counter
    from rummagebench.authoring.tasks.eligibility import EligibleCategory, ModelRecord
    ctx = build_ctx(wide_room=False)
    for c in ("chopping_board", "cutting_board"):
        ctx.eligibility.categories[c] = EligibleCategory(
            c, "chopping_board.n.01", 1.0, [f"category:{c}", "head:board"],
            [ModelRecord(f"{c}_m0", (0.3, 0.2, 0.02))])
    pri = ctx.priors
    from rummagebench.authoring.tasks.priors import PlacementPriors, PriorEntry
    entries = [e for e in pri.entries] + [PriorEntry(
        "chopping_board.n.01", cls, "kitchen", rel, 61, 58,
        object_categories=["chopping_board", "cutting_board"],
        covers_slot_types=covers, approved=True, compatibility="high",
        generation_weight=1.0, review={"reviewer": "test"})
        for cls, rel, covers in (("countertop", "on_top", ["countertop"]),
                                 ("cabinet_storage", "inside",
                                  ["cabinet_interior", "drawer"]))]
    ctx.priors = PlacementPriors(1, {}, entries, slot_classes=pri.slot_classes)
    assert ctx.priors.get("cutting_board", "countertop", "kitchen") is \
        ctx.priors.get("chopping_board", "countertop", "kitchen")
    c = Counter(p.target.category for p in
                (plan_episode(ctx, seed=5, index=i).plan for i in range(400))
                if p.mode == "on_surface")
    boards = c["chopping_board"] + c["cutting_board"]
    # on_surface-eligible synsets: bowl.n.01 and chopping_board.n.01 (apple has
    # a single candidate slot) -> boards ~1/2; per-category draws give ~2/3
    assert 0.38 < boards / sum(c.values()) < 0.6, c
    assert c["chopping_board"] and c["cutting_board"]  # both aliases still used


def test_non_causal_open_slot_is_never_an_in_container_target():
    """Probe evidence (no-top cabinet): invisible while closed but GRASPable
    from above -> OPEN is not causal -> never the in_container target slot,
    yet still counted as a search candidate."""
    c = build_ctx(wide_room=False)
    bad = f"{SCENE}/cab_a/link_0"
    c.overlays[SCENE].slots[bad].evidence["closed_graspable"] = True
    assert c.overlays[SCENE].slots[bad].open_causal is False
    plans = [o.plan for o in (plan_episode(c, seed=3, index=i) for i in range(30)) if o.plan]
    inside = [p for p in plans if p.mode == "in_container"]
    assert inside
    assert all(p.target.slot_id != bad for p in inside)
    assert any(bad in p.candidate_slots for p in inside)


def test_target_slot_must_be_visible_from_its_grasp_pose():
    """Probe: GRASP feasible but the revealed object out of view from that
    pose (low drawer under a level head camera) -> never a target slot."""
    c = build_ctx(wide_room=False)
    hidden = f"{SCENE}/cab_b/link_0"
    c.overlays[SCENE].slots[hidden].evidence["open_visibility"] = {
        "visible_px": 0, "reference_px": 0, "ratio": None}
    assert c.overlays[SCENE].slots[hidden].reveal_visibility == 0.0
    plans = [o.plan for o in (plan_episode(c, seed=5, index=i) for i in range(30)) if o.plan]
    assert plans and all(p.target.slot_id != hidden for p in plans)


def test_target_slot_needs_a_reveal_pose():
    c = build_ctx(wide_room=False)
    blind = f"{SCENE}/cab_a/link_1"
    c.overlays[SCENE].slots[blind].reveal_anchor = None
    plans = [o.plan for o in (plan_episode(c, seed=9, index=i) for i in range(30)) if o.plan]
    assert plans and all(p.target.slot_id != blind for p in plans)


def test_compiler_adds_the_reveal_anchor_to_the_oracle_plan(ctx, outcomes):
    from rummagebench.authoring.tasks.compile import compile_plan, reveal_anchor_name
    from rummagebench.authoring.tasks.embodiment_slots import Pose2

    plan = next(o.plan for o in outcomes if o.plan and o.plan.mode == "in_container")
    ov = ctx.overlays[SCENE]
    inter = ov.slots[plan.target.slot_id]
    saved = inter.reveal_anchor
    inter.reveal_anchor = Pose2((9.0, 9.0, 0.0), inter.navigation_anchor.orientation)
    try:
        ss = ctx.scenes[SCENE]
        doc = compile_plan(plan, ss, ov, ctx.tier)
    finally:
        inter.reveal_anchor = saved
    parent = ss.by_id()[plan.target.slot_id].parent_entity
    name = reveal_anchor_name(parent)
    assert doc["anchors"][name]["position"][:2] == [9.0, 9.0]
    skills = [(a["skill"], a["target"]["value"]) for a in doc["agent"]["scripted_success"]]
    assert skills == [("NAV", parent), ("OPEN", parent), ("NAV", name),
                      ("GRASP", plan.target.entity)]


def test_unplaceable_slots_never_receive_objects():
    """A slot where the probe could not place an object (builder placement
    failed) holds neither the target nor distractors."""
    c = build_ctx(wide_room=False)
    bad = f"{SCENE}/cab_b/link_0"
    c.overlays[SCENE].slots[bad].evidence["placed"] = False
    plans = [o.plan for o in (plan_episode(c, seed=4, index=i) for i in range(30)) if o.plan]
    assert plans
    assert all(o.slot_id != bad for p in plans for o in p.objects)


def _walled_ctx(only_far: bool) -> PlannerContext:
    """kitchen_0 gets a second start behind a wall (y in [-4.0, -3.9]) that no
    MOVE/TURN route crosses; measured obstacles enable the route filter."""
    ctx = build_ctx(wide_room=False)
    ss, ov = ctx.scenes[SCENE], ctx.overlays[SCENE]
    ss.footprint_obstacles = {"wall": [[-20.0, -4.0, 0.0], [20.0, -3.9, 2.0]]}
    far = StartPose("k0_far", Pose2((1.0, -6.0, 0.0), YAW90), [])
    ov.start_poses["kitchen_0"] = [far] if only_far else ov.start_poses["kitchen_0"] + [far]
    return ctx


def test_start_poses_without_a_route_to_the_target_are_never_drawn():
    ctx = _walled_ctx(only_far=False)
    plans = [o.plan for o in (plan_episode(ctx, seed=6, index=i) for i in range(30)) if o.plan]
    assert plans and all(p.start_pose == "k0_start" for p in plans)


def test_unreachable_room_is_rejected_with_reason():
    ctx = _walled_ctx(only_far=True)
    out = plan_episode(ctx, seed=6, index=0)
    assert out.plan is None
    assert all(gates.START_UNREACHABLE in a["codes"] for a in out.attempts)


def test_tall_narrow_items_do_not_fit_on_surfaces():
    # A.8.4: height / shorter footprint side > 2.5 topples off surfaces
    from rummagebench.authoring.tasks.planner import fits

    top = _slot("counter", None, "kitchen_0", "countertop", 2.0)
    drawer = _slot("cab_a", "link_0", "kitchen_0", "drawer", 0.0)
    spray_bottle = (0.10, 0.08, 0.26)    # 3.25
    mug = (0.12, 0.09, 0.07)
    assert not fits(spray_bottle, top) and fits(mug, top)
    assert fits((0.10, 0.08, 0.14), drawer)  # inside: only the height limit


def test_easy_keeps_containers_sparse():
    spec = load_tier("easy_v1")
    assert spec.target_slot.extra_items.max <= 1
    assert spec.other_containers.items.max <= 1


def test_excluded_room_types_are_never_drawn():
    import dataclasses

    ctx = build_ctx()
    assert plan_episode(ctx, 0, 0, 8).plan is not None
    no_kitchen = dataclasses.replace(ctx, exclude_room_types=frozenset({"kitchen"}))
    # the synthetic scene has kitchens only
    assert plan_episode(no_kitchen, 0, 0, 8).plan is None


def test_on_surface_starts_keep_out_of_reach_of_the_target_furniture():
    """A.9: on_surface starts <= 0.97 m from the target furniture were
    GRASP-feasible on GPU (TARGET_PREMATURELY_GRASPABLE)."""
    ctx = build_ctx(wide_room=False)
    ss, ov = ctx.scenes[SCENE], ctx.overlays[SCENE]
    ss.footprint_obstacles = {"counter_a": [[1.4, -0.3, 0.0], [2.6, 0.3, 0.9]]}
    far = StartPose("k0_far", Pose2((1.0, -2.5, 0.0), YAW90), [])
    ov.start_poses["kitchen_0"] = ov.start_poses["kitchen_0"] + [far]  # k0_start: 0.81 m
    plans = [o.plan for o in (plan_episode(ctx, seed=12, index=i) for i in range(40)) if o.plan]
    surf = [p for p in plans if p.mode == "on_surface"]
    cont = [p for p in plans if p.mode == "in_container"]
    assert surf and all(p.start_pose == "k0_far" for p in surf)
    assert any(p.start_pose == "k0_start" for p in cont)  # containers unaffected


def test_certifier_view_anchor_is_added_once_before_the_grasp(ctx, outcomes):
    from rummagebench.authoring.tasks.compile import (
        compile_plan, view_anchors_of, with_view_anchors)

    plan = next(o.plan for o in outcomes if o.plan)
    doc = compile_plan(plan, ctx.scenes[SCENE], ctx.overlays[SCENE], ctx.tier)
    va = {"cab_a__view": {"position": [0.1, -1.5, 0.0], "orientation": list(YAW90)}}
    once = with_view_anchors(doc, va)
    twice = with_view_anchors(once, view_anchors_of(once))
    assert once == twice and view_anchors_of(doc) == {}
    succ = once["agent"]["scripted_success"]
    assert succ[-2] == {"skill": "NAV", "target": {"type": "place", "value": "cab_a__view"}}
    assert succ[-1]["skill"] == "GRASP"
    ScenarioSpec.model_validate(once)
