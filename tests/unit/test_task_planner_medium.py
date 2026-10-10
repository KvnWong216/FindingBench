"""Medium planning (A.12): buried (drawer) and covered (counter) modes, cover
stacks, lookalikes, region scopes — on the synthetic easy-planner scene."""
from __future__ import annotations

import pytest

from test_task_planner_easy import SCENE, build_ctx

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.compile import compile_plan
from rummagebench.authoring.tasks.eligibility import (
    EligibleCategory, ModelRecord, ObjectEligibility)
from rummagebench.authoring.tasks.planner import plan_episode
from rummagebench.authoring.tasks.priors import PlacementPriors, PriorEntry
from rummagebench.authoring.tasks.tiers import load_tier
from rummagebench.authoring.tasks.tiers.medium import covers_target, is_flat, is_thin_target


def medium_ctx():
    ctx = build_ctx()

    def cat(name, keys, bbox):
        return EligibleCategory(name, f"{name}.n.01", 0.2, keys,
                                [ModelRecord(f"{name}_m0", bbox)])
    ctx.eligibility = ObjectEligibility(1, {}, {
        "tablespoon": cat("tablespoon", ["category:tablespoon", "parent:cutlery"],
                          (0.18, 0.04, 0.02)),
        "tablefork": cat("tablefork", ["category:tablefork", "parent:cutlery"],
                         (0.2, 0.03, 0.02)),
        "plate": cat("plate", ["category:plate"], (0.37, 0.26, 0.02)),
        "bowl": cat("bowl", ["category:bowl"], (0.15, 0.15, 0.08)),
    }, {})
    classes = {"drawer": "cabinet_storage", "cabinet_interior": "cabinet_storage"}

    def pe(c, st):
        cls = classes.get(st, st)
        covers = ["cabinet_interior", "drawer"] if cls == "cabinet_storage" else [st]
        return PriorEntry(f"{c}.n.01", cls, "kitchen",
                          "inside" if st == "drawer" else "on_top", 10, 5,
                          object_categories=[c], covers_slot_types=covers, approved=True,
                          compatibility="high", generation_weight=1.0,
                          review={"reviewer": "test"})
    ctx.priors = PlacementPriors(1, {}, [pe(c, st) for c in ("tablespoon", "plate", "bowl")
                                         for st in ("drawer", "countertop")]
                                 + [pe("tablefork", "drawer")], slot_classes=classes)
    ctx.tier = load_tier("medium_v1")
    ctx.cache.clear()
    return ctx


@pytest.fixture(scope="module")
def mctx():
    return medium_ctx()


@pytest.fixture(scope="module")
def plans(mctx):
    outs = [plan_episode(mctx, seed=3, index=i) for i in range(24)]
    return [o for o in outs if o.plan is not None]


def test_shape_rules():
    assert is_thin_target((0.18, 0.04, 0.02))  # a spoon may be a target ...
    assert not is_flat((0.18, 0.04, 0.02))      # ... but never a cover
    assert not is_thin_target((0.15, 0.15, 0.08))
    assert covers_target((0.37, 0.26, 0.02), (0.18, 0.04, 0.02))
    # 1.3x the target's long side, but only 0.035 m overhang: side-graspable
    assert not covers_target((0.25, 0.25, 0.02), (0.18, 0.04, 0.02))


def test_both_modes_plan_deterministically(mctx, plans):
    assert {o.plan.mode for o in plans} == {"buried", "covered"}
    again = plan_episode(mctx, seed=3, index=0)
    first = plan_episode(mctx, seed=3, index=0)
    assert again.plan.canonical_bytes() == first.plan.canonical_bytes()


def test_plans_satisfy_the_medium_contract(mctx, plans):
    for o in plans:
        p = o.plan
        covers = [d for d in p.distractors if d.role == "cover"]
        assert 1 <= len(covers) <= 2
        assert all(d.slot_id == p.target.slot_id for d in covers)
        assert p.target.category in ("tablespoon", "tablefork")
        assert all(d.category == "plate" for d in covers)
        assert p.expected["rearrangement_depth"] == len(covers)
        assert p.expected["lookalike_count"] >= 1
        if p.mode == "buried":
            assert p.expected["open_depth"] == 1 and p.expected["containment_depth"] == 1
        else:
            assert p.expected["open_depth"] == 0 and p.expected["containment_depth"] == 0
            assert p.target.slot_id.endswith("/on_top")
        assert not gates.structural_violations(o.certificate, mctx.tier)


def test_wide_room_uses_a_region_scope(plans):
    wide = [o.plan for o in plans if o.plan.room == "kitchen_1"]
    assert wide and all(p.search_region for p in wide)


def test_covers_compile_to_an_on_top_chain(mctx, plans):
    p = next(o.plan for o in plans if len([d for d in o.plan.distractors
                                           if d.role == "cover"]) == 2)
    doc = compile_plan(p, mctx.scenes[SCENE], mctx.overlay(SCENE), mctx.tier)
    covers = [d.entity for d in p.distractors if d.role == "cover"]
    chain = {pl["entity"]: pl for pl in doc["placements"]}
    assert chain[covers[0]] == {"entity": covers[0], "relation": "on_top",
                                "receptacle": p.target.entity}
    assert chain[covers[1]]["receptacle"] == covers[0]
    assert doc["termination"]["fail_on_wrong_grasp"] is False


def test_tier_lookalike_groups_are_tier_scoped(mctx):
    el = mctx.eligibility
    assert not el.lookalike("plate", "bowl")
    grouped = el.with_lookalike_groups([["plate", "bowl"]])
    assert grouped.lookalike("plate", "bowl") and grouped.lookalike("bowl", "plate")
    assert not el.lookalike("plate", "bowl")  # the original is untouched
    assert load_tier("easy_v1").lookalike_groups == []


def test_medium_lookalike_groups_name_eligible_categories():
    import yaml
    from pathlib import Path

    elig = Path("build/tasks/priors/eligibility_v1.yaml")
    if not elig.exists():
        pytest.skip("generated eligibility not built on this host")
    cats = set(yaml.safe_load(elig.read_text())["categories"])
    for group in load_tier("medium_v1").lookalike_groups:
        assert set(group) <= cats, set(group) - cats
