"""The tier contract can mechanically decide
which tier a synthetic certificate belongs to; gates are evidence-driven."""
from __future__ import annotations

import math

import pytest

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.search_certificate import (
    SearchStructureCertificate, SimulatorEvidence, minimum_required_interactions,
    navigation_lower_bound)
from rummagebench.authoring.tasks.tiers import load_all_tiers, load_tier
from rummagebench.authoring.tasks.tiers.base import Bound, TierSpec


def _cert(**kw) -> SearchStructureCertificate:
    base = dict(tier_proposed="easy", mode="in_container", room_count=1,
                candidate_slot_count=3, containment_depth=1, open_depth=1,
                rearrangement_depth=0, lookalike_count=0,
                target_slot_type="drawer", navigation_lower_bound=4,
                minimum_required_interactions=7)
    base.update(kw)
    return SearchStructureCertificate(**base)


def _good_evidence(**kw) -> SimulatorEvidence:
    base = dict(builder_ok=True, placements_verified=True, settle_ok=True,
                start_visibility=0.0, closed_visibility=0.0,
                pre_reveal_graspable=False, post_reveal_graspable=True,
                revealed_visibility=0.8, reachable_candidate_slot_count=3,
                rearrangement_depth=0, oracle_solvable=True, oracle_depth=3,
                replay_success=True, certified_execution_steps=9)
    base.update(kw)
    return SimulatorEvidence(**base)


@pytest.fixture(scope="module")
def tiers():
    return {t.tier: t for t in load_all_tiers()}


def test_tier_assets_load_and_keep_16_steps(tiers):
    assert set(tiers) == {"easy", "medium"}
    for spec in tiers.values():
        assert spec.budget.max_planning_steps == 16  # frozen horizon
        assert spec.gates.start_visibility_max == 0.0
    assert tiers["easy"].planner_enabled
    assert not tiers["medium"].planner_enabled


@pytest.mark.parametrize("fields,expected", [
    (dict(), ["easy"]),
    (dict(mode="on_surface", containment_depth=0, open_depth=0,
          target_slot_type="countertop"), ["easy"]),
    (dict(mode="rummage", rearrangement_depth=2, lookalike_count=1), ["medium"]),
    (dict(mode="rummage", rearrangement_depth=4, lookalike_count=4,
          candidate_slot_count=5), ["medium"]),
    # easy structure violated by a lookalike, but not medium either (no rearr.)
    (dict(lookalike_count=1), []),
    # too wide for easy (5 slots) and no rearrangement for medium
    (dict(candidate_slot_count=5), []),
    # one room with rearrangement 5 -> beyond medium cap
    (dict(rearrangement_depth=5, lookalike_count=1), []),
    # two rooms -> neither easy nor medium
    (dict(room_count=2, candidate_slot_count=6, rearrangement_depth=1), []),
])
def test_classification_is_disjoint_and_mechanical(tiers, fields, expected):
    assert gates.classify(_cert(**fields), tiers.values()) == expected


def test_proposed_label_is_not_trusted(tiers):
    # planner says "easy" but structure is medium -> easy gate rejects
    cert = _cert(tier_proposed="easy", rearrangement_depth=2, lookalike_count=1)
    codes = {r.code for r in gates.structural_violations(cert, tiers["easy"])}
    assert gates.REARRANGEMENT_DEPTH_MISMATCH in codes
    assert gates.TIER_GATE_FAILED in codes


def test_easy_in_container_requires_open_structurally(tiers):
    cert = _cert(open_depth=0)
    codes = {r.code for r in gates.structural_violations(cert, tiers["easy"])}
    assert codes == {gates.OPEN_NOT_REQUIRED}


def test_horizon_lower_bound_rejects_before_gpu(tiers):
    easy = tiers["easy"]
    limit = easy.budget.max_planning_steps - easy.budget.search_reserve_steps
    ok = _cert(minimum_required_interactions=limit)
    bad = _cert(minimum_required_interactions=limit + 1)
    assert not gates.structural_violations(ok, easy)
    assert [r.code for r in gates.structural_violations(bad, easy)] == [
        gates.HORIZON_LOWER_BOUND]


def test_minimum_required_interactions_formula():
    # OPEN + 2 x (GRASP, PLACE) + GRASP target + REPORT_DONE + nav
    assert minimum_required_interactions(1, 2, 5) == 1 + 4 + 1 + 1 + 5
    assert minimum_required_interactions(0, 0, 0) == 2


def test_navigation_lower_bound_is_admissible():
    # 2.5 m straight ahead along the heading: 3 MOVEs (<=1 m), no TURN
    assert navigation_lower_bound((0, 0), 0.0, (2.5, 0), 0.0, 1.0, 180.0) == 3
    # straight behind: backward MOVE is legal -> no turn needed
    assert navigation_lower_bound((0, 0), 0.0, (-2.5, 0), 0.0, 1.0, 180.0) == 3
    # sideways: one TURN + MOVEs
    assert navigation_lower_bound((0, 0), 0.0, (0, 1.5), 0.0, 1.0, 180.0) == 3
    # in place, facing away: turns only
    assert navigation_lower_bound((0, 0), 0.0, (0, 0), math.pi, 1.0, 180.0) == 1
    assert navigation_lower_bound((0, 0), 0.0, (0.01, 0), 0.0, 1.0, 180.0) == 0


def test_simulator_gate_happy_path_certifies(tiers):
    cert = _cert()
    cert.attach_evidence(_good_evidence())
    assert gates.certify(cert, tiers["easy"]) == []
    assert cert.status == "certified" and cert.tier_certified == "easy"
    assert cert.start_visibility == 0.0 and cert.revealed_visibility == 0.8


@pytest.mark.parametrize("override,code", [
    (dict(start_visibility=0.02), gates.START_VISIBLE),
    (dict(closed_visibility=0.1), gates.OPEN_NOT_REQUIRED),
    # OPEN not causally needed when the closed target is graspable
    (dict(pre_reveal_graspable=True), gates.OPEN_NOT_REQUIRED),
    (dict(revealed_visibility=0.62), gates.REVEAL_TOO_WEAK),  # < 0.65 margin
    (dict(post_reveal_graspable=False), gates.TARGET_NOT_GRASPABLE),
    (dict(oracle_solvable=False), gates.PHYSICAL_UNSOLVABLE),
    (dict(replay_success=False), gates.PHYSICAL_UNSOLVABLE),
    (dict(certified_execution_steps=17), gates.HORIZON_LOWER_BOUND),
    (dict(rearrangement_depth=1), gates.REARRANGEMENT_DEPTH_MISMATCH),
    (dict(reachable_candidate_slot_count=1), gates.SEARCH_WIDTH_MISMATCH),
    (dict(settle_ok=False), gates.SETTLE_FAILED),
    (dict(start_visibility=None), gates.START_VISIBLE),  # unmeasured != pass
])
def test_simulator_gate_failures(tiers, override, code):
    cert = _cert()
    cert.attach_evidence(_good_evidence(**override))
    codes = {r.code for r in gates.certify(cert, tiers["easy"])}
    assert code in codes
    assert cert.status == "rejected"


def test_benchmark_threshold_vs_generation_margin(tiers):
    # 0.62 passes the published 0.60 threshold but not the 0.65 generator margin
    cert = _cert()
    cert.attach_evidence(_good_evidence(revealed_visibility=0.62))
    assert gates.simulator_violations(cert, tiers["easy"], use_generation_margin=False) == []
    assert gates.simulator_violations(cert, tiers["easy"], use_generation_margin=True)


def test_no_evidence_never_certifies(tiers):
    cert = _cert()
    assert gates.certify(cert, tiers["easy"])
    assert cert.status == "rejected"


def test_medium_premature_grasp(tiers):
    cert = _cert(tier_proposed="medium", mode="rummage", rearrangement_depth=2,
                 lookalike_count=1)
    cert.attach_evidence(_good_evidence(pre_reveal_graspable=True,
                                        rearrangement_depth=2))
    codes = {r.code for r in gates.certify(cert, tiers["medium"])}
    assert codes == {gates.TARGET_PREMATURELY_GRASPABLE}


def test_certificate_roundtrip():
    cert = _cert()
    cert.attach_evidence(_good_evidence())
    again = gates.SearchStructureCertificate.from_dict(cert.to_dict())
    assert again.to_dict() == cert.to_dict()


def test_bound_forms():
    assert Bound.model_validate(3).contains(3) and not Bound.model_validate(3).contains(4)
    assert Bound.model_validate([2, 4]).contains(4)
    b = Bound.model_validate({"min": 4})
    assert b.contains(100) and not b.contains(3)
    with pytest.raises(Exception):
        Bound.model_validate([5, 1])


def test_generator_margin_cannot_be_looser():
    raw = load_tier("easy_v1").model_dump()
    raw["gates"]["modes"]["in_container"]["generator_visibility_margin"] = 0.5
    with pytest.raises(Exception):
        TierSpec.model_validate(raw)


def test_rejection_code_whitelist():
    with pytest.raises(ValueError):
        gates.Rejection("MADE_UP", "x")
