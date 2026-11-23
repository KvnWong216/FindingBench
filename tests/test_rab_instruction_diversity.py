"""Unit tests for diverse & unambiguous natural-language RAB instruction authoring."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from rab_instruction_diversity import (  # noqa: E402
    COUNTER_COFFEE_GRINDER_LAYOUTS,
    OBJECT_EXPRESSIONS,
    compose_diverse_instruction,
    verify_episode_instruction,
)


def test_counter_instructions_always_disambiguate_white_counter_and_vary():
    seen_dest = set()
    seen_inst = set()
    for i in range(40):
        lid = 34 if i % 5 == 0 else 22
        tid = f"rab_F{i + 1:03d}"
        inst = compose_diverse_instruction(
            seed_key=tid,
            target_assets=["cloth_clear_house"],
            delivery_fixture="counter",
            search_scope="living_room",
            layout_id=lid,
        )
        seen_inst.add(inst)
        ep = {
            "instruction": inst,
            "delivery": {"fixture": "counter"},
            "arrangement": {"kind": "articulated_drawer"},
            "placed": [{"asset": "cloth_clear_house", "target": True}],
        }
        assert verify_episode_instruction(ep, inst) == []
        low = inst.lower()
        assert "white" in low or "light-colored" in low
        for tag in ("white kitchen counter", "white countertop", "white counter", "light-colored kitchen counter"):
            if tag in low:
                seen_dest.add(tag)
    assert len(seen_inst) >= 15
    assert len(seen_dest) >= 3
    assert 34 in COUNTER_COFFEE_GRINDER_LAYOUTS


def test_all_catalog_assets_pass_singular_and_plural_disambiguation():
    for asset in sorted(OBJECT_EXPRESSIONS):
        for cnt, kind in ((1, "articulated_drawer"), (2, "occluded_open")):
            scope = "living_room" if kind == "articulated_drawer" else "kitchen_and_living"
            fixture = "side" if cnt == 1 else "coffee"
            inst = compose_diverse_instruction(
                seed_key=f"test_{asset}_{cnt}",
                target_assets=[asset] * cnt,
                delivery_fixture=fixture,
                search_scope=scope,
                layout_id=17,
            )
            ep = {
                "instruction": inst,
                "delivery": {"fixture": fixture},
                "arrangement": {"kind": kind},
                "placed": [{"asset": asset, "target": True} for _ in range(cnt)],
            }
            assert verify_episode_instruction(ep, inst) == [], f"{asset} (cnt={cnt}) failed: {inst}"


def test_validator_rejects_ambiguous_or_non_natural_prompts():
    ep_counter = {
        "delivery": {"fixture": "counter"},
        "arrangement": {"kind": "articulated_drawer"},
        "placed": [{"asset": "meal_knife", "target": True}],
    }
    # Ambiguous bare "kitchen counter" must fail
    errs_ambig = verify_episode_instruction(
        ep_counter,
        "Could you look for the table knife in the living room and put it on the kitchen counter?",
    )
    assert "counter_missing_white_or_light_disambiguator" in errs_ambig

    # Non-natural USD token leak must fail
    errs_usd = verify_episode_instruction(
        ep_counter,
        "Put the table knife from the living room onto CounterTop_17 white kitchen counter.",
    )
    assert any(e.startswith("non_natural_token:") for e in errs_usd)
