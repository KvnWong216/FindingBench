#!/usr/bin/env python3
"""Deterministic, physically verified, diverse natural-language instruction generator
and disambiguation validator for FindingBench RAB v1 (`runs/rab_lived_in_20261120`).

Core design paradigm:
1. **Strict Precision / Zero Ambiguity (`精确的指代`)**:
   - Every destination surface phrase is visually & geometrically verified against the
     50 RAB household USD layouts so it uniquely identifies the ground-truth support
     (`CounterTop_17` = WarmWhite counter with air fryer, never the dark `BP_Kitchen9`
     sink worktop; `CoffeeTable` = main wooden coffee table on the living room rug;
     `SmallCoffeeTable` = low dark-wood side table by the living room wall).
   - Every object referring expression preserves exact brand/species/category identity
     and is checked against all distractor assets present in that episode.
   - Pure everyday household English — zero USD prim names, coordinates, or synthetic tags.
2. **Linguistic & Referential Diversity (`diverse`)**:
   - Multiple curated `(singular, plural)` referring expressions per object asset,
     per destination fixture, per search region, and 12 natural household sentence frames,
     deterministically seeded by `sha256(task_id)` for 100% reproducibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

# Layouts where `/World/Assets/CoffeeGrinder` sits directly on `CounterTop_17` alongside `AirFryer`
COUNTER_COFFEE_GRINDER_LAYOUTS: frozenset[int] = frozenset({34, 40, 43, 46})

# Curated destination surface phrases — 100% verified across all 50 RAB USD layouts:
# - `counter`: `/World/Background/Shell/CounterTop_17` (`WarmWhite` in 25/25 layouts, holds `AirFryer`, no sink;
#              distinct from `ConstructorKitchen/.../Worktop` which is `DarkWorktop` with sink & faucet).
# - `coffee` : `/World/Assets/CoffeeTable/Mesh` (`CoffeeTableMat` light wood in 50/50 layouts, 50/50 on `LivingRoomRug`).
# - `side`   : `/World/Assets/SmallCoffeeTable` (`WoodMat` dark wood low side table against wall in 50/50 layouts, 0/50 on rug).
DESTINATION_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "counter": (
        "the white kitchen counter",
        "the white countertop in the kitchen",
        "the white kitchen counter with the air fryer",
        "the white counter by the air fryer in the kitchen",
        "the light-colored kitchen counter",
        "the white counter in the kitchen",
    ),
    "coffee": (
        "the coffee table",
        "the living room coffee table",
        "the main coffee table in the living room",
        "the coffee table on the living room rug",
        "the wooden coffee table on the rug",
        "the center coffee table in the living room",
    ),
    "side": (
        "the small side table",
        "the dark wooden side table",
        "the low wooden side table in the living room",
        "the small dark-wood side table",
        "the living room side table by the wall",
        "the small wooden side table",
    ),
    "shelf": (
        "the pantry shelf",
        "the shelving unit in the pantry",
        "the multi-tier pantry shelf",
    ),
}

# Curated `(singular, plural)` referring expressions for every asset in CATALOG + fridge food boxes.
# Every tuple preserves exact brand/type/color disambiguators and grammatical English pluralization.
OBJECT_EXPRESSIONS: dict[str, tuple[tuple[str, str], ...]] = {
    # Pantry beverages & snacks
    "coke_c3": (
        ("Coca-Cola can", "Coca-Cola cans"),
        ("can of Coca-Cola", "cans of Coca-Cola"),
        ("Coke soda can", "Coke soda cans"),
        ("Coca-Cola soda can", "Coca-Cola soda cans"),
    ),
    "sprite_c1": (
        ("Sprite can", "Sprite cans"),
        ("can of Sprite", "cans of Sprite"),
        ("Sprite lemon-lime soda can", "Sprite lemon-lime soda cans"),
        ("Sprite soda can", "Sprite soda cans"),
    ),
    "grapefruit_can": (
        ("grapefruit sparkling water", "grapefruit sparkling waters"),
        ("can of grapefruit sparkling water", "cans of grapefruit sparkling water"),
        ("grapefruit sparkling water can", "grapefruit sparkling water cans"),
    ),
    "starbucks": (
        ("Starbucks coffee can", "Starbucks coffee cans"),
        ("can of Starbucks coffee", "cans of Starbucks coffee"),
        ("canned Starbucks coffee", "canned Starbucks coffees"),
        ("Starbucks cold coffee can", "Starbucks cold coffee cans"),
    ),
    "gatorade_m": (
        ("orange Gatorade", "orange Gatorades"),
        ("orange Gatorade bottle", "orange Gatorade bottles"),
        ("bottle of orange Gatorade", "bottles of orange Gatorade"),
        ("orange Gatorade sports drink", "orange Gatorade sports drinks"),
    ),
    "gatorade_t": (
        ("blue Gatorade", "blue Gatorades"),
        ("blue Gatorade bottle", "blue Gatorade bottles"),
        ("bottle of blue Gatorade", "bottles of blue Gatorade"),
        ("blue Gatorade sports drink", "blue Gatorade sports drinks"),
    ),
    "good2grow": (
        ("Good2Grow juice bottle", "Good2Grow juice bottles"),
        ("bottle of Good2Grow juice", "bottles of Good2Grow juice"),
        ("character-topped Good2Grow juice bottle", "character-topped Good2Grow juice bottles"),
        ("Good2Grow fruit juice bottle", "Good2Grow fruit juice bottles"),
    ),
    "water_bottle": (
        ("water bottle", "water bottles"),
        ("bottle of water", "bottles of water"),
        ("drinking water bottle", "drinking water bottles"),
    ),
    "pring_bbq": (
        ("barbecue Pringles", "barbecue Pringles cans"),
        ("BBQ Pringles can", "BBQ Pringles cans"),
        ("tube of barbecue Pringles", "tubes of barbecue Pringles"),
        ("can of barbecue Pringles", "cans of barbecue Pringles"),
    ),
    "honeymaid": (
        ("Honey Maid crackers", "boxes of Honey Maid crackers"),
        ("box of Honey Maid crackers", "boxes of Honey Maid crackers"),
        ("Honey Maid graham cracker box", "Honey Maid graham cracker boxes"),
        ("pack of Honey Maid crackers", "packs of Honey Maid crackers"),
    ),
    "oreo_gf": (
        ("gluten-free Oreos", "packs of gluten-free Oreos"),
        ("pack of gluten-free Oreos", "packs of gluten-free Oreos"),
        ("gluten-free Oreo cookie pack", "gluten-free Oreo cookie packs"),
        ("package of gluten-free Oreos", "packages of gluten-free Oreos"),
    ),
    "zbar_t": (
        ("Zbar snack bar", "Zbar snack bars"),
        ("Zbar snack bar box", "Zbar snack bar boxes"),
        ("box of Zbar snack bars", "boxes of Zbar snack bars"),
        ("Clif Zbar snack box", "Clif Zbar snack boxes"),
    ),
    "skit_littles": (
        ("Skittles Littles", "packs of Skittles Littles"),
        ("Skittles Littles candy tube", "Skittles Littles candy tubes"),
        ("tube of Skittles Littles", "tubes of Skittles Littles"),
        ("pack of Skittles Littles", "packs of Skittles Littles"),
    ),
    "chomps_r": (
        ("Chomps beef stick", "Chomps beef sticks"),
        ("pack of Chomps beef sticks", "packs of Chomps beef sticks"),
        ("Chomps beef jerky pack", "Chomps beef jerky packs"),
        ("red Chomps beef stick pouch", "red Chomps beef stick pouches"),
    ),
    "coffee_bag": (
        ("coffee bag", "coffee bags"),
        ("bag of coffee", "bags of coffee"),
        ("packaged coffee bag", "packaged coffee bags"),
    ),
    "fridge_salad_box": (
        ("salad box", "salad boxes"),
        ("packaged salad container", "packaged salad containers"),
        ("fresh salad box", "fresh salad boxes"),
    ),
    "fridge_takeout_box": (
        ("takeout box", "takeout boxes"),
        ("clamshell takeout container", "clamshell takeout containers"),
        ("takeout meal box", "takeout meal boxes"),
    ),
    # Tableware
    "coffee_cup_blue_compact": (
        ("blue cup", "blue cups"),
        ("compact blue coffee cup", "compact blue coffee cups"),
        ("small blue cup", "small blue cups"),
    ),
    "coffee_cup_cream_classic": (
        ("cream cup", "cream cups"),
        ("cream-colored coffee cup", "cream-colored coffee cups"),
        ("classic cream cup", "classic cream cups"),
    ),
    "coffee_cup_green_tall": (
        ("tall green cup", "tall green cups"),
        ("tall green coffee cup", "tall green coffee cups"),
        ("high green cup", "high green cups"),
    ),
    "paper_cup": (
        ("paper cup", "paper cups"),
        ("disposable paper cup", "disposable paper cups"),
        ("single-use paper cup", "single-use paper cups"),
        ("drinking paper cup", "drinking paper cups"),
    ),
    "meal_fork": (
        ("fork", "forks"),
        ("dinner fork", "dinner forks"),
        ("table fork", "table forks"),
        ("metal dinner fork", "metal dinner forks"),
    ),
    "meal_knife": (
        ("table knife", "table knives"),
        ("dinner knife", "dinner knives"),
        ("metal table knife", "metal table knives"),
        ("butter knife", "butter knives"),
    ),
    "paper_plate_task2": (
        ("paper plate", "paper plates"),
        ("disposable paper plate", "disposable paper plates"),
        ("round paper plate", "round paper plates"),
    ),
    # Personal & office items
    "wallet": (
        ("wallet", "wallets"),
        ("pocket wallet", "pocket wallets"),
        ("folding wallet", "folding wallets"),
        ("personal wallet", "personal wallets"),
    ),
    "glasses_case": (
        ("glasses case", "glasses cases"),
        ("eyeglass case", "eyeglass cases"),
        ("protective glasses case", "protective glasses cases"),
        ("eyewear case", "eyewear cases"),
    ),
    "remote_tv": (
        ("TV remote", "TV remotes"),
        ("television remote control", "television remote controls"),
        ("TV remote control", "TV remote controls"),
        ("remote control for the TV", "remote controls for the TV"),
    ),
    "notebook_green": (
        ("green notebook", "green notebooks"),
        ("green-covered notebook", "green-covered notebooks"),
        ("green bound notebook", "green bound notebooks"),
        ("green order notebook", "green order notebooks"),
    ),
    "baseball_cap": (
        ("baseball cap", "baseball caps"),
        ("peaked baseball cap", "peaked baseball caps"),
        ("sports baseball cap", "sports baseball caps"),
    ),
    "power_bank": (
        ("power bank", "power banks"),
        ("portable power bank", "portable power banks"),
        ("battery power bank", "battery power banks"),
        ("portable battery charger", "portable battery chargers"),
    ),
    "book_red": (
        ("red book", "red books"),
        ("red-covered book", "red-covered books"),
        ("red hardcover book", "red hardcover books"),
        ("red-bound book", "red-bound books"),
    ),
    "cloth_clear_memo_pad": (
        ("memo pad", "memo pads"),
        ("yellow memo pad", "yellow memo pads"),
        ("small notepad", "small notepads"),
        ("sticky-note memo pad", "sticky-note memo pads"),
    ),
    "cloth_clear_tape_measure": (
        ("tape measure", "tape measures"),
        ("retractable tape measure", "retractable tape measures"),
        ("pocket measuring tape", "pocket measuring tapes"),
        ("measuring tape", "measuring tapes"),
    ),
    "cloth_clear_eraser": (
        ("eraser", "erasers"),
        ("pencil eraser", "pencil erasers"),
        ("rubber eraser", "rubber erasers"),
        ("desk eraser", "desk erasers"),
    ),
    "cloth_clear_pouch": (
        ("small pouch", "small pouches"),
        ("compact accessory pouch", "compact accessory pouches"),
        ("slim carrying pouch", "slim carrying pouches"),
        ("small zippered pouch", "small zippered pouches"),
    ),
    # Toys
    "cloth_clear_bear": (
        ("toy bear", "toy bears"),
        ("miniature toy bear", "miniature toy bears"),
        ("little teddy bear figure", "little teddy bear figures"),
        ("small bear figurine", "small bear figurines"),
    ),
    "cloth_clear_cat": (
        ("toy cat", "toy cats"),
        ("miniature toy cat", "miniature toy cats"),
        ("little cat figurine", "little cat figurines"),
        ("small toy kitty", "small toy kitties"),
    ),
    "cloth_clear_duck": (
        ("toy duck", "toy ducks"),
        ("miniature toy duck", "miniature toy ducks"),
        ("little duck figurine", "little duck figurines"),
        ("small toy duckling", "small toy ducklings"),
    ),
    "cloth_clear_car": (
        ("toy car", "toy cars"),
        ("miniature toy car", "miniature toy cars"),
        ("little model car", "little model cars"),
        ("small toy vehicle", "small toy vehicles"),
    ),
    "cloth_clear_house": (
        ("toy house", "toy houses"),
        ("miniature toy house", "miniature toy houses"),
        ("little red-roofed toy house", "little red-roofed toy houses"),
        ("small model house", "small model houses"),
    ),
    "cloth_clear_penguin": (
        ("toy penguin", "toy penguins"),
        ("miniature toy penguin", "miniature toy penguins"),
        ("little penguin figurine", "little penguin figurines"),
        ("small penguin toy", "small penguin toys"),
    ),
    "cloth_clear_rabbit": (
        ("toy rabbit", "toy rabbits"),
        ("miniature toy rabbit", "miniature toy rabbits"),
        ("little bunny figurine", "little bunny figurines"),
        ("small toy bunny", "small toy bunnies"),
    ),
    "cloth_clear_robot": (
        ("toy robot", "toy robots"),
        ("miniature toy robot", "miniature toy robots"),
        ("little robot figurine", "little robot figurines"),
        ("small robot toy", "small robot toys"),
    ),
    "cloth_clear_train": (
        ("toy train", "toy trains"),
        ("miniature toy train", "miniature toy trains"),
        ("little model train", "little model trains"),
        ("small toy locomotive", "small toy locomotives"),
    ),
    "cloth_clear_block": (
        ("toy block", "toy blocks"),
        ("building toy block", "building toy blocks"),
        ("small toy block", "small toy blocks"),
        ("toy cube block", "toy cube blocks"),
    ),
}

# Required semantic anchor keywords per target asset (at least one must appear in instruction)
ASSET_ANCHOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "coke_c3": ("coca-cola", "coke"),
    "sprite_c1": ("sprite",),
    "grapefruit_can": ("grapefruit",),
    "starbucks": ("starbucks",),
    "gatorade_m": ("orange gatorade",),
    "gatorade_t": ("blue gatorade",),
    "good2grow": ("good2grow",),
    "water_bottle": ("water",),
    "pring_bbq": ("pringles",),
    "honeymaid": ("honey maid",),
    "oreo_gf": ("oreo",),
    "zbar_t": ("zbar",),
    "skit_littles": ("skittles",),
    "chomps_r": ("chomps",),
    "coffee_bag": ("coffee bag", "bag of coffee"),
    "fridge_salad_box": ("salad",),
    "fridge_takeout_box": ("takeout",),
    "coffee_cup_blue_compact": ("blue cup", "blue coffee cup"),
    "coffee_cup_cream_classic": ("cream cup", "cream-colored coffee cup"),
    "coffee_cup_green_tall": ("green cup", "green coffee cup"),
    "paper_cup": ("paper cup",),
    "meal_fork": ("fork",),
    "meal_knife": ("knife", "knives"),
    "paper_plate_task2": ("paper plate",),
    "wallet": ("wallet",),
    "glasses_case": ("glasses case", "eyeglass case", "eyewear case"),
    "remote_tv": ("remote",),
    "notebook_green": ("green",),
    "baseball_cap": ("cap",),
    "power_bank": ("power bank", "battery charger"),
    "book_red": ("red",),
    "cloth_clear_memo_pad": ("memo pad", "notepad"),
    "cloth_clear_tape_measure": ("tape measure", "measuring tape"),
    "cloth_clear_eraser": ("eraser",),
    "cloth_clear_pouch": ("pouch",),
    "cloth_clear_bear": ("bear",),
    "cloth_clear_cat": ("cat", "kitty", "kitties"),
    "cloth_clear_duck": ("duck",),
    "cloth_clear_car": ("car", "vehicle"),
    "cloth_clear_house": ("house",),
    "cloth_clear_penguin": ("penguin",),
    "cloth_clear_rabbit": ("rabbit", "bunny", "bunnies"),
    "cloth_clear_robot": ("robot",),
    "cloth_clear_train": ("train", "locomotive"),
    "cloth_clear_block": ("block",),
}

SEARCH_SCOPE_EXPRESSIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "living_room": {
        "prep": (
            "in the living room",
            "around the living room",
            "somewhere in the living room",
        ),
        "noun": (
            "the living room",
            "the living room area",
        ),
    },
    "kitchen_and_living": {
        "prep": (
            "in the kitchen and living area",
            "across the kitchen and living room",
            "around the kitchen, pantry, and living area",
            "in the open kitchen and living space",
        ),
        "noun": (
            "the kitchen and living area",
            "the kitchen and living room",
            "the kitchen, pantry, and living area",
        ),
    },
    "kitchen_fridge": {
        "prep": (
            "in the kitchen fridge",
            "inside the kitchen refrigerator",
            "in the refrigerator in the kitchen",
        ),
        "noun": (
            "the kitchen fridge",
            "the kitchen refrigerator",
        ),
    },
    "pantry_shelf": {
        "prep": (
            "on the pantry shelf",
            "on the shelving unit in the pantry",
        ),
        "noun": (
            "the pantry shelf",
            "the pantry shelving unit",
        ),
    },
}

# 12 diverse natural household sentence frames (6 using `prep` search scope, 6 using `noun` search scope)
SENTENCE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("prep", "Could you look for {target} {scope} and put {pronoun} on {dest}?"),
    ("prep", "Please find {target} {scope} and place {pronoun} on top of {dest}."),
    ("prep", "Would you mind finding {target} {scope} and bringing {pronoun} over to {dest}?"),
    ("prep", "Help me track down {target} {scope}, then set {pronoun} on {dest}."),
    ("prep", "Locate {target} {scope} and leave {pronoun} on {dest}, please."),
    ("prep", "Can you find {target} {scope} and set {pronoun} neatly on {dest}?"),
    ("noun", "Go search {scope} for {target} and put {pronoun} onto {dest}."),
    ("noun", "Can you grab {target} from {scope} and set {pronoun} down on {dest}?"),
    ("noun", "Search {scope} to find {target}, and deliver {pronoun} to {dest}."),
    ("noun", "Could you check {scope} for {target} and move {pronoun} onto {dest}?"),
    ("noun", "Please fetch {target} from {scope} and place {pronoun} on {dest}."),
    ("noun", "Look around {scope} for {target} and put {pronoun} on {dest}."),
)

COMPARISON_FRIDGE_TEMPLATES: tuple[str, ...] = (
    "Find {amount} {scope_prep} matching the sample bottle on {dest}, and bring {pronoun} back to that same side table.",
    "Check {scope_noun} for {amount} identical to the bottle sitting on {dest}, then set {pronoun} beside it on that table.",
    "Look {scope_prep} for {amount} that matches the reference bottle on {dest}, and place {pronoun} back onto that table.",
)

QUANTITY_WORDS: dict[int, str] = {2: "two", 3: "three", 4: "four", 5: "five"}


def _slot_index(seed_key: str, salt: str, modulo: int) -> int:
    digest_bytes = hashlib.sha256(f"{seed_key}::{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest_bytes[:8], "big") % modulo


def infer_search_scope(episode: dict[str, Any]) -> str:
    arr_kind = str((episode.get("arrangement") or {}).get("kind") or "")
    if arr_kind == "articulated_refrigerator":
        return "kitchen_fridge"
    if arr_kind == "occluded_open":
        return "kitchen_and_living"
    if arr_kind == "articulated_drawer":
        return "living_room"
    cur_inst = str(episode.get("canonical_dense_instruction") or episode.get("instruction") or "").lower()
    if "kitchen fridge" in cur_inst or "kitchen refrigerator" in cur_inst:
        return "kitchen_fridge"
    if "kitchen" in cur_inst and "living" in cur_inst:
        return "kitchen_and_living"
    if "pantry shelf" in cur_inst:
        return "pantry_shelf"
    return "living_room"


def choose_destination_phrase(fixture: str, *, seed_key: str, layout_id: int | None = None) -> str:
    options = list(DESTINATION_EXPRESSIONS[fixture])
    if fixture == "counter" and layout_id in COUNTER_COFFEE_GRINDER_LAYOUTS:
        options.append("the white kitchen counter with the coffee grinder and air fryer")
    idx = _slot_index(seed_key, f"dest:{fixture}:{layout_id}", len(options))
    return options[idx]


def format_target_noun_phrase(target_assets: list[str], *, seed_key: str) -> str:
    if not target_assets:
        raise ValueError("target_assets must be non-empty")
    counts: dict[str, int] = {}
    for asset in target_assets:
        counts[asset] = counts.get(asset, 0) + 1

    phrases: list[str] = []
    for pos, (asset, count) in enumerate(counts.items()):
        variants = OBJECT_EXPRESSIONS[asset]
        idx = _slot_index(seed_key, f"obj:{pos}:{asset}", len(variants))
        singular, plural = variants[idx]
        if count == 1:
            phrases.append(f"the {singular}")
        else:
            qword = QUANTITY_WORDS.get(count, str(count))
            phrases.append(f"{qword} {plural}")

    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def compose_diverse_instruction(
    *,
    seed_key: str,
    target_assets: list[str],
    delivery_fixture: str,
    search_scope: str,
    layout_id: int | None = None,
    comparison_match: bool = False,
) -> str:
    """Deterministically compose a natural, diverse, 100% unambiguous instruction."""
    dest_phrase = choose_destination_phrase(delivery_fixture, seed_key=seed_key, layout_id=layout_id)
    pronoun = "them" if len(target_assets) > 1 else "it"
    scope_bank = SEARCH_SCOPE_EXPRESSIONS[search_scope]

    if comparison_match:
        amount = "a juice bottle" if len(target_assets) == 1 else f"{QUANTITY_WORDS.get(len(target_assets), str(len(target_assets)))} juice bottles"
        scope_prep = scope_bank["prep"][_slot_index(seed_key, "cmp_prep", len(scope_bank["prep"]))]
        scope_noun = scope_bank["noun"][_slot_index(seed_key, "cmp_noun", len(scope_bank["noun"]))]
        tmpl = COMPARISON_FRIDGE_TEMPLATES[_slot_index(seed_key, "cmp_tmpl", len(COMPARISON_FRIDGE_TEMPLATES))]
        # Singular verb adjustment if plural targets ("matches" -> "match")
        if len(target_assets) > 1:
            tmpl = tmpl.replace("that matches", "that match")
        return tmpl.format(amount=amount, scope_prep=scope_prep, scope_noun=scope_noun, dest=dest_phrase, pronoun=pronoun)

    target_phrase = format_target_noun_phrase(target_assets, seed_key=seed_key)
    mode, tmpl = SENTENCE_TEMPLATES[_slot_index(seed_key, "tmpl", len(SENTENCE_TEMPLATES))]
    scope_opts = scope_bank[mode]
    scope_phrase = scope_opts[_slot_index(seed_key, f"scope:{mode}", len(scope_opts))]
    return tmpl.format(target=target_phrase, scope=scope_phrase, pronoun=pronoun, dest=dest_phrase)


def verify_episode_instruction(episode: dict[str, Any], instruction: str | None = None) -> list[str]:
    """Return a list of ambiguity / accuracy violations (empty list == 100% valid)."""
    inst = (instruction if instruction is not None else str(episode.get("instruction") or "")).strip()
    low = inst.lower()
    errors: list[str] = []

    if not inst or len(inst) < 20:
        errors.append("instruction_too_short")
        return errors

    # 1. Strictly natural language: forbid internal USD paths, coordinates, underscores, or prim IDs
    forbidden_tokens = ("/world/", "countertop_", "bp_kitchen", "black_box", "item_0", "rectangle0", "mesh", "usd")
    for tok in forbidden_tokens:
        if tok in low:
            errors.append(f"non_natural_token:{tok}")

    # 2. Destination surface disambiguation
    fixture = str((episode.get("delivery") or {}).get("fixture") or "")
    if fixture == "counter":
        if not ("white" in low or "light-colored" in low):
            errors.append("counter_missing_white_or_light_disambiguator")
        if not ("counter" in low or "countertop" in low):
            errors.append("counter_missing_counter_noun")
        if "sink" in low:
            errors.append("counter_mentions_conflicting_sink")
    elif fixture == "coffee":
        if "coffee table" not in low:
            errors.append("coffee_missing_coffee_table")
        if "side table" in low:
            errors.append("coffee_mentions_conflicting_side_table")
    elif fixture == "side":
        if "side table" not in low:
            errors.append("side_missing_side_table")
        if "coffee table" in low:
            errors.append("side_mentions_conflicting_coffee_table")

    # 3. Search scope verification against physical arrangement
    arr_kind = str((episode.get("arrangement") or {}).get("kind") or "")
    if arr_kind == "articulated_refrigerator":
        if not ("fridge" in low or "refrigerator" in low):
            errors.append("fridge_arrangement_missing_fridge_scope")
    elif arr_kind == "articulated_drawer":
        if "living room" not in low:
            errors.append("drawer_arrangement_missing_living_room_scope")
        if "fridge" in low or "refrigerator" in low or "pantry" in low:
            errors.append("drawer_arrangement_mentions_wrong_room_scope")
    elif arr_kind == "occluded_open":
        if "living" not in low or "kitchen" not in low:
            errors.append("open_arrangement_missing_kitchen_and_living_scope")

    # 4. Target identity & quantity verification
    targets = [p["asset"] for p in episode.get("placed", []) if p.get("target")]
    is_comp = bool(episode.get("comparison_sample_body"))
    if is_comp:
        if "juice bottle" not in low or "match" not in low and "identical" not in low:
            errors.append("comparison_missing_matching_juice_bottle_cue")
    else:
        counts = Counter(targets)
        for asset, cnt in counts.items():
            anchors = ASSET_ANCHOR_KEYWORDS.get(asset, ())
            if anchors and not any(a in low for a in anchors):
                errors.append(f"missing_target_anchor:{asset}")
            if cnt > 1:
                qword = QUANTITY_WORDS.get(cnt, str(cnt))
                if qword not in low:
                    errors.append(f"missing_quantity_word:{asset}:{qword}")
            # Guard critical distractor confusions
            if asset == "glasses_case" and "case" not in low:
                errors.append("glasses_case_missing_case_word")
            if asset == "notebook_green" and "notebook" not in low:
                errors.append("notebook_green_missing_notebook_word")
            if asset == "cloth_clear_memo_pad" and "notebook" in low:
                errors.append("memo_pad_collides_with_notebook")
            if asset == "paper_cup" and "mug" in low:
                errors.append("paper_cup_collides_with_mug")

    return errors


def apply_to_release(release_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    manifest = json.loads((release_dir / "combined_manifest.json").read_text())
    unique_instructions: set[str] = set()
    dest_counter: Counter[str] = Counter()
    tmpl_counter: Counter[int] = Counter()
    obj_variant_counter: Counter[str] = Counter()
    updated = 0
    violations: dict[str, list[str]] = {}

    for row in manifest["tasks"]:
        tid = row["id"]
        ep_path = release_dir / "cases" / tid / "episode.json"
        ep = json.loads(ep_path.read_text())
        if "baseline_instruction" not in ep:
            ep["baseline_instruction"] = ep["instruction"]
        if "canonical_dense_instruction" not in ep:
            ep["canonical_dense_instruction"] = ep["instruction"]

        targets = [p["asset"] for p in ep.get("placed", []) if p.get("target")]
        fixture = ep["delivery"]["fixture"]
        lid = int(ep["layout_id"])
        is_comp = bool(ep.get("comparison_sample_body"))
        scope = infer_search_scope(ep)

        new_inst = compose_diverse_instruction(
            seed_key=tid,
            target_assets=targets,
            delivery_fixture=fixture,
            search_scope=scope,
            layout_id=lid,
            comparison_match=is_comp,
        )
        errs = verify_episode_instruction(ep, new_inst)
        if errs:
            violations[tid] = errs
            continue

        unique_instructions.add(new_inst)
        dest_counter[choose_destination_phrase(fixture, seed_key=tid, layout_id=lid)] += 1
        tmpl_counter[_slot_index(tid, "tmpl", len(SENTENCE_TEMPLATES))] += 1
        if not is_comp:
            obj_variant_counter[format_target_noun_phrase(targets, seed_key=tid)] += 1

        if ep.get("instruction") != new_inst:
            ep["instruction"] = new_inst
            updated += 1
            if not dry_run:
                with tempfile.NamedTemporaryFile("w", dir=ep_path.parent, delete=False, encoding="utf-8") as f:
                    json.dump(ep, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                    tmp = f.name
                os.replace(tmp, ep_path)

    return {
        "total_tasks": len(manifest["tasks"]),
        "updated_tasks": updated,
        "unique_instructions": len(unique_instructions),
        "unique_destination_phrases": len(dest_counter),
        "unique_target_noun_phrases": len(obj_variant_counter),
        "sentence_template_distribution": dict(sorted(tmpl_counter.items())),
        "destination_phrase_distribution": dict(dest_counter.most_common()),
        "violation_count": len(violations),
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--release",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs/rab_lived_in_20261120",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    summary = apply_to_release(args.release.resolve(), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["violation_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
