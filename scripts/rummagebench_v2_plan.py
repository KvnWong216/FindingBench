#!/usr/bin/env python3
"""Deterministic 1,000-episode multi-scene & domain-randomization planner for RummageBench v2.

Maps all 1,000 GPT-6 everyday tasks in `behavior_recipe_backup/prompts.csv`
(`F001..F334`, `P001..P333`, `U001..U333`) onto:
  - 6 Distinct Multi-Domain Scene Archetypes (`S1` Commercial Supermarket Aisles,
    `S2` Multi-Cabinet Culinary Kitchen, `S3` Bedroom & Private Study Suite,
    `S4` Living Salon & Media Lounge, `S5` Entryway Mudroom & Parcel Station,
    `S6` Cargo Trunk & Garage Rack Bay)
  - 60 Physical Base 3D Layouts (`SM_01..SM_10` standalone commercial supermarket
    configurations + `ID_01..ID_50` residential architectural shells)
  - 10 Balanced Physical Container & Occlusion Families (9%..11% each, strictly <= 12%,
    and rotating across 10+ distinct kitchen cupboards + Range Oven so NO single
    cupboard like `BP_Kitchen10` ever exceeds ~1.5%!)
  - Per-episode Lighting Mood (8 color-temperature & intensity profiles) x
    Interior Material Palette (12 wall/floor/cabinet/rug/upholstery themes)
  - 6 GPT-6 Pure-Rummage Cognitive Paradigms (`P1`..`P6`)
  - 60% In-Prior Semantic Zone / 40% Cross-Zone Misplaced Clutter
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/benchmarks/rummagebench_v2_spec.json"
PROMPTS_CSV = ROOT / "runs/rab_migration_20261118/behavior_recipe_backup/prompts.csv"

ARCHETYPE_COMPATIBLE_CONTAINERS = {
    "S1_commercial_supermarket_aisles": [
        "pantry_sliding_and_deep_shelf",
        "refrigerator_multibin",
        "deep_bins_and_bags_layered",
        "countertop_appliances",
        "kitchen_worktop_clutter",
        "cloth_and_tray_cover_occlusion",
    ],
    "S2_culinary_kitchen_and_appliance_hub": [
        "kitchen_cabinet_and_oven",
        "refrigerator_multibin",
        "countertop_appliances",
        "kitchen_worktop_clutter",
        "pantry_sliding_and_deep_shelf",
        "cloth_and_tray_cover_occlusion",
    ],
    "S3_bedroom_and_private_study_suite": [
        "entryway_study_and_trunk_storage",
        "furniture_blind_spots",
        "coffee_table_four_drawers",
        "cloth_and_tray_cover_occlusion",
        "deep_bins_and_bags_layered",
        "kitchen_cabinet_and_oven",
    ],
    "S4_living_salon_and_media_lounge": [
        "coffee_table_four_drawers",
        "furniture_blind_spots",
        "cloth_and_tray_cover_occlusion",
        "deep_bins_and_bags_layered",
        "pantry_sliding_and_deep_shelf",
        "kitchen_cabinet_and_oven",
        "refrigerator_multibin",
    ],
    "S5_entryway_mudroom_and_parcel_station": [
        "entryway_study_and_trunk_storage",
        "deep_bins_and_bags_layered",
        "pantry_sliding_and_deep_shelf",
        "furniture_blind_spots",
        "coffee_table_four_drawers",
        "countertop_appliances",
        "kitchen_cabinet_and_oven",
    ],
    "S6_cargo_trunk_and_garage_rack_bay": [
        "entryway_study_and_trunk_storage",
        "pantry_sliding_and_deep_shelf",
        "deep_bins_and_bags_layered",
        "kitchen_worktop_clutter",
        "countertop_appliances",
        "refrigerator_multibin",
        "kitchen_cabinet_and_oven",
    ],
}

SUPERMARKET_DELIVERY_SURFACES = [
    ("supermarket_checkout_counter", "/World/Assets/Task5PackStation/Cabinet/Top", "the supermarket checkout counter"),
    ("supermarket_shopping_cart", "/World/Objects/supermarket_v1_shopping_cart", "the chrome supermarket shopping cart in the center aisle"),
    ("supermarket_hand_basket", "/World/Assets/BasketInteriorFloor", "the red supermarket shopping basket by the aisle"),
    ("supermarket_deli_counter", "/World/Assets/DeliCounterTop", "the convenience-store deli counter beside the microwave"),
]

RESIDENTIAL_DELIVERY_SURFACES = [
    ("kitchen_dining_table", "/World/Assets/DiningSurface", "the dining table in the kitchen nook"),
    ("dark_sink_worktop", "/World/Background/Shell/ConstructorKitchen", "the main kitchen worktop near the sink"),
    ("living_coffee_table_top", "/World/Assets/CoffeeTable/Mesh", "the center coffee table on the living room rug"),
    ("wall_side_desk", "/World/Assets/SmallCoffeeTable", "the wooden side desk against the wall"),
    ("tv_console_surface", "/World/Assets/TvTable", "the media console table beneath the screen"),
    ("entryway_pack_top", "/World/Assets/Task5PackStation/Cabinet/Top", "the entryway sorting counter"),
    ("wicker_storage_basket", "/World/Assets/TableclothBasket", "the woven storage basket on the floor"),
    ("nightstand_or_trunk_ledge", "/World/Assets/ArchetypeSurface", "the auxiliary organizer surface"),
]


def classify_gpt6_prompt(instruction: str) -> str:
    low = instruction.lower()
    if any(k in low for k in ("should include", "missing item", "already there", "complete the set")):
        return "P5_set_completion_missing_item"
    if any(k in low for k in ("rather than", "instead of", "not the")):
        return "P6_contrastive_negative_foil"
    if any(k in low for k in ("labeled", "marked", "printed")) or '"' in instruction:
        return "P4_fine_grained_label_check"
    if any(k in low for k in ("sample", "matches the one", "matching")):
        return "P2_visual_sample_matching"
    if any(k in low for k in ("two ", "three ", "four ", "both ", "pair of ", " and a ", " and the ")):
        return "P3_multi_item_and_count"
    return "P1_situational_functional"


def infer_gpt6_scene_affinity(scene_name: str, instruction: str) -> str:
    s = f"{scene_name} {instruction}".lower()
    if any(k in s for k in ("grocery", "supermarket", "store", "market", "aisle", "cart", "shop")):
        return "S1_commercial_supermarket_aisles"
    if any(k in s for k in ("car", "trunk", "garage", "garden", "patio", "picnic", "camp", "load", "cooler")):
        return "S6_cargo_trunk_and_garage_rack_bay"
    if any(k in s for k in ("bedroom", "bed", "nightstand", "dresser", "office", "study", "desk", "notebook", "book")):
        return "S3_bedroom_and_private_study_suite"
    if any(k in s for k in ("entry", "hall", "coat", "umbrella", "leave", "bag", "tote", "mail", "package", "closet")):
        return "S5_entryway_mudroom_and_parcel_station"
    if any(k in s for k in ("kitchen", "fridge", "oven", "microwave", "plate", "mug", "fork", "knife", "coffee", "tea", "snack")):
        return "S2_culinary_kitchen_and_appliance_hub"
    return "S4_living_salon_and_media_lounge"


def generate_master_plan(seed: int = 202702) -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    rows = list(csv.DictReader(PROMPTS_CSV.open()))
    assert len(rows) == 1000

    rng = random.Random(seed)

    # 1. Assign 6 Scene Archetypes respecting GPT-6 prompt/scene affinity first
    arch_quotas = {a["id"]: a["quota"] for a in spec["scene_archetypes"]}
    assert sum(arch_quotas.values()) == 1000
    affinities = [infer_gpt6_scene_affinity(r["scene"], r["instruction"]) for r in rows]
    assigned_arch = [None] * 1000
    rem_arch = dict(arch_quotas)
    idxs = list(range(1000))
    rng.shuffle(idxs)
    for i in idxs:
        af = affinities[i]
        if rem_arch[af] > 0:
            assigned_arch[i] = af
            rem_arch[af] -= 1
    leftover_arch = [aid for aid, cnt in rem_arch.items() for _ in range(cnt)]
    rng.shuffle(leftover_arch)
    p_arch = 0
    for i in range(1000):
        if assigned_arch[i] is None:
            assigned_arch[i] = leftover_arch[p_arch]
            p_arch += 1

    # 2. Assign 60 Physical Base Layouts:
    #    - S1 (150 eps) -> SM_01..SM_10 (15 eps each, each with unique per-episode DR!)
    #    - S2..S6 (850 eps) -> ID_01..ID_50 (17 eps each across 50 architectural layouts!)
    sm_layouts = [f"SM_{k:02d}_SUPERMARKET" for k in range(1, 11)] * 15
    rng.shuffle(sm_layouts)
    res_layouts = [f"ID_{k:02d}_LIVING_STRAIGHT" for k in range(1, 51)] * 17
    rng.shuffle(res_layouts)

    assigned_layout = [None] * 1000
    sm_ptr, res_ptr = 0, 0
    for i in range(1000):
        if assigned_arch[i] == "S1_commercial_supermarket_aisles":
            assigned_layout[i] = sm_layouts[sm_ptr]
            sm_ptr += 1
        else:
            assigned_layout[i] = res_layouts[res_ptr]
            res_ptr += 1

    # 3. Assign 10 Container Families respecting Scene Archetype compatibility & exact quotas
    rem_cfam = {f["id"]: f["quota"] for f in spec["container_and_occlusion_families"]}
    assert sum(rem_cfam.values()) == 1000
    assigned_cfam = [None] * 1000

    # Assign most constrained archetype (S1 supermarket) first, then S3/S6/S5/S2/S4
    arch_priority = {
        "S1_commercial_supermarket_aisles": 0,
        "S3_bedroom_and_private_study_suite": 1,
        "S6_cargo_trunk_and_garage_rack_bay": 2,
        "S5_entryway_mudroom_and_parcel_station": 3,
        "S2_culinary_kitchen_and_appliance_hub": 4,
        "S4_living_salon_and_media_lounge": 5,
    }
    order_cfam = list(range(1000))
    rng.shuffle(order_cfam)
    order_cfam.sort(key=lambda i: arch_priority[assigned_arch[i]])

    for i in order_cfam:
        compat = [c for c in ARCHETYPE_COMPATIBLE_CONTAINERS[assigned_arch[i]] if rem_cfam[c] > 0]
        if not compat:
            compat = [c for c, cnt in rem_cfam.items() if cnt > 0]
        # Weighted pick proportional to remaining quota so every archetype gets rich container variety
        weights = [rem_cfam[c] for c in compat]
        picked = rng.choices(compat, weights=weights, k=1)[0]
        assigned_cfam[i] = picked
        rem_cfam[picked] -= 1

    # 4. Assign 6 GPT-6 Cognitive Paradigms preserving native prompt structure first
    rem_para = {p["id"]: p["quota"] for p in spec["cognitive_rummage_paradigms"]}
    native_tags = [classify_gpt6_prompt(r["instruction"]) for r in rows]
    assigned_para = [None] * 1000
    order_para = list(range(1000))
    rng.shuffle(order_para)
    for i in sorted(order_para, key=lambda idx: 0 if native_tags[idx] != "P1_situational_functional" else 1):
        nt = native_tags[i]
        if rem_para[nt] > 0:
            assigned_para[i] = nt
            rem_para[nt] -= 1
    left_para = [pid for pid, cnt in rem_para.items() for _ in range(cnt)]
    rng.shuffle(left_para)
    p_ptr = 0
    for i in range(1000):
        if assigned_para[i] is None:
            assigned_para[i] = left_para[p_ptr]
            p_ptr += 1

    # 5. Spatial prior (60/40), Robot spawn zones (6), Lighting (8), Material Palettes (12), Kitchen Cupboard index (10)
    prior_pool = ["in_prior"] * 600 + ["out_of_prior_misplaced"] * 400
    rng.shuffle(prior_pool)

    spawns = spec["robot_spawn_zones"]
    lights = spec["domain_randomization_axes"]["lighting_profiles"]
    palettes = spec["domain_randomization_axes"]["material_palettes"]

    # Round-robin cupboard slot index (0..9) for kitchen_cabinet_and_oven so NO cupboard repeats!
    cupboard_slot = 0

    episodes = []
    for i, r in enumerate(rows):
        arch = assigned_arch[i]
        cfam = assigned_cfam[i]
        lid = assigned_layout[i]

        if arch == "S1_commercial_supermarket_aisles":
            dest_pool = [d for d in SUPERMARKET_DELIVERY_SURFACES if not (cfam == "deep_bins_and_bags_layered" and d[0] in ("supermarket_shopping_cart", "supermarket_hand_basket"))]
            light_prof = rng.choice(["commercial_fluorescent_6400K", "overcast_studio_5800K", "neutral_daylight_4400K", "soft_indoor_3900K"])
            mat_pal = rng.choice(["commercial_retail_bright", "charcoal_slate_marble", "pebble_gray_granite", "nordic_ivory_oak", "sage_mist_walnut"])
        else:
            dest_pool = [d for d in RESIDENTIAL_DELIVERY_SURFACES if not (cfam == "deep_bins_and_bags_layered" and d[0] == "wicker_storage_basket")]
            if arch == "S3_bedroom_and_private_study_suite":
                light_prof = rng.choice(["warm_evening_2750K", "golden_hour_3300K", "cozy_lamp_spot_3500K", "soft_indoor_3900K", "crisp_morning_5200K"])
            elif arch == "S6_cargo_trunk_and_garage_rack_bay":
                light_prof = rng.choice(["overcast_studio_5800K", "crisp_morning_5200K", "commercial_fluorescent_6400K", "neutral_daylight_4400K"])
            else:
                light_prof = lights[(i * 3 + rng.randint(0, 7)) % len(lights)]
            mat_pal = palettes[(i * 5 + rng.randint(0, 11)) % len(palettes)]

        dest = dest_pool[i % len(dest_pool)]
        k_slot = None
        if cfam == "kitchen_cabinet_and_oven":
            k_slot = cupboard_slot % 10
            cupboard_slot += 1

        episodes.append({
            "episode_id": f"rb_{r['parent_task_id']}",
            "parent_task_id": r["parent_task_id"],
            "gpt6_original_scene": r["scene"],
            "gpt6_original_instruction": r["instruction"],
            "scene_archetype": arch,
            "layout_id": lid,
            "container_family": cfam,
            "kitchen_cupboard_slot": k_slot,
            "cognitive_paradigm": assigned_para[i],
            "spatial_prior_mode": prior_pool[i],
            "robot_spawn_zone": spawns[i % len(spawns)],
            "lighting_profile": light_prof,
            "material_palette": mat_pal,
            "delivery_surface_id": dest[0],
            "delivery_prim": dest[1],
            "delivery_phrase": dest[2],
        })

    return {
        "schema": "rummagebench.v2_master_plan.v2",
        "seed": seed,
        "total_episodes": len(episodes),
        "scene_archetype_distribution": dict(Counter(e["scene_archetype"] for e in episodes)),
        "unique_base_layouts": len({e["layout_id"] for e in episodes}),
        "container_distribution": dict(Counter(e["container_family"] for e in episodes)),
        "paradigm_distribution": dict(Counter(e["cognitive_paradigm"] for e in episodes)),
        "lighting_distribution": dict(Counter(e["lighting_profile"] for e in episodes)),
        "palette_distribution": dict(Counter(e["material_palette"] for e in episodes)),
        "prior_distribution": dict(Counter(e["spatial_prior_mode"] for e in episodes)),
        "episodes": episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "configs/benchmarks/rummagebench_v2_master_plan.json"))
    args = ap.parse_args()
    plan = generate_master_plan()
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print("Master plan written:", out_p)
    for k in (
        "scene_archetype_distribution",
        "unique_base_layouts",
        "container_distribution",
        "paradigm_distribution",
        "lighting_distribution",
        "palette_distribution",
    ):
        print(f"  {k}: {plan[k]}")


if __name__ == "__main__":
    main()
