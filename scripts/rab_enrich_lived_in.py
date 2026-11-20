"""Enrich all 1,000 FindingBench RAB episodes with V2 full-home 'lived-in' household items
across all visible household furniture & storage zones:
  - coffee      : Living room main coffee table top (5 items, preserving delivery drop clearance)
  - tv_console  : Living room open 4-bay media console shelves under the TV (5 items)
  - side        : Sofa side table (4 items)
  - dining      : Kitchen/dining table (`KitchenTable53` / `KitchenCoffeeTable03`, 5 items)
  - counter     : Kitchen countertop (`CounterTop_17` & modular `ConstructorKitchen`, 6 items)
  - shelf       : Multi-tier pantry shelving unit (7 items)
  - drawer      : Primary articulated coffee-table drawer `black_box_1` (3 items when non-task)
  - drawer_alt  : Secondary articulated coffee-table drawer `black_box_3` (3 items when non-task)
  - cubby_2     : Open under-coffee-table storage basket `black_box_2` (2 items)
  - cubby_4     : Open under-coffee-table storage basket `black_box_4` (2 items)

Strictly preserves:
  - Every existing task target, search-zone distractor, occlusion shield, and comparison sample.
  - Generous clear delivery drop footprints on the destination fixture.
  - Unique item identities so ambient items never duplicate the task target asset(s).
  - Full compatibility with spectator.py, play_data.py, rab_skills.py, and rab_mcp_server.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from rab_assets import AssetNamespace, LayerBundle  # noqa: E402
from rab_available_resources import namespace, provision  # noqa: E402
from rab_dense_search import (  # noqa: E402
    DRAWERS,
    compose_drawer,
    digest,
    drawer_headroom,
    drawer_support,
    load,
    mesh_footprint,
    quarter_turn,
    reference_table_transform,
    save,
    side_pose,
)

UNDER_TABLE_CUBBIES: dict[str, str] = {
    "cubby_2": "/World/Assets/CoffeeTable/black_box_2",
    "cubby_4": "/World/Assets/CoffeeTable/black_box_4",
}
ALL_UNDER_TABLE_BOXES: dict[str, str] = {**DRAWERS, **UNDER_TABLE_CUBBIES}

EXTRA_ZUP_HOUSEHOLD_ASSETS: tuple[str, ...] = (
    "context_trash_chipped_ceramic_mug",
    "context_trash_metal_cookie_tin",
    "context_trash_newspaper_bundle",
    "context_trash_cardboard_box",
    "context_trash_spent_ballpoint_pen",
    "cutlery_holder",
    "cake_frosted_white",
    "cake_plain_yellow",
    "grapefruit_can",
    "cayman_jacked",
)

STATIC_PROP_NAMES: tuple[str, ...] = (
    "Tv",
    "DiningTableMicrowave",
    "CoffeeGrinder",
    "AirFryer",
)

# Zone-appropriate household ambient pools ("lived-in" human home semantics)
ZONE_POOLS: dict[str, tuple[str, ...]] = {
    "coffee": (
        "context_trash_chipped_ceramic_mug",
        "paper_cup",
        "starbucks",
        "context_trash_metal_cookie_tin",
        "remote_tv",
        "glasses_case",
        "book_red",
        "context_trash_newspaper_bundle",
        "notebook_green",
        "wallet",
        "power_bank",
        "context_trash_spent_ballpoint_pen",
        "oreo_gf",
        "grapefruit_can",
        "coke_c3",
        "cake_frosted_white",
        "cloth_clear_memo_pad",
        "cloth_clear_cat",
        "cloth_clear_bear",
    ),
    "tv_console": (
        "context_trash_cardboard_box",
        "context_trash_metal_cookie_tin",
        "book_red",
        "notebook_green",
        "context_trash_newspaper_bundle",
        "remote_tv",
        "glasses_case",
        "power_bank",
        "wallet",
        "cloth_clear_car",
        "cloth_clear_train",
        "cloth_clear_house",
        "cloth_clear_block",
        "cloth_clear_pouch",
        "cloth_clear_bear",
        "cloth_clear_cat",
    ),
    "side": (
        "context_trash_chipped_ceramic_mug",
        "paper_cup",
        "book_red",
        "notebook_green",
        "glasses_case",
        "wallet",
        "power_bank",
        "context_trash_metal_cookie_tin",
        "context_trash_spent_ballpoint_pen",
        "remote_tv",
        "cloth_clear_duck",
        "cloth_clear_penguin",
        "cloth_clear_memo_pad",
    ),
    "dining": (
        "cutlery_holder",
        "cake_frosted_white",
        "cake_plain_yellow",
        "context_trash_chipped_ceramic_mug",
        "paper_cup",
        "meal_fork",
        "meal_knife",
        "coke_c3",
        "sprite_c1",
        "grapefruit_can",
        "good2grow",
        "honeymaid",
        "oreo_gf",
        "context_trash_metal_cookie_tin",
    ),
    "counter": (
        "cutlery_holder",
        "starbucks",
        "paper_cup",
        "context_trash_chipped_ceramic_mug",
        "meal_fork",
        "meal_knife",
        "coke_c3",
        "sprite_c1",
        "grapefruit_can",
        "cayman_jacked",
        "gatorade_m",
        "good2grow",
        "honeymaid",
        "oreo_gf",
        "cake_frosted_white",
        "zbar_t",
    ),
    "shelf": (
        "pring_bbq",
        "honeymaid",
        "oreo_gf",
        "context_trash_cardboard_box",
        "context_trash_metal_cookie_tin",
        "coke_c3",
        "sprite_c1",
        "grapefruit_can",
        "cayman_jacked",
        "gatorade_t",
        "gatorade_m",
        "starbucks",
        "zbar_t",
        "skit_littles",
        "chomps_r",
        "cloth_clear_pouch",
        "cloth_clear_train",
        "cloth_clear_car",
        "cloth_clear_rabbit",
        "cloth_clear_robot",
        "cloth_clear_block",
    ),
    "drawer": (
        "cloth_clear_memo_pad",
        "cloth_clear_tape_measure",
        "cloth_clear_eraser",
        "cloth_clear_pouch",
        "power_bank",
        "wallet",
        "glasses_case",
        "context_trash_spent_ballpoint_pen",
        "zbar_t",
        "chomps_r",
        "skit_littles",
    ),
    "drawer_alt": (
        "cloth_clear_tape_measure",
        "cloth_clear_eraser",
        "cloth_clear_memo_pad",
        "cloth_clear_pouch",
        "wallet",
        "power_bank",
        "glasses_case",
        "context_trash_metal_cookie_tin",
        "zbar_t",
        "chomps_r",
        "skit_littles",
    ),
    "cubby_2": (
        "context_trash_newspaper_bundle",
        "book_red",
        "notebook_green",
        "context_trash_metal_cookie_tin",
        "cloth_clear_pouch",
        "glasses_case",
        "wallet",
        "cloth_clear_bear",
    ),
    "cubby_4": (
        "context_trash_cardboard_box",
        "notebook_green",
        "cloth_clear_memo_pad",
        "cloth_clear_block",
        "cloth_clear_car",
        "cloth_clear_duck",
        "power_bank",
    ),
}

# Rich lived-in target item counts per household zone
ZONE_TARGET_COUNTS: dict[str, int] = {
    "coffee": 5,
    "tv_console": 5,
    "side": 4,
    "dining": 5,
    "counter": 6,
    "shelf": 7,
    "drawer": 3,
    "drawer_alt": 3,
    "cubby_2": 2,
    "cubby_4": 2,
}


def load_extra_household_assets(resources_root: Path, bundle: LayerBundle) -> dict[str, dict]:
    from pxr import Gf
    from rab_build_search import read_asset

    extra: dict[str, dict] = {}
    all_assets_dir = resources_root / "libraries/current_assets/all_assets"
    for name in EXTRA_ZUP_HOUSEHOLD_ASSETS:
        usd_path = all_assets_dir / name / "asset.usda"
        if not usd_path.is_file():
            continue
        parsed = read_asset(usd_path, bundle)
        if parsed is None:
            continue
        extra[name] = dict(
            parsed,
            rotation=Gf.Matrix4d(1.0),
            clearance=float(-parsed["box"][0, 2]),
            pose_variant="zup_native",
        )
    return extra


def extract_constructor_counter_boards(stage):
    from pxr import UsdGeom, UsdPhysics
    from rab_build_search import triangles

    root = stage.GetPrimAtPath("/World/Background/Shell/ConstructorKitchen")
    if not root or not root.IsActive():
        return []
    transforms = UsdGeom.XformCache()
    boards = []
    for sub in (
        "MetricScale/BP_Kitchen10/Body/Worktop",
        "MetricScale/RightWallBase400/Body/Worktop",
        "MetricScale/BP_Kitchen9/Body/Worktop",
    ):
        prim = stage.GetPrimAtPath(f"/World/Background/Shell/ConstructorKitchen/{sub}")
        if not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        tris = triangles(UsdGeom.Mesh(prim), transforms)
        flat = tris[np.ptp(tris[:, :, 2], axis=1) < 0.003]
        heights = np.rint(flat[:, :, 2].mean(axis=1) * 200) / 200
        plane = flat[heights == 0.92]
        if not len(plane):
            continue
        box = np.array([plane.min(axis=(0, 1)), plane.max(axis=(0, 1))])
        if min(box[1, :2] - box[0, :2]) < 0.25:
            continue
        boards.append((box, plane[:, :, :2], str(prim.GetPath())))
    return boards


def extract_dining_table_boards(stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    from rab_build_search import triangles

    transforms = UsdGeom.XformCache()
    boards = []
    for root_path in ("/World/Assets/KitchenTable53", "/World/Assets/KitchenCoffeeTable03"):
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsActive():
            continue
        best = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if prim.GetAttribute("physics:collisionEnabled").Get() is False:
                continue
            tris = triangles(UsdGeom.Mesh(prim), transforms)
            if not len(tris):
                continue
            flat = tris[np.ptp(tris[:, :, 2], axis=1) < 0.004]
            if not len(flat):
                continue
            heights = np.rint(flat[:, :, 2].mean(axis=1) * 100) / 100
            for h in sorted(set(heights), reverse=True):
                if not 0.70 <= h <= 1.05:
                    continue
                plane = flat[np.abs(flat[:, :, 2].mean(axis=1) - h) < 0.006]
                if not len(plane):
                    continue
                box = np.array([plane.min(axis=(0, 1)), plane.max(axis=(0, 1))])
                if min(box[1, :2] - box[0, :2]) < 0.35:
                    continue
                best.append((box, plane[:, :, :2], str(prim.GetPath())))
        best.sort(key=lambda r: r[0][1, 2], reverse=True)
        if best:
            boards.append(best[0])
    return boards


def extract_tv_console_bays(stage):
    """Extract the 4 open cubby bays (2 columns x 2 shelf levels z~0.13m, z~0.29m) of /World/Assets/TvTable."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from rab_build_search import triangles

    root = stage.GetPrimAtPath("/World/Assets/TvTable")
    if not root or not root.IsActive():
        return []
    transforms = UsdGeom.XformCache()
    bays = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.GetAttribute("physics:collisionEnabled").Get() is False:
            continue
        tris = triangles(UsdGeom.Mesh(prim), transforms)
        if not len(tris):
            continue
        flat = tris[np.ptp(tris[:, :, 2], axis=1) < 0.004]
        if not len(flat):
            continue
        heights = np.rint(flat[:, :, 2].mean(axis=1) * 100) / 100
        for target_z in (0.29, 0.13):
            plane = flat[np.abs(flat[:, :, 2].mean(axis=1) - target_z) < 0.008]
            if len(plane) < 20:
                continue
            full_box = np.array([plane.min(axis=(0, 1)), plane.max(axis=(0, 1))])
            span = full_box[1, :2] - full_box[0, :2]
            if max(span) < 1.20 or min(span) < 0.22:
                continue
            # Split along the long axis around the center vertical divider (leaving 5cm margin)
            long_ax = int(np.argmax(span))
            mid = 0.5 * (full_box[0, long_ax] + full_box[1, long_ax])
            for lo_v, hi_v in (
                (full_box[0, long_ax] + 0.055, mid - 0.050),
                (mid + 0.050, full_box[1, long_ax] - 0.055),
            ):
                if hi_v - lo_v < 0.28:
                    continue
                sub_box = full_box.copy()
                sub_box[0, long_ax] = lo_v
                sub_box[1, long_ax] = hi_v
                # Use None for triangle mask since sub_box is a clean solid rectangular shelf bay
                bays.append((sub_box, None, str(prim.GetPath())))
    return bays


def extract_usd_pose(stage, prim_path: str, xf_cache=None) -> list[float]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Missing prim in stage: {prim_path}")
    cache = xf_cache if xf_cache is not None else UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    tr = mat.ExtractTranslation()
    rot = mat.RemoveScaleShear().ExtractRotationQuat()
    im = rot.GetImaginary()
    return [
        float(tr[0]),
        float(tr[1]),
        float(tr[2]),
        float(im[0]),
        float(im[1]),
        float(im[2]),
        float(rot.GetReal()),
    ]


def build_variants(assets: dict[str, dict]) -> dict[str, tuple[dict, ...]]:
    variants: dict[str, tuple[dict, ...]] = {}
    for name, asset in assets.items():
        v = (asset, quarter_turn(asset))
        if name in (
            "zbar_t",
            "chomps_r",
            "skit_littles",
            "oreo_gf",
            "honeymaid",
            "coke_c3",
            "sprite_c1",
            "grapefruit_can",
            "starbucks",
        ):
            flat = side_pose(asset)
            v = v + (flat, quarter_turn(flat))
        variants[name] = v
    return variants


def compatible_with_zone(asset: dict, zone: str, fixtures: dict, headroom: dict[str, float]) -> bool:
    size = asset["box"][1] - asset["box"][0]
    if zone == "tv_console":
        # Inter-shelf clearance inside TvTable open bays is ~0.14m
        return bool(size[2] + 0.003 + max(0.0, asset.get("clearance", 0.0)) <= 0.115 and max(size[:2]) <= 0.29)
    if zone in ALL_UNDER_TABLE_BOXES:
        if zone not in fixtures:
            return False
        footprint = fixtures[zone][0][0]
        return bool(
            size[2] + 0.003 + max(0.0, asset.get("clearance", 0.0)) <= headroom.get(zone, 0.105)
            and np.all(size[:2] + 0.018 < footprint[1, :2] - footprint[0, :2])
        )
    return True


def enrich_all_episodes(source_release: Path, out_release: Path) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom
    from rab_build_search import bounds, load_assets, placement, supports

    run_migration = ROOT / "runs/rab_migration_20261118"
    repo = ROOT / "references/RobotAgentBench"
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))

    out_release.mkdir(parents=True, exist_ok=True)
    layers_dir = out_release / "asset_layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    resources = run_migration / "available_resources"
    mapping, _ = provision(repo, run_migration, resources)
    bundle = LayerBundle(namespace(resources), layers_dir)
    poses = load(run_migration / "pose_calibration/poses.json")["assets"]
    assets, _ = load_assets(mapping, repo, bundle, poses)
    extra_assets = load_extra_household_assets(resources, bundle)
    for k, v in extra_assets.items():
        assets.setdefault(k, v)
    if "remote_tv" in assets:
        assets["remote_tv"] = mesh_footprint(assets["remote_tv"])
    variants = build_variants(assets)

    ns = AssetNamespace.from_runtime(run_migration / "Runtime-room-0.1.7")
    logical = "/home/lingjun/robot_agent_arena/api-protocol/assets/house_scene_l1/demo_sim/"
    reference_tf = reference_table_transform(ns.resolve(logical + "scene_room_baseline.usda"))
    slide_layer = LayerBundle(ns, layers_dir).clone(ns.resolve(logical + "livingroom_drawer_slide.usda"))

    manifest = load(source_release / "combined_manifest.json")
    tasks_in = manifest["tasks"]
    seed_salt = os.environ.get("FINDINGBENCH_ENRICH_SALT", "v2_lived_in")

    layout_cache: dict[tuple[str, str], tuple] = {}
    new_tasks = []
    zone_added_totals = Counter()
    total_added_items = 0
    t0 = time.monotonic()

    for idx_task, item in enumerate(tasks_in, 1):
        orig_ep = load(item["episode"])
        orig_rec = load(item["physics_receipt"])
        ident = orig_ep["id"]
        kind = orig_ep["arrangement"]["kind"]
        if os.environ.get("FINDINGBENCH_REAUTHOR_NON_DRAWER") == "1" and kind == "articulated_drawer":
            ep_out_path = out_release / "cases" / ident / "episode.json"
            rec_out_path = out_release / "cases" / ident / "native_validation.json"
            ex_ep = load(ep_out_path)
            for p_row in ex_ep.get("placed", []):
                if p_row.get("ambient_enrichment"):
                    zone_added_totals[p_row["zone"]] += 1
                    total_added_items += 1
            new_item = deepcopy(item)
            new_item["episode"] = str(ep_out_path)
            new_item["physics_receipt"] = str(rec_out_path)
            new_item["ambient_enrichment_count"] = ex_ep.get("ambient_enrichment_count", 0)
            new_tasks.append(new_item)
            continue

        src_layer = Sdf.Layer.FindOrOpen(orig_ep["scene"])
        orig_sublayers = list(src_layer.subLayerPaths)
        base_sublayer = Path(orig_sublayers[0])
        cache_key = (str(base_sublayer), kind)

        if cache_key not in layout_cache:
            if kind == "occluded_open":
                composed_path = layers_dir / f"lived_in_{base_sublayer.stem}_drawer.usda"
                if not composed_path.exists():
                    compose_drawer(base_sublayer, composed_path, reference_tf, slide_layer, scale_xy=1.0)
                active_sublayer = composed_path
            else:
                active_sublayer = base_sublayer

            st_layout = Usd.Stage.Open(str(active_sublayer))
            fixtures, obstacles = supports(st_layout)
            if "counter" not in fixtures:
                c_boards = extract_constructor_counter_boards(st_layout)
                if c_boards:
                    fixtures["counter"] = c_boards
            dining_boards = extract_dining_table_boards(st_layout)
            if dining_boards:
                fixtures["dining"] = dining_boards
            tv_bays = extract_tv_console_bays(st_layout)
            if tv_bays:
                fixtures["tv_console"] = tv_bays
            box_map = ALL_UNDER_TABLE_BOXES if kind == "articulated_drawer" else (DRAWERS if kind == "occluded_open" else {})
            for z, p in box_map.items():
                if st_layout.GetPrimAtPath(p):
                    try:
                        fixtures[z] = drawer_support(st_layout, p)
                    except Exception:
                        pass
            headroom = drawer_headroom(st_layout, fixtures) if "drawer" in fixtures else {}
            drawer_origins = {
                z: extract_usd_pose(st_layout, p)[:3]
                for z, p in DRAWERS.items()
                if st_layout.GetPrimAtPath(p)
            }
            cache_b = UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"])
            robot_prim = st_layout.GetPrimAtPath("/World/Robot")
            robot_box = bounds(cache_b, robot_prim) if robot_prim else None

            static_prop_boxes = []
            for sp_name in STATIC_PROP_NAMES:
                sp_prim = st_layout.GetPrimAtPath(f"/World/Assets/{sp_name}")
                if sp_prim and sp_prim.IsActive():
                    sb = bounds(cache_b, sp_prim)
                    if sb is not None and np.isfinite(sb).all():
                        sb_pad = sb.copy()
                        sb_pad[0, :2] -= 0.025
                        sb_pad[1, :2] += 0.025
                        static_prop_boxes.append(sb_pad)

            layout_cache[cache_key] = (
                active_sublayer,
                fixtures,
                obstacles,
                headroom,
                drawer_origins,
                robot_box,
                static_prop_boxes,
            )

        (
            active_sublayer,
            fixtures,
            obstacles,
            headroom,
            drawer_origins,
            robot_box,
            static_prop_boxes,
        ) = layout_cache[cache_key]

        rng = random.Random(f"lived_in_{seed_salt}:{ident}")
        target_assets = {p["asset"] for p in orig_ep["placed"] if p["target"]}
        forbidden_assets = set(target_assets)

        occupied = [sb.copy() for sb in static_prop_boxes]
        for p in orig_ep["placed"]:
            b = np.array(p["expected_bounds_m"], dtype=float)
            b[0, :2] -= 0.018
            b[1, :2] += 0.018
            occupied.append(b)
        if robot_box is not None:
            r_pad = robot_box.copy()
            r_pad[0, :2] -= 0.14
            r_pad[1, :2] += 0.14
            r_pad[0, 2] = 0.0
            r_pad[1, 2] = 2.0
            occupied.append(r_pad)
        for dp in orig_ep.get("delivery", {}).get("placements", []):
            if "bounds_m" in dp:
                db = np.array(dp["bounds_m"], dtype=float)
                # Reserve a generous ~26cm+ clear delivery pad around the target placement zone
                db[0, :2] -= 0.085
                db[1, :2] += 0.085
                occupied.append(db)

        search_zone = orig_ep.get("search_fixture")
        alt_search_zone = orig_ep.get("alternative_search_fixture")

        zone_counts = Counter(p.get("zone") for p in orig_ep["placed"])
        asset_counts = Counter(p["asset"] for p in orig_ep["placed"])
        added_records = []

        for zone, want_count in ZONE_TARGET_COUNTS.items():
            if zone not in fixtures:
                continue
            # Never alter the primary search zone or primary decoy drawer of the episode
            if zone in (search_zone, alt_search_zone):
                continue
            current_in_zone = zone_counts[zone]
            to_add = max(0, want_count - current_in_zone)
            if to_add == 0:
                continue

            pool = [a for a in ZONE_POOLS[zone] if a in assets and a not in forbidden_assets]
            rng.shuffle(pool)

            if zone in ALL_UNDER_TABLE_BOXES:
                box_prefix = ALL_UNDER_TABLE_BOXES[zone] + "/"
                blockers = [
                    r for r in obstacles
                    if not r[0].startswith(box_prefix) and r[0] != "/World/Assets/CoffeeTable/Mesh"
                ]
            elif zone == "tv_console":
                blockers = [r for r in obstacles if not r[0].startswith("/World/Assets/TvTable/")]
            else:
                blockers = obstacles

            zone_inset = (
                0.055 if zone in ("counter", "shelf", "dining")
                else 0.025 if zone in ("coffee", "side")
                else 0.015 if zone == "tv_console"
                else 0.010
            )
            zone_sep = 0.022 if zone not in ALL_UNDER_TABLE_BOXES else 0.014

            for slot_idx in range(to_add):
                placed_ok = False
                # On the first 3 slots of coffee/side/dining, prefer taller visible 3D objects (>3.5cm)
                def sort_key(a_name: str):
                    h = float(assets[a_name]["box"][1, 2] - assets[a_name]["box"][0, 2])
                    prefer_3d = 0 if (zone not in ("coffee", "side", "dining") or slot_idx >= 3 or h >= 0.038) else 1
                    return (asset_counts[a_name], prefer_3d, rng.random())

                for candidate_name in sorted(pool, key=sort_key):
                    opts = list(variants[candidate_name])
                    if zone not in ALL_UNDER_TABLE_BOXES:
                        opts = opts[:2]
                    for asset_var in opts:
                        if not compatible_with_zone(asset_var, zone, fixtures, headroom):
                            continue
                        res = placement(
                            fixtures,
                            blockers,
                            zone,
                            asset_var["box"],
                            rng,
                            occupied,
                            asset_var.get("clearance", 0.0),
                            inset=zone_inset,
                            separation=zone_sep,
                        )
                        if res is not None:
                            box, support_path = res
                            occupied.append(box)
                            added_records.append(
                                dict(
                                    asset=candidate_name,
                                    zone=zone,
                                    box=box,
                                    target=False,
                                    support=support_path,
                                    source=asset_var,
                                    ambient_enrichment=True,
                                )
                            )
                            zone_counts[zone] += 1
                            asset_counts[candidate_name] += 1
                            zone_added_totals[zone] += 1
                            total_added_items += 1
                            placed_ok = True
                            break
                    if placed_ok:
                        break

        case_dir = out_release / "cases" / ident
        case_dir.mkdir(parents=True, exist_ok=True)
        new_scene_path = case_dir / "scene.usda"

        new_layer = Sdf.Layer.CreateAnonymous("lived_in_scene.usda")
        new_layer.TransferContent(src_layer)
        new_layer.subLayerPaths = []

        stage_mem = Usd.Stage.Open(new_layer)
        UsdGeom.SetStageUpAxis(stage_mem, "Z")
        UsdGeom.SetStageMetersPerUnit(stage_mem, 1.0)

        new_placed = deepcopy(orig_ep["placed"])
        base_n = len(new_placed)
        added_usd_poses: dict[str, list[float]] = {}
        xf_mem = UsdGeom.XformCache()

        for offset_i, rec in enumerate(added_records):
            item_id = f"Item_{base_n + offset_i:02d}"
            asset_var = rec["source"]
            root_prim = UsdGeom.Xform.Define(stage_mem, f"/World/FindingBench/{item_id}")
            model_prim = stage_mem.DefinePrim(f"/World/FindingBench/{item_id}/Model")
            model_prim.GetReferences().AddReference(asset_var["layer"])
            delta = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(rec["box"][0] - asset_var["box"][0])))
            root_prim.MakeMatrixXform().Set(asset_var["rotation"] * delta)
            body_path = f"/World/FindingBench/{item_id}/Model" + asset_var["body"]
            added_usd_poses[body_path] = extract_usd_pose(stage_mem, body_path, xf_mem)
            new_placed.append(
                dict(
                    body=body_path,
                    asset=rec["asset"],
                    target=False,
                    support=rec["support"],
                    expected_bounds_m=rec["box"].tolist(),
                    zone=rec["zone"],
                    source_sha256=asset_var["sha256"],
                    pose_variant=asset_var.get("pose_variant", "calibrated"),
                    ambient_enrichment=True,
                )
            )

        new_layer.subLayerPaths = [str(active_sublayer), *orig_sublayers[1:]]
        if not new_layer.Export(str(new_scene_path)):
            raise RuntimeError(f"Failed to export enriched USD scene for {ident}")
        new_scene_sha = digest(new_scene_path)

        new_ep = deepcopy(orig_ep)
        new_ep["scene"] = str(new_scene_path)
        new_ep["scene_sha256"] = new_scene_sha
        new_ep["pre_enrichment_scene_sha256"] = orig_ep["scene_sha256"]
        new_ep["placed"] = new_placed
        new_ep["ambient_enrichment_count"] = len(added_records)

        art_map = dict(new_ep["arrangement"].get("articulations") or {})
        if kind != "articulated_refrigerator":
            for z_k, d_path in DRAWERS.items():
                if z_k not in art_map:
                    art_map[z_k] = d_path + "_slide"
        new_ep["arrangement"]["articulations"] = art_map

        new_rec = deepcopy(orig_rec)
        new_rec["scene_sha256"] = new_scene_sha
        new_rec["physx_lived_in_verified"] = False
        ref = new_rec.get("dense_checks", {}).get("restoration_reference") or new_rec.get("restoration_reference")
        if ref:
            ref_bodies = ref.setdefault("bodies", {})
            ref_art_m = ref.setdefault("articulations_m", {})
            for z_k in DRAWERS:
                ref_art_m.setdefault(z_k, 0.0)
            for p_row in new_placed[base_n:]:
                b_path = p_row["body"]
                u_pose = added_usd_poses[b_path]
                entry = dict(
                    pose=u_pose,
                    support=p_row["support"],
                    zone=p_row["zone"],
                    target=False,
                )
                if p_row["zone"] in DRAWERS and p_row["zone"] in drawer_origins:
                    d_orig = drawer_origins[p_row["zone"]]
                    entry["drawer_relative_xyz_m"] = [
                        float(u_pose[0] - d_orig[0]),
                        float(u_pose[1] - d_orig[1]),
                        float(u_pose[2] - d_orig[2]),
                    ]
                ref_bodies[b_path] = entry
        for p_row in new_placed[base_n:]:
            b_path = p_row["body"]
            u_pose = added_usd_poses[b_path]
            new_rec.setdefault("objects", []).append(
                dict(
                    body=b_path,
                    stable=True,
                    drift_m=0.0,
                    final_motion_m=0.0,
                    rotation_rad=0.0,
                    initial_pose=u_pose,
                    settled_pose=u_pose,
                    ambient_enrichment=True,
                )
            )

        ep_out_path = case_dir / "episode.json"
        rec_out_path = case_dir / "native_validation.json"
        save(ep_out_path, new_ep)
        save(rec_out_path, new_rec)

        new_item = deepcopy(item)
        new_item["episode"] = str(ep_out_path)
        new_item["physics_receipt"] = str(rec_out_path)
        new_item["ambient_enrichment_count"] = len(added_records)
        new_tasks.append(new_item)

        if idx_task % 100 == 0 or idx_task == len(tasks_in):
            print(
                f"[lived-in-v2] Authored {idx_task}/{len(tasks_in)} episodes | total added items={total_added_items} ({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    out_summary = deepcopy(manifest.get("summary", {}))
    out_summary["ambient_enrichment"] = {
        "version": "v2_full_home_lived_in",
        "total_added_items": total_added_items,
        "mean_added_items_per_episode": round(total_added_items / len(new_tasks), 3),
        "added_by_zone": dict(zone_added_totals),
        "asset_catalog_count": len(assets),
        "parent_release": str(source_release),
        "parent_manifest_sha256": digest(source_release / "combined_manifest.json"),
    }
    out_manifest = dict(
        summary=out_summary,
        baseline_sha256=manifest.get("baseline_sha256"),
        acceptance_scope="Dense initialization with V2 full-home lived-in enrichment across 10 household zones (coffee, tv_console, side, dining, counter, shelf, drawers, cubbies)",
        tasks=new_tasks,
        pending=[],
    )
    save(out_release / "combined_manifest.json", out_manifest)
    save(out_release / "enrichment_report.json", out_summary["ambient_enrichment"])
    return out_summary


def verify_physx_subset(out_release: Path, task_ids: list[str]) -> dict:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "4")
    from isaacsim import SimulationApp

    app = SimulationApp(
        dict(
            headless=True,
            width=64,
            height=64,
            multi_gpu=False,
            limit_cpu_threads=4,
            extra_args=["--/log/level=error"],
        )
    )
    import carb.settings
    import omni.physx
    from pxr import UsdPhysics
    from rab_validate_search import validate

    settings = carb.settings.get_settings()
    settings.set_bool("/physics/suppressReadback", False)
    settings.set_bool("/physics/updateToUsd", True)
    sim = omni.physx.get_physx_simulation_interface()
    interface = omni.physx.get_physx_interface()

    manifest = load(out_release / "combined_manifest.json")
    by_id = {t["id"]: t for t in manifest["tasks"]}
    results = []
    passed_count = 0

    for idx, tid in enumerate(task_ids, 1):
        t_row = by_id[tid]
        ep_path = Path(t_row["episode"])
        rec_path = Path(t_row["physics_receipt"])
        rec = load(rec_path)
        if rec.get("physx_lived_in_verified"):
            passed_count += 1
            results.append(
                dict(
                    id=tid,
                    status="physx_initialization_pass",
                    passed=True,
                    total_objects=len(rec.get("objects", [])),
                    unsettled_bodies=[],
                    max_drift_m=round(max((o.get("drift_m", 0.0) for o in rec.get("objects", [])), default=0.0), 5),
                    cached=True,
                )
            )
            continue
        ep = load(ep_path)

        def before(stage):
            for zone, jpath in ep["arrangement"].get("articulations", {}).items():
                prim = stage.GetPrimAtPath(jpath)
                if not prim:
                    continue
                angular = prim.IsA(UsdPhysics.RevoluteJoint)
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular" if angular else "linear")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(1100.0 if angular else 4000.0)
                drive.CreateDampingAttr(130.0 if angular else 400.0)
                drive.CreateMaxForceAttr(180.0 if angular else 300.0)
                drive.CreateTargetPositionAttr(0.0)

        v_res = validate(ep, sim, interface, before_attach=before)
        ok = v_res["status"] == "physx_initialization_pass"
        if ok:
            passed_count += 1
            settled_by_body = {o["body"]: o["settled_pose"] for o in v_res["objects"]}
            rec["objects"] = v_res["objects"]
            ref = rec.get("dense_checks", {}).get("restoration_reference") or rec.get("restoration_reference")
            if ref:
                for b_path, b_entry in ref.get("bodies", {}).items():
                    if b_path in settled_by_body:
                        s_pose = settled_by_body[b_path]
                        b_entry["pose"] = s_pose
                        z_k = b_entry.get("zone")
                        if z_k in DRAWERS:
                            d_pose = interface.get_rigidbody_transformation(DRAWERS[z_k])
                            if d_pose.get("ret_val"):
                                d_pos = d_pose["position"]
                                b_entry["drawer_relative_xyz_m"] = [
                                    float(s_pose[0] - d_pos[0]),
                                    float(s_pose[1] - d_pos[1]),
                                    float(s_pose[2] - d_pos[2]),
                                ]
            rec["physx_lived_in_verified"] = True
            save(rec_path, rec)
        unsettled = [o["body"] for o in v_res["objects"] if not o["stable"]]
        max_drift = max((o["drift_m"] for o in v_res["objects"]), default=0.0)
        results.append(
            dict(
                id=tid,
                status=v_res["status"],
                passed=ok,
                total_objects=len(v_res["objects"]),
                unsettled_bodies=unsettled,
                max_drift_m=round(max_drift, 5),
            )
        )
        if idx % 25 == 0 or not ok or idx == len(task_ids):
            print(
                f"[physx-verify {idx}/{len(task_ids)}] {tid}: {v_res['status']} (objects={len(v_res['objects'])}, max_drift={max_drift:.4f}m, unsettled={unsettled})",
                flush=True,
            )

    report = dict(
        verified_count=len(task_ids),
        passed_count=passed_count,
        results=results,
    )
    save(out_release / "physx_verification_report.json", report)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if passed_count == len(task_ids) else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "runs/rab_dense_serial_20261119",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs/rab_lived_in_20261120",
    )
    parser.add_argument("--author-only", action="store_true")
    parser.add_argument("--verify-physx", action="store_true")
    parser.add_argument("--pilot-only", action="store_true", help="Verify all 48 pilot+reserve tasks in PhysX")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs to verify in PhysX")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_release = args.out.resolve()
    if not (out_release / "combined_manifest.json").exists() or args.author_only:
        summary = enrich_all_episodes(args.source.resolve(), out_release)
        print("Enrichment summary:", json.dumps(summary["ambient_enrichment"], indent=2, ensure_ascii=False), flush=True)
        if args.author_only and not args.verify_physx:
            return 0

    if args.verify_physx:
        if args.tasks:
            tids = [x.strip() for x in args.tasks.split(",") if x.strip()]
        elif args.pilot_only:
            pilot_cfg = load(ROOT / "configs/benchmarks/rab_v1_pilot_draft.json")
            tids = [r["id"] for r in pilot_cfg["primary_candidates"]]
            for group_rows in pilot_cfg.get("reserve_candidates", {}).values():
                for r in group_rows:
                    if r["id"] not in tids:
                        tids.append(r["id"])
            for extra_id in ("rab_F251", "rab_F087", "rab_F065", "rab_F262", "rab_F001"):
                if extra_id not in tids:
                    tids.append(extra_id)
        else:
            m = load(out_release / "combined_manifest.json")
            tids = [t["id"] for t in m["tasks"]]
        if args.limit:
            tids = tids[: args.limit]
        verify_physx_subset(out_release, tids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
