#!/usr/bin/env python3
"""Dual-Mode Ecological Scene Population Engine for RummageBench v2:
1. `S1` Commercial Supermarket Mode: High-density (`130–150` objs) multi-row retail
   shelf facings with 22% shopper depletion gaps + stocked gondolas & deli/checkout counters.
2. `S2–S6` Household Mode: Moderately sparser (`58–82` objs) yet organically MESSIER
   lived-in scatter — zero retail row cloning, heterogeneous miscellaneous items,
   continuous `[-180, 180]` yaw, 38% tipped/laid-flat cans/bottles/tubes (`rotX=90`),
   skewed 2-book stacks, crumpled paper balls, coffee-ring stain decals, and stray
   items near sofa/table/bin edges.

All ambient props use Static Colliders (`CollisionAPI=ON`, `physics:rigidBodyEnabled=False`)
for 0.0 ms PhysX dynamic rigid-body overhead and zero door/drawer swing drift.
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from rab_build_search import bounds


STATIC_PROP_SPECS = [
    # Surface decals & trays
    ("table_stain_grid", "aa", "table_stain_grid/asset.usda", "decal"),
    ("japanese_tray_static", "aa", "japanese_tray/asset.usda", "tray"),
    ("paper_ball_static", "aa", "paper_ball/asset.usda", "mess"),
    ("cutlery_holder_static", "aa", "cutlery_holder/asset.usda", "kitchen"),
    ("ceramic_fragments_static", "aa", "context_trash_ceramic_tile_fragments/asset.usda", "mess"),
    ("glass_jar_static", "aa", "context_trash_glass_jar/asset.usda", "kitchen"),
    ("spent_pen_static", "aa", "context_trash_spent_ballpoint_pen/asset.usda", "study"),
    ("worn_eraser_static", "aa", "context_trash_worn_eraser/asset.usda", "study"),
    ("chipped_mug_static", "aa", "context_trash_chipped_ceramic_mug/asset.usda", "mess"),
    ("empty_can_static", "aa", "empty_can/asset.usda", "tippable_can"),
    ("newspaper_static", "aa", "context_trash_newspaper_bundle/asset.usda", "mess"),
    ("cardboard_box_static", "aa", "context_trash_cardboard_box/asset.usda", "box"),
    ("cookie_tin_static", "aa", "context_trash_metal_cookie_tin/asset.usda", "box"),
    # Extra photorealistic grocery, snack, book & beverage packaging (`rbs=0, up=Z, mpu=1.0`)
    ("book_pink", "ds", "usdz/book_pink.usdz", "study"),
    ("note_green_pad", "ds", "usdz/note_green.usdz", "study"),
    ("blue_cereal_box", "ds", "photo_assets/blue_box/asset.usda", "shelf_tall"),
    ("cacao_pouch", "ds", "photo_assets/cacao_bag/asset.usda", "shelf_mid"),
    ("chomps_jerky_box", "ds", "photo_assets/chomps_box/asset.usda", "shelf_mid"),
    ("chomps_teal_pouch", "ds", "photo_assets/chomps_teal_bag/asset.usda", "shelf_mid"),
    ("coke_10pack_carton", "ds", "photo_assets/coke_mini_box/asset.usda", "carton"),
    ("sprite_10pack_carton", "ds", "photo_assets/sprite_mini_box/asset.usda", "carton"),
    ("diet_coke_silver_can", "ds", "photo_assets/diet_coke_can/asset.usda", "tippable_can"),
    ("gatorade_blue_static", "ds", "photo_assets/gatorade_blue/asset.usda", "tippable_bottle"),
    ("gatorade_orange_static", "ds", "photo_assets/gatorade_bottle/asset.usda", "tippable_bottle"),
    ("honeymaid_family_box", "ds", "photo_assets/honeymaid_box/asset.usda", "shelf_mid"),
    ("lays_party_chip_bag", "ds", "photo_assets/lays_party_bag/asset.usda", "snack_bag"),
    ("lindt_gold_bag_static", "ds", "photo_assets/lindt_bag/asset.usda", "snack_bag"),
    ("pringles_mini_cup", "ds", "photo_assets/pringles_cup/asset.usda", "snack_small"),
    ("pringles_purple_tube", "ds", "photo_assets/pringles_purple/asset.usda", "tippable_tube"),
    ("pringles_red_tube", "ds", "photo_assets/pringles_red/asset.usda", "tippable_tube"),
    ("pringles_smoky_tube", "ds", "photo_assets/pringles_smoky/asset.usda", "tippable_tube"),
    ("redbull_4pack_box", "ds", "photo_assets/redbull_pink_box/asset.usda", "shelf_mid"),
    ("skittles_share_pouch", "ds", "photo_assets/skittles_share_bag/asset.usda", "snack_bag"),
    ("smores_kit_carton", "ds", "photo_assets/smores_kit_box/asset.usda", "shelf_mid"),
    ("zbar_multipack_box", "ds", "photo_assets/zbar_box/asset.usda", "shelf_small"),
    ("sprite_open_empty_can", "ds", "newcans/cans/sprite_open_empty/asset.usda", "tippable_can"),
    ("redbull_silver_can", "ds", "newcans/cans/red_bull_silver_slim/asset.usda", "tippable_can"),
]

TIPPABLE_CATS = {"tippable_can", "tippable_bottle", "tippable_tube"}

HOME_MISC_SHELF_POOL = [
    "book_pink", "note_green_pad", "cookie_tin_static", "cardboard_box_static",
    "paper_ball_static", "newspaper_static", "chipped_mug_static", "spent_pen_static",
    "worn_eraser_static", "lays_party_chip_bag", "cacao_pouch", "lindt_gold_bag_static",
    "pringles_mini_cup", "pringles_purple_tube", "empty_can_static", "zbar_multipack_box",
    "honeymaid_family_box", "skittles_share_pouch", "diet_coke_silver_can",
]

RETAIL_SHELF_SKUS = [
    "blue_cereal_box", "cacao_pouch", "chomps_jerky_box", "chomps_teal_pouch",
    "coke_10pack_carton", "sprite_10pack_carton", "diet_coke_silver_can",
    "gatorade_blue_static", "gatorade_orange_static", "honeymaid_family_box",
    "lays_party_chip_bag", "lindt_gold_bag_static", "pringles_mini_cup",
    "pringles_purple_tube", "pringles_red_tube", "pringles_smoky_tube",
    "redbull_4pack_box", "skittles_share_pouch", "smores_kit_carton",
    "zbar_multipack_box", "redbull_silver_can", "cookie_tin_static", "cardboard_box_static",
]

CLUSTER_RECIPES = {
    "study_mess": [
        ["book_pink", "note_green_pad", "spent_pen_static", "worn_eraser_static", "paper_ball_static", "chipped_mug_static"],
        ["book_pink", "paper_ball_static", "spent_pen_static", "newspaper_static", "empty_can_static"],
    ],
    "living_snack_mess": [
        ["lays_party_chip_bag", "empty_can_static", "sprite_open_empty_can", "newspaper_static", "paper_ball_static"],
        ["skittles_share_pouch", "diet_coke_silver_can", "pringles_purple_tube", "japanese_tray_static", "paper_ball_static"],
    ],
    "kitchen_prep_mess": [
        ["cutlery_holder_static", "japanese_tray_static", "cacao_pouch", "chipped_mug_static", "empty_can_static", "ceramic_fragments_static"],
        ["honeymaid_family_box", "pringles_red_tube", "sprite_open_empty_can", "paper_ball_static", "chipped_mug_static"],
    ],
    "parcel_trunk_mess": [
        ["cardboard_box_static", "cookie_tin_static", "newspaper_static", "gatorade_orange_static", "paper_ball_static"],
        ["cardboard_box_static", "coke_10pack_carton", "zbar_multipack_box", "empty_can_static"],
    ],
}


def load_static_lived_in_props(mig_root: Path, cloth_demo: Path, bundle, out_asset_dir: Path) -> dict:
    aa_root = mig_root / "Runtime-room-0.1.7/libraries/current_assets/all_assets"
    props = {}
    out_asset_dir.mkdir(parents=True, exist_ok=True)
    bcache = UsdGeom.BBoxCache(0, ["default", "render", "proxy"])

    for pid, src_kind, rel_p, cat in STATIC_PROP_SPECS:
        fp = (aa_root / rel_p) if src_kind == "aa" else (cloth_demo / rel_p)
        if not fp.exists():
            continue
        if fp.suffix == ".usdz" or src_kind == "ds":
            layer_path = fp
        else:
            try:
                layer_path = bundle.clone(fp)
            except FileNotFoundError:
                continue

        st = Usd.Stage.Open(str(layer_path))
        if not st:
            continue
        root = st.GetDefaultPrim()
        if not root:
            continue
        bx = bounds(bcache, root)
        if bx is None:
            continue
        rbs = [str(p.GetPath())[len(str(root.GetPath())):] for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        has_col = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(root))
        props[pid] = {
            "layer": str(layer_path),
            "box": bx,
            "size": bx[1] - bx[0],
            "cat": cat,
            "rbs": rbs,
            "has_col": has_col,
        }
    return props


def _aabb_hits(cand: np.ndarray, occupied: list[np.ndarray], sep: float = 0.012) -> bool:
    for ob in occupied:
        if (
            cand[1, 0] + sep > ob[0, 0] and cand[0, 0] - sep < ob[1, 0]
            and cand[1, 1] + sep > ob[0, 1] and cand[0, 1] - sep < ob[1, 1]
            and cand[1, 2] > ob[0, 2] - 0.004 and cand[0, 2] < ob[1, 2] + 0.004
        ):
            return True
    return False


def _author_static_prop(
    ep_stage: Usd.Stage,
    prim_path: str,
    prop_info: dict,
    center_xy: tuple[float, float],
    support_z: float,
    yaw_deg: float,
    *,
    is_decal: bool = False,
    lay_flat: bool = False,
    add_collider: bool = True,
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    bx = prop_info["box"]
    scl_m = Gf.Matrix4d().SetScale(Gf.Vec3d(*scale_xyz))
    tilt_m = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)) if lay_flat else Gf.Matrix4d(1.0)
    rot_z = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(yaw_deg)))
    xf_m = scl_m * tilt_m * rot_z

    corners = np.array([
        xf_m.Transform(Gf.Vec3d(float(x), float(y), float(z)))
        for x in bx[:, 0] for y in bx[:, 1] for z in bx[:, 2]
    ])
    rmin, rmax = corners.min(axis=0), corners.max(axis=0)
    tx = float(center_xy[0] - (rmin[0] + rmax[0]) * 0.5)
    ty = float(center_xy[1] - (rmin[1] + rmax[1]) * 0.5)
    tz = float(support_z - rmin[2] + (0.0012 if is_decal else 0.001))
    xf_m.SetTranslateOnly(Gf.Vec3d(tx, ty, tz))

    xf_prim = UsdGeom.Xform.Define(ep_stage, prim_path)
    xf_prim.MakeMatrixXform().Set(xf_m)
    mod = ep_stage.DefinePrim(f"{prim_path}/Model")
    mod.GetReferences().AddReference(prop_info["layer"])

    for rb_sub in prop_info["rbs"]:
        rb_p = f"{prim_path}/Model{rb_sub}"
        ep_stage.OverridePrim(rb_p).CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    if not add_collider:
        for sub_p in Usd.PrimRange(mod):
            if sub_p.HasAPI(UsdPhysics.CollisionAPI):
                sub_p.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    elif not is_decal and not prop_info["has_col"]:
        col_c = UsdGeom.Cube.Define(ep_stage, f"{prim_path}/StaticCol")
        col_c.CreateSizeAttr(1.0)
        sz = rmax - rmin
        cm = Gf.Matrix4d().SetScale(Gf.Vec3d(float(max(0.02, sz[0] * 0.90)), float(max(0.02, sz[1] * 0.90)), float(max(0.015, sz[2] * 0.90))))
        cm.SetTranslateOnly(Gf.Vec3d(float((rmin[0] + rmax[0]) * 0.5), float((rmin[1] + rmax[1]) * 0.5), float((rmin[2] + rmax[2]) * 0.5)))
        col_c.MakeMatrixXform().Set(cm)
        UsdPhysics.CollisionAPI.Apply(col_c.GetPrim())
        col_c.CreatePurposeAttr("guide")

    return np.array([
        [center_xy[0] - (rmax[0] - rmin[0]) * 0.5, center_xy[1] - (rmax[1] - rmin[1]) * 0.5, support_z],
        [center_xy[0] + (rmax[0] - rmin[0]) * 0.5, center_xy[1] + (rmax[1] - rmin[1]) * 0.5, support_z + (rmax[2] - rmin[2])],
    ], float)


def populate_lived_in_scene_mess(
    ep_stage: Usd.Stage,
    ep_plan: dict,
    zones: dict,
    static_props: dict,
    occupied_dynamic: list[np.ndarray],
    countertop_keepouts: list[np.ndarray],
    rng: random.Random,
) -> int:
    ep_stage.DefinePrim("/World/LivedInClutter", "Xform")
    all_occ = [b.copy() for b in occupied_dynamic] + [b.copy() for b in countertop_keepouts]
    prop_counter = 0
    is_supermarket = ep_plan["scene_archetype"] == "S1_commercial_supermarket_aisles"

    # 1. Surface Weathering: Matte roughness perturbation on tables/worktops/shelves
    for sh_path in (
        "/World/Background/Shell/WorktopMat/S",
        "/World/Background/Shell/CabinetMat/S",
        "/World/Assets/CoffeeTable/Looks/CoffeeTableMat",
        "/World/SupermarketLooks/GondolaMat/PreviewSurface",
        "/World/SupermarketLooks/FloorTileMat/PreviewSurface",
    ):
        pr = ep_stage.GetPrimAtPath(sh_path)
        if pr and pr.IsA(UsdShade.Shader):
            UsdShade.Shader(ep_stage.OverridePrim(sh_path)).CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(round(rng.uniform(0.48, 0.76), 3))

    # 2. Coffee/Water Stain Decals (`table_stain_grid`)
    if "table_stain_grid" in static_props:
        stain_info = static_props["table_stain_grid"]
        stain_surfaces = [
            k for k in (
                "living_coffee_table_top", "kitchen_dining_table", "kitchen_side_table",
                "dark_sink_worktop", "wall_side_desk", "supermarket_deli_counter",
            ) if k in zones
        ]
        for sk in stain_surfaces:
            zbox, _, _ = zones[sk][0]
            if np.any((zbox[1, :2] - zbox[0, :2]) < 0.28):
                continue
            for _ in range(rng.randint(1, 2)):
                sx = rng.uniform(float(zbox[0, 0] + 0.12), float(zbox[1, 0] - 0.12))
                sy = rng.uniform(float(zbox[0, 1] + 0.12), float(zbox[1, 1] - 0.12))
                sc = rng.uniform(0.65, 1.12)
                _author_static_prop(
                    ep_stage, f"/World/LivedInClutter/Stain_{prop_counter:03d}",
                    stain_info, (sx, sy), float(zbox[1, 2]), rng.uniform(-180.0, 180.0),
                    is_decal=True, scale_xyz=(sc, sc, 1.0),
                )
                prop_counter += 1

    # 3A. SUPERMARKET MODE (`S1`): High-density Multi-Row Retail Shelf Facings + Depletion Gaps (`~95` shelf items!)
    if is_supermarket:
        valid_skus = [k for k in RETAIL_SHELF_SKUS if k in static_props]
        shelf_added = 0
        for szk in ("pantry_shelf", "supermarket_gondolas"):
            if szk not in zones:
                continue
            boards = list(zones[szk])
            rng.shuffle(boards)
            for zbox, _, max_h in boards:
                if shelf_added >= 98:
                    break
                z_top = float(zbox[1, 2])
                dx, dy = float(zbox[1, 0] - zbox[0, 0]), float(zbox[1, 1] - zbox[0, 1])
                along_x = dx >= dy
                length, depth = (dx, dy) if along_x else (dy, dx)
                if length < 0.25 or depth < 0.16:
                    continue
                fit_skus = [s for s in valid_skus if static_props[s]["size"][2] <= max_h - 0.01]
                if not fit_skus:
                    continue
                n_lanes = max(2, min(5, int(length / 0.24)))
                lane_w = (length - 0.06) / n_lanes
                n_rows = 2 if depth >= 0.34 else 1
                for lane_i in range(n_lanes):
                    sku = rng.choice(fit_skus)
                    pinfo = static_props[sku]
                    psz = pinfo["size"]
                    base_yaw = (0.0 if along_x else 90.0) + rng.choice((0.0, 180.0))
                    n_cols = max(1, min(3, int(lane_w / (max(psz[0], psz[1]) + 0.018))))
                    for col_i in range(n_cols):
                        for row_i in range(n_rows):
                            if rng.random() < 0.22 or shelf_added >= 98:
                                continue
                            u_pos = 0.04 + lane_i * lane_w + (col_i + 0.5) * (lane_w / n_cols) + rng.uniform(-0.008, 0.008)
                            v_pos = (0.28 if row_i == 0 else 0.72) * depth + rng.uniform(-0.010, 0.010)
                            px = float(zbox[0, 0] + (u_pos if along_x else v_pos))
                            py = float(zbox[0, 1] + (v_pos if along_x else u_pos))
                            r_half = max(psz[0], psz[1]) * 0.52
                            if px - r_half < zbox[0, 0] + 0.01 or px + r_half > zbox[1, 0] - 0.01 or py - r_half < zbox[0, 1] + 0.01 or py + r_half > zbox[1, 1] - 0.01:
                                continue
                            cand = np.array([[px - r_half, py - r_half, z_top], [px + r_half, py + r_half, z_top + psz[2]]], float)
                            if _aabb_hits(cand, all_occ, sep=0.014):
                                continue
                            b_placed = _author_static_prop(
                                ep_stage, f"/World/LivedInClutter/RetailShelf_{prop_counter:03d}",
                                pinfo, (px, py), z_top, base_yaw + rng.uniform(-12.0, 12.0),
                            )
                            all_occ.append(b_placed)
                            prop_counter += 1
                            shelf_added += 1
    else:
        # 3B. HOUSEHOLD MODE (`S2..S6`): Sparser than Supermarket, richer Garage/Pantry racks + Chaotic Heterogeneous Toss!
        home_pool = [k for k in HOME_MISC_SHELF_POOL if k in static_props]
        if "pantry_shelf" in zones and home_pool:
            boards = list(zones["pantry_shelf"])
            rng.shuffle(boards)
            home_shelf_budget = rng.randint(36, 46) if ep_plan["scene_archetype"] == "S6_cargo_trunk_and_garage_rack_bay" else rng.randint(18, 26)
            shelf_added = 0
            per_tier_lo, per_tier_hi = (4, 6) if ep_plan["scene_archetype"] == "S6_cargo_trunk_and_garage_rack_bay" else (2, 4)
            for zbox, _, max_h in boards:
                if shelf_added >= home_shelf_budget:
                    break
                z_top = float(zbox[1, 2])
                tier_items = rng.sample(home_pool, k=min(len(home_pool), rng.randint(per_tier_lo, per_tier_hi)))
                for sku in tier_items:
                    if shelf_added >= home_shelf_budget:
                        break
                    pinfo = static_props[sku]
                    tip = (pinfo["cat"] in TIPPABLE_CATS) and (rng.random() < 0.38)
                    eff_h = min(pinfo["size"][:2]) if tip else pinfo["size"][2]
                    eff_r = (max(pinfo["size"]) * 0.52) if tip else (max(pinfo["size"][:2]) * 0.52)
                    if eff_h > max_h - 0.01:
                        continue
                    for _try in range(8):
                        if (zbox[1, 0] - zbox[0, 0]) <= 2 * eff_r + 0.03 or (zbox[1, 1] - zbox[0, 1]) <= 2 * eff_r + 0.03:
                            break
                        px = rng.uniform(float(zbox[0, 0] + eff_r + 0.015), float(zbox[1, 0] - eff_r - 0.015))
                        py = rng.uniform(float(zbox[0, 1] + eff_r + 0.015), float(zbox[1, 1] - eff_r - 0.015))
                        cand = np.array([[px - eff_r, py - eff_r, z_top], [px + eff_r, py + eff_r, z_top + eff_h]], float)
                        if _aabb_hits(cand, all_occ, sep=0.016):
                            continue
                        b_placed = _author_static_prop(
                            ep_stage, f"/World/LivedInClutter/HomeShelf_{prop_counter:03d}",
                            pinfo, (px, py), z_top, rng.uniform(-180.0, 180.0), lay_flat=tip,
                        )
                        all_occ.append(b_placed)
                        prop_counter += 1
                        shelf_added += 1
                        break

    # 3C. REFRIGERATOR INTERIOR SHELVES: Stock chilled drinks, jars & snacks inside the main fridge cavity!
    if "fridge_interior_shelves" in zones:
        fridge_pool = [
            k for k in (
                "diet_coke_silver_can", "gatorade_blue_static", "gatorade_orange_static",
                "redbull_silver_can", "glass_jar_static", "pringles_mini_cup",
                "cacao_pouch", "chomps_teal_pouch", "cookie_tin_static",
                "coke_10pack_carton", "sprite_10pack_carton", "smores_kit_carton",
            ) if k in static_props
        ]
        if fridge_pool:
            f_occ = []
            f_budget = rng.randint(14, 16) if is_supermarket else rng.randint(12, 16)
            f_added = 0
            for zbox, _, max_h in zones["fridge_interior_shelves"]:
                if f_added >= f_budget:
                    break
                z_top = float(zbox[1, 2])
                rng.shuffle(fridge_pool)
                tier_cap = rng.randint(3, 4)
                tier_done = 0
                for sku in fridge_pool:
                    if f_added >= f_budget or tier_done >= tier_cap:
                        break
                    pinfo = static_props[sku]
                    tip = (pinfo["cat"] in TIPPABLE_CATS) and (rng.random() < 0.25)
                    eff_h = min(pinfo["size"][:2]) if tip else pinfo["size"][2]
                    eff_r = (max(pinfo["size"]) * 0.52) if tip else (max(pinfo["size"][:2]) * 0.52)
                    if eff_h > max_h - 0.010:
                        continue
                    if (zbox[1, 0] - zbox[0, 0]) <= 2 * eff_r + 0.02 or (zbox[1, 1] - zbox[0, 1]) <= 2 * eff_r + 0.02:
                        continue
                    for _try in range(10):
                        px = rng.uniform(float(zbox[0, 0] + eff_r + 0.01), float(zbox[1, 0] - eff_r - 0.01))
                        py = rng.uniform(float(zbox[0, 1] + eff_r + 0.01), float(zbox[1, 1] - eff_r - 0.01))
                        cand = np.array([[px - eff_r, py - eff_r, z_top], [px + eff_r, py + eff_r, z_top + eff_h]], float)
                        if _aabb_hits(cand, f_occ, sep=0.010):
                            continue
                        b_placed = _author_static_prop(
                            ep_stage, f"/World/LivedInClutter/FridgeInt_{prop_counter:03d}",
                            pinfo, (px, py), z_top, rng.uniform(-180.0, 180.0),
                            lay_flat=tip, add_collider=False,
                        )
                        f_occ.append(b_placed)
                        prop_counter += 1
                        f_added += 1
                        tier_done += 1
                        break

    # 4. Tabletop / Counter / Desk / Sofa Messy Scatter (Wide surfaces get 2 separated casual clusters!)
    if is_supermarket:
        surface_plan = [
            ("supermarket_deli_counter", "kitchen_prep_mess", 2),
            ("supermarket_checkout_counter", "parcel_trunk_mess", 2),
        ]
    else:
        surface_plan = [
            ("living_coffee_table_top", "living_snack_mess", 2),
            ("wall_side_desk", "study_mess", 2),
            ("tv_console_surface", "living_snack_mess", 1),
            ("kitchen_dining_table", "kitchen_prep_mess", 2),
            ("kitchen_side_table", "kitchen_prep_mess", 2),
            ("dark_sink_worktop", "kitchen_prep_mess", 2),
            ("entryway_pack_top", "parcel_trunk_mess", 1),
            ("nightstand_surface", "study_mess", 1),
            ("study_credenza_cubby", "study_mess", 1),
            ("nightstand_or_trunk_ledge", "parcel_trunk_mess", 1),
            ("sofa_surface", "living_snack_mess", 1),
        ]

    for zkey, recipe_key, n_clusters in surface_plan:
        if zkey not in zones:
            continue
        surf_slabs = list(zones[zkey])
        if zkey == "dark_sink_worktop" and len(surf_slabs) > 4:
            rng.shuffle(surf_slabs)
            surf_slabs = surf_slabs[:4]
        for zbox, _, max_h in surf_slabs:
            span = zbox[1, :2] - zbox[0, :2]
            if np.any(span < 0.16):
                continue
            z_top = float(zbox[1, 2])
            max_span = max(float(span[0]), float(span[1]))
            if max_span >= 1.85 and not is_supermarket:
                eff_clusters = 3
            elif max_span >= 0.85:
                eff_clusters = n_clusters
            else:
                eff_clusters = 1
            for cl_i in range(eff_clusters):
                if eff_clusters >= 2:
                    frac_lo = (0.08 + 0.84 * (cl_i / eff_clusters))
                    frac_hi = (0.08 + 0.84 * ((cl_i + 1) / eff_clusters))
                    if float(span[0]) >= float(span[1]):
                        x_lo = zbox[0, 0] + span[0] * frac_lo
                        x_hi = zbox[0, 0] + span[0] * frac_hi
                        anchor_x = rng.uniform(float(x_lo), float(max(x_lo + 0.02, x_hi)))
                        anchor_y = rng.uniform(float(zbox[0, 1] + 0.08), float(max(zbox[0, 1] + 0.10, zbox[1, 1] - 0.08)))
                    else:
                        y_lo = zbox[0, 1] + span[1] * frac_lo
                        y_hi = zbox[0, 1] + span[1] * frac_hi
                        anchor_x = rng.uniform(float(zbox[0, 0] + 0.08), float(max(zbox[0, 0] + 0.10, zbox[1, 0] - 0.08)))
                        anchor_y = rng.uniform(float(y_lo), float(max(y_lo + 0.02, y_hi)))
                else:
                    anchor_x = rng.uniform(float(zbox[0, 0] + 0.08), float(max(zbox[0, 0] + 0.10, zbox[1, 0] - 0.08)))
                    anchor_y = rng.uniform(float(zbox[0, 1] + 0.08), float(max(zbox[0, 1] + 0.10, zbox[1, 1] - 0.08)))
                recipe = list(rng.choice(CLUSTER_RECIPES[recipe_key]))
                rng.shuffle(recipe)
                if not is_supermarket:
                    recipe = recipe[: rng.randint(4, 6)]

                for item_id in recipe:
                    if item_id not in static_props:
                        continue
                    pinfo = static_props[item_id]
                    tip = (not is_supermarket) and (pinfo["cat"] in TIPPABLE_CATS) and (rng.random() < 0.45)
                    eff_h = min(pinfo["size"][:2]) if tip else pinfo["size"][2]
                    eff_r = (max(pinfo["size"]) * 0.52) if tip else (max(pinfo["size"][:2]) * 0.52)
                    if eff_h > max_h - 0.01:
                        continue
                    for _try in range(10):
                        if (zbox[1, 0] - zbox[0, 0]) <= 2 * eff_r + 0.02 or (zbox[1, 1] - zbox[0, 1]) <= 2 * eff_r + 0.02:
                            break
                        px = float(np.clip(anchor_x + rng.gauss(0.0, 0.22), zbox[0, 0] + eff_r + 0.01, zbox[1, 0] - eff_r - 0.01))
                        py = float(np.clip(anchor_y + rng.gauss(0.0, 0.22), zbox[0, 1] + eff_r + 0.01, zbox[1, 1] - eff_r - 0.01))
                        cand = np.array([[px - eff_r, py - eff_r, z_top], [px + eff_r, py + eff_r, z_top + eff_h]], float)
                        if _aabb_hits(cand, all_occ, sep=0.014):
                            continue
                        casual_yaw = rng.uniform(-180.0, 180.0)
                        b_placed = _author_static_prop(
                            ep_stage, f"/World/LivedInClutter/MessProp_{prop_counter:03d}",
                            pinfo, (px, py), z_top, casual_yaw, lay_flat=tip,
                        )
                        all_occ.append(b_placed)
                        prop_counter += 1

                        # Skewed 2-high book/notebook stack in study/living mess
                        if item_id in ("book_pink", "cardboard_box_static") and rng.random() < 0.55 and (eff_h * 2.1 < max_h):
                            top_id = "note_green_pad" if item_id == "book_pink" else "cookie_tin_static"
                            if top_id in static_props:
                                b2 = _author_static_prop(
                                    ep_stage, f"/World/LivedInClutter/StackProp_{prop_counter:03d}",
                                    static_props[top_id],
                                    (px + rng.uniform(-0.025, 0.025), py + rng.uniform(-0.025, 0.025)),
                                    float(b_placed[1, 2]), casual_yaw + rng.uniform(18.0, 48.0),
                                )
                                all_occ.append(b2)
                                prop_counter += 1
                        break

    return prop_counter
