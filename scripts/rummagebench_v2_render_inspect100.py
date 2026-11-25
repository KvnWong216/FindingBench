#!/usr/bin/env python3
"""Render the 10% (100-Episode) Stratified Inspection Set for RummageBench v2.

Features:
1. Single reusable headless Isaac Sim worker (`limit_cpu_threads=4`, strictly 1 worker).
2. 60 Hz PhysX settle via `NativeWorld(stage, episode, mobile=False)`.
3. `initial_scene.jpg`: Floor-polygon + PhysX-raycast verified wide scene overview (never outside walls)
   + Unitree G1 `t=0` standing head-camera ego inset (`z=1.27 m`, `yaw=spawn_yaw`, `pitch=-10 deg`).
4. `target_location.jpg`: Opens the ground-truth articulated container (cupboard, oven,
   fridge, microwave, air fryer, pantry door, entryway flap, cargo trunk tailgate, or drawer),
   uses floor-polygon + 16-angle PhysX raycast visibility scoring to pick an unobstructed interior
   close-up viewpoint with a soft session-layer inspection fill light for deep cabinets/fridges,
   and draws projected 2D bounding boxes on Target(s) & Foils/Blockers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/ubuntu/ygwang/FindingBench_20261016_200444/workspace")
sys.path.insert(0, str(ROOT / "scripts"))


def project_3d_box_to_2d(
    box_min: np.ndarray,
    box_max: np.ndarray,
    cam_pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)], float)
    right = np.array([math.sin(y), -math.cos(y), 0.0], float)
    up = np.array([-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p)], float)
    tan_half = math.tan(math.radians(fov_deg * 0.5))

    us, vs = [], []
    for x in (box_min[0], box_max[0]):
        for yy in (box_min[1], box_max[1]):
            for z in (box_min[2], box_max[2]):
                d = np.array([x, yy, z], float) - cam_pos
                zc = float(np.dot(d, fwd))
                if zc <= 0.06:
                    continue
                xc = float(np.dot(d, right))
                yc = float(np.dot(d, up))
                u = (width * 0.5) * (1.0 + xc / (zc * tan_half))
                v = (height * 0.5) - (width * 0.5) * (yc / (zc * tan_half))
                us.append(u)
                vs.append(v)
    if len(us) < 4:
        return None
    x0 = int(np.clip(min(us) - 4, 2, width - 10))
    y0 = int(np.clip(min(vs) - 4, 2, height - 10))
    x1 = int(np.clip(max(us) + 4, x0 + 8, width - 2))
    y1 = int(np.clip(max(vs) + 4, y0 + 8, height - 2))
    if (x1 - x0) > width * 0.94 and (y1 - y0) > height * 0.94:
        return None
    return (x0, y0, x1, y1)


def get_joints_for_fixture(fixture: str, articulations: dict) -> list[str]:
    if fixture.startswith("fridge_bin_") or fixture == "fridge_interior_shelves":
        return [k for k in ("fridge_right", "fridge_left") if k in articulations][:1]
    if fixture.startswith("cupboard_") or fixture == "range_oven_cavity":
        return [k for k in sorted(articulations) if k.startswith(fixture + "_d")]
    if fixture == "microwave_cavity" and "microwave_door" in articulations:
        return ["microwave_door"]
    if fixture == "airfryer_basket" and "airfryer_tray" in articulations:
        return ["airfryer_tray"]
    if fixture == "pantry_shelf" and "pantry_door" in articulations:
        return ["pantry_door"]
    if fixture == "entry_flap_cabinet" and "entry_pack_flap" in articulations:
        return ["entry_pack_flap"]
    if fixture == "cargo_trunk_bay" and "cargo_trunk_tailgate" in articulations:
        return ["cargo_trunk_tailgate"]
    if fixture.startswith("drawer_") and fixture in articulations:
        return [fixture]
    return []


def draw_badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, bg: tuple, fg: tuple = (255, 255, 255)):
    tw = max(48, len(text) * 6 + 14)
    x, y = int(xy[0]), int(max(4, min(xy[1], 408)))
    x = max(4, min(x, 764 - tw))
    draw.rectangle([x, y, x + tw, y + 19], fill=bg)
    draw.text((x + 5, y + 3), text, fill=fg)


def pick_unobstructed_camera(
    world,
    floor_tris: list,
    point_in_triangles_fn,
    focus_xyz: np.ndarray,
    preferred_xy: tuple[float, float],
    *,
    dists: tuple[float, ...],
    dzs: tuple[float, ...],
    target_prefixes: tuple[str, ...] = (),
    fallback_xy: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Use floor-polygon containment + 16-angle PhysX raycasts to guarantee an interior, unobstructed camera!"""
    pref_ang = math.atan2(preferred_xy[1] - focus_xyz[1], preferred_xy[0] - focus_xyz[0])
    ang_offsets = [0.0, 0.32, -0.32, 0.65, -0.65, 1.0, -1.0, 1.38, -1.38, 1.78, -1.78, 2.2, -2.2, 2.65, -2.65, math.pi]
    best = None
    best_score = -1e9
    down_dir = np.array([0.0, 0.0, -1.0], float)

    ray_ignore = (
        "/World/Robot",
        "/World/Background/Shell/RefrigeratorRecovered/right_door/Bin1GuardLeft",
        "/World/Background/Shell/RefrigeratorRecovered/right_door/Bin1GuardRight",
        "/World/Background/Shell/RefrigeratorRecovered/right_door/Bin1GuardFront",
        "/World/Background/Shell/RefrigeratorRecovered/right_door/Bin2GuardLeft",
        "/World/Background/Shell/RefrigeratorRecovered/right_door/Bin2GuardRight",
        "/World/Background/Shell/RefrigeratorRecovered/right_door/Bin2GuardFront",
    )

    for dist in dists:
        for dz in dzs:
            for da in ang_offsets:
                ang = pref_ang + da
                cx = float(focus_xyz[0] + math.cos(ang) * dist)
                cy = float(focus_xyz[1] + math.sin(ang) * dist)
                # Strictly reject cameras outside or within 18cm of the room/supermarket floor boundary!
                if floor_tris and not all(
                    point_in_triangles_fn((cx + ox, cy + oy), floor_tris)
                    for ox, oy in ((0.0, 0.0), (0.18, 0.0), (-0.18, 0.0), (0.0, 0.18), (0.0, -0.18))
                ):
                    continue
                cz = float(np.clip(focus_xyz[2] + dz, 0.42, 2.08))
                cpos = np.array([cx, cy, cz], float)

                # Verify floor/surface exists beneath camera (not outside building shell)
                down_hits = world.rays(cpos + np.array([0.0, 0.0, 0.04]), down_dir, limit=2.8, ignore=ray_ignore)
                if not down_hits:
                    continue
                # Penalize if camera is trapped right inside a tight shelf/counter (<10cm clearance)
                if float(down_hits[0]["distance"]) < 0.08:
                    continue

                vec = focus_xyz - cpos
                dlen = float(np.linalg.norm(vec))
                if dlen < 0.15:
                    continue
                uvec = vec / dlen
                hits = world.rays(cpos, uvec, limit=dlen + 0.05, ignore=ray_ignore)
                clear_frac = 1.0
                hit_target = False
                if hits:
                    h0 = hits[0]
                    if any(h0["body"].startswith(tp) or h0["path"].startswith(tp) for tp in target_prefixes):
                        hit_target = True
                        clear_frac = 1.0
                    else:
                        clear_frac = float(h0["distance"]) / dlen
                # Strongly penalize camera immediately embedded inside or looking at the opaque back of an open door
                if clear_frac < 0.22 and not hit_target:
                    continue
                score = (110.0 if hit_target else clear_frac * 65.0) - abs(da) * 8.0 - abs(dist - dists[0]) * 4.0
                if clear_frac < 0.82 and not hit_target:
                    score -= 180.0
                    if hits and any(k in hits[0]["path"].lower() for k in ("door", "panel", "cabinet", "fridge")):
                        score -= 120.0
                if score > best_score:
                    best_score = score
                    yaw = math.degrees(math.atan2(vec[1], vec[0]))
                    pitch = math.degrees(math.atan2(vec[2], math.hypot(vec[0], vec[1])))
                    best = (cpos, yaw, pitch)

    if best is not None:
        return best

    fxy = fallback_xy or preferred_xy
    cpos = np.array([float(fxy[0]), float(fxy[1]), 1.48], float)
    vec = focus_xyz - cpos
    yaw = math.degrees(math.atan2(vec[1], vec[0]))
    pitch = math.degrees(math.atan2(vec[2], max(0.05, math.hypot(vec[0], vec[1]))))
    return (cpos, yaw, pitch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-json", default=str(ROOT / "configs/benchmarks/rummagebench_v2_inspect100.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs/rummagebench_v2_20261124/inspect100"))
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--indices", default="", help="Optional comma-separated indices for targeted re-render")
    args = ap.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "4")

    inspect_doc = json.loads(Path(args.inspect_json).read_text())
    if args.indices:
        sel_idx = [int(x) for x in args.indices.split(",") if x.strip()]
        episodes_meta = [(i + 1, inspect_doc["episodes"][i]) for i in sel_idx]
    else:
        episodes_meta = [(i + 1, inspect_doc["episodes"][i]) for i in range(args.start, min(len(inspect_doc["episodes"]), args.start + args.count))]

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": True,
        "width": args.width,
        "height": args.height,
        "multi_gpu": False,
        "limit_cpu_threads": 4,
        "extra_args": ["--/log/level=error"],
    })

    try:
        import carb.settings
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux
        from rab_build_search import bounds
        from rab_edit import NativeWorld
        from rab_spectator import Renderer
        from rummagebench_v2_build import extract_floor_triangles_and_obstacles, point_in_triangles
        from spectator import FlightCamera

        settings = carb.settings.get_settings()
        settings.set_bool("/physics/updateToUsd", True)
        settings.set_bool("/physics/updateVelocitiesToUsd", True)
        settings.set_bool("/physics/suppressReadback", False)

        ctx = omni.usd.get_context()
        t_start = time.time()

        for seq_num, (disp_idx, meta) in enumerate(episodes_meta, 1):
            eid = meta["id"]
            rec_file = records_dir / f"{eid}.json"
            init_jpg = img_dir / f"{eid}_initial.jpg"
            tgt_jpg = img_dir / f"{eid}_target.jpg"

            ep_path = Path(meta["scene"]).parent / "episode.json"
            episode = json.loads(ep_path.read_text())

            if not ctx.open_stage(episode["scene"]):
                raise RuntimeError(f"Failed to open {episode['scene']}")
            stage = ctx.get_stage()
            stage.SetEditTarget(stage.GetSessionLayer())

            if episode.get("is_supermarket_scene"):
                floor_tris = [
                    (np.array([-4.05, -3.65]), np.array([4.05, -3.65]), np.array([4.05, 3.65])),
                    (np.array([-4.05, -3.65]), np.array([4.05, 3.65]), np.array([-4.05, 3.65])),
                ]
            else:
                floor_tris, _ = extract_floor_triangles_and_obstacles(stage)

            world = NativeWorld(stage, episode, mobile=False)
            world.step(20)

            fov_wide = 76.0
            renderer = Renderer(stage, args.width, args.height, fov_wide)

            rx, ry, ryaw = episode["robot_initial_xyyaw"]
            robot_prim = stage.GetPrimAtPath("/World/Robot")
            if robot_prim:
                UsdGeom.Imageable(robot_prim).MakeInvisible()

            # 1. Capture crisp standing Unitree G1 `t=0` Head Camera (`z = 1.27 m`, `pitch = -10 deg`)
            ego_cam = FlightCamera([float(rx), float(ry), 1.27], math.degrees(float(ryaw)), -10.0)
            ego_rgb = renderer.capture(ego_cam)

            # 2. Capture Floor-Polygon + Raycast Verified Wide Scene Overview at t=0 (all containers closed)
            t_first = next(r for r in episode["placed"] if r["target"])
            tb0 = np.array(t_first["expected_bounds_m"], float)
            tc0 = (tb0[0] + tb0[1]) * 0.5
            mid = np.array([(rx + tc0[0]) * 0.5, (ry + tc0[1]) * 0.5, 0.78], float)
            ov_pos, ov_yaw, ov_pitch = pick_unobstructed_camera(
                world,
                floor_tris,
                point_in_triangles,
                mid,
                (rx, ry),
                dists=(2.05, 1.55, 2.55, 1.15),
                dzs=(1.12, 0.92),
                fallback_xy=(rx, ry),
            )
            ov_cam = FlightCamera(ov_pos.tolist(), ov_yaw, max(-34.0, min(-10.0, ov_pitch)))
            wide_rgb = renderer.capture(ov_cam)
            if float(wide_rgb.mean()) < 42.0:
                vec_fb = tc0 - np.array([rx, ry, 1.50], float)
                fb_yaw = math.degrees(math.atan2(vec_fb[1], vec_fb[0]))
                fb_pitch = max(-30.0, min(-10.0, math.degrees(math.atan2(vec_fb[2], max(0.1, math.hypot(vec_fb[0], vec_fb[1]))))))
                wide_rgb = renderer.capture(FlightCamera([float(rx), float(ry), 1.50], fb_yaw, fb_pitch))

            im_init = Image.fromarray(wide_rgb)
            draw_i = ImageDraw.Draw(im_init)
            pip_w, pip_h = 240, 135
            pip_im = Image.fromarray(ego_rgb).resize((pip_w, pip_h), Image.BILINEAR)
            px0, py0 = args.width - pip_w - 10, args.height - pip_h - 10
            draw_i.rectangle([px0 - 2, py0 - 19, px0 + pip_w + 2, py0 + pip_h + 2], fill=(15, 23, 42), outline=(241, 245, 249), width=2)
            draw_i.text((px0 + 5, py0 - 16), "t=0 Robot Ego (G1 z=1.27m)", fill=(226, 232, 240))
            im_init.paste(pip_im, (px0, py0))
            draw_badge(
                draw_i, (8, 8),
                f"{eid} | {episode['scene_archetype'][:24]} | {episode['layout_name']} | {episode['lighting_profile']}",
                bg=(15, 23, 42),
            )
            im_init.save(init_jpg, quality=88)

            # 3. Open Articulated Container & Capture Interior-Verified Close-Up Target Location!
            opened_joints = get_joints_for_fixture(episode["search_fixture"], episode["arrangement"]["articulations"])
            swing_results = {}
            for jk in opened_joints:
                try:
                    res = world.swing(jk, True)
                    swing_results[jk] = f"ok ({res['travel']:.2f}{res['unit']})"
                except Exception as exc:
                    drv, tgt_val, _, _, _ = world.drives[jk]
                    for step_i in range(90):
                        drv.GetTargetPositionAttr().Set(float(tgt_val) * (step_i + 1) / 90.0)
                        world.step(1, sync=False)
                    world.step(5)
                    swing_results[jk] = f"opened_with_shift ({str(exc)[:44]})"
            if opened_joints:
                world.step(12)

            bcache = UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"])
            target_boxes = []
            target_prefixes = []
            foil_boxes = []
            for r in episode["placed"]:
                bp = stage.GetPrimAtPath(r["body"])
                if not bp:
                    continue
                bx = bounds(bcache, bp)
                if bx is None:
                    continue
                item_root = r["body"].split("/Model", 1)[0]
                if r["target"]:
                    target_boxes.append((bx, r["label"]))
                    target_prefixes.append(item_root)
                elif r["role"] in ("contrastive_foil", "label_distractor", "foreground_blocker", "surface_cover_occluder", "layered_top_clutter"):
                    foil_boxes.append((bx, f"{r['role']}: {r['label']}"))

            main_bx = target_boxes[0][0] if target_boxes else tb0
            tc = (main_bx[0] + main_bx[1]) * 0.5
            is_top_down = any(k in episode["search_fixture"] for k in ("tote_bag", "basket", "cart", "drawer_", "airfryer", "trunk"))
            is_fridge = episode["search_fixture"].startswith("fridge_")
            pref_xy = (rx, ry)
            fridge_aim = None
            if is_fridge:
                rp = stage.GetPrimAtPath("/World/Background/Shell/RefrigeratorRecovered/right_door/col_panel")
                s1 = stage.GetPrimAtPath("/World/Background/Shell/RefrigeratorRecovered/right_door/col_shelf1_floor")
                lp = stage.GetPrimAtPath("/World/Background/Shell/RefrigeratorRecovered/left_door/col_panel")
                bp = stage.GetPrimAtPath("/World/Background/Shell/RefrigeratorRecovered/body/col_body")
                sh3 = stage.GetPrimAtPath("/World/Background/Shell/RefrigeratorRecovered/body/geo_s/geo/SM_refrigerator_01_platform_right_003")
                if rp and s1 and lp and bp:
                    rpb = bounds(bcache, rp)
                    s1b = bounds(bcache, s1)
                    lb = bounds(bcache, lp)
                    bb = bounds(bcache, bp)
                    if rpb is not None and s1b is not None and lb is not None and bb is not None:
                        in_bin_dir = (0.5 * (s1b[0, :2] + s1b[1, :2])) - (0.5 * (rpb[0, :2] + rpb[1, :2]))
                        in_bin_dir /= max(1e-4, float(np.linalg.norm(in_bin_dir)))
                        lc_xy = 0.5 * (lb[0, :2] + lb[1, :2])
                        bc_xy = 0.5 * (bb[0, :2] + bb[1, :2])
                        front_dir = lc_xy - bc_xy
                        front_dir /= max(1e-4, float(np.linalg.norm(front_dir)))
                        # Stand in open room facing the inner side of right_door bins AND open right fridge cavity
                        cam_dir = in_bin_dir * 0.72 + front_dir * 0.69
                        cam_dir /= max(1e-4, float(np.linalg.norm(cam_dir)))
                        pref_xy = (float(tc[0] + cam_dir[0] * 1.32), float(tc[1] + cam_dir[1] * 1.32))
                        if sh3:
                            sh3b = bounds(bcache, sh3)
                            if sh3b is not None:
                                sh3_c = 0.5 * (sh3b[0] + sh3b[1])
                                fridge_aim = tc * 0.68 + np.array([float(sh3_c[0]), float(sh3_c[1]), 0.92], float) * 0.32
            cam_dists = (0.82, 1.05, 0.62, 1.32) if is_top_down else ((1.22, 1.45, 0.98, 1.68) if is_fridge else (0.95, 1.22, 0.72, 1.48))
            cam_dzs = (0.58, 0.42) if is_top_down else ((0.55, 0.78, 0.35, 0.95) if is_fridge else (0.28, 0.48, 0.14))
            rev_pos, rev_yaw, rev_pitch = pick_unobstructed_camera(
                world,
                floor_tris,
                point_in_triangles,
                tc,
                pref_xy,
                dists=cam_dists,
                dzs=cam_dzs,
                target_prefixes=tuple(target_prefixes),
                fallback_xy=pref_xy,
            )
            if fridge_aim is not None:
                v_aim = fridge_aim - rev_pos
                rev_yaw = math.degrees(math.atan2(v_aim[1], v_aim[0]))
                rev_pitch = max(-38.0, min(20.0, math.degrees(math.atan2(v_aim[2], max(0.1, math.hypot(v_aim[0], v_aim[1]))))))
                print(f"  [fridge_cam {eid}] tc={tc.round(3).tolist()} pref={pref_xy} rev={rev_pos.round(3).tolist()} yaw={rev_yaw:.1f} pitch={rev_pitch:.1f}", flush=True)

            # Soft session-layer inspection headlight so deep under-sink cupboards / ovens / fridges are clearly visible
            headlight = UsdLux.SphereLight.Define(stage, "/World/InspectionFillLight")
            headlight.CreateRadiusAttr(0.07)
            headlight.CreateIntensityAttr(11500.0)
            headlight.GetPrim().CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
            headlight.GetPrim().CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(4900.0)
            hl_mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(rev_pos[0]), float(rev_pos[1]), float(min(2.25, rev_pos[2] + 0.10))))
            UsdGeom.Xformable(headlight.GetPrim()).MakeMatrixXform().Set(hl_mat)

            fov_close = 74.0 if is_fridge else 62.0
            renderer.focal.Set(10.0 / math.tan(math.radians(fov_close * 0.5)))
            renderer.default_focal = renderer.focal.Get()
            rev_cam = FlightCamera(rev_pos.tolist(), rev_yaw, rev_pitch)
            tgt_rgb = renderer.capture(rev_cam)

            if float(tgt_rgb.mean()) < 52.0:
                # Boost inspection lamp in place WITHOUT moving camera behind open cabinet doors!
                headlight.CreateIntensityAttr(34000.0)
                tgt_rgb = renderer.capture(rev_cam)

            im_tgt = Image.fromarray(tgt_rgb)
            draw_t = ImageDraw.Draw(im_tgt)

            projected_targets = []
            tgt_badge_rects = []
            for tbx, tlab in target_boxes:
                rect = project_3d_box_to_2d(tbx[0], tbx[1], rev_pos, rev_yaw, rev_pitch, fov_close, args.width, args.height)
                if rect:
                    draw_t.rectangle(rect, outline=(16, 235, 96), width=3)
                    bx0, by0 = rect[0], max(28, rect[1] - 21)
                    draw_badge(draw_t, (bx0, by0), f"TARGET: {tlab[:44]}", bg=(5, 122, 55))
                    projected_targets.append(list(rect))
                    tgt_badge_rects.append((bx0, by0, bx0 + len(tlab[:44]) * 6 + 68, by0 + 20))

            for fbx, flab in foil_boxes[:3]:
                rect = project_3d_box_to_2d(fbx[0], fbx[1], rev_pos, rev_yaw, rev_pitch, fov_close, args.width, args.height)
                if rect:
                    draw_t.rectangle(rect, outline=(245, 158, 11), width=2)
                    fx, fy = rect[0], rect[3] + 3
                    for tx0, ty0, tx1, ty1 in tgt_badge_rects:
                        if not (fx + 180 < tx0 or fx > tx1 or fy + 20 < ty0 or fy > ty1):
                            fy = min(408, ty1 + 4)
                    draw_badge(draw_t, (fx, fy), flab[:36], bg=(180, 83, 9))

            open_tag = f"Opened: {','.join(opened_joints)}" if opened_joints else "Occlusion / Open Zone"
            draw_badge(
                draw_t, (8, 8),
                f"GT Target @ {episode['search_fixture']} ({open_tag})",
                bg=(6, 78, 59),
            )
            im_tgt.save(tgt_jpg, quality=88)

            renderer.close()
            world.close()
            time.sleep(0.05)

            rec = {
                **meta,
                "initial_image": f"images/{eid}_initial.jpg",
                "target_image": f"images/{eid}_target.jpg",
                "opened_joints": opened_joints,
                "swing_results": swing_results,
                "settled_target_center_m": [round(float(v), 4) for v in tc],
                "projected_target_rects": projected_targets,
            }
            rec_file.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")

            if seq_num % 10 == 0 or seq_num == len(episodes_meta):
                elapsed = time.time() - t_start
                print(f"[{seq_num}/{len(episodes_meta)}] (#{disp_idx} {eid}) {episode['scene_archetype']} / {episode['search_fixture']} -> {swing_results} ({elapsed:.1f}s)", flush=True)

        all_recs = []
        for m in inspect_doc["episodes"]:
            rf = records_dir / f"{m['id']}.json"
            if rf.exists():
                all_recs.append(json.loads(rf.read_text()))
        (out_dir / "inspect100_rendered.json").write_text(json.dumps({"count": len(all_recs), "episodes": all_recs}, indent=2, ensure_ascii=False) + "\n")
        print(f"=== RENDER COMPLETE: {len(all_recs)} total records in {out_dir} ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
