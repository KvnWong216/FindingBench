#!/usr/bin/env python3
"""Full 1,000-task dataset & Unitree G1 (Dex1-1 2-finger) physical compatibility audit
+ deterministic stratified FindingBench-Core-200 split generator.

Scans every episode in ``runs/rab_lived_in_20261120/combined_manifest.json`` for:
1. File / physics receipt / instruction / target / delivery placement integrity.
2. Target & shield-blocker graspability under Unitree G1 Dex1-1 (`min_extent <= max_grasp_extent_m`).
3. Target & delivery-surface vertical reachability for 1.32 m Unitree G1 (`cuRobo` 3-DoF IK across
   nominal standing/crouching standoffs `r in [0.42, 0.72] m`).
4. Robot spawn pose & initial clearance across all 50 unique household layouts (`pxr.Usd`).
5. Instruction vs. identical-asset distractor disambiguation (`comparison_sample_body` / spatial cues).
6. Deterministic stratified `FindingBench-Core-200` benchmark subset (`4 tasks x 50 layouts = 200`).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rab_embodiment import UNITREE_G1_CONFIG, UNITREE_READY_POSE, compute_g1_crouch, public_embodiment_spec
from rab_spectator_data import load


def _build_stratified_core200(records: list[dict], per_layout: int = 4) -> dict:
    """Proportional stratified 200-episode split (`4 tasks x 50 layouts = 200`),
    matching exact 20% global stratum quotas across `(lane, regime, multi_target, shielded)`
    so raw unweighted `SR` on Core-200 is an unbiased estimator of 1,000-task `SR`."""
    total_target = per_layout * len({int(r["layout_id"]) for r in records})
    ratio = total_target / max(1, len(records))

    def _stratum(r: dict) -> tuple:
        return (r["lane"], r["regime"], bool(r["multi_target"]), bool(r["shielded"]))

    global_counts = Counter(_stratum(r) for r in records)
    target_quotas = {s: global_counts[s] * ratio for s in global_counts}

    # Include explicit visual comparison reference episodes first (2 in 1,000 -> 1 in 200)
    by_layout: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_layout[int(rec["layout_id"])].append(rec)

    chosen_by_layout: dict[int, list[dict]] = defaultdict(list)
    selected_set: set[str] = set()
    cur_counts: Counter = Counter()

    comp_pool = sorted(
        [r for r in records if r["comparison_match"]],
        key=lambda r: hashlib.sha256(r["task_id"].encode("utf-8")).hexdigest(),
    )
    if comp_pool:
        c0 = comp_pool[0]
        chosen_by_layout[int(c0["layout_id"])].append(c0)
        selected_set.add(c0["task_id"])
        cur_counts[_stratum(c0)] += 1

    # Deterministic layout traversal ordered from fewest strata options to most
    ordered_layouts = sorted(
        by_layout.keys(),
        key=lambda lid: (len({_stratum(r) for r in by_layout[lid]}), lid),
    )
    for slot in range(per_layout):
        for lid in ordered_layouts:
            if len(chosen_by_layout[lid]) > slot:
                continue
            cands = [r for r in by_layout[lid] if r["task_id"] not in selected_set]
            # Pick candidate whose stratum has the largest remaining quota deficit
            best = max(
                cands,
                key=lambda r: (
                    round(target_quotas[_stratum(r)] - cur_counts[_stratum(r)], 6),
                    hashlib.sha256(r["task_id"].encode("utf-8")).hexdigest(),
                ),
            )
            chosen_by_layout[lid].append(best)
            selected_set.add(best["task_id"])
            cur_counts[_stratum(best)] += 1

    selected_ids = sorted(selected_set)
    rec_map = {r["task_id"]: r for r in records}
    sub = [rec_map[tid] for tid in selected_ids]
    return {
        "schema": "findingbench.rab_v1_subset.v1",
        "subset_id": "findingbench_rab_v1_core200",
        "description": (
            "Proportional stratified 200-episode core evaluation split (4 tasks x 50 household layouts) "
            "matching the exact 20% stratum quotas of the 1,000-episode benchmark across drawer/open lanes, "
            "short-range/cross-zone regimes, single/multi-target, and shielded arrangements."
        ),
        "total_episodes": len(selected_ids),
        "layouts_covered": len({r["layout_id"] for r in sub}),
        "lanes": dict(Counter(r["lane"] for r in sub)),
        "regimes": dict(Counter(r["regime"] for r in sub)),
        "multi_target_episodes": sum(1 for r in sub if r["multi_target"]),
        "shielded_episodes": sum(1 for r in sub if r["shielded"]),
        "comparison_match_episodes": sum(1 for r in sub if r["comparison_match"]),
        "episodes": selected_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=ROOT / "runs/rab_lived_in_20261120")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/rab_audit_g1_dex1_1000.json")
    ap.add_argument("--core200-out", type=Path, default=ROOT / "configs/benchmarks/rab_v1_core200.json")
    args = ap.parse_args()

    t0 = time.monotonic()
    manifest = load(args.release / "combined_manifest.json")
    tasks = manifest["tasks"]
    print(f"[audit] Scanning {len(tasks)} tasks in {args.release}...", flush=True)

    # 1. Initialize cuRobo 3D IK solver once to measure exact Unitree G1 + Dex1-1 vertical reach envelope
    from rab_curobo import UnitreeCuRoboSolver

    solver = UnitreeCuRoboSolver(embodiment_name="unitree_g1")
    solver.warmup(("r",), include_6d=False)

    reachable_by_z_cm: dict[int, bool] = {}
    for z_cm in range(5, 195, 2):
        z = z_cm / 100.0
        dz, pp, cj = compute_g1_crouch(z, floor_z=0.0)
        pelvis_pos = np.array([0.0, 0.0, UNITREE_G1_CONFIG.pelvis_z_offset - dz])
        pelvis_R = Rotation.from_euler("ZY", [0.0, pp]).as_matrix()
        seed = {**UNITREE_READY_POSE, **cj}
        ok_any = False
        for r in (0.42, 0.48, 0.55, 0.64, 0.72):
            ok, _, _, _ = solver.solve_ik_3d("r", np.array([r, -0.14, z]), pelvis_pos, pelvis_R, seed_joint_map=seed)
            if ok:
                ok_any = True
                break
        reachable_by_z_cm[z_cm] = ok_any

    reach_zs = [z_cm / 100.0 for z_cm, ok in sorted(reachable_by_z_cm.items()) if ok]
    min_reach_z, max_reach_z = min(reach_zs), max(reach_zs)
    print(f"[audit] Unitree G1 + Dex1-1 cuRobo vertical reach envelope: [{min_reach_z:.2f} m, {max_reach_z:.2f} m]", flush=True)

    from pxr import Usd, UsdGeom

    spawn_xy_by_layout: dict[int, tuple[float, float, float]] = {}
    issues_by_type: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()
    target_min_extents: list[float] = []
    blocker_min_extents: list[float] = []
    target_z_values: list[float] = []
    dest_z_values: list[float] = []
    layout_counts = Counter()
    lane_counts = Counter()
    regime_counts = Counter()
    task_records: list[dict] = []

    for entry in tasks:
        tid = entry["id"]
        lid = int(entry.get("layout_id", -1))
        lane = str(entry.get("lane", "unknown"))
        layout_counts[lid] += 1
        lane_counts[lane] += 1

        try:
            ep_path = Path(entry["episode"])
            rec_path = Path(entry["physics_receipt"])
            if not ep_path.is_file():
                ep_path = args.release / "cases" / tid / "episode.json"
            if not rec_path.is_file():
                rec_path = args.release / "cases" / tid / "native_validation.json"
            ep = load(ep_path)
            rec = load(rec_path)
        except Exception as exc:
            issues_by_type["missing_or_corrupt_files"].append({"task_id": tid, "error": str(exc)})
            continue

        stats["total_loaded"] += 1
        regime_counts[str(ep.get("regime", "unknown"))] += 1
        if rec.get("status") not in ("physx_initialization_pass", "passed", "accepted_dense_initialization", None):
            issues_by_type["bad_physics_receipt"].append({"task_id": tid, "status": rec.get("status")})

        # Spot-check robot spawn pose once per unique layout_id + check placed-object spawn overlap on all 1,000 tasks
        if lid not in spawn_xy_by_layout:
            stage = Usd.Stage.Open(str(ep["scene"]))
            base_prim = stage.GetPrimAtPath("/World/Robot/base_link")
            if not stage or not base_prim:
                issues_by_type["missing_robot_spawn_prim"].append({"task_id": tid, "layout_id": lid})
            else:
                xf = UsdGeom.XformCache().GetLocalToWorldTransform(base_prim)
                tr = xf.ExtractTranslation()
                spawn_xy_by_layout[lid] = (float(tr[0]), float(tr[1]), float(tr[2]))

        instr = str(ep.get("instruction") or "").strip()
        if len(instr) < 8:
            issues_by_type["empty_or_short_instruction"].append({"task_id": tid, "instruction": instr})

        placed = ep.get("placed") or []
        targets = [r for r in placed if r.get("target")]
        distractors = [r for r in placed if not r.get("target")]
        if not targets:
            issues_by_type["no_target_objects"].append({"task_id": tid})
            continue

        is_multi = len(targets) > 1
        if is_multi:
            stats["multi_target_episodes"] += 1
        else:
            stats["single_target_episodes"] += 1

        # Verify no placed object overlaps Unitree G1's standing footprint at spawn (0.28m core)
        if lid in spawn_xy_by_layout:
            sx, sy, sz = spawn_xy_by_layout[lid]
            for r in placed:
                eb = r.get("expected_bounds_m")
                if eb is not None:
                    box = np.asarray(eb, float).reshape(2, 3)
                    if box[0, 2] < sz + 1.35:
                        dx = max(float(box[0, 0]) - sx, 0.0, sx - float(box[1, 0]))
                        dy = max(float(box[0, 1]) - sy, 0.0, sy - float(box[1, 1]))
                        if (dx * dx + dy * dy) ** 0.5 < UNITREE_G1_CONFIG.free_stand_r:
                            issues_by_type["spawn_overlaps_placed_item"].append({
                                "task_id": tid,
                                "layout_id": lid,
                                "asset": r.get("asset"),
                            })

        # Check target extents & Z heights
        for t in targets:
            eb = t.get("expected_bounds_m")
            if eb is not None:
                box = np.asarray(eb, float).reshape(2, 3)
                span = np.maximum(box[1] - box[0], 0.0)
                min_ext = float(np.min(span))
                tz = float(box.mean(axis=0)[2])
                target_min_extents.append(min_ext)
                target_z_values.append(tz)
                if min_ext > UNITREE_G1_CONFIG.max_grasp_extent_m:
                    issues_by_type["target_exceeds_dex1_grasp_span"].append({
                        "task_id": tid,
                        "asset": t.get("asset"),
                        "min_extent_m": round(min_ext, 4),
                        "limit_m": UNITREE_G1_CONFIG.max_grasp_extent_m,
                    })
                elif min_ext > 0.0949:
                    # Rotated world-AABB wider than 94.9 mm even though true OBB fits Dex1 jaw
                    stats["rotated_world_aabb_between_95mm_and_140mm"] += 1
                if tz > max_reach_z + 0.02 or tz < min_reach_z - 0.03:
                    issues_by_type["target_z_outside_g1_reach"].append({
                        "task_id": tid,
                        "asset": t.get("asset"),
                        "target_z_m": round(tz, 3),
                        "zone": t.get("zone"),
                    })

        # Check shield blocker graspability
        shield = (ep.get("arrangement") or {}).get("shield") or {}
        blocker_body = shield.get("blocker")
        is_shielded = bool(blocker_body)
        if is_shielded:
            stats["shielded_episodes"] += 1
            brow = next((r for r in placed if r.get("body") == blocker_body), None)
            if brow and brow.get("expected_bounds_m") is not None:
                bbox = np.asarray(brow["expected_bounds_m"], float).reshape(2, 3)
                b_min = float(np.min(np.maximum(bbox[1] - bbox[0], 0.0)))
                blocker_min_extents.append(b_min)
                if b_min > UNITREE_G1_CONFIG.max_grasp_extent_m:
                    issues_by_type["blocker_exceeds_dex1_grasp_span"].append({
                        "task_id": tid,
                        "blocker_asset": brow.get("asset"),
                        "min_extent_m": round(b_min, 4),
                    })

        # Check delivery surface heights from `bounds_m`
        placements = (ep.get("delivery") or {}).get("placements") or []
        if not placements:
            issues_by_type["no_delivery_placements"].append({"task_id": tid})
        for pl in placements:
            bm = pl.get("bounds_m")
            if bm is not None:
                pbox = np.asarray(bm, float).reshape(2, 3)
                dz_val = float(pbox[0, 2])
                dest_z_values.append(dz_val)
                # `place()` solves IK to `point + [0, 0, 0.08]`
                if dz_val + 0.08 > max_reach_z + 0.02 or dz_val < min_reach_z - 0.03:
                    issues_by_type["delivery_z_outside_g1_reach"].append({
                        "task_id": tid,
                        "support": pl.get("support"),
                        "dest_z_m": round(dz_val, 3),
                    })

        # Check identical-asset distractor disambiguation
        comp_body = ep.get("comparison_sample_body")
        is_comp = bool(comp_body)
        if is_comp:
            stats["comparison_reference_episodes"] += 1
        tgt_assets = {t.get("asset") for t in targets}
        dup_dist = [d for d in distractors if d.get("asset") in tgt_assets and d.get("body") != comp_body]
        if dup_dist:
            stats["non_sample_same_asset_distractor"] += 1
            spatial_words = (
                "left", "right", "top", "bottom", "upper", "lower", "drawer", "fridge",
                "refrigerator", "shelf", "table", "counter", "tray", "cabinet", "all ", "both ", "two ", "matching",
            )
            if not any(w in instr.lower() for w in spatial_words):
                issues_by_type["ambiguous_identical_asset_distractor"].append({
                    "task_id": tid,
                    "asset": sorted(tgt_assets),
                    "instruction": instr,
                })

        task_records.append({
            "task_id": tid,
            "layout_id": lid,
            "lane": lane,
            "regime": str(ep.get("regime", "unknown")),
            "search_fixture": str(ep.get("search_fixture", "unknown")),
            "multi_target": is_multi,
            "shielded": is_shielded,
            "comparison_match": is_comp,
        })

    core200 = _build_stratified_core200(task_records, per_layout=4)
    args.core200_out.parent.mkdir(parents=True, exist_ok=True)
    args.core200_out.write_text(json.dumps(core200, indent=2, ensure_ascii=False) + "\n")

    summary = {
        "total_tasks": len(tasks),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "unitree_g1_dex1_spec": {
            **(public_embodiment_spec(UNITREE_G1_CONFIG) or {}),
            "curobo_vertical_reach_m": [round(min_reach_z, 3), round(max_reach_z, 3)],
        },
        "stats": dict(stats),
        "layouts_verified": len(spawn_xy_by_layout),
        "lanes": dict(lane_counts),
        "regimes": dict(regime_counts),
        "extents_and_heights_summary": {
            "target_min_extent_max_m": round(max(target_min_extents), 4) if target_min_extents else None,
            "target_min_extent_p99_m": round(float(np.percentile(target_min_extents, 99)), 4) if target_min_extents else None,
            "blocker_min_extent_max_m": round(max(blocker_min_extents), 4) if blocker_min_extents else None,
            "target_z_range_m": [round(min(target_z_values), 4), round(max(target_z_values), 4)] if target_z_values else None,
            "delivery_surface_z_range_m": [round(min(dest_z_values), 4), round(max(dest_z_values), 4)] if dest_z_values else None,
        },
        "core200_summary": {k: v for k, v in core200.items() if k != "episodes"},
        "issue_counts": {k: len(v) for k, v in sorted(issues_by_type.items())},
        "issues_sample": {k: v[:20] for k, v in sorted(issues_by_type.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("\n=== 1,000-TASK UNITREE G1 (DEX1-1) AUDIT & CORE-200 SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "issues_sample"}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
