#!/usr/bin/env python
"""Why is a skill infeasible? Collision pairs of OPEN/PLACE around an entity.

One scene load. Grasps the scenario target through the public protocol
(route oracle), then, for the first N footprint-free ring poses around
--entity, runs the production feasibility check for --skill twice:
with the held object modelled and with the held body dropped from the
collision context (held_offset=None). Prints / saves, per pose, the verdict
and the collision pairs of each rejected candidate.

    python scripts/diagnose_vla_feasibility.py build/vla_difficulty_v1/<id>/scenario.yaml \
        --skill OPEN --entity bottom_cabinet_no_top_qohxjq_0 [--open-first X] [--poses 6]

Evaluator-side debugging only; writes runs/vla_difficulty_v1/_diagnose/<variant>_<skill>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path


def _pairs(verdict) -> list:
    out = []
    for c in (verdict.details or {}).get("candidates", []) if isinstance(verdict.details, dict) else []:
        col = c.get("collision") or {}
        appr = c.get("approach") or {}
        pairs = [f"{p.get('robot_link')}~{p.get('other')}" for p in col.get("pairs", [])]
        if appr.get("collision"):
            pairs += [f"approach:{p.get('robot_link')}~{p.get('other')}"
                      for p in appr["collision"].get("pairs", [])]
        if not pairs and appr and not appr.get("clear", True):
            pairs.append(f"approach:{appr.get('reason')}")
        out.append(pairs)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", type=Path)
    p.add_argument("--skill", required=True, choices=["OPEN", "PLACE", "CLOSE", "GRASP"])
    p.add_argument("--entity", required=True)
    p.add_argument("--open-first", nargs="*", default=[],
                   help="entities to open (set_open) before grasping")
    p.add_argument("--open-entity", action="store_true",
                   help="open --entity before the checks (inside-PLACE)")
    p.add_argument("--no-grasp", action="store_true")
    p.add_argument("--poses", type=int, default=6)
    p.add_argument("--execute-place", action="store_true",
                   help="PLACE only: at the first feasible pose run the real symbolic_place "
                        "and dump object AABB, region and relation (no rollback)")
    p.add_argument("--out", type=Path, default=Path("runs/vla_difficulty_v1/_diagnose"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    out_path = a.out / f"{a.scenario.parent.name}_{a.skill}_{a.entity}.json"
    rep: dict = {"scenario": str(a.scenario), "skill": a.skill, "entity": a.entity, "poses": []}

    def flush():
        out_path.write_text(json.dumps(rep, indent=1, default=str) + "\n")

    try:
        import numpy as np

        from rummagebench.authoring.tasks.navigation import FootprintChecker
        from rummagebench.authoring.tasks.room_map import RoomMap
        from rummagebench.core.scenario import AnchorSpec, load_scenario
        from rummagebench.environment.env import InteractiveSearchEnv
        from rummagebench.evaluation.route_oracle import (BASE_HALF_EXTENT, BASE_MARGIN,
                                                          RouteOracleAgent, _ring_poses)
        from rummagebench.skills.move import quat_from_yaw

        scenario = load_scenario(a.scenario)
        target = scenario.target.entity
        env = InteractiveSearchEnv(a.scenario, mode="agent")
        b, vs = env._backend, env._session
        root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
        rm = RoomMap.load(root / "scenes" / scenario.scene.model,
                          root / "metadata" / "room_categories.txt")
        vs.reset()
        for e in a.open_first:
            b.set_open(e, True)
            b.settle(10)
        vs._capture_current()
        ws = vs.world_state
        if not a.no_grasp:
            oracle = RouteOracleAgent(vs, room_map=rm)
            oracle.reset(scenario.instruction)
            oracle._subgoals = [("GRASP", target)]
            steps = 0
            while ws.held_object != target and steps < 40:
                action = oracle.act(vs.observe())
                if action is None or action.get("skill") == "REPORT_DONE":
                    break
                vs.step(action)
                steps += 1
            rep["grasped"] = ws.held_object == target
            rep["grasp_steps"] = steps
            off = ws.held_offset
            rep["held_offset"] = None if off is None else {
                "position": [round(float(v), 4) for v in off.position],
                "orientation": [round(float(v), 4) for v in off.orientation]}
            ag = getattr(b._robot, "_ag_obj_in_hand", {})
            rep["holding_arm"] = [arm for arm, o in ag.items() if o is not None]
            rep["modelled_arm"] = b._modelled_arm()
            box = b.entity_aabb(target)
            rep["held_extent_m"] = [round(float(box[1][i] - box[0][i]), 3) for i in range(3)]
            print("grasp:", {k: rep.get(k) for k in ("grasped", "holding_arm", "modelled_arm",
                                                      "held_offset", "held_extent_m")}, flush=True)
        if a.open_entity:
            b.set_open(a.entity, True)
            b.settle(10)
        flush()

        pos, quat = b.robot_pose()
        z = float(pos[2])
        chk = FootprintChecker.from_backend(b, BASE_HALF_EXTENT, BASE_MARGIN)
        ring = _ring_poses(b.entity_aabb(a.entity))
        free = chk.free_many(np.array([r[0] for r in ring]), np.array([r[1] for r in ring]),
                             np.array([r[2] for r in ring]))
        cands = [r for r, ok in zip(ring, free) if ok]
        cands.sort(key=lambda r: r[3])
        k = max(1, len(cands) // max(1, a.poses))
        cands = cands[::k][:a.poses]
        oracle = RouteOracleAgent(vs, room_map=rm)
        held_offset = ws.held_offset
        for c in cands:
            b.teleport_robot(AnchorSpec(position=[c[0], c[1], z], orientation=quat_from_yaw(c[2])))
            row: dict = {"pose": [round(v, 3) for v in c[:3]], "ring_offset": c[3]}
            for label, offset in (("held_modelled", held_offset), ("held_dropped", None)):
                if label == "held_dropped" and (ws.held_object is None or a.skill == "PLACE"):
                    continue  # PLACE requires the offset
                ws.held_offset = offset
                t = time.time()
                v = oracle._verdict(a.skill, a.entity)
                row[label] = {"feasible": bool(v.feasible), "reason": str(v.reason),
                              "s": round(time.time() - t, 1),
                              "pair_counts": dict(Counter(x for ps in _pairs(v) for x in ps))}
            ws.held_offset = held_offset
            rep["poses"].append(row)
            print(json.dumps(row), flush=True)
            flush()
            if a.execute_place and a.skill == "PLACE" and row["held_modelled"]["feasible"]:
                from rummagebench.evaluation.goal_checker import check_placement_relation
                v = oracle._verdict("PLACE", a.entity)
                it = (v.details or {}).get("interaction_target") or {}
                point = it.get("surface_point")
                before = b.entity_aabb(target)
                ok = b.symbolic_place(target, a.entity, at=point)
                after = b.entity_aabb(target)
                rel = check_placement_relation(b, target, a.entity, backend_held=bool(b.is_holding(target)))
                pose6 = b.entity_pose6d(target)
                rep["execute_place"] = {
                    "point": point, "realized": ok,
                    "held_aabb_before": [[round(float(x), 3) for x in c] for c in before],
                    "aabb_after": [[round(float(x), 3) for x in c] for c in after],
                    "extent_after": [round(float(after[1][i] - after[0][i]), 3) for i in range(3)],
                    "center_after": [round(float(after[0][i] + after[1][i]) / 2, 3) for i in range(3)],
                    "origin_after": [round(float(x), 3) for x in pose6.position],
                    "relation": rel.to_dict()}
                print("execute_place:", json.dumps(rep["execute_place"], default=str), flush=True)
                flush()
                break
        b.teleport_robot(AnchorSpec(position=list(pos), orientation=list(quat)))
        rep["ok"] = True
    except BaseException as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
    flush()
    print(json.dumps({"ok": rep.get("ok"), "error": rep.get("error"), "out": str(out_path)}), flush=True)
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
