"""Certify candidate target / goal locations for vla_difficulty_v1 in ONE
scene load (GPU). Evaluator-only; nothing here is agent-facing.

For every candidate TARGET location (receptacle, relation[, link]):
  1. reset, move the scenario target there (containers: open -> place -> close)
  2. relation holds after settling?
  3. start-panorama visible pixels with the container closed (S gates)
  4. for containers: open it again (OPEN is part of the task)
  5. RouteOracleAgent pose search: a footprint-free base pose where GRASP is
     feasible AND a MOVE/TURN route to it from the start pose (steps)

For every candidate GOAL receptacle: the target is marked held
(semantic state only) and the same search runs for PLACE.

    python scripts/certify_vla_locations.py build/vla_difficulty_v1/T02_mug_to_table__L2_S1/scenario.yaml \
        --target-locations microwave_bfbeeb_0:inside:base_link countertop_tpuwys_4:on_top \
        --goal-receptacles breakfast_table_uhrsex_0 coffee_table_qlmqyy_0

Writes runs/vla_difficulty_v1/_locations/<scenario id>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _label_pixels(frame, entity: str) -> int:
    seg = np.asarray(frame.instance_segmentation)
    ids = [int(k) for k, v in frame.meta.get("instance_labels", {}).items() if v == entity]
    return int(np.isin(seg, ids).sum()) if ids else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", type=Path)
    p.add_argument("--target-locations", nargs="*", default=[],
                   help="receptacle:relation[:link]")
    p.add_argument("--goal-receptacles", nargs="*", default=[])
    p.add_argument("--out", type=Path, default=Path("runs/vla_difficulty_v1/_locations"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"scenario": str(a.scenario), "targets": [], "goals": []}
    out_path = a.out / f"{a.scenario.parent.name}.json"

    def flush():
        out_path.write_text(json.dumps(rep, indent=1, default=str) + "\n")

    try:
        from rummagebench.authoring.tasks.room_map import RoomMap
        from rummagebench.core.scenario import load_scenario
        from rummagebench.environment.env import InteractiveSearchEnv
        from rummagebench.evaluation.route_oracle import OracleFailure, RouteOracleAgent
        from rummagebench.sim.omnigibson.backend import resolve_object
        from rummagebench.sim.omnigibson.probe_support import relation_holds

        scenario = load_scenario(a.scenario)
        target = scenario.target.entity
        env = InteractiveSearchEnv(a.scenario, mode="agent")
        b, vs = env._backend, env._session
        root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
        rm = RoomMap.load(root / "scenes" / scenario.scene.model,
                          root / "metadata" / "room_categories.txt")

        def search(skill: str, entity: str) -> dict:
            oracle = RouteOracleAgent(vs, room_map=rm)
            t = time.time()
            res: dict = {}
            try:
                route = oracle._route_to_interaction_pose(skill, entity)
                res["route_steps"] = len(route)
                res["ok"] = True
            except OracleFailure as e:
                res["ok"] = False
                res["failure"] = str(e)
            res["search_log"] = oracle.log
            res["search_s"] = round(time.time() - t, 1)
            return res

        for spec in a.target_locations:
            rec, relation, *link = spec.split(":")
            link = link[0] if link else None
            row: dict = {"receptacle": rec, "relation": relation, "link": link}
            rep["targets"].append(row)
            try:
                vs.reset()
                obj = resolve_object(b._env.scene, target)
                robj = resolve_object(b._env.scene, rec)
                container = relation == "inside" and b.articulation_info(rec) is not None
                if container:
                    b.set_open(rec, True)
                    b.settle(10)
                if link is not None:
                    placed = b._place_in_link(obj, robj, link)
                else:
                    placed = b._place(obj, robj, relation)
                b.settle(20)
                row["placed"] = bool(placed)
                if container:
                    b.set_open(rec, False)
                    b.settle(20)
                row["relation_holds"] = relation_holds(b, target, rec, relation)
                box = b.entity_aabb(target)
                row["target_aabb"] = [[round(float(v), 3) for v in box[0]],
                                      [round(float(v), 3) for v in box[1]]]
                row["target_room"] = rm.room_at((box[0][0] + box[1][0]) / 2,
                                                (box[0][1] + box[1][1]) / 2)
                frame = b.capture_panorama_frame()
                row["start_panorama_px_closed"] = _label_pixels(frame, target)
                if container:
                    b.set_open(rec, True)
                    b.settle(10)
                    row["start_panorama_px_open"] = _label_pixels(b.capture_panorama_frame(), target)
                    row["relation_holds_after_open"] = relation_holds(b, target, rec, relation)
                vs._capture_current()  # oracle reads the session's current frame
                row["grasp"] = search("GRASP", target)
                chosen = next((l["chosen"] for l in row["grasp"]["search_log"]
                               if l.get("chosen")), None)
                if container and chosen is not None:
                    # does the container really gate the target? From the
                    # interaction pose, with the container CLOSED: pixels and
                    # GRASP feasibility (both must be 0 / False for S2)
                    from rummagebench.core.scenario import AnchorSpec
                    from rummagebench.skills.move import quat_from_yaw
                    z = b.robot_pose()[0][2]
                    x, y, yaw = chosen["pose"]
                    b.teleport_robot(AnchorSpec(position=[x, y, z], orientation=quat_from_yaw(yaw)))
                    b.set_open(rec, False)
                    b.settle(10)
                    row["at_pose_closed_px"] = _label_pixels(b.capture_panorama_frame(), target)
                    o = RouteOracleAgent(vs, room_map=rm)
                    row["at_pose_closed_grasp_feasible"] = o._feasible("GRASP", target)
                    b.set_open(rec, True)
                    b.settle(10)
                    row["at_pose_open_px"] = _label_pixels(b.capture_panorama_frame(), target)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            flush()

        for rec in a.goal_receptacles:
            row = {"receptacle": rec}
            rep["goals"].append(row)
            try:
                # really grasp the target at its scenario placement through the
                # public protocol (PLACE feasibility needs the grasp offset)
                vs.reset()
                oracle = RouteOracleAgent(vs, room_map=rm)
                oracle.reset(scenario.instruction)
                oracle._subgoals = [("GRASP", target)]
                steps = 0
                while vs.world_state.held_object != target and steps < 40:
                    action = oracle.act(vs.observe())
                    if action is None or action.get("skill") == "REPORT_DONE":
                        break
                    vs.step(action)
                    steps += 1
                row["grasped"] = vs.world_state.held_object == target
                row["grasp_steps"] = steps
                if not row["grasped"]:
                    raise RuntimeError(f"could not grasp {target}: {oracle.failure}")
                if b.articulation_info(rec) is not None and not b.is_open(rec):
                    b.set_open(rec, True)   # inside goals: OPEN precedes PLACE
                    b.settle(10)
                    region = b.receptacle_region(rec)
                    row["opened_region"] = None if region is None else {
                        "kind": region.kind, "lo": list(region.lo), "hi": list(region.hi)}
                row["place"] = search("PLACE", rec)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            flush()
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
