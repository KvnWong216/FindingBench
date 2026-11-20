"""vla_difficulty_v2 location certification (GPU, one scene load). Evaluator-only.

Input: runs/vla_difficulty_v2/_survey/<scene>.candidates.json (analyze_survey.py)

TARGET locations (every on_top surface / drawer interior candidate):
  1. reset, move the probe there (containers: open -> place in link -> close)
  2. relation holds after settling?
  3. from EVERY start pose: start-panorama RGB visibility (px + Lab contrast),
     with the container closed (and open, for containers)
  4. GRASP pose search from the scene's first start (RouteOracleAgent): all
     feasible base poses + route steps
  5. containers: at the chosen pose with the container CLOSED, probe pixels
     and GRASP feasibility (must be 0 / False for S2)

GOAL receptacles (same candidates): the probe is grasped ONCE for real
through the public protocol (PLACE feasibility depends on the grasp offset),
then for each receptacle (opened first if it is a container) the PLACE pose
search runs counterfactually and all feasible base poses are recorded.

    python scripts/v2/certify_locations.py --scene Rs_int [--only-targets|--only-goals]

Writes runs/vla_difficulty_v2/_locations/<scene>.json (flushed after every row)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_scene import COMMON, ROOT, scene_block  # noqa: E402
from visibility import rgb_visibility  # noqa: E402

SURVEY = ROOT / "runs/vla_difficulty_v2/_survey"
OUT = ROOT / "runs/vla_difficulty_v2/_locations"
PROBE = {"name": "probe_mug", "category": "mug", "model": "dhnxww"}   # v1-certified grasp/place
# furniture people naturally put things on / in (no oven tops, sinks, bushes, washers ...)
FURNITURE = ("countertop", "table", "desk", "shelf", "bookcase", "bottom_cabinet", "nightstand",
             "bar_", "dresser", "chest")


MIN_DRAWER_Z = 0.45
PLACE_SEARCH_S = 300.0   # route_oracle default 600 s; certification only needs a witness


def natural(candidate: dict) -> bool:
    return any(k in candidate["category"] for k in FURNITURE)


def quat(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def build_scenario(scene: str, cands: dict, seed: str | None = None) -> Path:
    common = yaml.safe_load(COMMON.read_text())
    starts = cands["starts"]
    # seed surface for the probe: a surface where GRASP was certified (goal
    # phase), else an on_top candidate in the first start's room
    seed = seed or next((c["entity"] for c in cands["candidates"]
                 if c["room"] == starts[0]["room"] and any(r["relation"] == "on_top" for r in c["relations"])),
                next(c["entity"] for c in cands["candidates"]
                     if any(r["relation"] == "on_top" for r in c["relations"])))
    spec = {
        "id": f"v2_certify_{scene}", "instruction": "certify",
        "scene": scene_block(scene),
        "robot": {**common["robot"], "init_anchor": starts[0]["name"]},
        "anchors": {s["name"]: {"position": [s["x"], s["y"], 0.0053], "orientation": quat(s["yaw"])}
                    for s in starts},
        "target": {"entity": PROBE["name"], "category": PROBE["category"]},
        "objects": [dict(PROBE)],
        "placements": [{"entity": PROBE["name"], "relation": "on_top", "receptacle": seed}],
        "termination": dict(common["termination"]), "feasibility": dict(common["feasibility"]),
        "skills": list(common["skills"]), "expose_available_skills": False,
        "goal": [{"kind": "holding", "entity": PROBE["name"]}],
    }
    path = ROOT / f"build/vla_difficulty_v2/_certify/{scene}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--only-targets", action="store_true")
    p.add_argument("--only-goals", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cands = json.loads((SURVEY / f"{a.scene}.candidates.json").read_text())
    cands["candidates"] = [c for c in cands["candidates"] if natural(c)]
    seed = None
    if a.only_goals:
        # the goal phase grasps the probe for real: start it on a surface whose
        # GRASP route was certified in the targets phase
        prev = json.loads((OUT / f"{a.scene}.json").read_text())
        seed = next((t["receptacle"] for t in prev["targets"] if t["relation"] == "on_top"
                     and (t.get("grasp") or {}).get("ok")), None)
        if seed is None:
            raise SystemExit(f"{a.scene}: no certified graspable surface for the goal phase")
    scen_path = build_scenario(a.scene, cands, seed)
    suffix = ".goals" if a.only_goals else ".targets" if a.only_targets else ""
    out_path = OUT / f"{a.scene}{suffix}.json"
    rep: dict = {"scene": a.scene, "scenario": str(scen_path), "probe": PROBE,
                 "starts": cands["starts"], "targets": [], "goals": []}

    def flush():
        out_path.write_text(json.dumps(rep, indent=1, default=str) + "\n")

    try:
        from rummagebench.authoring.tasks.room_map import RoomMap
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.environment.env import InteractiveSearchEnv
        import rummagebench.evaluation.route_oracle as route_oracle
        from rummagebench.evaluation.route_oracle import OracleFailure, RouteOracleAgent
        route_oracle.MAX_SEARCH_S_PLACE = PLACE_SEARCH_S
        from rummagebench.sim.omnigibson.backend import resolve_object
        from rummagebench.sim.omnigibson.probe_support import relation_holds

        env = InteractiveSearchEnv(scen_path, mode="agent")
        b, vs = env._backend, env._session
        root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
        rm = RoomMap.load(root / "scenes" / a.scene, root / "metadata" / "room_categories.txt")
        probe = PROBE["name"]
        z0 = 0.0053

        def to(x, y, yaw):
            b.teleport_robot(AnchorSpec(position=[x, y, z0], orientation=quat(yaw)))

        def search(skill: str, entity: str) -> dict:
            oracle = RouteOracleAgent(vs, room_map=rm)
            t = time.time()
            res: dict = {}
            try:
                res["route_steps"] = len(oracle._route_to_interaction_pose(skill, entity))
                res["ok"] = True
            except OracleFailure as e:
                res["ok"] = False
                res["failure"] = str(e)
            for entry in oracle.log:
                if entry.get("feasible_poses"):
                    res["feasible_poses"] = entry["feasible_poses"]
                    res["chosen"] = entry.get("chosen")
            res["search_s"] = round(time.time() - t, 1)
            return res

        def view_all(open_state: str) -> dict:
            out = {}
            here = b.robot_pose()
            for s in cands["starts"]:
                to(s["x"], s["y"], s["yaw"])
                out[s["name"]] = rgb_visibility(b.capture_panorama_frame(), probe)
            b.teleport_robot(AnchorSpec(position=list(here[0]), orientation=list(here[1])))
            return {f"{k}": v for k, v in out.items()} if open_state else out

        # drawers whose rim is below MIN_DRAWER_Z were never graspable for the
        # R1Pro arm (Rs_int, 2026-10-25: 0 of 9) -- skip their 4-minute searches
        targets = [] if a.only_goals else [
            (c["entity"], r["relation"], r.get("link"), c["room"])
            for c in cands["candidates"] for r in c["relations"]
            if r["relation"] == "on_top" or r["z"] >= MIN_DRAWER_Z]
        for rec, relation, link, room in targets:
            row: dict = {"receptacle": rec, "relation": relation, "link": link, "room": room}
            rep["targets"].append(row)
            t0 = time.time()
            try:
                vs.reset()
                obj = resolve_object(b._env.scene, probe)
                robj = resolve_object(b._env.scene, rec)
                container = relation == "inside" and b.articulation_info(rec) is not None
                if container:
                    b.set_open(rec, True)
                    b.settle(10)
                placed = b._place_in_link(obj, robj, link) if link else b._place(obj, robj, relation)
                b.settle(20)
                row["placed"] = bool(placed)
                if container:
                    b.set_open(rec, False)
                    b.settle(20)
                row["relation_holds"] = bool(relation_holds(b, probe, rec, relation))
                if not (row["placed"] and row["relation_holds"]):
                    continue
                box = b.entity_aabb(probe)
                row["probe_xy"] = [round((box[0][0] + box[1][0]) / 2, 3), round((box[0][1] + box[1][1]) / 2, 3)]
                row["probe_room"] = rm.room_at(*row["probe_xy"])
                row["vis_closed"] = view_all("")
                if container:
                    b.set_open(rec, True)
                    b.settle(10)
                    row["relation_holds_after_open"] = bool(relation_holds(b, probe, rec, relation))
                    row["vis_open"] = view_all("open")
                s0 = cands["starts"][0]
                to(s0["x"], s0["y"], s0["yaw"])
                vs._capture_current()
                row["grasp"] = search("GRASP", probe)
                chosen = (row["grasp"].get("chosen") or {}).get("pose")
                if container and chosen:
                    to(*chosen)
                    b.set_open(rec, False)
                    b.settle(10)
                    row["at_pose_closed_px"] = rgb_visibility(b.capture_panorama_frame(), probe)["px"]
                    row["at_pose_closed_grasp_feasible"] = RouteOracleAgent(vs, room_map=rm)._feasible("GRASP", probe)
                    b.set_open(rec, True)
                    b.settle(10)
                    row["at_pose_open_px"] = rgb_visibility(b.capture_panorama_frame(), probe)["px"]
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            finally:
                row["elapsed_s"] = round(time.time() - t0, 1)
                flush()

        if not a.only_targets:
            # grasp the probe once, for real, at its scenario placement
            vs.reset()
            oracle = RouteOracleAgent(vs, room_map=rm)
            oracle.reset("certify")
            oracle._subgoals = [("GRASP", probe)]
            steps = 0
            while vs.world_state.held_object != probe and steps < 40:
                action = oracle.act(vs.observe())
                if action is None or action.get("skill") == "REPORT_DONE":
                    break
                vs.step(action)
                steps += 1
            rep["goal_probe_grasped"] = vs.world_state.held_object == probe
            rep["goal_probe_grasp_steps"] = steps
            if not rep["goal_probe_grasped"]:
                raise RuntimeError(f"could not grasp the probe: {oracle.failure}")
            home = b.robot_pose()
            goals = [(c["entity"], c["room"], [r for r in c["relations"]])
                     for c in cands["candidates"]]
            for rec, room, rels in goals:
                for r in rels:
                    if r["relation"] == "inside" and r is not next(x for x in rels if x["relation"] == "inside"):
                        continue   # one PLACE search per container (OPEN opens all drawers)
                    row = {"receptacle": rec, "relation": r["relation"], "room": room}
                    rep["goals"].append(row)
                    t0 = time.time()
                    try:
                        b.teleport_robot(AnchorSpec(position=list(home[0]), orientation=list(home[1])))
                        container = r["relation"] == "inside" and b.articulation_info(rec) is not None
                        if container:
                            b.set_open(rec, True)
                            b.settle(10)
                            reg = b.receptacle_region(rec)
                            row["opened_region"] = None if reg is None else {
                                "kind": reg.kind, "lo": list(reg.lo), "hi": list(reg.hi)}
                        row["place"] = search("PLACE", rec)
                    except Exception as e:  # noqa: BLE001
                        row["error"] = f"{type(e).__name__}: {e}"
                    finally:
                        if r["relation"] == "inside":
                            try:
                                b.set_open(rec, False)
                                b.settle(10)
                            except Exception:  # noqa: BLE001
                                pass
                        row["elapsed_s"] = round(time.time() - t0, 1)
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
