"""Run ONE compiled vla_difficulty_v2 physical configuration (one process).

Same three levels as scripts/run_vla_variant.py (v1), with the v2 S gates:

  1. task_initialization : env builds, reset succeeds, target placement
     holds, goal initially unsatisfied, S gates measured on the start frame
       S1  same room, RGB-visible in the start panorama (visibility.s1_visible:
           >= 60 px AND Lab contrast >= 15 vs its surroundings)
       S2  same room, 0 px in the start panorama and best VIEW; container
           subtype: closed, target revealed (> 0 px) right after the oracle's
           OPEN; occluded subtype: hidden behind furniture on a surface
       S3  other room, 0 px in the start panorama and best VIEW
  2. null_baseline       : REPORT_DONE at step 0 must FAIL
  3. scripted_execution  : the route oracle solves it through the public
     protocol in <= MAX_ORACLE_STEPS (half the agent budget)

L1 / L2 / L3 variants of one (base task, S) differ only in the instruction
text, so one run certifies all three (compile_variants.py checks that the
scenarios are otherwise identical).

    python scripts/v2/run_variant.py build/vla_difficulty_v2/<variant_id>/scenario.yaml

Writes runs/vla_difficulty_v2/<variant_id>/report.json (+ null_baseline/, route_oracle/)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from run_vla_variant import _best_view, _label_pixels, _room_map  # noqa: E402
from visibility import rgb_visibility, s1_visible  # noqa: E402

ROOT = HERE.parents[1]
MAX_ORACLE_STEPS = 30


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", type=Path)
    p.add_argument("--out-root", type=Path, default=ROOT / "runs/vla_difficulty_v2")
    a = p.parse_args()

    meta_path = a.scenario.parent / "variant.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out = a.out_root / a.scenario.parent.name
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"scenario": str(a.scenario), "variant": meta,
                    "command": " ".join([sys.executable, *sys.argv]),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "task_initialization": {"ok": False}, "model_used": False}
    t0 = time.time()
    try:
        from rummagebench.core.scenario import load_scenario
        from rummagebench.environment.env import InteractiveSearchEnv
        from rummagebench.evaluation.episode_log import run_visual_episode
        from rummagebench.evaluation.goal_checker import check_goal
        from rummagebench.evaluation.route_oracle import RouteOracleAgent
        from rummagebench.sim.omnigibson.probe_support import relation_holds

        scenario = load_scenario(a.scenario)
        env = InteractiveSearchEnv(a.scenario, mode="agent")
        backend, vs = env._backend, env._session
        report["env_build_s"] = round(time.time() - t0, 1)
        root, RoomMap = _room_map()
        rm = RoomMap.load(root / "scenes" / scenario.scene.model, root / "metadata" / "room_categories.txt")

        # ---------------------------------------------- 1. initialization
        obs = env.reset()["observation"]
        target = scenario.target.entity
        tp = next(pl for pl in scenario.placements if pl.entity == target)
        init = report["task_initialization"]
        init["reset"] = True
        init["instruction"] = obs["instruction"]
        init["target_relation_holds"] = bool(relation_holds(backend, target, tp.receptacle, tp.relation))
        init["target_aabb"] = [list(map(float, v)) for v in backend.entity_aabb(target)]
        c = [(init["target_aabb"][0][i] + init["target_aabb"][1][i]) / 2 for i in range(2)]
        rpos = backend.robot_pose()[0]
        init["robot_start"] = [round(v, 3) for v in rpos]
        init["start_room"] = rm.room_at(rpos[0], rpos[1])
        init["target_room"] = rm.room_at(*c)
        init["start_to_target_m"] = round(float(np.hypot(c[0] - rpos[0], c[1] - rpos[1])), 2)
        gc = check_goal(vs.world_state, scenario, backend)
        init["goal_initially_satisfied"] = gc.satisfied
        pano = vs._panorama_frame
        vis = rgb_visibility(pano, target)
        init["start_panorama_target"] = vis
        npx, angles, vframe = _best_view(pano, backend.robot_pose(), init["target_aabb"], target,
                                         scenario.robot.image_width, scenario.robot.image_height)
        init["start_best_view_target_px"] = npx
        init["start_best_view_angles"] = angles
        if vframe is not None:
            from PIL import Image
            Image.fromarray(vframe.rgb).save(out / "start_best_view.png")
        (out / "start_panorama.png").write_bytes(base64.b64decode(obs["image_png_b64"]))
        level_s = meta.get("spatial_level")
        gates: dict = {}
        if level_s == "S1":
            gates["same_room"] = init["target_room"] == init["start_room"]
            gates["rgb_visible_at_start"] = s1_visible(vis)
        elif level_s == "S2":
            gates["same_room"] = init["target_room"] == init["start_room"]
            gates["invisible_at_start"] = vis["px"] == 0 and npx == 0
            if tp.relation == "inside":   # container S2 (occluded-surface S2 has no container)
                gates["container_closed"] = not backend.is_open(tp.receptacle)
        elif level_s == "S3":
            gates["other_room"] = init["target_room"] != init["start_room"]
            gates["invisible_at_start"] = vis["px"] == 0 and npx == 0
        init["ok"] = bool(init["target_relation_holds"] and not init["goal_initially_satisfied"])

        # ---------------------------------------------- 2. null baseline
        class ReportImmediately:
            name = "report-done-immediately"

            def reset(self, instruction):
                self._sent = False

            def act(self, observation):
                if self._sent:
                    return None
                self._sent = True
                return {"skill": "REPORT_DONE"}

        nb = run_visual_episode(vs, ReportImmediately(), out / "null_baseline")
        report["null_baseline"] = {"status": nb["status"], "ok": nb["status"] == "FAIL_FALSE_COMPLETION"}

        # ---------------------------------------------- 3. scripted oracle
        oracle = RouteOracleAgent(vs, room_map=rm)
        t1 = time.time()
        summ = run_visual_episode(vs, oracle, out / "route_oracle", save_images=True)
        gc = check_goal(vs.world_state, scenario, backend)
        se = report["scripted_execution"] = {
            "status": summ["status"], "success": summ["status"] == "SUCCESS",
            "run_valid": summ["run_valid"], "planning_steps_used": summ["planning_steps_used"],
            "max_planning_steps": summ["max_planning_steps"], "oracle_failure": oracle.failure,
            "oracle_log": oracle.log, "goal_check_final": gc.predicates, "error": summ["error"],
            "duration_s": round(time.time() - t1, 1)}
        if oracle.route_debug is not None:
            (out / "route_debug.json").write_text(json.dumps(oracle.route_debug) + "\n")
        if level_s == "S2" and tp.relation == "inside":
            reveal = None
            ev_path = out / "route_oracle" / "events.jsonl"
            if ev_path.exists():
                for line in ev_path.read_text().splitlines():
                    r = json.loads(line)
                    if r.get("selected_entity") == tp.receptacle and r.get("feedback") == "EXECUTED":
                        reveal = (r.get("observed_evidence") or {}).get(target, 0)
                        break
            se["target_px_after_open"] = reveal
            gates["revealed_after_open"] = bool(reveal)
        init["s_gates"] = gates
        init["s_gates_pass"] = all(gates.values()) if gates else None
        report["accepted"] = bool(
            init["ok"] and init["s_gates_pass"] and report["null_baseline"]["ok"]
            and se["success"] and se["planning_steps_used"] <= MAX_ORACLE_STEPS)
    except BaseException as error:  # noqa: BLE001 - report everything
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["accepted"] = False
    report["total_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"variant": out.name, "accepted": report.get("accepted"), "error": report.get("error")}),
          flush=True)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Kit teardown can hang; the report is already written
