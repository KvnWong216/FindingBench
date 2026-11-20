"""Run ONE compiled vla_difficulty_v1 variant in simulation (one process).

Three separately reported levels (never conflated):
  1. task_initialization : env builds, reset succeeds, target placement holds,
                           goal initially unsatisfied, S-level gates measured
  2. null_baseline       : REPORT_DONE at step 0 must FAIL (success checker
                           rejects an unachieved goal)
  3. scripted_execution  : the privileged route oracle (evaluation/route_oracle)
                           solves the task through the PUBLIC protocol
  4. policy_episode     : only with --agent module:factory — a model policy
                           through agents/policy_adapter (VLA Task Success)

    python scripts/run_vla_variant.py build/vla_difficulty_v1/T01_knife_fetch__L2_S1/scenario.yaml

Writes runs/vla_difficulty_v1/<variant_id>/{report.json, null_baseline/, route_oracle/}.
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
    labels = frame.meta.get("instance_labels", {})
    ids = [int(k) for k, v in labels.items() if v == entity]
    return int(np.isin(seg, ids).sum()) if ids else 0


def _best_view(pano, robot_pose, target_aabb, entity, width, height):
    """Best 640x480 VIEW aimed at the target from the current pose: the
    agent-visible pixel count (VIEW resamples the cached panorama)."""
    import math

    from rummagebench.perception.panorama import perspective_view
    from rummagebench.skills.move import yaw_from_quat

    (x, y, _), q = robot_pose
    c = [(target_aabb[0][i] + target_aabb[1][i]) / 2 for i in range(3)]
    bearing = math.degrees(math.atan2(c[1] - y, c[0] - x) - yaw_from_quat(q))
    best = (0, None, None)
    for dyaw in (-15, -5, 0, 5, 15):
        yaw = (bearing + dyaw + 180.0) % 360.0 - 180.0
        for pitch in range(-60, 31, 10):
            f = perspective_view(pano, yaw, pitch, width=width, height=height)
            n = _label_pixels(f, entity)
            if n > best[0]:
                best = (n, (round(yaw, 1), pitch), f)
    return best


def _room_map():
    from rummagebench.authoring.tasks.room_map import RoomMap

    root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
    return root, RoomMap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", type=Path)
    p.add_argument("--out-root", type=Path, default=Path("runs/vla_difficulty_v1"))
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--agent", help="policy to evaluate, 'module:factory' "
                   "(agents/policy_adapter.load_agent); reported as policy_episode")
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
        rm = RoomMap.load(root / "scenes" / scenario.scene.model,
                          root / "metadata" / "room_categories.txt")

        # ---------------------------------------------- 1. initialization
        obs = env.reset()["observation"]
        target = scenario.target.entity
        tp = next(pl for pl in scenario.placements if pl.entity == target)
        init = report["task_initialization"]
        init["reset"] = True
        init["instruction"] = obs["instruction"]
        init["target_relation_holds"] = relation_holds(backend, target, tp.receptacle, tp.relation)
        init["target_aabb"] = [list(map(float, v)) for v in backend.entity_aabb(target)]
        c = [(init["target_aabb"][0][i] + init["target_aabb"][1][i]) / 2 for i in range(2)]
        rpos = backend.robot_pose()[0]
        init["robot_start"] = [round(v, 3) for v in rpos]
        init["start_room"] = rm.room_at(rpos[0], rpos[1])
        init["target_room"] = rm.room_at(*c)
        init["start_to_target_m"] = round(float(np.hypot(c[0] - rpos[0], c[1] - rpos[1])), 2)
        init["initial_open"] = {e: backend.is_open(e) for e in scenario.initial_states}
        gc = check_goal(vs.world_state, scenario, backend)
        init["goal_initially_satisfied"] = gc.satisfied
        init["goal_check_initial"] = gc.predicates
        pano = vs._panorama_frame
        init["start_panorama_target_px"] = _label_pixels(pano, target)
        npx, angles, vframe = _best_view(pano, backend.robot_pose(), init["target_aabb"], target,
                                         scenario.robot.image_width, scenario.robot.image_height)
        init["start_best_view_target_px"] = npx
        init["start_best_view_angles"] = angles
        if vframe is not None:
            from PIL import Image
            Image.fromarray(vframe.rgb).save(out / "start_best_view.png")
        (out / "start_panorama.png").write_bytes(
            __import__("base64").b64decode(obs["image_png_b64"]))
        level_s = meta.get("spatial_level")
        gates = {}
        if level_s == "S1":
            # equal-angle panorama count; the best-VIEW count is recorded only
            # (perspective stretching at image edges inflates it)
            gates["visible_at_start(pano>=20px)"] = init["start_panorama_target_px"] >= 20
            gates["same_room"] = init["target_room"] == init["start_room"]
        elif level_s == "S2":
            gates["invisible_at_start"] = init["start_panorama_target_px"] == 0 and npx == 0
            gates["same_room"] = init["target_room"] == init["start_room"]
            gates["container_closed"] = not backend.is_open(tp.receptacle)
        elif level_s == "S3":
            gates["invisible_at_start"] = init["start_panorama_target_px"] == 0 and npx == 0
            gates["other_room"] = init["target_room"] != init["start_room"]
        init["s_gates"] = gates
        init["s_gates_pass"] = all(gates.values()) if gates else None
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
        report["null_baseline"] = {"status": nb["status"],
                                   "ok": nb["status"] == "FAIL_FALSE_COMPLETION"}

        # ---------------------------------------------- 3. scripted oracle
        if not a.no_oracle:
            oracle = RouteOracleAgent(vs, room_map=rm)
            t1 = time.time()
            summ = run_visual_episode(vs, oracle, out / "route_oracle", save_images=True)
            gc = check_goal(vs.world_state, scenario, backend)
            report["scripted_execution"] = {
                "status": summ["status"], "success": summ["status"] == "SUCCESS",
                "run_valid": summ["run_valid"],
                "planning_steps_used": summ["planning_steps_used"],
                "max_planning_steps": summ["max_planning_steps"],
                "oracle_failure": oracle.failure, "oracle_log": oracle.log,
                "goal_check_final": gc.predicates, "error": summ["error"],
                "duration_s": round(time.time() - t1, 1)}
            if oracle.route_debug is not None:
                (out / "route_debug.json").write_text(json.dumps(oracle.route_debug) + "\n")
            # S2 reveal evidence: target pixels in the frame right after OPEN
            reveal = None
            ev_path = out / "route_oracle" / "events.jsonl"
            if ev_path.exists():
                for line in ev_path.read_text().splitlines():
                    r = json.loads(line)
                    if r.get("selected_entity") == tp.receptacle and r.get("feedback") == "EXECUTED":
                        reveal = (r.get("observed_evidence") or {}).get(target, 0)
                        break
            report["scripted_execution"]["target_px_after_open"] = reveal
        # ---------------------------------------------- 4. model policy (optional)
        if a.agent:
            from rummagebench.agents.policy_adapter import load_agent

            agent = load_agent(a.agent)
            summ = run_visual_episode(vs, agent, out / "policy", save_images=True,
                                      model_version=a.agent)
            report["policy_episode"] = {
                "agent": a.agent, "status": summ["status"],
                "success": summ["status"] == "SUCCESS", "run_valid": summ["run_valid"],
                "planning_steps_used": summ["planning_steps_used"], "error": summ["error"]}
            report["model_used"] = True
    except BaseException as error:  # noqa: BLE001 - report everything
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["total_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report.get(k) for k in
                      ("task_initialization", "null_baseline", "scripted_execution", "policy_episode", "error")},
                     indent=1, default=str)[:6000], flush=True)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Kit teardown can hang; the report is already written
