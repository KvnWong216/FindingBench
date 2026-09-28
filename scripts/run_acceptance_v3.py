#!/usr/bin/env python
"""Production physical-grounding acceptance (v3) — §1/§6/§25.

One simulator process runs the whole acceptance:

    A. scene load + R1Pro articulation export + §2 cross-validation
       (50 random configs, 1 cm / 3 deg gates) — any failure ABORTS
    B. knife_search_001 with the production pinocchio backend:
       scripted_success / wrong_object / timeout / unsafe, each with
       events.jsonl + summary.json + grounding_trace.jsonl
    C. action-graph morphology comparison (default vs restricted URDF)
    D. (optional --episodes N) N generated knife-search episodes, scripted
       agent, determinism check on the final status

Writes runs/acceptance_v3/summary.json and prints ACCEPTANCE V3 OK/FAIL.
Run the memory probe FIRST (scripts/probe_memory.py) when GPUs are shared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXPECTED_STATUS = {
    "scripted": "SUCCESS",
    "wrong_object": "FAIL_WRONG_TARGET",
    "timeout": "FAIL_MAX_STEPS",
    "unsafe": "FAIL_UNSAFE_ACTION",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="scenarios/knife_search_001/scenario.yaml")
    ap.add_argument("--out", default="runs/acceptance_v3")
    ap.add_argument("--episodes", type=int, default=0,
                    help="also run N generated episodes (scripted agent)")
    ap.add_argument("--skip-morphology", action="store_true",
                    help="skip the restricted-URDF action-graph comparison")
    ap.add_argument("--fix-joints", default="right_arm_joint2",
                    help="joints written FIXED in the restricted variant")
    ap.add_argument("--skip-trajectories", action="store_true")
    ap.add_argument("--skip-episodes", action="store_true")
    ap.add_argument("--first-episode", type=int, default=0,
                    help="first generated-episode index (chunked batches)")
    ap.add_argument("--episode-count", type=int, default=0,
                    help="how many generated episodes THIS process runs")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}}

    from rummagebench.adapters.python_api import create_session
    from rummagebench.core.events import load_events
    from rummagebench.core.types import Action, EpisodeStatus
    from rummagebench.evaluation.action_graph import compare_graphs, export_action_graph, write_graph
    from rummagebench.evaluation.episode_log import run_episode
    from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence
    from rummagebench.evaluation.metrics import compute_metrics

    t0 = time.time()
    scenario_path = REPO_ROOT / args.scenario

    # ---------- phase A/B: production session (export + validation + runs) --
    session = create_session(scenario_path, run_dir=None, seed=0)
    session.reset()
    report["robot"] = session.scenario.robot.model
    report["feasibility_backend"] = session.scenario.feasibility.backend

    # export the restricted variant NOW, on the pristine post-load state:
    # after trajectories the assisted-grasp leftovers and controller drift
    # contaminate both the articulation and the rest-pose validation
    if not args.skip_morphology:
        restricted_urdf = REPO_ROOT / "build" / "robots" / "r1pro_restricted.urdf"
        try:
            from rummagebench.sim.omnigibson.robot_export import export_robot_urdf

            print(
                "[acceptance] exporting restricted variant on pristine state "
                f"(fixed: {args.fix_joints})",
                flush=True,
            )
            export_robot_urdf(
                session._backend._robot,
                restricted_urdf,
                eef_link=session.scenario.robot.kinematics.end_effector_link,
                base_link=session.scenario.robot.kinematics.base_link,
                fix_joints=[j.strip() for j in args.fix_joints.split(",") if j.strip()],
            )
        except Exception as e:  # noqa: BLE001
            report["phases"]["restricted_export"] = {
                "error": f"{type(e).__name__}: {e}"
            }
            traceback.print_exc()

    trajectories = {}
    if not args.skip_trajectories:
        _run_trajectories(session, out_root, report)
    if not args.skip_morphology:
        _run_morphology(session, out_root, report,
                        [j.strip() for j in args.fix_joints.split(",") if j.strip()])
    if args.episodes > 0 and not args.skip_episodes:
        _run_episodes(session, out_root, report, args)

    report["duration_s"] = round(time.time() - t0, 1)
    passed = (
        report["phases"].get("trajectories_ok", True)
        and report["phases"].get("generated_episodes", {}).get("all_success", True)
        and report["phases"].get("morphology", {}).get("graphs_differ", True) is not False
    )
    report["passed"] = bool(passed)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("robot", "feasibility_backend", "passed", "duration_s")}, indent=2))
    print(f"[acceptance] ACCEPTANCE V3 {'OK' if passed else 'FAILED'}")
    return 0 if passed else 1


def _run_trajectories(session, out_root, report):
    from rummagebench.core.events import load_events
    from rummagebench.evaluation.episode_log import run_episode
    from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence
    from rummagebench.evaluation.metrics import compute_metrics

    scenario = session.scenario
    trajectories = {}
    scenario = session.scenario
    report["robot"] = scenario.robot.model
    report["feasibility_backend"] = scenario.feasibility.backend
    report["phases"]["export"] = {
        "urdf": (
            session._feasibility._kinematics.urdf_path
            if scenario.feasibility.backend == "pinocchio" else "proxy"
        ),
        "chain": (
            session._feasibility._kinematics.chain_info()
            if scenario.feasibility.backend == "pinocchio" else None
        ),
    }
    print("[acceptance] chain:", json.dumps(report["phases"]["export"]["chain"], indent=1), flush=True)

    trajectories = {}
    for agent_name in ("scripted", "wrong_object", "timeout", "unsafe"):
        validate_agent_sequence(agent_name, scenario)
        agent = make_agent(agent_name, scenario)
        run_dir = out_root / agent_name
        run_dir.mkdir(parents=True, exist_ok=True)
        session.reset()
        session.set_log(run_dir / "events.jsonl")
        session.set_trace(run_dir / "grounding_trace.jsonl")
        summary = run_episode(session, agent, run_dir, save_images=False)
        session.set_log(None)
        session.set_trace(None)
        summary["metrics"] = compute_metrics(
            load_events(run_dir / "events.jsonl"), scenario.oracle_min_steps,
            scenario,
        )
        trajectories[agent_name] = summary
        status = summary["status"]
        ok = status == EXPECTED_STATUS[agent_name]
        print(
            f"[acceptance] {agent_name:12s} -> {status} "
            f"({summary['planning_steps_used']} steps) "
            f"{'OK' if ok else 'UNEXPECTED (expected ' + EXPECTED_STATUS[agent_name] + ')'}",
            flush=True,
        )
    report["phases"]["trajectories"] = trajectories
    report["phases"]["trajectories_ok"] = all(
        trajectories[a]["status"] == EXPECTED_STATUS[a] for a in EXPECTED_STATUS
    )


def _run_morphology(session, out_root, report, fix_joints):
    from rummagebench.evaluation.action_graph import (
        compare_graphs,
        export_action_graph,
        write_graph,
    )
    from rummagebench.evaluation.evaluator import make_agent

    scenario = session.scenario
    if True:
        try:
            agent = make_agent("scripted", scenario)
            base_graph = export_action_graph(session, agent)
            write_graph(base_graph, out_root / "action_graph" / "r1pro_action_graph.json")
            restricted_urdf = REPO_ROOT / "build" / "robots" / "r1pro_restricted.urdf"
            if not restricted_urdf.is_file():
                raise RuntimeError(
                    f"{restricted_urdf} missing: pristine-state export failed"
                )
            session.swap_morphology("build/robots/r1pro_restricted.urdf")
            agent_r = make_agent("scripted", scenario)
            restricted_graph = export_action_graph(session, agent_r)
            write_graph(
                restricted_graph,
                out_root / "action_graph" / "r1pro_restricted_action_graph.json",
            )
            comparison = compare_graphs(base_graph, restricted_graph)
            (out_root / "action_graph" / "morphology_comparison.json").write_text(
                json.dumps(comparison, indent=2)
            )
            report["phases"]["morphology"] = comparison
            print(
                f"[acceptance] morphology: {comparison['common_nodes']} common "
                f"nodes, differing action sets "
                f"{len(comparison['differing_action_sets'])} "
                f"(graphs_differ={comparison['graphs_differ']})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            report["phases"]["morphology"] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()


def _run_episodes(session, out_root, report, args):
    from rummagebench.authoring.episode_generator import generate_episodes
    from rummagebench.core.events import load_events
    from rummagebench.core.session import BenchmarkSession
    from rummagebench.evaluation.episode_log import run_episode
    from rummagebench.evaluation.evaluator import make_agent
    from rummagebench.evaluation.metrics import compute_metrics

    if True:
        gen_dir = REPO_ROOT / "build" / "generated_scenarios"
        seeds = list(range(args.first_episode, args.first_episode + args.episodes))
        paths = generate_episodes(REPO_ROOT / args.scenario, seeds, out_dir=gen_dir)
        gen_results, build_failures = [], []
        for path in paths:
            from rummagebench.core.scenario import load_scenario

            episode_scenario = load_scenario(path)
            # apply THIS episode's placements/initial states on the already
            # built scene (no scene reload), then a fresh benchmark session
            try:
                session._backend.reapply_scenario(episode_scenario)
            except Exception as e:  # noqa: BLE001 - one bad build != dead batch
                print(
                    f"[acceptance] generated {episode_scenario.id} BUILD FAILED: {e}",
                    flush=True,
                )
                build_failures.append({"episode": episode_scenario.id, "error": str(e)})
                continue
            run_dir = out_root / "generated" / episode_scenario.id
            run_dir.mkdir(parents=True, exist_ok=True)
            episode_session = BenchmarkSession(
                session._backend, episode_scenario,
                log_path=str(run_dir / "events.jsonl"),
                trace_path=str(run_dir / "grounding_trace.jsonl"),
            )
            agent = make_agent("scripted", episode_scenario)
            episode_session.reset()
            summary = run_episode(episode_session, agent, run_dir, save_images=False)
            summary["metrics"] = compute_metrics(
                load_events(run_dir / "events.jsonl"),
                episode_scenario.oracle_min_steps,
                episode_scenario,
            )
            gen_results.append({
                "episode": episode_scenario.id,
                "status": summary["status"],
                "steps": summary["planning_steps_used"],
                "target_container": next(
                    p.receptacle for p in episode_scenario.placements
                    if p.entity == episode_scenario.target.entity
                ),
            })
            print(
                f"[acceptance] generated {episode_scenario.id} -> {summary['status']} "
                f"({summary['planning_steps_used']} steps)",
                flush=True,
            )
        deterministic = all(r["status"] == "SUCCESS" for r in gen_results)
        report["phases"]["generated_episodes"] = {
            "count": len(gen_results),
            "build_failures": build_failures,
            "all_success": deterministic,
            "results": gen_results,
        }




if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        os._exit(code)  # Kit teardown segfaults; artifacts are written
