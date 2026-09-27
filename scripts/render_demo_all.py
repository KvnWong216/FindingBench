#!/usr/bin/env python3
"""Render all four demo episodes in ONE process (pays Kit startup once).

HumanCLAW-style demo assets: for every agent, an annotated egocentric MP4
(decision overlay per step) plus decisions.jsonl with the full step transcript
(action, executed, failure_reason, available_skills, status).

    CUDA_VISIBLE_DEVICES=5 python scripts/render_demo_all.py [--fps 2] [--scale 2]

Outputs under runs/demo_<agent>/ and a rollup runs/demo_index.json.
Exits via os._exit (Kit teardown segfault is benign).
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from render_episode import annotate, stitch  # noqa: E402

EXPECTED = {
    "scripted": "SUCCESS",
    "wrong_object": "FAIL_WRONG_TARGET",
    "timeout": "FAIL_MAX_STEPS",
    "unsafe": "FAIL_UNSAFE_ACTION",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="knife_search_001")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from rummagebench.core.scenario import load_scenario
    from rummagebench.core.session import BenchmarkSession
    from rummagebench.core.types import EpisodeStatus
    from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    scenario_path = REPO / "scenarios" / args.scenario / "scenario.yaml"
    scenario = load_scenario(scenario_path)

    print("== building scene (shared across all agents)", flush=True)
    backend = OmniGibsonBackend(seed=args.seed)
    build_report = backend.setup(scenario)
    print(f"== build OK: {len(build_report['spawned'])} objects spawned", flush=True)

    index: dict = {"scenario": args.scenario, "instruction": None, "episodes": {}}

    for agent_name, want in EXPECTED.items():
        out_dir = REPO / "runs" / f"demo_{args.scenario}_{agent_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        validate_agent_sequence(agent_name, scenario)
        agent = make_agent(agent_name, scenario)

        print(f"== rendering {args.scenario} / {agent_name} -> {out_dir}", flush=True)
        decisions_path = out_dir / "decisions.jsonl"
        decisions_path.write_text("")
        session = BenchmarkSession(backend, scenario, log_path=str(out_dir / "events.jsonl"))
        observation = session.reset()
        agent.reset(observation.instruction)
        index["instruction"] = observation.instruction.strip()

        frames: list[Path] = []
        frame_idx = 0

        def emit(lines: list[str], record: dict) -> None:
            nonlocal frame_idx
            rgb = backend.get_observation()
            annotated = annotate(rgb, lines, scale=args.scale)
            frame_path = out_dir / f"frame_{frame_idx:04d}.png"
            Image.fromarray(annotated).save(frame_path)
            frames.append(frame_path)
            frame_idx += 1
            with decisions_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            print(" | ".join(lines), flush=True)

        emit(
            ["step 0: RESET", "status: RUNNING", f"pos: {scenario.robot.init_anchor}"],
            {
                "step": 0,
                "action": None,
                "executed": None,
                "failure_reason": None,
                "episode_status": "RUNNING",
                "available_skills": observation.available_skills,
                "note": "initial observation",
            },
        )

        while session.status() == EpisodeStatus.RUNNING:
            action = agent.act(observation)
            t0 = time.time()
            result = session.act(action)
            observation = result.observation
            tgt = action.target.value or ""
            prev = observation.previous_action_result or {}
            emit(
                [
                    f"step {result.planning_step}: {action.skill}({tgt})",
                    f"executed: {result.executed}  postcond: {prev.get('postcondition_satisfied')}",
                    f"status: {result.episode_status.value}  ({round(time.time() - t0, 1)}s)",
                ],
                {
                    "step": result.planning_step,
                    "action": result.action,
                    "executed": result.executed,
                    "failure_reason": result.failure_reason.value,
                    "episode_status": result.episode_status.value,
                    "available_skills": observation.available_skills,
                    "events": result.events,
                },
            )

        final = session.status().value
        emit(
            [f"FINAL: {final}", f"steps: {observation.planning_step}"],
            {
                "step": observation.planning_step + 1,
                "action": None,
                "executed": None,
                "failure_reason": None,
                "episode_status": final,
                "available_skills": observation.available_skills,
                "note": "final",
            },
        )

        video = stitch(frames, out_dir, fps=args.fps)
        status_ok = final == want
        index["episodes"][agent_name] = {
            "status": final,
            "expected": want,
            "match": status_ok,
            "steps": observation.planning_step,
            "video": video,
            "decisions": str(decisions_path),
            "events": str(out_dir / "events.jsonl"),
            "frames_dir": str(out_dir),
        }
        print(f"== {agent_name}: {final} (expected {want}) video={video}", flush=True)

    (REPO / "runs" / "demo_index.json").write_text(json.dumps(index, indent=2))
    print("== demo index: runs/demo_index.json", flush=True)

    import os

    os._exit(0)


if __name__ == "__main__":
    main()
