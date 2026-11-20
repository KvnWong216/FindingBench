#!/usr/bin/env python3
"""Render an episode in the real environment with synchronized decision output.

For every semantic step the robot takes, this script:
  1. executes the action through the benchmark (validation pipeline included),
  2. immediately prints the decision to stdout (unbuffered, synchronous),
  3. captures the robot head-camera RGB and overlays the decision on the frame,
  4. appends the decision to <out>/decisions.jsonl.

Finally it stitches the annotated frames into an MP4 (ffmpeg if available)
and/or a GIF so the whole search can be replayed visually.

    python scripts/render_episode.py [--agent scripted] [--scenario knife_search_001]
        [--out runs/render] [--fps 2] [--scale 2]

GPU selection: set CUDA_VISIBLE_DEVICES before running.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def annotate(rgb: np.ndarray, lines: list[str], scale: int = 2) -> np.ndarray:
    """Overlay the decision text on a copy of the camera frame."""
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.BILINEAR
    )
    draw = ImageDraw.Draw(img)
    bar_h = 16 * len(lines) + 10
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((6, 6 + 16 * i), line, fill=(0, 255, 102))
    return np.asarray(img)


def stitch(frames: list[Path], out_dir: Path, fps: int) -> str:
    """MP4 via ffmpeg if present, else an animated GIF via PIL."""
    mp4 = out_dir / "episode.mp4"
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(out_dir / "frame_%04d.png"),
            "-pix_fmt", "yuv420p", "-r", str(fps), str(mp4),
        ]
        if subprocess.run(cmd).returncode == 0:
            return str(mp4)
    gif = out_dir / "episode.gif"
    imgs = [Image.open(f) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return str(gif)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="knife_search_001")
    parser.add_argument("--agent", default="scripted",
                        choices=["scripted", "wrong_object", "timeout", "unsafe"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --scenario accepts either a scenario id (scenarios/<id>/scenario.yaml)
    # or a direct path to any scenario YAML file
    arg = Path(args.scenario)
    if arg.suffix == ".yaml":
        scenario_path = arg if arg.is_absolute() else Path.cwd() / arg
    else:
        scenario_path = REPO / "scenarios" / args.scenario / "scenario.yaml"
    if not scenario_path.exists():
        available = sorted(
            d.name for d in (REPO / "scenarios").iterdir()
            if d.is_dir() and (d / "scenario.yaml").exists()
        )
        parser.error(
            f"scenario not found: {scenario_path}\n"
            f"available scenarios: {', '.join(available)}\n"
            f"(pass a scenario id, or a direct path to a scenario .yaml file)"
        )
    out_dir = Path(args.out) if args.out else REPO / "runs" / f"render_{args.scenario}_{args.agent}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from rummagebench.core.scenario import load_scenario
    from rummagebench.core.session import BenchmarkSession
    from rummagebench.core.types import EpisodeStatus
    from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    scenario = load_scenario(scenario_path)
    validate_agent_sequence(args.agent, scenario)
    agent = make_agent(args.agent, scenario)

    print(f"== rendering {args.scenario} / {args.agent} -> {out_dir}", flush=True)
    backend = OmniGibsonBackend(seed=args.seed)
    build_report = backend.setup(scenario)
    print(f"== build OK: {len(build_report['spawned'])} objects spawned", flush=True)

    decisions_path = out_dir / "decisions.jsonl"
    decisions_path.write_text("")  # truncate

    session = BenchmarkSession(backend, scenario, log_path=str(out_dir / "events.jsonl"))
    observation = session.reset()
    agent.reset(observation.instruction)

    print(f"== instruction: {observation.instruction.strip()}", flush=True)
    print(f"== init pose: living_room anchor, budget {observation.max_planning_steps} steps", flush=True)

    frame_paths: list[Path] = []
    frame_idx = 0

    def emit(text_lines: list[str], step: int, record: dict) -> None:
        nonlocal frame_idx
        rgb = backend.get_observation()
        annotated = annotate(rgb, text_lines, scale=args.scale)
        frame_path = out_dir / f"frame_{frame_idx:04d}.png"
        Image.fromarray(annotated).save(frame_path)
        frame_paths.append(frame_path)
        frame_idx += 1
        with decisions_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(" | ".join(text_lines), flush=True)

    # initial observation frame
    emit(
        [f"step 0: RESET", f"status: RUNNING", f"pos: {scenario.robot.init_anchor}"],
        0,
        {"step": 0, "action": None, "status": "RUNNING", "note": "initial observation"},
    )

    while session.status() == EpisodeStatus.RUNNING:
        action = agent.act(observation)
        t0 = time.time()
        result = session.act(action)
        observation = result.observation
        tgt = action.target.value or ""
        lines = [
            f"step {result.planning_step}: {action.skill}({tgt})",
            f"executed: {result.executed}  "
            f"postcond: {(observation.previous_action_result or {}).get('postcondition_satisfied')}",
            f"status: {result.episode_status.value}  "
            f"({round(time.time() - t0, 1)}s)",
        ]
        record = {
            "step": result.planning_step,
            "action": result.action,
            "executed": result.executed,
            "failure_reason": result.failure_reason.value,
            "episode_status": result.episode_status.value,
            "events": result.events,
        }
        emit(lines, result.planning_step, record)

    final = session.status().value
    print(f"== final status: {final} after {observation.planning_step} steps", flush=True)
    emit([f"FINAL: {final}", f"steps: {observation.planning_step}"],
         observation.planning_step,
         {"step": observation.planning_step + 1, "action": None,
          "status": final, "note": "final"})

    video = stitch(frame_paths, out_dir, fps=args.fps)
    print(f"== video: {video}", flush=True)
    print(f"== decisions: {decisions_path}", flush=True)

    import os

    os._exit(0)


if __name__ == "__main__":
    main()
