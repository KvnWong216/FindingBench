#!/usr/bin/env python3
"""Paired equivalence check: discrete mechanics vs. motion replay.

Runs the same scripted action sequence (``rab_taxonomy_probe.py``: refusals,
CHECK, GRASP, PLACE, REPORT_DONE, plus a premature-done episode) twice through
the real stdio MCP server - once with ``FINDINGBENCH_RAB_MOTION=discrete`` and
once with ``=replay`` - and asserts that the motion replay is causally inert:

* identical public feedback code / planning step / episode status per action;
* identical private ``private_error`` prototype and ``terminal_attribution``;
* restoration score equal within 1e-3;
* every replay segment restored the post-state within 5 mm;
* demo artifacts (mp4 or PNG frames, trajectory.jsonl, segments.jsonl) exist.

Serial by construction (one Isaac Sim worker on GPU 0 at a time).
Usage: python scripts/rab_motion_equivalence.py [--out runs/rab_motion_equivalence]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_probe(mode: str, out: Path) -> None:
    env = dict(os.environ, FINDINGBENCH_RAB_MOTION=mode, PYTHONPATH=f"{ROOT/'src'}:{ROOT/'scripts'}")
    subprocess.run(["nice", "-n", "19", sys.executable, str(ROOT / "scripts/rab_taxonomy_probe.py"),
                    "--out", str(out)], env=env, check=True, cwd=ROOT)


def private_records(out: Path) -> list[list[dict]]:
    episodes = []
    for actions in sorted((out / "sessions").glob("*/worker_trace/actions")):
        episodes.append([json.loads(p.read_text()) for p in sorted(actions.glob("*.json"))])
    return episodes


def head(err):
    return (err or "").split(":", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/rab_motion_equivalence")
    args = ap.parse_args()
    dirs = {m: args.out / m for m in ("discrete", "replay")}
    for mode, d in dirs.items():
        if d.exists():
            raise SystemExit(f"{d} exists; choose a fresh --out")
        run_probe(mode, d)

    problems: list[str] = []
    pub = {m: json.loads((d / "probe_public_log.json").read_text()) for m, d in dirs.items()}
    if [(a["code"], a["step"], a["status"]) for a in pub["discrete"]] != \
       [(a["code"], a["step"], a["status"]) for a in pub["replay"]]:
        problems.append("public feedback sequence differs")

    priv = {m: private_records(d) for m, d in dirs.items()}
    if len(priv["discrete"]) != len(priv["replay"]):
        problems.append("episode count differs")
    replay_segments = 0
    for ep_d, ep_r in zip(priv["discrete"], priv["replay"]):
        for i, (a, b) in enumerate(zip(ep_d, ep_r), 1):
            if head(a.get("private_error")) != head(b.get("private_error")):
                problems.append(f"step {i}: prototype {head(a.get('private_error'))} != {head(b.get('private_error'))}")
            ma, mb = a.get("measurements") or {}, b.get("measurements") or {}
            if ma.get("terminal_attribution") != mb.get("terminal_attribution"):
                problems.append(f"step {i}: attribution differs")
            ra, rb = a.get("restoration"), b.get("restoration")
            if ra is not None and rb is not None and abs(ra - rb) > 1e-3:
                problems.append(f"step {i}: restoration {ra:.4f} != {rb:.4f}")
            rep = mb.get("motion_replay")
            if rep:
                replay_segments += 1
                if rep.get("error") or rep.get("restore_error"):
                    problems.append(f"step {i}: replay error {rep.get('error') or rep.get('restore_error')}")
                if (rep.get("restore_max_err_m") or 0) > .005:
                    problems.append(f"step {i}: restore error {rep['restore_max_err_m']:.4f} m")
    if replay_segments == 0:
        problems.append("no motion_replay segments recorded")
    demos = list((dirs["replay"] / "sessions").glob("*/worker_trace/demo"))
    for demo in demos:
        if not ((demo / "trajectory.jsonl").is_file() and (demo / "segments.jsonl").is_file()):
            problems.append(f"{demo}: missing trajectory/segments")
        if not (list(demo.glob("episode_*.mp4")) or (demo / "frames").is_dir()):
            problems.append(f"{demo}: no video frames")
    if not demos:
        problems.append("no demo directories")

    report = dict(ok=not problems, problems=problems, replay_segments=replay_segments,
                  demos=[str(d) for d in demos])
    (args.out / "equivalence_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
