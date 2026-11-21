#!/usr/bin/env python3
"""Scripted probe that exercises the deterministic error-prototype taxonomy.

Drives the real stdio RAB MCP server (one Isaac Sim worker on GPU 0) through a
short scripted sequence on rab_F251 that is expected to trigger several
Layer-A prototypes and one Layer-B attribution per episode, then runs
``rab_eval_metrics.py`` on the produced private traces (after termination).

Episode 1 (taxonomy sweep):
    PLACE with empty hand      -> PRECONDITION_HAND_EMPTY
    GRASP on upper-frame pixel -> CLICK_NON_INTERACTABLE (or TOO_FAR, recorded as-is)
    VIEW / CHECK can           -> EXECUTED
    OPEN on the can            -> CLICK_NON_ARTICULATED
    GRASP with stale frame id  -> STALE_FRAME_ID
    GRASP can                  -> EXECUTED
    GRASP again                -> PRECONDITION_HAND_OCCUPIED
    PLACE / REPORT_DONE        -> terminal attribution
Episode 2: immediate REPORT_DONE -> PREMATURE_DONE
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _payload(resp):
    assert not resp.isError, resp
    return json.loads(next(c.text for c in resp.content if c.type == "text"))


async def run(args) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    sessions = Path(args.out).resolve() / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{root / 'src'}:{root / 'scripts'}", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="0", RUMMAGEBENCH_MCP_GPU="0", FINDINGBENCH_RAB_MCP_OUT=str(sessions))
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "rummagebench.cli", "serve-mcp", "--backend", "rab"],
                                   cwd=str(root), env=env)
    log: list[dict] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()

            async def act(tag, action):
                r = _payload(await s.call_tool("act", {"action": action}))
                log.append(dict(tag=tag, skill=action["skill"], code=r["feedback"]["code"],
                                step=r["planning_step"], status=r["episode_status"]))
                print(f"  {tag:<28} {action['skill']:<12} -> {r['feedback']['code']:<18} "
                      f"step={r['planning_step']} status={r['episode_status']}", flush=True)
                return r

            def pt(frame, x, y):
                return {"frame_id": frame, "x": x, "y": y}

            print("[episode 1] taxonomy sweep", flush=True)
            res = _payload(await s.call_tool("reset_episode", {"scenario_id": args.scenario}))
            f = res["reset"]["frame_id"]
            r = await act("place_empty_hand", {"skill": "PLACE", "point": pt(f, .51, .65)}); f = r["observation"]["frame_id"]
            r = await act("grasp_upper_frame", {"skill": "GRASP", "point": pt(f, .50, .04)}); f = r["observation"]["frame_id"]
            r = await act("view_down", {"skill": "VIEW", "yaw": -27.5, "pitch": -60.0, "zoom": 1.0}); f = r["observation"]["frame_id"]
            r = await act("check_can", {"skill": "CHECK", "point": pt(f, .42, .48)}); f = r["observation"]["frame_id"]
            r = await act("open_non_articulated", {"skill": "OPEN", "point": pt(f, .42, .48)}); f_prev, f = f, r["observation"]["frame_id"]
            r = await act("grasp_stale_frame", {"skill": "GRASP", "point": pt(f_prev, .42, .48)}); f = r["observation"]["frame_id"]
            r = await act("grasp_can", {"skill": "GRASP", "point": pt(f, .42, .48)}); f = r["observation"]["frame_id"]
            r = await act("grasp_hand_occupied", {"skill": "GRASP", "point": pt(f, .42, .48)}); f = r["observation"]["frame_id"]
            r = await act("place_can", {"skill": "PLACE", "point": pt(f, .51, .65)}); f = r["observation"]["frame_id"]
            await act("report_done", {"skill": "REPORT_DONE"})

            print("[episode 2] premature REPORT_DONE", flush=True)
            _payload(await s.call_tool("reset_episode", {"scenario_id": args.scenario}))
            await act("premature_done", {"skill": "REPORT_DONE"})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="rab_F251")
    ap.add_argument("--out", default="runs/rab_taxonomy_probe_20261121")
    args = ap.parse_args()
    log = asyncio.run(run(args))
    out = Path(args.out).resolve()
    (out / "probe_public_log.json").write_text(json.dumps(log, indent=2) + "\n")
    # Episodes are terminated now -> evaluator-private traces may be read.
    subprocess.run([sys.executable, str(Path(__file__).with_name("rab_eval_metrics.py")),
                    str(out / "sessions"), "--label", "taxonomy_probe", "--out", str(out / "metrics")],
                   check=True)


if __name__ == "__main__":
    main()
