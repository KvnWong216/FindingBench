"""vla_difficulty_v2: private doorway witnesses for a clearance manifest (GPU).

Lenient sibling of scripts/environment/check_clearance.py (unchanged): every
widened opening is tried from both sides with the unchanged public MOVE skill
(2 x 65 cm straight across the wall plane), each opening is recorded
independently and a rejection does not abort the rest. Turning inside the
opening is NOT required (no task route needs it); it is attempted once in the
middle and recorded for information only.

The robot is placed at the approach pose by evaluator-private teleport, never
exposed to a player. Reset is verified at the end.

    python scripts/v2/replay_doorways.py --scenario build/vla_difficulty_v2/_survey/<scene>.yaml \
        --output runs/vla_difficulty_v2/_clearance/<scene>_replay
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np

APPROACH = 0.65   # robot centre to wall plane before / after crossing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Straight public-MOVE crossing of each widened opening; not task solvability",
              "model_used": False, "supplied_to_player": False, "portals": []}
    try:
        from rummagebench.authoring.tasks.navigation import FootprintChecker
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.environment.env import InteractiveSearchEnv
        e = InteractiveSearchEnv(a.scenario, mode="agent", trace_path=out / "private_trace.jsonl")
        e.reset()
        b = e._backend
        start = b.robot_pose()
        doc = json.loads(Path(e._scenario.scene.clearance_manifest).read_text())
        report["manifest"] = e._scenario.scene.clearance_manifest
        checker = FootprintChecker.from_backend(b, .4, .03)
        for w in doc["wall_warps"]:
            axis, normal = w["axis"], 1 - w["axis"]
            lo, hi = w["target"][1:3]
            center = [0.0, 0.0, start[0][2]]
            center[axis], center[normal] = (lo + hi) / 2, w["wall_plane"]
            item = {"entity": w["entity"], "center": center[:2], "width_m": hi - lo, "sides": []}
            for sign in (-1, 1):
                pose = list(center)
                pose[normal] += sign * APPROACH
                d = [0.0, 0.0]
                d[normal] = -sign
                yaw = math.atan2(d[1], d[0])
                side = {"approach": pose[:2], "yaw": yaw}
                if not checker.free(pose[0], pose[1], yaw):
                    side["result"] = "approach_pose_blocked_by_furniture"
                    item["sides"].append(side)
                    continue
                b.teleport_robot(AnchorSpec(position=pose, orientation=[0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]))
                def act(action):
                    code = e.step(action)["feedback"]["code"]
                    return str(getattr(code, "value", code))
                moves = [act({"skill": "MOVE", "distance_cm": 65})]
                if moves[0] == "EXECUTED":
                    # in-opening turn is informative only; undone only if it executed
                    side["turn_in_opening"] = act({"skill": "TURN", "angle_deg": 90})
                    if side["turn_in_opening"] == "EXECUTED":
                        act({"skill": "TURN", "angle_deg": -90})
                    moves.append(act({"skill": "MOVE", "distance_cm": 65}))
                side["moves"] = moves
                side["crossed"] = moves == ["EXECUTED", "EXECUTED"]
                end = b.robot_pose()[0]
                side["end"] = [round(float(v), 3) for v in end[:2]]
                item["sides"].append(side)
                e.reset()
            item["ok"] = any(s.get("crossed") for s in item["sides"])
            report["portals"].append(item)
            print(json.dumps({k: item[k] for k in ("entity", "center", "ok")}), flush=True)
        e.reset()
        report["reset_verified"] = bool(np.allclose(b.robot_pose()[0], start[0], atol=1e-4))
        report["crossed"] = sum(x["ok"] for x in report["portals"])
        report["ok"] = report["reset_verified"]
    except Exception as error:  # noqa: BLE001
        report["ok"] = False
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report.get(k) for k in ("ok", "crossed", "error")}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    os._exit(code)
