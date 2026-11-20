#!/usr/bin/env python
"""Does GRASP at a given base pose lift the robot base? (2026-10-19)

One scene load. For each --forward value (0 = object origin on the EEF, the
legacy realization; -1 = auto: ahead of the fingertips by half the object's
largest extent + 2 cm; >0 = metres) : reset, teleport the base to --pose,
symbolic_grasp(target), settle, then teleport the base back by --back m and
settle again (what MOVE does), recording base z, held-object AABB and the
robot links / scene entities whose AABB overlaps it.

    python scripts/probe_grasp_lift.py <scenario> --pose -1.331 -6.754 -1.745 --forward 0 -1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path


def _box(b):
    return [[round(float(v), 3) for v in b[0]], [round(float(v), 3) for v in b[1]]]


def _overlap(a, b, pad=0.0):
    return all(float(a[0][i]) - pad < float(b[1][i]) and float(b[0][i]) - pad < float(a[1][i]) for i in range(3))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scenario", type=Path)
    p.add_argument("--pose", nargs=3, type=float, required=True)
    p.add_argument("--forward", nargs="+", type=float, default=[0.0, -1.0])
    p.add_argument("--back", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("runs/vla_difficulty_v1/_diagnose/grasp_lift.json"))
    a = p.parse_args()
    rep: dict = {"scenario": str(a.scenario), "pose": a.pose, "trials": []}
    try:
        import numpy as np
        from rummagebench.core.scenario import AnchorSpec, load_scenario
        from rummagebench.environment.env import InteractiveSearchEnv
        from rummagebench.skills.move import quat_from_yaw

        scenario = load_scenario(a.scenario)
        target = scenario.target.entity
        env = InteractiveSearchEnv(a.scenario, mode="agent")
        b, vs = env._backend, env._session
        x, y, yaw = a.pose
        for fwd in a.forward:
            vs.reset()
            z0 = float(b.robot_pose()[0][2])
            b.teleport_robot(AnchorSpec(position=[x, y, z0], orientation=quat_from_yaw(yaw)))
            b.settle(5)
            b.held_forward_m = fwd
            ok = b.symbolic_grasp(target)
            b.settle(30)
            row = {"forward": fwd, "grasped": bool(ok), "z_after_grasp": round(float(b.robot_pose()[0][2]), 4)}
            held = b.entity_aabb(target)
            row["held_aabb"] = _box(held)
            row["robot_links_overlapping"] = sorted(
                n for n, l in (b._robot.links or {}).items()
                if _overlap(held, [np.asarray(v.detach().cpu() if hasattr(v, "detach") else v) for v in l.aabb]))
            row["entities_overlapping"] = sorted(
                n for n in b.entity_names()
                if n != target and n not in b.robot_entity_names()
                and (bb := b.entity_aabb(n)) is not None and _overlap(held, bb))
            bx, by = x - a.back * math.cos(yaw), y - a.back * math.sin(yaw)
            b.teleport_robot(AnchorSpec(position=[bx, by, z0], orientation=quat_from_yaw(yaw)))
            b.settle(30)
            row["z_after_move_back"] = round(float(b.robot_pose()[0][2]), 4)
            row["still_holding"] = bool(b.is_holding(target))
            off = b.eef_pose().inverse().compose(b.entity_pose6d(target))
            row["held_offset_pos"] = [round(float(v), 3) for v in off.position]
            rep["trials"].append(row)
            print(json.dumps(row), flush=True)
            b.held_forward_m = -1.0  # back to the backend default (auto)
        rep["ok"] = True
    except BaseException as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"ok": rep.get("ok"), "error": rep.get("error"), "out": str(a.out)}), flush=True)
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
