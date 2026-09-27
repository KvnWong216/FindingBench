#!/usr/bin/env python
"""Export a robot's kinematic model (URDF) from the OmniGibson articulation.

Run on the simulator host (behavior conda env; needs omnigibson + pin):

    python scripts/export_robot_kinematics.py --scenario scenarios/knife_search_001/scenario.yaml
    python scripts/export_robot_kinematics.py --robot r1pro --scene Beechwood_0_int \
        --out assets/robots/r1pro/r1pro.urdf

The URDF is derived from the simulator asset (link frames, joint tree,
limits, collider-bound boxes) — nothing is hand-authored. The exporter
cross-validates rest-pose FK against the simulator and refuses to write a
wrong model. After export, the scenario's robot.kinematics.urdf_path
resolves and the pinocchio feasibility backend becomes usable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=str, help="scenario YAML whose robot to export")
    parser.add_argument("--robot", type=str, default="r1pro", help="robot model name")
    parser.add_argument("--scene", type=str, default="Beechwood_0_int")
    parser.add_argument(
        "--out", type=str, default=None,
        help="output URDF path (default: build/robots/<robot>.urdf)",
    )
    args = parser.parse_args()

    out = args.out or str(REPO_ROOT / "build" / "robots" / f"{args.robot}.urdf")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        from rummagebench.core.scenario import load_scenario

        scenario = load_scenario(REPO_ROOT / args.scenario)
        robot_model = scenario.robot.model
        scene_model = scenario.scene.model
        if scenario.robot.kinematics and scenario.robot.kinematics.urdf_path:
            from rummagebench.robots.model_loader import resolve_urdf_path

            try:
                existing = resolve_urdf_path(scenario.robot.kinematics.urdf_path)
                print(f"URDF already exists: {existing} (delete to re-export)")
                return 0
            except Exception:
                pass
    else:
        robot_model, scene_model = args.robot, args.scene

    # simulator import happens here so --help stays cheap
    os.environ.setdefault("OMNIGIBSON_DATA_PATH", "")
    from omnigibson.envs import Environment
    from omnigibson.scenes import InteractiveTraversableScene

    print(f"loading {robot_model} in {scene_model} (Kit launch takes minutes)...")
    env = Environment(
        configs={
            "scene": {"type": "InteractiveTraversableScene", "model": scene_model},
            "robots": [{"type": robot_model, "obs_modalities": []}],
        }
    )
    robot = env.scene.robots[0]

    from rummagebench.sim.omnigibson.robot_export import export_robot_urdf

    manifest = export_robot_urdf(robot, out_path)
    print(f"EXPORT OK: {out_path}")
    print(f"  links: {len(manifest['links'])}, joints: {len(manifest['joints'])}")
    print(f"  validation: {manifest['validation']}")
    print(
        "Next: set robot.kinematics.urdf_path to this file in the scenario YAML "
        "(pip install pin on this host for the feasibility backend)."
    )
    os._exit(0)  # Kit teardown segfaults; artifacts are already written


if __name__ == "__main__":
    sys.exit(main())
