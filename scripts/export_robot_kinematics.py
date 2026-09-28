#!/usr/bin/env python
"""Export a robot's kinematic model (URDF) from the OmniGibson articulation.

Run on the simulator host (behavior conda env; needs omnigibson + pin):

    # full export + §2 cross-validation (50 random configs, 1cm / 3deg gates)
    python scripts/export_robot_kinematics.py --scenario scenarios/knife_search_001/scenario.yaml

    # §3 chain inspection from an existing URDF (no Kit launch)
    python scripts/export_robot_kinematics.py --inspect \
        --urdf build/robots/r1pro.urdf --eef-link eef_link

    # restricted morphology variant: write selected distal joints as fixed
    python scripts/export_robot_kinematics.py --robot r1pro \
        --fix-joints joint_a,joint_b --out build/robots/r1pro_restricted.urdf

The URDF is derived from the simulator asset (link frames, joint tree,
limits, collider-bound boxes) — nothing is hand-authored. Cross-validation:
rest-pose link poses AND N>=50 random joint configurations are compared
between the simulator and Pinocchio FK; any sample beyond the tolerance
REJECTS the export and the benchmark must not run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def inspect_urdf(urdf: Path, eef_link: str | None, controlled: str | None) -> int:
    """§3: print base link / controlled chain / locked joints from a URDF."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    controlled_names = [model.names[j] for j in range(1, model.njoints)]
    eef_joint = None
    if eef_link:
        fid = model.getFrameId(eef_link, pin.FrameType.BODY)
        if fid >= model.nframes:
            print(f"EEF link {eef_link!r} NOT FOUND in {urdf}")
            return 1
        eef_joint = model.frames[fid].parentJoint

    print(f"URDF: {urdf}")
    print("Base link:", model.names[1] if model.njoints > 1 else "(none)")
    print("EEF link:", eef_link or "(not specified)")
    print("Controlled chain:")
    for name in controlled_names:
        marker = ""
        print(f"  {name}{marker}")
    locked = locked_joints_of_urdf(urdf)
    print("Locked (fixed joints in URDF):")
    for name in locked:
        print(f"  {name}")
    if eef_joint is not None and controlled is not None:
        keep = {n.strip() for n in controlled.split(",") if n.strip()}
        chain = set()
        j = eef_joint
        while j > 0:
            chain.add(model.names[j])
            j = model.parents[j]
        if not (chain & keep):
            print(
                f"FAIL LOUDLY: EEF {eef_link!r} is not under the controlled "
                f"chain {sorted(keep)}"
            )
            return 1
    print("INSPECT OK")
    return 0


def locked_joints_of_urdf(urdf: Path) -> list[str]:
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf).getroot()
    return [j.get("name") for j in root.findall("joint") if j.get("type") == "fixed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=str, help="scenario YAML whose robot to export")
    parser.add_argument("--robot", type=str, default="r1pro", help="robot model name")
    parser.add_argument("--scene", type=str, default="Beechwood_0_int")
    parser.add_argument(
        "--out", type=str, default=None,
        help="output URDF path (default: build/robots/<robot>.urdf)",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="print the kinematic chain of an existing URDF and exit (no sim)",
    )
    parser.add_argument("--urdf", type=str, help="URDF to inspect (--inspect mode)")
    parser.add_argument("--eef-link", type=str, default=None,
                        help="end-effector link (validation / inspect)")
    parser.add_argument(
        "--fix-joints", type=str, default=None,
        help="comma-separated joint names written as FIXED (restricted "
             "morphology variant, e.g. a wrist); derived from the same asset",
    )
    parser.add_argument("--num-samples", type=int, default=50,
                        help="§2 random-config validation samples (>=50 required)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="DEV ONLY: skip the random-config gate")
    args = parser.parse_args()

    if args.inspect:
        if not args.urdf:
            parser.error("--inspect requires --urdf")
        return inspect_urdf(Path(args.urdf), args.eef_link, None)

    if args.fix_joints and not args.out:
        out = str(REPO_ROOT / "build" / "robots" / f"{args.robot}_restricted.urdf")
    else:
        out = args.out or str(REPO_ROOT / "build" / "robots" / f"{args.robot}.urdf")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_samples < 50 and not args.skip_validation:
        parser.error("--num-samples must be >= 50 (§2) unless --skip-validation")

    if args.scenario:
        from rummagebench.core.scenario import load_scenario

        scenario = load_scenario(REPO_ROOT / args.scenario)
        robot_model = scenario.robot.model
        scene_model = scenario.scene.model
        if args.eef_link is None and scenario.robot.kinematics:
            args.eef_link = scenario.robot.kinematics.end_effector_link
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

    # simulator import happens here so --help / --inspect stay cheap
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

    manifest = export_robot_urdf(
        robot,
        out_path,
        eef_link=args.eef_link,
        validate_samples=0 if args.skip_validation else args.num_samples,
        fix_joints=[j.strip() for j in args.fix_joints.split(",")] if args.fix_joints else None,
    )

    print(f"EXPORT OK: {out_path}")
    print(f"  links: {len(manifest['links'])}, joints: {len(manifest['joints'])}")
    print(f"  rest-pose validation: {manifest['validation'].get('rest_pose_ok')}")
    if "random_config_validation" in manifest:
        v = manifest["random_config_validation"]
        print(
            f"  §2 random-config validation: n={v['num_samples']} "
            f"max_trans={v['max_translation_error_m']}m "
            f"max_rot={v['max_rotation_error_deg']}deg passed={v['passed']}"
        )
    print(
        "Next: set robot.kinematics.urdf_path to this file in the scenario YAML "
        "(pip install pin on this host for the feasibility backend)."
    )
    os._exit(0)  # Kit teardown segfaults; artifacts are already written


if __name__ == "__main__":
    sys.exit(main())
