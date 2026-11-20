#!/usr/bin/env python3
"""Phase 1 environment verification.

Verifies BEHAVIOR-1K version, OmniGibson import, dataset availability,
one InteractiveTraversableScene load, robot load, and RGB observation.
Writes runs/setup_report.json. Do not proceed to authoring until this passes.

    python scripts/verify_setup.py [--scene Beechwood_0_int]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPORT_PATH = Path("runs/setup_report.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Beechwood_0_int")
    args = parser.parse_args()

    report = {"timestamp": time.time(), "checks": {}}

    # 1. BEHAVIOR-1K / OmniGibson version
    import os

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")

    import omnigibson as og

    report["checks"]["omnigibson_version"] = og.__version__
    report["checks"]["omnigibson_import"] = True

    from omnigibson.macros import gm

    report["checks"]["data_path"] = str(gm.DATA_PATH)
    report["checks"]["headless"] = gm.HEADLESS

    # 2. dataset availability
    from omnigibson.utils.asset_utils import (
        get_available_behavior_1k_scenes,
        get_dataset_path,
    )

    scenes = get_available_behavior_1k_scenes()
    report["checks"]["dataset_available"] = len(scenes) > 0
    report["checks"]["num_scenes"] = len(scenes)
    report["checks"]["scene_requested_available"] = args.scene in scenes

    # 3. load one InteractiveTraversableScene with a robot and capture RGB
    from omnigibson.envs import Environment

    t0 = time.time()
    env = Environment(
        configs={
            "env": {"action_frequency": 30, "physics_frequency": 120},
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": args.scene,
                "include_robots": False,
            },
            "robots": [
                {
                    "model": "r1pro",
                    "name": "robot_0",
                    "obs_modalities": ["rgb"],
                    "action_type": "continuous",
                    "grasping_mode": "sticky",
                    "sensor_config": {
                        "VisionSensor": {"sensor_kwargs": {"image_height": 224, "image_width": 224}}
                    },
                }
            ],
            "objects": [],
            "task": {"type": "DummyTask"},
        }
    )
    report["checks"]["scene_load_seconds"] = round(time.time() - t0, 1)
    report["checks"]["scene_loaded"] = True

    scene = env.scene
    robot = scene.robots[0]
    report["checks"]["robot_loaded"] = robot.name
    report["checks"]["num_scene_objects"] = len(scene.objects)

    obs_list, _ = env.get_obs()
    obs = obs_list[0]
    rgb = None
    for sensor_name, sensor_obs in obs[robot.name].items():
        if isinstance(sensor_obs, dict) and "rgb" in sensor_obs:
            rgb = sensor_obs["rgb"]
            report["checks"]["rgb_sensor_name"] = sensor_name
            break
    report["checks"]["rgb_works"] = rgb is not None
    if rgb is not None:
        rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else rgb
        report["checks"]["rgb_shape"] = list(rgb.shape)

    # room types available in this scene (useful for Phase 2)
    seg_map = scene.seg_map
    report["checks"]["room_types"] = sorted(seg_map.room_sem_name_to_ins_name.keys())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    ok = (
        report["checks"]["omnigibson_import"]
        and report["checks"]["dataset_available"]
        and report["checks"]["scene_loaded"]
        and report["checks"]["robot_loaded"]
        and report["checks"]["rgb_works"]
    )
    print(f"\nSETUP {'OK' if ok else 'FAILED'} -> {REPORT_PATH}")
    # Kit teardown can segfault after a fully successful run; exit cleanly.
    import os

    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
