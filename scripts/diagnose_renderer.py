"""Standalone synthetic renderer diagnostic; never an AGENT acceptance result."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", default="rgb,depth_linear,seg_instance")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--scene-template", action="store_true")
    parser.add_argument("--task-object-count", type=int, default=0)
    parser.add_argument("--task-object-offset", type=int, default=0)
    args = parser.parse_args()
    import omnigibson as og
    import numpy as np
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    config = {
        "scene": {"type": "Scene"},
        "robots": [],
        "objects": [{"type": "PrimitiveObject", "name": "diagnostic_cube",
                     "primitive_type": "Cube", "size": 0.5, "fixed_base": False,
                     "position": [0.0, 0.0, 1.0]}],
        "env": {"use_external_obs": True, "external_sensors": [
            {"sensor_type": "VisionSensor", "name": "diagnostic_camera",
             "modalities": args.modalities.split(","),
             "sensor_kwargs": {"image_width": args.resolution, "image_height": args.resolution}, "position": [0.0, 0.0, 3.0],
             "orientation": [0.0, 0.0, 0.0, 1.0]}]},
        "task": {"type": "DummyTask"},
    }

    if args.task_object_count:
        from rummagebench.core.scenario import load_scenario
        from rummagebench.sim.omnigibson.dataset import pick_model_for_category
        source = load_scenario(Path(__file__).resolve().parents[1] /
                               "scenarios/knife_search_001/scenario.yaml")
        selected = source.objects[args.task_object_offset:
                                  args.task_object_offset + args.task_object_count]
        for i, spec in enumerate(selected):
            config["objects"].append({
                "type": "DatasetObject", "name": spec.name,
                "category": spec.category,
                "model": spec.model or pick_model_for_category(spec.category),
                "fixed_base": spec.fixed_base,
                "position": [(i % 4 - 1.5) * 0.6, (i // 4 - 1) * 0.6, 1.0]})
        print("[diagnostic] task asset count", len(selected), flush=True)
    if args.scene_template:
        from rummagebench.sim.omnigibson.env_factory import load_env_config
        config["scene"] = load_env_config()["scene"]
    if args.robot:
        from rummagebench.sim.omnigibson.env_factory import load_robot_config
        robot = load_robot_config("r1pro")
        robot.update({"model": "r1pro", "name": "diagnostic_robot",
                      "obs_modalities": args.modalities.split(",")})
        robot["sensor_config"]["VisionSensor"]["sensor_kwargs"].update(
            {"image_height": args.resolution, "image_width": args.resolution})
        config["robots"] = [robot]
        config["env"] = {"use_external_obs": False}
    print("[diagnostic] building synthetic scene", flush=True)
    try:
        env = og.Environment(configs=config)
        print("[diagnostic] environment ready", flush=True)
        rows = []
        for i in range(3):
            og.sim.step()
            obs, info = env.get_obs()
            if args.robot:
                data = next(v for v in obs[0]["diagnostic_robot"].values()
                            if isinstance(v, dict) and "rgb" in v)
            else:
                data = obs[0]["external"]["diagnostic_camera"]
            rows.append({k: {"shape": list(array(v).shape),
                              "finite": int(np.isfinite(array(v)).sum()),
                              "nonzero": int(np.count_nonzero(array(v)))}
                         for k, v in data.items()})
            print("[diagnostic] captured", i, rows[-1], flush=True)
        Path(args.out).write_text(json.dumps({"kind": "synthetic_diagnostic", "frames": rows, "completed": True}))
        print("[diagnostic] completed", flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        Path(args.out).write_text(json.dumps({"kind": "synthetic_diagnostic",
                                             "error": repr(error), "ok": False}))
    finally:
        og.shutdown()

if __name__ == "__main__":
    main()
