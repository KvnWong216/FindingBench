"""Independent synthetic camera calibration; does not use task geometry."""
import argparse
import json
import math
from pathlib import Path
import traceback

import numpy as np

from rummagebench.perception.camera_geometry import (
    to_world, unproject_range, unproject_z_depth, world_from_usd_camera,
)

MARKERS = (
    ("marker_red", [0.4, -0.25, 1.0], 0.4, [1., 0.03, 0.03, 1.]),
    ("marker_green", [-0.4, 0.3, 1.25], 0.4, [0.03, 1., 0.03, 1.]),
    ("marker_blue", [0.15, 0.45, 0.7], 0.4, [0.03, 0.03, 1., 1.]),
)


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def box_surface_error(point, center, half_size):
    # Independent analytic reference: known axis-aligned constructed geometry.
    signed = np.abs(np.asarray(point) - center) - half_size
    return float(np.linalg.norm(np.maximum(signed, 0.))) if np.any(signed > 0) else float(-np.max(signed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    import omnigibson as og
    from PIL import Image
    from scipy.ndimage import binary_erosion

    objects = [
        {"type": "PrimitiveObject", "name": name, "primitive_type": "Cube",
         "size": size, "fixed_base": True, "position": pos, "rgba": color}
        for name, pos, size, color in MARKERS]
    # A dynamic rigid body is required by this OG version's global PhysX view.
    objects.append({"type": "PrimitiveObject", "name": "dynamic_reference",
                    "primitive_type": "Sphere", "radius": .05,
                    "position": [3., 3., 1.], "fixed_base": False})
    cfg = {"scene": {"type": "Scene"}, "robots": [], "objects": objects,
           "task": {"type": "DummyTask"},
           "env": {"use_external_obs": True, "external_sensors": [
               {"sensor_type": "VisionSensor", "name": "calibration_camera",
                "modalities": ["rgb", "depth", "depth_linear", "seg_instance"],
                "sensor_kwargs": {"image_width": args.resolution, "image_height": args.resolution},
                "position": [0., 0., 3.], "orientation": [0., 0., 0., 1.]}]}}
    theta_y = math.atan2(1., 2.)
    theta_x = -math.atan2(1., 1.5)
    poses = [
        ([0., 0., 3.], [0., 0., 0., 1.]),
        ([0.3, -0.15, 4.], [0., 0., 0., 1.]),
        ([1., 0., 3.], [0., math.sin(theta_y/2), 0., math.cos(theta_y/2)]),
        ([0., 1., 2.5], [math.sin(theta_x/2), 0., 0., math.cos(theta_x/2)]),
    ]
    report = {"kind": "independent_constructed_geometry_calibration",
              "tolerance_m": .01, "views": [], "ok": False}
    try:
        env = og.Environment(configs=cfg)
        sensor = env._external_sensors[0]["calibration_camera"]
        K = array(sensor.intrinsic_matrix)
        for index, (position, orientation) in enumerate(poses):
            sensor.set_position_orientation(position=position, orientation=orientation)
            for _ in range(4):
                og.sim.render()
            obs, info = env.get_obs()
            data = obs[0]["external"]["calibration_camera"]
            labels = info[0]["external"]["calibration_camera"]["seg_instance"]
            seg = array(data["seg_instance"])
            camera_position, camera_quaternion = sensor.get_position_orientation()
            T = world_from_usd_camera(array(camera_position), array(camera_quaternion))
            Image.fromarray(array(data["rgb"])[..., :3].astype(np.uint8)).save(
                output.parent / f"{output.stem}_rgb_{index}.png")
            view = {"view": index, "markers": []}
            for name, center, size, _color in MARKERS:
                ids = [int(key) for key, label in labels.items() if label == name]
                mask = binary_erosion(np.isin(seg, ids), iterations=3)
                rows, cols = np.where(mask)
                if len(rows) < 25:
                    raise RuntimeError(f"insufficient independent marker pixels: view {index}, {name}")
                chosen = np.linspace(0, len(rows)-1, 25, dtype=int)
                marker = {"name": name, "samples_per_modality": len(chosen), "errors": {}}
                for modality, unproject in (("depth", unproject_range), ("depth_linear", unproject_z_depth)):
                    depth = array(data[modality])
                    errors = []
                    for j in chosen:
                        u, v = int(cols[j]), int(rows[j])
                        distance = float(depth[v, u])
                        if not np.isfinite(distance) or distance <= 0:
                            raise RuntimeError(f"invalid {modality} on marker")
                        point = to_world(unproject(u, v, distance, K), T)
                        errors.append(box_surface_error(point, np.asarray(center), size / 2))
                    marker["errors"][modality] = {
                        "max_m": max(errors), "median_m": float(np.median(errors))}
                view["markers"].append(marker)
            report["views"].append(view)
            print("[calibration] completed view", index, flush=True)
        report["max_error_m"] = max(
            stats["max_m"] for view in report["views"]
            for marker in view["markers"] for stats in marker["errors"].values())
        report["ok"] = report["max_error_m"] <= report["tolerance_m"]
    except Exception as error:
        report["error"] = repr(error)
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(report, indent=2))
        print("[calibration] result", report.get("max_error_m"), report["ok"], flush=True)
        og.shutdown()


if __name__ == "__main__":
    main()
