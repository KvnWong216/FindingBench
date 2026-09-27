"""Observation capture: head-camera RGB only.

Depth and segmentation remain internal to validators / debug tooling and are
never exposed to the evaluated agent unless a track explicitly enables them.
"""

from __future__ import annotations

import numpy as np


def capture_head_rgb(env, robot_name: str) -> np.ndarray:
    """Return the robot head camera RGB as HxWx3 uint8 numpy array."""
    obs_list, _info = env.get_obs()
    obs = obs_list[0]
    robot_obs = obs.get(robot_name)
    if not robot_obs:
        raise RuntimeError(f"no observations for robot {robot_name!r}")
    for sensor_name, sensor_obs in robot_obs.items():
        if sensor_name == "proprio" or not isinstance(sensor_obs, dict):
            continue
        rgb = sensor_obs.get("rgb")
        if rgb is None:
            continue
        arr = rgb
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8)
    raise RuntimeError(f"no RGB sensor found for robot {robot_name!r}")


def camera_sensor_names(env, robot_name: str) -> list[str]:
    obs_list, _info = env.get_obs()
    obs = obs_list[0]
    robot_obs = obs.get(robot_name, {})
    return [k for k in robot_obs.keys() if k != "proprio"]
