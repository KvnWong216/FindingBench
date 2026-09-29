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


# ============================================================================
# Visual protocol (§7): synchronized PRIVATE modality capture. The agent sees
# only the RGB + frame_id; depth / instance segmentation / camera geometry
# live in the evaluator-private FrameStore.
# ============================================================================

def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def capture_private_frame(env, robot_name: str) -> dict:
    """One synchronized modality bundle from the head camera:
    {rgb, depth, seg, seg_is_paths} — same render. OmniGibson modalities:
    'rgb', 'depth' (float meters), 'seg_instance' (ids or prim-path strings
    depending on version — both treated as opaque instance keys)."""
    rgb = depth = seg = None
    obs_list, _info = env.get_obs()
    obs = obs_list[0]
    robot_obs = obs.get(robot_name, {})
    for _sensor, sensor_obs in robot_obs.items():
        if not isinstance(sensor_obs, dict):
            continue
        if rgb is None and sensor_obs.get("rgb") is not None:
            rgb = _to_numpy(sensor_obs["rgb"])
        if depth is None and sensor_obs.get("depth") is not None:
            depth = _to_numpy(sensor_obs["depth"]).astype(np.float64)
        if seg is None and sensor_obs.get("seg_instance") is not None:
            seg = _to_numpy(sensor_obs["seg_instance"])
    if rgb is None:
        raise RuntimeError(f"no RGB sensor for robot {robot_name!r}")
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8)

    H, W = rgb.shape[:2]
    if depth is None or depth.shape[:2] != (H, W):
        depth = np.full((H, W), np.nan)
    if seg is None:
        seg = np.zeros((H, W))
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    if seg.shape[:2] != (H, W):
        seg = np.resize(seg, (H, W))
    return {"rgb": rgb, "depth": depth, "seg": seg,
            "seg_is_paths": seg.dtype.kind in "USO"}


def camera_extrinsics(env, robot_name: str, sensor_name: str | None) -> np.ndarray:
    """T_world_camera 4x4 from the sensor prim; identity fallback (unprojection
    then lives in camera space). Never exposed to the agent."""
    import numpy as _np

    try:
        robot = next((r for r in env.scene.robots if r.name == robot_name), None)
        if robot is None or not sensor_name:
            return _np.eye(4)
        sensor = robot.sensors.get(sensor_name) if hasattr(robot, "sensors") else None
        prim_path = getattr(sensor, "prim_path", None)
        if not prim_path:
            return _np.eye(4)
        import omnigibson.lazy as lazy

        prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
        if prim is None:
            return _np.eye(4)
        if prim.HasAttribute("xformOp:transform"):
            T = _np.asarray(prim.GetAttribute("xformOp:transform").Get()).reshape(4, 4).T
            return T
        T = _np.eye(4)
        if prim.HasAttribute("xformOp:translate"):
            T[:3, 3] = _np.asarray(prim.GetAttribute("xformOp:translate").Get())[:3]
        if prim.HasAttribute("xformOp:orient"):
            q = _np.asarray(prim.GetAttribute("xformOp:orient").Get())
            q = q[[1, 2, 3, 0]]  # wxyz -> xyzw
            from scipy.spatial.transform import Rotation as R

            T[:3, :3] = R.from_quat(q).as_matrix()
        return T
    except Exception:
        return _np.eye(4)


def build_visual_frame(env, robot_name: str, frame_id: str,
                       instance_to_entity, reference_point_world=None):
    """Assemble a VisualFramePrivate with auto-calibrated depth convention
    (§8.5: verified, never assumed)."""
    from rummagebench.perception.frame_store import VisualFramePrivate
    from rummagebench.perception.camera_geometry import (
        calibrate_depth_convention, default_intrinsics,
    )

    bundle = capture_private_frame(env, robot_name)
    H, W = bundle["rgb"].shape[:2]
    K = default_intrinsics(W, H)
    sensors = camera_sensor_names(env, robot_name)
    T_world_camera = camera_extrinsics(env, robot_name, sensors[0] if sensors else None)

    frame = VisualFramePrivate(
        frame_id=frame_id,
        rgb=bundle["rgb"],
        depth=bundle["depth"],
        instance_segmentation=bundle["seg"],
        camera_intrinsics=K,
        camera_extrinsics=T_world_camera,
        image_width=W,
        image_height=H,
    )
    if reference_point_world is not None and not bundle["seg_is_paths"]:
        frame.depth_convention = calibrate_depth_convention(
            bundle["depth"], bundle["seg"],
            _entity_instance_in_frame(bundle["seg"], instance_to_entity),
            K, T_world_camera, reference_point_world,
        )
    else:
        frame.depth_convention = "z_depth"
    return frame


def _entity_instance_in_frame(seg, instance_to_entity):
    """A representative instance key that canonicalizes to a benchmark entity."""
    if seg.dtype.kind in "USO":
        flat = seg.ravel()
        for s in flat[:: max(1, len(flat) // 64)]:
            key = s.item() if hasattr(s, "item") else s
            if key and instance_to_entity(key) is not None:
                return key
        return None
    ids, counts = np.unique(seg, return_counts=True)
    for inst in ids[np.argsort(-counts)]:
        if inst == 0:
            continue
        if instance_to_entity(inst.item() if hasattr(inst, "item") else inst) is not None:
            return inst.item() if hasattr(inst, "item") else inst
    return None
