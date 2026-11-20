"""Camera geometry: pixel conversion, unprojection, depth-convention handling.

Active capture obtains conventions from the OmniGibson modality API:
``depth`` is Euclidean range; ``depth_linear`` is optical-axis Z depth.
Real-host pixel correspondence still requires independent calibration tests.
The calibration helper below is diagnostic, not evidence those tests ran.
"""

from __future__ import annotations

import numpy as np


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Protocol §8.1: u = round(x * (W-1)), v = round(y * (H-1))."""
    u = int(round(float(x) * (width - 1)))
    v = int(round(float(y) * (height - 1)))
    u = min(max(u, 0), width - 1)
    v = min(max(v, 0), height - 1)
    return u, v


def default_intrinsics(width: int, height: int, hfov_deg: float = 90.0) -> np.ndarray:
    """Pinhole K for a horizontal FOV (OmniGibson default 90 deg), principal
    point at the image center."""
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = fx  # square pixels
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def unproject_z_depth(u: int, v: int, z: float, K: np.ndarray) -> np.ndarray:
    """Optical-axis depth: Xc=(u-cx)z/fx, Yc=(v-cy)z/fy, Zc=z."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def unproject_range(u: int, v: int, rng: float, K: np.ndarray) -> np.ndarray:
    """Euclidean range: p = range * normalize(inv(K) @ [u,v,1])."""
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray = ray / np.linalg.norm(ray)
    return rng * ray


def to_world(p_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (T_world_camera @ np.append(p_camera, 1.0))[:3]


def calibrate_depth_convention(
    depth: np.ndarray,
    seg: np.ndarray,
    instance: object,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    reference_point_world: np.ndarray,
) -> str:
    """§8.5: decide the camera's depth convention from a known 3D surface
    point. The reference is projected (z_depth pinhole) to a pixel; in a
    small window around it BOTH conventions unproject the captured depth and
    the one whose points land closest to the reference wins. Off-axis pixels
    distinguish the conventions (on-axis they coincide)."""
    seg = np.asarray(seg)
    depth = np.asarray(depth)
    mask = seg == instance
    if not mask.any():
        return "z_depth"

    # project the reference assuming z_depth pinhole model
    T_cam_world = np.linalg.inv(T_world_camera)
    p_cam = T_cam_world @ np.append(reference_point_world, 1.0)
    if p_cam[2] <= 0:
        return "z_depth"
    u_ref = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
    v_ref = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
    H, W = seg.shape[:2]
    if not (0 <= u_ref < W and 0 <= v_ref < H):
        return "z_depth"

    errs = {"z_depth": [], "euclidean_range": []}
    for v in range(max(0, v_ref - 3), min(H, v_ref + 4)):
        for u in range(max(0, u_ref - 3), min(W, u_ref + 4)):
            if not mask[v, u]:
                continue
            d = float(depth[v, u])
            if not np.isfinite(d) or d <= 0:
                continue
            errs["z_depth"].append(float(np.linalg.norm(
                to_world(unproject_z_depth(u, v, d, K), T_world_camera)
                - reference_point_world)))
            errs["euclidean_range"].append(float(np.linalg.norm(
                to_world(unproject_range(u, v, d, K), T_world_camera)
                - reference_point_world)))
    if not errs["z_depth"]:
        return "z_depth"
    med = {k: float(np.median(v)) for k, v in errs.items() if v}
    if "euclidean_range" not in med:
        return "z_depth"
    return "z_depth" if med["z_depth"] <= med["euclidean_range"] else "euclidean_range"


def world_from_usd_camera(position, quaternion_xyzw) -> np.ndarray:
    """World-from-optical transform from an OmniGibson sensor world pose.

    OG XFormPrim returns XYZW. USD cameras look down -Z with +Y up;
    our optical convention is +Z forward, +Y down, +X right.
    Source: https://openusd.org/dev/api/class_usd_geom_camera.html
    This convention conversion is not a substitute for real-host calibration.
    """
    p = np.asarray(position, dtype=float)
    q = np.asarray(quaternion_xyzw, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid camera pose")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero camera quaternion")
    x, y, z, w = q / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation @ np.diag([1., -1., -1.])
    result[:3, 3] = p
    return result
