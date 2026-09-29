"""§39 camera geometry: pixel conversion, unprojection, convention check."""

import numpy as np

from rummagebench.perception.camera_geometry import (
    calibrate_depth_convention,
    default_intrinsics,
    normalized_to_pixel,
    to_world,
    unproject_range,
    unproject_z_depth,
)


def test_pixel_conversion_exact():
    assert normalized_to_pixel(0.0, 0.0, 640, 480) == (0, 0)
    assert normalized_to_pixel(1.0, 1.0, 640, 480) == (639, 479)
    u, v = normalized_to_pixel(0.43, 0.61, 640, 480)
    assert (u, v) == (round(0.43 * 639), round(0.61 * 479))


def test_z_depth_unprojection_center():
    K = default_intrinsics(640, 480)
    p = unproject_z_depth(319, 239, 2.0, K)  # principal point -> [0,0,z]
    assert abs(p[0]) <= 0.01 and abs(p[1]) <= 0.01 and p[2] == 2.0


def test_range_unprojection_norm_matches():
    K = default_intrinsics(640, 480)
    p = unproject_range(500, 300, 3.0, K)
    assert abs(np.linalg.norm(p) - 3.0) < 1e-9


def test_projection_unprojection_roundtrip_under_1cm():
    """§8.6: project a known world point, unproject it, error <= 1 cm."""
    K = default_intrinsics(640, 480)
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.5]  # camera 1.5 m up
    # a point 2 m ahead of the camera, on the ground plane
    p_world = np.array([0.5, -3.0, 0.0])
    p_cam = (np.linalg.inv(T) @ np.append(p_world, 1.0))[:3]
    # project via z-depth pinhole
    u = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
    v = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
    z = p_cam[2]
    back_cam = unproject_z_depth(u, v, z, K)
    back_world = to_world(back_cam, T)
    # pixel rounding dominates: at 2 m with fx=320, 1 px ~ 6.3 mm
    assert np.linalg.norm(back_world - p_world) <= 0.01 + 0.007


def _plane_frame(depth_values):
    """Instance patch at an OFF-AXIS pixel block; camera 64x64."""
    K = default_intrinsics(64, 64)
    seg = np.zeros((64, 64), dtype=np.int64)
    seg[28:35, 40:47] = 2
    depth = np.full((64, 64), np.nan)
    for v in range(28, 35):
        for u in range(40, 47):
            depth[v, u] = depth_values(u, v, K)
    return depth, seg, K


def test_depth_convention_calibration_picks_z_depth():
    # data generated as a plane z = 2 parallel to the image plane: the stored
    # value IS optical-axis depth
    depth, seg, K = _plane_frame(lambda u, v, K: 2.0)
    # reference: true world point at pixel (42, 30)
    u0, v0 = 42, 30
    reference = np.array([(u0 - K[0, 2]) * 2.0 / K[0, 0],
                          (v0 - K[1, 2]) * 2.0 / K[1, 1], 2.0])
    assert calibrate_depth_convention(depth, seg, 2, K, np.eye(4), reference) == "z_depth"


def test_depth_convention_calibration_picks_range():
    # data generated as euclidean range from the camera origin
    K = None

    def range_fn(u, v, K):
        ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        return 2.2 * float(np.linalg.norm(ray))

    depth, seg, K = _plane_frame(range_fn)
    u0, v0 = 42, 30
    ray = np.array([(u0 - K[0, 2]) / K[0, 0], (v0 - K[1, 2]) / K[1, 1], 1.0])
    ray = ray / np.linalg.norm(ray)
    reference = 2.2 * ray
    assert calibrate_depth_convention(depth, seg, 2, K, np.eye(4), reference) == "euclidean_range"
