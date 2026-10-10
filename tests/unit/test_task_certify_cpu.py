"""CPU parts of the GPU certifier (authoring/tasks/certify.py)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rummagebench.authoring.tasks.certify import _vis_value, click_point
from rummagebench.perception.camera_geometry import normalized_to_pixel


def _frame(seg, labels):
    return SimpleNamespace(instance_segmentation=seg, meta={"instance_labels": labels})


def test_click_point_lands_inside_the_entity_interior():
    seg = np.zeros((60, 80), np.int64)
    seg[10:50, 10:30] = 5   # cabinet body
    seg[20:30, 50:52] = 7   # thin sliver of the same cabinet (other instance)
    seg[40:44, 60:64] = 9   # target
    f = _frame(seg, {"5": "cab", "7": "cab", "9": "knife"})
    x, y = click_point(f, None, "cab")
    u, v = normalized_to_pixel(x, y, 80, 60)
    assert seg[v, u] == 5 and 15 <= u <= 25 and 15 <= v <= 45
    x, y = click_point(f, None, "knife")
    u, v = normalized_to_pixel(x, y, 80, 60)
    assert seg[v, u] == 9
    assert click_point(f, None, "absent") is None


def test_out_of_view_counts_as_invisible():
    assert _vis_value({"ratio": None, "visible_px": 0, "reference_px": 0}) == 0.0
    assert _vis_value({"ratio": 0.7, "visible_px": 7, "reference_px": 10}) == 0.7


def test_target_ring_faces_the_target():
    import math

    from rummagebench.authoring.tasks.certify import target_ring

    ring = target_ring((2.0, -1.0), distances=(1.0,), bearings_deg=(0, 90, 180))
    assert len(ring) == 3
    for x, y, yaw in ring:
        assert math.isclose(math.hypot(2.0 - x, -1.0 - y), 1.0, abs_tol=1e-9)
        # heading points from the base to the target
        assert math.isclose(math.atan2(-1.0 - y, 2.0 - x), yaw, abs_tol=1e-9)


def test_view_prediction_matches_head_camera_geometry():
    """R1Pro head camera (A.5 F2): 1.62 m high, ~20 deg down, vertical FOV
    58.7 deg -> a counter-height object is below the view when the base is
    close and inside it from ~1.1 m."""
    import math

    from rummagebench.authoring.tasks.certify import predicts_in_view

    W, H = 640, 480
    fy = (H / 2) / math.tan(math.radians(58.7 / 2))
    K = np.array([[fy, 0, (W - 1) / 2], [0, fy, (H - 1) / 2], [0, 0, 1]])
    p = math.radians(20)
    # optical frame (+Z forward, +Y down) in the base frame, pitched down
    R = np.array([[0, -math.sin(p), math.cos(p)],
                  [-1, 0, 0],
                  [0, -math.cos(p), -math.sin(p)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.0, 0.0, 1.62]
    box = ([-0.04, -0.04, 0.90], [0.04, 0.04, 0.98])  # mug on a counter at x=0
    assert not predicts_in_view(K, T, (-0.6, 0.0, 0.0), 0.0, box, W, H)
    assert predicts_in_view(K, T, (-1.3, 0.0, 0.0), 0.0, box, W, H)
    assert not predicts_in_view(K, T, (-1.3, 0.0, math.pi), 0.0, box, W, H)  # facing away
