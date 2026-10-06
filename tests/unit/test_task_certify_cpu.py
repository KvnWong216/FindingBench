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
