"""§39 visual bridge tests: patch selection, dominance, depth, canonicalize."""

import numpy as np
import pytest

from rummagebench.perception.frame_store import VisualFramePrivate
from rummagebench.perception.camera_geometry import default_intrinsics
from rummagebench.perception.visual_bridge import VisualBridge, VisualGroundingError

H = W = 64


def make_frame(blocks: dict[str, tuple[int, int, int, int]], depth_m: float = 1.0):
    """blocks: instance label -> (u, v, w, h) pixel rect."""
    seg = np.zeros((H, W), dtype=np.int64)
    depth = np.full((H, W), np.nan)
    labels: dict[str, str] = {}
    for idx, (label, (u, v, w, h)) in enumerate(blocks.items(), start=2):
        seg[v:v + h, u:u + w] = idx
        depth[v:v + h, u:u + w] = depth_m
        labels[str(idx)] = label
    return VisualFramePrivate(
        frame_id="frame_test", rgb=np.zeros((H, W, 3), dtype=np.uint8),
        depth=depth, instance_segmentation=seg,
        camera_intrinsics=default_intrinsics(W, H), camera_extrinsics=np.eye(4),
        image_width=W, image_height=H, depth_convention="z_depth",
    ), labels


def test_point_on_visible_object_resolves_to_entity():
    frame, labels = make_frame({"cabinet_B": (4, 24, 20, 20)})
    bridge = VisualBridge()
    entity, point, detail = bridge.resolve(
        frame, 0.1, 0.5, lambda inst: labels.get(str(inst.item() if hasattr(inst, 'item') else inst))
    )
    assert entity == "cabinet_B"
    assert 0 <= detail["pixel"][0] < W


def test_background_point_invalid():
    frame, _ = make_frame({"cabinet_B": (4, 24, 20, 20)})
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError) as e:
        bridge.select_instance(frame, 0.9, 0.05)  # far from any instance
    assert e.value.subreason == "NO_VISUAL_TARGET"


def test_too_few_pixels_invalid():
    frame, _ = make_frame({"cabinet_B": (30, 30, 3, 3)})  # 9 px < 20
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError):
        bridge.select_instance(frame, 0.5, 0.5)


def test_ambiguous_patch_invalid():
    # 11x11 patch split 5 rows / 4 rows / 2 rows across three instances:
    # the dominant instance holds 5/11 = 0.45 < 0.50
    seg = np.zeros((H, W), dtype=np.int64)
    seg[27:32, 27:38] = 2
    seg[32:36, 27:38] = 3
    seg[36:38, 27:38] = 4
    depth = np.full((H, W), 1.0)
    frame = VisualFramePrivate(
        frame_id="f", rgb=np.zeros((H, W, 3), dtype=np.uint8), depth=depth,
        instance_segmentation=seg, camera_intrinsics=default_intrinsics(W, H),
        camera_extrinsics=np.eye(4), image_width=W, image_height=H,
        depth_convention="z_depth",
    )
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError) as e:
        bridge.select_instance(frame, 0.5, 0.5)
    assert e.value.subreason == "AMBIGUOUS_VISUAL_TARGET"


def test_surface_point_uses_median_depth():
    frame, labels = make_frame({"cabinet_B": (4, 24, 20, 20)}, depth_m=2.0)
    bridge = VisualBridge()
    entity, point, _ = bridge.resolve(
        frame, 0.1, 0.5, lambda inst: labels.get(str(inst.item() if hasattr(inst, 'item') else inst))
    )
    # z_depth at the frame center ray with identity extrinsics: x,y ~ 0
    assert abs(point[2] - 2.0) < 1e-6


def test_off_camera_object_cannot_be_selected():
    frame, labels = make_frame({})  # nothing in frame at all
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError):
        bridge.resolve(frame, 0.5, 0.5, lambda inst: labels.get(str(inst)))
