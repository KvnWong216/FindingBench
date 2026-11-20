"""§8/rev§4 visual bridge tests: center-pixel authoritative selection.

The 11x11 neighborhood never overrides the center-pixel choice: a thin
target on any background is selectable, a background center never selects a
nearby target, and surface samples come only from the selected object.
"""
import numpy as np
import pytest

from rummagebench.perception.frame_store import VisualFramePrivate
from rummagebench.perception.camera_geometry import default_intrinsics
from rummagebench.perception.visual_bridge import (
    GROUNDING_VERSION,
    VisualBridge,
    VisualGroundingError,
)

H = W = 64


def make_frame(blocks: dict[str, tuple[int, int, int, int]],
               depth_by_label: dict[str, float] | float = 1.0):
    """blocks: instance label -> (u, v, w, h) pixel rect."""
    seg = np.zeros((H, W), dtype=np.int64)
    depth = np.full((H, W), np.nan)
    labels: dict[str, str] = {}
    for idx, (label, (u, v, w, h)) in enumerate(blocks.items(), start=2):
        seg[v:v + h, u:u + w] = idx
        d = (depth_by_label.get(label, 1.0)
             if isinstance(depth_by_label, dict) else depth_by_label)
        depth[v:v + h, u:u + w] = d
        labels[str(idx)] = label
    return VisualFramePrivate(
        frame_id="frame_test", rgb=np.zeros((H, W, 3), dtype=np.uint8),
        depth=depth, instance_segmentation=seg,
        camera_intrinsics=default_intrinsics(W, H), camera_extrinsics=np.eye(4),
        image_width=W, image_height=H, depth_convention="z_depth",
    ), labels


def canon(labels: dict[str, str]):
    return lambda inst: labels.get(str(inst.item() if hasattr(inst, "item")
                                          else inst))


def test_point_on_visible_object_resolves_to_entity():
    frame, labels = make_frame({"cabinet_B": (4, 24, 20, 20)})
    bridge = VisualBridge()
    entity, point, detail = bridge.resolve(frame, 0.1, 0.5, canon(labels))
    assert entity == "cabinet_B"
    assert 0 <= detail["pixel"][0] < W


def test_thin_knife_on_table_still_selects_knife():
    """Revision §4 headline case: a 2px-wide knife across a dominating table
    is selected when the crosshair is ON the knife (the old 11x11 dominance
    rule returned the table)."""
    # table = large block; knife = 2px-thick horizontal strip across it
    frame, labels = make_frame({"table": (10, 10, 44, 30)})
    import numpy as np
    seg = np.asarray(frame.instance_segmentation).copy()
    seg[24:26, 12:52] = 9  # knife strip
    np.asarray(frame.instance_segmentation)[:] = seg
    labels["9"] = "knife"
    frame.depth[24:26, 12:52] = 0.8  # knife closer than the table
    bridge = VisualBridge()
    entity, _, detail = bridge.resolve(frame, 0.5, 24.5 / H, canon(labels))
    assert entity == "knife"


def test_background_center_near_knife_does_not_select_knife():
    frame, labels = make_frame({"knife": (28, 31, 8, 2)})
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError) as e:
        bridge.resolve(frame, 0.5, 0.3, canon(labels))  # background above knife
    assert e.value.subreason == "NO_VISUAL_TARGET"


def test_center_on_bowl_does_not_select_hidden_knife():
    """The knife lies fully hidden under an inverted bowl: a bowl center
    selects the bowl, never the invisible knife."""
    frame, labels = make_frame({"inverted_bowl": (20, 20, 24, 24)})
    bridge = VisualBridge()
    entity, _, _ = bridge.resolve(frame, 0.5, 0.5, canon(labels))
    assert entity == "inverted_bowl"
    assert entity != "hidden_knife"


def test_link_pixels_map_to_interaction_entity():
    """Handle and door pixels both canonicalize to the owning cabinet."""
    frame, labels = make_frame({"cabinet_door": (10, 10, 20, 40),
                                "cabinet_handle": (34, 24, 8, 6)})
    # verified owning-object merge: the handle instance maps to the cabinet
    mapping = {**labels, **{k: "cabinet_door"
                            for k, v in labels.items()
                            if v == "cabinet_handle"}}
    bridge = VisualBridge()
    e1, _, _ = bridge.resolve(frame, 0.3, 0.5, canon(mapping))
    e2, _, _ = bridge.resolve(frame, 0.62, 0.42, canon(mapping))
    assert e1 == e2 == "cabinet_door"


def test_surface_point_uses_selected_objects_own_samples():
    """Knife in front of a table: the surface point comes from the knife's
    own depth samples, never from the table behind it."""
    frame, labels = make_frame(
        {"table": (0, 0, 64, 64), "knife": (28, 30, 12, 3)},
        depth_by_label={"table": 2.0, "knife": 1.0})
    bridge = VisualBridge()
    entity, point, _ = bridge.resolve(frame, 0.53, 0.49, canon(labels))
    assert entity == "knife"
    assert abs(point[2] - 1.0) < 1e-6


def test_background_point_invalid():
    frame, _ = make_frame({"cabinet_B": (4, 24, 20, 20)})
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError) as e:
        bridge.select_instance(frame, 0.9, 0.05)
    assert e.value.subreason == "NO_VISUAL_TARGET"


def test_small_object_center_still_selects():
    """min-pixel dominance is retired: a 3x3 object under the crosshair is
    selected (the neighborhood no longer outvotes the center)."""
    frame, labels = make_frame({"peeler": (30, 30, 3, 3)})
    bridge = VisualBridge()
    entity, _, _ = bridge.resolve(frame, 0.5, 0.5, canon(labels))
    assert entity == "peeler"


def test_off_camera_object_cannot_be_selected():
    frame, labels = make_frame({})
    bridge = VisualBridge()
    with pytest.raises(VisualGroundingError):
        bridge.resolve(frame, 0.5, 0.5, canon(labels))


def test_grounding_version_bound():
    assert GROUNDING_VERSION == "center_pixel_v2"
