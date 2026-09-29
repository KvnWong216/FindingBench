"""Perception package: synchronized private frames + visual grounding."""

from rummagebench.perception.camera_geometry import (
    calibrate_depth_convention,
    default_intrinsics,
    normalized_to_pixel,
    to_world,
    unproject_range,
    unproject_z_depth,
)
from rummagebench.perception.frame_store import FrameStore, VisualFramePrivate
from rummagebench.perception.visual_bridge import VisualBridge, VisualGroundingError

__all__ = [
    "FrameStore",
    "VisualFramePrivate",
    "VisualBridge",
    "VisualGroundingError",
    "normalized_to_pixel",
    "default_intrinsics",
    "unproject_z_depth",
    "unproject_range",
    "to_world",
    "calibrate_depth_convention",
]
