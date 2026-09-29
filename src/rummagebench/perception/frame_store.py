"""Evaluator-private synchronized camera frames.

For every public RGB frame the environment captures a synchronized private
bundle (rgb + depth + instance segmentation + camera geometry) and registers
it here under an opaque frame_id. ONLY the RGB and the frame_id cross the
agent boundary; every other modality stays private. Only the LATEST current
frame is a valid point-reference target (OBSERVE auxiliary views are never
registered as current).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VisualFramePrivate:
    frame_id: str
    rgb: Any                      # HxWx3 uint8
    depth: Any                    # HxW float (convention auto-calibrated)
    instance_segmentation: Any    # HxW (string prim path per pixel, or ids)
    camera_intrinsics: Any        # 3x3 K
    camera_extrinsics: Any        # 4x4 T_world_camera
    image_width: int
    image_height: int
    depth_convention: str = "unknown"  # "z_depth" | "euclidean_range"
    meta: dict[str, Any] = field(default_factory=dict)


class FrameStore:
    """Registry of captured frames. `current` is the only actionable frame."""

    def __init__(self, max_frames: int = 64):
        self._frames: dict[str, VisualFramePrivate] = {}
        self._counter = itertools.count()
        self._current: Optional[str] = None
        self._max = max_frames

    def register(self, frame: VisualFramePrivate) -> str:
        self._frames[frame.frame_id] = frame
        self._current = frame.frame_id
        if len(self._frames) > self._max:  # bounded memory, FIFO eviction
            oldest = next(iter(self._frames))
            if oldest != self._current:
                self._frames.pop(oldest, None)
        return frame.frame_id

    def new_frame_id(self) -> str:
        return f"frame_{next(self._counter):06d}"

    def get(self, frame_id: str) -> Optional[VisualFramePrivate]:
        return self._frames.get(frame_id)

    @property
    def current_id(self) -> Optional[str]:
        return self._current

    def is_current(self, frame_id: str) -> bool:
        return frame_id is not None and frame_id == self._current

    def reset(self) -> None:
        self._frames.clear()
        self._current = None
