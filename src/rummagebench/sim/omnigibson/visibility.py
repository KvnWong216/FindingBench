"""Evaluator-only target visibility ratio (task-authoring certification).

    ratio = visible target pixels / isolated reference pixels

Both counts come from the private renderer instance segmentation of the SAME
camera pose. The reference render hides EVERY scene object except the target
(render-only USD visibility; physics untouched), robot included, so it is the
target's unoccluded projection inside the field of view. With nothing else
visible, the reference is simply the number of non-background pixels: it
does not depend on the instance-label mapping, which the renderer
re-assigns whenever prim visibility changes (verified on this host: every
instance id changes after a hide/restore cycle).

The occluded count keeps the robot visible: its own body occludes what the
agent's camera sees, exactly as in AGENT mode.

A reference of zero pixels means the target is outside the view: the ratio
is undefined (None), never 0 — callers decide what that means. A reference
smaller than the visible count is an inconsistent measurement; it is retried
and, if it persists, reported with ``consistent=False`` and ratio None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VisibilityMeasurement:
    visible_px: int
    reference_px: int
    consistent: bool = True
    attempts: int = 1

    @property
    def ratio(self) -> Optional[float]:
        if self.reference_px <= 0 or not self.consistent:
            return None
        return min(1.0, self.visible_px / self.reference_px)

    def to_dict(self) -> dict:
        return {"visible_px": self.visible_px, "reference_px": self.reference_px,
                "ratio": self.ratio, "consistent": self.consistent,
                "attempts": self.attempts}


def capture_counts(backend, entities: list[str]) -> dict[str, int]:
    """Visible pixels per entity in one fresh frame."""
    frame = backend.capture_visual_frame()
    return {e: backend.entity_visible_pixels(frame, e) for e in entities}


def _foreground_px(backend) -> int:
    frame = backend.capture_visual_frame()
    return int(np.count_nonzero(np.asarray(frame.instance_segmentation)))


def measure_visibility(backend, entity: str, renders: int = 3,
                       retries: int = 3) -> VisibilityMeasurement:
    """Occluded vs isolated pixel counts of ``entity`` from the current pose."""
    # stable read: instance ids are re-assigned after visibility changes, so
    # require two consecutive identical counts (max 4 frames)
    backend.flush_render(renders)
    visible = capture_counts(backend, [entity])[entity]
    for _ in range(3):
        again = capture_counts(backend, [entity])[entity]
        if again == visible:
            break
        visible = again
    hidden = []
    reference, attempts = 0, 0
    try:
        for obj in backend._env.scene.objects:
            if obj.name == entity:
                continue
            if obj.visible:
                obj.visible = False
                hidden.append(obj)
        for attempts in range(1, retries + 1):
            backend.flush_render(renders * attempts)
            reference = _foreground_px(backend)
            if reference >= visible:
                break
    finally:
        for obj in hidden:
            obj.visible = True
        backend.flush_render(renders)
    return VisibilityMeasurement(int(visible), int(reference),
                                 consistent=reference >= visible, attempts=attempts)


def visible_entities_in_view(backend, min_px: int) -> dict[str, int]:
    """Every labelled entity with at least ``min_px`` pixels in a fresh frame."""
    backend.flush_render(2)
    frame = backend.capture_visual_frame()
    seg = np.asarray(frame.instance_segmentation)
    labels = frame.meta.get("instance_labels", {})
    ids, counts = np.unique(seg, return_counts=True)
    per_entity: dict[str, int] = {}
    for i, n in zip(ids.tolist(), counts.tolist()):
        name = labels.get(str(i))
        if name:
            per_entity[name] = per_entity.get(name, 0) + int(n)
    return {k: v for k, v in per_entity.items() if v >= min_px}
