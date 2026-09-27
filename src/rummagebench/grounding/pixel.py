"""Pixel grounding (future work, stub).

The evaluated agent will return a normalized pixel on a named camera:

    {"type": "pixel", "camera": "head_rgb", "x": 0.63, "y": 0.41}

The evaluator privately maps that point to a simulator instance using
ground-truth instance segmentation, which is never exposed to the agent.
The MVP does not implement the mapping yet; it raises a clear error so no
code path silently pretends visual grounding works.
"""

from __future__ import annotations

from rummagebench.core.errors import UnresolvableTargetError
from rummagebench.core.types import TargetKind, TargetRef
from rummagebench.sim.base import ResolvedTarget, SimBackend


class PixelGrounding:
    """Maps a pixel on a robot camera to an entity via evaluator-private segmentation."""

    def __init__(self, backend: SimBackend):
        self._backend = backend

    def resolve(self, target: TargetRef) -> ResolvedTarget:
        if target.type == TargetKind.ENTITY:
            return self._backend.resolve_entity(target.value or "")
        raise UnresolvableTargetError(
            "pixel grounding is not implemented in the MVP; "
            "use oracle entity grounding or extend sim/omnigibson segmentation capture"
        )
