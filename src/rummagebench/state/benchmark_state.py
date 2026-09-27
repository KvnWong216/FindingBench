"""Benchmark-owned world state.

The holding state is benchmark semantic truth: it is owned by the benchmark
core and never derived from the simulator's assisted-grasp internals
(``_ag_obj_in_hand``). The backend's assisted-grasp realization is
visualization/optional realization only and cannot decide ``is_holding()``,
``held_count()`` or task success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rummagebench.feasibility.ik_solver import Pose


@dataclass
class BenchmarkWorldState:
    """Semantic state the benchmark core is the authority for.

    held_object: entity name currently grasped (hand_capacity=1 semantics)
    held_offset: grasp-time pose of the held object relative to the end
      effector (EEF frame), used to include the held body in configuration-
      space collision checks; None when the backend cannot provide an EEF
      pose (the held body is then excluded from collision checks, logged).
    """

    held_object: str | None = None
    held_offset: Pose | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def held_count(self) -> int:
        return 1 if self.held_object is not None else 0

    def grasp(self, entity: str, held_offset: Pose | None) -> None:
        self.held_object = entity
        self.held_offset = held_offset

    def release(self) -> str | None:
        held = self.held_object
        self.held_object = None
        self.held_offset = None
        return held

    def clear(self) -> None:
        self.held_object = None
        self.held_offset = None
