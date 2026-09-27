"""Geometry validator: is the interaction physically admissible?

MVP implementation: point-vs-AABB overlap. The interaction point (handle
pose, grasp pose, place pose) must not be buried inside another entity's
volume. Admissibility semantics (allowed-collision matrix):

    allowed:  interaction with the target entity itself
    allowed:  interaction at the mouth of an OPEN container (openable
              entities that are currently open do not block)
    allowed:  contact with the entity that currently holds/frames the target
              while it is held (hand <-> grasped object)
    forbidden: interaction point inside any CLOSED foreign volume

This answers 'is this interaction admissible', not 'is the trajectory safe'
(that stays with SafetyValidator and future motion-level checkers).
"""

from __future__ import annotations

from rummagebench.core.types import FeasibilityVerdict
from rummagebench.sim.base import SimBackend


class CollisionChecker:
    def __init__(self, backend: SimBackend):
        self._backend = backend

    def check(
        self,
        target_entity: str,
        interaction_pose: list[float],
    ) -> FeasibilityVerdict:
        for name in self._backend.entity_names():
            if name == target_entity:
                continue  # finger <-> target object: allowed
            # relevance filter: skip entities whose center is far from the
            # interaction point (furniture AABB half-extents are bounded)
            center = self._backend.entity_pose(name)
            if sum((a - b) ** 2 for a, b in zip(center, interaction_pose)) > 9.0:
                continue  # > 3 m away: cannot bury the interaction point
            aabb = self._backend.entity_aabb(name)
            if aabb is None:
                continue
            lo, hi = aabb
            if not all(lo[i] <= interaction_pose[i] <= hi[i] for i in range(3)):
                continue
            # an OPEN container's mouth is an admissible interaction volume;
            # a CLOSED one burying the interaction point is a real collision
            if self._backend.is_open(name):
                continue
            return FeasibilityVerdict(
                feasible=False,
                reason="COLLISION",
                details={"blocker": name, "target": target_entity},
            )
        return FeasibilityVerdict(feasible=True)
