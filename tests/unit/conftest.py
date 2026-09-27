"""Unit-test fixtures: a minimal in-memory SimBackend double.

These tests verify the benchmark core (schema, validators, session, grounding,
feasibility, failure branches) WITHOUT the simulator, which also validates
that the backend abstraction is real and OmniGibson is genuinely pluggable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rummagebench.sim.base import (
    EntityInfo,
    ResolvedTarget,
    SimBackend,
)
from rummagebench.core.types import TargetKind


class FakeBackend(SimBackend):
    """Minimal backend double for unit tests.

    entities: name -> EntityInfo; holding: entity currently grasped;
    opens: set of open entities; poses: name -> [x,y,z]; aabbs: name -> (lo,hi).
    """

    def __init__(
        self,
        entities: dict[str, EntityInfo],
        anchors: set[str],
        poses: dict[str, list[float]] | None = None,
        aabbs: dict[str, tuple[list[float], list[float]]] | None = None,
    ):
        self.entities = entities
        self.anchors = anchors
        self.poses = poses or {}
        self.aabbs = aabbs or {}
        self.holding_entity: str | None = None
        self.opens: set[str] = set()
        self._counter = 0

    def setup(self, scenario):
        return {"ok": True}

    def reset(self) -> None:
        self.holding_entity = None
        self.opens = set()

    def get_observation(self) -> Any:
        self._counter += 1
        return np.full((2, 2, 3), self._counter % 255, dtype=np.uint8)

    def resolve_entity(self, name: str) -> ResolvedTarget:
        info = self.entities.get(name)
        if info is None:
            from rummagebench.core.errors import UnresolvableTargetError

            raise UnresolvableTargetError(f"unknown entity {name!r}")
        return ResolvedTarget(kind=TargetKind.ENTITY, entity=name, info=info)

    def teleport_robot(self, anchor) -> None:
        self._last_anchor_position = list(anchor.position)

    def robot_pose(self):
        return getattr(self, "_last_anchor_position", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]

    def set_open(self, entity: str, open_value: bool) -> bool:
        if entity not in self.entities or not self.entities[entity].openable:
            return False
        if open_value:
            self.opens.add(entity)
        else:
            self.opens.discard(entity)
        return True

    def is_open(self, entity: str) -> bool:
        return entity in self.opens

    def symbolic_grasp(self, entity: str) -> bool:
        if entity not in self.entities or not self.entities[entity].graspable:
            return False
        self.holding_entity = entity
        return True

    def is_holding(self, entity: str) -> bool:
        return self.holding_entity == entity

    def settle(self, steps: int = 10) -> None:
        pass

    def dump_state(self) -> Any:
        return {"opens": set(self.opens), "holding": self.holding_entity}

    def load_state(self, state: Any) -> None:
        self.opens = set(state["opens"])
        self.holding_entity = state["holding"]

    def render_snapshot(self, path: str) -> None:
        pass

    def close(self) -> None:
        pass

    # ------------------------------------------------- embodiment grounding

    def entity_names(self) -> list[str]:
        return list(self.entities.keys())

    def entity_pose(self, name: str) -> list[float]:
        return list(self.poses.get(name, [0.0, 0.0, 0.5]))

    def entity_aabb(self, name: str):
        return self.aabbs.get(name)

    def held_count(self) -> int:
        return 1 if self.holding_entity is not None else 0

    def symbolic_place_held(self, receptacle: str) -> bool:
        if self.holding_entity is None:
            return False
        self.holding_entity = None
        return True

    def describe_entity(self, name: str) -> EntityInfo | None:
        return self.entities.get(name)


@pytest.fixture
def fake_backend():
    return FakeBackend(
        entities={
            "target_knife": EntityInfo(name="target_knife", category="knife"),
            "distractor_spoon": EntityInfo(name="distractor_spoon", category="spoon"),
            "cabinet_A": EntityInfo(
                name="cabinet_A", category="cabinet", fixed_base=True, openable=True,
                is_receptacle=True,
            ),
            "cabinet_B": EntityInfo(
                name="cabinet_B", category="cabinet", fixed_base=True, openable=True,
                is_receptacle=True,
            ),
            "drawer_A": EntityInfo(
                name="drawer_A", category="drawer", fixed_base=True, openable=True,
                is_receptacle=True,
            ),
            "countertop": EntityInfo(
                name="countertop", category="countertop", fixed_base=True
            ),
            "hot_pot": EntityInfo(name="hot_pot", category="pot_nonstick"),
        },
        anchors={"living_room", "kitchen"},
        poses={
            "target_knife": [3.05, 1.05, 0.35],
            "distractor_spoon": [3.02, 1.02, 0.30],
            "cabinet_A": [2.30, 2.30, 0.45],
            "cabinet_B": [3.00, 1.00, 0.45],
            "drawer_A": [3.65, 0.50, 0.30],
            "countertop": [2.80, 0.80, 0.60],
            "hot_pot": [3.30, 1.30, 0.90],
        },
        aabbs={
            "cabinet_B": ([2.7, 0.7, 0.0], [3.3, 1.3, 0.9]),
            "cabinet_A": ([2.0, 2.0, 0.0], [2.6, 2.6, 0.9]),
            "drawer_A": ([3.4, 0.2, 0.0], [3.9, 0.8, 0.6]),
            "countertop": ([2.4, 0.4, 0.5], [3.6, 1.6, 0.7]),
        },
    )
