"""Abstract simulator backend.

Only OmniGibson may import omnigibson, and it may only do so under
rummagebench/sim/omnigibson/. The rest of the benchmark talks to this
interface so later backends (mock, Habitat, real robot) plug in without
touching benchmark logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from rummagebench.core.scenario import AnchorSpec, ScenarioSpec
from rummagebench.core.types import TargetKind


@dataclass
class EntityInfo:
    """Privileged metadata the validators may use but the agent never sees."""

    name: str
    category: str
    model: str | None = None
    fixed_base: bool = False
    openable: bool = False
    graspable: bool = True
    is_receptacle: bool = False
    in_rooms: list[str] = field(default_factory=list)


@dataclass
class ResolvedTarget:
    """The grounding result for one action target."""

    kind: TargetKind
    place: str | None = None  # anchor id for PLACE targets
    entity: str | None = None  # entity name for ENTITY targets
    info: EntityInfo | None = None
    anchor: Any = None  # AnchorSpec for PLACE targets (kept Any to avoid an import cycle)


class SimBackend(ABC):
    """The minimal simulator surface the MVP needs. Do not grow speculatively."""

    @abstractmethod
    def setup(self, scenario: ScenarioSpec) -> dict[str, Any]:
        """Load the scene, spawn scenario objects, place them, force initial states,
        settle physics, and return a build report dict. Called once per process."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the deterministic initial snapshot captured after setup."""

    @abstractmethod
    def get_observation(self) -> Any:
        """Return the current head-camera RGB as an HxWx3 uint8 array."""

    @abstractmethod
    def resolve_entity(self, name: str) -> ResolvedTarget:
        """Ground an entity name to a ResolvedTarget, or raise UnresolvableTargetError."""

    @abstractmethod
    def teleport_robot(self, anchor: AnchorSpec) -> None:
        """Move the robot base to the anchor pose (semantic navigation)."""

    @abstractmethod
    def robot_pose(self) -> tuple[list[float], list[float]]:
        """Current robot (position, orientation) in world frame."""

    @abstractmethod
    def set_open(self, entity: str, open_value: bool) -> bool:
        """Symbolically set the Open state. Returns the achieved value."""

    @abstractmethod
    def is_open(self, entity: str) -> bool:
        """Whether the entity is currently open."""

    @abstractmethod
    def symbolic_grasp(self, entity: str) -> bool:
        """Privileged grasp: attach the object to the robot end-effector."""

    @abstractmethod
    def is_holding(self, entity: str) -> bool:
        """Whether the robot currently holds the entity."""

    @abstractmethod
    def settle(self, steps: int = 10) -> None:
        """Advance the simulation so poses/contacts become consistent."""

    @abstractmethod
    def dump_state(self) -> Any:
        """Capture a full simulator state snapshot."""

    @abstractmethod
    def load_state(self, state: Any) -> None:
        """Restore a snapshot captured by dump_state (deterministic reset)."""

    @abstractmethod
    def render_snapshot(self, path: str) -> None:
        """Save the current head-camera RGB as an image file (authoring previews)."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

    # ------------------------------------------------- embodiment grounding
    # Minimal surface for the dynamic skill grounder / feasibility engine.

    @abstractmethod
    def entity_names(self) -> list[str]:
        """Names of all interactable scene entities (robots excluded)."""

    @abstractmethod
    def entity_pose(self, name: str) -> list[float]:
        """World-frame position of the entity (interaction pose proxy)."""

    @abstractmethod
    def entity_aabb(self, name: str) -> tuple[list[float], list[float]] | None:
        """World-frame AABB (lo, hi) or None if unavailable."""

    @abstractmethod
    def held_count(self) -> int:
        """Number of objects currently grasped by the robot."""

    @abstractmethod
    def symbolic_place_held(self, receptacle: str) -> bool:
        """Release the held entity onto/into the receptacle (instant transition)."""

    @abstractmethod
    def describe_entity(self, name: str) -> EntityInfo:
        """Privileged EntityInfo for one entity (grounder input)."""
