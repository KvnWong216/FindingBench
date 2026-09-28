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
from rummagebench.feasibility.ik_solver import Pose
from rummagebench.feasibility.interaction_target import InteractionRegion


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
class ArticulationJointInfo:
    """One joint of an articulated entity (from the underlying articulation).

    limits are (lower, upper) in radians for revolute / meters for prismatic.
    position is the current joint value (None when unavailable).
    """

    name: str
    joint_type: str  # revolute | prismatic | fixed | continuous ...
    parent_link: str | None
    child_link: str | None
    axis: list[float] | None = None
    limits: tuple[float, float] | None = None
    position: float | None = None


@dataclass
class ArticulationInfo:
    """Articulation decomposition of one entity (OPEN/CLOSE grounding input).

    The interaction interface of an articulated object is its moving link /
    handle, never the entity root pose; this is the backend surface that
    exposes joints, links and handle candidates.
    """

    entity: str
    joints: list[ArticulationJointInfo] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    # link name of an explicitly annotated handle, when the asset has one
    handle_link: str | None = None


@dataclass
class WorldCollisionObject:
    """One collision body of the world (backend-provided, geometry-agnostic).

    geometry holds a coal/hpp-fcl CollisionGeometry (box/sphere/cylinder/mesh)
    placed at ``pose`` (world frame). ``approximation`` records the fidelity
    of the geometry: 'urdf_collision' | 'physics_collider' | 'convex_hull' |
    'aabb_primitive'. ``aabb`` is a conservative world-frame bound used only
    for broadphase filtering.
    """

    entity: str
    link: str | None
    geometry: Any
    pose: Pose
    category: str = ""
    approximation: str = "aabb_primitive"
    aabb: tuple[list[float], list[float]] | None = None


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
        """Privileged grasp realization: attach the object to the end effector.

        Realization only (visualization/physics bookkeeping). It MUST NOT move
        the robot base and MUST NOT define benchmark holding state — the
        benchmark-owned state (state/benchmark_state.py) is the semantic truth.
        """

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
    def symbolic_place(self, entity: str, receptacle: str) -> bool:
        """Release ``entity`` (the benchmark-held object) onto/into the
        receptacle (instant transition). The held object is passed explicitly
        by the benchmark core; the backend never re-derives it from an
        internal grasp dict."""

    @abstractmethod
    def describe_entity(self, name: str) -> EntityInfo:
        """Privileged EntityInfo for one entity (grounder input)."""

    # ------------------------------------------------- physical grounding
    # Optional capabilities backing the geometry/kinematics feasibility
    # engine. Defaults degrade gracefully (the proxy test mode never calls
    # them); the pinocchio mode requires them and fails loudly through the
    # validator when a backend cannot supply the data.

    def articulation_info(self, entity: str) -> ArticulationInfo | None:
        """Articulation decomposition (joints/links/handle) for an entity."""
        return None

    def collision_geometries(self) -> list[WorldCollisionObject]:
        """World collision bodies (link-granular when available), with the
        approximation level of each body recorded."""
        return []

    def receptacle_region(self, entity: str) -> InteractionRegion | None:
        """Support region of a receptacle (top surface / inside volume)."""
        return None

    def link_pose(self, entity: str, link: str) -> Pose | None:
        """World-frame pose of one link of an entity."""
        return None

    def link_aabb(self, entity: str, link: str) -> tuple[list[float], list[float]] | None:
        """World-frame AABB of one link, or None when unavailable."""
        return None

    def eef_pose(self) -> Pose | None:
        """Current end-effector pose (world frame), None when unavailable."""
        return None

    def robot_joint_positions(self) -> dict[str, float] | None:
        """Current joint positions by joint name (IK seed), None when
        unavailable (the solver then uses neutral + deterministic restarts)."""
        return None

    def visible_entities(self) -> list[str]:
        """Entities the AGENT may currently be offered skills for.

        Default: everything (backends without a visibility model). Real
        backends must override: closed-container contents are NOT observable
        and must not leak into the agent-facing action space (oracle entity
        grounding stays available for action execution and evaluation).
        """
        return self.entity_names()
