"""Collision checking for the feasibility engine.

Two layers live here:

1. ``AllowedCollisionMatrix`` / ``CollisionPair`` / ``CollisionResult`` — the
   vocabulary of configuration-space collision checking (see
   hpp_fcl_checker.PinocchioCollisionChecker for the q-based engine: robot
   configuration -> robot collision links x world collision geometry ->
   pairwise coal queries, with self-collision and robot-world separation).

2. ``CollisionChecker`` — the legacy interaction-point-vs-AABB checker.
   PROXY MODE ONLY (unit tests / coarse prefilter): it inspects the single
   interaction point, not the robot configuration, and cannot see link
   geometry. The production benchmark must use the configuration-space
   checker; the proxy checker may never silently replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rummagebench.core.types import FeasibilityVerdict
from rummagebench.sim.base import SimBackend


# ---------------------------------------------------------------------------
# structured results + allowed collision matrix (configuration-space layer)
# ---------------------------------------------------------------------------


@dataclass
class CollisionPair:
    """One colliding pair, attributed for failure reporting."""

    robot_link: str | None  # None when both sides are world/self-other side
    other: str  # world 'entity[:link]' or robot link (self-collision)
    kind: str  # 'self' | 'world'
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"robot_link": self.robot_link, "other": self.other,
                "kind": self.kind, "distance": self.distance}


@dataclass
class CollisionResult:
    """Outcome of one configuration-space collision query."""

    collision_free: bool
    self_collision: bool
    world_collision: bool
    pairs: list[CollisionPair] = field(default_factory=list)
    min_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collision_free": self.collision_free,
            "self_collision": self.self_collision,
            "world_collision": self.world_collision,
            "pairs": [p.to_dict() for p in self.pairs],
            "min_distance": self.min_distance,
        }


@dataclass
class InteractionCollisionContext:
    """Who may touch what for THIS interaction (feeds the ACM).

    skill: GRASP | OPEN | CLOSE | PLACE
    target_entity: the entity the skill acts on
    interaction_link: the articulated link of the target the tool may touch
      (e.g. drawer front); None -> any link of the target entity is allowed
      for gripper-class contacts
    held_entity: object currently held (travels with the end effector)
    held_offset: held-object pose relative to the end effector, captured at
      grasp time (Pose, EEF frame); None -> the held body is not modelled
    base_pose: robot base pose in the world frame (world<->base conversion)
    """

    skill: str
    target_entity: str | None = None
    interaction_link: str | None = None
    held_entity: str | None = None
    held_offset: Any = None  # feasibility.ik_solver.Pose | None
    base_pose: Any = None  # feasibility.ik_solver.Pose


class AllowedCollisionMatrix:
    """Which robot links may touch which world bodies for one skill.

    Rules (prompt §9):

    GRASP:
        gripper/finger <-> target object  = allowed
        arm            <-> target object  = forbidden
        robot          <-> foreign object = forbidden
    OPEN/CLOSE:
        gripper <-> interaction link     = allowed
        arm     <-> cabinet body         = forbidden
        robot   <-> foreign object       = forbidden
    PLACE:
        held object <-> target receptacle = allowed
        gripper     <-> target receptacle = allowed (setting the object down
                                                implies gripper proximity)
        robot arm   <-> receptacle body   = forbidden

    Physical safety (self-collision, illegal contact) is NOT relaxed here:
    the ACM only encodes task-admissible contact.
    """

    def __init__(self, link_classes: dict[str, str] | None = None):
        # link class map: robot link name -> 'gripper' | 'arm'
        self._link_classes = dict(link_classes or {})

    def set_link_classes(self, link_classes: dict[str, str]) -> None:
        self._link_classes = dict(link_classes)

    def link_class(self, robot_link: str) -> str:
        if robot_link in self._link_classes:
            return self._link_classes[robot_link]
        lowered = robot_link.lower()
        if any(h in lowered for h in ("finger", "gripper", "hand")):
            return "gripper"
        return "arm"

    def is_allowed(
        self,
        robot_link: str,
        world_entity: str,
        skill: str,
        target_entity: str | None,
        interaction_link: str | None = None,
        world_link: str | None = None,
        held_entity: str | None = None,
    ) -> bool:
        # the held object travels with the tool: touching it is unavoidable
        if held_entity is not None and world_entity == held_entity:
            return True

        is_target = world_entity == target_entity
        gripper = self.link_class(robot_link) == "gripper"

        if skill == "GRASP":
            return is_target and gripper

        if skill in ("OPEN", "CLOSE"):
            if not is_target:
                return False
            if gripper:
                # contact admissible on the interaction interface itself; when
                # no link decomposition is available the whole entity counts
                # as the interface (approximation level is logged upstream)
                return interaction_link is None or world_link in (None, interaction_link)
            return False

        if skill == "PLACE":
            if not is_target:
                return False
            return gripper or held_entity is not None

        return False


# ---------------------------------------------------------------------------
# legacy point-vs-AABB checker (proxy mode only)
# ---------------------------------------------------------------------------


class CollisionChecker:
    """PROXY geometry validator: interaction point vs foreign AABBs.

    Admissibility semantics (proxy allowed-collision matrix):

        allowed:  interaction with the target entity itself
        allowed:  interaction at the mouth of an OPEN container
        forbidden: interaction point inside any CLOSED foreign volume

    It answers 'is this single point admissible' — NOT 'is the robot
    configuration collision-free'. Unit-test / coarse-prefilter use only.
    """

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
