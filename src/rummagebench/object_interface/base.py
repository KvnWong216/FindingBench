"""Object adapter layer: BEHAVIOR assets are never used directly by skills
or the grounder. Each adapter answers 'which skills does this object offer
to THIS robot in THIS world state' — semantic candidacy only; geometric
feasibility is applied afterwards by the feasibility engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rummagebench.core.types import WorldState
from rummagebench.robots.robot import RobotEmbodiment

if TYPE_CHECKING:  # pragma: no cover
    from rummagebench.feasibility.interaction_target import InteractionTarget


@dataclass
class SkillCandidate:
    """A semantically valid skill offered by an object (not yet feasibility-checked)."""

    skill: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.skill}({self.target})"


@dataclass
class ObjectInterface:
    """Base adapter wrapping one simulator entity."""

    info: Any  # EntityInfo
    backend: Any  # SimBackend

    @property
    def id(self) -> str:
        return self.info.name

    @property
    def category(self) -> str:
        return self.info.category

    def available_skills(
        self, robot: RobotEmbodiment, world: WorldState
    ) -> list[SkillCandidate]:
        return []

    def interaction_targets(self, skill_name: str, world: WorldState) -> list["InteractionTarget"]:
        """Interaction interfaces this object offers for one skill.

        Default: no interface (the object does not host this skill). Subclasses
        ground OPEN/CLOSE at articulated links, GRASP at canonical rigid-body
        candidates and PLACE at receptacle regions — never at the entity root
        pose.
        """
        return []


def build_object_interface(info, backend) -> ObjectInterface:
    """Factory: pick the adapter class from entity metadata.

    fixed_base + openable          -> ArticulatedObject (cabinet/drawer/fridge)
    fixed_base + receptacle-holding -> Receptacle (table/countertop)
    otherwise                      -> RigidObject (knife/cup/apple)
    """
    from rummagebench.object_interface.articulated import ArticulatedObject
    from rummagebench.object_interface.receptacle import Receptacle
    from rummagebench.object_interface.rigid import RigidObject

    is_receptacle = bool(getattr(info, "is_receptacle", False))
    if info.fixed_base and info.openable:
        return ArticulatedObject(info, backend, is_receptacle=is_receptacle)
    if info.fixed_base and is_receptacle:
        return Receptacle(info, backend)
    if info.fixed_base:
        # fixed, non-interactive furniture: no skills, but keep the adapter
        # type predictable for the grounder
        return ObjectInterface(info, backend)
    return RigidObject(info, backend)
