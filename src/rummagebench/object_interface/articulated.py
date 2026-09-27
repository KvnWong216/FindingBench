"""Articulated objects (cabinet, drawer, fridge): OPEN/CLOSE by joint state.

Grounded at the articulated interaction interface (handle link or the
auto-derived moving-link anchor), never at the cabinet root origin.
"""

from __future__ import annotations

from rummagebench.core.types import WorldState
from rummagebench.feasibility.interaction_target import (
    articulated_interaction_targets,
)
from rummagebench.object_interface.base import ObjectInterface, SkillCandidate
from rummagebench.robots.robot import RobotEmbodiment


class ArticulatedObject(ObjectInterface):
    def __init__(self, info, backend, is_receptacle: bool = False):
        super().__init__(info, backend)
        self.is_receptacle = is_receptacle

    def available_skills(
        self, robot: RobotEmbodiment, world: WorldState
    ) -> list[SkillCandidate]:
        skills: list[SkillCandidate] = []
        if self.backend.is_open(self.id):
            skills.append(SkillCandidate(skill="CLOSE", target=self.id))
        else:
            skills.append(SkillCandidate(skill="OPEN", target=self.id))
        if self.is_receptacle and world.held_count > 0:
            skills.append(
                SkillCandidate(
                    skill="PLACE",
                    target=self.id,
                    params={"held": True},
                )
            )
        return skills

    def interaction_targets(self, skill_name: str, world: WorldState):
        if skill_name in ("OPEN", "CLOSE"):
            return articulated_interaction_targets(self.id, self.backend)
        return super().interaction_targets(skill_name, world)
