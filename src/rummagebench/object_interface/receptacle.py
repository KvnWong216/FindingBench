"""Receptacles (table, countertop, cabinet interior): PLACE targets."""

from __future__ import annotations

from rummagebench.core.types import WorldState
from rummagebench.object_interface.base import ObjectInterface, SkillCandidate
from rummagebench.robots.robot import RobotEmbodiment


class Receptacle(ObjectInterface):
    def available_skills(
        self, robot: RobotEmbodiment, world: WorldState
    ) -> list[SkillCandidate]:
        skills: list[SkillCandidate] = []
        if world.held_count > 0:
            skills.append(SkillCandidate(skill="PLACE", target=self.id))
        return skills
