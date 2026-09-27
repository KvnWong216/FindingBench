"""Receptacles (table, countertop, cabinet interior): PLACE targets.

PLACE grounding comes from the receptacle support region (top surface /
inside volume), never the receptacle root origin.
"""

from __future__ import annotations

from rummagebench.core.types import WorldState
from rummagebench.feasibility.interaction_target import receptacle_place_targets
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

    def interaction_targets(self, skill_name: str, world: WorldState):
        if skill_name == "PLACE":
            return receptacle_place_targets(self.id, self.backend)
        return super().interaction_targets(skill_name, world)
