"""Rigid objects (knife, cup, apple): graspable, no grasp-pose annotation.

Deliberately NOT modeled: grasp quality, finger closure, force closure.
"""

from __future__ import annotations

from rummagebench.core.types import WorldState
from rummagebench.object_interface.base import ObjectInterface, SkillCandidate
from rummagebench.robots.robot import RobotEmbodiment


class RigidObject(ObjectInterface):
    def available_skills(
        self, robot: RobotEmbodiment, world: WorldState
    ) -> list[SkillCandidate]:
        skills: list[SkillCandidate] = []
        if self.info.graspable and world.held_count < robot.hand_capacity:
            skills.append(SkillCandidate(skill="GRASP", target=self.id))
        return skills
