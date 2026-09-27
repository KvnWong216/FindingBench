"""Dynamic skill grounding: regenerate the available action space each step.

    A_t = Ground(Robot, Object, WorldState_t)

For every entity, its adapter proposes semantically valid skill candidates;
the feasibility engine (reach + collision) filters them into the skills that
actually exist for THIS robot at THIS world state. The result feeds the agent
observation (available_skills) and the event log.
"""

from __future__ import annotations

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import TargetKind, WorldState
from rummagebench.feasibility.collision import CollisionChecker
from rummagebench.feasibility.ik_solver import IKSolver
from rummagebench.object_interface.base import build_object_interface
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.validation.feasibility import FeasibilityValidator


class SkillGrounder:
    """Enumerates the skills that exist for the robot right now."""

    def __init__(
        self,
        backend: SimBackend,
        scenario: ScenarioSpec,
        feasibility: FeasibilityValidator | None = None,
        ik_solver: IKSolver | None = None,
        collision_checker: CollisionChecker | None = None,
    ):
        self._backend = backend
        self._scenario = scenario
        if feasibility is None:
            feasibility = FeasibilityValidator(
                backend,
                ik_solver=ik_solver,
                collision_checker=collision_checker,
            )
        self._feasibility = feasibility

    def ground(self, robot: RobotEmbodiment) -> list[str]:
        world = WorldState(held_count=self._backend.held_count())
        labels: list[str] = []

        # NAV: navigation executor is perfect; every verified anchor is a skill
        for anchor in self._scenario.anchors:
            labels.append(f"NAV({anchor})")

        # entity-offered skills, filtered by the feasibility engine
        for name in self._backend.entity_names():
            info = self._backend.describe_entity(name)
            if info is None:
                continue
            obj = build_object_interface(info, self._backend)
            for cand in obj.available_skills(robot, world):
                resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity=name, info=info)
                verdict = self._feasibility.check(cand.skill, resolved, robot, world)
                if verdict.feasible:
                    labels.append(cand.label())

        return sorted(set(labels))
