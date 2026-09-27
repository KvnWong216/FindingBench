"""Dynamic skill grounding: regenerate the available action space each step.

    A_t = Ground(RobotGeometry, ObjectInterface, WorldGeometry, WorldState_t)

For every entity, its adapter proposes semantically valid skill candidates;
the feasibility engine (URDF kinematics IK + configuration-space collision +
state preconditions) filters them into the skills that actually exist for
THIS robot at THIS world state. The result feeds the agent observation
(available_skills) and the event log.
"""

from __future__ import annotations

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import TargetKind
from rummagebench.object_interface.base import build_object_interface
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.validation.feasibility import FeasibilityValidator


class SkillGrounder:
    """Enumerates the skills that exist for the robot right now.

    This is the benchmark's core mechanism, the Embodied Action Grounding
    Engine: the admissible action space is not fixed by the skill library but
    derived per step from robot geometry x object interface x world state.
    """

    def __init__(
        self,
        backend: SimBackend,
        scenario: ScenarioSpec,
        feasibility: FeasibilityValidator | None = None,
    ):
        self._backend = backend
        self._scenario = scenario
        if feasibility is None:
            from rummagebench.core.session import _build_feasibility

            feasibility = _build_feasibility(backend, scenario)
        self._feasibility = feasibility

    def ground(self, robot: RobotEmbodiment, world) -> list[str]:
        """``world`` is the benchmark-owned state (BenchmarkWorldState)."""
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


# Paper-facing alias: the grounder is the Embodied Action Grounding Engine
# (EAGE) described in the benchmark design notes.
EmbodiedActionGroundingEngine = SkillGrounder
