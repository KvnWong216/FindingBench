"""GRASP(target): privileged, symbolic object acquisition.

Validated GRASP execution goes through the backend's symbolic semantic
executor: the object becomes held by the robot; no grasp-pose generation,
IK, motion planning or grasp stability is evaluated. Validation happens
BEFORE execution, in the benchmark core, never inside the skill.

Rule 4: this skill does NOT decide benchmark success or wrong-target
failure — BenchmarkSession does, from the skill result.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


class GraspSkill:
    name = "GRASP"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget) -> SkillResult:
        assert resolved.entity is not None
        acquired = backend.symbolic_grasp(resolved.entity)
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": resolved.entity},
            executed=True,
            postcondition_satisfied=bool(acquired),
            events=[
                {
                    "event": "grasp_executed",
                    "entity": resolved.entity,
                    "acquired": bool(acquired),
                }
            ],
            details={"acquired": bool(acquired)},
        )
