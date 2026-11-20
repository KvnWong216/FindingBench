"""GRASP(target): privileged, symbolic object acquisition.

Validated GRASP execution is an instant symbolic transition:

    feasibility passed -> robot base does NOT move
                       -> standardized realization establishes the attach
                       -> benchmark state becomes held_object = target

The backend's ``symbolic_grasp`` is a realization step (assisted-grasp
joint) and must never relocate the robot: implicit NAV inside GRASP is
forbidden. Commit discipline (skills/realization.py): when the realization
does not establish the held postcondition, the backend is rolled back and an
infrastructure fault is raised — the benchmark state is never committed on a
failed realization, and the step is never reported as EXECUTED.

Rule 4: this skill does NOT decide benchmark success or wrong-target
failure — BenchmarkSession does, from the skill result + benchmark state.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.feasibility.ik_solver import Pose
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.skills.realization import commit_realization


class GraspSkill:
    name = "GRASP"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        entity = resolved.entity

        # standardized realization: attach the assisted-grasp joint.
        # MUST NOT move the robot base (no implicit NAV, ever). Raises
        # FeasibilityBackendError (with backend rollback) when the attach
        # postcondition cannot be established.
        commit_realization(
            backend,
            realize=lambda: backend.symbolic_grasp(entity),
            verify=lambda: backend.is_holding(entity),
            what=f"GRASP({entity})",
        )

        # capture the grasp offset (held object relative to the EEF) so the
        # held body can be included in later configuration-space collision
        # checks; when the backend cannot provide poses, offset stays None
        # and the held body is excluded (logged by the collision checker).
        held_offset: Pose | None = None
        eef_pose = backend.eef_pose()
        obj_pose = backend.entity_pose6d(entity)
        if eef_pose is not None and obj_pose is not None:
            held_offset = eef_pose.inverse().compose(obj_pose)

        # benchmark-owned state transition (semantic truth, committed AFTER
        # the realization landed)
        state.grasp(entity, held_offset)

        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.ENTITY.value, "value": entity},
            executed=True,
            postcondition_satisfied=True,
            events=[
                {
                    "event": "grasp_executed",
                    "entity": entity,
                    "held_state": entity,
                    "realized": True,
                    "base_moved": False,
                }
            ],
            details={
                "held": entity,
                "realized": True,
                "held_offset": held_offset is not None,
            },
        )
