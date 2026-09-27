"""GRASP(target): privileged, symbolic object acquisition.

Validated GRASP execution is an instant symbolic transition:

    feasibility passed -> robot base does NOT move
                       -> benchmark state becomes held_object = target

The backend's ``symbolic_grasp`` is a realization/visualization step only
(assisted-grasp joint) and must never relocate the robot: implicit NAV inside
GRASP is forbidden. The realization result is logged but the benchmark state
is the semantic truth (state/benchmark_state.py).

Rule 4: this skill does NOT decide benchmark success or wrong-target
failure — BenchmarkSession does, from the skill result + benchmark state.
"""

from __future__ import annotations

import logging

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.feasibility.ik_solver import Pose
from rummagebench.sim.base import ResolvedTarget, SimBackend

logger = logging.getLogger(__name__)


class GraspSkill:
    name = "GRASP"
    target_kind = TargetKind.ENTITY

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.entity is not None
        entity = resolved.entity

        # realization only: attach the assisted-grasp joint for visualization.
        # MUST NOT move the robot base (no implicit NAV, ever).
        realized = backend.symbolic_grasp(entity)

        # capture the grasp offset (held object relative to the EEF) so the
        # held body can be included in later configuration-space collision
        # checks; when the backend cannot provide poses, offset stays None
        # and the held body is excluded (logged by the collision checker).
        held_offset: Pose | None = None
        eef_pose = backend.eef_pose()
        obj_pose = backend.entity_pose6d(entity)
        if eef_pose is not None and obj_pose is not None:
            held_offset = eef_pose.inverse().compose(obj_pose)

        # benchmark-owned state transition (semantic truth)
        state.grasp(entity, held_offset)
        if not realized:
            logger.warning(
                "GRASP realization for %s failed (assisted-grasp joint); "
                "benchmark state is still authoritative", entity
            )

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
                    "realized": bool(realized),
                    "base_moved": False,
                }
            ],
            details={
                "held": entity,
                "realized": bool(realized),
                "held_offset": held_offset is not None,
            },
        )
