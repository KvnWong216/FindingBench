"""Feasibility validator: the geometry/state gate between an agent decision
and its instant symbolic execution.

Answers (in order): is the interaction state-valid (hand free, container
closed/open as required), is the interaction pose reachable by THIS robot,
and is it geometrically admissible (collision). Infeasible actions are
structured non-terminal failures (FAIL_UNREACHABLE / FAIL_COLLISION /
FAIL_INVALID_STATE) that consume one planning step.
"""

from __future__ import annotations

from rummagebench.core.types import (
    FeasibilityVerdict,
    WorldState,
)
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.feasibility.collision import CollisionChecker
from rummagebench.feasibility.ik_solver import IKSolver


class FeasibilityValidator:
    def __init__(
        self,
        backend: SimBackend,
        ik_solver: IKSolver | None = None,
        collision_checker: CollisionChecker | None = None,
    ):
        self._backend = backend
        self._ik = ik_solver
        self._collision = collision_checker

    def _reach(self, pose: list[float], robot: RobotEmbodiment) -> FeasibilityVerdict:
        if self._ik is None:
            return FeasibilityVerdict(feasible=True)
        q = self._ik.solve(pose, self._backend.robot_pose(), robot)
        if q is None:
            return FeasibilityVerdict(
                feasible=False, reason="UNREACHABLE", details={"pose": pose}
            )
        return FeasibilityVerdict(feasible=True, details=q)

    def _collision(self, entity: str, pose: list[float]) -> FeasibilityVerdict:
        if self._collision is None:
            return FeasibilityVerdict(feasible=True)
        return self._collision.check(entity, pose)

    def check(
        self,
        skill_name: str,
        resolved: ResolvedTarget,
        robot: RobotEmbodiment,
        world: WorldState,
    ) -> FeasibilityVerdict:
        if skill_name == "NAV":
            # navigation executor is perfect; anchors are verified at build time
            return FeasibilityVerdict(feasible=True)

        assert resolved.entity is not None
        info = resolved.info
        pose = self._backend.entity_pose(resolved.entity)

        # state preconditions (FAIL_INVALID_STATE)
        if skill_name == "OPEN" and (info is None or not info.openable or self._backend.is_open(resolved.entity)):
            return FeasibilityVerdict(
                feasible=False, reason="INVALID_STATE", details={"entity": resolved.entity}
            )
        if skill_name == "CLOSE" and (info is None or not info.openable or not self._backend.is_open(resolved.entity)):
            return FeasibilityVerdict(
                feasible=False, reason="INVALID_STATE", details={"entity": resolved.entity}
            )
        if skill_name == "GRASP":
            if info is None or info.fixed_base or world.held_count >= robot.hand_capacity:
                return FeasibilityVerdict(
                    feasible=False,
                    reason="INVALID_STATE",
                    details={"entity": resolved.entity, "held": world.held_count},
                )
        if skill_name == "PLACE" and world.held_count == 0:
            return FeasibilityVerdict(
                feasible=False, reason="INVALID_STATE", details={"hand": "empty"}
            )

        # geometric preconditions (FAIL_UNREACHABLE then FAIL_COLLISION)
        reach = self._reach(pose, robot)
        if not reach.feasible:
            return reach
        return self._collision.check(resolved.entity, pose)
