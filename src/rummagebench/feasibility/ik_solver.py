"""Kinematic feasibility: answer 'CAN this robot perform this interaction?',
never 'HOW does it execute it' (execution stays symbolic/instant).

ik_solver.py defines the solver interface. The default backend is a
reachability proxy (reach radius + height band) which needs no external
dependencies and works for any robot we can teleport. For URDF-defined
robots (fetch, franka_mobile, spot_arm), a TRAC-IK/Pinocchio backend can be
plugged in behind the same IKSolver protocol without benchmark-core changes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rummagebench.robots.robot import RobotEmbodiment


@runtime_checkable
class IKSolver(Protocol):
    """solve() returns a joint configuration dict if the interaction pose is
    reachable, or None if UNREACHABLE. It never plans trajectories."""

    def solve(
        self,
        target_pose: list[float],
        robot_pose: tuple[list[float], list[float]],
        robot: RobotEmbodiment,
    ) -> dict[str, Any] | None: ...


class ReachabilityIKSolver:
    """Geometry-proxy IK: reach radius + interaction height band.

    Deliberately simple: it models 'the arm can extend to this pose' as
    (horizontal+vertical distance from base) <= reach_radius and the target
    height inside the robot's interaction band. Good enough to make the
    action space embodiment-aware; replace with TRAC-IK for real URDFs.
    """

    def solve(
        self,
        target_pose: list[float],
        robot_pose: tuple[list[float], list[float]],
        robot: RobotEmbodiment,
    ) -> dict[str, Any] | None:
        base = robot_pose[0]
        dist = (
            sum((a - b) ** 2 for a, b in zip(base, target_pose)) ** 0.5
        )
        if dist > robot.reach_radius:
            return None
        if not (robot.z_min <= target_pose[2] <= robot.z_max):
            return None
        return {"q": None, "distance": round(dist, 4)}


def default_solver() -> IKSolver:
    return ReachabilityIKSolver()
