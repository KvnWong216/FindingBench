"""Kinematic feasibility toolbox: answer 'CAN this robot perform this
interaction?', never 'HOW does it execute it' (execution stays symbolic).

This module defines the kinematics-facing contracts used by the feasibility
engine:

- ``Pose``                minimal SE(3) primitive (position + xyzw quaternion)
- ``IKResult``            structured IK outcome with fine-grained failure reason
- ``RobotKinematicsBackend``  the real-embodiment protocol (URDF-backed;
  see pinocchio_solver.PinocchioKinematics for the reference implementation)
- ``IKSolver`` / ``ReachabilityIKSolver``  the legacy reach-radius proxy.

DESIGN RULE (fail loudly, never silently degrade):
``ReachabilityIKSolver`` is a TEST/PROXY backend only. It models "the arm can
extend to this pose" as a distance + height-band check and carries no robot
geometry. It must never be the runtime default of the production benchmark:
the production feasibility config requires ``backend: pinocchio`` and a URDF
(see robots/model_loader.py). Silent fallback to the proxy is forbidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rummagebench.robots.robot import RobotEmbodiment

# ---------------------------------------------------------------------------
# SE(3) primitive (numpy-only so it stays importable without pinocchio)
# ---------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw convention. Inputs are NOT normalized (a pure
    vector quaternion [vx,vy,vz,0] must survive the product unchanged)."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=float).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (xyzw)."""
    t = quat_multiply(quat_multiply(q, np.array([*v, 0.0])), quat_conjugate(q))
    return t[:3]


@dataclass
class Pose:
    """Rigid transform T_parent_frame: frame expressed in ``parent``.

    position: (3,) world/parent-frame position; orientation: (4,) xyzw
    quaternion of the frame rotation w.r.t. the parent.
    """

    position: np.ndarray
    orientation: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    @staticmethod
    def from_lists(
        position: list[float] | tuple[float, ...],
        orientation: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0, 1.0),
    ) -> "Pose":
        return Pose(
            np.asarray(position, dtype=float).reshape(3),
            quat_normalize(np.asarray(orientation, dtype=float)),
        )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.orientation = quat_normalize(self.orientation)

    def compose(self, other: "Pose") -> "Pose":
        """T_parent_self . T_self_other = T_parent_other."""
        return Pose(
            self.position + quat_rotate(self.orientation, other.position),
            quat_multiply(self.orientation, other.orientation),
        )

    def inverse(self) -> "Pose":
        inv_q = quat_conjugate(self.orientation)
        return Pose(-quat_rotate(inv_q, self.position), inv_q)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Point expressed in this frame -> expressed in the parent frame."""
        return self.position + quat_rotate(self.orientation, np.asarray(point, dtype=float))

    def apply_inverse(self, point_world: np.ndarray) -> np.ndarray:
        """Point expressed in the parent frame -> expressed in this frame."""
        return quat_rotate(quat_conjugate(self.orientation), np.asarray(point_world) - self.position)

    def to_lists(self) -> tuple[list[float], list[float]]:
        return (
            [float(v) for v in self.position],
            [float(v) for v in self.orientation],
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Pose(pos={[round(float(v), 4) for v in self.position]}, "
            f"quat={[round(float(v), 4) for v in self.orientation]})"
        )


# ---------------------------------------------------------------------------
# IK contracts
# ---------------------------------------------------------------------------


class IKFailureReason:
    """Fine-grained IK failure attribution (benchmark-facing reason is
    mapped to UNREACHABLE later; the internal log keeps the distinction)."""

    NO_IK_SOLUTION = "NO_IK_SOLUTION"
    JOINT_LIMIT = "JOINT_LIMIT"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass
class IKResult:
    """Structured outcome of one IK query."""

    success: bool
    q: np.ndarray | None = None
    reason: str | None = None  # None on success; one of IKFailureReason otherwise
    position_error: float | None = None
    orientation_error: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RobotKinematicsBackend(Protocol):
    """Real-embodiment kinematics (URDF-backed; geometry-derived).

    ``target_pose`` is expressed in the ROBOT BASE frame; the feasibility
    pipeline converts world-frame interaction poses through the current base
    pose before calling. Implementations must be deterministic: identical
    inputs produce identical outputs (random restarts use a fixed seed).
    """

    def solve_ik(
        self,
        target_pose: Pose,
        seed_q: np.ndarray | None = None,
    ) -> IKResult: ...

    def forward_kinematics(
        self,
        q: np.ndarray,
    ) -> dict[str, Pose]: ...


@runtime_checkable
class IKSolver(Protocol):
    """Legacy proxy-solver protocol (test mode only, see module docstring).

    solve() returns a joint configuration dict if the interaction pose is
    reachable, or None if UNREACHABLE. It never plans trajectories.
    """

    def solve(
        self,
        target_pose: list[float],
        robot_pose: tuple[list[float], list[float]],
        robot: RobotEmbodiment,
    ) -> dict[str, Any] | None: ...


class ReachabilityIKSolver:
    """Geometry-proxy IK: reach radius + interaction height band.

    PROXY BACKEND — unit-test / coarse-prefilter use only (prompt rule: it may
    gate nothing in the production benchmark). It models 'the arm can extend
    to this pose' as (distance from base) <= reach_radius and the target
    height inside the robot's interaction band. It knows nothing about link
    geometry, joint limits or collisions.
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
    """Proxy solver for unit tests. Production sessions must build their
    kinematics backend via robots.model_loader (pinocchio + URDF)."""
    return ReachabilityIKSolver()


# ---------------------------------------------------------------------------
# small geometric helpers shared by the feasibility engine
# ---------------------------------------------------------------------------


def look_at_quaternion(direction: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Quaternion (xyzw) whose local +Z axis maps to ``direction`` (world).

    Used to give canonical interaction candidates a deterministic approach
    orientation (tool z-axis = approach normal).
    """
    z = np.asarray(direction, dtype=float)
    n = np.linalg.norm(z)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    z = z / n
    up = np.asarray(up, dtype=float)
    if abs(float(np.dot(z, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[2]) > 0.5 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    x = x / nx
    y = np.cross(z, x)
    rot = np.column_stack([x, y, z])
    # rotation matrix -> xyzw quaternion
    tr = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x_ = (rot[2, 1] - rot[1, 2]) / s
        y_ = (rot[0, 2] - rot[2, 0]) / s
        z_ = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        w = (rot[2, 1] - rot[1, 2]) / s
        x_ = 0.25 * s
        y_ = (rot[0, 1] + rot[1, 0]) / s
        z_ = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        w = (rot[0, 2] - rot[2, 0]) / s
        x_ = (rot[0, 1] + rot[1, 0]) / s
        y_ = 0.25 * s
        z_ = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        w = (rot[1, 0] - rot[0, 1]) / s
        x_ = (rot[0, 2] + rot[2, 0]) / s
        y_ = (rot[1, 2] + rot[2, 1]) / s
        z_ = 0.25 * s
    return quat_normalize(np.array([x_, y_, z_, w]))


def aabb_surface_point(
    lo: list[float] | np.ndarray,
    hi: list[float] | np.ndarray,
    outward: np.ndarray,
) -> np.ndarray:
    """The point on the AABB surface that the unit ``outward`` direction faces.

    Deterministic face anchor used by fallback interaction anchors
    (source='fallback_link_anchor'): project the outward direction onto the
    box and take the corresponding face point.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    d = np.asarray(outward, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return center
    d = d / n
    # scale so that center + t*d hits the box surface (t>0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(d) > 1e-12, half / np.abs(d), np.inf)
    step = float(np.min(t))
    return center + d * step


def aabb_face_candidates(
    lo: list[float] | np.ndarray,
    hi: list[float] | np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Canonical rigid-object interaction candidates: AABB center plus the six
    face centers, each with its outward-normal approach orientation.

    Only used to answer 'does at least one reachable, collision-free
    interaction configuration exist?' — no grasp-quality modelling.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    out: list[tuple[np.ndarray, np.ndarray]] = [(center, np.array([0.0, 0.0, 1.0]))]
    for axis in range(3):
        for sign in (1.0, -1.0):
            pos = center.copy()
            pos[axis] += sign * half[axis]
            normal = np.zeros(3)
            normal[axis] = sign
            out.append((pos, normal))
    return out
