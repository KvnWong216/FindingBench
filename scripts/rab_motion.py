"""Continuous-motion trajectories for FindingBench RAB demo replay (pure numpy).

The benchmark mechanics stay discrete: every skill is validated and realized by
``rab_skills.py`` exactly as before (A* route + clearance rays, Jacobian IK +
hand-path rays, PhysX settle checks).  This module only turns the trajectories
those mechanics ALREADY computed into smooth, time-parameterized robot motion:

* navigation: the A* waypoint list returned by ``Skills.route`` (plus the start
  pose and the final facing yaw) -> rotate / translate / rotate base frames;
* manipulation: the joint-space iterates of the damped-least-squares IK solver
  (``Skills._ik_solve``) -> an arc-length re-timed joint trajectory.

All timing uses minimum-jerk profiles so recorded demos look like a smooth,
ideal low-level controller ("ideal cerebellum").  No simulator imports here, so
everything is unit-testable without Isaac Sim.

Rigid poses are ``[x, y, z, qx, qy, qz, qw]`` (PhysX / scipy xyzw order).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
MIN_JERK_PEAK = 1.875  # peak velocity of a min-jerk profile = 1.875 * D / T


def min_jerk(s: float | np.ndarray) -> float | np.ndarray:
    """Minimum-jerk time scaling 10s^3 - 15s^4 + 6s^5 on [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def duration_for(distance: float, vmax: float, tmin: float = 0.0) -> float:
    """Duration so a min-jerk move of ``distance`` peaks at ``vmax``."""
    if distance <= 0 or vmax <= 0:
        return tmin
    return max(MIN_JERK_PEAK * distance / vmax, tmin)


def frame_count(duration: float, fps: float) -> int:
    return max(1, int(math.ceil(duration * fps)))


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------------------
# Base (navigation) trajectories
# ---------------------------------------------------------------------------
@dataclass
class BaseLimits:
    lin_vmax: float = 0.35      # m/s
    ang_vmax: float = 1.0       # rad/s
    min_segment_s: float = 0.25
    eps_xy: float = 0.01        # m; below this a segment is a pure rotation
    eps_yaw: float = math.radians(1.0)


def _rotate_frames(xyz, yaw0, yaw1, fps, lim: BaseLimits):
    d = wrap_angle(yaw1 - yaw0)
    if abs(d) < lim.eps_yaw:
        return []
    n = frame_count(duration_for(abs(d), lim.ang_vmax, lim.min_segment_s), fps)
    return [(float(xyz[0]), float(xyz[1]), float(xyz[2]), wrap_angle(yaw0 + d * float(min_jerk((i + 1) / n))))
            for i in range(n)]


def _translate_frames(p0, p1, yaw, fps, lim: BaseLimits):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dist = float(np.linalg.norm(p1[:2] - p0[:2]))
    n = frame_count(duration_for(dist, lim.lin_vmax, lim.min_segment_s), fps)
    out = []
    for i in range(n):
        s = float(min_jerk((i + 1) / n))
        p = p0 + (p1 - p0) * s
        out.append((float(p[0]), float(p[1]), float(p[2]), float(yaw)))
    return out


def base_frames(poses: Sequence[Sequence[float]], fps: float,
                limits: BaseLimits | None = None) -> list[tuple[float, float, float, float]]:
    """Frames for a differential-drive style replay of executed base poses.

    ``poses`` is the sequence actually realized by the mechanics:
    ``[(x, y, z, yaw)_start, (x, y, z, yaw)_wp1, ..., (x, y, z, yaw)_final]``.
    Between consecutive poses with a translation the robot first turns to the
    segment heading, drives straight, and the last pose's yaw is reached by an
    in-place turn.  Returns per-frame ``(x, y, z, yaw)`` (start pose excluded).
    """
    lim = limits or BaseLimits()
    if len(poses) < 2:
        return []
    frames: list[tuple[float, float, float, float]] = []
    cur = tuple(float(v) for v in poses[0])
    for nxt in poses[1:]:
        nxt = tuple(float(v) for v in nxt)
        dxy = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        if dxy >= lim.eps_xy:
            heading = math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])
            frames += _rotate_frames(cur[:3], cur[3], heading, fps, lim)
            frames += _translate_frames(cur[:3], nxt[:3], heading, fps, lim)
            cur = (nxt[0], nxt[1], nxt[2], heading)
        frames += _rotate_frames(cur[:3], cur[3], nxt[3], fps, lim)
        cur = nxt
    return frames


# ---------------------------------------------------------------------------
# Joint-space trajectories (re-timed IK iterates)
# ---------------------------------------------------------------------------
def dedupe_path(path: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    path = np.asarray(path, float)
    if len(path) <= 1:
        return path
    keep = [0]
    for i in range(1, len(path)):
        if np.linalg.norm(path[i] - path[keep[-1]]) > tol:
            keep.append(i)
    if keep[-1] != len(path) - 1:
        keep[-1] = len(path) - 1
    return path[keep]


def retime_path(path: Sequence[Sequence[float]], fps: float, *, vmax: float = 0.9,
                tmin: float = 0.4, weights: Sequence[float] | None = None) -> np.ndarray:
    """Arc-length re-time a joint polyline with a min-jerk profile.

    ``path`` rows are full DOF vectors (first row = start configuration,
    last row = IK solution).  ``weights`` masks which DOFs count toward the
    arc length (e.g. arm joints only).  Returns ``(n_frames, dof)`` with the
    first frame strictly after the start and the last frame equal to the goal.
    """
    q = dedupe_path(np.asarray(path, float))
    if len(q) < 2:
        return q[-1:].copy() if len(q) else q
    w = np.ones(q.shape[1]) if weights is None else np.asarray(weights, float)
    seg = np.linalg.norm((q[1:] - q[:-1]) * w, axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = float(cum[-1])
    if total <= 1e-9:
        return q[-1:].copy()
    n = frame_count(duration_for(total, vmax, tmin), fps)
    out = np.empty((n, q.shape[1]))
    for i in range(n):
        s = float(min_jerk((i + 1) / n)) * total
        k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        f = 0.0 if seg[k] <= 1e-12 else (s - cum[k]) / seg[k]
        out[i] = q[k] + (q[k + 1] - q[k]) * min(max(f, 0.0), 1.0)
    out[-1] = q[-1]
    return out


def blend(q0: Sequence[float], q1: Sequence[float], fps: float, duration: float) -> np.ndarray:
    """Min-jerk straight-line blend between two DOF vectors (e.g. gripper close)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = frame_count(duration, fps)
    return np.stack([q0 + (q1 - q0) * float(min_jerk((i + 1) / n)) for i in range(n)])


# ---------------------------------------------------------------------------
# Rigid transforms (xyzw quaternions)
# ---------------------------------------------------------------------------
def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_inv(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, float)
    return np.array([-q[0], -q[1], -q[2], q[3]]) / float(np.dot(q, q))


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    return quat_mul(quat_mul(q, [*v, 0.0]), quat_inv(q))[:3]


def compose(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Pose a * b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.r_[a[:3] + quat_rotate(a[3:], b[:3]), quat_mul(a[3:], b[3:])]


def inverse(a: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, float)
    qi = quat_inv(a[3:])
    return np.r_[-quat_rotate(qi, a[:3]), qi]


def relative(parent: Sequence[float], child: Sequence[float]) -> np.ndarray:
    """child expressed in parent's frame: parent^-1 * child."""
    return compose(inverse(parent), child)


def lerp_pose(a: Sequence[float], b: Sequence[float], s: float) -> np.ndarray:
    """Linear translation + normalized-lerp rotation (adequate for short moves)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    qb = b[3:] if np.dot(a[3:], b[3:]) >= 0 else -b[3:]
    q = a[3:] * (1 - s) + qb * s
    return np.r_[a[:3] + (b[:3] - a[:3]) * s, q / np.linalg.norm(q)]


def quat_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, float)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def rot_error_vec(target_R: np.ndarray, current_R: np.ndarray) -> np.ndarray:
    """World-frame axis-angle rotation error rotating ``current_R`` to ``target_R``.

    Matches the world-frame angular-velocity Jacobian ``jac[:, 3:6, :]``:
    ``exp([e]_x) @ current_R == target_R``.
    """
    R = np.asarray(target_R, float) @ np.asarray(current_R, float).T
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(tr)
    if theta < 1e-6:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    if math.pi - theta < 1e-4:
        # Near-180 deg: extract unit axis from the largest diagonal of (R + I) / 2.
        diag = np.maximum(0.5 * (np.diag(R) + 1.0), 0.0)
        k = int(np.argmax(diag))
        axis = np.zeros(3)
        axis[k] = math.sqrt(diag[k])
        for j in range(3):
            if j != k:
                axis[j] = R[k, j] / (2.0 * axis[k]) if axis[k] > 1e-8 else 0.0
        n = float(np.linalg.norm(axis))
        return (axis / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])) * theta
    s = theta / (2.0 * math.sin(theta))
    return s * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)


# ---------------------------------------------------------------------------
# Geometric 6-DoF grasp / place pose synthesis (calibrated to G1 OmniPicker)
# ---------------------------------------------------------------------------
# Measured from /World/Robot/gripper_{l,r}_center_link in the native G1 USD:
#   * local +Z = approach axis (palm at z = -0.1431 m -> fingertip pinch center z = 0)
#   * local ±X = pinch / finger-opening axis (inner_link at +X, outer_link at -X)
#   * local ±Y = palm-plane normal (Y = Z x X)
#   * outer joint1 revolute range: 0..45 deg (0..0.785 rad), 4-bar linkage span ~0.17*sin(q)
GRIP_JOINT_MIN = 0.01
GRIP_JOINT_MAX_CLOSE = 0.58
GRIP_JOINT_OPEN_MIN = 0.64
GRIP_JOINT_OPEN_MAX = 0.74
GRIP_ZERO_GAP_M = 0.005
GRIP_LINKAGE_SPAN_M = 0.17


def gripper_close_angle(width_m: float) -> float:
    """Calibrated OmniPicker outer joint angle (rad) that hugs an object of ``width_m``."""
    ratio = float(np.clip((float(width_m) - GRIP_ZERO_GAP_M) / GRIP_LINKAGE_SPAN_M, 0.0, 0.99))
    return float(np.clip(math.asin(ratio), GRIP_JOINT_MIN, GRIP_JOINT_MAX_CLOSE))


def gripper_open_angle(close_angle: float) -> float:
    """Pre-grasp / release opening angle (rad), strictly wider than ``close_angle``."""
    return float(np.clip(max(GRIP_JOINT_OPEN_MIN, float(close_angle) + 0.16),
                         GRIP_JOINT_OPEN_MIN, GRIP_JOINT_OPEN_MAX))


def _unit(v: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.asarray(fallback, float)


def gripper_frame(approach: Sequence[float], pinch: Sequence[float],
                  cur_R: np.ndarray | None = None) -> np.ndarray:
    """Right-handed 3x3 world rotation for ``gripper_{l,r}_center_link``.

    Columns are ``[X=±pinch, Y=Z×X, Z=approach]``.  Parallel-jaw grippers are
    180-deg symmetric around ``+Z``; when ``cur_R`` is provided, picks the
    symmetry branch closest to ``cur_R`` to avoid unnecessary wrist flips.
    """
    z = _unit(approach, [0.0, 0.0, -1.0])
    x = np.asarray(pinch, float) - float(np.dot(pinch, z)) * z
    if float(np.linalg.norm(x)) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, ref)
    x = _unit(x, [1.0, 0.0, 0.0])
    y = _unit(np.cross(z, x), [0.0, 1.0, 0.0])
    R_pos = np.column_stack((x, y, z))
    if cur_R is None:
        return R_pos
    R_neg = np.column_stack((-x, -y, z))
    cur_R = np.asarray(cur_R, float)
    return R_pos if float(np.trace(R_pos.T @ cur_R)) >= float(np.trace(R_neg.T @ cur_R)) else R_neg


def _shortest_horizontal_pinch(box: np.ndarray, obj_quat: Sequence[float] | None) -> tuple[np.ndarray, float]:
    """Return ``(unit_pinch_xy, width_m)`` along the object's narrowest horizontal axis."""
    ext = np.maximum(box[1] - box[0], 1e-3)
    candidates: list[np.ndarray] = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    if obj_quat is not None:
        R_obj = quat_to_rotmat(obj_quat)
        for col in range(3):
            h = np.array([R_obj[0, col], R_obj[1, col], 0.0], float)
            if float(np.linalg.norm(h)) > 0.5:
                candidates.append(_unit(h, [1.0, 0.0, 0.0]))
    # Project AABB half-extents (upper bound) onto each candidate axis; when OBB axes
    # are aligned with AABB, this matches exact width.
    best_axis, best_w = candidates[0], float("inf")
    for ax in candidates:
        w = float(abs(ax[0]) * ext[0] + abs(ax[1]) * ext[1])
        if w < best_w:
            best_axis, best_w = ax, w
    return best_axis, best_w


@dataclass
class GraspPlan:
    mode: str                  # "top_down" or "side"
    grasp_pos: np.ndarray      # (3,) fingertip center at grasp
    pre_pos: np.ndarray        # (3,) approach hover point along -approach
    lift_pos: np.ndarray       # (3,) post-grasp vertical clearance point
    rotmat: np.ndarray         # (3, 3) target gripper_center_link rotation
    pinch_width_m: float       # object extent along pinch axis
    q_open: float              # pre-grasp gripper joint angle (rad)
    q_close: float             # closed gripper joint angle hugging object (rad)


def plan_grasp_pose(box: Sequence[Sequence[float]], obj_pose: Sequence[float],
                    base_xyz: Sequence[float], base_yaw: float, *,
                    in_open_drawer: bool = False, cur_R: np.ndarray | None = None,
                    force_mode: str | None = None) -> GraspPlan:
    """Synthesize a physically faithful 6-DoF grasp plan for G1 OmniPicker."""
    box = np.asarray(box, float).reshape(2, 3)
    ext = np.maximum(box[1] - box[0], 1e-3)
    center = box.mean(axis=0)
    dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])
    obj_quat = obj_pose[3:7] if obj_pose is not None and len(obj_pose) >= 7 else None

    if force_mode is not None:
        mode = force_mode
    elif in_open_drawer or dz < 0.085 or (dz < 0.11 and max(dx, dy) > 1.6 * dz):
        mode = "top_down"
    else:
        mode = "side"

    if mode == "top_down":
        approach = np.array([0.0, 0.0, -1.0])
        pinch, width = _shortest_horizontal_pinch(box, obj_quat)
        gz = float(np.clip(center[2], box[0, 2] + 0.015, box[1, 2] - 0.008))
        grasp_pos = np.array([center[0], center[1], gz], float)
        clearance = 0.14 if in_open_drawer else 0.10
        pre_pos = grasp_pos + np.array([0.0, 0.0, clearance])
        lift_pos = grasp_pos + np.array([0.0, 0.0, clearance + 0.02])
    else:
        base_xy = np.asarray(base_xyz, float)[:2]
        to_obj = np.array([center[0] - base_xy[0], center[1] - base_xy[1], 0.0], float)
        if float(np.linalg.norm(to_obj)) < 1e-4:
            to_obj = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0], float)
        u = _unit(to_obj, [1.0, 0.0, 0.0])
        # Natural ~12 deg downward pitch keeps forearm above tabletops and under shelves.
        approach = _unit(u + np.array([0.0, 0.0, -0.22]), u)
        pinch = _unit(np.array([-u[1], u[0], 0.0]), [0.0, 1.0, 0.0])
        width = float(abs(pinch[0]) * dx + abs(pinch[1]) * dy)
        grasp_pos = np.array([center[0], center[1], box[0, 2] + 0.55 * dz], float)
        pre_pos = grasp_pos - 0.11 * approach + np.array([0.0, 0.0, 0.02])
        lift_pos = grasp_pos + np.array([0.0, 0.0, 0.10])

    R = gripper_frame(approach, pinch, cur_R=cur_R)
    q_close = gripper_close_angle(width)
    q_open = gripper_open_angle(q_close)
    return GraspPlan(mode=mode, grasp_pos=grasp_pos, pre_pos=pre_pos, lift_pos=lift_pos,
                     rotmat=R, pinch_width_m=width, q_open=q_open, q_close=q_close)


@dataclass
class PlacePlan:
    mode: str
    pre_pos: np.ndarray        # (3,) above target spot before vertical descent
    place_pos: np.ndarray      # (3,) fingertip center at object release
    lift_pos: np.ndarray       # (3,) vertical clearance after opening gripper
    rotmat: np.ndarray         # (3, 3) target gripper_center_link rotation
    q_open: float
    q_close: float


def plan_place_pose(box: Sequence[Sequence[float]], point: Sequence[float],
                    base_xyz: Sequence[float], base_yaw: float, *,
                    cur_R: np.ndarray | None = None,
                    preferred_mode: str | None = None) -> PlacePlan:
    """Synthesize a level 6-DoF placement plan (pre-place -> vertical down -> open -> lift)."""
    box = np.asarray(box, float).reshape(2, 3)
    ext = np.maximum(box[1] - box[0], 1e-3)
    dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])
    pt = np.asarray(point, float)
    mode = preferred_mode or ("top_down" if dz < 0.085 else "side")

    base_xy = np.asarray(base_xyz, float)[:2]
    to_pt = np.array([pt[0] - base_xy[0], pt[1] - base_xy[1], 0.0], float)
    if float(np.linalg.norm(to_pt)) < 1e-4:
        to_pt = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0], float)
    u = _unit(to_pt, [1.0, 0.0, 0.0])
    pinch = _unit(np.array([-u[1], u[0], 0.0]), [0.0, 1.0, 0.0])
    width = float(min(dx, dy)) if mode == "top_down" else float(abs(pinch[0]) * dx + abs(pinch[1]) * dy)

    if mode == "top_down":
        approach = np.array([0.0, 0.0, -1.0])
        place_pos = np.array([pt[0], pt[1], pt[2] + 0.008 + max(0.015, 0.5 * dz)], float)
    else:
        approach = _unit(u + np.array([0.0, 0.0, -0.22]), u)
        place_pos = np.array([pt[0], pt[1], pt[2] + 0.008 + 0.55 * dz], float)

    pre_pos = place_pos + np.array([0.0, 0.0, 0.09])
    lift_pos = place_pos + np.array([0.0, 0.0, 0.11]) if mode == "top_down" else place_pos - 0.09 * approach + np.array([0.0, 0.0, 0.04])
    R = gripper_frame(approach, pinch, cur_R=cur_R)
    q_close = gripper_close_angle(width)
    q_open = gripper_open_angle(q_close)
    return PlacePlan(mode=mode, pre_pos=pre_pos, place_pos=place_pos, lift_pos=lift_pos,
                     rotmat=R, q_open=q_open, q_close=q_close)


# ---------------------------------------------------------------------------
# Demo cameras
# ---------------------------------------------------------------------------
def chase_camera(base_xyz: Sequence[float], base_yaw: float, *, back: float = 2.3,
                 height: float = 1.85, side: float = 0.35, pitch_deg: float = -27.0
                 ) -> tuple[list[float], float, float]:
    """Third-person camera behind/above the robot: (position, yaw_deg, pitch_deg)."""
    fx, fy = math.cos(base_yaw), math.sin(base_yaw)
    sx, sy = -fy, fx
    pos = [float(base_xyz[0]) - back * fx + side * sx,
           float(base_xyz[1]) - back * fy + side * sy,
           float(base_xyz[2]) + height]
    return pos, math.degrees(base_yaw), pitch_deg


# ---------------------------------------------------------------------------
# Recorder: frames -> mp4 (or PNG fallback) + continuous-control log
# ---------------------------------------------------------------------------
def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [round(float(x), 6) for x in v.ravel()]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclass
class DemoRecorder:
    """Writes one continuous demo per episode.

    Layout under ``root``::

        episode_<camera>.mp4      (or frames/<camera>/%06d.png without imageio)
        trajectory.jsonl          one row per rendered frame (continuous control)
        segments.jsonl            one row per discrete decision -> frame range
        meta.json                 fps, cameras, dof names, schema
    """
    root: Path
    fps: float = 20.0
    cameras: tuple[str, ...] = ("chase", "head")
    dof_names: Sequence[str] = ()
    frame: int = 0
    _writers: dict = field(default_factory=dict)
    _traj: Any = None
    _segments: Any = None
    _segment: dict | None = None
    _png: bool = False

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio  # noqa: F401
            self._imageio = imageio
        except Exception:
            self._imageio, self._png = None, True
        self._traj = (self.root / "trajectory.jsonl").open("a", encoding="utf-8")
        self._segments = (self.root / "segments.jsonl").open("a", encoding="utf-8")
        (self.root / "meta.json").write_text(json.dumps(dict(
            schema="findingbench.rab_demo.v1", fps=self.fps, cameras=list(self.cameras),
            dof_names=list(self.dof_names), video="png" if self._png else "mp4",
            trajectory_fields=["frame", "t", "action_index", "skill", "phase", "base_xyz_yaw",
                               "q", "attached_body", "attached_pose", "drive_targets"]),
            indent=2) + "\n")

    # -- segments ---------------------------------------------------------
    def begin(self, action_index: int, skill: str, raw: dict, code: str) -> None:
        self._segment = dict(action_index=action_index, skill=skill, raw=raw, code=code,
                             frame_start=self.frame)

    def end(self, **extra: Any) -> None:
        if self._segment is None:
            return
        self._segment.update(frame_end=self.frame, frames=self.frame - self._segment["frame_start"], **extra)
        self._segments.write(json.dumps(_jsonable(self._segment), ensure_ascii=False) + "\n")
        self._segments.flush()
        self._segment = None

    # -- frames -----------------------------------------------------------
    def _write_image(self, cam: str, rgb: np.ndarray) -> None:
        if self._png:
            from PIL import Image
            d = self.root / "frames" / cam
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(d / f"{self.frame:06d}.png")
            return
        w = self._writers.get(cam)
        if w is None:
            w = self._imageio.get_writer(str(self.root / f"episode_{cam}.mp4"), fps=self.fps,
                                         codec="libx264", quality=8, macro_block_size=8)
            self._writers[cam] = w
        w.append_data(rgb)

    def add(self, images: dict[str, np.ndarray], state: dict[str, Any]) -> None:
        for cam in self.cameras:
            if cam in images:
                self._write_image(cam, images[cam])
        seg = self._segment or {}
        row = dict(frame=self.frame, t=self.frame / self.fps, action_index=seg.get("action_index"),
                   skill=seg.get("skill"), **state)
        self._traj.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        self.frame += 1

    def flush(self) -> None:
        self._traj.flush()

    def close(self) -> None:
        self.end()
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        for f in (self._traj, self._segments):
            if f and not f.closed:
                f.close()
