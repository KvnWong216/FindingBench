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
