"""Visual replay of executed FindingBench RAB skills as continuous robot motion.

Enabled with ``FINDINGBENCH_RAB_MOTION=replay`` (default ``discrete``).

Contract ("ideal cerebellum", mechanics unchanged)
--------------------------------------------------
1. ``snapshot()`` is taken right before a physical skill (MOVE_TO, TURN, OPEN,
   CLOSE, GRASP, PLACE) runs.  Snapshots only READ simulator state.
2. The skill runs exactly as in discrete mode (A* + clearance rays, DLS IK +
   hand-path rays, PhysX settle).  If it is refused, nothing is animated and the
   failure is returned unchanged.
3. Only after ``EXECUTED``: the authoritative post-state is snapshotted, the
   pre-state is restored, the trajectory the mechanics already computed
   (``Skills.motion``: A* waypoints / IK iterates / drawer drive targets) is
   played back kinematically while frames are recorded, and finally the
   post-state is restored.  The replay therefore cannot change feedback,
   episode status, python-side game state, or (up to the reported
   ``restore_max_err_m``) the physical world.
4. Any exception inside the replay is caught and logged in the private action
   record; it never turns into an agent-visible failure.

Outputs (evaluator-private, under ``<run_output>/demo``): ``episode_chase.mp4``,
``episode_head.mp4``, ``trajectory.jsonl`` (per-frame continuous control:
base pose, full DOF vector, attached object, drive targets), ``segments.jsonl``
(discrete decision -> frame range) and ``meta.json``.

Env knobs: ``FINDINGBENCH_RAB_DEMO_FPS`` (20), ``FINDINGBENCH_RAB_DEMO_CAMERAS``
(``chase,head``), ``FINDINGBENCH_RAB_DEMO_HOLD_S`` (0.3).
"""
from __future__ import annotations

import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from rab_edit import ROBOT
from rab_motion import (BaseLimits, DemoRecorder, base_frames, blend, chase_camera, compose,
                        frame_count, lerp_pose, min_jerk, relative, retime_path)

PHYSICAL = {"MOVE_TO", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE"}
GRIP_OPEN, GRIP_CLOSED = .64, .01


class ReplayDirector:
    def __init__(self, skills, out: Path | None):
        self.sk = skills
        self.out = Path(out) if out else None
        self.fps = float(os.environ.get("FINDINGBENCH_RAB_DEMO_FPS", 20))
        self.cameras = tuple(c.strip() for c in os.environ.get(
            "FINDINGBENCH_RAB_DEMO_CAMERAS", "chase,head").split(",") if c.strip())
        self.hold_s = float(os.environ.get("FINDINGBENCH_RAB_DEMO_HOLD_S", .3))
        self.rec: DemoRecorder | None = None
        self._touched_gravity: set[str] = set()
        self._phase = ""
        self._side = "r"
        self._attached: tuple[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ io
    @property
    def world(self):
        return self.sk.world

    @property
    def robot(self):
        return self.sk.world.mobile.view

    def _recorder(self) -> DemoRecorder | None:
        if self.out is None:
            return None
        if self.rec is None:
            self.rec = DemoRecorder(self.out, fps=self.fps, cameras=self.cameras, dof_names=list(self.sk.names))
        return self.rec

    def close(self) -> None:
        if self.rec is not None:
            self.rec.close()
            self.rec = None

    # ------------------------------------------------------------ snapshot
    def _row_bodies(self) -> list[str]:
        return [r["body"] for r in self.world.rows]

    def _gravity(self, body: str) -> bool | None:
        from pxr import PhysxSchema
        prim = self.world.stage.GetPrimAtPath(body)
        if not prim or not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            return None
        attr = PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr()
        return bool(attr.Get()) if attr and attr.HasAuthoredValue() else None

    def _state(self) -> dict[str, Any]:
        w = self.world
        poses = {}
        for body in self._row_bodies() + [d[2] for d in w.drives.values()]:
            try:
                poses[body] = w.pose(body)
            except ValueError:
                pass
        p, yaw = self.sk.base()
        return dict(base=(p.copy(), float(yaw)),
                    q=self.robot.get_dof_positions().copy(),
                    t=self.robot.get_dof_position_targets().copy(),
                    poses=poses,
                    gravity={b: self._gravity(b) for b in self._row_bodies()},
                    drives={k: float(d[0].GetTargetPositionAttr().Get() or 0.) for k, d in w.drives.items()})

    def snapshot(self, raw: Any) -> dict[str, Any] | None:
        skill = raw.get("skill") if isinstance(raw, dict) else None
        if skill not in PHYSICAL:
            return None
        try:
            return self._state()
        except Exception:
            traceback.print_exc()
            return None

    def _set_gravity(self, body: str, disabled: bool) -> None:
        from pxr import PhysxSchema
        prim = self.world.stage.GetPrimAtPath(body)
        if prim:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(bool(disabled))

    def _apply_poses(self, poses: dict[str, np.ndarray]) -> None:
        import omni.physics.tensors
        view = omni.physics.tensors.create_simulation_view("numpy", self.world.sid.ToLongInt())
        idx = np.array([0], np.uint32)
        for body, pose in poses.items():
            actor = view.create_rigid_body_view(body)
            if actor.count != 1:
                continue
            actor.set_transforms(np.asarray([pose], np.float32), idx)
            actor.set_velocities(np.zeros((1, 6), np.float32), idx)

    def _set_robot(self, q: np.ndarray) -> None:
        idx = np.array([0], np.uint32)
        q = np.asarray(q, np.float32).reshape(1, -1)
        self.robot.set_dof_positions(q, idx)
        self.robot.set_dof_velocities(np.zeros_like(q), idx)
        self.robot.set_dof_position_targets(q, idx)

    def _restore(self, st: dict[str, Any]) -> float:
        """Restore a snapshot; returns the max body translation error (m)."""
        w = self.world
        self.sk.teleport_base(st["base"][0], st["base"][1])
        idx = np.array([0], np.uint32)
        self.robot.set_dof_positions(st["q"], idx)
        self.robot.set_dof_velocities(np.zeros_like(st["q"]), idx)
        self.robot.set_dof_position_targets(st["t"], idx)
        for key, value in st["drives"].items():
            w.drives[key][0].GetTargetPositionAttr().Set(value)
        for body, g in st["gravity"].items():
            if g is not None:
                self._set_gravity(body, g)
            elif body in self._touched_gravity:
                self._set_gravity(body, False)
        w.sim.flush_changes()
        self._apply_poses(st["poses"])
        w.step(1, sync=True)
        err = 0.
        for body, pose in st["poses"].items():
            try:
                err = max(err, float(np.linalg.norm(w.pose(body)[:3] - pose[:3])))
            except ValueError:
                pass
        return err

    # -------------------------------------------------------------- frames
    def _images(self) -> dict[str, np.ndarray]:
        from spectator import FlightCamera
        out = {}
        if "chase" in self.cameras:
            p, yaw = self.sk.base()
            pos, yaw_deg, pitch = chase_camera(p, yaw)
            out["chase"] = self.sk.renderer.capture(FlightCamera(pos, yaw_deg, pitch))
        if "head" in self.cameras:
            out["head"] = self.sk.renderer.capture(self.sk.view())
        return out

    def _log_state(self) -> dict[str, Any]:
        p, yaw = self.sk.base()
        att = self._attached
        return dict(phase=self._phase, base_xyz_yaw=[*map(float, p), float(yaw)],
                    q=self.robot.get_dof_positions()[0],
                    attached_body=att[0] if att else None,
                    attached_pose=self.world.pose(att[0]) if att else None,
                    drive_targets={k: float(d[0].GetTargetPositionAttr().Get() or 0.)
                                   for k, d in self.world.drives.items()})

    def _emit(self, physics_steps: int = 1) -> None:
        """Advance kinematic playback one frame and record it."""
        w = self.world
        if physics_steps:
            w.step(physics_steps, sync=True)
        if self._attached:
            body, rel = self._attached
            grip = w.pose(f"{ROBOT}/gripper_{self._side}_center_link")
            self._apply_poses({body: compose(grip, rel)})
            w.step(0, sync=True)  # push the new transform to USD for rendering
        rec = self._recorder()
        if rec is not None:
            rec.add(self._images(), self._log_state())

    def _hold(self, seconds: float) -> None:
        rec = self._recorder()
        if rec is None:
            return
        imgs, state = self._images(), self._log_state()
        for _ in range(frame_count(seconds, self.fps)):
            rec.add(imgs, state)

    def _play_joints(self, frames: np.ndarray, phase: str) -> None:
        self._phase = phase
        for q in frames:
            self._set_robot(q)
            self._emit(1)

    # ------------------------------------------------------------- helpers
    def _gripper_ix(self, side: str) -> tuple[int, int]:
        from arena_api.robot_profiles import INNER
        grip = 1 if side == "r" else 0
        return self.sk.names.index(self.sk.joints.GRIPPERS[grip]), self.sk.names.index(INNER[grip])

    def _with_gripper(self, q: np.ndarray, side: str, value: float) -> np.ndarray:
        q = np.array(q, float, copy=True)
        outer, inner = self._gripper_ix(side)
        q[..., outer], q[..., inner] = value, -value
        return q

    def _arm_weights(self, side: str) -> np.ndarray:
        w = np.zeros(len(self.sk.names))
        arm = list(self.sk.joints.RIGHT_ARM if side == "r" else self.sk.joints.LEFT_ARM)
        for name in arm + ["idx01_body_joint1", "idx02_body_joint2"]:
            if name in self.sk.names:
                w[self.sk.names.index(name)] = 1.
        return w

    def _retime(self, path, side) -> np.ndarray:
        return retime_path(np.asarray(path, float), self.fps, vmax=.9, tmin=.5, weights=self._arm_weights(side))

    def _attach(self, body: str, rel: np.ndarray) -> None:
        self._set_gravity(body, True)
        self._touched_gravity.add(body)
        self.world.sim.flush_changes()
        self._attached = (body, np.asarray(rel, float))

    # ------------------------------------------------------------- replays
    def _replay_base(self, plan: dict, pre: dict) -> None:
        q = pre["q"][0]
        self._phase = "base"
        for x, y, z, yaw in base_frames(plan["poses"], self.fps, BaseLimits()):
            self.sk.teleport_base((x, y, z), yaw)
            self._set_robot(q)
            self._emit(0)

    def _replay_grasp(self, plan: dict, pre: dict) -> None:
        side = self._side = plan["side"]
        q0 = pre["q"][0]
        path = [self._with_gripper(q, side, GRIP_OPEN) for q in plan["ik_path"]] or [q0]
        self._play_joints(blend(q0, path[0], self.fps, .35), "open_gripper")
        reach = self._retime(path, side)
        self._play_joints(reach, "reach")
        goal = reach[-1] if len(reach) else path[-1]
        self._play_joints(blend(goal, self._with_gripper(goal, side, GRIP_CLOSED), self.fps, .35), "close_gripper")
        grip = self.world.pose(f"{ROBOT}/gripper_{side}_center_link")
        self._attach(plan["body"], relative(grip, np.asarray(plan["object_pose"], float)))
        back = [self._with_gripper(q, side, GRIP_CLOSED) for q in plan["ik_path"][::-1]]
        self._play_joints(self._retime(back, side), "lift_retract")

    def _replay_place(self, plan: dict, pre: dict) -> None:
        side = self._side = plan["side"]
        release = np.asarray(plan["release_pose"], float)
        carry = np.r_[0., 0., -.03, 0., 0., 0., 1.]
        grip = self.world.pose(f"{ROBOT}/gripper_{side}_center_link")
        carry[3:] = relative(grip, release)[3:]  # keep the release orientation while carried
        self._attach(plan["body"], carry)
        path = [self._with_gripper(q, side, GRIP_CLOSED) for q in plan["ik_path"]]
        self._play_joints(self._retime(path, side), "carry")
        top = path[-1]
        self._play_joints(blend(top, self._with_gripper(top, side, GRIP_OPEN), self.fps, .3), "open_gripper")
        body, _ = self._attached
        self._attached = None
        start = self.world.pose(body)
        self._phase = "release"
        n = frame_count(.4, self.fps)
        for i in range(n):
            self._apply_poses({body: lerp_pose(start, release, float(min_jerk((i + 1) / n)))})
            self._emit(1)
        back = [self._with_gripper(q, side, GRIP_OPEN) for q in plan["ik_path"][::-1]]
        self._play_joints(self._retime(back, side), "retract")

    def _replay_articulate(self, plan: dict, pre: dict, info: dict) -> None:
        from rab_skills import Refused
        w, sk = self.world, self.sk
        handle = np.asarray(plan["handle"], float)
        p, yaw = sk.base()
        side = self._side = "l" if np.dot(handle[:2] - p[:2], [-math.sin(yaw), math.cos(yaw)]) > .06 else "r"
        toward = np.r_[math.cos(yaw), math.sin(yaw), 0.]
        q0 = pre["q"][0]
        approach: list = []
        try:  # animation-only IK (OPEN/CLOSE mechanics never needed it)
            _, approach = sk._ik_solve(handle - .05 * toward, side, tolerance=.04, record=True)
        except Refused as e:
            info.setdefault("degraded", []).append(f"handle_ik:{e.reason}")
        if approach:
            approach = [self._with_gripper(q, side, GRIP_OPEN) for q in approach]
            self._play_joints(self._retime(approach, side), "reach_handle")
            hold = approach[-1]
            self._play_joints(blend(hold, self._with_gripper(hold, side, GRIP_CLOSED), self.fps, .3), "grip_handle")
        drive, _, body, _, _ = w.drives[plan["key"]]
        t0 = w.pose(body)
        handle_local = relative(t0, np.r_[handle - .05 * toward, 0., 0., 0., 1.])
        a, b = plan["drive_from"], plan["drive_to"]
        self._phase = "pull" if plan["opening"] else "push"
        n = frame_count(1.6, self.fps)
        steps = max(1, int(round(60 / self.fps)))
        for i in range(n):
            f = (i + 1) / n
            drive.GetTargetPositionAttr().Set(a + (b - a) * f * f * (3 - 2 * f))
            if approach:
                goal = compose(w.pose(body), handle_local)[:3]
                try:
                    _, follow = sk._ik_solve(goal, side, tolerance=.03, record=True)
                    if follow:
                        self._set_robot(self._with_gripper(follow[-1], side, GRIP_CLOSED))
                except Refused:
                    pass  # hand lags this frame; purely visual
            self._emit(steps)
        if approach:
            q_now = self.robot.get_dof_positions()[0]
            self._play_joints(blend(q_now, self._with_gripper(q_now, side, GRIP_OPEN), self.fps, .3), "release_handle")
            self._play_joints(retime_path([self._with_gripper(q_now, side, GRIP_OPEN), q0], self.fps,
                                          vmax=.9, tmin=.6, weights=self._arm_weights(side)), "retract")

    # -------------------------------------------------------------- driver
    def replay(self, action_index: int, raw: Any, code: str, pre: dict | None,
               plan: dict) -> dict | None:
        skill = raw.get("skill") if isinstance(raw, dict) else str(raw)
        rec = self._recorder()
        if rec is None:
            return None
        rec.begin(action_index, skill, raw if isinstance(raw, dict) else {}, code)
        info: dict[str, Any] = dict(mode="replay", animated=False)
        if code != "EXECUTED" or pre is None or not plan:
            # Refused / non-physical: no motion, just keep the demo timeline.
            self._phase = "idle" if code == "EXECUTED" else "refused"
            try:
                self._hold(self.hold_s)
            except Exception:
                info["error"] = traceback.format_exc(limit=3)
            rec.end(animated=False)
            return None
        started, frame0 = time.monotonic(), rec.frame
        post = None
        try:
            post = self._state()
            self._restore(pre)
            kind = plan.get("kind")
            if kind == "base":
                self._replay_base(plan, pre)
            elif kind == "grasp":
                self._replay_grasp(plan, pre)
            elif kind == "place":
                self._replay_place(plan, pre)
            elif kind == "articulate":
                self._replay_articulate(plan, pre, info)
            info["animated"] = True
        except Exception:
            info["error"] = traceback.format_exc(limit=5)
            print("[motion-replay] " + info["error"], flush=True)
        finally:
            self._attached = None
            if post is not None:
                try:
                    info["restore_max_err_m"] = self._restore(post)
                except Exception:
                    info["restore_error"] = traceback.format_exc(limit=5)
                    print("[motion-replay] RESTORE FAILED " + info["restore_error"], flush=True)
            try:
                self._phase = "settled"
                self._hold(self.hold_s)
            except Exception:
                pass
            rec.flush()
        info.update(frames=rec.frame - frame0, wall_s=round(time.monotonic() - started, 3), kind=plan.get("kind"))
        rec.end(animated=info["animated"], kind=plan.get("kind"), restore_max_err_m=info.get("restore_max_err_m"))
        return info
