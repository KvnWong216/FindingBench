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
                        frame_count, gripper_frame, lerp_pose, min_jerk, plan_grasp_pose,
                        plan_place_pose, quat_to_rotmat, relative, retime_path, rot_error_vec)

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
        self._carry_meta: dict[str, Any] | None = None

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
            grip = w.pose(self.sk.emb.ee_link_path(self._side))
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
        if self.sk.emb.name == "unitree_g1":
            from rab_embodiment import dex3_hand_targets
            openness = float(np.clip((float(value) - GRIP_CLOSED) / max(1e-6, GRIP_OPEN - GRIP_CLOSED), 0.0, 1.0))
            for name, jval in dex3_hand_targets(side, openness).items():
                if name in self.sk.names:
                    q[..., self.sk.names.index(name)] = jval
            return q
        outer, inner = self._gripper_ix(side)
        q[..., outer], q[..., inner] = value, -value
        return q

    def _arm_weights(self, side: str) -> np.ndarray:
        w = np.zeros(len(self.sk.names))
        for name in self.sk.emb.arm_joints(side):
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

    def _ik_solve_6d(self, position: np.ndarray, rotmat: np.ndarray | None, side: str, *,
                     seed_q: np.ndarray | None = None, pos_tol: float = .025,
                     rot_tol: float = .16, w_rot: float = .25, max_iters: int = 140,
                     damp: float = .024, step_clip: float = .10) -> tuple[float, float, list[np.ndarray]]:
        """Animation-only 6-DoF weighted DLS-IK using the full 6-row Articulation Jacobian.

        Returns ``(pos_err_m, rot_err_rad, joint_iterates)`` and always restores the
        robot's joint state before returning. Raises ``ValueError`` if tolerances are not met.
        """
        robot = self.robot
        jmeta = robot.shared_metatype
        arm = [j for j in self.sk.emb.arm_joints(side) if j in self.sk.names]
        ix = [self.sk.names.index(x) for x in arm]
        link = self.sk.emb.ee_link(side)
        link_path = self.sk.emb.ee_link_path(side)
        links = list(jmeta.link_names)
        lim = robot.get_dof_limits()[0, ix]
        idx = np.array([0], np.uint32)
        saved_q = robot.get_dof_positions().copy()
        saved_t = robot.get_dof_position_targets().copy()
        target_p = np.asarray(position, float)
        target_R = None if rotmat is None else np.asarray(rotmat, float)
        try:
            q_start = saved_q.copy() if seed_q is None else np.asarray(seed_q, np.float32).reshape(1, -1).copy()
            robot.set_dof_positions(q_start, idx)
            robot.set_dof_velocities(np.zeros_like(q_start), idx)
            robot.set_dof_position_targets(q_start, idx)
            self.world.step(1, sync=True)
            path = [q_start[0].copy()]
            if seed_q is None and float(target_p[2]) < .45:
                seed = q_start.copy()
                if "idx01_body_joint1" in self.sk.names:
                    seed[0, self.sk.names.index("idx01_body_joint1")] = .02
                if "idx02_body_joint2" in self.sk.names:
                    seed[0, self.sk.names.index("idx02_body_joint2")] = .65
                robot.set_dof_positions(seed, idx)
                robot.set_dof_position_targets(seed, idx)
                self.world.step(2, sync=True)
                path.append(seed[0].copy())
            for _ in range(max_iters):
                pose = self.world.pose(link_path)
                e_pos = target_p - pose[:3]
                pos_err = float(np.linalg.norm(e_pos))
                if target_R is not None and w_rot > 0.:
                    e_rot = rot_error_vec(target_R, quat_to_rotmat(pose[3:]))
                    rot_err = float(np.linalg.norm(e_rot))
                else:
                    e_rot, rot_err = np.zeros(3), 0.
                if pos_err <= pos_tol and (target_R is None or w_rot <= 0. or rot_err <= rot_tol):
                    return pos_err, rot_err, path
                jac = robot.get_jacobians()[0]
                i = links.index(link) - (1 if jac.shape[0] == len(links) - 1 else 0)
                offset = jac.shape[-1] - len(self.sk.names)
                cols = np.array(ix) + offset
                j_lin = jac[i, :3, :][:, cols]
                if target_R is not None and w_rot > 0.:
                    j_ang = jac[i, 3:6, :][:, cols]
                    j6 = np.vstack((j_lin, w_rot * j_ang))
                    e6 = np.r_[e_pos, w_rot * e_rot]
                    d = j6.T @ np.linalg.solve(j6 @ j6.T + np.eye(6) * (damp ** 2), e6)
                else:
                    d = j_lin.T @ np.linalg.solve(j_lin @ j_lin.T + np.eye(3) * (.018 ** 2), e_pos)
                q = robot.get_dof_positions().copy()
                q[0, ix] = np.clip(q[0, ix] + np.clip(d, -step_clip, step_clip), lim[:, 0], lim[:, 1])
                robot.set_dof_positions(q, idx)
                robot.set_dof_position_targets(q, idx)
                self.world.step(1, sync=True)
                path.append(q[0].copy())
            pose = self.world.pose(link_path)
            pos_err = float(np.linalg.norm(target_p - pose[:3]))
            rot_err = float(np.linalg.norm(rot_error_vec(target_R, quat_to_rotmat(pose[3:])))) if target_R is not None else 0.
            raise ValueError(f"ik6d_failed:pos={pos_err:.4f}:rot={rot_err:.3f}")
        finally:
            robot.set_dof_positions(saved_q, idx)
            robot.set_dof_velocities(np.zeros_like(saved_q), idx)
            robot.set_dof_position_targets(saved_t, idx)
            self.world.step(1, sync=True)

    def _cartesian_6d(self, p0: np.ndarray, p1: np.ndarray, rotmat: np.ndarray | None, side: str,
                      seed_q: np.ndarray, *, n_sub: int = 6, w_rot: float = .25,
                      pos_tol: float = .022, rot_tol: float = .18) -> list[np.ndarray]:
        """Solve a straight-line Cartesian segment by warm-starting 6-DoF IK along p0 -> p1."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        q_cur = np.asarray(seed_q, float).copy()
        out = [q_cur]
        for k in range(1, n_sub + 1):
            pt = p0 + (p1 - p0) * (k / n_sub)
            _, _, seg = self._ik_solve_6d(pt, rotmat, side, seed_q=q_cur, w_rot=w_rot,
                                          pos_tol=pos_tol, rot_tol=rot_tol, max_iters=45)
            q_cur = seg[-1]
            out.append(q_cur)
        return out

    # ------------------------------------------------------------- replays
    def _replay_base(self, plan: dict, pre: dict) -> None:
        q = pre["q"][0]
        self._phase = "base"
        for x, y, z, yaw in base_frames(plan["poses"], self.fps, BaseLimits()):
            self.sk.teleport_base((x, y, z), yaw)
            self._set_robot(q)
            self._emit(0)

    def _plan_grasp_6d(self, plan: dict, pre: dict, side: str):
        """Solve a 4-stage 6-DoF grasp (ready -> pre -> grasp -> lift) with graceful fallback."""
        p, yaw = self.sk.base()
        cur_R = quat_to_rotmat(self.world.pose(self.sk.emb.ee_link_path(side))[3:])
        box = plan.get("box")
        if box is None:
            t = np.asarray(plan["target"], float)
            box = [(t - .03).tolist(), (t + .03).tolist()]
        obj_pose = np.asarray(plan["object_pose"], float)
        primary = plan_grasp_pose(box, obj_pose, p, yaw,
                                  in_open_drawer=bool(plan.get("in_open_drawer")), cur_R=cur_R)
        alt_mode = "side" if primary.mode == "top_down" else "top_down"
        attempts = [
            (f"6dof_{primary.mode}", primary, .25, .18),
            (f"6dof_{primary.mode}_relaxed", primary, .12, .32),
        ]
        if not plan.get("in_open_drawer"):
            alt = plan_grasp_pose(box, obj_pose, p, yaw, in_open_drawer=False, cur_R=cur_R, force_mode=alt_mode)
            attempts.append((f"6dof_{alt.mode}", alt, .22, .22))
        q0 = pre["q"][0]
        for tag, gp, w_rot, rot_tol in attempts:
            try:
                _, _, to_pre = self._ik_solve_6d(gp.pre_pos, gp.rotmat, side, w_rot=w_rot,
                                                 pos_tol=.030, rot_tol=rot_tol, max_iters=140)
                to_grasp = self._cartesian_6d(gp.pre_pos, gp.grasp_pos, gp.rotmat, side,
                                              to_pre[-1], n_sub=6, w_rot=w_rot, pos_tol=.022, rot_tol=rot_tol)
                try:
                    to_lift = self._cartesian_6d(gp.grasp_pos, gp.lift_pos, gp.rotmat, side,
                                                 to_grasp[-1], n_sub=5, w_rot=w_rot, pos_tol=.030, rot_tol=rot_tol + .08)
                except ValueError:
                    to_lift = [to_grasp[-1], to_pre[-1]]
                return tag, gp, to_pre, to_grasp, to_lift
            except ValueError:
                continue
        return "3dof_fallback", primary, None, None, None

    def _replay_grasp(self, plan: dict, pre: dict, info: dict) -> None:
        side = self._side = plan["side"]
        q0 = pre["q"][0]
        solver, gp, to_pre, to_grasp, to_lift = self._plan_grasp_6d(plan, pre, side)
        info.update(grasp_solver=solver, grasp_mode=gp.mode,
                    pinch_width_m=round(gp.pinch_width_m, 4),
                    q_open=round(gp.q_open, 3), q_close=round(gp.q_close, 3))
        if to_pre is not None:
            p_pre = [self._with_gripper(q, side, gp.q_open) for q in to_pre]
            p_approach = [self._with_gripper(q, side, gp.q_open) for q in to_grasp]
            self._play_joints(blend(q0, p_pre[0], self.fps, .25), "open_gripper")
            self._play_joints(self._retime(p_pre, side), "pre_grasp")
            self._play_joints(retime_path(p_approach, self.fps, vmax=.45, tmin=.45,
                                          weights=self._arm_weights(side)), "approach_grasp")
            goal = p_approach[-1]
            self._play_joints(blend(goal, self._with_gripper(goal, side, gp.q_close), self.fps, .35), "close_gripper")
            grip = self.world.pose(self.sk.emb.ee_link_path(side))
            rel = relative(grip, np.asarray(plan["object_pose"], float))
            self._attach(plan["body"], rel)
            self._carry_meta = dict(body=plan["body"], mode=gp.mode, q_open=gp.q_open,
                                    q_close=gp.q_close, rel=rel)
            p_lift = [self._with_gripper(q, side, gp.q_close) for q in to_lift]
            self._play_joints(retime_path(p_lift, self.fps, vmax=.45, tmin=.40,
                                          weights=self._arm_weights(side)), "lift")
            p_back = [self._with_gripper(p_lift[-1], side, gp.q_close),
                      self._with_gripper(q0, side, gp.q_close)]
            self._play_joints(self._retime(p_back, side), "retract")
            return
        # Fallback: reuse validated 3-DoF IK path from rab_skills.py
        info.setdefault("degraded", []).append("grasp_6dof_fallback_to_3dof")
        path = [self._with_gripper(q, side, gp.q_open) for q in plan["ik_path"]] or [q0]
        self._play_joints(blend(q0, path[0], self.fps, .35), "open_gripper")
        reach = self._retime(path, side)
        self._play_joints(reach, "reach")
        goal = reach[-1] if len(reach) else path[-1]
        self._play_joints(blend(goal, self._with_gripper(goal, side, gp.q_close), self.fps, .35), "close_gripper")
        grip = self.world.pose(self.sk.emb.ee_link_path(side))
        rel = relative(grip, np.asarray(plan["object_pose"], float))
        self._attach(plan["body"], rel)
        self._carry_meta = dict(body=plan["body"], mode=gp.mode, q_open=gp.q_open,
                                q_close=gp.q_close, rel=rel)
        back = [self._with_gripper(q, side, gp.q_close) for q in plan["ik_path"][::-1]]
        self._play_joints(self._retime(back, side), "lift_retract")

    def _replay_place(self, plan: dict, pre: dict, info: dict) -> None:
        from scipy.spatial.transform import Rotation
        side = self._side = plan["side"]
        q0 = pre["q"][0]
        release = np.asarray(plan["release_pose"], float)
        p, yaw = self.sk.base()
        cur_R = quat_to_rotmat(self.world.pose(self.sk.emb.ee_link_path(side))[3:])
        box = plan.get("box")
        if box is None:
            pt = np.asarray(plan["point"], float)
            box = [(pt - .03).tolist(), (pt + .03).tolist()]
        meta = self._carry_meta if self._carry_meta and self._carry_meta.get("body") == plan["body"] else None
        pp = plan_place_pose(box, plan["point"], p, yaw, cur_R=cur_R,
                             preferred_mode=meta["mode"] if meta else None)
        q_close = meta["q_close"] if meta else pp.q_close
        q_open = meta["q_open"] if meta else pp.q_open

        to_pre = to_down = to_lift = None
        solver = "3dof_fallback"
        for tag, w_rot, rot_tol in ((f"6dof_{pp.mode}", .25, .18), (f"6dof_{pp.mode}_relaxed", .12, .32)):
            try:
                _, _, to_pre = self._ik_solve_6d(pp.pre_pos, pp.rotmat, side, w_rot=w_rot,
                                                 pos_tol=.030, rot_tol=rot_tol, max_iters=140)
                to_down = self._cartesian_6d(pp.pre_pos, pp.place_pos, pp.rotmat, side,
                                             to_pre[-1], n_sub=5, w_rot=w_rot, pos_tol=.022, rot_tol=rot_tol)
                try:
                    to_lift = self._cartesian_6d(pp.place_pos, pp.lift_pos, pp.rotmat, side,
                                                 to_down[-1], n_sub=4, w_rot=w_rot, pos_tol=.030, rot_tol=rot_tol + .08)
                except ValueError:
                    to_lift = [to_down[-1], to_pre[-1]]
                solver = tag
                break
            except ValueError:
                to_pre = to_down = to_lift = None
        info.update(place_solver=solver, place_mode=pp.mode)

        if to_pre is not None:
            # Attach object with exact offset so at place_pos it coincides with release_pose.
            q_place = Rotation.from_matrix(pp.rotmat).as_quat()
            carry = relative(np.r_[pp.place_pos, q_place], release)
            self._attach(plan["body"], carry)
            p_pre = [self._with_gripper(q, side, q_close) for q in to_pre]
            p_down = [self._with_gripper(q, side, q_close) for q in to_down]
            self._play_joints(self._retime(p_pre, side), "carry_pre_place")
            self._play_joints(retime_path(p_down, self.fps, vmax=.40, tmin=.40,
                                          weights=self._arm_weights(side)), "descend_place")
            bottom = p_down[-1]
            self._play_joints(blend(bottom, self._with_gripper(bottom, side, q_open), self.fps, .30), "open_gripper")
            body, _ = self._attached
            self._attached = None
            start = self.world.pose(body)
            self._phase = "settle"
            n = frame_count(.25, self.fps)
            for i in range(n):
                self._apply_poses({body: lerp_pose(start, release, float(min_jerk((i + 1) / n)))})
                self._emit(1)
            p_lift = [self._with_gripper(q, side, q_open) for q in to_lift]
            self._play_joints(retime_path(p_lift, self.fps, vmax=.45, tmin=.35,
                                          weights=self._arm_weights(side)), "lift_clear")
            p_back = [self._with_gripper(p_lift[-1], side, q_open),
                      self._with_gripper(q0, side, q_open)]
            self._play_joints(self._retime(p_back, side), "retract")
            self._carry_meta = None
            return

        # Fallback: reuse validated 3-DoF IK path from rab_skills.py
        info.setdefault("degraded", []).append("place_6dof_fallback_to_3dof")
        carry = np.r_[0., 0., -.03, 0., 0., 0., 1.]
        grip = self.world.pose(self.sk.emb.ee_link_path(side))
        carry[3:] = relative(grip, release)[3:]
        self._attach(plan["body"], carry)
        path = [self._with_gripper(q, side, q_close) for q in plan["ik_path"]]
        self._play_joints(self._retime(path, side), "carry")
        top = path[-1]
        self._play_joints(blend(top, self._with_gripper(top, side, q_open), self.fps, .3), "open_gripper")
        body, _ = self._attached
        self._attached = None
        start = self.world.pose(body)
        self._phase = "release"
        n = frame_count(.4, self.fps)
        for i in range(n):
            self._apply_poses({body: lerp_pose(start, release, float(min_jerk((i + 1) / n)))})
            self._emit(1)
        back = [self._with_gripper(q, side, q_open) for q in plan["ik_path"][::-1]]
        self._play_joints(self._retime(back, side), "retract")
        self._carry_meta = None

    def _replay_articulate(self, plan: dict, pre: dict, info: dict) -> None:
        from rab_skills import Refused
        w, sk = self.world, self.sk
        handle = np.asarray(plan["handle"], float)
        p, yaw = sk.base()
        side = self._side = "l" if np.dot(handle[:2] - p[:2], [-math.sin(yaw), math.cos(yaw)]) > .06 else "r"
        toward = np.r_[math.cos(yaw), math.sin(yaw), 0.]
        cur_R = quat_to_rotmat(w.pose(self.sk.emb.ee_link_path(side))[3:])
        # Align fingertip +Z toward the drawer/door handle with horizontal pinch axis.
        handle_R = gripper_frame(toward, [0., 0., 1.], cur_R=cur_R)
        q0 = pre["q"][0]
        approach: list = []
        try:
            _, _, approach = self._ik_solve_6d(handle - .05 * toward, handle_R, side, w_rot=.20,
                                               pos_tol=.035, rot_tol=.28, max_iters=130)
        except ValueError:
            try:
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
                q_prev = self.robot.get_dof_positions()[0]
                try:
                    _, _, follow = self._ik_solve_6d(goal, handle_R, side, seed_q=q_prev, w_rot=.18,
                                                     pos_tol=.030, rot_tol=.30, max_iters=35)
                    if follow:
                        self._set_robot(self._with_gripper(follow[-1], side, GRIP_CLOSED))
                except ValueError:
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
                self._replay_grasp(plan, pre, info)
            elif kind == "place":
                self._replay_place(plan, pre, info)
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
