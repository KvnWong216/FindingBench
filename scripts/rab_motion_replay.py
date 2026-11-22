"""Visual replay of executed FindingBench RAB skills as continuous robot motion.

Enabled with ``FINDINGBENCH_RAB_MOTION=replay`` (default ``discrete``).

Contract ("ideal cerebellum", mechanics unchanged)
--------------------------------------------------
1. ``snapshot()`` is taken right before a physical skill (MOVE_TO, TURN, OPEN,
   CLOSE, GRASP, PLACE) runs.  Snapshots only READ simulator state.
2. The skill runs exactly as in discrete mode (A* + clearance rays, cuRobo/DLS IK +
   hand-path rays, PhysX settle).  If it is refused, nothing is animated and the
   failure is returned unchanged.
3. Only after ``EXECUTED``: the authoritative post-state is snapshotted, the
   pre-state is restored, the trajectory the mechanics already computed
   (``Skills.motion``: A* waypoints / 6-DoF cuRoboV2-Fast IK / drawer drive targets)
   is played back kinematically while frames are recorded, and finally the
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

from rab_motion import (
    BaseLimits,
    DemoRecorder,
    base_frames,
    blend,
    chase_camera,
    compose,
    frame_count,
    gripper_frame,
    lerp_pose,
    min_jerk,
    plan_grasp_pose,
    plan_place_pose,
    quat_to_rotmat,
    relative,
    retime_path,
    rot_error_vec,
    wrap_angle,
)

PHYSICAL = {"MOVE_TO", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE"}
GRIP_OPEN, GRIP_CLOSED = 0.64, 0.01

# Posture tuple: (base_xyz (3,), yaw (float), crouch_dz (float), pelvis_pitch (float))
Posture = tuple[np.ndarray, float, float, float]


def _crouch_leg_joints(crouch_dz: float, pelvis_pitch: float, waist_pitch: float) -> dict[str, float]:
    """Exact foot-planted symmetric leg + waist joints at interpolated (crouch_dz, pelvis_pitch)."""
    from rab_embodiment import G1_SHIN_LEN, G1_THIGH_LEN

    L1, L2 = G1_THIGH_LEN, G1_SHIN_LEN
    dz = float(max(0.0, crouch_dz))
    pp = float(pelvis_pitch)
    wp = float(waist_pitch)
    if dz < 1e-4:
        return {
            "left_hip_pitch_joint": -pp,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "right_hip_pitch_joint": -pp,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "waist_pitch_joint": wp,
        }
    splay = 0.65 * min(1.0, dz / 0.08)
    d = float(np.clip((L1 + L2) - dz, 0.26, L1 + L2 - 1e-5))
    cos_knee = float(np.clip((L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2), -1.0, 1.0))
    knee = math.pi - math.acos(cos_knee)
    cos_hip = float(np.clip((L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d), -1.0, 1.0))
    hip_up = -math.acos(cos_hip)
    ankle = float(np.clip(-(knee + hip_up), math.radians(-48.0), math.radians(28.0)))
    hip = float(np.clip(hip_up - pp, math.radians(-138.0), 0.0))
    knee = float(np.clip(knee, 0.0, math.radians(155.0)))
    return {
        "left_hip_pitch_joint": hip,
        "left_hip_yaw_joint": splay,
        "left_knee_joint": knee,
        "left_ankle_pitch_joint": ankle,
        "right_hip_pitch_joint": hip,
        "right_hip_yaw_joint": -splay,
        "right_knee_joint": knee,
        "right_ankle_pitch_joint": ankle,
        "waist_pitch_joint": wp,
    }


class ReplayDirector:
    def __init__(self, skills, out: Path | None):
        self.sk = skills
        self.out = Path(out) if out else None
        self.fps = float(os.environ.get("FINDINGBENCH_RAB_DEMO_FPS", 20))
        self.cameras = tuple(
            c.strip()
            for c in os.environ.get("FINDINGBENCH_RAB_DEMO_CAMERAS", "chase,head").split(",")
            if c.strip()
        )
        self.hold_s = float(os.environ.get("FINDINGBENCH_RAB_DEMO_HOLD_S", 0.3))
        self.rec: DemoRecorder | None = None
        self._touched_gravity: set[str] = set()
        self._phase = ""
        self._side = "r"
        # (body_path, rel_pose_in_pinch_frame, attach_side)
        self._attached: tuple[str, np.ndarray, str] | None = None
        self._carry_meta: dict[str, Any] | None = None
        self._cur_posture: Posture | None = None
        self._cached_scene_bodies: list[str] | None = None
        self._sim_view: Any = None
        self._rb_views: dict[str, Any] = {}

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
            self.rec = DemoRecorder(
                self.out, fps=self.fps, cameras=self.cameras, dof_names=list(self.sk.names)
            )
        return self.rec

    def close(self) -> None:
        if self.rec is not None:
            self.rec.close()
            self.rec = None

    # ------------------------------------------------------------ snapshot
    def _row_bodies(self) -> list[str]:
        return [r["body"] for r in self.world.rows]

    def _scene_bodies(self) -> list[str]:
        """All movable non-robot rigid bodies in `/World` (placed items + ambient lived-in props)."""
        if self._cached_scene_bodies is None:
            from pxr import UsdPhysics
            from rab_edit import prefix

            drv_bodies = {d[2] for d in self.world.drives.values()}
            found = set(self._row_bodies())
            for prim in self.world.stage.Traverse():
                p = str(prim.GetPath())
                if (
                    not p.startswith("/World/")
                    or p.startswith("/World/Background/")
                    or prefix(p, self.sk.robot_prim)
                    or prefix(p, "/World/Robot")
                    or any(prefix(p, db) for db in drv_bodies)
                ):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    kin = rb.GetKinematicEnabledAttr()
                    if kin and kin.Get():
                        continue
                    found.add(p)
            self._cached_scene_bodies = sorted(found)
        return self._cached_scene_bodies

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
        all_bodies = self._scene_bodies()
        for body in all_bodies:
            try:
                poses[body] = w.pose(body)
            except ValueError:
                pass
        p, yaw = self.sk.base()
        held_info = (self.sk.held[0]["body"], self.sk.held[1]) if self.sk.held else None
        return dict(
            base=(p.copy(), float(yaw)),
            q=self.robot.get_dof_positions().copy(),
            t=self.robot.get_dof_position_targets().copy(),
            poses=poses,
            held=held_info,
            gravity={b: self._gravity(b) for b in all_bodies},
            drives={k: float(d[0].GetTargetPositionAttr().Get() or 0.0) for k, d in w.drives.items()},
        )

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

    def _apply_poses(self, poses: dict[str, np.ndarray], *, only_changed: bool = False) -> None:
        import omni.physics.tensors

        if not poses:
            return
        if self._sim_view is None:
            self._sim_view = omni.physics.tensors.create_simulation_view("numpy", self.world.sid.ToLongInt())
        idx = np.array([0], np.uint32)
        for body, pose in poses.items():
            if only_changed:
                try:
                    cur = self.world.pose(body)
                    if float(np.linalg.norm(cur[:3] - pose[:3])) < 2e-4 and abs(float(np.dot(cur[3:], pose[3:]))) > 0.99999:
                        continue
                except ValueError:
                    continue
            if body not in self._rb_views:
                self._rb_views[body] = self._sim_view.create_rigid_body_view(body)
            actor = self._rb_views[body]
            if getattr(actor, "_backend", None) is None or actor.count != 1:
                continue
            actor.set_transforms(np.asarray([pose], np.float32), idx)
            actor.set_velocities(np.zeros((1, 6), np.float32), idx)

    def _set_robot(self, q: np.ndarray) -> None:
        idx = np.array([0], np.uint32)
        q_arr = np.asarray(q, np.float32).reshape(1, -1)
        self.robot.set_dof_positions(q_arr, idx)
        self.robot.set_dof_velocities(np.zeros_like(q_arr), idx)
        self.robot.set_dof_position_targets(q_arr, idx)

    def _set_posture_and_joints(self, posture: Posture, q: np.ndarray) -> None:
        """Set base/pelvis root pose (including crouch_dz & hip-hinge pitch) and DOFs."""
        from scipy.spatial.transform import Rotation

        xyz, yaw, crouch_dz, pitch = posture
        self._cur_posture = (np.asarray(xyz, float).copy(), float(yaw), float(crouch_dz), float(pitch))
        if not self.sk.emb.supports_crouch:
            p_now, yaw_now = self.sk.base()
            if float(np.linalg.norm(p_now[:2] - np.asarray(xyz, float)[:2])) > 1e-4 or abs(wrap_angle(yaw_now - yaw)) > 1e-4:
                self.sk.teleport_base(xyz, yaw)
            self._set_robot(q)
            return

        mobile = self.world.mobile
        root_z = float(xyz[2]) + self.sk.emb.pelvis_z_offset - float(crouch_dz)
        q_xyzw = Rotation.from_euler("ZY", [float(yaw), float(pitch)]).as_quat()
        qx, qy, qz, qw = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
        root = np.array([[float(xyz[0]), float(xyz[1]), root_z, qx, qy, qz, qw]], np.float32)
        idx = np.array([0], np.uint32)
        q_arr = np.asarray(q, np.float32).reshape(1, -1)

        mobile.anchor.GetJointEnabledAttr().Set(False)
        mobile.flush()
        mobile.view.set_root_transforms(root, idx)
        mobile.view.set_root_velocities(np.zeros((1, 6), np.float32), idx)
        mobile.view.set_dof_positions(q_arr, idx)
        mobile.view.set_dof_velocities(np.zeros_like(q_arr), idx)
        mobile.view.set_dof_position_targets(q_arr, idx)
        mobile.anchor.GetLocalPos0Attr().Set(mobile.Gf.Vec3f(float(xyz[0]), float(xyz[1]), root_z))
        mobile.anchor.GetLocalRot0Attr().Set(mobile.Gf.Quatf(qw, mobile.Gf.Vec3f(qx, qy, qz)))
        mobile.anchor.GetJointEnabledAttr().Set(True)
        mobile.flush()
        self.sk._crouch_dz = float(crouch_dz)

    def _restore(self, st: dict[str, Any]) -> float:
        """Restore a snapshot; returns the max body translation error (m)."""
        w = self.world
        self.sk.teleport_base(st["base"][0], st["base"][1], crouch_dz=0.0, pitch=0.0,
                              dof_positions=st["q"], dof_targets=st["t"])
        self._cur_posture = (st["base"][0].copy(), float(st["base"][1]), 0.0, 0.0)
        for key, value in st["drives"].items():
            w.drives[key][0].GetTargetPositionAttr().Set(value)
        for body, g in st["gravity"].items():
            if g is not None:
                self._set_gravity(body, g)
            elif body in self._touched_gravity:
                self._set_gravity(body, False)
        w.sim.flush_changes()
        self._apply_poses(st["poses"], only_changed=True)
        w.step(1, sync=True)
        err = 0.0
        for body, pose in st["poses"].items():
            try:
                err = max(err, float(np.linalg.norm(w.pose(body)[:3] - pose[:3])))
            except ValueError:
                pass
        return err

    # -------------------------------------------------------------- frames
    def _pinch_pose_world(self, side: str) -> np.ndarray:
        """Exact world 6-DoF pose [x, y, z, qx, qy, qz, qw] of the fingertip pinch tool frame."""
        ee_pose = self.world.pose(self.sk.emb.ee_link_path(side))
        if self.sk.emb.gripper_type in ("dex1", "dex3"):
            off = self.sk.emb.ee_offset(side)
            # Matches rab_curobo._build_arm_robot_cfg: q_xyzw = [0.5, 0.5, 0.5, 0.5]
            t_pinch = np.array([float(off[0]), float(off[1]), float(off[2]), 0.5, 0.5, 0.5, 0.5], float)
            return compose(ee_pose, t_pinch)
        return ee_pose

    def _images(self) -> dict[str, np.ndarray]:
        from pxr import UsdGeom
        from spectator import FlightCamera

        out = {}
        stage = self.world.stage
        rprim = stage.GetPrimAtPath(self.sk.robot_prim)
        img_api = UsdGeom.Imageable(rprim) if rprim else None
        vis_attr = img_api.GetVisibilityAttr() if img_api else None
        saved_vis = vis_attr.Get() if vis_attr else None

        if "chase" in self.cameras:
            try:
                if img_api is not None:
                    img_api.MakeVisible()
                p, yaw = self.sk.base()
                pos, yaw_deg, pitch = chase_camera(p, yaw)
                out["chase"] = self.sk.renderer.capture(FlightCamera(pos, yaw_deg, pitch))
            finally:
                if vis_attr is not None and saved_vis is not None:
                    vis_attr.Set(saved_vis)
        if "head" in self.cameras:
            if img_api is not None:
                img_api.MakeInvisible()
            out["head"] = self.sk.renderer.capture(self.sk.view())
            if vis_attr is not None and saved_vis is not None:
                vis_attr.Set(saved_vis)
        return out

    def _log_state(self) -> dict[str, Any]:
        p, yaw = self.sk.base()
        att = self._attached
        return dict(
            phase=self._phase,
            base_xyz_yaw=[*map(float, p), float(yaw)],
            crouch_dz=round(float(self.sk._crouch_dz), 4),
            q=self.robot.get_dof_positions()[0],
            attached_body=att[0] if att else None,
            attached_pose=self.world.pose(att[0]) if att else None,
            drive_targets={
                k: float(d[0].GetTargetPositionAttr().Get() or 0.0) for k, d in self.world.drives.items()
            },
        )

    def _emit(self, physics_steps: int = 1) -> None:
        """Advance kinematic playback one frame and record it."""
        w = self.world
        if physics_steps:
            w.step(physics_steps, sync=True)
        else:
            w.step(0, sync=True)
        if self._attached:
            body, rel, att_side = self._attached
            grip = self._pinch_pose_world(att_side)
            self._apply_poses({body: compose(grip, rel)})
            w.step(0, sync=True)
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

    def _play_joints(self, frames: np.ndarray, phase: str, *, posture: Posture | None = None) -> None:
        self._phase = phase
        for q in frames:
            if posture is not None:
                self._set_posture_and_joints(posture, q)
            else:
                self._set_robot(q)
            self._emit(1)

    def _play_posture_transition(
        self,
        q_frames: np.ndarray,
        p_start: Posture,
        p_end: Posture,
        phase: str,
    ) -> None:
        """Play joint frames while smoothly interpolating pelvis squat/bow with planted feet."""
        self._phase = phase
        n = len(q_frames)
        if n == 0:
            return
        if not self.sk.emb.supports_crouch or (
            abs(p_start[2] - p_end[2]) < 1e-4
            and abs(p_start[3] - p_end[3]) < 1e-4
            and float(np.linalg.norm(p_start[0][:2] - p_end[0][:2])) < 1e-4
        ):
            self._play_joints(q_frames, phase, posture=p_end)
            return

        wp_idx = self.sk.names.index("waist_pitch_joint") if "waist_pitch_joint" in self.sk.names else None
        all_lim = self.robot.get_dof_limits()[0]
        for i, q_raw in enumerate(q_frames):
            u = float(min_jerk((i + 1) / n))
            xyz_i = p_start[0] + (p_end[0] - p_start[0]) * u
            yaw_i = wrap_angle(p_start[1] + wrap_angle(p_end[1] - p_start[1]) * u)
            dz_i = p_start[2] + (p_end[2] - p_start[2]) * u
            pitch_i = p_start[3] + (p_end[3] - p_start[3]) * u
            q_i = np.array(q_raw, float, copy=True)
            wp_val = float(q_i[wp_idx]) if wp_idx is not None else 0.0
            leg_j = _crouch_leg_joints(dz_i, pitch_i, wp_val)
            for jn, jv in leg_j.items():
                if jn == "waist_pitch_joint":
                    continue
                if jn in self.sk.names:
                    ji = self.sk.names.index(jn)
                    q_i[ji] = float(np.clip(jv, all_lim[ji, 0], all_lim[ji, 1]))
            self._set_posture_and_joints((xyz_i, yaw_i, dz_i, pitch_i), q_i)
            self._emit(1)

    # ------------------------------------------------------------- helpers
    def _gripper_ix(self, side: str) -> tuple[int, int]:
        from arena_api.robot_profiles import INNER

        grip = 1 if side == "r" else 0
        return self.sk.names.index(self.sk.joints.GRIPPERS[grip]), self.sk.names.index(INNER[grip])

    def _with_gripper(
        self,
        q: np.ndarray,
        side: str,
        value: float,
        *,
        width_m: float | None = None,
        opening: bool = False,
    ) -> np.ndarray:
        q = np.array(q, float, copy=True)
        gtype = self.sk.emb.gripper_type
        if gtype == "dex1":
            from rab_embodiment import DEX1_JAW_MAX_M, dex1_hand_targets, dex1_jaw_target

            if width_m is not None:
                if opening:
                    q_jaw = min(DEX1_JAW_MAX_M, max(0.014, dex1_jaw_target(width_m) + 0.020))
                    targets = {jn: q_jaw for jn in dex1_hand_targets(side, 1.0)}
                else:
                    targets = dex1_hand_targets(side, width_m=width_m)
            else:
                openness = float(np.clip((float(value) - GRIP_CLOSED) / max(1e-6, GRIP_OPEN - GRIP_CLOSED), 0.0, 1.0))
                targets = dex1_hand_targets(side, openness)
            for name, jval in targets.items():
                if name in self.sk.names:
                    q[..., self.sk.names.index(name)] = jval
            return q
        if gtype == "dex3":
            from rab_embodiment import dex3_hand_targets

            openness = float(np.clip((float(value) - GRIP_CLOSED) / max(1e-6, GRIP_OPEN - GRIP_CLOSED), 0.0, 1.0))
            for name, jval in dex3_hand_targets(side, openness).items():
                if name in self.sk.names:
                    q[..., self.sk.names.index(name)] = jval
            return q
        outer, inner = self._gripper_ix(side)
        q[..., outer], q[..., inner] = value, -value
        return q

    def _carry_joint_vector(
        self,
        q_stand: np.ndarray,
        side: str,
        *,
        width_m: float | None = None,
        q_close: float = GRIP_CLOSED,
    ) -> np.ndarray:
        """Comfortable close-body carry posture for holding an object while standing/walking."""
        q_out = self._with_gripper(q_stand, side, q_close, width_m=width_m, opening=False)
        if not self.sk.emb.supports_crouch:
            return q_out
        prefix_s = "right" if side == "r" else "left"
        roll_sign = -1.0 if side == "r" else 1.0
        carry_arm = {
            f"{prefix_s}_shoulder_pitch_joint": -0.32,
            f"{prefix_s}_shoulder_roll_joint": 0.20 * roll_sign,
            f"{prefix_s}_shoulder_yaw_joint": 0.0,
            f"{prefix_s}_elbow_joint": 1.12,
            f"{prefix_s}_wrist_roll_joint": 0.0,
            f"{prefix_s}_wrist_pitch_joint": -0.20,
            f"{prefix_s}_wrist_yaw_joint": 0.0,
        }
        all_lim = self.robot.get_dof_limits()[0]
        for jn, jv in carry_arm.items():
            if jn in self.sk.names:
                ji = self.sk.names.index(jn)
                q_out[ji] = float(np.clip(jv, all_lim[ji, 0], all_lim[ji, 1]))
        return q_out

    def _setup_existing_carry(self, pre: dict, stand_posture: Posture) -> tuple[np.ndarray, str | None]:
        """If an item was already held before a base/articulate skill, attach it to the carry hand."""
        q_base = pre["q"][0].copy()
        held = pre.get("held")
        if not held:
            return q_base, None
        held_body, held_side = held
        meta = self._carry_meta if (self._carry_meta and self._carry_meta.get("body") == held_body) else None
        width_m = meta.get("width_m") if meta else 0.055
        q_close = meta.get("q_close", GRIP_CLOSED) if meta else GRIP_CLOSED
        q_carry = self._carry_joint_vector(q_base, held_side, width_m=width_m, q_close=q_close)
        self._set_posture_and_joints(stand_posture, q_carry)
        self.world.step(0, sync=True)
        rel = meta["rel"] if (meta and "rel" in meta) else np.array([0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 1.0], float)
        self._attach(held_body, rel, held_side)
        return q_carry, held_side

    def _arm_weights(self, side: str) -> np.ndarray:
        w = np.zeros(len(self.sk.names))
        for name in self.sk.emb.arm_joints(side):
            if name in self.sk.names:
                w[self.sk.names.index(name)] = 1.0
        return w

    def _retime(self, path, side) -> np.ndarray:
        return retime_path(np.asarray(path, float), self.fps, vmax=1.8, tmin=0.45, weights=self._arm_weights(side))

    def _attach(self, body: str, rel: np.ndarray, side: str | None = None) -> None:
        self._set_gravity(body, True)
        self._touched_gravity.add(body)
        self.world.sim.flush_changes()
        self._attached = (body, np.asarray(rel, float), side or self._side)

    def _work_posture(self, target_p: np.ndarray, stand_posture: Posture, q_seed: np.ndarray) -> tuple[Posture, dict[str, float]]:
        """Compute validated crouch/bow posture (and leg/waist joints) for a manipulation target."""
        if not self.sk.emb.supports_crouch:
            return stand_posture, {}
        from rab_embodiment import compute_g1_crouch

        base_pair = (stand_posture[0], stand_posture[1])
        crouch_xy, crouch_dz, pelvis_pitch, crouch_j = self.sk._compute_crouch_plan(target_p, base_pair)
        if crouch_dz <= 1e-4 and pelvis_pitch <= 1e-4:
            return stand_posture, {}

        # Validate in PhysX that bent knees clear low furniture; fall back to pure hip-hinge bow if needed
        saved_q = self.robot.get_dof_positions().copy()
        saved_t = self.robot.get_dof_position_targets().copy()
        all_lim = self.robot.get_dof_limits()[0]
        test_q = np.asarray(q_seed, np.float32).reshape(1, -1).copy()
        for jn, jv in crouch_j.items():
            if jn in self.sk.names:
                ji = self.sk.names.index(jn)
                test_q[0, ji] = float(np.clip(jv, all_lim[ji, 0], all_lim[ji, 1]))
        chosen_posture: Posture = (crouch_xy.copy(), stand_posture[1], float(crouch_dz), float(pelvis_pitch))
        chosen_j = dict(crouch_j)
        try:
            self.sk.teleport_base(
                crouch_xy, stand_posture[1], crouch_dz=crouch_dz, pitch=pelvis_pitch,
                dof_positions=test_q, dof_targets=test_q,
            )
        except RuntimeError:
            _, bow_pitch, bow_j = compute_g1_crouch(float(target_p[2]), floor_z=float(stand_posture[0][2]), max_knee_x=0.0)
            bow_q = np.asarray(q_seed, np.float32).reshape(1, -1).copy()
            for jn, jv in bow_j.items():
                if jn in self.sk.names:
                    ji = self.sk.names.index(jn)
                    bow_q[0, ji] = float(np.clip(jv, all_lim[ji, 0], all_lim[ji, 1]))
            try:
                self.sk.teleport_base(
                    stand_posture[0], stand_posture[1], crouch_dz=0.0, pitch=bow_pitch,
                    dof_positions=bow_q, dof_targets=bow_q,
                )
                chosen_posture = (stand_posture[0].copy(), stand_posture[1], 0.0, float(bow_pitch))
                chosen_j = dict(bow_j)
            except RuntimeError:
                chosen_posture = stand_posture
                chosen_j = {}
        finally:
            self.sk.teleport_base(
                stand_posture[0], stand_posture[1], crouch_dz=0.0, pitch=0.0,
                dof_positions=saved_q, dof_targets=saved_t,
            )
        return chosen_posture, chosen_j

    def _ik_solve_6d(
        self,
        position: np.ndarray,
        rotmat: np.ndarray | None,
        side: str,
        *,
        posture: Posture | None = None,
        crouch_joints: dict[str, float] | None = None,
        seed_q: np.ndarray | None = None,
        track_seed: bool = False,
        pos_tol: float = 0.025,
        rot_tol: float = 0.16,
        w_rot: float = 0.25,
        max_iters: int = 140,
        damp: float = 0.024,
        step_clip: float = 0.10,
    ) -> tuple[float, float, list[np.ndarray]]:
        """Animation-only 6-DoF IK using cuRoboV2-Fast GPU IK when available, else weighted DLS-IK."""
        robot = self.robot
        target_p = np.asarray(position, float)
        target_R = None if rotmat is None else np.asarray(rotmat, float)

        # Fast GPU 6-DoF cuRoboV2-Fast path (zero PhysX stepping during planning!)
        if getattr(self.sk, "ik_backend", "dls") == "curobo" and getattr(self.sk, "curobo", None) is not None:
            from scipy.spatial.transform import Rotation

            q_base = (
                robot.get_dof_positions()[0].copy()
                if seed_q is None
                else np.asarray(seed_q, float).reshape(-1).copy()
            )
            if posture is not None:
                crouch_xy, yaw, crouch_dz, pelvis_pitch = posture
                crouch_j = dict(crouch_joints or {})
            else:
                saved_base = self.sk.base()
                yaw = float(saved_base[1])
                if self.sk.emb.supports_crouch:
                    crouch_xy, crouch_dz, pelvis_pitch, crouch_j = self.sk._compute_crouch_plan(target_p, saved_base)
                else:
                    crouch_xy, crouch_dz, pelvis_pitch, crouch_j = saved_base[0], 0.0, 0.0, {}
            pelvis_pos = np.array(
                [
                    float(crouch_xy[0]),
                    float(crouch_xy[1]),
                    float(crouch_xy[2]) + self.sk.emb.pelvis_z_offset - float(crouch_dz),
                ],
                float,
            )
            pelvis_R = Rotation.from_euler("ZY", [float(yaw), float(pelvis_pitch)]).as_matrix()
            seed_map = {jn: float(q_base[i]) for i, jn in enumerate(self.sk.names)}
            seed_map.update(crouch_j)
            ok, pos_err, rot_err, sol_map, _ = self.sk.curobo.solve_ik_6d(
                side,
                target_p,
                target_R if (target_R is not None and w_rot > 0.0) else None,
                pelvis_pos,
                pelvis_R,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                seed_joint_map=seed_map,
                track_seed=track_seed,
            )
            if not ok:
                raise ValueError(f"curobo_ik6d_failed:pos={pos_err:.4f}:rot={rot_err:.3f}")
            q_sol = q_base.copy()
            all_lim = robot.get_dof_limits()[0]
            for jname, jval in {**crouch_j, **sol_map}.items():
                if jname in self.sk.names:
                    ji = self.sk.names.index(jname)
                    q_sol[ji] = float(np.clip(jval, all_lim[ji, 0], all_lim[ji, 1]))
            return pos_err, rot_err, [q_base, q_sol]

        jmeta = robot.shared_metatype
        arm = [j for j in self.sk.emb.arm_joints(side) if j in self.sk.names]
        ix = [self.sk.names.index(x) for x in arm]
        link = self.sk.emb.ee_link(side)
        links = list(jmeta.link_names)
        lim = robot.get_dof_limits()[0, ix]
        idx = np.array([0], np.uint32)
        saved_q = robot.get_dof_positions().copy()
        saved_t = robot.get_dof_position_targets().copy()
        saved_base = self.sk.base()
        try:
            q_start = saved_q.copy() if seed_q is None else np.asarray(seed_q, np.float32).reshape(1, -1).copy()
            if posture is not None and self.sk.emb.supports_crouch:
                self._set_posture_and_joints(posture, q_start[0])
            else:
                robot.set_dof_positions(q_start, idx)
                robot.set_dof_velocities(np.zeros_like(q_start), idx)
                robot.set_dof_position_targets(q_start, idx)
            self.world.step(1, sync=True)
            path = [q_start[0].copy()]
            if not self.sk.emb.supports_crouch and seed_q is None and float(target_p[2]) < 0.45:
                seed = q_start.copy()
                if "idx01_body_joint1" in self.sk.names:
                    seed[0, self.sk.names.index("idx01_body_joint1")] = 0.02
                if "idx02_body_joint2" in self.sk.names:
                    seed[0, self.sk.names.index("idx02_body_joint2")] = 0.65
                robot.set_dof_positions(seed, idx)
                robot.set_dof_position_targets(seed, idx)
                self.world.step(2, sync=True)
                path.append(seed[0].copy())
            for _ in range(max_iters):
                pose = self._pinch_pose_world(side)
                _, _, r_world = self.sk.ee_world(side)
                e_pos = target_p - pose[:3]
                pos_err = float(np.linalg.norm(e_pos))
                if target_R is not None and w_rot > 0.0:
                    e_rot = rot_error_vec(target_R, quat_to_rotmat(pose[3:]))
                    rot_err = float(np.linalg.norm(e_rot))
                else:
                    e_rot, rot_err = np.zeros(3), 0.0
                if pos_err <= pos_tol and (target_R is None or w_rot <= 0.0 or rot_err <= rot_tol):
                    return pos_err, rot_err, path
                jac = robot.get_jacobians()[0]
                i = links.index(link) - (1 if jac.shape[0] == len(links) - 1 else 0)
                offset = jac.shape[-1] - len(self.sk.names)
                cols = np.array(ix) + offset
                j_lin = jac[i, :3, :][:, cols]
                if np.any(r_world):
                    rx = np.array(
                        [[0.0, -r_world[2], r_world[1]], [r_world[2], 0.0, -r_world[0]], [-r_world[1], r_world[0], 0.0]]
                    )
                    j_lin = j_lin - rx @ jac[i, 3:6, :][:, cols]
                if target_R is not None and w_rot > 0.0:
                    j_ang = jac[i, 3:6, :][:, cols]
                    j6 = np.vstack((j_lin, w_rot * j_ang))
                    e6 = np.r_[e_pos, w_rot * e_rot]
                    d = j6.T @ np.linalg.solve(j6 @ j6.T + np.eye(6) * (damp ** 2), e6)
                else:
                    d = j_lin.T @ np.linalg.solve(j_lin @ j_lin.T + np.eye(3) * (0.018 ** 2), e_pos)
                q = robot.get_dof_positions().copy()
                q[0, ix] = np.clip(q[0, ix] + np.clip(d, -step_clip, step_clip), lim[:, 0], lim[:, 1])
                robot.set_dof_positions(q, idx)
                robot.set_dof_position_targets(q, idx)
                self.world.step(1, sync=True)
                path.append(q[0].copy())
            pose = self._pinch_pose_world(side)
            pos_err = float(np.linalg.norm(target_p - pose[:3]))
            rot_err = (
                float(np.linalg.norm(rot_error_vec(target_R, quat_to_rotmat(pose[3:]))))
                if target_R is not None
                else 0.0
            )
            raise ValueError(f"ik6d_failed:pos={pos_err:.4f}:rot={rot_err:.3f}")
        finally:
            if posture is not None and self.sk.emb.supports_crouch:
                self._set_posture_and_joints((saved_base[0], float(saved_base[1]), 0.0, 0.0), saved_q[0])
            else:
                robot.set_dof_positions(saved_q, idx)
                robot.set_dof_velocities(np.zeros_like(saved_q), idx)
                robot.set_dof_position_targets(saved_t, idx)
            self.world.step(1, sync=True)

    def _cartesian_6d(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        rotmat: np.ndarray | None,
        side: str,
        seed_q: np.ndarray,
        *,
        posture: Posture | None = None,
        crouch_joints: dict[str, float] | None = None,
        n_sub: int = 6,
        w_rot: float = 0.25,
        pos_tol: float = 0.022,
        rot_tol: float = 0.18,
    ) -> list[np.ndarray]:
        """Solve a straight-line Cartesian segment by warm-starting 6-DoF IK along p0 -> p1."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        q_cur = np.asarray(seed_q, float).copy()
        out = [q_cur]
        for k in range(1, n_sub + 1):
            pt = p0 + (p1 - p0) * (k / n_sub)
            _, _, seg = self._ik_solve_6d(
                pt,
                rotmat,
                side,
                posture=posture,
                crouch_joints=crouch_joints,
                seed_q=q_cur,
                track_seed=True,
                w_rot=w_rot,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                max_iters=45,
            )
            q_cur = seg[-1]
            out.append(q_cur)
        return out

    # ------------------------------------------------------------- replays
    def _replay_base(self, plan: dict, pre: dict) -> None:
        stand_posture: Posture = (pre["base"][0].copy(), float(pre["base"][1]), 0.0, 0.0)
        q_base, carry_side = self._setup_existing_carry(pre, stand_posture)
        self._phase = "base"
        lim = BaseLimits(lin_vmax=0.95, ang_vmax=1.80, min_segment_s=0.20) if self.sk.emb.supports_crouch else BaseLimits()
        frames = base_frames(plan["poses"], self.fps, lim)
        if not frames:
            return
        if not self.sk.emb.supports_crouch:
            for x, y, z, yaw in frames:
                self.sk.teleport_base((x, y, z), yaw)
                self._set_robot(q_base)
                self._emit(0)
            return

        from rab_embodiment import unitree_walk_gait_joints

        all_lim = self.robot.get_dof_limits()[0]
        prev_x, prev_y, _, prev_yaw = float(pre["base"][0][0]), float(pre["base"][0][1]), float(pre["base"][0][2]), float(pre["base"][1])
        phase_rad = 0.0
        n_total = len(frames)
        for idx_f, (x, y, z, yaw) in enumerate(frames):
            ds_lin = math.hypot(x - prev_x, y - prev_y)
            ds_ang = abs(wrap_angle(yaw - prev_yaw))
            prev_x, prev_y, prev_yaw = x, y, yaw
            # Advance gait phase smoothly (~0.52 m per full stride cycle, ~1.1 rad per turn cycle)
            dphase = 2.0 * math.pi * (ds_lin / 0.52 + ds_ang / 1.15)
            phase_rad += dphase
            # Smooth start/stop envelope so feet settle cleanly together at segment boundaries
            edge_env = min(1.0, (idx_f + 1) / 4.0, (n_total - 1 - idx_f) / 4.0)
            speed_env = float(np.clip((ds_lin * self.fps) / 0.16 + (ds_ang * self.fps) / 0.45, 0.0, 1.0))
            amp = edge_env * speed_env
            q_step = q_base.copy()
            for jn, jv in unitree_walk_gait_joints(phase_rad, amp, carry_side=carry_side).items():
                if jn in self.sk.names:
                    ji = self.sk.names.index(jn)
                    q_step[ji] = float(np.clip(jv, all_lim[ji, 0], all_lim[ji, 1]))
            bob_dz = 0.018 * (amp * math.sin(phase_rad)) ** 2
            self._set_posture_and_joints((np.array([x, y, z], float), yaw, bob_dz, 0.0), q_step)
            self._emit(0)

    def _plan_grasp_6d(self, plan: dict, pre: dict, side: str, stand_posture: Posture):
        """Solve a 4-stage 6-DoF grasp (ready -> pre -> grasp -> lift) with crouch + graceful fallback."""
        p, yaw = stand_posture[0], stand_posture[1]
        cur_R = quat_to_rotmat(self._pinch_pose_world(side)[3:])
        box = plan.get("box")
        if box is None:
            t = np.asarray(plan["target"], float)
            box = [(t - 0.03).tolist(), (t + 0.03).tolist()]
        obj_pose = np.asarray(plan["object_pose"], float)
        primary = plan_grasp_pose(
            box, obj_pose, p, yaw, in_open_drawer=bool(plan.get("in_open_drawer")), cur_R=cur_R
        )
        work_posture, crouch_j = self._work_posture(primary.grasp_pos, stand_posture, pre["q"][0])
        alt_mode = "side" if primary.mode == "top_down" else "top_down"
        attempts = [
            (f"6dof_{primary.mode}", primary, 0.25, 0.18),
            (f"6dof_{primary.mode}_relaxed", primary, 0.12, 0.32),
        ]
        if not plan.get("in_open_drawer"):
            alt = plan_grasp_pose(box, obj_pose, p, yaw, in_open_drawer=False, cur_R=cur_R, force_mode=alt_mode)
            attempts.append((f"6dof_{alt.mode}", alt, 0.22, 0.22))
        for tag, gp, w_rot, rot_tol in attempts:
            try:
                _, _, to_pre = self._ik_solve_6d(
                    gp.pre_pos, gp.rotmat, side, posture=work_posture, crouch_joints=crouch_j,
                    w_rot=w_rot, pos_tol=0.030, rot_tol=rot_tol, max_iters=140,
                )
                to_grasp = self._cartesian_6d(
                    gp.pre_pos, gp.grasp_pos, gp.rotmat, side, to_pre[-1],
                    posture=work_posture, crouch_joints=crouch_j,
                    n_sub=6, w_rot=w_rot, pos_tol=0.022, rot_tol=rot_tol,
                )
                try:
                    to_lift = self._cartesian_6d(
                        gp.grasp_pos, gp.lift_pos, gp.rotmat, side, to_grasp[-1],
                        posture=work_posture, crouch_joints=crouch_j,
                        n_sub=5, w_rot=w_rot, pos_tol=0.030, rot_tol=rot_tol + 0.08,
                    )
                except ValueError:
                    to_lift = [to_grasp[-1], to_pre[-1]]
                return tag, gp, work_posture, to_pre, to_grasp, to_lift
            except ValueError:
                continue
        return "3dof_fallback", primary, work_posture, None, None, None

    def _replay_grasp(self, plan: dict, pre: dict, info: dict) -> None:
        side = self._side = plan["side"]
        q0 = pre["q"][0]
        stand_posture: Posture = (pre["base"][0].copy(), float(pre["base"][1]), 0.0, 0.0)
        solver, gp, work_posture, to_pre, to_grasp, to_lift = self._plan_grasp_6d(plan, pre, side, stand_posture)
        w_m = float(gp.pinch_width_m)
        info.update(
            grasp_solver=solver,
            grasp_mode=gp.mode,
            pinch_width_m=round(w_m, 4),
            crouch_dz_m=round(float(work_posture[2]), 3),
            q_open=round(gp.q_open, 3),
            q_close=round(gp.q_close, 3),
        )
        q_carry = self._carry_joint_vector(q0, side, width_m=w_m, q_close=gp.q_close)
        if to_pre is not None:
            q0_open = self._with_gripper(q0, side, gp.q_open, width_m=w_m, opening=True)
            p_pre = [self._with_gripper(q, side, gp.q_open, width_m=w_m, opening=True) for q in to_pre]
            p_approach = [self._with_gripper(q, side, gp.q_open, width_m=w_m, opening=True) for q in to_grasp]
            self._play_joints(blend(q0, q0_open, self.fps, 0.22), "open_gripper", posture=stand_posture)
            self._play_posture_transition(self._retime(p_pre, side), stand_posture, work_posture, "pre_grasp")
            self._play_joints(
                retime_path(p_approach, self.fps, vmax=0.45, tmin=0.45, weights=self._arm_weights(side)),
                "approach_grasp",
                posture=work_posture,
            )
            goal = p_approach[-1]
            closed_goal = self._with_gripper(goal, side, gp.q_close, width_m=w_m, opening=False)
            self._play_joints(blend(goal, closed_goal, self.fps, 0.32), "close_gripper", posture=work_posture)
            grip = self._pinch_pose_world(side)
            rel = relative(grip, np.asarray(plan["object_pose"], float))
            self._attach(plan["body"], rel, side)
            self._carry_meta = dict(
                body=plan["body"], side=side, mode=gp.mode, width_m=w_m,
                q_open=gp.q_open, q_close=gp.q_close, rel=rel,
            )
            p_lift = [self._with_gripper(q, side, gp.q_close, width_m=w_m, opening=False) for q in to_lift]
            self._play_joints(
                retime_path(p_lift, self.fps, vmax=0.45, tmin=0.40, weights=self._arm_weights(side)),
                "lift",
                posture=work_posture,
            )
            p_back = [p_lift[-1], q_carry]
            self._play_posture_transition(self._retime(p_back, side), work_posture, stand_posture, "retract")
            return

        # Fallback: reuse validated 3-DoF IK path from rab_skills.py
        info.setdefault("degraded", []).append("grasp_6dof_fallback_to_3dof")
        path = [self._with_gripper(q, side, gp.q_open, width_m=w_m, opening=True) for q in plan["ik_path"]] or [q0]
        q0_open = self._with_gripper(q0, side, gp.q_open, width_m=w_m, opening=True)
        self._play_joints(blend(q0, q0_open, self.fps, 0.30), "open_gripper", posture=stand_posture)
        reach = self._retime(path, side)
        self._play_posture_transition(reach, stand_posture, work_posture, "reach")
        goal = reach[-1] if len(reach) else path[-1]
        closed_goal = self._with_gripper(goal, side, gp.q_close, width_m=w_m, opening=False)
        self._play_joints(blend(goal, closed_goal, self.fps, 0.32), "close_gripper", posture=work_posture)
        grip = self._pinch_pose_world(side)
        rel = relative(grip, np.asarray(plan["object_pose"], float))
        self._attach(plan["body"], rel, side)
        self._carry_meta = dict(
            body=plan["body"], side=side, mode=gp.mode, width_m=w_m,
            q_open=gp.q_open, q_close=gp.q_close, rel=rel,
        )
        back = [closed_goal, q_carry]
        self._play_posture_transition(self._retime(back, side), work_posture, stand_posture, "lift_retract")

    def _replay_place(self, plan: dict, pre: dict, info: dict) -> None:
        from scipy.spatial.transform import Rotation

        side = self._side = plan["side"]
        q0 = pre["q"][0]
        stand_posture: Posture = (pre["base"][0].copy(), float(pre["base"][1]), 0.0, 0.0)
        release = np.asarray(plan["release_pose"], float)
        p, yaw = stand_posture[0], stand_posture[1]
        cur_R = quat_to_rotmat(self._pinch_pose_world(side)[3:])
        box = plan.get("box")
        if box is None:
            pt = np.asarray(plan["point"], float)
            box = [(pt - 0.03).tolist(), (pt + 0.03).tolist()]
        meta = self._carry_meta if self._carry_meta and self._carry_meta.get("body") == plan["body"] else None
        pp = plan_place_pose(
            box, plan["point"], p, yaw, cur_R=cur_R, preferred_mode=meta["mode"] if meta else None
        )
        ext = np.maximum(np.asarray(box, float).reshape(2, 3)[1] - np.asarray(box, float).reshape(2, 3)[0], 1e-3)
        w_m = float(meta["width_m"]) if (meta and "width_m" in meta) else float(min(ext[0], ext[1]))
        q_close = meta["q_close"] if meta else pp.q_close
        q_open = meta["q_open"] if meta else pp.q_open
        q_start_carry = self._carry_joint_vector(q0, side, width_m=w_m, q_close=q_close)
        work_posture, crouch_j = self._work_posture(pp.place_pos, stand_posture, q_start_carry)

        to_pre = to_down = to_lift = None
        solver = "3dof_fallback"
        for tag, w_rot, rot_tol in ((f"6dof_{pp.mode}", 0.25, 0.18), (f"6dof_{pp.mode}_relaxed", 0.12, 0.32)):
            try:
                _, _, to_pre = self._ik_solve_6d(
                    pp.pre_pos, pp.rotmat, side, posture=work_posture, crouch_joints=crouch_j,
                    seed_q=q_start_carry, w_rot=w_rot, pos_tol=0.030, rot_tol=rot_tol, max_iters=140,
                )
                to_down = self._cartesian_6d(
                    pp.pre_pos, pp.place_pos, pp.rotmat, side, to_pre[-1],
                    posture=work_posture, crouch_joints=crouch_j,
                    n_sub=5, w_rot=w_rot, pos_tol=0.022, rot_tol=rot_tol,
                )
                try:
                    to_lift = self._cartesian_6d(
                        pp.place_pos, pp.lift_pos, pp.rotmat, side, to_down[-1],
                        posture=work_posture, crouch_joints=crouch_j,
                        n_sub=4, w_rot=w_rot, pos_tol=0.030, rot_tol=rot_tol + 0.08,
                    )
                except ValueError:
                    to_lift = [to_down[-1], to_pre[-1]]
                solver = tag
                break
            except ValueError:
                to_pre = to_down = to_lift = None
        info.update(place_solver=solver, place_mode=pp.mode, crouch_dz_m=round(float(work_posture[2]), 3))

        if to_pre is not None:
            self._set_posture_and_joints(stand_posture, q_start_carry)
            self.world.step(0, sync=True)
            q_place = Rotation.from_matrix(pp.rotmat).as_quat()
            carry = relative(np.r_[pp.place_pos, q_place], release)
            self._attach(plan["body"], carry, side)
            p_pre = [q_start_carry, *[self._with_gripper(q, side, q_close, width_m=w_m, opening=False) for q in to_pre[1:]]]
            p_down = [self._with_gripper(q, side, q_close, width_m=w_m, opening=False) for q in to_down]
            self._play_posture_transition(self._retime(p_pre, side), stand_posture, work_posture, "carry_pre_place")
            self._play_joints(
                retime_path(p_down, self.fps, vmax=0.40, tmin=0.40, weights=self._arm_weights(side)),
                "descend_place",
                posture=work_posture,
            )
            bottom = p_down[-1]
            bottom_open = self._with_gripper(bottom, side, q_open, width_m=w_m, opening=True)
            self._play_joints(blend(bottom, bottom_open, self.fps, 0.28), "open_gripper", posture=work_posture)
            body = self._attached[0]
            self._attached = None
            start = self.world.pose(body)
            self._phase = "settle"
            n = frame_count(0.25, self.fps)
            for i in range(n):
                self._apply_poses({body: lerp_pose(start, release, float(min_jerk((i + 1) / n)))})
                self._emit(1)
            p_lift = [self._with_gripper(q, side, q_open, width_m=w_m, opening=True) for q in to_lift]
            self._play_joints(
                retime_path(p_lift, self.fps, vmax=0.45, tmin=0.35, weights=self._arm_weights(side)),
                "lift_clear",
                posture=work_posture,
            )
            q0_open = self._with_gripper(q0, side, q_open, width_m=w_m, opening=True)
            p_back = [p_lift[-1], q0_open]
            self._play_posture_transition(self._retime(p_back, side), work_posture, stand_posture, "retract")
            self._carry_meta = None
            return

        # Fallback: reuse validated 3-DoF IK path from rab_skills.py
        info.setdefault("degraded", []).append("place_6dof_fallback_to_3dof")
        carry = np.r_[0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]
        grip = self._pinch_pose_world(side)
        carry[3:] = relative(grip, release)[3:]
        self._attach(plan["body"], carry, side)
        path = [self._with_gripper(q, side, q_close, width_m=w_m, opening=False) for q in plan["ik_path"]]
        self._play_posture_transition(self._retime(path, side), stand_posture, work_posture, "carry")
        top = path[-1]
        top_open = self._with_gripper(top, side, q_open, width_m=w_m, opening=True)
        self._play_joints(blend(top, top_open, self.fps, 0.28), "open_gripper", posture=work_posture)
        body = self._attached[0]
        self._attached = None
        start = self.world.pose(body)
        self._phase = "release"
        n = frame_count(0.35, self.fps)
        for i in range(n):
            self._apply_poses({body: lerp_pose(start, release, float(min_jerk((i + 1) / n)))})
            self._emit(1)
        q0_open = self._with_gripper(q0, side, q_open, width_m=w_m, opening=True)
        back = [top_open, q0_open]
        self._play_posture_transition(self._retime(back, side), work_posture, stand_posture, "retract")
        self._carry_meta = None

    def _replay_articulate(self, plan: dict, pre: dict, info: dict) -> None:
        from rab_skills import Refused

        w, sk = self.world, self.sk
        stand_posture: Posture = (pre["base"][0].copy(), float(pre["base"][1]), 0.0, 0.0)
        q_start, carry_side = self._setup_existing_carry(pre, stand_posture)
        handle = np.asarray(plan["handle"], float)
        p, yaw = stand_posture[0], stand_posture[1]
        if carry_side is not None:
            side = "l" if carry_side == "r" else "r"
        else:
            side = "l" if np.dot(handle[:2] - p[:2], [-math.sin(yaw), math.cos(yaw)]) > 0.06 else "r"
        self._side = side
        toward = np.r_[math.cos(yaw), math.sin(yaw), 0.0]
        cur_R = quat_to_rotmat(self._pinch_pose_world(side)[3:])
        handle_R = gripper_frame(toward, [0.0, 0.0, 1.0], cur_R=cur_R)
        work_posture, crouch_j = self._work_posture(handle - 0.05 * toward, stand_posture, q_start)
        info["crouch_dz_m"] = round(float(work_posture[2]), 3)

        approach: list = []
        try:
            _, _, approach = self._ik_solve_6d(
                handle - 0.05 * toward,
                handle_R,
                side,
                posture=work_posture,
                crouch_joints=crouch_j,
                seed_q=q_start,
                w_rot=0.20,
                pos_tol=0.035,
                rot_tol=0.28,
                max_iters=130,
            )
        except ValueError:
            try:
                _, approach = sk._ik_solve(handle - 0.05 * toward, side, tolerance=0.04, record=True)
            except Refused as e:
                info.setdefault("degraded", []).append(f"handle_ik:{e.reason}")
        if approach:
            approach = [self._with_gripper(q, side, GRIP_OPEN, width_m=0.030, opening=True) for q in approach]
            self._play_posture_transition(self._retime(approach, side), stand_posture, work_posture, "reach_handle")
            hold = approach[-1]
            hold_closed = self._with_gripper(hold, side, GRIP_CLOSED, width_m=0.022, opening=False)
            self._play_joints(blend(hold, hold_closed, self.fps, 0.28), "grip_handle", posture=work_posture)
        else:
            self._set_posture_and_joints(work_posture, q_start)

        drive, _, body, _, _ = w.drives[plan["key"]]
        t0 = w.pose(body)
        handle_local = relative(t0, np.r_[handle - 0.05 * toward, 0.0, 0.0, 0.0, 1.0])
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
                    _, _, follow = self._ik_solve_6d(
                        goal,
                        handle_R,
                        side,
                        posture=work_posture,
                        crouch_joints=crouch_j,
                        seed_q=q_prev,
                        track_seed=True,
                        w_rot=0.18,
                        pos_tol=0.030,
                        rot_tol=0.30,
                        max_iters=35,
                    )
                    if follow:
                        q_f = self._with_gripper(follow[-1], side, GRIP_CLOSED, width_m=0.022, opening=False)
                        self._set_posture_and_joints(work_posture, q_f)
                except ValueError:
                    pass
            self._emit(steps)
        if approach:
            q_now = self.robot.get_dof_positions()[0]
            q_rel = self._with_gripper(q_now, side, GRIP_OPEN, width_m=0.030, opening=True)
            self._play_joints(blend(q_now, q_rel, self.fps, 0.25), "release_handle", posture=work_posture)
            retract_frames = retime_path([q_rel, q_start], self.fps, vmax=0.9, tmin=0.55, weights=self._arm_weights(side))
            self._play_posture_transition(retract_frames, work_posture, stand_posture, "retract")

    # -------------------------------------------------------------- driver
    def replay(
        self,
        action_index: int,
        raw: Any,
        code: str,
        pre: dict | None,
        plan: dict,
    ) -> dict | None:
        skill = raw.get("skill") if isinstance(raw, dict) else str(raw)
        rec = self._recorder()
        if rec is None:
            return None
        rec.begin(action_index, skill, raw if isinstance(raw, dict) else {}, code)
        info: dict[str, Any] = dict(mode="replay", animated=False)
        if code != "EXECUTED" or pre is None or not plan:
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
