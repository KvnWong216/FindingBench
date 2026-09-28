"""Pinocchio-backed kinematics: the real, URDF-derived embodiment backend.

Responsibilities (real computation, no proxies):

- URDF parsing: link frames, joint tree, joint limits (joint_limits_source=urdf)
- forward kinematics for every link
- numerical SE(3) IK: damped-least-squares CLIK, joint-limit projection,
  deterministic multi-seed restarts, fine-grained failure attribution
  (NO_IK_SOLUTION / JOINT_LIMIT / NUMERICAL_FAILURE)
- URDF <collision> geometry as coal (hpp-fcl) bodies for configuration-space
  collision checking (visual meshes are never used)

Targets are expressed in the ROBOT BASE frame; the feasibility pipeline
converts world-frame interaction poses through the current base pose. The
class is fully deterministic: restart randomness comes from a fixed seed.

No motion planning happens here: single-configuration queries only.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from rummagebench.feasibility.ik_solver import (
    IKFailureReason,
    IKResult,
    Pose,
)

logger = logging.getLogger(__name__)


class PinocchioKinematics:
    """Fixed-base kinematics model loaded from a URDF via Pinocchio.

    Joints not listed in ``controlled_joints`` (wheels, caster joints, ... of
    a mobile-manipulator URDF) are locked at the URDF rest pose, so the model
    root frame equals ``base_link`` and the IK search space is exactly the
    controlled arm chain.
    """

    def __init__(
        self,
        urdf_path: str,
        base_link: str,
        eef_link: str,
        controlled_joints: list[str] | None = None,
        gripper_links: list[str] | None = None,
        pos_tol: float = 0.005,
        rot_tol: float = 0.0873,
        max_iters: int = 200,
        restarts: int = 8,
        seed: int = 0,
        damping: float = 1e-3,
        collision_padding: float = 0.0,
        free_tool_roll: bool = True,
    ):
        import pinocchio as pin

        self._pin = pin
        self.urdf_path = str(urdf_path)
        self.base_link = base_link
        self.eef_link = eef_link
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)
        self.max_iters = int(max_iters)
        self.restarts = int(restarts)
        self.seed = int(seed)
        self.damping = float(damping)
        self.collision_padding = float(collision_padding)
        self.free_tool_roll = bool(free_tool_roll)

        full_model = pin.buildModelFromUrdf(self.urdf_path)
        lock_ids = self._joints_to_lock(full_model, controlled_joints)
        self._locked_joint_names = [full_model.names[j] for j in lock_ids]
        if lock_ids:
            self.model = pin.buildReducedModel(
                full_model, lock_ids, np.zeros(full_model.nq)
            )
        else:
            self.model = full_model
        self.data = self.model.createData()

        self._validate_frames()
        self.frame_id = self.model.getFrameId(eef_link, pin.FrameType.BODY)
        self._validate_eef_in_controlled_chain(controlled_joints)

        # joint limits (urdf-sourced); +-inf = continuous joint (no clamp)
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.lower[~np.isfinite(self.lower)] = -np.pi
        self.upper[~np.isfinite(self.upper)] = np.pi

        # neutral pose: mid-range where finite, 0 clamped into range otherwise
        lo_c = np.where(np.isfinite(np.asarray(self.model.lowerPositionLimit, dtype=float)),
                        self.lower, 0.0)
        up_c = np.where(np.isfinite(np.asarray(self.model.upperPositionLimit, dtype=float)),
                        self.upper, 0.0)
        self.q_neutral = np.clip((lo_c + up_c) / 2.0, self.lower, self.upper)

        # collision geometry: URDF <collision> bodies on the reduced tree
        self.geom_model = pin.buildGeomFromUrdf(
            self.model, self.urdf_path, pin.GeometryType.COLLISION
        )
        self.geom_data = self.geom_model.createData()

        # exact kinematic reach bound of the controlled chain: for a chain of
        # revolute joints, max |p_eef| <= sum of ||joint origin translations||
        # (+ prismatic travel). A target beyond this bound is PROVABLY
        # NO_IK_SOLUTION — skipping the numerical search there is
        # semantics-preserving (identical observable result, no restart cost).
        reach = 0.0
        for jid in range(1, self.model.njoints):
            origin_t = self.model.jointPlacements[jid].translation
            reach += float(np.linalg.norm(origin_t))
            jtype = str(self.model.joints[jid].shortname())
            if "Prismatic" in jtype or "prismatic" in jtype:
                lo_j = float(self.model.lowerPositionLimit[self.model.joints[jid].idx_q])
                hi_j = float(self.model.upperPositionLimit[self.model.joints[jid].idx_q])
                if math.isfinite(lo_j) and math.isfinite(hi_j):
                    reach += abs(hi_j - lo_j)
                else:
                    reach += 2.0 * math.pi  # unbounded prismatic (unphysical)
        self.max_reach = reach + self.pos_tol

        self._link_classes = self._compute_link_classes(gripper_links)
        self._req: Any = None
        self._dreq: Any = None

    # ------------------------------------------------------------ validation

    def _joints_to_lock(self, model, controlled_joints) -> list[int]:
        """Movable joints absent from controlled_joints are locked (id 0 =
        universe is never a lock candidate).

        controlled_joints="auto" derives the chain from the asset: every
        movable joint on the kinematic path from the base to the end-effector
        frame (torso/arm joints of a mobile manipulator), so wheels/casters/
        suspension are locked by construction.
        """
        movable = [
            jid
            for jid in range(1, model.njoints)
            if model.joints[jid].nv > 0
        ]
        if controlled_joints is None:
            return []
        if controlled_joints == "auto":
            keep = self._auto_controlled_chain(model)
            return [jid for jid in movable if jid not in keep]
        names = list(model.names)
        keep = set(controlled_joints)
        missing = keep - set(names)
        if missing:
            raise ValueError(
                f"controlled_joints not present in URDF {self.urdf_path}: {sorted(missing)}"
            )
        keep_ids = {names.index(n) for n in keep}
        self._explicit_controlled_ids = keep_ids
        return [jid for jid in movable if jid not in keep_ids]

    def _auto_controlled_chain(self, model) -> set[int]:
        """Movable joint ids on the path from the root to the EEF joint."""
        frame_id = model.getFrameId(self.eef_link, self._pin.FrameType.BODY)
        if frame_id >= model.nframes:
            raise ValueError(
                f"end_effector_link {self.eef_link!r} not found in URDF {self.urdf_path}"
            )
        eef_joint = model.frames[frame_id].parentJoint
        chain: set[int] = set()
        j = eef_joint
        while j > 0:
            if model.joints[j].nv > 0:
                chain.add(j)
            j = model.parents[j]
        return chain

    def _validate_frames(self) -> None:
        pin = self._pin
        n_frames = self.model.nframes
        base_id = self.model.getFrameId(self.base_link, pin.FrameType.BODY)
        if base_id >= n_frames:
            raise ValueError(
                f"base_link {self.base_link!r} not found in URDF {self.urdf_path}"
            )
        base_frame = self.model.frames[base_id]
        if base_frame.parentJoint != 0:
            raise ValueError(
                f"base_link {self.base_link!r} is not the kinematic root; "
                "the URDF root link must be the robot base (lock mobile-base "
                "joints via controlled_joints instead)."
            )
        eef_id = self.model.getFrameId(self.eef_link, pin.FrameType.BODY)
        if eef_id >= n_frames:
            raise ValueError(
                f"end_effector_link {self.eef_link!r} not found in URDF {self.urdf_path}"
            )

    def _validate_eef_in_controlled_chain(self, controlled_joints) -> None:
        """§3: the EEF must be reachable by the controlled chain, otherwise
        IK could never move it — fail loudly instead of grounding blindly."""
        pin = self._pin
        eef_joint = self.model.frames[self.frame_id].parentJoint
        controlled_ids = set(range(1, self.model.njoints))
        if controlled_joints == "auto" or controlled_joints is None:
            return  # chain derived from / includes the EEF path by construction
        j = eef_joint
        while j > 0:
            if j in getattr(self, "_explicit_controlled_ids", set()):
                return
            j = self.model.parents[j]
        raise ValueError(
            f"end_effector_link {self.eef_link!r} (joint "
            f"{self.model.names[eef_joint]!r}) is not under the controlled "
            f"joint chain {sorted(controlled_joints)} — IK could never move "
            "it. Fix controlled_joints or end_effector_link."
        )

    def chain_info(self) -> dict:
        """§3 inspect payload: base link, controlled chain, locked joints.

        Locked = joints present in the source URDF but excluded from the
        kinematic model (mobile base, suspension, or a restricted variant's
        fixed distal joints).
        """
        controlled = [self.model.names[j] for j in range(1, self.model.njoints)]
        return {
            "base_link": self.base_link,
            "eef_link": self.eef_link,
            "controlled_joints": controlled,
            "locked_joints": list(self._locked_joint_names),
        }

    def _compute_link_classes(self, gripper_links: list[str] | None) -> dict[str, str]:
        """Allowed-collision-matrix link classes for every collision body.

        Explicit configuration wins; otherwise gripper class = the EEF link
        itself, its frame descendants (fingers hang below the tool frame) and
        links whose name matches a gripper hint.
        """
        pin = self._pin
        classes: dict[str, str] = {}
        if gripper_links:
            gripper = set(gripper_links) | {self.eef_link}
        else:
            eef_joint = self.model.frames[self.frame_id].parentJoint
            subtree: set[int] = set()
            for jid in range(self.model.njoints):
                j = jid
                while j > 0:
                    if j == eef_joint:
                        subtree.add(jid)
                        break
                    j = self.model.parents[j]
            gripper = {self.eef_link}
            for jid in subtree:
                gripper.add(self.model.names[jid])
            from rummagebench.robots.model_loader import _GRIPPER_NAME_HINTS

            for fname in (f.name for f in self.model.frames):
                if any(h in fname.lower() for h in _GRIPPER_NAME_HINTS):
                    gripper.add(fname)

        for go in self.geom_model.geometryObjects:
            link = self._link_name_of(go)
            classes[link] = "gripper" if link in gripper else "arm"
        return classes

    def _link_name_of(self, go) -> str:
        # parentFrame is the frame index in pinocchio >= 4
        frame = self.model.frames[go.parentFrame]
        return frame.name if frame is not None else go.name

    # ---------------------------------------------------------------- queries

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def controlled_joint_names(self) -> list[str]:
        return list(self.model.names[1:])

    @property
    def link_classes(self) -> dict[str, str]:
        return dict(self._link_classes)

    def q_seed_neutral(self) -> np.ndarray:
        return self.q_neutral.copy()

    def _se3_to_pose(self, M) -> Pose:
        pin = self._pin
        quat = np.asarray(pin.Quaternion(np.asarray(M.rotation, dtype=float)).coeffs(), dtype=float)
        return Pose(np.asarray(M.translation, dtype=float).copy(), quat)

    def quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """xyzw quaternion (Pose convention) -> 3x3 rotation matrix."""
        pin = self._pin
        return np.asarray(
            pin.Quaternion(np.asarray(quat, dtype=float)).toRotationMatrix(), dtype=float
        )

    def _pose_to_se3(self, pose: Pose):
        pin = self._pin
        R = np.asarray(pin.Quaternion(
            np.asarray(pose.orientation, dtype=float)
        ).toRotationMatrix(), dtype=float)
        return pin.SE3(R, np.asarray(pose.position, dtype=float))

    def forward_kinematics(self, q: np.ndarray) -> dict[str, Pose]:
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(self.model, self.data)
        out: dict[str, Pose] = {}
        for fid in range(self.model.nframes):
            frame = self.model.frames[fid]
            if frame.type == pin.FrameType.BODY:
                out[frame.name] = self._se3_to_pose(self.data.oMf[fid])
        return out

    def eef_pose(self, q: np.ndarray) -> Pose:
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(self.model, self.data)
        return self._se3_to_pose(self.data.oMf[self.frame_id])

    # --------------------------------------------------------------------- IK

    def solve_ik(
        self,
        target_pose: Pose,
        seed_q: np.ndarray | None = None,
    ) -> IKResult:
        """Damped-least-squares CLIK with joint-limit projection and
        deterministic multi-seed restarts.

        Seeds: current joint state (if provided) -> neutral -> seeded random
        restarts. The first converging seed wins; failure attribution
        distinguishes joint-limit blockage from genuine unreachability by
        re-running the search with limit projection disabled.
        """
        target = self._pose_to_se3(target_pose)
        # provable unreachability: beyond the chain's kinematic reach bound
        target_dist = float(np.linalg.norm(target.translation))
        if target_dist > self.max_reach:
            return IKResult(
                success=False,
                q=None,
                reason=IKFailureReason.NO_IK_SOLUTION,
                position_error=target_dist - self.max_reach + self.pos_tol,
                orientation_error=None,
                details={"beyond_kinematic_reach": True, "max_reach": self.max_reach},
            )
        seeds: list[np.ndarray] = []
        if seed_q is not None:
            seeds.append(np.clip(np.asarray(seed_q, dtype=float), self.lower, self.upper))
        seeds.append(self.q_neutral.copy())
        rng = np.random.default_rng(self.seed)
        for _ in range(self.restarts):
            seeds.append(rng.uniform(self.lower, self.upper))

        best: IKResult | None = None
        for idx, seed in enumerate(seeds):
            result = self._ik_from_seed(target, seed, enforce_limits=True)
            result.details["seed_index"] = idx
            if result.success:
                return result
            if best is None or (
                result.position_error is not None
                and best.position_error is not None
                and result.position_error < best.position_error
            ):
                best = result

        # attribution pass: would the target be solvable without joint limits?
        # (multi-seed, same deterministic sequence)
        relaxed_ok = False
        for seed in seeds:
            relaxed = self._ik_from_seed(target, seed, enforce_limits=False)
            if relaxed.success:
                relaxed_ok = True
                break
        if relaxed_ok:
            return IKResult(
                success=False,
                q=None,
                reason=IKFailureReason.JOINT_LIMIT,
                position_error=best.position_error if best else None,
                orientation_error=best.orientation_error if best else None,
                details={"clamped_solution_exists": True, **(best.details if best else {})},
            )
        reason = best.reason if best is not None else IKFailureReason.NO_IK_SOLUTION
        return IKResult(
            success=False,
            q=None,
            reason=reason if reason in (IKFailureReason.NO_IK_SOLUTION, IKFailureReason.NUMERICAL_FAILURE)
            else IKFailureReason.NO_IK_SOLUTION,
            position_error=best.position_error if best else None,
            orientation_error=best.orientation_error if best else None,
            details={**(best.details if best else {})},
        )

    def _ik_from_seed(self, target, q0: np.ndarray, enforce_limits: bool) -> IKResult:
        pin = self._pin
        q = np.asarray(q0, dtype=float).copy()
        eps = 1e-10
        best_pos_err = float("inf")
        best_rot_err = float("inf")
        for _ in range(self.max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.frame_id]
            err = pin.log6(current.actInv(target)).vector
            # 5D interaction constraint: roll about the tool z-axis (the
            # approach axis) is free — grippers are symmetric about it. The
            # LOCAL frame z IS the tool z, so drop the local angular-z term.
            if self.free_tool_roll:
                err = err.copy()
                err[5] = 0.0
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err < best_pos_err:
                best_pos_err, best_rot_err = pos_err, rot_err
            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                return IKResult(
                    success=True,
                    q=q.copy(),
                    reason=None,
                    position_error=pos_err,
                    orientation_error=rot_err,
                )
            if not np.all(np.isfinite(err)) or not np.all(np.isfinite(q)):
                return IKResult(
                    success=False,
                    q=None,
                    reason=IKFailureReason.NUMERICAL_FAILURE,
                    position_error=best_pos_err,
                    orientation_error=best_rot_err,
                )
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, pin.LOCAL
            )
            JT = J.T
            A = J @ JT + (self.damping**2) * np.eye(6)
            try:
                dq = JT @ np.linalg.solve(A, err)
            except np.linalg.LinAlgError:
                return IKResult(
                    success=False,
                    q=None,
                    reason=IKFailureReason.NUMERICAL_FAILURE,
                    position_error=best_pos_err,
                    orientation_error=best_rot_err,
                )
            # step clamp for stability
            step_norm = float(np.linalg.norm(dq))
            max_step = 0.5
            if step_norm > max_step:
                dq *= max_step / step_norm
            q = pin.integrate(self.model, q, dq)
            if enforce_limits:
                q = np.clip(q, self.lower, self.upper)
            # divergence guard
            if pos_err > 10.0 * max(best_pos_err, 1.0) + 10.0:
                break
        return IKResult(
            success=False,
            q=None,
            reason=IKFailureReason.NO_IK_SOLUTION,
            position_error=best_pos_err if best_pos_err != float("inf") else None,
            orientation_error=best_rot_err if best_rot_err != float("inf") else None,
            details={"iters": self.max_iters},
        )

    # ------------------------------------------------- collision body access
    # Used by hpp_fcl_checker.PinocchioCollisionChecker: URDF <collision>
    # geometries placed at their world (base-frame) poses for configuration q.

    def collision_bodies(self, q: np.ndarray) -> list[tuple[str, Any]]:
        """[(link_name, coal.CollisionObject at base-frame pose)] for every
        URDF <collision> body. Visual meshes are never loaded."""
        pin = self._pin
        coal = _coal()
        pin.updateGeometryPlacements(
            self.model, self.data, self.geom_model, self.geom_data,
            np.asarray(q, dtype=float),
        )
        bodies: list[tuple[str, Any]] = []
        for i, go in enumerate(self.geom_model.geometryObjects):
            M = self.geom_data.oMg[i]
            T = coal.Transform3s(np.asarray(M.translation, dtype=float))
            T.setRotation(np.asarray(M.rotation, dtype=float))
            bodies.append((self._link_name_of(go), coal.CollisionObject(go.geometry, T)))
        return bodies

    def collision_geometry_names(self) -> list[str]:
        return [self._link_name_of(go) for go in self.geom_model.geometryObjects]


def _coal():
    """Import coal (hpp-fcl's successor shipped with the pin wheel)."""
    try:
        import coal
        return coal
    except ImportError as e:  # pragma: no cover - old pin wheels expose hppfcl
        try:
            import pinocchio as pin
            return pin.hppfcl
        except Exception:
            raise ImportError(
                "coal/hpp-fcl is required for configuration-space collision "
                "checking (pip install pin provides it)"
            ) from e
