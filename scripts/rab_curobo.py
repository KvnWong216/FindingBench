"""CUDA-accelerated Unitree G1 IK and collision-free trajectory backend via NVLabs cuRoboV2-Fast.

Pluggable backend selected via ``FINDINGBENCH_RAB_IK_BACKEND``:
- ``curobo`` (default for ``unitree_g1`` / ``unitree_g1_dex3`` when ``curobo`` is installed):
  Uses NVLabs cuRoboV2-Fast GPU multi-seed IK (``InverseKinematics`` with CUDA graphs)
  and optional B-spline ``MotionPlanner`` on the pelvis-rooted 10-DoF chain
  (3 waist + 7 arm joints) with exact fingertip pinch tool frames and pre-fitted
  Unitree G1 self-collision spheres (``unitree_g1_29dof.yml`` for Dex1-1 2-finger
  parallel gripper, ``unitree_g1.yml`` for Dex3 3-finger hand). Zero PhysX stepping
  during IK search.
- ``dls``: Legacy live-PhysX damped-least-squares Jacobian IK.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CUROBO_G1_29DOF_YML = ROOT / "references/curobo/curobo/content/configs/robot/unitree_g1_29dof.yml"
CUROBO_G1_DEX3_YML = ROOT / "references/curobo/curobo/content/configs/robot/unitree_g1.yml"
# Default Unitree G1 YAML points to the 29-DoF + parallel-gripper-compatible config
CUROBO_G1_YML = CUROBO_G1_29DOF_YML

# Upper-body collision links (excludes lower legs so base proximity to low tables
# is governed cleanly by the simulator's base/knee clearance checks, while torso,
# head, arms, and wrists/hands get full self-collision checking).
UPPER_BODY_COLLISION_LINKS = {
    "pelvis",
    "pelvis_contour_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "logo_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "left_hand_palm_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
    "right_hand_palm_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
}


def _yml_for_embodiment(embodiment_name: str = "unitree_g1") -> Path:
    return CUROBO_G1_DEX3_YML if embodiment_name == "unitree_g1_dex3" else CUROBO_G1_29DOF_YML


def curobo_available(embodiment_name: str = "unitree_g1") -> bool:
    try:
        import curobo.inverse_kinematics  # noqa: F401
        return _yml_for_embodiment(embodiment_name).is_file()
    except Exception:
        return False


def selected_ik_backend(embodiment_name: str = "unitree_g1") -> str:
    raw = os.environ.get("FINDINGBENCH_RAB_IK_BACKEND", "auto").strip().lower()
    is_g1 = embodiment_name.startswith("unitree_g1")
    if raw in ("dls", "physx", "legacy"):
        return "dls"
    if raw in ("curobo", "curobo_v2", "curobov2"):
        if not is_g1:
            return "dls"
        if not curobo_available(embodiment_name):
            raise RuntimeError("FINDINGBENCH_RAB_IK_BACKEND=curobo requested, but cuRoboV2 is not available")
        return "curobo"
    if raw == "auto":
        return "curobo" if (is_g1 and curobo_available(embodiment_name)) else "dls"
    raise ValueError(f"Unknown FINDINGBENCH_RAB_IK_BACKEND={raw!r}; expected 'auto', 'curobo', or 'dls'")


def _build_arm_robot_cfg(side: str, embodiment_name: str = "unitree_g1") -> dict[str, Any]:
    """Build pelvis-rooted 10-DoF (3 waist + 7 arm) cuRoboV2 robot config for Unitree G1."""
    import yaml
    from rab_embodiment import (
        UNITREE_LEFT_ARM,
        UNITREE_READY_POSE,
        UNITREE_RIGHT_ARM,
        UNITREE_TORSO_JOINTS,
        get_embodiment,
    )

    emb = get_embodiment(embodiment_name)
    yml_path = _yml_for_embodiment(emb.name)
    cfg_dict = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    kin = cfg_dict["kinematics"]
    kin["base_link"] = "pelvis"

    # Keep only upper-body collision links + spheres
    kin["collision_link_names"] = [
        ln for ln in kin.get("collision_link_names", []) if ln in UPPER_BODY_COLLISION_LINKS
    ]
    if "collision_spheres" in kin and isinstance(kin["collision_spheres"], dict):
        kin["collision_spheres"] = {
            k: v for k, v in kin["collision_spheres"].items() if k in UPPER_BODY_COLLISION_LINKS
        }
    if "self_collision_buffer" in kin and isinstance(kin["self_collision_buffer"], dict):
        kin["self_collision_buffer"] = {
            k: v for k, v in kin["self_collision_buffer"].items() if k in UPPER_BODY_COLLISION_LINKS
        }
    if "self_collision_ignore" in kin and isinstance(kin["self_collision_ignore"], dict):
        cleaned = {}
        for k, vals in kin["self_collision_ignore"].items():
            if k in UPPER_BODY_COLLISION_LINKS:
                cleaned[k] = [v for v in vals if v in UPPER_BODY_COLLISION_LINKS]
        kin["self_collision_ignore"] = cleaned

    # Exact fingertip pinch center tool frame matching rab_embodiment.get_embodiment(emb.name).
    # Oriented to match rab_motion.gripper_frame convention:
    #   +Z_pinch = +X_wrist (along parallel jaws / approach axis)
    #   +X_pinch = +Y_wrist (jaw opening/closing pinch axis)
    #   +Y_pinch = +Z_wrist (palm plane normal)  =>  q_wxyz = [0.5, 0.5, 0.5, 0.5]
    off = emb.ee_offset(side).tolist()
    tool_name = "right_pinch_frame" if side == "r" else "left_pinch_frame"
    parent_link = emb.ee_link(side)
    kin["extra_links"] = {
        tool_name: {
            "parent_link_name": parent_link,
            "link_name": tool_name,
            "joint_name": f"{tool_name}_joint",
            "fixed_transform": [float(off[0]), float(off[1]), float(off[2]), 0.5, 0.5, 0.5, 0.5],
            "joint_type": "FIXED",
        }
    }
    kin["tool_frames"] = [tool_name]

    arm_set = set(UNITREE_TORSO_JOINTS) | set(UNITREE_RIGHT_ARM if side == "r" else UNITREE_LEFT_ARM)
    cspace = kin["cspace"]
    orig_names = list(cspace["joint_names"])
    lock_joints: dict[str, float] = {}
    keep_idx: list[int] = []
    for i, jn in enumerate(orig_names):
        if jn.startswith("base_j_") or any(tok in jn for tok in ("_hip_", "_knee_", "_ankle_")):
            continue
        if jn in arm_set:
            keep_idx.append(i)
        else:
            lock_joints[jn] = float(UNITREE_READY_POSE.get(jn, 0.0))
    kin["lock_joints"] = lock_joints

    for field, val in list(cspace.items()):
        if isinstance(val, list) and len(val) == len(orig_names):
            cspace[field] = [val[i] for i in keep_idx]

    cspace["default_joint_position"] = [
        float(UNITREE_READY_POSE.get(jn, 0.0)) for jn in cspace["joint_names"]
    ]
    # Strongly regularize waist yaw/roll so cuRobo prefers natural arm reach over 90+ deg torso twists
    weights = []
    for jn in cspace["joint_names"]:
        if jn == "waist_roll_joint":
            weights.append(8.0)
        elif jn == "waist_yaw_joint":
            weights.append(6.0)
        elif jn == "waist_pitch_joint":
            weights.append(2.5)
        elif "shoulder" in jn:
            weights.append(1.4)
        elif "wrist" in jn:
            weights.append(0.8)
        else:
            weights.append(1.0)
    cspace["cspace_distance_weight"] = weights
    return cfg_dict


def _rotmat_to_wxyz(R: np.ndarray) -> list[float]:
    q_xyzw = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def _world_to_pelvis(
    pos_world: np.ndarray,
    rotmat_world: np.ndarray | None,
    pelvis_pos: np.ndarray,
    pelvis_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    p_local = pelvis_R.T @ (np.asarray(pos_world, float) - np.asarray(pelvis_pos, float))
    R_local = None if rotmat_world is None else (pelvis_R.T @ np.asarray(rotmat_world, float))
    return p_local, R_local


class UnitreeCuRoboSolver:
    """Fast GPU multi-seed IK + smooth trajectory solver for Unitree G1 using cuRoboV2-Fast.

    Solvers are initialized lazily on first use and cached per side ('l' / 'r') with
    CUDA graphs enabled so steady-state IK calls execute in ~4-10 ms on RTX 3090 with
    zero PhysX stepping and full upper-body self-collision checking.
    """

    def __init__(
        self,
        *,
        embodiment_name: str = "unitree_g1",
        num_seeds: int = 16,
        use_cuda_graph: bool = True,
    ) -> None:
        self.embodiment_name = embodiment_name
        self.num_seeds = int(num_seeds)
        self.use_cuda_graph = bool(use_cuda_graph)
        self._cfgs: dict[str, dict[str, Any]] = {}
        self._ik_3d: dict[str, Any] = {}
        self._ik_6d: dict[str, Any] = {}
        self._mp: dict[str, Any] = {}
        self.stats: dict[str, list[float]] = {"ik_3d_ms": [], "ik_6d_ms": [], "traj_ms": []}

    def _robot_cfg(self, side: str) -> dict[str, Any]:
        if side not in self._cfgs:
            self._cfgs[side] = _build_arm_robot_cfg(side, self.embodiment_name)
        return copy.deepcopy(self._cfgs[side])

    def _tool_frame(self, side: str) -> str:
        return "right_pinch_frame" if side == "r" else "left_pinch_frame"

    def _make_seed_batch(self, solver, seed_joint_map: dict[str, float] | None, *, track_seed: bool = False):
        """Build fixed-shape (1, num_seeds, D) seed tensor for CUDA graph reuse."""
        import torch

        dev = solver.device_cfg.device
        q_def = solver.default_joint_position.detach().reshape(-1)
        if seed_joint_map:
            q_cur = torch.tensor(
                [float(seed_joint_map.get(jn, float(q_def[i].item()))) for i, jn in enumerate(solver.joint_names)],
                dtype=torch.float32,
                device=dev,
            )
        else:
            q_cur = q_def.clone()

        n_s = self.num_seeds
        seeds = q_cur.view(1, 1, -1).repeat(1, n_s, 1)
        gen = torch.Generator(device=dev)
        gen.manual_seed(42)
        if track_seed:
            if n_s > 1:
                noise = torch.randn((1, n_s - 1, q_cur.numel()), generator=gen, device=dev, dtype=torch.float32) * 0.025
                seeds[:, 1:, :] = q_cur.view(1, 1, -1) + noise
        else:
            if n_s > 1:
                seeds[:, 1, :] = q_def
            if n_s > 2:
                scale = torch.tensor(
                    [0.20 if "waist" in jn else 0.75 for jn in solver.joint_names],
                    dtype=torch.float32,
                    device=dev,
                ).view(1, 1, -1)
                noise = torch.randn((1, n_s - 2, q_cur.numel()), generator=gen, device=dev, dtype=torch.float32) * scale
                base = 0.5 * (q_cur.view(1, 1, -1) + q_def.view(1, 1, -1))
                seeds[:, 2:, :] = base + noise
        bounds = getattr(solver.kinematics, "get_joint_limits", lambda: None)()
        if bounds is not None and hasattr(bounds, "position"):
            lo = bounds.position[0].view(1, 1, -1)
            hi = bounds.position[1].view(1, 1, -1)
            seeds = torch.clamp(seeds, lo + 1e-4, hi - 1e-4)
        return q_cur, seeds

    def _get_ik(self, side: str, *, six_dof: bool, track_mode: bool = False):
        key = f"{side}_{'track' if track_mode else 'global'}"
        cache = self._ik_6d if six_dof else self._ik_3d
        if key in cache:
            return cache[key]
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        n_seeds = 1 if track_mode else self.num_seeds
        opt_cfgs = ["ik/lbfgs_ik.yml"] if track_mode else ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"]
        use_cg = False if track_mode else self.use_cuda_graph
        cfg = InverseKinematicsCfg.create(
            robot=self._robot_cfg(side),
            optimizer_configs=opt_cfgs,
            num_seeds=n_seeds,
            self_collision_check=True,
            position_tolerance=0.020 if six_dof else 0.035,
            orientation_tolerance=0.22 if six_dof else 3.15,
            seed_orientation_weight=1.0 if six_dof else 0.0,
            orientation_rank_weight=None if six_dof else 0.0,
            velocity_regularization_weight=0.0,
            acceleration_regularization_weight=0.0,
            optimization_dt=None,
            use_cuda_graph=use_cg,
            cspace_rank_weight=0.25 if track_mode else 1.0,
        )
        if track_mode:
            cfg.use_lm_seed = False
        solver = InverseKinematics(cfg)
        tool = self._tool_frame(side)
        if not six_dof:
            solver.update_tool_pose_criteria({tool: ToolPoseCriteria.track_position()})
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        seed_t = solver.default_joint_position.clone().view(1, 1, -1).repeat(1, n_seeds, 1)
        cur_js = JointState.from_position(solver.default_joint_position.clone().view(1, -1), joint_names=solver.joint_names)
        fk = solver.compute_kinematics(solver.default_joint_state)
        goal_ok = GoalToolPose.from_poses(
            fk.tool_poses.to_dict(), ordered_tool_frames=solver.tool_frames, num_goalset=1
        )
        goal_far = GoalToolPose.from_poses(
            {tool: Pose.from_list([1.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
            ordered_tool_frames=[tool],
            num_goalset=1,
        )
        solver.solve_pose(goal_ok, current_state=cur_js, seed_config=seed_t)
        solver.solve_pose(goal_far, current_state=cur_js, seed_config=seed_t)
        torch.cuda.synchronize()
        cache[key] = solver
        return solver

    def _get_motion_planner(self, side: str):
        if side in self._mp:
            return self._mp[side]
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        cfg = MotionPlannerCfg.create(robot=self._robot_cfg(side))
        mp = MotionPlanner(cfg)
        mp.warmup()
        self._mp[side] = mp
        return mp

    def warmup(self, sides: tuple[str, ...] = ("r", "l"), *, include_6d: bool = False) -> float:
        t0 = time.monotonic()
        for s in sides:
            self._get_ik(s, six_dof=False, track_mode=False)
            if include_6d:
                self._get_ik(s, six_dof=True, track_mode=False)
                self._get_ik(s, six_dof=True, track_mode=True)
        return time.monotonic() - t0

    def solve_ik_3d(
        self,
        side: str,
        target_world: np.ndarray,
        pelvis_pos: np.ndarray,
        pelvis_R: np.ndarray,
        *,
        tolerance: float = 0.040,
        seed_joint_map: dict[str, float] | None = None,
    ) -> tuple[bool, float, dict[str, float], float]:
        """Solve 3-DoF position-only collision-free IK in pelvis frame on GPU.

        Returns ``(success, pos_err_m, joint_solution_map, solve_ms)``.
        """
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        ik = self._get_ik(side, six_dof=False)
        p_local, _ = _world_to_pelvis(target_world, None, pelvis_pos, pelvis_R)
        tool = self._tool_frame(side)
        goal = GoalToolPose.from_poses(
            {tool: Pose.from_list([float(p_local[0]), float(p_local[1]), float(p_local[2]), 1.0, 0.0, 0.0, 0.0])},
            ordered_tool_frames=[tool],
            num_goalset=1,
        )
        q_cur, seed_t = self._make_seed_batch(ik, seed_joint_map, track_seed=False)
        cur_js = JointState.from_position(q_cur.view(1, -1), joint_names=ik.joint_names)

        t0 = time.monotonic()
        res = ik.solve_pose(goal, current_state=cur_js, seed_config=seed_t)
        torch.cuda.synchronize()
        solve_ms = (time.monotonic() - t0) * 1000.0
        self.stats["ik_3d_ms"].append(solve_ms)

        pos_err = float(res.position_error.reshape(-1)[0].item())
        q_sol = res.js_solution.position.reshape(-1).detach().cpu().numpy()
        sol_map = {jn: float(q_sol[i]) for i, jn in enumerate(ik.joint_names)}
        feasible = bool(res.success.reshape(-1)[0].item())
        if not feasible and pos_err <= float(tolerance):
            sc = getattr(res, "self_collision_cost", None)
            if sc is None or float(sc.reshape(-1)[0].item()) <= 1e-4:
                feasible = True
        return feasible, pos_err, sol_map, solve_ms

    def solve_ik_6d(
        self,
        side: str,
        target_world: np.ndarray,
        rotmat_world: np.ndarray | None,
        pelvis_pos: np.ndarray,
        pelvis_R: np.ndarray,
        *,
        pos_tol: float = 0.025,
        rot_tol: float = 0.25,
        seed_joint_map: dict[str, float] | None = None,
        track_seed: bool = False,
    ) -> tuple[bool, float, float, dict[str, float], float]:
        """Solve 6-DoF collision-free IK in pelvis frame on GPU.

        Returns ``(success, pos_err_m, rot_err_rad, joint_solution_map, solve_ms)``.
        """
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        six_dof = rotmat_world is not None
        ik = self._get_ik(side, six_dof=six_dof, track_mode=bool(track_seed and seed_joint_map))
        p_local, R_local = _world_to_pelvis(target_world, rotmat_world, pelvis_pos, pelvis_R)
        qwxyz = _rotmat_to_wxyz(R_local) if R_local is not None else [1.0, 0.0, 0.0, 0.0]
        tool = self._tool_frame(side)
        goal = GoalToolPose.from_poses(
            {tool: Pose.from_list([float(p_local[0]), float(p_local[1]), float(p_local[2]), *qwxyz])},
            ordered_tool_frames=[tool],
            num_goalset=1,
        )
        if track_seed and seed_joint_map:
            q_s = [float(seed_joint_map.get(jn, 0.0)) for jn in ik.joint_names]
            q_cur = torch.tensor(q_s, dtype=torch.float32, device=ik.device_cfg.device)
            seed_t = q_cur.view(1, 1, -1)
        else:
            q_cur, seed_t = self._make_seed_batch(ik, seed_joint_map, track_seed=False)
        cur_js = JointState.from_position(q_cur.view(1, -1), joint_names=ik.joint_names)

        t0 = time.monotonic()
        res = ik.solve_pose(goal, current_state=cur_js, seed_config=seed_t)
        torch.cuda.synchronize()
        solve_ms = (time.monotonic() - t0) * 1000.0
        self.stats["ik_6d_ms"].append(solve_ms)

        pos_err = float(res.position_error.reshape(-1)[0].item())
        rot_err = float(res.rotation_error.reshape(-1)[0].item()) if six_dof else 0.0
        q_sol = res.js_solution.position.reshape(-1).detach().cpu().numpy()
        sol_map = {jn: float(q_sol[i]) for i, jn in enumerate(ik.joint_names)}
        ok = (pos_err <= float(pos_tol)) and (not six_dof or rot_err <= float(rot_tol))
        return ok, pos_err, rot_err, sol_map, solve_ms

    def plan_pose_trajectory(
        self,
        side: str,
        start_joint_map: dict[str, float],
        target_world: np.ndarray,
        rotmat_world: np.ndarray | None,
        pelvis_pos: np.ndarray,
        pelvis_R: np.ndarray,
    ) -> tuple[bool, list[dict[str, float]], float]:
        """Plan a smooth, collision-free B-spline joint trajectory via cuRoboV2 MotionPlanner."""
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        mp = self._get_motion_planner(side)
        p_local, R_local = _world_to_pelvis(target_world, rotmat_world, pelvis_pos, pelvis_R)
        qwxyz = _rotmat_to_wxyz(R_local) if R_local is not None else [1.0, 0.0, 0.0, 0.0]
        tool = self._tool_frame(side)
        goal = GoalToolPose.from_poses(
            {tool: Pose.from_list([float(p_local[0]), float(p_local[1]), float(p_local[2]), *qwxyz])},
            ordered_tool_frames=[tool],
            num_goalset=1,
        )
        q_start = [float(start_joint_map.get(jn, 0.0)) for jn in mp.joint_names]
        cur = JointState.from_position(
            torch.tensor([q_start], dtype=torch.float32, device=mp.device_cfg.device),
            joint_names=mp.joint_names,
        )
        t0 = time.monotonic()
        res = mp.plan_pose(goal, cur)
        torch.cuda.synchronize()
        plan_ms = (time.monotonic() - t0) * 1000.0
        self.stats["traj_ms"].append(plan_ms)
        if res is None or not bool(res.success.reshape(-1)[0].item()):
            return False, [], plan_ms
        traj_js = res.get_interpolated_plan()
        pos_np = traj_js.position.detach().cpu().numpy()
        if pos_np.ndim == 3:
            pos_np = pos_np[0]
        waypoints = [{jn: float(row[i]) for i, jn in enumerate(mp.joint_names)} for row in pos_np]
        return True, waypoints, plan_ms
