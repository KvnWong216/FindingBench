"""Embodiment specifications, session-layer Unitree G1 swap, and static body prior.

Supports switching FindingBench RAB episodes between:
- ``unitree_g1`` (default): Unitree G1 29-DoF bipedal humanoid + Dex3 3-finger hands
  (dynamically swapped into existing RAB scenes in the ephemeral session layer).
- ``agibot_g1``: Legacy AgiBot G1 wheeled dual-arm humanoid native to the RAB scenes.

Environment variables:
- ``FINDINGBENCH_RAB_ROBOT``: ``unitree_g1`` (default) | ``agibot_g1``
- ``FINDINGBENCH_RAB_EMBODIMENT_PRIOR``: ``text`` (default) | ``none``
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITREE_G1_DEX3_USD = ROOT / "assets/robots/unitree_g1/g1_29dof_dex3/g1_29dof_dex3.usd"

# Unitree G1 exact kinematic constants measured from g1_29dof_dex3.usd (m)
G1_PELVIS_STANDING_Z = 0.793        # feet bottom at -0.7923 in pelvis frame
G1_HIP_OFFSET_Z = 0.1027            # pelvis -> hip_pitch_link
G1_THIGH_LEN = 0.3366               # hip_pitch_link -> knee_link
G1_SHIN_LEN = 0.3176                # knee_link -> ankle_roll_link
G1_ANKLE_TO_SOLE = 0.0361           # ankle_roll_link -> floor contact
G1_D435_LOCAL = (0.0537, 0.0, 0.4739)  # centered head camera relative to pelvis
G1_MAX_CROUCH_DZ = 0.34             # max pelvis drop while keeping feet on floor

UNITREE_LEFT_ARM = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
UNITREE_RIGHT_ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
UNITREE_TORSO_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
UNITREE_LEFT_HAND = (
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
)
UNITREE_RIGHT_HAND = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

# Dex3 open vs closed joint presets (rad), verified against USD joint limits:
# Left index/middle close with negative Z angle; right index/middle close with positive Z angle.
DEX3_OPEN_LEFT = {
    "left_hand_index_0_joint": -0.05,
    "left_hand_index_1_joint": -0.05,
    "left_hand_middle_0_joint": -0.05,
    "left_hand_middle_1_joint": -0.05,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": -0.20,
    "left_hand_thumb_2_joint": 0.10,
}
DEX3_CLOSED_LEFT = {
    "left_hand_index_0_joint": -1.05,
    "left_hand_index_1_joint": -1.10,
    "left_hand_middle_0_joint": -1.05,
    "left_hand_middle_1_joint": -1.10,
    "left_hand_thumb_0_joint": 0.15,
    "left_hand_thumb_1_joint": 0.55,
    "left_hand_thumb_2_joint": 0.85,
}
DEX3_OPEN_RIGHT = {
    "right_hand_index_0_joint": 0.05,
    "right_hand_index_1_joint": 0.05,
    "right_hand_middle_0_joint": 0.05,
    "right_hand_middle_1_joint": 0.05,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.20,
    "right_hand_thumb_2_joint": -0.10,
}
DEX3_CLOSED_RIGHT = {
    "right_hand_index_0_joint": 1.05,
    "right_hand_index_1_joint": 1.10,
    "right_hand_middle_0_joint": 1.05,
    "right_hand_middle_1_joint": 1.10,
    "right_hand_thumb_0_joint": -0.15,
    "right_hand_thumb_1_joint": -0.55,
    "right_hand_thumb_2_joint": -0.85,
}

# Standing ready pose for Unitree G1 (rad): upright legs/waist, relaxed arms bent forward.
UNITREE_READY_POSE: dict[str, float] = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.25,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.15,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.25,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    **DEX3_OPEN_LEFT,
    **DEX3_OPEN_RIGHT,
}


@dataclass(frozen=True)
class EmbodimentConfig:
    name: str
    robot_prim: str
    root_link: str
    articulation_prim: str
    pelvis_z_offset: float
    camera_local_offset: tuple[float, float, float] | None  # (fwd, left, up_from_floor); None => use USD HEAD
    head_yaw_joint: str | None
    left_arm: tuple[str, ...]
    right_arm: tuple[str, ...]
    torso_joints: tuple[str, ...]
    ee_link_left: str
    ee_link_right: str
    ee_offset_left: tuple[float, float, float]
    ee_offset_right: tuple[float, float, float]
    reach_radius_m: float
    articulation_reach_m: float
    max_grasp_extent_m: float
    turn_clearance_r: float
    free_stand_r: float
    clear_base_lateral: tuple[float, ...]
    clear_base_z_bands: tuple[float, ...]
    turn_z_bands: tuple[float, ...]
    supports_crouch: bool
    public_spec: dict[str, Any]

    def arm_joints(self, side: str) -> list[str]:
        base = list(self.right_arm if side == "r" else self.left_arm)
        for j in self.torso_joints:
            if j not in base:
                base.append(j)
        return base

    def ee_link(self, side: str) -> str:
        return self.ee_link_right if side == "r" else self.ee_link_left

    def ee_link_path(self, side: str) -> str:
        return f"{self.robot_prim}/{self.ee_link(side)}"

    def ee_offset(self, side: str) -> np.ndarray:
        return np.asarray(self.ee_offset_right if side == "r" else self.ee_offset_left, float)


UNITREE_G1_CONFIG = EmbodimentConfig(
    name="unitree_g1",
    robot_prim="/World/UnitreeG1",
    root_link="/World/UnitreeG1/pelvis",
    articulation_prim="/World/UnitreeG1/pelvis",
    pelvis_z_offset=G1_PELVIS_STANDING_Z,
    camera_local_offset=(G1_D435_LOCAL[0], 0.0, G1_PELVIS_STANDING_Z + G1_D435_LOCAL[2]),
    head_yaw_joint=None,
    left_arm=UNITREE_LEFT_ARM,
    right_arm=UNITREE_RIGHT_ARM,
    torso_joints=UNITREE_TORSO_JOINTS,
    ee_link_left="left_hand_palm_link",
    ee_link_right="right_hand_palm_link",
    # Fingertip pinch center in palm local frame (+X along fingers, inward toward thumb)
    ee_offset_left=(0.085, -0.025, 0.0),
    ee_offset_right=(0.085, 0.025, 0.0),
    reach_radius_m=0.80,
    articulation_reach_m=0.85,
    max_grasp_extent_m=0.15,
    turn_clearance_r=0.34,
    free_stand_r=0.35,
    clear_base_lateral=(-0.34, -0.18, 0.0, 0.18, 0.34),
    clear_base_z_bands=(0.12, 0.27, 0.42, 0.65, 0.95, 1.28),
    turn_z_bands=(0.15, 0.27, 0.40, 0.70, 1.05, 1.25),
    supports_crouch=True,
    public_spec={
        "embodiment_id": "unitree_g1_dex3",
        "display_name": "Unitree G1 (29-DoF bipedal humanoid with Dex3 3-finger hands)",
        "height_m": 1.32,
        "eye_camera_height_m": 1.27,
        "body_footprint_radius_m": 0.34,
        "nominal_arm_reach_m": 0.78,
        "hand_grasp_span_m": 0.14,
        "torso_and_posture": (
            "Bipedal humanoid with 3-DoF waist (yaw/roll/pitch) and automatic crouching "
            "for low workspaces; standing vertical reach up to ~1.55 m."
        ),
        "hand_configuration": (
            "Dual 7-DoF arms with 3-finger Dex3 dexterous hands (single held item at a time)."
        ),
        "camera_mount": (
            "Head-mounted forward RGB pinhole camera at ~1.27 m eye height; "
            "virtual gaze pans, tilts, and zooms from the head position."
        ),
    },
)

AGIBOT_G1_CONFIG = EmbodimentConfig(
    name="agibot_g1",
    robot_prim="/World/Robot",
    root_link="/World/Robot/base_link",
    articulation_prim="/World/Robot",
    pelvis_z_offset=0.0,
    camera_local_offset=None,
    head_yaw_joint="idx11_head_joint1",
    left_arm=(
        "idx21_arm_l_joint1",
        "idx22_arm_l_joint2",
        "idx23_arm_l_joint3",
        "idx24_arm_l_joint4",
        "idx25_arm_l_joint5",
        "idx26_arm_l_joint6",
        "idx27_arm_l_joint7",
    ),
    right_arm=(
        "idx61_arm_r_joint1",
        "idx62_arm_r_joint2",
        "idx63_arm_r_joint3",
        "idx64_arm_r_joint4",
        "idx65_arm_r_joint5",
        "idx66_arm_r_joint6",
        "idx67_arm_r_joint7",
    ),
    torso_joints=("idx01_body_joint1", "idx02_body_joint2"),
    ee_link_left="gripper_l_center_link",
    ee_link_right="gripper_r_center_link",
    ee_offset_left=(0.0, 0.0, 0.0),
    ee_offset_right=(0.0, 0.0, 0.0),
    reach_radius_m=0.92,
    articulation_reach_m=0.94,
    max_grasp_extent_m=0.16,
    turn_clearance_r=0.38,
    free_stand_r=0.39,
    clear_base_lateral=(-0.40, -0.22, 0.0, 0.22, 0.40),
    clear_base_z_bands=(0.13, 0.30, 0.53, 0.78, 1.05, 1.35),
    turn_z_bands=(0.17, 0.40, 0.70, 1.10),
    supports_crouch=False,
    public_spec={
        "embodiment_id": "agibot_g1_omnipicker",
        "display_name": "AgiBot G1 (wheeled dual-arm humanoid with OmniPicker parallel grippers)",
        "height_m": 1.55,
        "eye_camera_height_m": 1.40,
        "body_footprint_radius_m": 0.38,
        "nominal_arm_reach_m": 0.90,
        "hand_grasp_span_m": 0.16,
        "torso_and_posture": (
            "Wheeled mobile base with vertical torso lift and torso pitch for low and high workspaces."
        ),
        "hand_configuration": (
            "Dual 7-DoF arms with parallel-jaw grippers (single held item at a time)."
        ),
        "camera_mount": (
            "Head-mounted forward RGB pinhole camera at ~1.40 m eye height; "
            "virtual gaze pans, tilts, and zooms from the head position."
        ),
    },
)

EMBODIMENTS: dict[str, EmbodimentConfig] = {
    "unitree_g1": UNITREE_G1_CONFIG,
    "agibot_g1": AGIBOT_G1_CONFIG,
}


def selected_robot_name() -> str:
    raw = os.environ.get("FINDINGBENCH_RAB_ROBOT", "unitree_g1").strip().lower()
    if raw in ("unitree_g1", "unitree", "g1_29dof", "g1_dex3"):
        return "unitree_g1"
    if raw in ("agibot_g1", "agibot", "legacy"):
        return "agibot_g1"
    raise ValueError(f"Unknown FINDINGBENCH_RAB_ROBOT={raw!r}; expected 'unitree_g1' or 'agibot_g1'")


def get_embodiment(name: str | None = None) -> EmbodimentConfig:
    key = selected_robot_name() if name is None else name.strip().lower()
    if key not in EMBODIMENTS:
        raise ValueError(f"Unknown embodiment {key!r}; expected one of {sorted(EMBODIMENTS)}")
    return EMBODIMENTS[key]


def embodiment_prior_mode() -> str:
    raw = os.environ.get("FINDINGBENCH_RAB_EMBODIMENT_PRIOR", "text").strip().lower()
    if raw in ("none", "off", "false", "0"):
        return "none"
    if raw in ("text", "on", "true", "1", "spec"):
        return "text"
    raise ValueError(
        f"Unknown FINDINGBENCH_RAB_EMBODIMENT_PRIOR={raw!r}; expected 'text' or 'none'"
    )


def public_embodiment_spec(cfg: EmbodimentConfig | None = None) -> dict[str, Any] | None:
    """Return static, zero-leakage body schema dict for public observations, or None."""
    if embodiment_prior_mode() == "none":
        return None
    emb = cfg or get_embodiment()
    return dict(emb.public_spec)


def compute_g1_crouch(
    target_z: float,
    floor_z: float = 0.0,
    max_knee_x: float | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Compute pelvis drop ``crouch_dz`` (m), hip-hinge ``pelvis_pitch`` (rad), and
    symmetrical leg + waist pose (rad) so Unitree G1 reaches low targets cleanly.

    Uses a natural splayed squat (``hip_yaw = ±1.0 rad``) plus forward hip-hinge
    (``pelvis_pitch`` compensated by ``-pelvis_pitch`` on ``hip_pitch_joint`` so legs
    and feet remain planted on the floor with minimal forward knee protrusion).
    Returns ``(crouch_dz, pelvis_pitch, joint_targets_rad)``.
    """
    h = float(target_z) - float(floor_z)
    if h >= 0.85:
        return 0.0, 0.0, {}
    frac = float(np.clip((0.85 - h) / 0.40, 0.0, 1.0))
    crouch_dz = float(np.clip(frac * 0.28, 0.0, G1_MAX_CROUCH_DZ))
    pelvis_pitch = float(np.clip(frac * 0.62, 0.0, 0.65))  # up to ~37 deg forward hip-hinge
    waist_pitch = float(np.clip(frac * 0.50, 0.0, 0.50))

    L1, L2 = G1_THIGH_LEN, G1_SHIN_LEN
    splay_yaw = 1.00  # ~57 deg outward knee splay => cos(1.0)=0.54 halves knee protrusion
    if max_knee_x is not None:
        lim_sin = float(np.clip(max(0.0, max_knee_x) / (L1 * math.cos(splay_yaw)), 0.0, 0.92))
        max_hp = math.asin(lim_sin)
        min_d = L1 * math.cos(max_hp) + math.sqrt(max(1e-6, L2 * L2 - (L1 * math.sin(max_hp)) ** 2))
        max_dz = max(0.0, (L1 + L2) - min_d)
        crouch_dz = min(crouch_dz, max_dz)

    if crouch_dz < 0.01:
        # Pure hip-hinge bow with straight legs on floor
        hip_pitch = -pelvis_pitch
        joints = {
            "left_hip_pitch_joint": hip_pitch,
            "right_hip_pitch_joint": hip_pitch,
            "waist_pitch_joint": waist_pitch,
        }
        return 0.0, pelvis_pitch, joints

    # 2-link sagittal leg IK keeping ankle under hip and sole flat on floor
    d = float(np.clip((L1 + L2) - crouch_dz, 0.26, L1 + L2 - 1e-4))
    cos_knee = float(np.clip((L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2), -1.0, 1.0))
    knee = math.pi - math.acos(cos_knee)
    cos_hip = float(np.clip((L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d), -1.0, 1.0))
    hip_pitch_upright = -math.acos(cos_hip)
    ankle_pitch = -(knee + hip_pitch_upright)

    # Compensate pelvis forward pitch on hip_pitch so thighs/shins/feet stay fixed in world
    hip_pitch = float(np.clip(hip_pitch_upright - pelvis_pitch, math.radians(-138.0), 0.0))
    knee = float(np.clip(knee, 0.0, math.radians(155.0)))
    ankle_pitch = float(np.clip(ankle_pitch, math.radians(-48.0), math.radians(28.0)))

    joints = {
        "left_hip_pitch_joint": hip_pitch,
        "left_hip_yaw_joint": splay_yaw,
        "left_knee_joint": knee,
        "left_ankle_pitch_joint": ankle_pitch,
        "right_hip_pitch_joint": hip_pitch,
        "right_hip_yaw_joint": -splay_yaw,
        "right_knee_joint": knee,
        "right_ankle_pitch_joint": ankle_pitch,
        "waist_pitch_joint": waist_pitch,
    }
    return crouch_dz, pelvis_pitch, joints


def dex3_hand_targets(side: str, openness: float) -> dict[str, float]:
    """Interpolate Dex3 7-DoF hand targets for ``openness`` in [0.0 (closed), 1.0 (open)]."""
    f = float(np.clip(openness, 0.0, 1.0))
    op = DEX3_OPEN_RIGHT if side == "r" else DEX3_OPEN_LEFT
    cl = DEX3_CLOSED_RIGHT if side == "r" else DEX3_CLOSED_LEFT
    return {k: cl[k] + (op[k] - cl[k]) * f for k in op}


def swap_stage_to_unitree_g1(stage) -> tuple[np.ndarray, float]:
    """Swap `/World/Robot` (AgiBot) to `/World/UnitreeG1` in the ephemeral session layer.

    Reads the original spawn pose from `/World/Robot/base_link`, deactivates `/World/Robot`,
    references `g1_29dof_dex3.usd` at `/World/UnitreeG1` at `(x, y, floor_z + G1_PELVIS_STANDING_Z)`
    with the exact same spawn yaw, authors strong PD joint drives, and authors the fixed
    initialization anchor at `/World/UnitreeG1/pelvis/FixedJoint`.

    Returns ``(spawn_pelvis_xyz, spawn_yaw_rad)``.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError("Unitree G1 swap must be authored in the episode session layer")
    if not UNITREE_G1_DEX3_USD.is_file():
        raise FileNotFoundError(f"Unitree G1 Dex3 USD not found: {UNITREE_G1_DEX3_USD}")

    old_robot = stage.GetPrimAtPath("/World/Robot")
    old_base = stage.GetPrimAtPath("/World/Robot/base_link")
    if not old_robot or not old_base:
        raise ValueError("Expected /World/Robot/base_link in RAB scene before Unitree G1 swap")

    cache = UsdGeom.XformCache()
    base_xf = cache.GetLocalToWorldTransform(old_base)
    base_pos = base_xf.ExtractTranslation()
    fwd = base_xf.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    spawn_yaw = math.atan2(float(fwd[1]), float(fwd[0]))
    pelvis_pos = Gf.Vec3d(float(base_pos[0]), float(base_pos[1]), float(base_pos[2]) + G1_PELVIS_STANDING_Z)
    spawn_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(spawn_yaw)).GetQuat()

    # Deactivate original AgiBot G1 completely in session layer
    old_robot.SetActive(False)

    # Reference Unitree G1 Dex3 USD at /World/UnitreeG1
    g1_prim = stage.DefinePrim(UNITREE_G1_CONFIG.robot_prim, "Xform")
    g1_prim.GetReferences().AddReference(str(UNITREE_G1_DEX3_USD))
    xf = UsdGeom.Xformable(g1_prim)
    mat = Gf.Matrix4d().SetRotate(spawn_rot)
    mat.SetTranslateOnly(pelvis_pos)
    xf.MakeMatrixXform().Set(mat)

    # Strengthen PD drive gains on all Unitree G1 revolute joints so PhysX holds poses stiffly
    from pxr import Usd
    for p in Usd.PrimRange(g1_prim):
        if not p.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = p.GetName()
        drv = UsdPhysics.DriveAPI.Apply(p, "angular")
        drv.CreateTypeAttr("force")
        if "_hand_" in name:
            kp, kd, mf = 40.0, 4.0, 25.0
        elif "shoulder" in name or "elbow" in name or "wrist" in name:
            kp, kd, mf = 220.0, 22.0, 120.0
        else:
            # Legs and waist
            kp, kd, mf = 450.0, 45.0, 250.0
        drv.CreateStiffnessAttr(kp)
        drv.CreateDampingAttr(kd)
        drv.CreateMaxForceAttr(mf)
        drv.CreateTargetPositionAttr(math.degrees(UNITREE_READY_POSE.get(name, 0.0)))

    # Author fixed anchor on /World/UnitreeG1/pelvis
    anchor_path = f"{UNITREE_G1_CONFIG.root_link}/FixedJoint"
    anchor = UsdPhysics.FixedJoint.Define(stage, anchor_path)
    anchor.CreateBody1Rel().SetTargets([UNITREE_G1_CONFIG.root_link])
    anchor.CreateExcludeFromArticulationAttr(True)
    anchor.CreateJointEnabledAttr(True)
    anchor.CreateLocalPos0Attr(Gf.Vec3f(pelvis_pos))
    anchor.CreateLocalRot0Attr(Gf.Quatf(spawn_rot))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1.0))

    return np.array([float(pelvis_pos[0]), float(pelvis_pos[1]), float(pelvis_pos[2])], float), float(spawn_yaw)


class UnitreeMobileController:
    """Drop-in mobile controller for Unitree G1 matching `arena_api.mobile_base_native.Controller`."""

    ZERO = {"base_linear_x": 0.0, "base_angular_z": 0.0}

    def __init__(self, stage, cfg: EmbodimentConfig = UNITREE_G1_CONFIG):
        from pxr import Gf, UsdGeom, UsdPhysics

        self.stage, self.Gf, self.cfg = stage, Gf, cfg
        self.previous = dict(self.ZERO)
        self.mode = "anchor"
        self.view = self.simulation_view = None
        self.flush = None
        self.joints: dict[str, Any] = {}
        self.drives: dict[str, Any] = {}

        base = stage.GetPrimAtPath(cfg.root_link)
        self.anchor = UsdPhysics.FixedJoint(stage.GetPrimAtPath(cfg.root_link + "/FixedJoint"))
        if not base or not self.anchor:
            raise ValueError(f"UnitreeMobileController requires {cfg.root_link}/FixedJoint")

    def bind(self):
        import omni.physics.tensors as tensors
        import omni.physx

        self.simulation_view = tensors.create_simulation_view("numpy")
        self.simulation_view.set_subspace_roots("/")
        self.view = self.simulation_view.create_articulation_view(self.cfg.articulation_prim)
        if self.view.count != 1:
            raise RuntimeError(
                f"Expected 1 Unitree G1 articulation at {self.cfg.articulation_prim}, got {self.view.count}"
            )
        self.flush = omni.physx.get_physx_simulation_interface().flush_changes
        p = self.pose()
        if not np.all(np.isfinite(p)):
            raise RuntimeError(f"Non-finite initial Unitree G1 root pose: {p}")

    def pose(self) -> np.ndarray:
        return self.view.get_root_transforms()[0].copy()

    def apply(self, velocity: dict[str, float]) -> dict[str, Any]:
        self.previous = dict(velocity)
        return dict(command=self.previous, constraint_mode=self.mode)

    def verify(self) -> None:
        if bool(self.anchor.GetJointEnabledAttr().Get()) != (self.mode == "anchor"):
            raise RuntimeError("Unexpected Unitree G1 anchor state")


def apply_unitree_ready_pose(world) -> None:
    """Set Unitree G1 to its upright standing ready pose inside PhysX."""
    robot = world.mobile.view
    names = list(robot.shared_metatype.dof_names)
    target = robot.get_dof_positions().copy()
    limits = robot.get_dof_limits()[0]
    for name, val in UNITREE_READY_POSE.items():
        if name not in names:
            continue
        j = names.index(name)
        target[0, j] = float(np.clip(val, limits[j, 0], limits[j, 1]))
    idx = np.array([0], np.uint32)
    robot.set_dof_positions(target, idx)
    robot.set_dof_velocities(np.zeros_like(target), idx)
    robot.set_dof_position_targets(target, idx)
    world.step(30)
