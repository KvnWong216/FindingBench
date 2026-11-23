"""CPU-only unit tests for Unitree G1 (Dex1-1 2-finger & Dex3) / AgiBot G1 embodiment abstraction."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rab_embodiment import (
    AGIBOT_G1_CONFIG,
    DEX1_JAW_MAX_M,
    DEX1_JAW_MIN_M,
    DEX1_ZERO_GAP_M,
    G1_PELVIS_STANDING_Z,
    UNITREE_G1_CONFIG,
    UNITREE_G1_DEX1_USD,
    UNITREE_G1_DEX3_CONFIG,
    UNITREE_G1_DEX3_USD,
    compute_g1_crouch,
    dex1_hand_targets,
    dex1_jaw_target,
    dex3_hand_targets,
    embodiment_prior_mode,
    get_embodiment,
    public_embodiment_spec,
    selected_robot_name,
    unitree_walk_gait_joints,
)
from rummagebench.adapters.rab_mcp_server import PUBLIC_OBSERVATION_KEYS, _strip_public_observation

ROOT = Path(__file__).resolve().parents[1]


def test_default_embodiment_is_unitree_g1_dex1(monkeypatch):
    monkeypatch.delenv("FINDINGBENCH_RAB_ROBOT", raising=False)
    assert selected_robot_name() == "unitree_g1"
    cfg = get_embodiment()
    assert cfg is UNITREE_G1_CONFIG
    assert cfg.gripper_type == "dex1"
    assert cfg.robot_prim == "/World/UnitreeG1"
    assert cfg.articulation_prim == "/World/UnitreeG1/Geometry/pelvis"
    assert cfg.ee_link("r") == "right_wrist_yaw_link"
    assert cfg.ee_link_path("r") == "/World/UnitreeG1/Geometry/right_wrist_yaw_link"
    assert math.isclose(cfg.pelvis_z_offset, G1_PELVIS_STANDING_Z, abs_tol=1e-6)
    assert math.isclose(cfg.free_stand_r, 0.22, abs_tol=1e-6)
    assert math.isclose(cfg.turn_clearance_r, 0.22, abs_tol=1e-6)
    assert math.isclose(max(cfg.clear_base_lateral), 0.22, abs_tol=1e-6)
    assert math.isclose(cfg.public_spec["body_footprint_radius_m"], 0.22, abs_tol=1e-6)
    assert UNITREE_G1_DEX1_USD.is_file()


def test_dex3_and_agibot_switches(monkeypatch):
    monkeypatch.setenv("FINDINGBENCH_RAB_ROBOT", "unitree_g1_dex3")
    assert selected_robot_name() == "unitree_g1_dex3"
    cfg_d3 = get_embodiment()
    assert cfg_d3 is UNITREE_G1_DEX3_CONFIG
    assert cfg_d3.gripper_type == "dex3"
    assert cfg_d3.articulation_prim == "/World/UnitreeG1/pelvis"
    assert UNITREE_G1_DEX3_USD.is_file()

    monkeypatch.setenv("FINDINGBENCH_RAB_ROBOT", "agibot_g1")
    assert selected_robot_name() == "agibot_g1"
    cfg = get_embodiment()
    assert cfg is AGIBOT_G1_CONFIG
    assert cfg.robot_prim == "/World/Robot"
    assert cfg.pelvis_z_offset == 0.0


def test_dex1_parallel_jaw_calibration():
    assert math.isclose(dex1_jaw_target(DEX1_ZERO_GAP_M), 0.0, abs_tol=1e-9)
    assert math.isclose(dex1_jaw_target(0.15), DEX1_JAW_MAX_M, abs_tol=1e-9)
    assert math.isclose(dex1_jaw_target(0.001), DEX1_JAW_MIN_M, abs_tol=1e-9)
    # Symmetric displacement on both finger_joint_1 and finger_joint_2
    r_w = dex1_hand_targets("r", width_m=0.0659)
    assert math.isclose(r_w["right_dex1_finger_joint_1"], 0.010, abs_tol=1e-6)
    assert math.isclose(r_w["right_dex1_finger_joint_2"], 0.010, abs_tol=1e-6)
    l_open = dex1_hand_targets("l", 1.0)
    l_closed = dex1_hand_targets("l", 0.0)
    assert math.isclose(l_open["left_dex1_finger_joint_1"], DEX1_JAW_MAX_M, abs_tol=1e-9)
    assert math.isclose(l_closed["left_dex1_finger_joint_1"], DEX1_JAW_MIN_M, abs_tol=1e-9)


def test_embodiment_prior_zero_leakage_and_switch(monkeypatch):
    monkeypatch.delenv("FINDINGBENCH_RAB_EMBODIMENT_PRIOR", raising=False)
    assert embodiment_prior_mode() == "text"
    spec = public_embodiment_spec(UNITREE_G1_CONFIG)
    assert isinstance(spec, dict)
    assert spec["embodiment_id"] == "unitree_g1_dex1"
    assert "2-finger" in spec["display_name"]
    assert 1.2 < spec["height_m"] < 1.4
    assert 1.2 < spec["eye_camera_height_m"] < 1.35
    flat = " ".join(str(v).lower() for v in spec.values())
    for forbidden in ("move_to", "grasp(", "drawer", "coffee", "target", "episode", "threshold", "try ", "you should"):
        assert forbidden not in flat

    monkeypatch.setenv("FINDINGBENCH_RAB_EMBODIMENT_PRIOR", "none")
    assert embodiment_prior_mode() == "none"
    assert public_embodiment_spec(UNITREE_G1_CONFIG) is None


def test_strip_public_observation_respects_embodiment_spec():
    import base64

    fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    assert "embodiment_spec" in PUBLIC_OBSERVATION_KEYS

    obs_with = {
        "instruction": "Find the cup.",
        "frame_id": "f1",
        "planning_step": 0,
        "max_planning_steps": 120,
        "skill_library": {},
        "embodiment_spec": UNITREE_G1_CONFIG.public_spec,
        "feedback": None,
        "image_png_b64": fake_png,
    }
    meta, _, _, _ = _strip_public_observation(obs_with)
    assert meta["embodiment_spec"]["embodiment_id"] == "unitree_g1_dex1"

    obs_without = dict(obs_with)
    obs_without.pop("embodiment_spec")
    meta2, _, _, _ = _strip_public_observation(obs_without)
    assert "embodiment_spec" not in meta2


def test_compute_g1_crouch_kinematics_and_walk_gait():
    dz_hi, pitch_hi, j_hi = compute_g1_crouch(0.85, floor_z=0.0)
    assert dz_hi == 0.0 and pitch_hi == 0.0 and j_hi == {}

    dz_low, pitch_low, j_low = compute_g1_crouch(0.26, floor_z=0.0)
    assert 0.18 <= dz_low <= 0.34
    assert 0.35 <= pitch_low <= 0.62
    assert j_low["left_knee_joint"] > 0.8
    assert j_low["left_hip_pitch_joint"] < -0.9
    assert j_low["waist_pitch_joint"] > 0.2
    assert math.isclose(j_low["left_knee_joint"], j_low["right_knee_joint"])
    assert math.isclose(j_low["left_hip_pitch_joint"], j_low["right_hip_pitch_joint"])

    dz_tight, pitch_tight, j_tight = compute_g1_crouch(0.26, floor_z=0.0, max_knee_x=0.0)
    assert dz_tight == 0.0 and pitch_tight > 0.35 and j_tight["left_hip_pitch_joint"] < -0.35

    g0 = unitree_walk_gait_joints(1.2, amp=0.0)
    assert math.isclose(g0["left_knee_joint"], 0.0, abs_tol=1e-9)
    gc = unitree_walk_gait_joints(1.2, amp=1.0, carry_side="r")
    assert "right_shoulder_pitch_joint" not in gc
    assert "left_shoulder_pitch_joint" in gc


def test_dex3_hand_targets_mirror_signs():
    l_cl = dex3_hand_targets("l", 0.0)
    r_cl = dex3_hand_targets("r", 0.0)
    assert l_cl["left_hand_index_0_joint"] < -0.5
    assert r_cl["right_hand_index_0_joint"] > 0.5


def test_eval_runner_forwards_embodiment_spec_and_shards():
    from run_rab_mcp_eval import build_pinhole_prompt, resolve_episode_list

    obs = {
        "instruction": "Move the cup to the table.",
        "frame_id": "f42",
        "planning_step": 0,
        "max_planning_steps": 120,
        "skill_library": {},
        "embodiment_spec": UNITREE_G1_CONFIG.public_spec,
        "feedback": None,
        "inspection_view": None,
    }
    prompt = build_pinhole_prompt(obs, [], 60, "default_natural")
    assert '"embodiment_id": "unitree_g1_dex1"' in prompt

    rel = ROOT / "runs/rab_lived_in_20261120"
    if (rel / "combined_manifest.json").is_file():
        all_eps = resolve_episode_list({"episodes": "all"}, rel)
        assert len(all_eps) == 1000
        merged = []
        for s in ("0/4", "1/4", "2/4", "3/4"):
            part = resolve_episode_list({"episodes": "all"}, rel, shard=s)
            assert len(part) == 250
            merged.extend(part)
        assert sorted(merged) == sorted(all_eps)
