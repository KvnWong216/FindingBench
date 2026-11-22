"""CPU-only unit tests for Unitree G1 / AgiBot G1 embodiment abstraction and static body prior."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rab_embodiment import (
    AGIBOT_G1_CONFIG,
    G1_PELVIS_STANDING_Z,
    UNITREE_G1_CONFIG,
    UNITREE_G1_DEX3_USD,
    compute_g1_crouch,
    dex3_hand_targets,
    embodiment_prior_mode,
    get_embodiment,
    public_embodiment_spec,
    selected_robot_name,
)
from rummagebench.adapters.rab_mcp_server import PUBLIC_OBSERVATION_KEYS, _strip_public_observation


def test_default_embodiment_is_unitree_g1(monkeypatch):
    monkeypatch.delenv("FINDINGBENCH_RAB_ROBOT", raising=False)
    assert selected_robot_name() == "unitree_g1"
    cfg = get_embodiment()
    assert cfg is UNITREE_G1_CONFIG
    assert cfg.robot_prim == "/World/UnitreeG1"
    assert cfg.articulation_prim == "/World/UnitreeG1/pelvis"
    assert math.isclose(cfg.pelvis_z_offset, G1_PELVIS_STANDING_Z, abs_tol=1e-6)
    assert UNITREE_G1_DEX3_USD.is_file()


def test_agibot_switch(monkeypatch):
    monkeypatch.setenv("FINDINGBENCH_RAB_ROBOT", "agibot_g1")
    assert selected_robot_name() == "agibot_g1"
    cfg = get_embodiment()
    assert cfg is AGIBOT_G1_CONFIG
    assert cfg.robot_prim == "/World/Robot"
    assert cfg.pelvis_z_offset == 0.0


def test_embodiment_prior_zero_leakage_and_switch(monkeypatch):
    monkeypatch.delenv("FINDINGBENCH_RAB_EMBODIMENT_PRIOR", raising=False)
    assert embodiment_prior_mode() == "text"
    spec = public_embodiment_spec(UNITREE_G1_CONFIG)
    assert isinstance(spec, dict)
    assert spec["embodiment_id"] == "unitree_g1_dex3"
    assert 1.2 < spec["height_m"] < 1.4
    assert 1.2 < spec["eye_camera_height_m"] < 1.35
    # Verify zero task leakage and zero imperative hints
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
    assert meta["embodiment_spec"]["embodiment_id"] == "unitree_g1_dex3"

    obs_without = dict(obs_with)
    obs_without.pop("embodiment_spec")
    meta2, _, _, _ = _strip_public_observation(obs_without)
    assert "embodiment_spec" not in meta2


def test_compute_g1_crouch_kinematics():
    dz_hi, pitch_hi, j_hi = compute_g1_crouch(0.85, floor_z=0.0)
    assert dz_hi == 0.0 and pitch_hi == 0.0 and j_hi == {}

    dz_low, pitch_low, j_low = compute_g1_crouch(0.26, floor_z=0.0)
    assert 0.18 <= dz_low <= 0.34
    assert 0.35 <= pitch_low <= 0.62
    assert j_low["left_knee_joint"] > 0.8
    assert j_low["left_hip_pitch_joint"] < -0.9
    assert j_low["waist_pitch_joint"] > 0.2
    # Symmetrical legs
    assert math.isclose(j_low["left_knee_joint"], j_low["right_knee_joint"])
    assert math.isclose(j_low["left_hip_pitch_joint"], j_low["right_hip_pitch_joint"])

    # Tight knee clearance falls back cleanly to pure hip-hinge bow
    dz_tight, pitch_tight, j_tight = compute_g1_crouch(0.26, floor_z=0.0, max_knee_x=0.0)
    assert dz_tight == 0.0 and pitch_tight > 0.35 and j_tight["left_hip_pitch_joint"] < -0.35


def test_dex3_hand_targets_mirror_signs():
    l_cl = dex3_hand_targets("l", 0.0)
    r_cl = dex3_hand_targets("r", 0.0)
    assert l_cl["left_hand_index_0_joint"] < -0.5
    assert r_cl["right_hand_index_0_joint"] > 0.5
