"""CPU-only unit tests for cuRoboV2-Fast backend configuration and frame alignment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rab_curobo as C


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("FINDINGBENCH_RAB_IK_BACKEND", "dls")
    assert C.selected_ik_backend("unitree_g1") == "dls"
    assert C.selected_ik_backend("unitree_g1_dex3") == "dls"
    assert C.selected_ik_backend("agibot_g1") == "dls"

    monkeypatch.setenv("FINDINGBENCH_RAB_IK_BACKEND", "auto")
    assert C.selected_ik_backend("agibot_g1") == "dls"
    expected = "curobo" if C.curobo_available("unitree_g1") else "dls"
    assert C.selected_ik_backend("unitree_g1") == expected
    expected_d3 = "curobo" if C.curobo_available("unitree_g1_dex3") else "dls"
    assert C.selected_ik_backend("unitree_g1_dex3") == expected_d3


def test_build_arm_robot_cfg_dex1_and_dex3():
    if not C.CUROBO_G1_29DOF_YML.is_file():
        return
    for emb_name, exp_parent_r, exp_x in (
        ("unitree_g1", "right_wrist_yaw_link", 0.155),
        ("unitree_g1_dex3", "right_hand_palm_link", 0.085),
    ):
        for side, tool in (("r", "right_pinch_frame"), ("l", "left_pinch_frame")):
            cfg = C._build_arm_robot_cfg(side, emb_name)
            kin = cfg["kinematics"]
            assert kin["base_link"] == "pelvis"
            assert kin["tool_frames"] == [tool]
            jnames = kin["cspace"]["joint_names"]
            assert len(jnames) == 10  # 3 waist + 7 arm
            assert "waist_pitch_joint" in jnames
            for lk in kin["lock_joints"]:
                assert "_hip_" not in lk and "_knee_" not in lk and "_ankle_" not in lk
            ex = kin["extra_links"][tool]
            if side == "r":
                assert ex["parent_link_name"] == exp_parent_r
            ft = ex["fixed_transform"]
            assert np.isclose(ft[0], exp_x, atol=1e-6)
            w, x, y, z = ft[3], ft[4], ft[5], ft[6]
            R = Rotation.from_quat([x, y, z, w]).as_matrix()
            assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-6)  # +Z_pinch = +X_wrist
            assert np.allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-6)  # +X_pinch = +Y_wrist


def test_world_to_pelvis_roundtrip():
    pelvis_pos = np.array([1.2, -0.5, 0.55])
    pelvis_R = Rotation.from_euler("ZY", [0.8, 0.45]).as_matrix()
    p_w = np.array([1.6, -0.1, 0.26])
    R_w = Rotation.from_euler("xyz", [0.2, -0.3, 1.1]).as_matrix()
    p_l, R_l = C._world_to_pelvis(p_w, R_w, pelvis_pos, pelvis_R)
    assert np.allclose(pelvis_pos + pelvis_R @ p_l, p_w, atol=1e-9)
    assert np.allclose(pelvis_R @ R_l, R_w, atol=1e-9)
