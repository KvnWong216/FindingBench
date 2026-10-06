"""CPU parts of the embodiment probe."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from rummagebench.authoring.tasks.embodiment_slots import (
    Pose2, SlotEmbodimentOverlay, SlotInteraction, StartPose)
from rummagebench.authoring.tasks.probe import (
    ProbeConfig, anchor_candidates, farthest_points, pose_from_xyyaw, wrap_yaw)
from rummagebench.authoring.tasks.slots import Furniture, SceneSlots

REPO = Path(__file__).resolve().parents[2]
GENERATED = REPO / "build" / "tasks"
BEHAVIOR = Path("/data1/ygwang/codes/BEHAVIOR-1K/datasets/behavior-1k-assets")


def _furn(yaw: float, pos=(1.0, 2.0, 0.4), size=(0.6, 1.0, 0.8)) -> Furniture:
    return Furniture(entity="cab_0", category="bottom_cabinet", model="m", room="kitchen_0",
                     room_type="kitchen", position=pos, yaw=yaw, openable=True, size=size)


def test_anchor_candidates_stand_in_front_and_face_the_furniture():
    cfg = ProbeConfig(standoffs_m=(0.5,), lateral_fracs=(0.0,))
    for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
        f = _furn(yaw)
        (x, y, ayaw), = anchor_candidates(f, cfg)
        # base centre = front face (depth/2) + standoff along local +x
        assert math.hypot(x - 1.0, y - 2.0) == pytest.approx(0.3 + 0.5)
        assert math.cos(ayaw - wrap_yaw(yaw + math.pi)) == pytest.approx(1.0)
        # heading points from the anchor at the furniture
        to_f = math.atan2(2.0 - y, 1.0 - x)
        assert math.cos(to_f - ayaw) == pytest.approx(1.0)


@pytest.mark.skipif(not (GENERATED / "scenes").exists(), reason="scene slots not built")
def test_anchor_candidates_reproduce_gpu_picked_anchor():
    """knife_search_001 anchor of bottom_cabinet_no_top_qohxjq_0 (picked on
    the simulator) lies on the probe's standoff ray with the same heading."""
    ss = SceneSlots.load(GENERATED / "scenes" / "Beechwood_0_int.yaml")
    f = ss.furniture_by_entity()["bottom_cabinet_no_top_qohxjq_0"]
    picked = Pose2((0.62, -6.15, 0.0053), (0.0, 0.0, 0.7071, 0.7071))
    d = math.hypot(picked.xy[0] - f.position[0], picked.xy[1] - f.position[1]) - f.size[0] / 2
    cfg = ProbeConfig(standoffs_m=(d,), lateral_fracs=(0.0,))
    (x, y, yaw), = anchor_candidates(f, cfg)
    assert math.hypot(x - picked.xy[0], y - picked.xy[1]) < 0.01
    assert math.cos(yaw - picked.yaw) == pytest.approx(1.0, abs=1e-4)


def test_anchor_candidates_order_and_count():
    cfg = ProbeConfig()
    c = anchor_candidates(_furn(0.0), cfg)
    assert len(c) == len(cfg.standoffs_m) * len(cfg.lateral_fracs)
    dists = [x - 1.0 for x, _, _ in c]  # yaw 0: standoff along +x
    assert dists == sorted(dists)  # nearest standoff first


def test_farthest_points_is_deterministic_and_spread():
    pts = [(float(i), float(j)) for i in range(5) for j in range(5)]
    a = farthest_points(pts, 4)
    assert a == farthest_points(list(reversed(pts)), 4)
    assert a[0] == (2.0, 2.0)  # nearest the centroid first
    assert set(a[1:]) <= {(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0)}
    assert farthest_points(pts[:3], 4) == pts[:3]


def test_pose_from_xyyaw_round_trips_yaw():
    for yaw in (-3.0, -1.0, 0.0, 0.7, 3.1):
        assert math.cos(pose_from_xyyaw(0, 0, 0, yaw).yaw - yaw) == pytest.approx(1.0, abs=1e-5)


def test_overlay_evidence_round_trip(tmp_path):
    p = Pose2((1.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    ov = SlotEmbodimentOverlay(
        scene="S", robot_id="r1pro", urdf_path="u", urdf_hash="h", status="probed",
        source="probe", robot_config={"model": "r1pro"}, feasibility={},
        start_poses={"kitchen_0": [StartPose("k_p0_y000", p, ["S/a/link_0"])]},
        slots={"S/a/link_0": SlotInteraction(p, True, False, True, "",
                                             {"placed": True, "grasp_from": [0, 2]}),
               "S/a/top": SlotInteraction(p, None, True, True)})
    ov.save(tmp_path / "o.yaml")
    back = SlotEmbodimentOverlay.load(tmp_path / "o.yaml")
    assert back.to_dict() == ov.to_dict()
    assert back.slots["S/a/link_0"].evidence["grasp_from"] == [0, 2]
    assert not back.slots["S/a/link_0"].usable(requires_open=True)
    assert back.slots["S/a/top"].usable(requires_open=False)


@pytest.mark.skipif(not BEHAVIOR.exists() or not (GENERATED / "scenes").exists(),
                    reason="BEHAVIOR assets / scene slots not on this host")
def test_room_map_matches_scene_json_rooms():
    from rummagebench.authoring.tasks.room_map import RoomMap

    for scene in ("Beechwood_0_int", "Merom_1_int"):
        rm = RoomMap.load(BEHAVIOR / "scenes" / scene,
                          BEHAVIOR / "metadata" / "room_categories.txt")
        ss = SceneSlots.load(GENERATED / "scenes" / f"{scene}.yaml")
        assert sorted(set(rm.names.values())) == sorted(ss.rooms)
        for f in ss.furniture:
            assert rm.room_at(f.position[0], f.position[1]) == f.room, f.entity
        pts = rm.free_points("kitchen_0", 0.5, 0.25)
        assert pts and all(rm.room_at(x, y) == "kitchen_0" for x, y in pts)


# --- v5: the reveal pose must see the whole opened drawer floor -------------

def _r1pro_like_camera(pitch_deg=20.0, height=1.62, vfov_deg=58.7, W=640, H=480):
    """Camera at (0, 0, height) looking along world +x, pitched down; optical
    frame +Z forward, +X right, +Y down (world_from_usd_camera convention)."""
    import numpy as np
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, [0.0, 0.0, height]
    fy = (H / 2) / math.tan(math.radians(vfov_deg / 2))
    K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
    return K, T, W, H


def test_floor_region_below_the_view_is_not_in_view():
    from rummagebench.authoring.tasks.probe import (
        drawer_floor_slab, floor_region_corners, points_in_view)
    K, T, W, H = _r1pro_like_camera()
    # opened drawer floor 0.6 m in front of the camera at z 0.75: below the
    # lower image edge
    near = ([0.45, -0.17, 0.745], [0.78, 0.17, 0.75])
    wall = ([0.45, -0.17, 0.75], [0.47, 0.17, 0.82])  # thick walls are not the floor
    slab = drawer_floor_slab([wall, near])
    assert slab == ([0.45, -0.17, 0.745], [0.78, 0.17, 0.75])
    assert not points_in_view(K, T, W, H, floor_region_corners(slab))
    far = ([1.05, -0.17, 0.745], [1.38, 0.17, 0.75])  # stepped back ~0.6 m
    assert points_in_view(K, T, W, H, floor_region_corners(far))
    # behind the camera / non-finite pose never counts as seen
    assert not points_in_view(K, T, W, H, [(-1.0, 0.0, 1.0)])
    import numpy as np
    assert not points_in_view(K, np.full((4, 4), np.nan), W, H, [(1.0, 0.0, 1.0)])
    assert drawer_floor_slab([wall]) is None
