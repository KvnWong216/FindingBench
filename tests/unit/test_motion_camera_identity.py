"""Regressions for confirmed robot-self obstacle and wrist/head confusion."""
from types import SimpleNamespace
import numpy as np
import pytest
from rummagebench.skills.move import base_pose_collision_free
from rummagebench.sim.omnigibson.observation import select_head_rgb_sensor

def test_controlled_robot_is_not_an_obstacle_but_other_robots_are():
    box=([-1,-1,0],[1,1,1])
    b=SimpleNamespace(entity_names=lambda:["self"],
        robot_entity_names=lambda:{"self"},
        describe_entity=lambda n:SimpleNamespace(fixed_base=True,category="agent"),
        entity_aabb=lambda n:box)
    assert base_pose_collision_free(b,0,0,0,.3,.03)
    b.entity_names=lambda:["self","other_robot"]
    assert not base_pose_collision_free(b,0,0,0,.3,.03)
    b.entity_names=lambda:["self","wall"]
    assert not base_pose_collision_free(b,0,0,0,.3,.03)

def test_head_camera_is_selected_after_both_wrist_cameras():
    rgb={"rgb":np.zeros((2,2,3))}
    obs={n:rgb for n in ["robot:left_realsense_link:Camera:0",
        "robot:right_realsense_link:Camera:0","robot:zed_link:Camera:0"]}
    assert select_head_rgb_sensor(obs)[0]=="robot:zed_link:Camera:0"
    del obs["robot:zed_link:Camera:0"]
    with pytest.raises(RuntimeError,match="ambiguous"):select_head_rgb_sensor(obs)

def test_missing_rgb_does_not_choose_segmentation_sensor():
    with pytest.raises(RuntimeError):select_head_rgb_sensor({"head":{"seg_instance":np.ones((2,2))}})


def test_movable_furniture_blocks_motion_but_held_body_is_not_external_obstacle():
    b=SimpleNamespace(entity_names=lambda:["table"],
        robot_entity_names=lambda:set(),
        describe_entity=lambda n:SimpleNamespace(fixed_base=False),
        entity_aabb=lambda n:([-1,-1,0],[1,1,.5]),
        is_holding=lambda n:False)
    assert not base_pose_collision_free(b,0,0,0,.4,.03)
    b.is_holding=lambda n:True
    assert base_pose_collision_free(b,0,0,0,.4,.03)
