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


def test_sole_known_wrist_cannot_masquerade_as_head():
    with pytest.raises(RuntimeError):
        select_head_rgb_sensor({'robot:left_realsense_link:Camera:0':{'rgb':np.zeros((2,2,3))}})


def test_private_modalities_never_mix_wrist_data_into_head_rgb():
    from rummagebench.sim.omnigibson.observation import capture_private_frame
    wrist={'rgb':np.ones((2,2,3),dtype=np.uint8),'depth':np.ones((2,2)),
           'seg_instance':np.full((2,2),9)}
    head={'rgb':np.full((2,2,3),77,dtype=np.uint8),'seg_instance':np.full((2,2),7)}
    env=SimpleNamespace(get_obs=lambda:([{'robot':{'robot:left_realsense_link:Camera:0':wrist,
                                                  'robot:zed_link:Camera:0':head}}],None))
    frame=capture_private_frame(env,'robot')
    assert np.all(frame['rgb']==77) and np.all(frame['seg']==7)
    assert np.isnan(frame['depth']).all() and frame['sensor_name']=='robot:zed_link:Camera:0'


def test_default_camera_pose_and_prim_match_rgb_head_identity(monkeypatch):
    """Unit sensors only; no simulator asset or camera protocol acceptance."""
    import sys
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    b=OmniGibsonBackend()
    wrist=SimpleNamespace(prim_path='/unit_wrist',get_position_orientation=lambda:([9,0,0],[0,0,0,1]))
    head=SimpleNamespace(prim_path='/unit_head',get_position_orientation=lambda:([0,0,2],[0,0,0,1]))
    sensors={'robot:left_realsense_link:Camera:0':wrist,'robot:zed_link:Camera:0':head}
    b._robot=SimpleNamespace(name='robot',sensors=sensors)
    b._env=SimpleNamespace(get_obs=lambda:([{'robot':{n:{'rgb':np.zeros((2,2,3))} for n in sensors}}],None))
    lazy=SimpleNamespace(omni=SimpleNamespace(isaac=SimpleNamespace(core=SimpleNamespace(utils=
        SimpleNamespace(prims=SimpleNamespace(get_prim_at_path=lambda p:p))))))
    parent=SimpleNamespace(lazy=lazy)
    monkeypatch.setitem(sys.modules,'omnigibson',parent);monkeypatch.setitem(sys.modules,'omnigibson.lazy',lazy)
    assert b._camera_prim()=='/unit_head'
    assert np.array_equal(b._camera_pose()[0],[0,0,2])
    assert b._camera_prim('missing') is None and b._camera_pose('missing') is None


def test_movable_furniture_blocks_motion_but_held_body_is_not_external_obstacle():
    b=SimpleNamespace(entity_names=lambda:["table"],
        robot_entity_names=lambda:set(),
        describe_entity=lambda n:SimpleNamespace(fixed_base=False),
        entity_aabb=lambda n:([-1,-1,0],[1,1,.5]),
        is_holding=lambda n:False)
    assert not base_pose_collision_free(b,0,0,0,.4,.03)
    b.is_holding=lambda n:True
    assert base_pose_collision_free(b,0,0,0,.4,.03)
