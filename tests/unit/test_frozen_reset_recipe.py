"""UNIT ONLY: reset must not insert physics steps into a frozen recipe."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rummagebench.authoring.tasks import household_state
from rummagebench.sim.omnigibson import backend as module


@pytest.mark.parametrize('motion_ok',[True,False])
def test_early_direct_motion_avoids_idle_wait_but_retains_unstable_rejection(tmp_path,monkeypatch,motion_ok):
    import validate_dense_pilot as validator
    monkeypatch.setenv('YGBENCH_NATIVE_MOTION_CONVERGENCE','1')
    obj=SimpleNamespace(links={},get_position_orientation=lambda:([0.,0.,0.],[0.,0.,0.,1.]))
    settles=[];observations=[]
    backend=SimpleNamespace(_env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args:obj)),settle=settles.append)
    monkeypatch.setattr(validator,'root_speed',lambda *args:100.)
    def observe(*args):
        observations.append(True);return dict(converged=motion_ok,physics_steps=240)
    monkeypatch.setattr(validator,'observe_native_motion',observe)
    spec=SimpleNamespace(placements=[SimpleNamespace(entity='jar')])
    if motion_ok:
        result,motion=validator.observe_convergence(backend,spec,tmp_path)
        assert result['converged'] and result['raw_velocity_windows']==4
        assert motion['physics_steps']==240 and len(settles)==3 and len(observations)==1
    else:
        with pytest.raises(ValueError,match='LAYOUT_NOT_CONVERGED'):
            validator.observe_convergence(backend,spec,tmp_path)
        assert len(settles)==60 and len(observations)==2


def test_direct_motion_detects_oscillation_hidden_at_render_cadence():
    from validate_dense_pilot import pose_motion_metrics
    poses=np.zeros((33,7));poses[:,6]=1
    poses[:,0]=.002*np.sin(np.arange(33)*np.pi/2)
    assert np.max(abs(poses[::4,0]))<1e-12
    measured=pose_motion_metrics(poses,1/120)
    assert measured['max_displacement_m']>.001 and measured['max_linear_speed']>.01
    # Quaternion scale noise and sign flips do not represent rotation.
    poses[:,:3]=0;poses[:,6]=np.where(np.arange(33)%2,1.0000001,-.9999999)
    measured=pose_motion_metrics(poses,1/120)
    assert measured['max_rotation_rad']==0 and measured['max_angular_speed']==0


@pytest.mark.parametrize('dt',[0,float('inf'),float('nan')])
def test_direct_motion_rejects_invalid_timebase(dt):
    from validate_dense_pilot import pose_motion_metrics
    with pytest.raises(ValueError,match='INVALID_NATIVE_POSE_TRAJECTORY'):
        pose_motion_metrics([[0,0,0,0,0,0,1]]*2,dt)


@pytest.mark.parametrize('moving',[False,True])
def test_direct_motion_observer_uses_native_poses_and_never_changes_actors(tmp_path,moving):
    from validate_dense_pilot import observe_native_motion
    calls=[];step=[0]
    def forbidden(*args):raise AssertionError('Motion observation cannot freeze or move an actor')
    body=SimpleNamespace(root_link=SimpleNamespace(prim_path='/native/body'),n_joints=0,
        set_position_orientation=forbidden,set_linear_velocity=forbidden,sleep=forbidden)
    def transform():
        x=.002*np.sin(step[0]*np.pi/2) if moving else 0.
        return np.array([[x,0,0,0,0,0,1.]])
    view=SimpleNamespace(count=1,get_transforms=transform,get_velocities=lambda:np.array([[.05,0,0,0,.3,0]]))
    def advance():step[0]+=1;calls.append('physics')
    sim=SimpleNamespace(get_physics_dt=lambda:1/120,
        physics_sim_view=SimpleNamespace(create_rigid_body_view=lambda path:view),step_physics=advance)
    backend=SimpleNamespace(_sim=sim,_env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args:body)))
    spec=SimpleNamespace(placements=[SimpleNamespace(entity='body',receptacle='table')],initial_states={})
    report=observe_native_motion(backend,spec,tmp_path,steps=32)
    assert report['converged'] is (not moving) and len(calls)==32
    assert report['objects']['body']['native_reported_linear_speed_max']==.05
    assert (tmp_path/'native_motion_convergence.private.json').is_file()


def test_frozen_reset_uses_original_member_recipe_after_releasing_grasp(monkeypatch):
    calls=[]
    robot=SimpleNamespace(arm_names=['left'],_ag_obj_in_hand={'left':'UNIT_OBJECT'},
                          _release_grasp=lambda **_:calls.append('release'))
    backend=SimpleNamespace(_initial_state='UNIT_DUMP_NOT_THE_RESTORE_INPUT',_robot=robot,
        _scenario=SimpleNamespace(scene=SimpleNamespace(frozen_member_manifest='unit_member.json')),
        validate_physics_state=lambda:calls.append('finite'),
        _validate_spawn_clearance=lambda:calls.append('spawn'))
    def restore(b,path,scenario):
        assert b is backend and path==Path('unit_member.json') and scenario is backend._scenario
        calls.append('original_recipe')
    monkeypatch.setattr(household_state,'restore_frozen_member',restore)
    monkeypatch.setattr(module,'load_state',lambda *a,**k:(_ for _ in ()).throw(AssertionError('legacy load/step prohibited')))
    module.OmniGibsonBackend.reset(backend)
    assert calls==['release','original_recipe','finite','spawn']


@pytest.mark.parametrize('moving',[False,True])
def test_direct_motion_observes_fixed_geometry_without_a_dynamic_body_view(tmp_path,moving):
    from validate_dense_pilot import observe_native_motion
    step=[0]
    body=SimpleNamespace(kinematic_only=True,root_link=SimpleNamespace(),n_joints=0,
        get_position_orientation=lambda:([.002*np.sin(step[0]*np.pi/2) if moving else 0.,0,0],[0,0,0,1]))
    def forbidden(*args):raise AssertionError('Fixed geometry has no dynamic-body view')
    sim=SimpleNamespace(get_physics_dt=lambda:1/120,step_physics=lambda:step.__setitem__(0,step[0]+1),
        physics_sim_view=SimpleNamespace(create_rigid_body_view=forbidden))
    backend=SimpleNamespace(_sim=sim,_env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args:body)))
    spec=SimpleNamespace(placements=[SimpleNamespace(entity='table',receptacle='floor')],initial_states={})
    report=observe_native_motion(backend,spec,tmp_path,steps=32)
    assert report['converged'] is (not moving)
    assert report['objects']['table']['native_reported_linear_speed_max'] is None


def test_same_input_may_reconstruct_different_but_repeatable_world():
    original=np.array([1.,-0.],dtype=np.float32)
    observed=np.array([1.0000001,-0.],dtype=np.float32)
    current=[None];inputs=[]
    def restore(snapshot):
        inputs.append(snapshot['sim'].copy())
        # UNIT fake SDK mutates the supplied input. Caller must deep-copy it.
        snapshot['sim'][0]=99.
        current[0]=observed.copy()
    backend=SimpleNamespace(restore_observe_state=restore,validate_physics_state=lambda:None,
        _validate_spawn_clearance=lambda:None,dump_state=lambda:current[0])
    recipe=household_state.reconstruct_checkpoint(backend,original)
    assert recipe['checkpoint_input_state_sha256']!=recipe['observed_world_state_sha256']
    household_state.reconstruct_checkpoint(backend,original,expected_world_hash=recipe['observed_world_state_sha256'])
    assert original[0]==1. and all(np.array_equal(a,original) for a in inputs)
    observed[1]=0. # Signed-zero drift is still a different world serialization.
    with pytest.raises(ValueError,match='EXACT_WORLD_RECONSTRUCTION_MISMATCH'):
        household_state.reconstruct_checkpoint(backend,original,expected_world_hash=recipe['observed_world_state_sha256'])


def unit_state(objects):
    return {0:{'pos':np.zeros(3,dtype=np.float32),'ori':np.array([0,0,0,1],dtype=np.float32),
               'registry':{'object_registry':objects,'system_registry':{}}}}


def test_composition_preserves_native_inputs_and_current_sdk_registry_order():
    native=unit_state({'background':np.array([1.]),'robot':np.array([2.])})
    current=unit_state({'robot':np.array([2.1]),'target':np.array([3.]),'background':np.array([1.1])})
    output=household_state.merge_checkpoint_objects(native,current,'target')
    objects=output[0]['registry']['object_registry']
    assert list(objects)==['robot','target','background']
    assert objects['background'][0]==1. and objects['robot'][0]==2. and objects['target'][0]==3.
    objects['background'][0]=10.
    assert native[0]['registry']['object_registry']['background'][0]==1.
    assert current[0]['registry']['object_registry']['background'][0]==1.1


@pytest.mark.parametrize('change',['missing_native','new_native','scene_frame','multiple_scenes'])
def test_composition_refuses_topology_or_frame_changes(change):
    native=unit_state({'background':np.array([1.])})
    current=unit_state({'background':np.array([1.1]),'target':np.array([3.])})
    if change=='missing_native':del current[0]['registry']['object_registry']['background']
    elif change=='new_native':current[0]['registry']['object_registry']['extra']=np.array([4.])
    elif change=='scene_frame':current[0]['pos'][0]=1.
    else:current[1]=current[0]
    with pytest.raises(ValueError,match='FROZEN_CHECKPOINT_'):
        household_state.merge_checkpoint_objects(native,current,'target')
