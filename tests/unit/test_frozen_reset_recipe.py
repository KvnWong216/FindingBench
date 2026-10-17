"""UNIT ONLY: reset must not insert physics steps into a frozen recipe."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rummagebench.authoring.tasks import household_state
from rummagebench.sim.omnigibson import backend as module


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
