"""CPU contracts only. All objects/state/OG modules here are synthetic test fixtures.

No result in this file is simulator acceptance or a production asset.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from rummagebench.authoring.tasks.native_occupancy import (
    background_mismatches, capture_native_occupancy, checked_box,
    require_known_slots, state_hash, summarize_slot, verify_source,
)
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.authoring.tasks.slots import SceneSlots, Slot, SlotGeometry


def slot(link=None, relation='inside'):
    return Slot('S/cab/interior', 'S', 'kitchen_0', 'kitchen', 'cab', 'cabinet', 'test_model',
                relation, 'cabinet_interior', link, True, True,
                SlotGeometry((1., 1., 1.), 1., (1., 1.), 0.), (0., 0., 0.), 0.)


def obj(box):
    return {'aabb': box, 'state': {'pose': [0., 0., 0.]}}


BOX = [[0., 0., 0.], [1., 1., 1.]]
FAR = [[2., 2., 2.], [3., 3., 3.]]


def test_empty_requires_measured_envelope_and_complete_geometry():
    assert summarize_slot(slot(), {'cab': obj(BOX), 'mug': obj(FAR)}, [], BOX, [])['status'] == 'confirmed_empty'
    for objects, envelope, errors in [({'mug': obj(None)}, BOX, []),
                                      ({}, None, []), ({}, BOX, ['PARTICLE_SYSTEM_OCCUPANCY_UNRESOLVED'])]:
        assert summarize_slot(slot(), objects, [], envelope, errors)['status'] == 'unknown'


def test_intersection_without_relation_is_unknown_and_never_empty():
    rec = summarize_slot(slot(), {'cab': obj(BOX), 'native_item': obj(BOX)}, [], BOX, [])
    assert rec['status'] == 'unknown' and rec['possible_occupants'] == ['native_item']


def test_nested_contents_preserve_ids_once_without_double_summing_volume():
    relations = [{'entity': 'bowl', 'parent': 'cab', 'relation': 'inside'},
                 {'entity': 'apple', 'parent': 'bowl', 'relation': 'inside'},
                 {'entity': 'apple', 'parent': 'cab', 'relation': 'inside'}]
    rec = summarize_slot(slot(), {'cab': obj(BOX), 'bowl': obj(BOX), 'apple': obj(BOX)}, relations, BOX, [])
    assert rec['status'] == 'confirmed_occupied'
    assert rec['occupants'] == ['apple', 'bowl'] and rec['possible_occupants'] == []
    assert 'used_volume' not in rec


def test_parent_inside_does_not_prove_drawer_membership():
    rec = summarize_slot(slot(link='drawer'), {'cab': obj(BOX), 'bowl': obj(BOX)},
                         [{'entity': 'bowl', 'parent': 'cab', 'relation': 'inside'}], BOX, [])
    assert rec['status'] == 'unknown'


def test_unknown_candidates_reject_whole_requested_set_without_pruning():
    ids = ['one', 'two']
    doc = {'slots': [{'slot_id': 'one', 'status': 'confirmed_empty', 'geometry_complete': True},
                     {'slot_id': 'two', 'status': 'unknown', 'geometry_complete': False}]}
    with pytest.raises(ValueError, match='two'):
        require_known_slots(doc, ids)
    assert ids == ['one', 'two']
    with pytest.raises(ValueError, match='absent'):
        require_known_slots(doc, ['absent'])


@pytest.mark.parametrize('bad', [None, [[0, 0], [1, 1]], [[0, 0, 0], [-1, 1, 1]],
                                [[0, 0, 0], [float('nan'), 1, 1]]])
def test_invalid_geometry_is_rejected(bad):
    with pytest.raises(ValueError):
        checked_box(bad)


def test_serialized_state_hash_binds_dtype_shape_and_values():
    a = np.array([1., 2.], dtype=np.float32)
    assert state_hash(a) == state_hash(a.copy())
    assert state_hash(a) != state_hash(a.astype(np.float64))
    assert state_hash(a) != state_hash(a.reshape((1, 2)))
    assert state_hash(a) != state_hash(a + 1)
    with pytest.raises(ValueError):
        state_hash(np.array([float('nan')]))


def test_background_pose_state_instance_or_membership_change_is_rejected():
    frozen = {'mug': {'model': 'fixture', 'instance': 'native_mug', 'pose': [1, 2, 3], 'open': False}}
    assert background_mismatches(frozen, deepcopy(frozen)) == []
    for key, value in [('model', 'other'), ('instance', 'new'), ('pose', [1, 2, 4]), ('open', True)]:
        current = deepcopy(frozen); current['mug'][key] = value
        assert background_mismatches(frozen, current) == ['mug']
    assert background_mismatches(frozen, {}) == ['mug']


@pytest.fixture
def fake_loaded_scene(monkeypatch, tmp_path):
    """A stub scene for testing the real extractor and pre-import builder hook."""
    import rummagebench.sim.omnigibson.backend as backend_module
    import rummagebench.sim.omnigibson.env_factory as factory
    asset_path = tmp_path / 'fixture.usd'
    asset_path.write_text('synthetic unit test asset; never a real USD')
    class Inside: pass
    class OnTop: pass
    class State:
        def __init__(self, names): self.names = names
        def get_value(self, parent): return parent.name in self.names
    class Object:
        def __init__(self, name, box, inside=(), on_top=(), rooms=()):
            self.name = name; self.box = box; self.category = 'fixture'; self.model = 'fixture'
            self._usd_path = str(asset_path)
            self.in_rooms = rooms; self.states = {Inside: State(inside), OnTop: State(on_top)}
        def dump_state(self, serialized=False): return {'pose': [0., 0., 0.], 'joints': []}
    objects = [Object('cab', BOX), Object('bowl', BOX, inside=('cab',)),
               Object('apple', BOX, inside=('bowl', 'cab')),
               Object('other_room', FAR, rooms=('living_room_0',))]
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'objects_info': {'init_info': {o.name: {'class_name': 'DatasetObject'} for o in objects}}}))
    robot = SimpleNamespace(name='robot_0')
    scene = SimpleNamespace(scene_file=str(source), objects=objects + [robot], robots=[robot], systems={})
    state = np.array([1., 2., 3.], dtype=np.float32)
    class Sim:
        def is_playing(self): return True
        def dump_state(self, serialized=True): return state.copy()
    sim = Sim()
    og = ModuleType('omnigibson'); og.sim = sim
    states = ModuleType('omnigibson.object_states'); states.Inside = Inside; states.OnTop = OnTop
    spawns = ModuleType('omnigibson.objects')
    def forbid_spawn(*args, **kwargs): raise AssertionError('native-only capture must precede task import')
    spawns.DatasetObject = forbid_spawn
    for name, module in [('omnigibson', og), ('omnigibson.object_states', states), ('omnigibson.objects', spawns)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backend_module, 'seed_everything', lambda _: None)
    monkeypatch.setattr(backend_module, 'apply_runtime_env', lambda: None)
    monkeypatch.setattr(backend_module, 'build_env', lambda _: (SimpleNamespace(scene=scene), og))
    monkeypatch.setattr(factory, 'build_env_config', lambda _: {'scene': {'scene_model': 'S'}})
    backend = backend_module.OmniGibsonBackend()
    backend._env = SimpleNamespace(scene=scene); backend._sim = sim
    backend.entity_aabb = lambda name: next(o.box for o in objects if o.name == name)
    scenario = SimpleNamespace(scene=SimpleNamespace(model='S', source='behavior', frozen_member_manifest=None), objects=[SimpleNamespace(name='target')])
    backend._scenario = scenario
    slots = SceneSlots('S', 1, {'scene_json_sha256': sha256_path(source)}, {'kitchen_0': 'kitchen'}, [slot()])
    return backend, scenario, slots, source, scene, state


def test_builder_hook_captures_same_loaded_native_scene_before_task_import(fake_loaded_scene):
    backend, scenario, slots, source, _, _ = fake_loaded_scene
    captured = {}
    def hook(b):
        doc, state = capture_native_occupancy(b, slots, source_scene_path=source)
        captured.update(doc)
        return {'inventory_sha256': doc['inventory_sha256']}
    report = backend.setup(scenario, native_scene_capture=hook, native_scene_only=True)
    assert report['spawned'] == [] and report['stage'] == 'native_loaded'
    assert set(captured['objects']) == {'cab', 'bowl', 'apple', 'other_room'}
    assert captured['physical_acceptance'] == 'pending'
    assert captured['slots'][0]['status'] == 'confirmed_occupied'
    verify_source(captured, backend, slots)


def test_inventory_reuse_rejects_hash_state_and_source_conflicts(fake_loaded_scene):
    backend, _, slots, source, _, state = fake_loaded_scene
    doc, _ = capture_native_occupancy(backend, slots, source_scene_path=source)
    corrupted = deepcopy(doc); corrupted['counts'] = {}
    with pytest.raises(ValueError, match='inventory hash'):
        verify_source(corrupted, backend, slots)
    state[0] += 1
    with pytest.raises(ValueError, match='source state'):
        verify_source(doc, backend, slots)
    state[0] -= 1
    source.write_text('{}')
    with pytest.raises(ValueError, match='scene source'):
        verify_source(doc, backend, slots)


def test_loaded_inventory_omission_or_wrong_source_is_rejected(fake_loaded_scene):
    backend, _, slots, source, scene, _ = fake_loaded_scene
    bad = deepcopy(slots); bad.source['scene_json_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='scene source'):
        capture_native_occupancy(backend, bad, source_scene_path=source)
    scene.objects.pop(0)
    with pytest.raises(ValueError, match='native entity inventory mismatch'):
        capture_native_occupancy(backend, slots, source_scene_path=source)


def test_native_only_without_capture_fails_before_gpu_import():
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    with pytest.raises(ValueError, match='requires an occupancy capture'):
        OmniGibsonBackend().setup(None, native_scene_only=True)


def test_native_asset_and_restored_geometry_changes_fail_closed(fake_loaded_scene):
    backend, _, slots, source, scene, _ = fake_loaded_scene
    doc, _ = capture_native_occupancy(backend, slots, source_scene_path=source)
    old_box = scene.objects[0].box
    scene.objects[0].box = FAR
    with pytest.raises(ValueError, match='restored geometry'):
        verify_source(doc, backend, slots)
    scene.objects[0].box = old_box
    asset = Path(scene.objects[0]._usd_path)
    asset.write_text('changed synthetic unit test geometry')
    with pytest.raises(ValueError, match='asset source'):
        verify_source(doc, backend, slots)


def test_native_restore_rejects_changed_scene_identity(fake_loaded_scene):
    backend, scenario, slots, source, _, _ = fake_loaded_scene
    doc, _ = capture_native_occupancy(backend, slots, source_scene_path=source)
    scenario.scene.model = 'different_scene'
    with pytest.raises(ValueError, match='scene identity mismatch'):
        verify_source(doc, backend, slots)
