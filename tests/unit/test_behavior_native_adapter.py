"""Synthetic metadata here is a unit fixture, never a generated dataset."""
from pathlib import Path
import json
import math
import numpy as np
import pytest
from rummagebench.authoring.layout.spec import LayoutConfig
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes
from rummagebench.sim.omnigibson.behavior_adapter import (
    asset_record, prepare_layout, load_manifest, assembly_env_config)
from rummagebench.core.scenario import SceneSpec
from rummagebench.feasibility.ik_solver import quat_rotate


@pytest.fixture
def fixture_assets(tmp_path):
    root = tmp_path/'unit_only_assets'
    for cat, model, size in [('cabinet', 'unitcab', [.4, .8, .9]),
                              ('table', 'unittable', [.8, 1.2, .8]),
                              ('bowl', 'unitbowl', [.1, .12, .06])]:
        d = root/'objects'/cat/model
        (d/'misc').mkdir(parents=True)
        (d/'usd').mkdir()
        (d/'usd'/f'{model}.encrypted.usd').write_text('UNIT TEST ONLY; NOT USD')
        (d/'misc/metadata.json').write_text(json.dumps(dict(
            bbox_size=size, base_link_offset=[.07, -.03, .01],
            openable_joint_ids=[[1, 'door']] if cat == 'cabinet' else [])))
    return root


def make(root, seed=31):
    cfg = LayoutConfig(id='unit', base_scene='__empty__', storage_units=2,
                       work_surfaces=1, clutter_per_surface=1,
                       placement_regions=[[-3., 1., 3., 5.]], robot_spawn=[0., 0.])
    pool = dict(storage=[dict(category='cabinet', model='unitcab')],
                surface=[dict(category='table', model='unittable')],
                clutter=[dict(category='bowl', model='unitbowl')])
    return prepare_layout(cfg, pool, root, seed)


def write(tmp_path, doc):
    path = tmp_path/'assembly.json'
    path.write_text(json.dumps(doc))
    return path


def test_native_model_and_origin_and_front(fixture_assets):
    doc = make(fixture_assets)
    assert doc == make(fixture_assets)
    assert doc != make(fixture_assets, seed=32)
    assert doc['evaluation_ready_episodes'] == 0
    for item in doc['objects']:
        p, cfg = item['placement'], item['og_config']
        a = doc['assets'][f"{p['category']}/{p['model_id']}"]
        assert cfg['type'] == 'DatasetObject' and cfg['scale'] == [1., 1., 1.]
        assert p['aabb_size'] == [a['bbox_size'][1], a['bbox_size'][0], a['bbox_size'][2]]
        assert np.allclose(np.array(cfg['position']) + quat_rotate(cfg['orientation'], a['base_link_offset']), p['position'])
        assert np.allclose(quat_rotate(cfg['orientation'], [1., 0., 0.]), [0., -1., 0.])


def test_source_binding_and_complete_house(fixture_assets, tmp_path):
    doc = make(fixture_assets)
    path = write(tmp_path, doc)
    assert load_manifest(path) == doc
    assert assembly_env_config(path, {}, '__empty__')['scene']['type'] == 'Scene'
    with pytest.raises(ValueError, match='BASE_SCENE'):
        assembly_env_config(path, {}, 'wrong_house')
    doc['base_scene'] = 'Beechwood_0_int'
    doc['manifest_sha256'] = sha256_bytes(canonical_json({k:v for k,v in doc.items() if k != 'manifest_sha256'}))
    path = write(tmp_path, doc)
    template = dict(type='InteractiveTraversableScene', load_room_types=None)
    cfg = assembly_env_config(path, template, 'Beechwood_0_int')
    assert cfg['scene'] == dict(template, scene_model='Beechwood_0_int')
    assert 'scene_model' not in template
    assert SceneSpec(model='x', source='behavior_layout', assembly_manifest=str(path))
    assert SceneSpec(model='x').assembly_manifest is None
    with pytest.raises(ValueError):
        SceneSpec(model='x', source='infinigen')
    with pytest.raises(ValueError):
        SceneSpec(model='x', source='behavior_layout')
    with pytest.raises(ValueError):
        SceneSpec(model='x', source='behavior', assembly_manifest=str(path))


def test_missing_or_changed_asset_rejected(fixture_assets, tmp_path):
    with pytest.raises(ValueError, match='MISSING'):
        asset_record(fixture_assets, 'cabinet', 'scene_instance_unitcab_0')
    path = write(tmp_path, make(fixture_assets))
    (fixture_assets/'objects/cabinet/unitcab/usd/unitcab.encrypted.usd').write_text('CHANGED')
    with pytest.raises(ValueError, match='ASSET_CHANGED'):
        load_manifest(path)


def test_resigned_scale_and_duplicate_rejected(fixture_assets, tmp_path):
    for mutation in ('scale', 'duplicate'):
        doc = make(fixture_assets)
        if mutation == 'scale':
            doc['objects'][0]['og_config']['scale'] = [1., 1., 2.]
        else:
            doc['objects'].append(doc['objects'][0])
        doc['manifest_sha256'] = sha256_bytes(canonical_json({k:v for k,v in doc.items() if k != 'manifest_sha256'}))
        with pytest.raises(ValueError, match='TRANSFORM_MISMATCH'):
            load_manifest(write(tmp_path, doc))


def test_impossible_layout_rejected_whole(fixture_assets):
    cfg = LayoutConfig(id='unit', base_scene='__empty__', storage_units=2,
                       work_surfaces=0, clutter_per_surface=0,
                       placement_regions=[[0., 0., .1, .1]])
    with pytest.raises(ValueError, match='LAYOUT_REJECTED'):
        prepare_layout(cfg, dict(storage=[dict(category='cabinet', model='unitcab')]), fixture_assets, 31)


def test_existing_builder_dispatches_native_layout(fixture_assets, tmp_path):
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.sim.omnigibson.env_factory import build_env_config
    path = write(tmp_path, make(fixture_assets))
    scenario = ScenarioSpec.model_validate(dict(
        id='unit', instruction='unit fixture',
        scene=dict(model='__empty__', source='behavior_layout', assembly_manifest=str(path)),
        robot=dict(model='r1pro', init_anchor='start'),
        anchors=dict(start=dict(position=[0, 0, .0053], orientation=[0, 0, 0, 1])),
        target=dict(entity='target', category='bowl'),
        objects=[dict(name='target', category='bowl', model='unitbowl')]))
    cfg = build_env_config(scenario)
    assert cfg['scene']['type'] == 'Scene'
    assert cfg['robots'][0]['model'] == 'r1pro'
    assert len(cfg['objects']) == 4
    assert all(o['type'] == 'DatasetObject' for o in cfg['objects'])
    assert all(o['name'] != 'target' for o in cfg['objects'])  # backend owns task spawning
