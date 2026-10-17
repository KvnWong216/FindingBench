"""Unit-only synthetic native records; never used by production generation."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from rummagebench.authoring.layout.indoor_background import native_context, native_start
from rummagebench.authoring.tasks.plan import canonical_json


@pytest.mark.parametrize('offset,gap,excluded',[(0.,.02,True),(.4,.02,False),(0.,.20,False)])
def test_stacked_support_exclusion_keeps_bodies_and_partial_surfaces(offset,gap,excluded):
    from rummagebench.authoring.layout.indoor_background import exclude_covered_support_envelopes
    # Synthetic rotated geometry; the same classification applies in any yaw.
    q=[0.,0.,2**-.5,2**-.5]
    rows=[dict(name='base',size=[1.,2.,.8],center=[0.,0.,.4],quaternion=q,support=True),
          dict(name='slab',size=[1.1,2.1,.04],center=[0.,offset,.82+gap],quaternion=q,support=True)]
    skipped=exclude_covered_support_envelopes(rows)
    assert len(rows)==2 and rows[1]['support']
    assert rows[0]['support']==(not excluded)
    assert bool(skipped)==excluded
    if excluded:assert skipped[0]['covering_entity']=='slab'


@pytest.mark.parametrize('lamp_width,active',[(.32,False),(.05,True)])
def test_native_occupants_bound_background_additions_without_deleting_entities(lamp_width,active):
    from rummagebench.authoring.layout.indoor_background import qualify_supports_for_pool
    context=dict(objects=[dict(name='table',support=True,size=[.38,.43,.5],center=[0,0,.25],quaternion=[0,0,0,1]),
        dict(name='lamp',support=False,size=[lamp_width,.35,.6],center=[0,0,.8],quaternion=[0,0,0,1])],
        skipped_supports=[],retained_native_objects=['table','lamp'])
    qualify_supports_for_pool(context,{'unit/mug':dict(bbox_size=[.10,.10,.10])})
    assert context['objects'][0]['support']==active
    assert len(context['objects'])==2 and context['retained_native_objects']==['table','lamp']
    if not active:assert context['skipped_supports'][0]['blocking_entities']==['lamp']


def test_context_preserves_native_occupants_and_rooms(tmp_path):
    root=tmp_path
    records={}; states={}
    for name,cat,model,pos,rooms in [
        ('counter','countertop','unitcounter',[1.,2.,.8],['kitchen_0']),
        ('occupant','bowl','unitbowl',[1.,2.,1.],['kitchen_0']),
        ('table','console_table','unittable',[5.,2.,.7],['living_room_0'])]:
        p=root/'objects'/cat/model/'misc/metadata.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(bbox_size=[1.,2.,.2],base_link_offset=[.1,.2,0.])))
        records[name]={'args':dict(category=cat,model=model,scale=[1.,1.,1.],in_rooms=rooms)}
        states[name]={'root_link':dict(pos=pos,ori=[0.,0.,0.,1.])}
    path=root/'scenes/unit/json/unit_best.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(objects_info={'init_info':records},state=dict(
        pos=[0.,0.,0.],ori=[0.,0.,0.,1.],registry={'object_registry':states}))))
    context=native_context(root,'unit')
    assert context['retained_native_objects']==['counter','occupant','table']
    assert len(context['objects'])==3
    assert {o['name'] for o in context['objects'] if o['support']}=={'counter','table'}
    assert next(o for o in context['objects'] if o['name']=='occupant')['support'] is False
    assert np.allclose(context['objects'][0]['center'],[1.1,2.2,.8])
    assert b'target' not in canonical_json(context)
    states['occupant']['root_link']['pos']=[1.4,2.,1.]
    doc=json.loads(path.read_text());doc['state']['registry']['object_registry']=states
    path.write_text(json.dumps(doc))
    assert native_context(root,'unit')['inputs'] != context['inputs']


def test_native_start_cpu_has_plain_floats_and_excludes_occupancy(tmp_path):
    root=tmp_path;layout=root/'scenes/unit/layout';layout.mkdir(parents=True)
    (root/'metadata').mkdir();(root/'metadata/room_categories.txt').write_text('kitchen\n')
    ins=np.ones((300,300),np.uint8);sem=ins.copy();trav=ins*255
    trav[:,:80]=0
    for name,array in [('floor_insseg_0.png',ins),('floor_semseg_0.png',sem),('floor_trav_0.png',trav)]:
        Image.fromarray(array).save(layout/name)
    xy,inputs=native_start(root,'unit','kitchen_0')
    assert all(type(v) is float for v in xy)
    assert xy[0]>0 and len(inputs)==4
    with pytest.raises(ValueError,match='ROOM_MISSING'):native_start(root,'unit','bathroom_0')


def test_cropped_clearance_matches_full_map_exactly(tmp_path):
    from scipy.ndimage import distance_transform_edt
    root=tmp_path;layout=root/'scenes/unit/layout';layout.mkdir(parents=True)
    (root/'metadata').mkdir();(root/'metadata/room_categories.txt').write_text('kitchen\n')
    ins=np.zeros((800,900),np.uint8);ins[137:410,522:803]=1
    sem=ins.copy();trav=ins*255;trav[180:240,640:670]=0
    for name,array in [('floor_insseg_0.png',ins),('floor_semseg_0.png',sem),('floor_trav_0.png',trav)]:
        Image.fromarray(array).save(layout/name)
    expected=distance_transform_edt(np.pad((ins==1)&(trav>0),1))[1:-1,1:-1]
    row,col=np.unravel_index(int(expected.argmax()),expected.shape)
    xy,_=native_start(root,'unit','kitchen_0')
    assert xy==[(col-450)*.01,(row-400)*.01]


def test_per_surface_coverage_cannot_be_replaced_by_total_count():
    from rummagebench.authoring.layout.indoor_background import verify_result
    req=dict(native_context=dict(objects=[dict(name='a',support=True),dict(name='b',support=True)],
                                retained_native_objects=['a','b']),
             per_surface=1,assets={'bowl/unit':{}})
    def item(name,support):
        return dict(name=name,support_name=support,category='bowl',model_id='unit',role='clutter',
                    position=[0.,0.,1.],orientation_deg=0.,aabb_size=[.1,.1,.1])
    res=dict(status='constraints_satisfied',violations={'daily_items_a':0,'daily_items_b':0},
             placements=[item('one','a'),item('two','b')])
    verify_result(req,res)
    res['placements'][1]['support_name']='a'
    with pytest.raises(ValueError,match='INCOMPLETE_LIVING_BACKGROUND'):verify_result(req,res)
    res['placements'][1]['support_name']='b'
    del res['violations']['daily_items_b']
    with pytest.raises(ValueError,match='CONSTRAINT_EVIDENCE'):verify_result(req,res)


def test_home_initial_view_does_not_follow_hidden_target(tmp_path, monkeypatch):
    """Unit-only generator substitutes isolate the public scenario boundary."""
    import copy
    import importlib.util
    from types import SimpleNamespace
    import yaml
    import rummagebench.authoring.layout.indoor_background as background

    scripts = Path(__file__).resolve().parents[2] / 'scripts/tasks'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('unit_behavior_pipeline', scripts/'behavior_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = dict(retained_native_objects=['left', 'right'], objects=[
        dict(name=name, category='cabinet', openable=True, rooms=['kitchen_0'], center=center)
        for name, center in [('left', [-2., 1., 1.]), ('right', [2., -1., 1.])]])
    def prepare(*args, **kwargs):
        args[4].mkdir()
        return dict(living_context=dict(native=copy.deepcopy(native), inputs={}), assets={}, objects=[])
    monkeypatch.setattr(background, 'prepare_home', prepare)
    monkeypatch.setattr(background, 'native_start', lambda *args: ([0., 0.], {}))
    monkeypatch.setattr(module, 'asset_record', lambda *args, **kw: dict(category='bowl', model='unit'))
    configs = []
    for name in ('left', 'right'):
        config = tmp_path/f'{name}.yaml'
        config.write_text(yaml.safe_dump(dict(scene='unit', room='kitchen_0', daily_assets=[],
            per_surface=1, solver_steps=10, smoke_container=name, target={})))
        out = tmp_path/name
        module.prepare_living_home(SimpleNamespace(config=config, assets=tmp_path,
            seed=31, out=out, upstream=tmp_path, python='unused'))
        configs.append(yaml.safe_load((out/'scenario.yaml').read_text()))
    assert configs[0]['anchors'] == configs[1]['anchors']
    assert configs[0]['instruction'] == configs[1]['instruction']
    assert configs[0]['initial_states'] == configs[1]['initial_states']
    assert configs[0]['placements'] != configs[1]['placements']
    assert all(c['expose_available_skills'] is False for c in configs)
