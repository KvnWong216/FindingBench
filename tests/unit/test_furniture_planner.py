"""Contracts that prevent furnishing from changing tasks or certifying fake support."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts'))
import furniture_planner as p


def setup(room='bathroom',mode='inside'):
    loc=dict(id='cab::drawer',support='cab',mode=mode,room_kind=room,
        bounds=[[0,0,.6],[.6,.5,.9]],floor_z=.6)
    assets={c:[dict(category=c,model='native_'+c,size=[.09,.07,.025])]
        for c in {c for r in p.SEMANTICS.values() for cs in r.values() for c in cs}}
    policy=dict(semantics=p.SEMANTICS,assets=assets,max_coverage_fraction=.12,
        edge_clearance_m=.025,neighbour_clearance_m=.025)
    task=dict(id='test',instruction='Find the medicine.',objects=[dict(category='bottle_of_medicine')])
    return loc,policy,task


def test_semantic_room_and_storage_selection():
    loc,policy,task=setup()
    choices=p.candidates(loc,policy,task,[])
    assert choices and all(c['category'] in p.SEMANTICS['bathroom']['inside'] for c in choices)
    assert all(c['category'] not in ('mug','open_end_wrench') for c in choices)
    loc['room_kind']='garage'
    assert {c['category'] for c in p.candidates(loc,policy,task,[])} <= set(p.SEMANTICS['garage']['inside'])


def test_targets_and_comparison_families_cannot_be_reintroduced():
    loc,policy,task=setup('kitchen')
    task.update(instruction='Bring all the fruit.',objects=[dict(category='apple'),dict(category='tablefork')])
    assert not {c['category'] for c in p.candidates(loc,policy,task,[])} & {'apple','tablefork'}
    task['instruction']='Collect all the utensils and all the dishes.'
    assert not {c['category'] for c in p.candidates(loc,policy,task,[])} & {'plate','mug','bowl','coffee_cup'}


def test_cup_asset_alias_does_not_create_a_second_valid_target():
    loc,policy,task=setup('kitchen')
    task.update(instruction='Find my cup.',objects=[dict(category='mug')])
    assert not {c['category'] for c in p.candidates(loc,policy,task,[])} & {'mug','coffee_cup'}


def test_shallow_drawer_cannot_take_a_tall_native_asset():
    loc,policy,task=setup();loc['bounds'][1][2]=.62
    assert p.candidates(loc,policy,task,[])==[]


def test_finite_solver_rejects_occupied_and_excessive_coverage():
    loc,policy,task=setup()
    assert p.candidates(loc,policy,task,[[[0,0,.6],[.6,.5,.9]]])==[]
    policy['max_coverage_fraction']=.001
    assert p.candidates(loc,policy,task,[])==[]


def test_finite_solver_edge_and_neighbour_clearance():
    loc,policy,task=setup();obstacle=[[.2,0,.6],[.6,.5,.9]]
    choices=p.candidates(loc,policy,task,[obstacle]);assert choices
    for c in choices:
        assert c['xy'][0]+c['size'][0]/2+.025 < .2
        assert c['xy'][0]-c['size'][0]/2>=.025


def test_existing_items_count_but_furniture_shells_and_robots_do_not():
    loc,_,_=setup()
    rows={name:dict(fixed_base=False,n_joints=0,robot=False,aabb=[[.1,.1,.6],[.2,.2,.7]])
          for name in ['cab','book','robot','door','outside']}
    rows['robot']['robot']=True;rows['door']['n_joints']=1
    rows['outside']['aabb']=[[.1,.1,1],[.2,.2,1.1]]
    assert p.occupied(loc,rows)==['book']


def test_table_lamp_not_misclassified_as_table_and_cabinets_included():
    assert not p.eligible('table_lamp')
    assert all(p.eligible(c) for c in ('top_cabinet','bottom_cabinet_no_top','desk','nightstand'))


def test_unknown_rooms_never_get_arbitrary_kitchen_props():
    loc,policy,task=setup();loc['room_kind']=None
    assert not p.candidates(loc,policy,task,[])


def test_native_private_office_and_pantry_labels_have_household_semantics():
    assert p.room_kind('private_office_0')=='office'
    assert p.room_kind('pantry_room_0')=='kitchen'
    assert p.room_kind('home_office_12')=='office'
    assert p.room_kind('bathroom_2')=='bathroom'
    assert p.room_kind('unclassified_room_0') is None


def test_sparse_policy_one_per_empty_location_and_no_resizing(tmp_path):
    native=tmp_path/'objects/bar_soap/native/misc';native.mkdir(parents=True)
    (native/'metadata.json').write_text(json.dumps(dict(bbox_size=[.1,.08,.03])))
    tall=tmp_path/'objects/mug/tall/misc';tall.mkdir(parents=True)
    (tall/'metadata.json').write_text(json.dumps(dict(bbox_size=[.05,.05,.25])))
    policy=p.build_policy(tmp_path)
    assert policy['desired_count']==1 and policy['max_coverage_fraction']==.12
    assert policy['assets']['bar_soap'][0]['size']==[.1,.08,.03]
    assert not policy['assets']['mug']


def test_queue_old_sparse_pair_not_accepted_for_all_furniture(tmp_path):
    from enrich_background_queue import accepted_case
    parent=dict(id='old',parent_task_id='F001')
    for name,status in [('validation.json','initialization_validated'),('restore_validation.json','restore_validated')]:
        (tmp_path/name).write_text(json.dumps(dict(status=status,parent_task=parent)))
    assert accepted_case(tmp_path,parent,tmp_path,p.VERSION) is None


def test_quick_authoring_requires_explicit_mode_native_checks_and_valid_cache(tmp_path,monkeypatch):
    from enrich_background_queue import accepted_case
    import spectator_cache
    parent=dict(id='parent',parent_task_id='F001',instruction='Find soy sauce.',runtime_root='/pinned')
    member=dict(parent,id='parent_furnished',parent_variant='parent',enrichment_validation='sparse_authoring')
    report=dict(status='initialization_validated',validation_mode='sparse_authoring',
        policy_version=p.VERSION,parent_task=parent,task_content_preserved=True,navigation_verified=True,
        cache='/cache',process_id=10)
    (tmp_path/'human_ready.json').write_text(json.dumps(dict(tasks=[member])))
    (tmp_path/'validation.json').write_text(json.dumps(report))
    monkeypatch.setattr(spectator_cache,'lookup',lambda *_:('/cache',{}))
    assert accepted_case(tmp_path,parent,tmp_path,p.VERSION) is None
    assert accepted_case(tmp_path,parent,tmp_path,p.VERSION,allow_quick=True)[0]==member
    report['navigation_verified']=False
    (tmp_path/'validation.json').write_text(json.dumps(report))
    assert accepted_case(tmp_path,parent,tmp_path,p.VERSION,allow_quick=True) is None
    report['navigation_verified']=True
    (tmp_path/'validation.json').write_text(json.dumps(report))
    monkeypatch.setattr(spectator_cache,'lookup',lambda *_:None)
    assert accepted_case(tmp_path,parent,tmp_path,p.VERSION,allow_quick=True) is None


def test_furniture_cannot_add_a_new_navigation_obstacle():
    parent=[[[0,0,0],[1,1,.8]]]
    assert p.preserves_navigation([[[.1,.1,.35],[.2,.2,.4]]],parent)
    assert p.preserves_navigation([[[2,2,.8],[3,3,1]]],parent)
    assert not p.preserves_navigation([[[1.01,.1,.35],[1.2,.2,.4]]],parent)


def test_navigation_fast_path_requires_complete_body_outside_floor_relative_band():
    assert p.outside_navigation_band([[0,0,.8],[.1,.1,.9]],0.)
    assert p.outside_navigation_band([[0,0,2.8],[.1,.1,2.9]],2.)
    assert p.outside_navigation_band([[0,0,.1],[.1,.1,.2]],0.)
    assert not p.outside_navigation_band([[0,0,.4],[.1,.1,.8]],0.)
    assert not p.outside_navigation_band([[0,0,.501],[.1,.1,.8]],0.)
    assert not p.outside_navigation_band([[0,0,.8],[.1,.1,.7]],0.)
    assert not p.outside_navigation_band([[0,0,.8],[.1,.1,.9]],float('nan'))


def test_native_probe_logs_do_not_leak_tensor_objects_to_json():
    import torch
    from furnish_native import probe_record
    hit=dict(hit=True,rigidBody='/cabinet/drawer',position=torch.tensor([1.,2.,3.]),
             normal=torch.tensor([0.,0.,1.]),unused_tensor=torch.ones(2))
    record=probe_record(hit)
    assert json.loads(json.dumps(record))['position']==[1.,2.,3.]


def test_physics_only_settle_preserves_steps_and_restores_render_on_exception():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from furnish_native import settle_physics
    class Sim:
        _render_on_step=True
        @contextmanager
        def render_on_step(self,value):
            before=self._render_on_step;self._render_on_step=value
            yield
            self._render_on_step=before
    sim=Sim();calls=[]
    def settle(n):calls.append((n,sim._render_on_step))
    backend=SimpleNamespace(_sim=sim,settle=settle)
    settle_physics(backend,90)
    assert calls==[(90,False)] and sim._render_on_step is True
    def fail(n):raise ValueError('physics failed')
    backend.settle=fail
    with pytest.raises(ValueError,match='physics failed'):settle_physics(backend,90)
    assert sim._render_on_step is True
    backend._furniture_physics_only=False;backend.settle=settle
    settle_physics(backend,90)
    assert calls[-1]==(90,True)


def test_parked_new_props_are_not_compared_as_original_world_motion():
    original={'door':{'position':[0,0,0]}}
    whole={'door':{'position':[0,0,0]},'furnishing_000':{'position':[100,100,5]}}
    assert p.original_control_poses(whole,original)==original
    with pytest.raises(ValueError,match='ORIGINAL_OBJECT_REMOVED_FROM_CONTROL'):
        p.original_control_poses({},original)


def test_passive_door_must_stay_inside_measured_unchanged_hinge_range():
    import math
    def pose(theta,x=0):return dict(links={'hinge':[[x,0,0],[0,0,math.sin(theta/2),math.cos(theta/2)]]},joints=[theta])
    assert p.passive_door_within_baseline(pose(1.28),pose(1.33),pose(1.32))
    assert not p.passive_door_within_baseline(pose(1.28),pose(1.33),pose(1.35))
    assert not p.passive_door_within_baseline(pose(1.28),pose(1.33),pose(1.32,.01))


def test_frozen_relations_do_not_reapply_clearance_translation():
    loc=dict(entity='prop',support='table',mode='surface',position_xy=[1.,2.])
    assert p.frozen_placement(loc)==dict(entity='prop',receptacle='table',relation='on_top')
    loc.update(mode='inside',link='drawer')
    assert p.frozen_placement(loc)['link']=='drawer'
    assert loc['position_xy']==[1.,2.]


def test_reload_instability_blacklists_only_the_observed_failed_location(tmp_path):
    from enrich_background_queue import unstable_furniture_locations
    (tmp_path/'resolved_plan.json').write_text(json.dumps(dict(locations=[
        dict(id='drawer_a',entity='furnishing_000'),dict(id='table_b',entity='furnishing_001')])))
    (tmp_path/'restore_validation.json').write_text(json.dumps(dict(error='RELOAD_FURNITURE_UNSTABLE:furnishing_001')))
    assert unstable_furniture_locations(tmp_path)=={'table_b'}
    (tmp_path/'restore_validation.json').write_text(json.dumps(dict(error='FROZEN_MEMBER_SCENARIO_OR_HASH_MISMATCH')))
    assert not unstable_furniture_locations(tmp_path)
