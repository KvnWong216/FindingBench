from pathlib import Path
import copy
import sys
import pytest


def test_clearance_journal_path_alone_does_not_create_a_new_physical_proposal(tmp_path):
    import json
    from expert_storage_repair import authoring_recipe_identity
    first=tmp_path/'a.json';second=tmp_path/'b.json'
    physical=dict(scene_model='house',wall_warps=[],translations=[dict(entity='cabinet',delta=[.2,0,0])],removed_objects=[])
    first.write_text(json.dumps(dict(physical,expert_fixture_replacements=['old journal'])))
    second.write_text(json.dumps(dict(physical,expert_fixture_replacements=['new journal'])))
    a=dict(scene=dict(model='house',clearance_manifest=str(first)),placements=[dict(entity='jar',position_xy=[0,0])])
    b=copy.deepcopy(a);b['scene']['clearance_manifest']=str(second)
    assert authoring_recipe_identity(a)==authoring_recipe_identity(b)
    physical['translations'][0]['delta']=[.3,0,0];second.write_text(json.dumps(physical))
    assert authoring_recipe_identity(a)!=authoring_recipe_identity(b)
    b=copy.deepcopy(a);b['placements'][0]['position_xy']=[.1,0]
    assert authoring_recipe_identity(a)!=authoring_recipe_identity(b)


def test_flat_stock_piles_reserve_all_real_footprints_and_keep_nonflat_stock_separate():
    from expert_storage_repair import flat_stock_stack_proposals
    objects={n:dict(category='hardback') for n in ['a','b','c','d','e','target','sample']}
    objects['bookend']=dict(category='bookend')
    items={n:[.3-i*.02,.2-i*.01,.02] for i,n in enumerate(['a','b','c','d','e','target'])}
    items['bookend']=[.1,.1,.2];before=copy.deepcopy(items)
    stacks,envelopes=flat_stock_stack_proposals(objects,items,['target'])
    assert len(stacks)==2 and {n for pile in stacks for n in pile}==set(items)-{'bookend'}
    assert next(p for p in stacks if 'target' in p)[1]=='target'
    for pile in stacks:
        envelope=envelopes[pile[0]]
        assert all(envelope[0]>=items[n][0] and envelope[1]>=items[n][1] for n in pile)
        assert envelope[2]==pytest.approx(sum(items[n][2] for n in pile))
    assert items==before and 'sample' not in envelopes


def test_open_boxes_and_game_sets_are_not_assumed_to_have_flat_closed_tops():
    from expert_storage_repair import flat_stock_stack_proposals
    objects={name:dict(category=name) for name in ['storage_box','tray','plate','chess_set','jigsaw_puzzle']}
    items={name:[.3,.3,.1] for name in objects}
    assert flat_stock_stack_proposals(objects,items,['storage_box'])==([],{})


def test_open_storage_data_revision_is_explicit_and_preserves_task_inventory(tmp_path,monkeypatch):
    import json,yaml,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as r
    original=dict(id='task',instruction='Find the box among the storage-room containers and bring it to the table.',
        scene=dict(model='house'),target=dict(entity='box'),goal=[dict(entity='box',kind='on_top',receptacle='table')],
        robot=dict(init_anchor='start'),anchors=dict(start=dict(position=[1,1,0])),
        objects=[dict(name='box',category='storage_box',model='native'),dict(name='jar',category='jar',model='native'),
                 dict(name='source',category='bottom_cabinet',model='native',fixed_base=True)],
        placements=[dict(entity='box',relation='inside',receptacle='source'),dict(entity='jar',relation='inside',receptacle='source'),
                    dict(entity='source',relation='on_top',receptacle='floor',position_xy=[2,2])],
        initial_states=dict(source=dict(open=False)))
    path=tmp_path/'scenario.yaml';path.write_text(yaml.safe_dump(original))
    row=dict(id='task',scenario=str(path),source_receptacle='source',source_room='storage_room_0',
        destination_receptacle='table',destination_room='storage_room_0',target_entities=['box'],
        inserted_furniture=[dict(name='source')],restoration_entities=['jar','source'])
    assert r.reauthor_generic_storage_on_work_surface(row,'evidence',tmp_path/'no-approval') is None
    assets=tmp_path/'assets';meta=assets/'objects/breakfast_table/iaritq/misc/metadata.json';meta.parent.mkdir(parents=True);meta.write_text('{"bbox_size":[0.7,0.7,0.75]}')
    monkeypatch.setattr(r.g,'ASSETS',assets)
    entities=dict(floor=dict(category='floors',aabb=[[0,0,0],[10,10,0]]),table=dict(category='desk',aabb=[[4,4,0],[5,5,1]],in_rooms=['storage_room_0']))
    def replay(_,furniture):
        bodies=copy.deepcopy(entities)
        if furniture:bodies['source']=dict(category='bottom_cabinet',aabb=[[1.5,1.5,0],[2.5,2.5,1]])
        return NS(entities=bodies,inserted_furniture=furniture,map=NS(size=3,to_map=lambda *args:(1,1)),
            free=np.ones((3,3),bool),labels=np.ones((3,3)),route=lambda *args:dict(start=[1,1,0]))
    monkeypatch.setattr(r,'layout_for_spec',lambda _:None);monkeypatch.setattr(r.storage,'replay_layout',replay)
    def augment(base,room,kind,name,**kwargs):
        assert r.storage.MODELS[kind]==('breakfast_table','iaritq')
        added=dict(name=name,category='breakfast_table',model='iaritq',floor='floor',position_xy=[2,2])
        layout=copy.copy(base);layout.entities=dict(base.entities,**{name:dict(category='breakfast_table',aabb=[[1.6,1.6,0],[2.4,2.4,.75]])})
        layout.inserted_furniture=[added]
        return layout,dict(name=name)  # augment returns a source record, not a name.
    monkeypatch.setattr(r.storage,'augment',augment);monkeypatch.setattr(r,'dimensions',lambda _:dict(box=[.2]*3,jar=[.1]*3))
    monkeypatch.setattr(r,'plan_explicit_surfaces',lambda *args,**kwargs:[dict(physical_validation_required=True)])
    monkeypatch.setattr(r,'preflight_recipe',lambda *args:dict(status='heuristic_fit'));monkeypatch.setattr(r.g,'load_scenario',lambda _:None)
    result,journal=r.reauthor_generic_storage_on_work_surface(row,'evidence',tmp_path/'out',expert_approved=True)
    spec=yaml.safe_load(Path(result['scenario']).read_text())
    for key in ['instruction','goal','target','robot','anchors']:assert spec[key]==original[key]
    assert spec['objects'][:2]==original['objects'][:2] and 'source' not in spec['initial_states']
    assert all(p['relation']=='on_top' for p in spec['placements'])
    assert result['restoration_entities']==['jar'] and len(journal['relation_changes'])==2
    assert journal['physical_validation_required'] and not journal['human_completion_verified']
    entities['native_source']=dict(category='desk',aabb=[[1.6,1.6,0],[2.4,2.4,.75]],in_rooms=['storage_room_0'])
    def unnecessary_augmentation(*args,**kwargs):raise AssertionError('Existing valid native surface must be tried before constructing furniture')
    monkeypatch.setattr(r.storage,'augment',unnecessary_augmentation)
    reused,_=r.reauthor_generic_storage_on_work_surface(row,'evidence',tmp_path/'reuse',expert_approved=True)
    assert reused['source_receptacle']=='native_source'
    original['instruction']='Find the box inside the storage-room cupboard.';path.write_text(yaml.safe_dump(original))
    assert r.reauthor_generic_storage_on_work_surface(row,'evidence',tmp_path/'explicit-cupboard',expert_approved=True) is None


def test_double_translation_repair_requires_matching_native_error_and_preserves_world_target(tmp_path,monkeypatch):
    import hashlib,json,yaml,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as r
    manifest=tmp_path/'clearance.json';manifest.write_text(json.dumps(dict(translations=[
        dict(entity='slab',delta=[.5,-1.4,0]),dict(entity='other',delta=[9,9,0])])))
    original=dict(id='task',instruction='Find the apple in the kitchen basket.',scene=dict(clearance_manifest=str(manifest)),
        objects=[dict(name='basket'),dict(name='apple')],robot={},anchors={},target=dict(entity='apple'),goal=[],
        placements=[dict(entity='basket',relation='on_top',receptacle='slab',position_xy=[1.1,-6.8]),
                    dict(entity='apple',relation='inside',receptacle='basket')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original));evidence=tmp_path/'evidence';evidence.mkdir()
    report=dict(status='failed',scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        error='position_xy [1.6, -8.2] is outside the top surface of slab')
    path=evidence/'validation.json';path.write_text(json.dumps(report))
    monkeypatch.setattr(r,'layout_for_spec',lambda _:NS(entities=dict(slab=dict(aabb=[[0,-7,.8],[1.4,-6,.9]]))))
    monkeypatch.setattr(r.g,'load_scenario',lambda _:None)
    row=dict(id='task',scenario=str(source))
    result,journal=r.repair_double_translated_surface_pose(row,evidence,tmp_path/'out')
    changed=yaml.safe_load(Path(result['scenario']).read_text())
    point=np.asarray(changed['placements'][0]['position_xy'])
    assert np.allclose(point+[.5,-1.4],[1.1,-6.8])
    expected=copy.deepcopy(original);expected['placements'][0]['position_xy']=point.tolist()
    assert changed==expected and journal['physical_validation_required']
    report['error']='position_xy [9, 9] is outside the top surface of slab';path.write_text(json.dumps(report))
    assert r.repair_double_translated_surface_pose(row,evidence,tmp_path/'unrelated') is None

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/v2'))
from expert_storage_repair import assert_preserved,blocking_storage,repair_kind


def test_capacity_repair_never_rebinds_a_target_or_described_basket(tmp_path,monkeypatch):
    import yaml
    import expert_storage_repair as repair
    spec=dict(instruction='Find the basket.',target=dict(entity='basket'),goal=[],
        objects=[dict(name='basket',category='wicker_basket',model='native')],
        placements=[dict(entity='apple',relation='inside',receptacle='basket')])
    source=tmp_path/'scenario.yaml';source.write_text(yaml.safe_dump(spec));row=dict(scenario=str(source),target_entities=['basket'])
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:pytest.fail('Protected basket cannot enter geometry selection'))
    assert repair.repair_rigid_container_capacity(row,'evidence',tmp_path/'out') is None
    spec['target']['entity']='apple';row['target_entities']=['apple'];spec['instruction']='Find the apple in the red wicker basket.'
    source.write_text(yaml.safe_dump(spec))
    assert repair.repair_rigid_container_capacity(row,'evidence',tmp_path/'colour') is None


def test_covered_counter_binding_routes_a_basket_through_its_real_support(tmp_path,monkeypatch):
    import json,numpy as np,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    survey=tmp_path/'survey.json';survey.write_text('survey')
    original=dict(id='task',instruction='Find the apple in the kitchen basket and bring it to the table.',
        scene=dict(model='house'),objects=[dict(name='basket'),dict(name='apple')],robot=dict(init_anchor='start'),
        anchors=dict(start=dict(position=[0,0,0])),target=dict(entity='apple'),
        placements=[dict(entity='basket',relation='on_top',receptacle='cabinet'),dict(entity='apple',relation='inside',receptacle='basket')],
        goal=[dict(entity='apple',kind='on_top',receptacle='table')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original))
    entities=dict(cabinet=dict(category='bottom_cabinet',aabb=[[0,0,0],[1,1,.8]],in_rooms=['kitchen']),
        slab=dict(category='countertop',aabb=[[0,0,.8],[1,1,.84]],in_rooms=['kitchen']),
        table=dict(category='table',aabb=[[2,0,0],[3,1,.8]],in_rooms=['kitchen']))
    calls=[]
    def route(src,room,dst,dstroom):
        calls.append((src,dst));assert (src,dst)==('slab','table');return dict(start=[0,0])
    layout=NS(path=survey,entities=entities,map=NS(size=1,to_map=lambda *args:(0,0)),
        free=np.ones((1,1),bool),labels=np.ones((1,1)),route=route)
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:layout);monkeypatch.setattr(repair.storage,'replay_layout',lambda l,f:l)
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(basket=[.2,.2,.2],apple=[.05,.05,.05]))
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    row=dict(id='task',scenario=str(source),source_receptacle='basket',source_room='kitchen',destination_receptacle='table',destination_room='kitchen',target_entities=['apple'])
    result,journal=repair.repair_covered_surface_binding(row,'evidence',tmp_path/'out')
    spec=yaml.safe_load(Path(result['scenario']).read_text())
    assert result['source_receptacle']=='basket' and spec['placements'][0]['receptacle']=='slab'
    assert spec['placements'][1]==original['placements'][1] and spec['goal']==original['goal']
    assert spec['objects']==original['objects'] and calls==[('slab','table')]


def test_registered_background_contact_rule_retains_closed_cabinet_and_removes_stale_translation(tmp_path,monkeypatch):
    import hashlib,json,numpy as np,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    import track_full_asset_queue as tracking
    monkeypatch.setattr(repair.g,'ROOT',tmp_path)
    survey=tmp_path/'survey.json';survey.write_text('native survey')
    log=tmp_path/'native.log';log.write_text('measured inter-cabinet contacts')
    manifest=tmp_path/'original_clearance.json';manifest.write_text(json.dumps(dict(scene_model='house',translations=[dict(entity='obstacle',delta=[0,-2,0])],removed_objects=[])))
    original=dict(id='task',instruction='Find the jar in the living room and bring it to the counter.',
        scene=dict(model='house',clearance_manifest=str(manifest)),robot=dict(init_anchor='start'),anchors=dict(start=dict(position=[0,0,0])),
        objects=[dict(name='jar')],placements=[dict(entity='jar',relation='inside',receptacle='task_cabinet')],
        target=dict(entity='jar'),goal=[dict(entity='jar',kind='on_top',receptacle='counter')],initial_states=dict(background=dict(open=False),task_cabinet=dict(open=False)))
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original))
    rule=dict(revision_id='verified',source_entity='background',source_asset='bottom_cabinet/native',
        required_declared_state=dict(open=False),required_translation=dict(entity='obstacle',delta=[0,-2,0]),
        base_survey_sha256=hashlib.sha256(survey.read_bytes()).hexdigest(),native_failure_evidence=str(log),
        native_failure_evidence_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),translations=[],removed_objects=['obstacle'],
        positive_native_pairs=[dict(recipe='proof.yaml',evidence='proof')])
    primary=dict(rule,source_entity='unrelated',additional_rules=[rule])
    registry=tmp_path/'build/expert_scene_contact_repairs/index.json';registry.parent.mkdir(parents=True);registry.write_text(json.dumps(dict(scenes=dict(house=primary))))
    layout=NS(path=survey,entities=dict(background=dict(category='bottom_cabinet',model='native'),obstacle={}),
        map=NS(size=1,to_map=lambda *args:(0,0)),free=np.ones((1,1),bool),labels=np.ones((1,1)),route=lambda *args:dict(start=[0,0]))
    monkeypatch.setattr(repair,'layout_for',lambda _:layout);monkeypatch.setattr(repair,'layout_for_spec',lambda _:layout)
    monkeypatch.setattr(repair.storage,'replay_layout',lambda l,f:l);monkeypatch.setattr(repair,'dimensions',lambda _:{})
    monkeypatch.setattr(repair,'preflight_recipe',lambda *args:dict(status='heuristic_fit'));monkeypatch.setattr(repair,'clear_unstated_links',lambda _:[])
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None);monkeypatch.setattr(tracking,'accepted_native_pair',lambda *args:True)
    row=dict(id='task',scenario=str(source),source_receptacle='task_cabinet',source_room='living_room',destination_receptacle='counter',destination_room='kitchen')
    revised,journal=repair.apply_registered_scene_contact_repair(row,'evidence',tmp_path/'out')
    spec=yaml.safe_load(Path(revised['scenario']).read_text());actual=json.loads(Path(spec['scene']['clearance_manifest']).read_text())
    assert spec['initial_states']==original['initial_states'] and spec['objects']==original['objects']
    assert actual['removed_objects']==['obstacle'] and actual['translations']==[]
    # A body becoming part of a task, or a lost positive proof, invalidates reuse.
    protected=copy.deepcopy(original);protected['objects'].append(dict(name='obstacle'));source.write_text(yaml.safe_dump(protected))
    assert repair.apply_registered_scene_contact_repair(row,'evidence',tmp_path/'protected') is None
    source.write_text(yaml.safe_dump(original));monkeypatch.setattr(tracking,'accepted_native_pair',lambda *args:False)
    assert repair.apply_registered_scene_contact_repair(row,'evidence',tmp_path/'unverified') is None


@pytest.mark.parametrize('nested_source',[False,True])
def test_initial_goal_overlap_moves_stock_but_preserves_matching_sample(tmp_path,monkeypatch,nested_source):
    import hashlib,json,numpy as np,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    original=dict(id='task',instruction='There is a sample on the counter. Find matching cereal in kitchen storage and bring it to the sample.',
        target=dict(entity='target'),robot=dict(init_anchor='start'),anchors=dict(start=dict(position=[0,0,0])),
        objects=[dict(name=n,category='box_of_cereal',model='native') for n in ['target','clutter','comparison_sample']],
        placements=[dict(entity=n,relation='on_top',receptacle='delivery',position_xy=[0,0]) for n in ['target','clutter','comparison_sample']],
        goal=[dict(entity='target',kind='on_top',receptacle='delivery')])
    if nested_source:
        original['objects'].append(dict(name='basket',category='shopping_basket',model='native'))
        for p in original['placements'][:2]:p.update(relation='inside',receptacle='basket');p.pop('position_xy')
        original['placements'].append(dict(entity='basket',relation='on_top',receptacle='delivery',position_xy=[0,0]))
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original));evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(error='INVALID_INITIAL_GOAL',scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
    (evidence/'native_validation.log').write_text('all placements physically valid')
    entities={name:dict(category='countertop',model='native',aabb=[[x,0,0],[x+1,1,1]],in_rooms=['kitchen']) for name,x in [('delivery',0),('search',2)]}
    layout=NS(entities=entities,map=NS(size=1,to_map=lambda *args:(0,0)),free=np.ones((1,1),bool),labels=np.ones((1,1)),route=lambda *args:dict(start=[0,0]))
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:layout)
    monkeypatch.setattr(repair.storage,'replay_layout',lambda l,f:l)
    monkeypatch.setattr(repair,'dimensions',lambda _:{n:[.1,.1,.1] for n in ['target','clutter','comparison_sample','basket']})
    monkeypatch.setattr(repair,'plan_explicit_surfaces',lambda *args,**kwargs:[dict(receptacle='search')])
    monkeypatch.setattr(repair,'preflight_recipe',lambda *args:dict(status='heuristic_fit'))
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    row=dict(id='task',scenario=str(source),source_receptacle='delivery',destination_receptacle='delivery',source_room='kitchen',destination_room='kitchen')
    if nested_source:row['source_receptacle']='basket'
    result,journal=repair.repair_open_surface_source(row,evidence,tmp_path/'out',93,
        **(dict(reviewed_surface_entity='basket',reviewed_surface_categories=('countertop',)) if nested_source else {}))
    revised=yaml.safe_load(Path(result['scenario']).read_text())
    placements={p['entity']:p for p in revised['placements']}
    assert placements['target']['receptacle']==placements['clutter']['receptacle']==('basket' if nested_source else 'search')
    if nested_source:
        assert placements['basket']['receptacle']=='search'
        assert placements['target']['relation']==placements['clutter']['relation']=='inside'
    assert placements['comparison_sample']==original['placements'][2]
    assert revised['goal']==original['goal'] and revised['instruction']==original['instruction']
    assert revised['objects']==original['objects'] and result['source_receptacle']==('basket' if nested_source else 'search')


def test_local_spawn_repair_preserves_task_heading_and_uses_physical_evidence(tmp_path,monkeypatch):
    import hashlib,json,numpy as np,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    n=100;free=np.ones((n,n),dtype=bool)
    layout=NS(map=NS(size=n,resolution=.1,room_ins=np.ones((n,n),dtype=int),names={1:'kitchen'}),
        free=free,labels=np.ones((n,n),dtype=int),room_clear=np.ones((n,n))*.4,radius=.38,
        points=np.array([(r,c) for r in range(40,61) for c in range(40,61)]),
        entities={'cabinet':dict(aabb=[[.4,-.3,0],[1.,.3,1.]]),
                  'table':dict(aabb=[[1.8,-.3,0],[2.2,.3,1.]])},
        approaches=lambda name,room:[(0.,(50,60),[1.,0.],0.)])
    original=dict(id='task',instruction='Find the jar and put it on the table in front of you.',
        scene={'model':'house'},robot=dict(init_anchor='start'),
        anchors=dict(start=dict(position=[0.,0.,.0053],orientation=[0.,0.,0.,1.])),
        target=dict(entity='jar'),goal=[dict(kind='on_top',entity='jar',receptacle='table')],
        objects=[dict(name='jar')],placements=[dict(entity='jar',relation='inside',receptacle='cabinet')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original))
    row=dict(id='task',scenario=str(source),source_receptacle='cabinet',source_room='kitchen',
        destination_receptacle='table',destination_room='kitchen')
    evidence=tmp_path/'evidence';evidence.mkdir()
    report=dict(scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),error='cabinet open=True (declared False)')
    (evidence/'validation.json').write_text(json.dumps(report))
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:layout)
    monkeypatch.setattr(repair.storage,'replay_layout',lambda l,f:l)
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    # An Open error alone is not proof that the robot caused it.
    assert repair.repair_robot_spawn(row,evidence,tmp_path/'absent') is None
    pair=['/World/scene_0/articulated__cabinet/link_0','/World/scene_0/controllable__g1_bench__robot_0/hand']
    event=dict(entity='cabinet',contacts=[pair])
    (evidence/'native_validation.log').write_text('[open-state-audit] '+json.dumps(event)+'\n')
    result,journal=repair.repair_robot_spawn(row,evidence,tmp_path/'out')
    revised=yaml.safe_load(Path(result['scenario']).read_text())
    position=revised['anchors']['start']['position']
    assert .3<=np.linalg.norm(np.array(position[:2]))<=1.2
    assert position[0]<0 and position[2]==original['anchors']['start']['position'][2]
    assert revised['anchors']['start']['orientation']==original['anchors']['start']['orientation']
    contract=copy.deepcopy(original);contract['anchors']['start']['position']=position
    assert revised==contract
    assert journal['edit']['observed_robot_contacts']==[pair] and journal['physical_validation_required']
    assert result['route']['path_xy'][0]==position[:2]
    assert journal['edit']['screening']['front_of_you_preserved']
    # Do not use another recipe's native evidence or silently relax room/heading constraints.
    source.write_text(source.read_text()+'\n# changed recipe\n')
    assert repair.repair_robot_spawn(row,evidence,tmp_path/'stale') is None
    layout.points=np.array([[50,54]])
    layout.entities['table']['aabb']=[[-2.2,-.3,0],[-1.8,.3,1.]]
    assert repair.nearby_spawn_proposal(original,row,layout) is None
    layout.entities['table']['aabb']=[[1.8,-.3,0],[2.2,.3,1.]]
    layout.map.room_ins[50,54]=2
    assert repair.nearby_spawn_proposal(original,row,layout) is None


def test_repair_contract_preserves_targets_and_required_negatives():
    original=dict(id='task',instruction='Find the salt.',goal=[dict(kind='on_top',entity='salt',receptacle='table')],
        target={'entity':'salt'},objects=[{'name':'salt'},{'name':'sugar'}],
        placements=[dict(entity=n,relation='inside',receptacle='old') for n in ['salt','sugar']])
    repaired=copy.deepcopy(original)
    for p in repaired['placements']:p['receptacle']='new'
    repaired['objects'].append({'name':'new'})
    assert_preserved(original,repaired)
    repaired['objects'].remove({'name':'sugar'})
    with pytest.raises(ValueError,match='Original asset changed'):assert_preserved(original,repaired)
    repaired=copy.deepcopy(original);repaired['instruction']='Find the sugar.'
    with pytest.raises(ValueError,match='Semantic field changed'):assert_preserved(original,repaired)


def test_failed_reference_center_uses_same_counter_clear_of_raised_geometry(tmp_path,monkeypatch):
    import json,hashlib,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    original=dict(id='task',instruction='Find the plate matching the sample on this counter.',
        scene=dict(model='house'),target=dict(entity='target'),goal=[],
        objects=[dict(name='comparison_sample')],
        placements=[dict(entity='comparison_sample',relation='on_top',receptacle='counter',position_xy=[1.,1.])])
    source=tmp_path/'scenario.yaml';source.write_text(yaml.safe_dump(original))
    evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(status='failed',
        scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        error="placement failed verification after physics settled: ['comparison_sample on_top counter']")))
    layout=NS(entities={'counter':dict(aabb=[[0,0,1],[2,2,1.1]]),
        'stove':dict(aabb=[[.8,.8,1.1],[1.2,1.2,1.3]])})
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:layout)
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(comparison_sample=[.1,.1,.1]))
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    row=dict(id='task',scenario=str(source),reference_entity='comparison_sample')
    result,journal=repair.repair_unstable_reference_surface(row,evidence,tmp_path/'out')
    revised=yaml.safe_load(Path(result['scenario']).read_text())
    position=revised['placements'][0]['position_xy']
    assert position!=[1.,1.] and revised['placements'][0]['receptacle']=='counter'
    expected=copy.deepcopy(original);expected['placements'][0]['position_xy']=position
    assert revised==expected
    assert journal['previous_position_xy']==[1.,1.] and journal['physical_validation_required']
    assert position[0]-.075>=1.2 or position[0]+.075<=.8 or position[1]-.075>=1.2 or position[1]+.075<=.8
    assert result['reference_pose_repairs'][0]['before_position_xy']==[1.,1.]


@pytest.mark.parametrize('authored_tall_fixture,shared_unanchored',[(False,False),(True,False),('short',False),('blocked',False),(False,True)])
def test_failed_scene_cabinet_is_replaced_by_real_asset_at_same_site(tmp_path,monkeypatch,authored_tall_fixture,shared_unanchored):
    import json,hashlib,yaml,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    original=dict(id='task',instruction='Find the jar in the kitchen cupboards.',scene=dict(model='house'),
        target=dict(entity='jar'),goal=[dict(kind='on_top',entity='jar',receptacle='table')],
        robot=dict(init_anchor='start'),anchors=dict(start=dict(position=[1.,1.,.0053],orientation=[0,0,0,1])),
        objects=[dict(name='jar',category='jar',model='original')],
        placements=[dict(entity='jar',relation='inside',receptacle='cabinet')],initial_states=dict(cabinet=dict(open=False)))
    if authored_tall_fixture:
        original['objects'].append(dict(name='cabinet',category='wardrobe',model='broken',fixed_base=True))
        original['placements'].append(dict(entity='cabinet',relation='on_top',receptacle='floor',position_xy=[0.,0.]))
    if shared_unanchored:
        original['goal'][0]['receptacle']='cabinet'
        original['objects'].append(dict(name='sample',category='jar',model='original'))
        original['placements'].append(dict(entity='sample',relation='on_top',receptacle='cabinet'))
    recipe=tmp_path/'scenario.yaml';recipe.write_text(yaml.safe_dump(original))
    evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(status='failed',error='EXPERT_SUPPORT_UNRESOLVED',
        scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())))
    entities=dict(cabinet=dict(category='bottom_cabinet',model='broken',fixed_base=True,
        aabb=[[-.3,-.3,0],[.3,.3,.8]],in_rooms=['kitchen']),
        floor=dict(category='floors',aabb=[[-10,-10,0],[10,10,0]]),
        table=dict(category='table',aabb=[[2,2,0],[3,3,1]]))
    if authored_tall_fixture:
        entities['cabinet'].update(category='wardrobe',aabb=[[-.3,-.3,0],[.3,.3,.4 if authored_tall_fixture=='short' else 1.9]])
        entities['cabinet'].pop('fixed_base')
    if shared_unanchored:entities['cabinet'].update(fixed_base=False,graspable=False)
    base=NS(entities=entities,map=NS(size=3,to_map=lambda *args:(1,1)),
        free=np.ones((3,3),dtype=bool),labels=np.ones((3,3),dtype=int),route=lambda *args:dict(start=[1.,1.,0.]))
    if authored_tall_fixture=='blocked':
        base.free[1,1]=False
        monkeypatch.setattr(repair,'nearby_spawn_proposal',lambda *args:dict(
            position=[2.,2.,.0053],route=dict(start=[2.,2.,0.]),screening={'same_room':True}))
    def replay(layout,furniture):
        result=copy.copy(layout);result.entities=copy.deepcopy(layout.entities)
        for f in furniture:result.entities[f['name']]=dict(f,in_rooms=[f['room']])
        return result
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:base)
    monkeypatch.setattr(repair.storage,'replay_layout',replay)
    monkeypatch.setattr(repair,'blocking_storage',lambda *args:'cabinet')
    monkeypatch.setattr(repair,'candidate_models',lambda *args:[('bottom_cabinet','replacement')])
    monkeypatch.setattr(repair.storage,'model_openable',lambda *args:True)
    monkeypatch.setattr(repair.storage,'enrich_constructed_interiors',lambda *args:None)
    monkeypatch.setattr(repair.storage,'footprint_fits_floor',lambda *args:True)
    monkeypatch.setattr(repair,'compartment_assignment',lambda *args,**kwargs:({'jar':'drawer'},dict(status='feasible_model')))
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(jar=[.1]*3,cabinet=[.5,.5,.75]))
    monkeypatch.setattr(repair,'preflight_recipe',lambda *args:dict(status='heuristic_fit'))
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    real_read=repair.read
    monkeypatch.setattr(repair,'read',lambda path,*args:dict(bbox_size=[.5,.5,.75]) if str(path).endswith('metadata.json') else real_read(path,*args))
    row=dict(id='task',scenario=str(recipe),source_receptacle='cabinet',source_room='kitchen',
        destination_receptacle='table',destination_room='kitchen')
    if authored_tall_fixture:
        row['inserted_furniture']=[dict(entities['cabinet'],name='cabinet',room='kitchen',
                                         floor='floor',position_xy=[0.,0.])]
    if shared_unanchored:
        row['destination_receptacle']='cabinet'
        assert repair.repair_scene_fixture_in_place(row,evidence,tmp_path/'default') is None
    result,journal=repair.repair_scene_fixture_in_place(row,evidence,tmp_path/'out',
        allow_unanchored_scene_source=shared_unanchored,
        allow_shared_delivery_replacement=shared_unanchored)
    spec=yaml.safe_load(Path(result['scenario']).read_text())
    replacement_name='cabinet' if authored_tall_fixture else 'replacement_cabinet'
    contract=copy.deepcopy(original)
    if authored_tall_fixture=='blocked':contract['anchors']['start']['position']=[2.,2.,.0053]
    if shared_unanchored:contract['goal'][0]['receptacle']=replacement_name
    repair.assert_task_inventory_preserved(contract,spec,editable_fixture_names={'cabinet'} if authored_tall_fixture else ())
    assert spec['goal']==contract['goal'] and spec['initial_states']=={replacement_name:dict(open=False)}
    assert spec['objects'][0]==original['objects'][0]
    assert dict(name=replacement_name,category='bottom_cabinet',model='replacement',fixed_base=True) in spec['objects']
    assert spec['placements'][0]['receptacle']==replacement_name
    clearance=json.loads(Path(spec['scene']['clearance_manifest']).read_text())
    assert clearance['removed_objects']==([] if authored_tall_fixture else ['cabinet'])
    assert result['inserted_furniture'][0]['position_xy']==[0.,0.]
    assert journal['edit']['before_asset']==('wardrobe/broken' if authored_tall_fixture else 'bottom_cabinet/broken') and journal['physical_validation_required']
    if shared_unanchored:
        assert next(p for p in spec['placements'] if p['entity']=='sample')['receptacle']==replacement_name
        assert result['destination_receptacle']==replacement_name
        assert journal['shared_delivery_alias']['physical_site_unchanged']
    if authored_tall_fixture=='blocked':
        assert result['spawn_repairs'][-1]['authorization']=='user_requested_change_initial_spawn_20261114'
        assert spec['anchors']['start']['orientation']==original['anchors']['start']['orientation']


def test_furniture_replay_cache_respects_different_derived_scene_geometry():
    import numpy as np
    from types import SimpleNamespace as NS
    import constructed_storage as storage
    first=NS(scene='cache_test_house',entities={'original':{}},free=np.ones((3,3),dtype=bool),
        map=NS(size=3),_authoring_geometry_identity='before-removal')
    second=copy.copy(first);second.entities={};second._authoring_geometry_identity='after-removal'
    a=storage.replay_layout(first,[]);b=storage.replay_layout(second,[])
    assert 'original' in a.entities and 'original' not in b.entities
    assert a is not b


def test_removed_only_scene_edit_drops_entity_without_inventing_free_floor(tmp_path,monkeypatch):
    import json,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    manifest=tmp_path/'clearance.json';manifest.write_text(json.dumps(dict(removed_objects=['cabinet'])))
    free=np.ones((3,3),dtype=bool);free[1,1]=False
    base=NS(entities={'cabinet':dict(aabb=[[0,0,0],[1,1,1]])},free=free,
        doc=dict(footprint_boxes=[[0,0,1,1]]),map=NS(size=3))
    monkeypatch.setattr(repair,'layout_for',lambda _:base)
    result=repair.layout_for_spec(dict(scene=dict(model='house',clearance_manifest=str(manifest))))
    assert 'cabinet' not in result.entities and 'cabinet' in base.entities
    assert np.array_equal(result.free,free) and not result.free[1,1]
    assert result._authoring_geometry_identity


def test_surface_spacing_fallback_changes_only_authoring_proposal(monkeypatch):
    import expert_storage_repair as repair
    attempts=[]
    def solve(items,volumes,**kwargs):
        attempts.append(kwargs)
        return dict(status='feasible_model' if kwargs['margin_m']==.015 else 'unknown')
    monkeypatch.setattr(repair,'solve_layout',solve)
    result=repair.solve_surface_proposal({'jar':[.1]*3},[.4,.8])
    assert result['status']=='feasible_model' and result['physical_validation_required']
    assert [a['margin_m'] for a in attempts]==[.025,.015]
    assert all(a['allowed_yaws']=={'jar':[0]} for a in attempts)


def test_runtime_counterfactual_uses_prior_native_3d_placements_not_2d_rejection(tmp_path,monkeypatch):
    import hashlib,json,yaml
    from types import SimpleNamespace
    import expert_storage_repair as repair
    original=dict(id='task',objects=[dict(name='apple')],
                  placements=[dict(entity='apple',relation='inside',receptacle='basket')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original));evidence=tmp_path/'evidence';evidence.mkdir()
    report=dict(scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),status='failed',
                error='LAYOUT_NOT_CONVERGED: see settling.private.json',build=dict(placements=[
                    dict(entity='apple',placed=True,verified=True)]))
    (evidence/'validation.json').write_text(json.dumps(report))
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:SimpleNamespace(entities={}))
    monkeypatch.setattr(repair.storage,'replay_layout',lambda l,f:l)
    calls=[]
    def include(layout,spec,**kwargs):calls.append(kwargs);return layout
    monkeypatch.setattr(repair,'include_task_receptacles',include)
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(apple=[.1]*3))
    monkeypatch.setattr(repair,'preflight_recipe',lambda *args:dict(status='rejected_candidate'))
    row=dict(id='task',scenario=str(source))
    result=repair.request(dict(action='direct',case=dict(record=row,evidence=str(evidence)),out=str(tmp_path/'out')))
    assert result['status']=='proposed' and Path(result['record']['scenario']).read_bytes()==source.read_bytes()
    assert calls[0]['source_sha']==report['scenario_sha256']
    assert result['evidence']['prior_verified_native_placements'] and result['evidence']['physical_validation_required']
    report['build']['placements'][0]['verified']=False
    (evidence/'validation.json').write_text(json.dumps(report))
    result=repair.request(dict(action='direct',case=dict(record=row,evidence=str(evidence)),out=str(tmp_path/'bad')))
    assert result['status']=='deferred'


def test_prune_only_unused_added_fixtures_preserving_task_and_survey(tmp_path,monkeypatch):
    import yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    import track_full_asset_queue as tracker
    spec=dict(id='task',instruction='Find the camera among the bedroom electronics.',
        scene=dict(model='house'),target=dict(entity='camera'),
        goal=[dict(kind='on_top',entity='camera',receptacle='table')],
        objects=[dict(name='camera',category='digital_camera',model='native')]+[
            dict(name=n,category='cabinet',model='native',fixed_base=True)
            for n in ['used','orphan','scene_fixture','declared_closed']],
        placements=[dict(entity='camera',relation='inside',receptacle='used')]+[
            dict(entity=n,relation='on_top',receptacle='floor')
            for n in ['used','orphan','scene_fixture','declared_closed']],
        initial_states={'declared_closed':dict(open=False)})
    source=tmp_path/'scenario.yaml';source.write_text(yaml.safe_dump(spec));raw=source.read_bytes()
    row=dict(id='task',scenario=str(source),source_receptacle='used',destination_receptacle='table',
        target_entities=['camera'],distractor_count=0,restoration_entities=['orphan','used'],
        inserted_furniture=[dict(name=n) for n in ['used','orphan','scene_fixture','declared_closed']])
    base=NS(entities={'scene_fixture':dict()})
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:base)
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    monkeypatch.setattr(tracker,'accepted_native_pair',lambda *args:False)
    result,actual,journal=repair.prune_unused_authoring_fixtures(row,tmp_path/'evidence',tmp_path/'out')
    assert journal['removed_entities']==['orphan'] and journal['original_scene_occupancy_retained']
    assert actual['objects']==[o for o in spec['objects'] if o['name']!='orphan']
    assert actual['placements']==[p for p in spec['placements'] if p['entity']!='orphan']
    assert actual['goal']==spec['goal'] and actual['initial_states']==spec['initial_states']
    assert source.read_bytes()==raw and result['distractor_count']==0
    assert 'scene_fixture' in base.entities and result['restoration_entities']==['used']
    assert repair.prune_unused_authoring_fixtures(result,tmp_path/'evidence',tmp_path/'repeat') is None
    monkeypatch.setattr(tracker,'accepted_native_pair',lambda *args:True)
    assert repair.prune_unused_authoring_fixtures(row,tmp_path/'evidence',tmp_path/'accepted') is None


def test_thin_drawer_repair_adds_real_rigid_support_only_for_empty_native_domains(tmp_path,monkeypatch):
    import hashlib,json,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    spec=dict(id='task',instruction='Find the comb in the vanity drawers.',goal=[],target={'entity':'comb'},robot={},anchors={},
        scene=dict(model='house'),objects=[dict(name='comb',category='comb',model='native')],
        placements=[dict(entity='comb',relation='inside',receptacle='drawer')])
    source=tmp_path/'scenario.yaml';source.write_text(yaml.safe_dump(spec));evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        error='EXPERT_LAYOUT_UNRESOLVED: infeasible_model No allowed bin/orientation fits an item')))
    problem=dict(receptacle='drawer',items=dict(comb=[.01,.05,.002]),fixed=dict(comb=[]),volumes=dict(bin=[.3,.5,.2]))
    def save():
        (evidence/'native_validation.log').write_text('[expert-layout-problem] '+json.dumps(problem)+'\n')
    save();metadata=tmp_path/'assets/objects/tray/real/misc/metadata.json';metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(dict(bbox_size=[.15,.2,.012])))
    monkeypatch.setattr(repair.g,'ASSETS',tmp_path/'assets')
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:NS(entities={}))
    monkeypatch.setattr(repair.storage,'replay_layout',lambda layout,*args:layout)
    monkeypatch.setattr(repair,'include_task_receptacles',lambda layout,*args,**kwargs:layout)
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(comb=[.01,.05,.002]))
    monkeypatch.setattr(repair,'preflight_recipe',lambda *args:dict(status='heuristic_fit'))
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    row=dict(id='task',scenario=str(source),distractor_count=0)
    result,journal=repair.repair_thin_drawer_contents_with_native_tray(row,evidence,tmp_path/'out')
    actual=yaml.safe_load(Path(result['scenario']).read_text());assert_preserved(spec,actual)
    assert actual['objects'][0]==spec['objects'][0] and actual['placements'][0]==spec['placements'][0]
    tray=actual['objects'][-1];assert tray['category']=='tray' and tray['model']=='real' and tray['fixed_base'] is False
    assert actual['placements'][-1]['receptacle']=='drawer' and journal['physical_validation_required']
    assert repair.repair_thin_drawer_contents_with_native_tray(dict(row,drawer_liner_repairs=[dict(receptacle='drawer')]),evidence,tmp_path/'repeat') is None
    problem['fixed']['comb']=['bin'];save()
    assert repair.repair_thin_drawer_contents_with_native_tray(row,evidence,tmp_path/'not-needed') is None
    # An explicitly selected compartment can use a real liner, provided that
    # the support is bound to that SAME compartment and the original stays.
    problem['fixed']['comb']=[];save()
    spec['placements'][0]['link']='bin';source.write_text(yaml.safe_dump(spec))
    (evidence/'validation.json').write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        error='EXPERT_LAYOUT_UNRESOLVED: infeasible_model No allowed bin/orientation fits an item')))
    result,journal=repair.repair_thin_drawer_contents_with_native_tray(row,evidence,tmp_path/'same-drawer')
    actual=yaml.safe_load(Path(result['scenario']).read_text())
    assert actual['placements'][0]==spec['placements'][0]
    assert actual['placements'][-1]['link']=='bin' and journal['edit']['original_compartment_link']=='bin'
    problem['items']['comb']=[.25,.25,.002];save()
    result,journal=repair.repair_thin_drawer_contents_with_native_tray(row,evidence,tmp_path/'overhang')
    assert result is not None and 'Overhang is a proposal only' in journal['edit']['support_policy']
    assert journal['physical_validation_required']


def test_compiler_link_release_retains_named_locations_and_single_drawer():
    from expert_storage_repair import clear_unstated_links
    spec=dict(instruction='Find salt in the kitchen cupboards.',placements=[dict(entity='salt',relation='inside',receptacle='cabinet',link='drawer0')])
    assert clear_unstated_links(spec)==[dict(entity='salt',previous_link='drawer0')]
    assert 'link' not in spec['placements'][0]
    for instruction in ['Find salt in the upper cupboard.','Find a comb in the drawer.','Find salt in the left cabinet.']:
        spec=dict(instruction=instruction,placements=[dict(entity='target',relation='inside',receptacle='cabinet',link='drawer0')])
        assert not clear_unstated_links(spec)
        assert spec['placements'][0]['link']=='drawer0'


def test_collective_supplies_allow_multiple_sources_without_changing_named_drawers():
    from expert_storage_repair import source_allows_multiple_fixtures
    assert source_allows_multiple_fixtures('Find the matching cream among the bedroom travel supplies and bring it to the counter.')
    assert source_allows_multiple_fixtures('Find a cookie box on the grocery shelves.')
    assert source_allows_multiple_fixtures('Find salt in the kitchen cupboards.')
    assert not source_allows_multiple_fixtures('Find salt in the kitchen cupboard.')
    assert not source_allows_multiple_fixtures('Find the comb in the upper drawer.')


def test_destination_and_reference_top_do_not_pin_hidden_source_drawers():
    from expert_storage_repair import clear_unstated_links,repair_kind
    text='There is a sample on the cabinet top in front of you. Find its matching cookie box among the kitchen groceries and bring it to the upper cabinet.'
    spec=dict(instruction=text,placements=[dict(entity='cookie',relation='inside',receptacle='source',link='drawer0')])
    assert clear_unstated_links(spec) and 'link' not in spec['placements'][0]
    assert repair_kind({'category':'bottom_cabinet'},[],'Find salt in the kitchen cupboard and put it in the dining-room drawer.')=='cabinet'
    for text in ['Find salt in the upper kitchen cupboard.','Find salt in the cabinet on the left.','Find salt in the second drawer and bring it to the table.']:
        spec=dict(instruction=text,placements=[dict(entity='salt',relation='inside',receptacle='source',link='drawer0')])
        assert not clear_unstated_links(spec) and spec['placements'][0]['link']=='drawer0'


def test_storage_repair_separates_contents_from_reference_on_same_counter(tmp_path,monkeypatch):
    import yaml,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as e
    spec=dict(instruction='Find the matching box in the cupboards and put it on this counter.',scene={'model':'house'},
        objects=[dict(name='target'),dict(name='sample')],placements=[
            dict(entity='target',relation='inside',receptacle='cabinet'),
            dict(entity='sample',relation='on_top',receptacle='cabinet')],
        goal=[dict(kind='on_top',entity='target',receptacle='cabinet')])
    recipe=tmp_path/'scenario.yaml';recipe.write_text(yaml.safe_dump(spec))
    row=dict(scenario=str(recipe),source_room='kitchen',destination_receptacle='cabinet')
    base=NS(entities={'cabinet':dict(category='bottom_cabinet',model='native',aabb=[[0.,0.,0.],[1.,1.,1.]],in_rooms=['kitchen'])})
    monkeypatch.setattr(e,'blocking_storage',lambda *args:'cabinet')
    monkeypatch.setattr(e,'layout_for',lambda *args:base);monkeypatch.setattr(e.storage,'replay_layout',lambda *args:base)
    monkeypatch.setattr(e,'dimensions',lambda *args:{'target':[.1]*3,'sample':[.1]*3})
    observed=[]
    def reuse(original,row,base,old,group,*args,**kwargs):
        observed.extend(group);return dict(row,repaired=True),dict(status='proposal')
    monkeypatch.setattr(e,'reuse_existing_storage',reuse)
    result,doc=e.curate(row,tmp_path,tmp_path/'out')
    assert result['repaired'] and [p['entity'] for p in observed]==['target']
    assert yaml.safe_load(recipe.read_text())==spec


def test_task_basket_uses_only_geometry_from_matching_native_snapshot(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace as NS
    import expert_storage_repair as e
    monkeypatch.setattr(e.g,'ROOT',tmp_path)
    evidence=tmp_path/'case';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(scenario_sha256='verified')))
    (evidence/'geometry.private.json').write_text(json.dumps(dict(basket=dict(volumes={'fillable':[[99.,99.,5.],[99.3,99.2,5.15]]}))))
    spec=dict(objects=[dict(name='basket',category='shopping_basket',model='native')],
              placements=[dict(entity='milk',relation='inside',receptacle='basket')])
    original=NS(entities={})
    result=e.include_task_receptacles(original,spec,evidence=evidence,source_sha='verified')
    assert not original.entities
    extent=result.entities['basket']['interiors'][0]['hi']
    assert extent==pytest.approx([.3,.2,.28])
    assert not e.include_task_receptacles(original,spec,evidence=evidence,source_sha='different').entities


def test_multi_source_repair_chains_private_partial_changes_before_native_proposal(tmp_path,monkeypatch):
    import yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as e
    spec=dict(id='task',instruction='Find salt in the cupboards.',goal=[],target={'entity':'salt'},robot={},anchors={},
        scene={'model':'house'},objects=[{'name':'salt'},{'name':'sugar'}],placements=[
            dict(entity='salt',relation='inside',receptacle='bad_a'),dict(entity='sugar',relation='inside',receptacle='bad_b')])
    source=tmp_path/'original.yaml';source.write_text(yaml.safe_dump(spec));original_bytes=source.read_bytes()
    row=dict(id='task',scenario=str(source));calls=[]
    def preflight(spec):
        findings=[dict(receptacle=p['receptacle'],status='rejected_candidate' if p['receptacle'].startswith('bad_') else 'heuristic_fit') for p in spec['placements']]
        return dict(status='rejected_candidate' if any(f['status']!='heuristic_fit' for f in findings) else 'heuristic_fit',findings=findings)
    monkeypatch.setattr(e,'layout_for',lambda scene:None)
    monkeypatch.setattr(e.storage,'replay_layout',lambda *args:NS(entities={}))
    monkeypatch.setattr(e,'dimensions',lambda spec:{})
    monkeypatch.setattr(e,'preflight_recipe',lambda spec,*args:preflight(spec))
    monkeypatch.setattr(e.g,'load_scenario',lambda path:None)
    def curate(row,evidence,out,*,variant,old_storage=None,allow_partial=False,**kwargs):
        if not allow_partial:return None,dict(reason='Two invalid groups')
        calls.append(old_storage);spec=yaml.safe_load(Path(row['scenario']).read_text())
        for p in spec['placements']:
            if p['receptacle']==old_storage:p['receptacle']=old_storage.replace('bad_','good_')
        path=Path(out)/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(spec))
        return dict(row,scenario=str(path),layout_preflight=preflight(spec)),dict(status='static_partial_not_native_ready')
    monkeypatch.setattr(e,'curate',curate)
    result,doc=e.expert_proposal(row,'evidence',tmp_path/'out',policy={'method':'correct_proposal'})
    assert calls==['bad_a','bad_b']
    assert result['layout_preflight']['status']=='heuristic_fit' and doc['physical_validation_required']
    assert source.read_bytes()==original_bytes
    e.assert_preserved(spec,yaml.safe_load(Path(result['scenario']).read_text()))


def test_first_real_rejection_controls_repair_group(tmp_path):
    spec={'placements':[dict(entity='a',receptacle='first'),dict(entity='b',receptacle='second')]}
    (tmp_path/'native_validation.log').write_text('[placement] end a placed=False elapsed_s=2\n[placement] end b placed=False elapsed_s=0\n')
    assert blocking_storage(spec,tmp_path)=='first'


def test_storage_profile_retains_fridge_and_drawer_semantics():
    assert repair_kind({'category':'fridge'},[], 'Find milk.')=='fridge'
    assert repair_kind({'category':'bottom_cabinet'},[], 'Find the peeler in the drawer.')=='drawer'
    assert repair_kind({'category':'bottom_cabinet'},[], 'Find salt in the cupboard.')=='cabinet'
    assert repair_kind({'category':'shelf'},[], 'Find a book.') is None


def test_unsettled_container_is_identified_before_any_placement_rejects(tmp_path):
    import json
    (tmp_path/'validation.json').write_text(json.dumps(dict(error='EXPERT_CONTAINER_JOINTS_NOT_SETTLED')))
    (tmp_path/'native_validation.log').write_text('[placement] attempt=1 start jar inside auxiliary link=None\n'
        '[container-settling] '+json.dumps(dict(entity='auxiliary',quiet_windows=0))+'\n')
    assert blocking_storage({'placements':[]},tmp_path)=='auxiliary'


def test_missing_support_diagnosis_retains_actual_blocking_storage(tmp_path):
    import json
    (tmp_path/'validation.json').write_text(json.dumps(dict(error='EXPERT_SUPPORT_UNRESOLVED: no measured floor')))
    (tmp_path/'native_validation.log').write_text('[placement] attempt=1 start jar inside cabinet link=drawer\n')
    assert blocking_storage({'placements':[]},tmp_path)=='cabinet'


def test_nonfinite_native_link_identifies_its_real_source_without_pose_reset(tmp_path):
    import json
    error='/World/scene_0/articulated__cabinet_b/link_0 orientation tensor([nan, nan, nan, nan]) is not a unit quaternion.'
    (tmp_path/'validation.json').write_text(json.dumps(dict(error=error)))
    (tmp_path/'native_validation.log').write_text('[placement] end a placed=True\n')
    spec=dict(placements=[dict(entity='a',receptacle='cabinet_a'),dict(entity='b',receptacle='cabinet_b')])
    assert blocking_storage(spec,tmp_path)=='cabinet_b'


def test_post_physics_relation_failure_identifies_source_without_placement_reject(tmp_path):
    import json
    spec={'placements':[dict(entity='jar',receptacle='cabinet'),dict(entity='sample',receptacle='table')]}
    (tmp_path/'validation.json').write_text(json.dumps(dict(error=
        "placement failed verification after physics settled: ['jar inside cabinet', 'sample on_top table']")))
    (tmp_path/'native_validation.log').write_text('[placement] end jar placed=True\n')
    assert blocking_storage(spec,tmp_path)=='cabinet'


def test_missing_native_open_state_identifies_affected_fixture(tmp_path):
    import json
    (tmp_path/'validation.json').write_text(json.dumps(dict(error='wardrobe has no Open state')))
    assert blocking_storage({'placements':[]},tmp_path)=='wardrobe'


def test_failed_closed_state_identifies_actual_neighbour_not_last_placement(tmp_path):
    import json
    (tmp_path/'validation.json').write_text(json.dumps(dict(error=
        "placement failed verification after physics settled: ['neighbour open=True (declared False)']")))
    (tmp_path/'native_validation.log').write_text('[placement] end target placed=True\n')
    assert blocking_storage({'placements':[]},tmp_path)=='neighbour'


def test_explicit_xy_refers_to_closed_bbox_even_when_open_drawers_extend_aabb(monkeypatch):
    import ast,numpy as np,sys,types
    from types import SimpleNamespace as NS
    path=Path(__file__).resolve().parents[2]/'src/rummagebench/sim/omnigibson/backend.py'
    method=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_place_at_xy')
    namespace={'np':np,'_arr':lambda x:x,'SimBackendError':ValueError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'xy-contract','exec'),namespace)
    state={'position':np.array([100.,100.,5.])};calls=[]
    def set_pose(position=None,orientation=None):
        if position is not None:state['position']=np.array(position)
    obj=NS(name='cabinet',fixed_base=True,root_link=NS(),
           aabb=([99.3,99.8,4.6],[100.5,100.2,5.6]),
           scaled_bbox_center_in_base_frame=[-.2,.1,0.],
           get_position_orientation=lambda:(state['position'],[0,0,0,1]),set_position_orientation=set_pose)
    fake=types.ModuleType('omnigibson.object_states');fake.Inside=object();fake.OnTop=object()
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',fake)
    backend=NS(receptacle_region=lambda name:NS(kind='top_surface',lo=[0,0,0],hi=[30,30,0]),
               settle=lambda n:calls.append(n),_verify_relation=lambda *args:True)
    assert namespace['_place_at_xy'](backend,obj,NS(name='floor'),[10.,20.])
    assert np.allclose(state['position'],[9.8,20.1,.4])
    assert np.allclose(state['position'][:2]-np.array(obj.scaled_bbox_center_in_base_frame[:2]),[10.,20.])
    assert calls==[20]
    # An articulated table's actual top can extend beyond the root frame.
    # Propose there, but still return the native predicate's rejection when
    # there is no real contact; whole-body bounds are never a certificate.
    backend.receptacle_region=lambda _:NS(kind='top_surface',lo=[0,0,0],hi=[.1,.1,.4])
    backend.entity_aabb=lambda _:([0,0,0],[1,1,.8])
    backend._verify_relation=lambda *args:False
    rec=NS(name='table',fixed_base=True,links={'root':object(),'top':object()})
    state['position']=np.array([100.,100.,5.]);calls.clear()
    assert not namespace['_place_at_xy'](backend,obj,rec,[.5,.5])
    assert np.allclose(state['position'],[.3,.6,1.2]) and calls==[20]
    with pytest.raises(ValueError,match='outside the top surface'):
        namespace['_place_at_xy'](backend,obj,rec,[1.5,.5])


def test_storage_packer_checks_complete_inventory_not_individual_fit():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from expert_storage_repair import compartment_assignment
    entity={'interiors':[dict(link='drawer',lo=[0.,0.,0.],hi=[.21,.21,.2])]}
    assert compartment_assignment(entity,{'a':[.15,.15,.1]},'cabinet') is not None
    assert compartment_assignment(entity,{'a':[.15,.15,.1],'b':[.15,.15,.1]},'cabinet') is None


def test_storage_authoring_spacing_fallback_retains_true_item_dimensions():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from expert_storage_repair import compartment_assignment
    cavity={'interiors':[dict(link='bin',lo=[0.,0.,0.],hi=[.12,.12,.12])]}
    _,proposal=compartment_assignment(cavity,{'part':[.1,.1,.1]},'cabinet')
    assert proposal['margin_m']==.005 and proposal['physical_validation_required']
    cavity['interiors'][0]['hi']=[.105,.105,.12]
    assert compartment_assignment(cavity,{'part':[.1,.1,.1]},'cabinet') is None


def test_known_proposal_bug_is_not_evidence_for_geometry_blacklisting():
    from expert_storage_repair import proposal_coordinate_bug
    source='src/rummagebench/sim/omnigibson/inside_volume.py'
    assert proposal_coordinate_bug({'runtime_sources':{source:'broken'}},{'broken'})
    assert not proposal_coordinate_bug({'runtime_sources':{source:'corrected'}},{'broken'})
    assert not proposal_coordinate_bug({}, {'broken'})


def test_accepted_layout_is_positive_evidence_and_never_blacklisted(tmp_path,monkeypatch):
    import expert_storage_repair as e
    import track_full_asset_queue as tracker
    learned=[];case=dict(id='task',record={'scenario':'unused'},evidence=str(tmp_path/'evidence'))
    monkeypatch.setattr(tracker,'accepted_native_pair',lambda *args:True)
    monkeypatch.setattr(e,'learn_verified_storage_sites',lambda row,evidence:learned.append(evidence))
    result=e.request(dict(action='signature',case=case,out=str(tmp_path/'out')))
    assert result['signature'] is None and learned==[case['evidence']]
    monkeypatch.setattr(tracker,'accepted_native_pair',lambda *args:False)
    monkeypatch.setattr(e,'proposal_coordinate_bug',lambda *args:True)
    e.request(dict(action='signature',case=case,out=str(tmp_path/'out')))
    assert learned==[case['evidence']]


def test_native_support_experience_filters_bins_only_for_matching_native_assets(tmp_path,monkeypatch):
    import json,torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import expert_storage_repair as e
    import audit_dense_layouts as audit
    entity=dict(category='bottom_cabinet_no_top',model='native',interiors=[
        dict(link='drawer0',lo=[0.,0.,0.],hi=[.3,.3,.2]),
        dict(link='drawer2',lo=[0.,0.,0.],hi=[.3,.3,.2])])
    profile=dict(metadata_sha256='verified',encrypted_asset_sha256='verified',components={
        'drawer0':dict(status='unknown'),
        'drawer2':dict(status='measured_plane',floor_minus_fillable_bottom_m=-.02)})
    path=tmp_path/'build/native_support_geometry/index.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(assets={'bottom_cabinet_no_top/native':profile})))
    monkeypatch.setattr(e.g,'ROOT',tmp_path)
    monkeypatch.setattr(audit,'asset_digest',lambda path:'verified')
    monkeypatch.setattr(e,'native_drawer_links',lambda *args:{'drawer0','drawer2'})
    result=e.compartment_assignment(entity,{'jar':[.1,.1,.08]},'drawer',native_size=True)
    assert result[0]=={'jar':'drawer2'}
    assert e.compartment_assignment(entity,{'thin_tool':[.1,.1,.01]},'drawer',native_size=True) is None
    assert e.compartment_assignment(entity,{'thin_tool':[.1,.1,.01]},'drawer') is not None
    # Hash mismatch cannot transfer a fixture's measurements to another asset.
    monkeypatch.setattr(audit,'asset_digest',lambda path:'changed')
    assert e.compartment_assignment(entity,{'thin_tool':[.1,.1,.01]},'drawer',native_size=True) is not None


def test_native_drawer_classification_uses_joint_ancestry_not_link_number():
    from expert_storage_repair import native_drawer_links
    assert 'meta__drawer2_fillable_0_0_link' in native_drawer_links('bottom_cabinet_no_top','ncgmqr')
    assert 'meta__link_0_fillable_0_0_link' in native_drawer_links('bottom_cabinet_no_top','ufhpbn')
    assert not native_drawer_links('bottom_cabinet','jrhgeu')


def test_single_native_drawer_can_use_all_its_disconnected_components():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from expert_storage_repair import compartment_assignment
    entity={'offline_interior_hints':[dict(link='drawer0',component=i,extent_m=[.2,.2,.2]) for i in [0,1]]}
    assignment,solution=compartment_assignment(entity,{'a':[.15,.15,.1],'b':[.15,.15,.1]},'drawer')
    assert assignment=={'a':'drawer0','b':'drawer0'}
    assert len(set(solution['assignment'].values()))==2


def test_native_cavity_profile_does_not_double_count_overlapping_components(tmp_path,monkeypatch):
    import json,torch
    import audit_dense_layouts as audit
    import expert_storage_repair as repair
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    entity=dict(category='cabinet',model='native',offline_interior_hints=[
        dict(link='fillable',component=i,extent_m=[.5,.5,.2]) for i in [0,1]])
    domains=dict(volumes={'a':[.15,.15,.2],'b':[.15,.15,.2]},native_links={'a':'fillable','b':'fillable'},
        supports={k:dict(status='measured_plane',floor_minus_fillable_bottom_m=0.,height_spread_m=0.) for k in ['a','b']},
        overlap_groups={'a':'same_physical_space','b':'same_physical_space'},center_halfspaces={},obstacles={})
    profile=dict(metadata_sha256='verified',encrypted_asset_sha256='verified',native_proposal_domains=domains)
    path=tmp_path/'build/native_support_geometry/index.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(assets={'cabinet/native':profile})))
    monkeypatch.setattr(repair.g,'ROOT',tmp_path)
    monkeypatch.setattr(audit,'asset_digest',lambda _: 'verified')
    items={'one':[.1,.1,.02],'two':[.1,.1,.02]}
    assert repair.compartment_assignment(entity,items,'cabinet') is not None
    assert repair.compartment_assignment(entity,items,'cabinet',native_size=True) is None
    monkeypatch.setattr(audit,'asset_digest',lambda _: 'different asset')
    assert repair.compartment_assignment(entity,items,'cabinet',native_size=True) is not None


def test_native_floor_inset_can_defer_thin_prop_to_existing_real_flat_support(tmp_path,monkeypatch):
    import json,torch
    import expert_storage_repair as repair
    import audit_dense_layouts as audit
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    domains=dict(volumes={'bin':[.3,.3,.2]},native_links={'bin':'fillable'},
        supports={'bin':dict(status='measured_plane',floor_minus_fillable_bottom_m=-.004,height_spread_m=0.)},
        overlap_groups={},center_halfspaces={},obstacles={})
    path=tmp_path/'build/native_support_geometry/index.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(assets={'cabinet/native':dict(metadata_sha256='verified',encrypted_asset_sha256='verified',native_proposal_domains=domains)})))
    monkeypatch.setattr(repair.g,'ROOT',tmp_path);monkeypatch.setattr(audit,'asset_digest',lambda _: 'verified')
    entity=dict(category='cabinet',model='native',interiors=[])
    items=dict(board=[.1,.1,.02],card=[.05,.05,.001])
    assert repair.compartment_assignment(entity,items,'cabinet',native_size=True) is None
    assignment,proposal=repair.compartment_assignment(entity,items,'cabinet',native_size=True,support_entities={'board'})
    assert assignment=={'board':'fillable','card':'fillable'}
    assert proposal['deferred_native_stacking']=={'card':'board'}
    assert set(proposal['poses'])=={'board'} and proposal['physical_validation_required']


def test_existing_storage_repair_preserves_inventory_and_avoids_new_furniture(tmp_path,monkeypatch):
    import numpy as np,yaml
    from types import SimpleNamespace as NS
    import expert_storage_repair as e
    original=dict(id='task',instruction='Find the salt in the cupboards.',goal=[dict(kind='on_top',entity='salt',receptacle='table')],
        target={'entity':'salt'},robot={'init_anchor':'start'},anchors={'start':{'position':[0.,0.,0.]}},
        objects=[{'name':'salt'},{'name':'sugar'}],initial_states={'old':{'open':False}},
        placements=[dict(entity=n,relation='inside',receptacle='old') for n in ['salt','sugar']])
    entity=lambda room='kitchen':dict(category='bottom_cabinet',model='native',in_rooms=[room],aabb=[[0.,0.,0.],[1.,1.,1.]])
    entities={'old':entity(),'occupied':entity(),'wrong_room':entity('bathroom'),'good':entity(),'table':entity()}
    original['placements'].append(dict(entity='other',relation='inside',receptacle='occupied'))
    row=dict(id='task',scenario='original.yaml',source_receptacle='old',destination_receptacle='table',destination_room='kitchen',restoration_entities=['sugar'])
    base=NS(entities=entities,free=np.ones((1,1),bool),labels=np.ones((1,1)),map=NS(to_map=lambda *xy:(0,0)),
        route=lambda *args:dict(start=[0.,0.,0.]))
    seen=[]
    monkeypatch.setattr(e.storage,'model_openable',lambda *args:True)
    def packed(entity,items,kind):
        seen.append(entity)
        return {'salt':'fillable','sugar':'fillable'},dict(status='feasible_model')
    monkeypatch.setattr(e,'compartment_assignment',packed)
    monkeypatch.setattr(e,'dimensions',lambda spec:{})
    monkeypatch.setattr(e,'preflight_recipe',lambda *args:dict(status='heuristic_fit'))
    monkeypatch.setattr(e.g,'load_scenario',lambda *args:None)
    result,doc=e.reuse_existing_storage(original,row,base,'old',original['placements'][:2],'kitchen','cabinet',{},tmp_path,'evidence')
    assert seen==[entities['good']]
    repaired=yaml.safe_load(Path(result['scenario']).read_text())
    e.assert_preserved(original,repaired)
    assert repaired['objects']==original['objects']
    assert result['source_receptacle']=='good' and doc['reused_native_storage']=='good'
    assert all('link' not in p for p in repaired['placements'][:2])
    assert len(doc['released_compiler_links'])==2
    assert repaired['placements'][-1]==original['placements'][-1]
    original['goal'].append(dict(kind='closed',entity='old'))
    assert e.reuse_existing_storage(original,row,base,'old',original['placements'][:2],'kitchen','cabinet',{},tmp_path,'evidence') is None
    original['goal'].pop();original['objects'].append(dict(name='other'))
    # When empty fixtures are unavailable, include every existing object in
    # the capacity witness; a group-only fit must never certify an occupied bin.
    base.entities={n:entities[n] for n in ('old','occupied','table')}
    monkeypatch.setattr(e,'dimensions',lambda _:{'other':[.3,.3,.2]})
    sizes=dict(salt=[.1,.1,.1],sugar=[.1,.1,.1]);packed_items=[]
    def joint_packed(entity,items,kind):
        packed_items.append(items)
        assert items==dict(sizes,other=[.3,.3,.2])
        return {n:'fillable' for n in items},dict(status='feasible_model')
    monkeypatch.setattr(e,'compartment_assignment',joint_packed)
    result,doc=e.reuse_existing_storage(original,row,base,'old',original['placements'][:2],'kitchen','cabinet',sizes,tmp_path/'joint','evidence')
    repaired=yaml.safe_load(Path(result['scenario']).read_text())
    assert doc['existing_contents_retained']==['other'] and len(packed_items)==1
    assert all(p['receptacle']=='occupied' for p in repaired['placements'])
    assert repaired['objects']==original['objects']
    monkeypatch.setattr(e,'compartment_assignment',lambda *args,**kwargs:None)
    assert e.reuse_existing_storage(original,row,base,'old',original['placements'][:2],'kitchen','cabinet',sizes,tmp_path/'full','evidence') is None


def test_furniture_floor_domain_includes_wall_sites_and_low_raised_tiles():
    import numpy as np
    from types import SimpleNamespace as NS
    from constructed_storage import storage_floor_domain, screen_furniture_footprints
    layout=NS(map=NS(names={1:'kitchen',2:'bathroom'},room_ins=np.array([[1,1,2],[1,1,2]])),
              points=np.array([[1,1]]), entities={
                  'tile':dict(category='floors',aabb=[[-2,-2,0],[2,2,.12]]),
                  'upper_floor':dict(category='floors',aabb=[[-2,-2,2],[2,2,2.12]])},
              doc={'footprint_boxes':[[.1,-1,.4,1]]})
    floors,points=storage_floor_domain(layout,'kitchen')
    assert [name for name,_ in floors]==['tile']
    assert points.tolist()==[[0,0],[0,1],[1,0],[1,1]]
    assert storage_floor_domain(layout,'bathroom')[1].tolist()==[[0,2],[1,2]]
    # Admitting a semantic floor cell never bypasses actual collision screening.
    assert screen_furniture_footprints(layout,[[0,0],[-1,0]],[[-.2,-.2],[.2,.2]]).tolist()==[False,True]


def test_constructed_cabinet_site_reserves_open_drawer_over_floor_edge():
    from constructed_storage import footprint_fits_floor
    # F082: the closed cabinet fits but the opened drawer crosses x=-1.05.
    floor={'aabb':[[-8.63,-3.02,-.3],[-1.05,4.235,0.]]}
    size=[.624,.426,.696]
    assert footprint_fits_floor([-1.55,-.05],size,floor)
    assert not footprint_fits_floor([-1.55,-.05],size,floor,max(size[:2]))
    assert footprint_fits_floor([-3.,-.05],size,floor,max(size[:2]))


@pytest.mark.parametrize('surface_category',['countertop','bottom_cabinet'])
def test_failed_surface_repacking_preserves_body_and_avoids_raised_geometry(tmp_path,monkeypatch,surface_category):
    import hashlib,json,yaml,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as repair
    original=dict(id='task',instruction='Find the jar on the kitchen counter.',
        goal=[dict(kind='on_top',entity='jar',receptacle='table')],target=dict(entity='jar'),
        objects=[dict(name='jar',category='jar',model='native'),dict(name='sample',category='jar',model='native')],
        placements=[dict(entity='jar',relation='on_top',receptacle='counter',position_xy=[.1,.1]),
                    dict(entity='sample',relation='on_top',receptacle='table',position_xy=[2.,2.])])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original));evidence=tmp_path/'evidence';evidence.mkdir()
    report=dict(scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),status='failed',
                error="placement failed verification after physics settled: ['jar on_top counter']")
    (evidence/'validation.json').write_text(json.dumps(report))
    layout=NS(entities=dict(counter=dict(category=surface_category,aabb=[[0.,0.,.9],[1.,1.,1.]]),
                            stove=dict(aabb=[[0.,0.,1.],[.5,.5,1.2]]),
                            table=dict(category='table',aabb=[[1.5,1.5,.9],[2.5,2.5,1.]])))
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:layout)
    monkeypatch.setattr(repair.storage,'replay_layout',lambda l,f:l)
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(jar=[.1]*3,sample=[.1]*3))
    monkeypatch.setattr(repair,'preflight_recipe',lambda *args:dict(status='heuristic_fit'))
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    result,journal=repair.repair_failed_surface_poses(dict(id='task',scenario=str(source)),evidence,tmp_path/'out')
    revised=yaml.safe_load(Path(result['scenario']).read_text());xy=revised['placements'][0]['position_xy']
    assert xy!=[.1,.1] and (xy[0]-.05>=.5 or xy[1]-.05>=.5)
    assert revised['placements'][1]==original['placements'][1]
    assert revised['placements'][0]['receptacle']=='counter'
    assert_preserved(original,revised)
    assert journal['plans'][0]['measured_scene_obstacles'] and journal['physical_validation_required']
    source.write_text(source.read_text()+'\n# different recipe\n')
    assert repair.repair_failed_surface_poses(dict(id='task',scenario=str(source)),evidence,tmp_path/'stale') is None


def test_fixture_envelope_requires_matching_native_recipe_and_keeps_direction(tmp_path):
    import hashlib,json,yaml
    from expert_storage_repair import observed_fixture_envelope
    recipe=tmp_path/'scenario.yaml';recipe.write_text(yaml.safe_dump(dict(
        placements=[dict(entity='cabinet',position_xy=[-1.55,-.05])])) )
    (tmp_path/'validation.json').write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())))
    (tmp_path/'native_validation.log').write_text('[expert-pose-result] '+json.dumps(dict(entity='cabinet',
        orientation=[0.,0.,0.,1.],aabb=[[-1.862,-.263,0.],[-.724,.163,.696]]))+'\n')
    result=observed_fixture_envelope({'scenario':str(recipe)},tmp_path,'cabinet')
    assert result[0]==pytest.approx([-.332,-.233])
    assert result[1]==pytest.approx([.846,.233])
    recipe.write_text(recipe.read_text()+'changed: true\n')
    assert observed_fixture_envelope({'scenario':str(recipe)},tmp_path,'cabinet') is None


def test_measured_support_patches_replace_optimistic_full_cavity(tmp_path,monkeypatch):
    import json,torch
    import expert_storage_repair as e
    import audit_dense_layouts as audit
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    entity=dict(category='cabinet',model='native',interiors=[dict(link='fillable',lo=[0.,0.,0.],hi=[.5,.5,.4])])
    profile=dict(metadata_sha256='verified',encrypted_asset_sha256='verified',components={
        'fillable':dict(status='unknown',patches=[dict(extent_m=[.12,.12,.2],floor_minus_fillable_bottom_m=0.)])})
    path=tmp_path/'build/native_support_geometry/index.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(assets={'cabinet/native':profile})))
    monkeypatch.setattr(e.g,'ROOT',tmp_path);monkeypatch.setattr(audit,'asset_digest',lambda path:'verified')
    assert e.compartment_assignment(entity,{'jar':[.2,.2,.1]},'cabinet') is not None
    assert e.compartment_assignment(entity,{'jar':[.2,.2,.1]},'cabinet',native_size=True,allow_patch_support=True) is None
    assignment,solution=e.compartment_assignment(entity,{'jar':[.08,.08,.1]},'cabinet',native_size=True,allow_patch_support=True)
    assert assignment=={'jar':'fillable'}
    assert solution['assignment']['jar']=='fillable#support_patch:0'


def test_native_support_learning_is_hash_bound_and_rejects_scene_scaled_assets(tmp_path,monkeypatch):
    import json,yaml,hashlib
    import expert_storage_repair as e
    import audit_dense_layouts as audit
    monkeypatch.setattr(e.g,'ROOT',tmp_path);monkeypatch.setattr(audit,'asset_digest',lambda path:'verified')
    recipe=tmp_path/'scenario.yaml'
    spec=dict(objects=[dict(name='cabinet',category='cabinet',model='native',fixed_base=True)],
              placements=[dict(entity='cabinet',relation='on_top',position_xy=[0.,0.])])
    recipe.write_text(yaml.safe_dump(spec));row={'scenario':str(recipe)}
    report=tmp_path/'validation.json'
    report.write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())))
    log=tmp_path/'native_validation.log'
    log.write_text('[expert-pose-result] '+json.dumps(dict(entity='cabinet',placed=True,orientation=[0,0,0,1]))+'\n'
        +'[expert-support-plane] '+json.dumps(dict(receptacle='cabinet',component='fillable',status='measured_plane',floor_minus_fillable_bottom_m=-.001))+'\n')
    e.learn_native_support_profiles(row,tmp_path)
    index=tmp_path/'build/native_support_geometry/index.json'
    assert json.loads(index.read_text())['assets']['cabinet/native']['components']['fillable']['floor_minus_fillable_bottom_m']==-.001
    index.unlink();spec['objects'][0]['scale']=2.;recipe.write_text(yaml.safe_dump(spec))
    report.write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())))
    e.learn_native_support_profiles(row,tmp_path)
    assert not index.exists()


def test_registered_support_rebinding_preserves_every_task_item_and_rejects_target_change(tmp_path,monkeypatch):
    import json,yaml,hashlib,copy
    from types import SimpleNamespace as NS
    import expert_storage_repair as e
    monkeypatch.setattr(e.g,'ROOT',tmp_path)
    original=dict(id='task',instruction='Find the milk.',goal=[dict(kind='on_top',entity='milk',receptacle='table')],
        target={'entity':'milk'},robot={},anchors={},scene={'model':'house'},
        objects=[dict(name='basket',category='shopping_basket',model='small'),dict(name='milk',category='milk',model='native'),dict(name='juice',category='juice',model='native')],
        placements=[dict(entity='basket',relation='on_top',receptacle='table'),dict(entity='milk',relation='inside',receptacle='basket'),dict(entity='juice',relation='inside',receptacle='basket')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(original));before=source.read_bytes()
    row=dict(id='task',parent_task_id='Ftest',scenario=str(source),target_entities=['milk'])
    decision=dict(instruction_sha256=hashlib.sha256(original['instruction'].encode()).hexdigest(),entity='basket',
        original_object=original['objects'][0],replacement=dict(original['objects'][0],model='large'))
    config=tmp_path/'build/expert_support_overrides/index.json';config.parent.mkdir(parents=True)
    config.write_text(json.dumps(dict(tasks={'Ftest':decision})))
    monkeypatch.setattr(e,'layout_for',lambda *args:None)
    monkeypatch.setattr(e.storage,'replay_layout',lambda *args:NS(entities={}))
    monkeypatch.setattr(e,'include_task_receptacles',lambda layout,*args:layout)
    monkeypatch.setattr(e,'dimensions',lambda spec:{})
    monkeypatch.setattr(e,'preflight_recipe',lambda *args:dict(status='heuristic_fit'))
    monkeypatch.setattr(e.g,'load_scenario',lambda *args:None)
    result,doc=e.apply_registered_support_rebinding(row,tmp_path,tmp_path/'out')
    changed=yaml.safe_load(Path(result['scenario']).read_text())
    assert changed['objects'][1:]==original['objects'][1:] and changed['placements']==original['placements']
    assert changed['instruction']==original['instruction'] and changed['goal']==original['goal']
    assert doc['physical_validation_required'] and source.read_bytes()==before
    decision.update(entity='milk',original_object=original['objects'][1],replacement=dict(original['objects'][1],model='other'))
    config.write_text(json.dumps(dict(tasks={'Ftest':decision})))
    with pytest.raises(ValueError,match='CANNOT_CHANGE_TARGET'):e.apply_registered_support_rebinding(row,tmp_path,tmp_path/'target_change')


def test_hidden_compiler_drawer_link_does_not_restrict_generic_storage_semantics():
    from expert_storage_repair import repair_kind,candidate_models
    placement=[dict(entity='mirror',link='meta__drawer0_fillable_0_0_link')]
    assert repair_kind({'category':'nightstand'},placement,'Find the makeup mirror among the bedroom dresser accessories.')=='cabinet'
    assert repair_kind({'category':'nightstand'},placement,'Find the mirror in the vanity drawers.')=='drawer'
    assert any(c=='wardrobe' for c,m in candidate_models('cabinet','bedroom_0'))
    assert not any(c=='wardrobe' for c,m in candidate_models('cabinet','kitchen_0'))


def test_failed_auxiliary_fixture_can_change_but_task_objects_cannot():
    from expert_storage_repair import assert_task_inventory_preserved
    original=dict(id='task',instruction='Find salt.',goal=[],target={'entity':'salt'},robot={},anchors={},
        objects=[dict(name='salt',model='native'),dict(name='sugar',model='native'),dict(name='expert_storage_1',category='bottom_cabinet',model='bad',fixed_base=True)],
        placements=[dict(entity='salt',relation='inside',receptacle='expert_storage_1'),dict(entity='sugar',relation='inside',receptacle='expert_storage_1'),dict(entity='expert_storage_1',relation='on_top',receptacle='floor')])
    changed=copy.deepcopy(original);changed['objects'][2]['model']='fitting_native'
    assert_task_inventory_preserved(original,changed)
    changed['objects'][1]['model']='smaller_sugar'
    with pytest.raises(ValueError,match='Original task asset changed: sugar'):assert_task_inventory_preserved(original,changed)


def test_authored_source_replacement_uses_provenance_and_protects_reference_and_goal():
    from expert_storage_repair import editable_authored_storage,assert_task_inventory_preserved
    original=dict(id='task',instruction='Find salt.',goal=[],target={'entity':'salt'},robot={},anchors={},
        objects=[dict(name='salt',model='native'),dict(name='search_storage',category='nightstand',model='unopenable',fixed_base=True)],
        placements=[dict(entity='salt',relation='inside',receptacle='search_storage'),dict(entity='search_storage',relation='on_top',receptacle='floor')])
    row=dict(target_entities=['salt'],inserted_furniture=[{'name':'search_storage'}])
    assert editable_authored_storage(original,row,'search_storage')
    changed=copy.deepcopy(original);changed['objects'][1].update(category='bottom_cabinet',model='openable')
    assert_task_inventory_preserved(original,changed,editable_fixture_names={'search_storage'})
    assert not editable_authored_storage(original,dict(row,inserted_furniture=[]),'search_storage')
    original['goal']=[dict(kind='on_top',entity='salt',receptacle='search_storage')]
    assert not editable_authored_storage(original,row,'search_storage')
    original['goal']=[];original['placements'].append(dict(entity='sample',relation='on_top',receptacle='search_storage'))
    assert not editable_authored_storage(original,row,'search_storage')


def test_task_container_uses_its_real_surveyed_parent_as_authoring_anchor():
    from types import SimpleNamespace as NS
    from expert_storage_repair import declared_source_anchor,repair_kind
    spec=dict(placements=[dict(entity='fruit',receptacle='basket'),dict(entity='basket',receptacle='table')])
    layout=NS(entities={'table':{'aabb':[[1,2,0],[2,3,1]]}})
    assert declared_source_anchor(spec,layout,'basket')=='table'
    assert set(layout.entities)=={'table'}  # no fabricated basket world geometry
    assert repair_kind({'category':'wicker_basket'},[],'Find a lemon in the kitchen cupboards.')=='cabinet'
    assert declared_source_anchor({'placements':[dict(entity='basket',receptacle='basket')]},layout,'basket') is None


def test_scene_translation_updates_real_geometry_and_never_frees_unknown_occupancy(tmp_path,monkeypatch):
    import json,hashlib,numpy as np
    from types import SimpleNamespace as NS
    import expert_storage_repair as e
    survey=tmp_path/'survey.json';survey.write_text('a hash-bound native survey')
    free=np.ones((30,30),bool);free[1,1]=False
    base=NS(path=survey,doc={'footprint_boxes':[]},entities={'cabinet':dict(aabb=[[-.2,-.2,0],[.2,.2,1]],
        interiors=[dict(lo=[-.1,-.1,.1],hi=[.1,.1,.8])])},free=free,radius=.1,
        map=NS(size=30,to_map=lambda x,y:(int(y*10+15),int(x*10+15))))
    monkeypatch.setattr(e,'layout_for',lambda scene:base)
    manifest=tmp_path/'clearance.json';manifest.write_text(json.dumps(dict(
        expert_base_survey_sha256=hashlib.sha256(survey.read_bytes()).hexdigest(),
        expert_scene_translations=[dict(entity='cabinet',delta=[.4,0,0])])))
    spec=dict(scene=dict(model='house',clearance_manifest=str(manifest)))
    changed=e.layout_for_spec(spec)
    assert changed.entities['cabinet']['aabb'][0]==pytest.approx([.2,-.2,0])
    assert changed.entities['cabinet']['interiors'][0]['lo']==pytest.approx([.3,-.1,.1])
    assert base.entities['cabinet']['aabb'][0]==[-.2,-.2,0]
    assert not changed.free[1,1] and np.all(changed.free<=base.free)
    survey.write_text('another survey')
    with pytest.raises(ValueError,match='SURVEY_CHANGED'):e.layout_for_spec(spec)


def test_failed_auxiliary_floor_site_can_try_another_native_fixture(tmp_path,monkeypatch):
    import yaml
    import expert_storage_repair as e
    original=dict(id='task',instruction='Find salt.',scene={'model':'house'},target={'entity':'salt'},goal=[],
        objects=[dict(name='salt'),dict(name='expert_storage_1',category='bottom_cabinet',model='large',fixed_base=True)],
        placements=[dict(entity='salt',relation='inside',receptacle='expert_storage_1')])
    path=tmp_path/'scenario.yaml';path.write_text(yaml.safe_dump(original))
    row=dict(scenario=str(path),target_entities=['salt'],inserted_furniture=[dict(name='expert_storage_1')])
    monkeypatch.setattr(e,'relocate_failed_fixture',lambda *args:(None,dict(entity='expert_storage_1',reason='no site for this large model')))
    calls=[]
    def curate(*args,**kwargs):
        calls.append(kwargs);return dict(row,another_fixture=True),dict(status='proposal')
    monkeypatch.setattr(e,'curate',curate)
    result,doc=e.expert_proposal(row,tmp_path,tmp_path/'out',policy={'method':'relocate_storage'})
    assert result['another_fixture'] and calls[0]['old_storage']=='expert_storage_1'
    assert doc['prior_fixture_relocation']['entity']=='expert_storage_1'


def test_plural_drawers_can_use_multiple_native_drawers_but_singular_cannot():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from expert_storage_repair import compartment_assignment
    entity={'offline_interior_hints':[dict(link='drawer'+str(i),extent_m=[.2,.2,.2]) for i in [0,1]]}
    items={'a':[.15,.15,.1],'b':[.15,.15,.1]}
    assert compartment_assignment(entity,items,'drawer') is None
    assigned,_=compartment_assignment(entity,items,'drawer',single_drawer=False)
    assert len(set(assigned.values()))==2


def test_task_local_storage_experience_requires_geometry_rejection_not_timeout(tmp_path):
    import json,yaml,hashlib
    from expert_storage_repair import observed_storage_model_rejection,storage_inventory_signature
    spec=dict(objects=[dict(name='a',category='card',model='native')],placements=[dict(entity='a',relation='inside',receptacle='cabinet')])
    recipe=tmp_path/'scenario.yaml';recipe.write_text(yaml.safe_dump(spec));row={'scenario':str(recipe)}
    report=dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest());(tmp_path/'validation.json').write_text(json.dumps(report))
    event=dict(receptacle='cabinet',status='infeasible_model',reason='No allowed bin/orientation fits an item')
    log=tmp_path/'native_validation.log';log.write_text('[expert-layout-plan] '+json.dumps(event)+'\n')
    rejected=observed_storage_model_rejection(row,tmp_path,spec,'cabinet',spec['placements'],dict(category='cabinet',model='native'))
    assert rejected['inventory_sha256']==storage_inventory_signature(spec,spec['placements'])
    assert rejected['asset']=='cabinet/native'
    event['status']='unknown';log.write_text('[expert-layout-plan] '+json.dumps(event)+'\n')
    assert observed_storage_model_rejection(row,tmp_path,spec,'cabinet',spec['placements'],dict(category='cabinet',model='native')) is None
    event['status']='infeasible_model';log.write_text('[expert-layout-plan] '+json.dumps(event)+'\n')
    spec['placements'][0]['link']='compiler_drawer'
    assert observed_storage_model_rejection(row,tmp_path,spec,'cabinet',spec['placements'],dict(category='cabinet',model='native')) is None


def test_editorial_rewrite_requires_exact_authorized_source_and_archives_changes(tmp_path,monkeypatch):
    import json,yaml,hashlib
    import expert_storage_repair as e
    monkeypatch.setattr(e.g,'ROOT',tmp_path);monkeypatch.setattr(e.g,'ASSETS',tmp_path/'assets')
    monkeypatch.setattr(e.g,'load_scenario',lambda *args:None)
    before=dict(id='task',instruction='Find the matching mirror.',scene={'model':'house'},
        robot={'init_anchor':'start'},anchors={'start':{'position':[0,0,0]}},target={'entity':'target','category':'mirror'},
        goal=[dict(kind='on_top',entity='target',receptacle='table')],
        objects=[dict(name=n,category='mirror',model='oversize') for n in ('target','comparison_sample')]+
                [dict(name='negative',category='comb',model='other')],
        placements=[dict(entity=n,relation='inside',receptacle='drawer') for n in ('target','negative')]+
                   [dict(entity='comparison_sample',relation='on_top',receptacle='table')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(before));raw=source.read_bytes()
    changes=[dict(before=o,after=dict(o,category='comb',model='small')) for o in before['objects'][:2]]
    asset=tmp_path/'assets/objects/comb/small';(asset/'misc').mkdir(parents=True);(asset/'usd').mkdir()
    (asset/'misc/metadata.json').write_text(json.dumps(dict(bbox_size=[.10,.03,.02])))
    (asset/'usd/small.encrypted.usd').write_bytes(b'fixture')
    decision=dict(revision_id='revision',authorization='user_approved_find_and_deliver_rewrite',
        original_scenario_sha256=hashlib.sha256(raw).hexdigest(),before_instruction=before['instruction'],
        after_instruction='Find the matching comb.',object_changes=changes,reason='Oversized mirror in a drawer.')
    config=tmp_path/'build/expert_editorial_revisions/index.json';config.parent.mkdir(parents=True)
    def save():config.write_text(json.dumps(dict(tasks={'Ptest':decision})))
    save();row=dict(id='task',parent_task_id='Ptest',scenario=str(source),target_entities=['target'],binding_sha256='old')
    result,journal=e.apply_registered_editorial_revision(row,tmp_path/'out')
    after=yaml.safe_load(Path(result['scenario']).read_text())
    assert source.read_bytes()==raw==Path(journal['original_archive']).read_bytes()
    for key in ('goal','placements','robot','anchors','scene'):assert before[key]==after[key]
    assert after['target']=={'entity':'target','category':'comb'}
    assert after['objects'][2]==before['objects'][2] and len(after['objects'])==len(before['objects'])
    assert result['target_key']=='comb/small' and result['previous_binding_sha256']=='old'
    assert journal['physical_validation_required'] and not journal['human_completion_verified']
    with pytest.raises(ValueError,match='Semantic field changed'):e.assert_preserved(before,after)
    decision['original_scenario_sha256']='wrong';save()
    assert e.apply_registered_editorial_revision(row,tmp_path/'wrong') is None
    decision['original_scenario_sha256']=hashlib.sha256(raw).hexdigest();decision['authorization']='';save()
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):e.apply_registered_editorial_revision(row,tmp_path/'unauthorized')
    decision['authorization']='user_approved_find_and_deliver_rewrite';decision['object_changes'][1]['after']['model']='other';save()
    other=tmp_path/'assets/objects/comb/other';(other/'misc').mkdir(parents=True);(other/'usd').mkdir()
    (other/'misc/metadata.json').write_text(json.dumps(dict(bbox_size=[.10,.03,.02])))
    (other/'usd/other.encrypted.usd').write_bytes(b'fixture')
    with pytest.raises(ValueError,match='SAMPLE_TARGET_MISMATCH'):e.apply_registered_editorial_revision(row,tmp_path/'mismatch')


def test_surface_binding_follows_measured_slab_and_refuses_ambiguous_aliases():
    from types import SimpleNamespace as NS
    import generate_dense_pilot as g
    cabinet=dict(category='bottom_cabinet_no_top',in_rooms=['bathroom_0'],aabb=[[0,0,0],[1,.6,.8]])
    slab=dict(category='countertop',in_rooms=['bathroom_0'],aabb=[[-.01,-.01,.803],[1.01,.61,.825]])
    entities=dict(cabinet=cabinet,slab=slab)
    assert g.exposed_surface_entity(entities,'cabinet')=='slab'
    assert g.surfaces(NS(entities=entities),'bathroom_0')==['slab']
    # The room, height, extent and unique covering-body checks all matter.
    for changes in [dict(in_rooms=['bedroom_0']),dict(aabb=[[0,0,1.],[1,.6,1.02]]),dict(aabb=[[0,0,.803],[.4,.6,.825]])]:
        assert g.exposed_surface_entity(dict(cabinet=cabinet,slab={**slab,**changes}),'cabinet')=='cabinet'
    assert g.exposed_surface_entity({**entities,'another_slab':slab},'cabinet')=='cabinet'


def test_runtime_candidate_format_error_does_not_blacklist_scene(tmp_path):
    import json
    from expert_storage_repair import request
    (tmp_path/'validation.json').write_text(json.dumps(dict(error='ValueError: INVALID_EXPERT_LAYOUT_CANDIDATE')))
    result=request(dict(action='signature',case=dict(record=dict(scenario=str(tmp_path/'not_needed.yaml')),evidence=str(tmp_path)),out=str(tmp_path/'out')))
    assert result['signature'] is None


def test_task_basket_route_uses_support_without_changing_delivery_goal():
    from expert_storage_repair import scene_route_anchor
    spec=dict(placements=[dict(entity='basket',receptacle='tray'),dict(entity='tray',receptacle='table')],
              goal=[dict(kind='inside',entity='target',receptacle='basket')])
    before=copy.deepcopy(spec)
    assert scene_route_anchor(spec,'basket',{'table':{}})=='table'
    assert spec==before
    with pytest.raises(ValueError,match='SUPPORT_NOT_IN_SCENE'):
        scene_route_anchor(spec,'unplaced',{'table':{}})
    spec['placements'].append(dict(entity='table',receptacle='basket'))
    with pytest.raises(ValueError,match='CYCLIC'):scene_route_anchor(spec,'basket',{})


def test_shelf_native_sampling_revision_preserves_task_and_receptacle(tmp_path,monkeypatch):
    import hashlib,json,yaml
    import expert_storage_repair as repair
    original=dict(id='task',instruction='Find the jam on the kitchen shelf.',
        target={'entity':'jam'},goal=[dict(kind='on_top',entity='jam',receptacle='table')],
        objects=[dict(name='jam',category='jar',model='one'),dict(name='honey',category='jar',model='two'),
                 dict(name='shelf',category='shelf',model='native',fixed_base=True)],
        placements=[dict(entity=n,relation='on_top',receptacle='shelf',position_xy=[.1,.2]) for n in ['jam','honey']])
    source=tmp_path/'scenario.yaml';source.write_text(yaml.safe_dump(original))
    evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
    (evidence/'native_validation.log').write_text('[placement] end honey placed=False elapsed_s=1\n')
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    row=dict(id='task',scenario=str(source))
    result,journal=repair.repair_explicit_shelf_sampling(row,evidence,tmp_path/'out')
    revised=yaml.safe_load(Path(result['scenario']).read_text())
    assert_preserved(original,revised)
    assert all(p['receptacle']=='shelf' and 'position_xy' not in p for p in revised['placements'])
    assert revised['objects']==original['objects'] and journal['physical_validation_required']
    # Rejected native sampling must fall through to a new physical support,
    # rather than returning to the same outer-box XY proposal indefinitely.
    next_evidence=tmp_path/'native_rejection';next_evidence.mkdir()
    (next_evidence/'validation.json').write_text(json.dumps(dict(
        scenario_sha256=hashlib.sha256(Path(result['scenario']).read_bytes()).hexdigest(),
        error="placement failed verification after physics settled: ['honey on_top shelf']")))
    monkeypatch.setattr(repair,'layout_for_spec',lambda _:pytest.fail('Repeated surface packing must be skipped'))
    assert repair.repair_failed_surface_poses(result,next_evidence,tmp_path/'repack_loop') is None
    # A substantive fixture/inventory/native-policy change still gets a new chance.
    key=repair.surface_attempt_key(revised,'shelf',{})
    changed=copy.deepcopy(revised);changed['objects'][0]['model']='different native model'
    assert repair.surface_attempt_key(changed,'shelf',{})!=key
    # The same rejected recipe without a matching physical report is not evidence.
    source.write_text(source.read_text()+'\n# changed\n')
    assert repair.repair_explicit_shelf_sampling(row,evidence,tmp_path/'other') is None


def test_measured_opening_envelope_is_hash_bound_and_screened_against_floor(tmp_path,monkeypatch):
    import hashlib,json,numpy as np
    import constructed_storage as storage
    monkeypatch.setattr(storage.g,'ROOT',tmp_path)
    monkeypatch.setattr(storage.g,'ASSETS',tmp_path/'assets')
    asset=tmp_path/'assets/objects/cabinet/native'
    metadata=asset/'misc/metadata.json';metadata.parent.mkdir(parents=True);metadata.write_text('metadata')
    usd=asset/'usd/native.encrypted.usd';usd.parent.mkdir();usd.write_text('mesh')
    profile=tmp_path/'build/native_support_geometry/index.json';profile.parent.mkdir(parents=True)
    bounds=[[-.3,-.3],[.8,.3]]
    profile.write_text(json.dumps(dict(assets={'cabinet/native':dict(opened_footprint_offsets_m=bounds,
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        encrypted_asset_sha256=hashlib.sha256(usd.read_bytes()).hexdigest())})))
    observed=storage.observed_opening_offsets('cabinet','native')
    assert np.allclose(observed,bounds)
    floor=dict(aabb=[[0,0,0],[2,2,0]])
    assert storage.footprint_fits_floor([1.5,1],[.6,.6,1],floor)
    assert not storage.footprint_fits_floor([1.5,1],[.6,.6,1],floor,footprint_offsets_m=observed)
    usd.write_text('different native mesh')
    assert storage.observed_opening_offsets('cabinet','native') is None


def test_registered_surface_rebinding_cannot_change_goal_predicate_or_target(tmp_path,monkeypatch):
    import json,yaml,hashlib
    import expert_storage_repair as repair
    monkeypatch.setattr(repair.g,'ROOT',tmp_path)
    monkeypatch.setattr(repair.g,'load_scenario',lambda _:None)
    before=dict(id='task',instruction='Find the jar and put it on the counter.',scene={'model':'house'},
        target={'entity':'target','category':'jar'},objects=[dict(name='target',category='jar',model='native')],
        goal=[dict(kind='on_top',entity='target',receptacle='cabinet')],
        placements=[dict(entity='target',relation='inside',receptacle='source')])
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(before))
    decision=dict(revision_id='surface',authorization='user_approved_find_and_deliver_rewrite',
        original_scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        before_instruction=before['instruction'],after_instruction=before['instruction'],object_changes=[],
        reason='Bind the real counter body.',surface_binding_changes=[dict(field='goal',before=before['goal'][0],
            after=dict(kind='on_top',entity='target',receptacle='countertop'))])
    config=tmp_path/'build/expert_editorial_revisions/index.json';config.parent.mkdir(parents=True)
    def save():config.write_text(json.dumps(dict(tasks={'task':decision})))
    save();row=dict(id='task',parent_task_id='task',scenario=str(source))
    revised,journal=repair.apply_registered_editorial_revision(row,tmp_path/'good')
    after=yaml.safe_load(Path(revised['scenario']).read_text())
    assert after['objects']==before['objects'] and after['placements']==before['placements']
    assert journal['before_goal']==before['goal'] and journal['after_goal']==after['goal']
    decision['surface_binding_changes'][0]['after']['kind']='inside';save()
    with pytest.raises(ValueError,match='RETAIN_ON_TOP_AND_ENTITY'):
        repair.apply_registered_editorial_revision(row,tmp_path/'bad_predicate')
    decision['surface_binding_changes'][0]['after'].update(kind='on_top',entity='other');save()
    with pytest.raises(ValueError,match='RETAIN_ON_TOP_AND_ENTITY'):
        repair.apply_registered_editorial_revision(row,tmp_path/'bad_target')


def test_reviewed_surface_spawn_preserves_room_and_a_unique_forward_table(monkeypatch):
    from types import SimpleNamespace as N
    import expert_storage_repair as repair
    import numpy as np
    grid=N(size=20,resolution=.1,room_ins=np.ones((20,20),int),to_map=lambda *_:(10,10))
    grid.room_ins[10,11]=2
    free=np.zeros((20,20),bool);free[10,9]=free[10,11]=True
    layout=N(map=grid,free=free,labels=np.ones((20,20),int),points=[(10,11),(10,9)],
        entities={name:dict(aabb=[[x-.1,-.1,.7],[x+.1,.1,.8]]) for name,x in [('source',-.5),('dest',.5)]},
        approaches=lambda *_:[(0.,(10,9))])
    monkeypatch.setattr(repair,'scene_route_anchor',lambda spec,name,entities:name)
    spec=dict(instruction='Bring the box to the table in front of you.',robot={'init_anchor':'start'},
        anchors={'start':dict(position=[0.,0.,.03],orientation=[0.,0.,2**-.5,2**-.5])})
    row=dict(source_room='storage_room_0',destination_room='storage_room_0',destination_receptacle='dest')
    result=repair.work_surface_spawn_proposal(spec,row,layout,'source')
    assert result and result['after']['position']==pytest.approx([-.1,0.,.03])
    assert result['after']['orientation']==pytest.approx([0.,0.,0.,1.])
    assert result['before']==spec['anchors']['start'] and result['same_initial_room']
    assert result['physical_validation_required']
    # A closer candidate in another room must never replace the original room.
    free[10,9]=False
    assert repair.work_surface_spawn_proposal(spec,row,layout,'source') is None


def test_work_surface_capacity_filter_runs_before_pose_search(tmp_path,monkeypatch):
    import json
    import expert_storage_repair as repair
    monkeypatch.setattr(repair.g,'ASSETS',tmp_path)
    for model,size in [('small',[.7,.7,.75]),('large',[.8,1.3,.75])]:
        path=tmp_path/'objects/breakfast_table'/model/'misc/metadata.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(bbox_size=size)))
    tried=[]
    def solve(items,size):
        tried.append(size);return dict(status='feasible_model')
    monkeypatch.setattr(repair,'solve_surface_proposal',solve)
    assert repair.fitting_work_surface_models({'a':[.6,.6,.1],'b':[.6,.6,.1]})==[('breakfast_table','large')]
    assert tried==[[.8,1.3]]


def test_generic_stock_collects_source_piles_and_retains_matching_references():
    from expert_storage_repair import generic_source_stock_placements
    spec=dict(objects=[dict(name=n) for n in ['base','cover','target','comparison_sample','negative']]
        +[dict(name='source',fixed_base=True)],placements=[
            dict(entity='source',relation='on_top',receptacle='floor'),
            dict(entity='base',relation='on_top',receptacle='source'),
            dict(entity='target',relation='on_top',receptacle='base'),
            dict(entity='cover',relation='on_top',receptacle='target'),
            dict(entity='comparison_sample',relation='on_top',receptacle='delivery'),
            dict(entity='negative',relation='on_top',receptacle='delivery')])
    row=dict(source_receptacle='source',destination_receptacle='delivery')
    assert {p['entity'] for p in generic_source_stock_placements(spec,row)}=={'base','target','cover'}


def test_open_source_relayout_requires_explicit_review_and_retains_container():
    from expert_storage_repair import generic_source_stock_placements
    spec=dict(objects=[dict(name='basket',category='shopping_basket'),dict(name='apple',category='apple')],
        placements=[dict(entity='basket',relation='on_top',receptacle='counter'),
                    dict(entity='apple',relation='inside',receptacle='basket')])
    row=dict(source_receptacle='basket',destination_receptacle='table',target_entities=['apple'])
    assert generic_source_stock_placements(spec,row)==[]
    assert {p['entity'] for p in generic_source_stock_placements(spec,row,include_open_container_contents=True)}=={'basket','apple'}


def test_reviewed_surface_yaw_is_serialized_into_the_real_initial_pose(monkeypatch):
    from types import SimpleNamespace as N
    import numpy as np,torch
    from scipy.spatial.transform import Rotation
    import expert_storage_repair as repair
    if not torch.cuda.is_available():pytest.skip('CUDA layout solver required')
    native=[.8,.2,.1]
    assert repair.solve_surface_proposal({'long':native},[.4,.95])['status']=='infeasible_model'
    spec=dict(objects=[dict(name='long',category='package',model='native')],
        placements=[dict(entity='long',relation='on_top',receptacle='table')])
    monkeypatch.setattr(repair,'dimensions',lambda _:dict(long=native))
    layout=N(entities={'table':dict(category='table',aabb=[[1.,2.,0.],[1.4,2.95,.8]])})
    plans=repair.plan_explicit_surfaces(spec,layout,replan_receptacles={'table'},allow_yaw=True)
    assert plans
    p=spec['placements'][0];shape=np.abs(Rotation.from_quat(p['orientation_hint_xyzw']).as_matrix())@native
    assert shape==pytest.approx([.2,.8,.1])
    xy=np.asarray(p['position_xy']);half=shape[:2]/2
    assert np.all(xy-half>=np.array([1.,2.])) and np.all(xy+half<=np.array([1.4,2.95]))


def test_rigid_table_quarter_turn_updates_floor_fit_and_serialized_pose(monkeypatch):
    from types import SimpleNamespace as N
    import numpy as np
    import constructed_storage as storage
    monkeypatch.setattr(storage,'model_for',lambda *args,**kw:('breakfast_table','native',np.array([.6,1.2,.75])))
    monkeypatch.setattr(storage,'model_openable',lambda *args:False)
    monkeypatch.setattr(storage,'observed_opening_offsets',lambda *args:None)
    monkeypatch.setattr(storage,'enrich_constructed_interiors',lambda *args:None)
    floor=dict(category='floors',aabb=[[-.9,-.45,0.],[.9,.45,0.]])
    monkeypatch.setattr(storage,'storage_floor_domain',lambda *args:([('floor',floor)],np.array([[10,10]])))
    monkeypatch.setattr(storage,'screen_furniture_footprints',lambda *args:np.array([True]))
    grid=N(size=20,resolution=.1,to_map=lambda x,y:(int(round(y/.1+10)),int(round(x/.1+10))))
    layout=N(entities={'floor':floor},map=grid,room_clear=np.ones((20,20)),free=np.ones((20,20),bool),radius=.1)
    assert storage.augment(layout,'storage_room_0','table','source') is None
    turned,_=storage.augment(layout,'storage_room_0','table','source',yaw_degrees=90)
    fixture=turned.inserted_furniture[0]
    assert np.subtract(fixture['aabb'][1],fixture['aabb'][0])==pytest.approx([1.2,.6,.75])
    spec=dict(objects=[],placements=[]);row={};storage.add_to_recipe(spec,row,turned)
    from rummagebench.core.scenario import PlacementSpec
    assert PlacementSpec(**spec['placements'][0]).orientation_hint_xyzw==fixture['orientation_hint_xyzw']
    # Do not guess how an articulated cabinet's swept opening should rotate.
    monkeypatch.setattr(storage,'model_openable',lambda *args:True)
    assert storage.augment(layout,'storage_room_0','table','source',yaw_degrees=90) is None
