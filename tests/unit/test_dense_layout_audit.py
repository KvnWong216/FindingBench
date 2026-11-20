from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_layered_open_container_witness_has_no_3d_overlaps_and_respects_height(monkeypatch):
    import pytest,torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import layered_open_container_proposal
    items={str(i):[.15,.15,.05] for i in range(2)}
    result=layered_open_container_proposal(items,{'basket':[.2,.2,.15]})
    assert result['status']=='feasible_model' and result['physical_validation_required']
    boxes=[]
    for name,p in result['poses'].items():
        low=[*p['lower_xy_m'],p['lower_z_m']];high=[low[0]+p['footprint_m'][0],low[1]+p['footprint_m'][1],low[2]+items[name][2]]
        assert all(0<=a<b<=limit for a,b,limit in zip(low,high,[.2,.2,.15]))
        boxes.append((low,high))
    assert any(boxes[0][1][k]<=boxes[1][0][k] or boxes[1][1][k]<=boxes[0][0][k] for k in range(3))
    assert layered_open_container_proposal({str(i):[.15,.15,.05] for i in range(3)}, {'basket':[.2,.2,.15]}) is None

def test_rigid_container_does_not_need_articulated_joints_for_geometry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import offline_interior_hints
    geometry=dict(joints=[],volumes=[dict(link='fillable',components=[dict(vertices=[[0,0,0],[.3,.2,.15]])])])
    hints=offline_interior_hints(geometry,[1,1,1],[0,0,0,1])
    assert len(hints)==1 and hints[0]['link']=='fillable'

def test_upright_fit_does_not_assume_carton_can_lie_down(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import upright_fit
    assert not upright_fit([.07,.24,.24],[.4,.4,.10])
    assert upright_fit([.07,.24,.24],[.3,.3,.3])

def test_allocator_spreads_across_native_compartments(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import allocate
    plan=allocate({str(i):[.1,.1,.1] for i in range(4)},
                  {'first':[.3,.3,.2],'second':[.3,.3,.2]})
    assert set(plan.values())=={'first','second'}

def test_allocator_requires_review_when_capacity_is_insufficient(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import allocate
    assert allocate({'large':[.4,.4,.4]},{'small':[.2,.2,.2]}) is None

def test_fixed_items_reserve_capacity_first(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import allocate
    plan=allocate({'free':[.2,.2,.2],'fixed':[.2,.2,.1]},
                  {'a':[.3,.3,.3],'b':[.3,.3,.3]}, {'fixed':'a'})
    assert plan == {'fixed':'a','free':'b'}

def test_preflight_counts_required_negatives_and_inventory(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import preflight_recipe
    names=['target','required_negative','initial_inventory']
    spec={'instruction':'Find the mug.', 'placements':[
        dict(entity=n,relation='inside',receptacle='cabinet') for n in names]}
    entities={'cabinet':{'interiors':[dict(link='link_0',lo=[0,0,0],hi=[.3,.3,.3])]}}
    result=preflight_recipe(spec,entities,{n:[.2,.2,.1] for n in names})
    assert result['status']=='rejected_candidate'
    assert result['assignments']=={}

def test_preflight_preserves_location_and_unknown_geometry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import preflight_recipe
    spec={'instruction':'Find the mug in the upper drawer.', 'placements':[
        dict(entity='mug',relation='inside',receptacle='cabinet')]}
    entities={'cabinet':{'interiors':[dict(link='link_0',lo=[0,0,0],hi=[.3,.3,.3])]}}
    result=preflight_recipe(spec,entities,{'mug':[.1,.1,.1]})
    assert result['assignments']=={}
    assert result['physical_validation_required']
    assert preflight_recipe(spec,{}, {'mug':[.1,.1,.1]})['status']=='unknown'

def test_preflight_balances_without_mutating_recipe(monkeypatch):
    import copy
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import preflight_recipe
    spec={'instruction':'Find the mug.', 'placements':[
        dict(entity=n,relation='inside',receptacle='cabinet') for n in ['a','b']]}
    original=copy.deepcopy(spec)
    entities={'cabinet':{'interiors':[
        dict(link=k,lo=[0,0,0],hi=[.3,.3,.3]) for k in ['link_0','link_1']]}}
    result=preflight_recipe(spec,entities,{'a':[.2,.2,.1],'b':[.2,.2,.1]})
    assert set(result['assignments'].values())=={'link_0','link_1'}
    assert spec==original


def test_constructed_storage_uses_model_capability_not_category(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import constructed_storage as construction
    monkeypatch.setattr(construction.g,'ASSETS',tmp_path)
    for model,tags in [('plain',{'base_link':[]}),('drawer',{'base_link':[],'link_0':['openable']})]:
        path=tmp_path/'objects/nightstand'/model/'misc/metadata.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'link_tags':tags}))
    assert not construction.model_openable('nightstand','plain')
    assert construction.model_openable('nightstand','drawer')
    assert not construction.model_openable('nightstand','missing')
    assert construction.MODELS['drawer']==('bottom_cabinet_no_top','qohxjq')


def test_offline_fillable_geometry_preserves_components_and_root_frame(monkeypatch):
    import pytest
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, Gf
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import fillable_geometry_from_stage
    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/asset');stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(100,200,300))
    link=UsdGeom.Xform.Define(stage,'/asset/meta__drawer_fillable_0_0_link')
    link.AddTranslateOp().Set(Gf.Vec3d(1,2,3))
    UsdGeom.Scope.Define(stage,str(link.GetPath())+'/visuals')
    for name,shift in [('left',0),('right',5)]:
        mesh=UsdGeom.Mesh.Define(stage,str(link.GetPath())+'/visuals/'+name)
        mesh.GetPointsAttr().Set([(0,0,0),(1,2,3)])
        mesh.AddTranslateOp().Set(Gf.Vec3d(shift,0,0))
        UsdGeom.Imageable(mesh).MakeInvisible()
    ordinary=UsdGeom.Mesh.Define(stage,'/asset/decor')
    ordinary.GetPointsAttr().Set([(0,0,0),(99,99,99)])
    result=fillable_geometry_from_stage(stage)
    assert len(result['volumes'])==1
    parts=result['volumes'][0]['components']
    assert len(parts)==2
    assert parts[0]['lo']==[1,2,3] and parts[0]['hi']==[2,4,6]
    assert parts[1]['lo']==[6,2,3] and parts[1]['hi']==[7,4,6]
    assert result['physical_validation_required'] is True


def test_offline_hints_fail_closed_and_preserve_survey(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import offline_interior_hints,authoring_volumes
    from copy import deepcopy
    geometry={'joints':[{'type':'PhysicsPrismaticJoint'}],
        'volumes':[{'link':'drawer','components':[{'vertices':[[0,0,0],[.4,.5,.1]]}]}]}
    hint=offline_interior_hints(geometry,[1,2,1],[0,0,0,1])
    assert hint[0]['extent_m']==[.402,1.002,.10200000000000001]
    assert offline_interior_hints(geometry,[1,-1,1],[0,0,0,1])==[]
    assert offline_interior_hints(geometry,[1,1,1],[.5,0,0,.8660254])==[]
    hinged=deepcopy(geometry);hinged['joints'][0]['type']='PhysicsRevoluteJoint'
    assert offline_interior_hints(hinged,[1,1,1],[0,0,0,1])==[]
    split=deepcopy(geometry);split['volumes'][0]['components']*=2
    assert len(offline_interior_hints(split,[1,1,1],[0,0,0,1]))==2
    row={'interiors':[{'link':'surveyed','lo':[0,0,0],'hi':[1,2,3]}],
         'offline_interior_hints':hint}
    assert authoring_volumes(row)=={'surveyed':[1,2,3]}
    spec={'instruction':'Find the cup','placements':[{'entity':'cup','receptacle':'cabinet','relation':'inside'}]}
    from audit_dense_layouts import preflight_recipe
    assert preflight_recipe(spec,{'cabinet':{'offline_interior_hints':hint}}, {'cup':[.14,.18,.18]})['status']=='rejected_candidate'
    assert preflight_recipe(spec,{'cabinet':{}},{'cup':[.14,.18,.18]})['status']=='unknown'


def test_offline_enrichment_checks_model_scale_and_asset_hashes(monkeypatch,tmp_path):
    import hashlib
    from copy import deepcopy
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import enrich_offline_interiors
    asset=tmp_path/'cabinet/model';(asset/'usd').mkdir(parents=True);(asset/'misc').mkdir()
    (asset/'usd/model.encrypted.usd').write_bytes(b'asset')
    (asset/'misc/metadata.json').write_bytes(b'{}')
    geometry={'joints':[{'type':'PhysicsPrismaticJoint'}],
        'volumes':[{'link':'drawer','components':[{'vertices':[[0,0,0],[.4,.5,.1]]}]}],
        'encrypted_asset_sha256':hashlib.sha256(b'asset').hexdigest(),
        'metadata_sha256':hashlib.sha256(b'{}').hexdigest()}
    index={'assets':{'cabinet/model':geometry}}
    scene={'objects_info':{'init_info':{'x':{'args':{'category':'cabinet','model':'model','scale':[1,1,1]}}}},
           'state':{'registry':{'object_registry':{'x':{'root_link':{'ori':[0,0,0,1]}}}}}}
    rows={'x':{'category':'cabinet','model':'model','interiors':[]}}
    good=deepcopy(rows)
    assert enrich_offline_interiors(good,scene,index,tmp_path)==1
    assert good['x']['offline_interior_provenance']['physical_validation_required']
    mismatch=deepcopy(scene);mismatch['objects_info']['init_info']['x']['args']['model']='different'
    assert enrich_offline_interiors(deepcopy(rows),mismatch,index,tmp_path)==0
    (asset/'misc/metadata.json').write_bytes(b'{"changed":true}')
    assert enrich_offline_interiors(deepcopy(rows),scene,index,tmp_path)==0


def test_explicit_drawer_alias_checks_size_without_rewriting_task(monkeypatch):
    from copy import deepcopy
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import preflight_recipe
    spec={'instruction':'Find it in the drawer','placements':[{'entity':'item','relation':'inside','receptacle':'cabinet','link':'link_1'}]}
    original=deepcopy(spec)
    entities={'cabinet':{'offline_interior_hints':[{'link':'meta__link_1_fillable_0_0_link','extent_m':[.4,.5,.1]}]}}
    result=preflight_recipe(spec,entities,{'item':[.15,.15,.15]})
    assert result['status']=='rejected_candidate' and spec==original
    result=preflight_recipe(spec,entities,{'item':[.02,.03,.04]})
    assert result['status']=='heuristic_fit' and result['assignments']=={} and spec==original
    entities['cabinet']['offline_interior_hints'].append({'link':'meta__link_1_fillable_1_0_link','extent_m':[1,1,1]})
    assert preflight_recipe(spec,entities,{'item':[.15,.15,.15]})['status']=='unknown'


def test_hinged_door_does_not_invalidate_fixed_cavity(monkeypatch):
    from copy import deepcopy
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import offline_interior_hints
    cavity='/asset/meta__base_fillable_0_0_link'
    geometry={'volumes':[{'link':cavity.rsplit('/',1)[1],'components':[{'prim_path':cavity+'/visuals/mesh','vertices':[[0,0,0],[1,1,1]]}]}],
        'joints':[{'type':'PhysicsRevoluteJoint','body0':['/asset/base'],'body1':['/asset/door']},
                  {'type':'PhysicsFixedJoint','body0':['/asset/base'],'body1':[cavity]}]}
    assert len(offline_interior_hints(geometry,[1,1,1],[0,0,0,1]))==1
    moving=deepcopy(geometry);moving['joints'][1]['body0']=['/asset/door']
    assert offline_interior_hints(moving,[1,1,1],[0,0,0,1])==[]
    cycle=deepcopy(geometry);cycle['joints'][1]['body0']=[cavity]
    assert offline_interior_hints(cycle,[1,1,1],[0,0,0,1])==[]


def test_component_packing_never_fuses_partitions_or_emits_virtual_links(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import preflight_recipe
    link='meta__base_fillable_0_0_link'
    entities={'cabinet':{'offline_interior_hints':[
        {'link':link,'component':0,'extent_m':[.2,.2,.2]},
        {'link':link,'component':1,'extent_m':[.2,.2,.2]}]}}
    spec={'instruction':'Find the items','placements':[
        {'entity':name,'relation':'inside','receptacle':'cabinet'} for name in ['a','b']]}
    result=preflight_recipe(spec,entities,{'a':[.15,.15,.1],'b':[.15,.15,.1]})
    assert result['status']=='heuristic_fit'
    assert result['assignments']=={'a':link,'b':link}
    assert len(set(result['findings'][0]['component_allocation'].values()))==2
    assert preflight_recipe(spec,entities,{'a':[.3,.15,.1],'b':[.15,.15,.1]})['status']=='rejected_candidate'
    for p in spec['placements']:p['link']=link
    result=preflight_recipe(spec,entities,{'a':[.15,.15,.1],'b':[.15,.15,.1]})
    assert result['status']=='heuristic_fit' and result['assignments']=={}


def test_model_selection_requires_openability_even_when_size_fits(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import constructed_storage as c
    monkeypatch.setattr(c.g,'ASSETS',tmp_path)
    monkeypatch.setitem(c.MODELS,'cabinet',('bottom_cabinet','closed'))
    for name,tags in [('closed',{}),('working',{'door':['openable']})]:
        path=tmp_path/'objects'/'bottom_cabinet'/name/'misc'/'metadata.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(bbox_size=[.8,.8,.8],link_tags=tags)))
    assert c.model_for('cabinet',[.1,.1,.1])[1]=='working'
    (tmp_path/'objects/bottom_cabinet/working/misc/metadata.json').unlink()
    assert c.model_for('cabinet',[.1,.1,.1]) is None


def test_long_narrow_native_furniture_uses_rectangular_clearance():
    import numpy as np
    from types import SimpleNamespace as NS
    import constructed_storage as c
    layout=NS(doc={'footprint_boxes':[[-.6,-2.,-.5,2.],[.5,-2.,.6,2.]]},inserted_furniture=[])
    points=np.array([[0.,0.],[.45,0.]])
    # A 2.4-m shelf fits lengthways in this aisle; its circumscribed circle
    # cannot fit. The actual width, orientation and obstacle boxes decide.
    assert c.screen_furniture_footprints(layout,points,[[-.15,-1.2],[.15,1.2]]).tolist()==[True,False]
    assert not c.screen_furniture_footprints(layout,points,[[-1.2,-.15],[1.2,.15]]).any()
    layout.inserted_furniture=[dict(aabb=[[-.1,-.1,0],[.1,.1,1.]])]
    assert not c.screen_furniture_footprints(layout,points,[[-.15,-1.2],[.15,1.2]]).any()


def test_constructed_source_requires_real_compartment_not_shell(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import constructed_storage as c
    monkeypatch.setattr(c.g,'ASSETS',tmp_path)
    monkeypatch.setitem(c.MODELS,'cabinet',('bottom_cabinet','partitioned'))
    for name in ['partitioned','tall','unknown']:
        path=tmp_path/'objects/bottom_cabinet'/name/'misc/metadata.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(bbox_size=[.8,.8,.8],link_tags={'door':['openable']})))
    def measured(entities):
        row=entities['candidate']
        heights={'partitioned':.18,'tall':.4}
        if row['model'] in heights:
            row['interiors']=[dict(link='shelf',lo=[0,0,0],hi=[.6,.6,heights[row['model']]])]
    monkeypatch.setattr(c,'enrich_constructed_interiors',measured)
    # All three shells fit, but only the tall measured compartment holds this bottle.
    assert c.model_for('cabinet',[.1,.1,.25],require_interior=True)[1]=='tall'
    # A delivery surface need not accommodate the bottle inside its cupboard.
    assert c.model_for('cabinet',[.1,.1,.25])[1]=='partitioned'
    (tmp_path/'objects/bottom_cabinet/tall/misc/metadata.json').unlink()
    assert c.model_for('cabinet',[.1,.1,.25],require_interior=True) is None


def test_failure_taxonomy_keeps_symptoms_separate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import classify_failure
    assert classify_failure('native_timeout')==('timeout','cause_unresolved')
    assert classify_failure('POST_SETTLE_RELATION_FAILED: target')==('placement','post_settle_relation_lost')
    assert classify_failure('OPEN_STATE_CHANGED: cabinet')==('articulation','declared_open_state_changed')
    assert classify_failure('search_storage has no Open state')[0]=='asset_capability'


def test_sampling_size_gate_rejects_unknown_and_oversized(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from generate_dense_pilot import fits_known_interior
    assert not fits_known_interior({},[.1,.1,.1])
    row={'offline_interior_hints':[{'link':'drawer','extent_m':[.327,.268,.122]}]}
    assert not fits_known_interior(row,[.449,.044,.032])
    assert fits_known_interior(row,[.225,.039,.013])


def test_source_layout_signature_tracks_container_and_location(monkeypatch):
    from copy import deepcopy
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import source_layout_signature
    spec={'scene':{'model':'house'},'objects':[{'name':'box','category':'cabinet','model':'a'}],
          'placements':[{'entity':'target','relation':'inside','receptacle':'box'},
                        {'entity':'box','relation':'on_top','receptacle':'floor','position_xy':[1,2]}]}
    first=source_layout_signature(spec,'target');changed=deepcopy(spec)
    changed['objects'][0]['model']='b'
    assert source_layout_signature(changed,'target')!=first
    changed=deepcopy(spec);changed['placements'][1]['position_xy']=[2,2]
    assert source_layout_signature(changed,'target')!=first
    changed=deepcopy(spec);changed['instruction']='Another expression of the same task'
    assert source_layout_signature(changed,'target')==first


def test_joint_capacity_failure_is_rejected_before_sampling(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import preflight_recipe
    spec={'instruction':'Find a cup','placements':[
        dict(entity=n,relation='inside',receptacle='cabinet') for n in ['a','b','c']]}
    entities={'cabinet':{'offline_interior_hints':[{'link':'drawer','extent_m':[.3,.3,.2]}]}}
    result=preflight_recipe(spec,entities,{n:[.2,.2,.1] for n in ['a','b','c']})
    assert result['status']=='rejected_candidate'
    assert not result['findings'][0]['native_sampling_required']


def test_binding_sync_preserves_per_object_compartments(tmp_path,monkeypatch):
    import json,yaml
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import sync_bound_recipes as sync
    monkeypatch.setattr(sync,'load_scenario',lambda path:None)
    spec={'objects':[dict(name=n,category=c,model=m) for n,c,m in
        [('target','cup','a'),('other_target','cup','a'),('required_negative_0','bowl','b')]],
        'placements':[dict(entity=n,relation='inside',receptacle='cabinet',link=link) for n,link in
            [('target','link_0'),('other_target','link_1'),('required_negative_0','link_2')]],
        'goal':[dict(kind='on_top',entity=n,receptacle='table') for n in ['target','other_target']],
        'instruction':'Find both cups.'}
    path=tmp_path/'scenario.yaml';path.write_text(yaml.safe_dump(spec))
    binding=dict(status='ready',target_categories=['cup','cup'],target_model_overrides=['cup/a','cup/a'],
        required_distractor_models=['bowl/b'],destination_kind='table',source_kind='cabinet',
        source_room_type='kitchen',destination_room_type='kitchen',instruction='Find both cups.')
    row=dict(id='test',parent_task_id='T',scenario=str(path),target_entities=['target','other_target'])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(tasks=[row],binding_reviews={'T':binding})))
    sync.synchronize(tmp_path)
    actual=yaml.safe_load(path.read_text())
    assert {p['entity']:p['link'] for p in actual['placements']}=={
        'target':'link_0','other_target':'link_1','required_negative_0':'link_2'}
    assert sync.synchronize(tmp_path)==[]


def test_diagnostic_motion_delta_handles_quaternion_sign_and_scale(monkeypatch):
    import math
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from pinpoint_controls import motion_delta
    assert motion_delta([0,0,0,0,0,0,2],[0,0,0,0,0,0,-1])=={'translation_m':0.,'rotation_rad':0.}
    result=motion_delta([0,0,0,0,0,0,1],[.001,0,0,0,0,1,0])
    assert abs(result['translation_m']-.001)<1e-10
    assert abs(result['rotation_rad']-math.pi)<1e-10


def test_exact_layout_catches_area_feasible_but_geometrically_impossible(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import solve_layout
    # The previous 75%-area heuristic accepts .18*.18*2 < .3*.3*.75.
    # Two axis-aligned .18 squares cannot fit side by side along either axis.
    result=solve_layout({'a':[.18,.18,.1],'b':[.18,.18,.1]}, {'drawer':[.3,.3,.2]})
    assert result['status']=='infeasible_model'


def test_exact_layout_returns_nonoverlapping_positions_and_yaw(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import solve_layout
    result=solve_layout({'a':[.22,.11,.1],'b':[.22,.11,.1]}, {'drawer':[.26,.26,.2]},time_limit_s=2.)
    assert result['status']=='feasible_model' and len(result['poses'])==2
    a,b=result['poses'].values()
    ax,ay=a['lower_xy_m'];aw,ah=a['footprint_m'];bx,by=b['lower_xy_m'];bw,bh=b['footprint_m']
    assert ax+aw+.005<=bx+1e-8 or bx+bw+.005<=ax+1e-8 or ay+ah+.005<=by+1e-8 or by+bh+.005<=ay+1e-8
    assert result['physical_validation_required']


def test_exact_layout_distinguishes_solver_timeout_from_infeasibility(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import solve_layout
    result=solve_layout({'unique_timeout_item':[.100123,.100123,.1]}, {'unique_timeout_bin':[.41234,.41234,.3]},max_nodes=0)
    assert result['status']=='unknown'


def test_cuda_layout_matches_exhaustive_tiny_grid(monkeypatch):
    import itertools,torch,pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import solve_layout
    def exhaustive(shapes,placed=()):
        if not shapes:return True
        for w,h in set([shapes[0],shapes[0][::-1]]):
            for x in range(4-w+1):
                for y in range(3-h+1):
                    if all(x+w<=ox or ox+ow<=x or y+h<=oy or oy+oh<=y for ox,oy,ow,oh in placed):
                        if exhaustive(shapes[1:],placed+((x,y,w,h),)):return True
        return False
    shapes=[(1,1),(2,1),(2,2),(3,2)]
    for a,b,c in itertools.combinations_with_replacement(shapes,3):
        items={str(i):[w*.01,h*.01,.01] for i,(w,h) in enumerate([a,b,c])}
        result=solve_layout(items,{'bin':[.04,.03,.02]},grid_m=.01,margin_m=0,time_limit_s=2.)
        assert result['status']!='unknown'
        assert (result['status']=='feasible_model')==exhaustive([a,b,c]),(a,b,c,result)
        assert result['backend']=='torch_cuda_finite_csp'


def test_layout_solver_has_no_implicit_cpu_fallback(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import solve_layout
    result=solve_layout({'item':[.1,.1,.1]},{'bin':[.3,.3,.3]},device='cpu')
    assert result['status']=='unknown' and 'no automatic CPU' in result['reason']


def test_survey_union_never_recombines_native_shelves(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from audit_dense_layouts import authoring_volumes,authoring_native_links,preflight_recipe
    native='meta__link_0_fillable_0_0_link'
    row={'interiors':[{'link':'link_0','lo':[0,0,0],'hi':[.48,.21,.65]}],
        'offline_interior_hints':[{'link':native,'component':i,'extent_m':[.48,.21,h]} for i,h in enumerate([.177,.185,.244])]}
    bins=authoring_volumes(row)
    assert len(bins)==3 and max(v[2] for v in bins.values())==.244
    assert set(authoring_native_links(row).values())=={native}
    spec={'instruction':'Find the syrup','placements':[{'entity':'syrup','relation':'inside','receptacle':'cabinet','link':'link_0'}]}
    result=preflight_recipe(spec,{'cabinet':row},{'syrup':[.075,.075,.258]})
    assert result['status']=='rejected_candidate'


def test_openable_joint_evidence_does_not_require_optional_link_tags(tmp_path,monkeypatch):
    import json,hashlib
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import constructed_storage as c
    monkeypatch.setattr(c.g,'ROOT',tmp_path);monkeypatch.setattr(c.g,'ASSETS',tmp_path/'assets')
    asset=tmp_path/'assets/objects/cabinet/model';(asset/'misc').mkdir(parents=True);(asset/'usd').mkdir()
    metadata=asset/'misc/metadata.json';source=asset/'usd/model.encrypted.usd'
    metadata.write_text(json.dumps({'link_tags':{'drawer':[]}}));source.write_bytes(b'native joint asset')
    index=tmp_path/'build/native_fillable_geometry/index.json';index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'assets':{'cabinet/model':dict(joints=[{'type':'PhysicsPrismaticJoint'}],
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        encrypted_asset_sha256=hashlib.sha256(source.read_bytes()).hexdigest())}}))
    assert c.model_openable('cabinet','model')
    source.write_bytes(b'changed asset')
    assert not c.model_openable('cabinet','model')
