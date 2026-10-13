"""CPU-only tests. Synthetic assets and subprocess outputs are TEST FIXTURES ONLY.

The USD tests use real standalone OpenUSD APIs, never mock USD interfaces and
never count these fixture stages as generated benchmark geometry.
"""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from rummagebench.authoring.tasks.household import (
    HouseholdTaskSpec, SearchLocation, TargetAsset, UPSTREAM_COMMIT,
    sample_spec, validate_spec_assets, family_members, grouped_split)
from rummagebench.authoring.tasks.difficulty import exact_search_cost, interval_label, euclidean_proxies
from rummagebench.authoring.tasks import infinigen as ig
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path

REPO = Path(__file__).resolve().parents[2]


def make_spec():
    return HouseholdTaskSpec(task_id="unit_test_only", family_id="fixture_family", base_layout_id="fixture_layout",
        seed=42, index=0, scene_seed=45, instruction="Find the bowl in the kitchen cabinets and pick it up.",
        target=TargetAsset(category="bowl", model="test_model_only", bbox_m=[.1, .1, .1]),
        locations=[SearchLocation(location_id="storage_0", prior=.5), SearchLocation(location_id="storage_1", prior=.5)],
        target_location="storage_0", background_policy="preserve_upstream_household_semantics_all_surfaces",
        appearance={"light_intensity_multiplier": 1.2, "light_color_temperature_k": 4500., "ambient_multiplier": .9,
                    "light_direction_yaw_deg": 20., "wall_color": "warm_beige", "support_texture_swap": True},
        robot_config_ref="configs/tasks/robots/r1pro.yaml", required_skills=["MOVE", "TURN", "OBSERVE", "OPEN", "GRASP", "REPORT_DONE"],
        max_planning_steps=16, initial_visibility_max=0, requested_difficulty={"spatial": "low", "operations": "high"},
        input_hashes={"unit_fixture_only": "a"*64})


@pytest.fixture
def policy(tmp_path):
    from rummagebench.authoring.tasks.eligibility import ObjectEligibility, EligibleCategory, ModelRecord
    from rummagebench.authoring.tasks.priors import PlacementPriors, PriorEntry
    doc = yaml.safe_load((REPO / 'assets/tasks/household_families_v1.yaml').read_text())
    doc.update(placement_priors='priors.yaml', eligibility='eligibility.yaml', robot_config='robot.yaml',
               appearance_config='appearance.yaml', target_categories=['bowl'])
    (tmp_path / 'policy.yaml').write_text(yaml.safe_dump(doc))
    (tmp_path / 'robot.yaml').write_text('# synthetic unit-test robot configuration only')
    (tmp_path / 'appearance.yaml').write_text((REPO / 'configs/level1/appearance_v1.yaml').read_text())
    ObjectEligibility(1, {'synthetic_test_only': True}, {'bowl': EligibleCategory('bowl', 'bowl.n.01', .1, [],
                        [ModelRecord('test_model_only', (.1,.1,.1))])}, {}).save(tmp_path/'eligibility.yaml')
    PlacementPriors(3, {'synthetic_test_only': True}, [PriorEntry('bowl.n.01', 'cabinet_interior', 'kitchen', 'inside',
        2, 1, ['bowl'], ['cabinet_interior'], True, 'high', 1.)]).save(tmp_path/'priors.yaml')
    return tmp_path, tmp_path/'policy.yaml'


def test_round_trip_and_public_privilege_boundary():
    spec = make_spec()
    assert HouseholdTaskSpec.model_validate_json(spec.canonical_bytes()) == spec
    assert set(spec.public_task()) == {'instruction', 'max_planning_steps'}
    assert spec.target_location not in json.dumps(spec.public_task())
    assert spec.target.internal_entity not in json.dumps(spec.public_task())


@pytest.mark.parametrize('changes', [
    {'family': 'occlusion'}, {'status': 'validated'}, {'final_difficulty': {'spatial': 'high'}},
    {'wrong_grasp_terminal': True}, {'initial_visibility_max': .1}, {'max_planning_steps': 0},
    {'target_location': 'unknown'}, {'required_skills': ['NAV','GRASP']},
    {'requested_difficulty': {'spatial': 'low'}}, {'unapproved_field': 1},
])
def test_invalid_spec_rejected(changes):
    doc = make_spec().model_dump(); doc.update(changes)
    with pytest.raises((ValueError, ValidationError)):
        HouseholdTaskSpec.model_validate(doc)


@pytest.mark.parametrize('changes', [{'relation':'on_top'}, {'prior':0}, {'prior':float('nan')}, {'initial_open':True}])
def test_invalid_search_location(changes):
    with pytest.raises(ValidationError):
        SearchLocation(location_id='a', prior=.5, **({k:v for k,v in changes.items() if k!='prior'})) if 'prior' not in changes else SearchLocation(location_id='a', **changes)


def test_seed_reproducibility_and_independent_streams(policy):
    repo, path = policy
    a = sample_spec(path, repo, 216, 0)
    b = sample_spec(path, repo, 216, 0)
    assert a.canonical_bytes() == b.canonical_bytes()
    config = yaml.safe_load((repo/'appearance.yaml').read_text())
    config['randomize']['light_intensity_multiplier'] = [2.,3.]
    c = sample_spec(path, repo, 216, 0, appearance_config=config)
    assert c.appearance != a.appearance
    assert (c.target, c.locations, c.target_location, c.scene_seed, c.base_layout_id, c.requested_difficulty) == (
            a.target, a.locations, a.target_location, a.scene_seed, a.base_layout_id, a.requested_difficulty)
    validate_spec_assets(a, repo, path)


@pytest.mark.parametrize('change,reason', [('category','INVALID_TARGET'),('bbox','TARGET_GEOMETRY'),('input','SPEC_INPUT')])
def test_semantic_gate_uses_actual_whitelist(policy,change,reason):
    repo,path = policy; spec=sample_spec(path,repo,1,0)
    if change=='category': spec.target.category='made_up'
    if change=='bbox': spec.target.bbox_m=[3.,3.,3.]
    if change=='input': (repo/'robot.yaml').write_text('changed unit fixture')
    with pytest.raises(ValueError, match=reason): validate_spec_assets(spec,repo,path)


def test_complete_family_background_independence_and_grouped_split():
    members = family_members(make_spec())
    assert len(members)==2 and len({s.target_location for s in members})==2
    assert family_members(members[0]) == members == family_members(members[1])
    assert ig.generation_request(members[0]) == ig.generation_request(members[1])
    members[1].appearance['light_intensity_multiplier']=5.
    members[1].robot_config_ref='another_config'
    assert len(set(grouped_split(members).values())) == 1
    assert ig.generation_command(members[0], '/cpu/python', REPO, REPO/'build/geometry') == ig.generation_command(
        members[1], '/cpu/python', REPO, REPO/'build/geometry')


def test_real_upstream_command_and_deferred_gpu_export():
    cmd=ig.generation_command(make_spec(),'/cpu/python',REPO,REPO/'build/geometry')
    assert cmd['argv'][1:3]==['-m','infinigen_examples.generate_indoors']
    assert 'configure_cycles_devices.use_gpu=False' in cmd['argv']
    assert "restrict_solving.restrict_parent_rooms=['Kitchen']" in cmd['argv']
    assert cmd['environment']['CUDA_VISIBLE_DEVICES']==''
    exp=ig.export_command('/cpu/python',REPO,REPO/'build/geometry',REPO/'build/export')
    assert exp['execution_status']=='deferred' and not exp['whole_scene_articulation_preserved']
    assert '--omniverse' in exp['argv']


def test_discrete_model_hand_calculation_and_finish_report():
    costs=[[0,3,1],[3,0,1],[1,1,0]]
    result=exact_search_cost(costs,[.5,.5],[2.,2.])
    assert result['order_on_negative_results']==[0,1]
    assert result['expected_movement']==2.5
    assert result['expected_operations']==5. # two per reveal plus GRASP+REPORT
    assert result['weighted_optimum']==7.5 and not result['real_scene_difficulty_certified']


def test_dp_matches_exhaustive_asymmetric_model_without_actual_target():
    d=np.array([[0,3,1,4],[2,0,5,1],[1,4,0,2],[3,1,2,0]],float); p=[.2,.3,.5]; check=[2,3,1]
    values=[]
    for order in itertools.permutations(range(3)):
        last=3; move=ops=value=0
        for i in order:
            move+=d[last,i];ops+=check[i];value+=p[i]*(move+ops+2);last=i
        values.append((value,order))
    result=exact_search_cost(d,p,check)
    assert (result['weighted_optimum'],tuple(result['order_on_negative_results']))==min(values)


@pytest.mark.parametrize('value,label', [(0,'low'),(7.999,'low'),(8,'high'),(100,'high')])
def test_difficulty_half_open_boundaries(value,label):
    assert interval_label(value,{'low':[0,8],'high':[8,None]})==label


@pytest.mark.parametrize('case',[float('nan'),-1,float('inf')])
def test_invalid_cost(case):
    with pytest.raises(ValueError): interval_label(case,{'low':[0,8]})


def test_proxy_is_not_a_path_or_certificate():
    doc=euclidean_proxies([0,0],[[3,4],[0,1]])
    assert doc['start_to_candidate']==[5.,1.] and not doc['traversable_path_measured']
    with pytest.raises(ValueError): interval_label(3,{'a':[0,4],'b':[2,6]})
    with pytest.raises(ValueError): exact_search_cost([[0,1],[1,0]],[0],[2])


@pytest.mark.parametrize('error,code,expected', [
    ('ModuleNotFoundError: bpy',1,'UPSTREAM_DEPENDENCY_MISSING'),
    ('Unknown configurable: invalid',1,'CONFIGURATION_INVALID'),
    ('constraints unsatisfied',1,'CONSTRAINT_UNSATISFIABLE'),
    ('unexpected crash',139,'GENERATION_FAILED'),
    ('',0,'GENERATION_OUTPUT_INVALID'),
    ('timeout',None,'GENERATION_TIMEOUT'),
])
def test_runner_failure_timeout_and_idempotency(tmp_path,monkeypatch,error,code,expected):
    monkeypatch.setattr(ig,'runtime_audit',lambda python:{'unit_fixture_only': True})
    monkeypatch.setattr(ig,'source_audit',lambda root: {'commit':UPSTREAM_COMMIT})
    calls=[]
    def run(argv,**kw):
        calls.append(argv)
        assert kw['shell'] is False and kw['env']['CUDA_VISIBLE_DEVICES']==''
        if code is None: raise subprocess.TimeoutExpired(argv,kw['timeout'])
        kw['stderr'].write(error); return SimpleNamespace(returncode=code)
    monkeypatch.setattr(ig.subprocess,'run',run)
    out=tmp_path/'unit_only_run'
    receipt=ig.run_cpu_generation(make_spec(),tmp_path,sys.executable,out,timeout_s=.1,retries=1)
    assert receipt['attempts'][-1]['reason']==expected and len(calls)==2
    assert receipt['generated_geometry_scenes']==0 and receipt['evaluation_ready_episodes']==0
    assert receipt==ig.run_cpu_generation(make_spec(),tmp_path,sys.executable,out,timeout_s=.1,retries=1)
    assert len(calls)==2
    changed=make_spec();changed.scene_seed+=1
    with pytest.raises(ValueError,match='RESUME_INPUT_CONFLICT'):
        ig.run_cpu_generation(changed,tmp_path,sys.executable,out)


def test_checkpoint_resume_does_not_rerun_completed_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(ig,'runtime_audit',lambda python:{'unit_fixture_only': True})
    monkeypatch.setattr(ig,'source_audit',lambda root:{'commit':UPSTREAM_COMMIT})
    calls=[]
    def run(argv,**kw):
        calls.append(argv);kw['stderr'].write('constraints unsatisfied');return SimpleNamespace(returncode=1)
    monkeypatch.setattr(ig.subprocess,'run',run)
    out=tmp_path/'unit_resume';ig.run_cpu_generation(make_spec(),tmp_path,sys.executable,out)
    (out/'generation.json').unlink()
    result=ig.run_cpu_generation(make_spec(),tmp_path,sys.executable,out,retries=1)
    assert len(calls)==2 and len(result['attempts'])==2 # only second attempt newly executed
    (out/'generation.json').unlink();(out/'attempt_1/attempt.json').unlink()
    result=ig.run_cpu_generation(make_spec(),tmp_path,sys.executable,out,retries=1)
    assert len(calls)==2 and result['attempts'][-1]['reason']=='GENERATION_INTERRUPTED'


def test_missing_export_and_relation_targets_fail_closed(tmp_path):
    with pytest.raises(ValueError,match='EXPORT_FILES_MISSING'): ig.inspect_export(tmp_path/'absent',tmp_path/'absent.json')
    p=tmp_path/'solve.json';p.write_text(json.dumps({'objs':{'a':{'obj':'a','tags':[],
        'relations':[{'target_name':'missing','relation':{'relation_type':'StableAgainst'}}]}}}))
    with pytest.raises(ValueError,match='UNRESOLVED_RELATION'): ig.read_solve_state(p)


def test_module_imports_do_not_load_simulator_or_blender():
    code="""import sys
import rummagebench.authoring.tasks.infinigen
import rummagebench.authoring.tasks.household_binding
import rummagebench.sim.omnigibson.infinigen_adapter
assert not any(k.split('.')[0] in {'omnigibson','isaacsim','omni','bpy','pxr'} for k in sys.modules)
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=REPO,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.fixture
def usd_bundle(tmp_path):
    pxr=pytest.importorskip('pxr', reason='standalone usd-core required; never start Isaac Sim')
    from pxr import Usd,UsdGeom,UsdLux,Gf
    scene=tmp_path/'synthetic_unit_scene.usda';solve=tmp_path/'solve_state.json'
    stage=Usd.Stage.CreateNew(str(scene));root=UsdGeom.Xform.Define(stage,'/World').GetPrim();stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    meta={'kitchen':{'obj':'kitchen','tags':['Semantics.Kitchen']}}
    UsdGeom.Xform.Define(stage,'/World/kitchen')
    points=[(-.5,-.5,-.5),(.5,-.5,-.5),(.5,.5,-.5),(-.5,.5,-.5),(-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5)]
    for i,name in enumerate(['cab0','cab1','table','prop','outside']):
        mesh=UsdGeom.Mesh.Define(stage,'/World/'+name)
        mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr([0,1,2,3,4,5,6,7,0,4,5,1,1,5,6,2,2,6,7,3,3,7,4,0])
        mesh.CreateExtentAttr([(-.5,-.5,-.5),(.5,.5,.5)])
        mesh.AddTranslateOp().Set(Gf.Vec3d(i*2,0,.5))
        tags=(['Semantics.Storage'] if i<2 else ['Semantics.Table'] if i==2 else ['Semantics.HandheldItem'] if i==3 else [])
        meta[name]={'obj':name,'tags':tags,'active':False,'generator':'<infinigen.assets.objects.tableware.PlateFactory object at unit_fixture>' if name=='prop' else None, 'relations':[{'relation':{'relation_type':'StableAgainst'},'target_name':'kitchen'}]}
    light=UsdLux.DistantLight.Define(stage,'/World/Light');light.CreateIntensityAttr(10)
    stage.GetRootLayer().Save();solve.write_text(json.dumps({'objs':meta}))
    assets=tmp_path/'unit_assets_only';model=assets/'objects/bottom_cabinet/unit_model';(model/'misc').mkdir(parents=True);(model/'usd').mkdir()
    (model/'misc/metadata.json').write_text(json.dumps({'link_tags':{'door':['openable']},'bbox_size':[1,1,1]}))
    # A synthetic file intentionally does NOT supply joints. Preparation is
    # pending; runtime_articulation_checks must still reject this asset.
    (model/'usd/unit_model.usd').write_text('# synthetic unit asset, no real articulation')
    bindings=[{'location_id':f'storage_{i}','source_id':f'cab{i}','category':'bottom_cabinet','model':'unit_model'} for i in range(2)]
    return scene,solve,assets,bindings


def test_real_usd_inspection_retains_inactive_and_nonsearch_geometry(usd_bundle):
    scene,solve,_,_=usd_bundle;doc=ig.inspect_export(scene,solve)
    assert doc['mesh_count']==5 and doc['source_to_usd']['outside']=='/World/outside'
    assert doc['joint_paths']==[] and doc['evaluation_ready_episodes']==0


def test_real_usd_assembly_is_pending_with_all_geometry_and_manifest_checks(usd_bundle,tmp_path):
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly,load_assembly_manifest,assembly_env_config,scene_slots_from_assembly,runtime_articulation_checks
    scene,solve,assets,bindings=usd_bundle;out=tmp_path/'prepared'
    doc=prepare_assembly(make_spec(),scene,solve,bindings,assets,out)
    assert len(doc['objects'])==5 and len(doc['candidate_bindings'])==2
    assert {o['role'] for o in doc['objects']}=={'static_background','movable_rigid_body','articulated_asset'}
    assert doc['status']=='pending_simulation' and doc['evaluation_ready_episodes']==0
    assert 'target_location' not in doc
    assert load_assembly_manifest(out/'assembly.json')==doc
    config=assembly_env_config(out/'assembly.json');assert len(config['objects'])==5
    slots=scene_slots_from_assembly(out/'assembly.json');assert len(slots.slots)==2
    failures=runtime_articulation_checks(SimpleNamespace(articulation_info=lambda _:None),doc)
    assert len(failures)==2
    (out/'lighting.usda').write_text('tampered unit asset')
    with pytest.raises(ValueError,match='ASSEMBLY_ARTIFACT_HASH_MISMATCH'):load_assembly_manifest(out/'assembly.json')


@pytest.mark.parametrize('failure', ['incomplete','shared','metadata'])
def test_missing_candidate_mapping_and_real_asset_evidence_reject(usd_bundle,tmp_path,failure):
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    scene,solve,assets,bindings=usd_bundle
    if failure=='incomplete':bindings.pop()
    if failure=='shared':bindings[1]['source_id']='cab0'
    if failure=='metadata':(assets/'objects/bottom_cabinet/unit_model/misc/metadata.json').write_text(json.dumps({'link_tags':{},'bbox_size':[1,1,1]}))
    with pytest.raises(ValueError):prepare_assembly(make_spec(),scene,solve,bindings,assets,tmp_path/'rejected')
    assert not (tmp_path/'rejected/assembly.json').exists()


def test_light_application_changes_real_light_attrs_only_and_records_pending(usd_bundle):
    from pxr import Usd,UsdLux
    from rummagebench.sim.omnigibson.infinigen_adapter import apply_light_appearance
    scene,_,_,_=usd_bundle;stage=Usd.Stage.Open(str(scene))
    before={str(p.GetPath()):{a.GetName():str(a.Get()) for a in p.GetAttributes()}
            for p in stage.Traverse() if p.GetName()!='Light'}
    doc=apply_light_appearance(stage,'/World',make_spec().appearance)
    assert UsdLux.LightAPI(stage.GetPrimAtPath('/World/Light')).GetIntensityAttr().Get()==pytest.approx(12)
    after={str(p.GetPath()):{a.GetName():str(a.Get()) for a in p.GetAttributes()}
           for p in stage.Traverse() if p.GetName()!='Light'}
    assert before==after and 'wall_color' in doc['material_requests_pending']


def test_unverified_family_cannot_bind_or_grant_acceptance(usd_bundle,tmp_path):
    from rummagebench.authoring.tasks.household_binding import bind_family
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
    scene,solve,assets,bindings=usd_bundle;out=tmp_path/'prepared'
    prepare_assembly(make_spec(),scene,solve,bindings,assets,out)
    overlay=SlotEmbodimentOverlay('fixture_layout','r1pro','absent',None,'partial','unit_only',{}, {},{}, {})
    path=tmp_path/'overlay.yaml';overlay.save(path)
    with pytest.raises(ValueError,match='OVERLAY_NOT_PROBED'):
        bind_family(family_members(make_spec()),out/'assembly.json',path,tmp_path/'compiled')


def test_infinigen_source_guard_cannot_be_bypassed_by_removing_family_markers(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(REPO/'scripts/tasks'))
    promoter=importlib.import_module('promote_tasks')
    (tmp_path/'candidates').mkdir()
    (tmp_path/'candidates/id.yaml').write_text('scene: {source: infinigen, model: unit_fixture_only}')
    with pytest.raises(SystemExit,match='legacy promoter is disabled'):
        promoter.preflight_episodes(tmp_path,{'id':'dev'},{'id':{'status':'certified'}},'easy_v1')


def test_candidate_spec_scene_source_contract():
    from rummagebench.core.scenario import SceneSpec
    assert SceneSpec(model='old_behavior').source=='behavior'
    with pytest.raises(ValidationError):SceneSpec(model='fixture',source='infinigen')
    with pytest.raises(ValidationError):SceneSpec(model='fixture',assembly_manifest='unit_fixture')


def test_full_family_compiles_via_existing_schema_with_current_protocol(usd_bundle,tmp_path):
    from rummagebench.authoring.tasks.household_binding import bind_family
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly,scene_slots_from_assembly
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay,SlotInteraction,StartPose,Pose2
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.env_factory import build_env_config
    scene,solve,assets,bindings=usd_bundle;out=tmp_path/'prepared'
    prepare_assembly(make_spec(),scene,solve,bindings,assets,out)
    slots=scene_slots_from_assembly(out/'assembly.json')
    urdf=tmp_path/'unit_robot.urdf';urdf.write_text('synthetic unit-test-only URDF evidence')
    pose=Pose2((1.,-2.,.0053),(0.,0.,0.,1.))
    evidence={'closed_visibility':{'visible_px':0},'closed_graspable':False}
    overlay=SlotEmbodimentOverlay(slots.scene,'r1pro',str(urdf),sha256_path(urdf),'probed','synthetic_unit_only',
        yaml.safe_load((REPO/'configs/tasks/robots/r1pro.yaml').read_text())['robot'],{},
        {'kitchen_0':[StartPose('common_start',pose)]},
        {s.slot_id:SlotInteraction(pose,True,True,True,evidence=evidence) for s in slots.slots},
        notes=[f'assembly_manifest_sha256:{sha256_path(out/"assembly.json")}'])
    op=tmp_path/'unit_overlay.yaml';overlay.save(op)
    compiled=tmp_path/'compiled';members=family_members(make_spec())
    report=bind_family(members,out/'assembly.json',op,compiled)
    assert report['counts']=={'compiled_candidates':2,'evaluation_ready_episodes':0}
    scenarios=[load_scenario(compiled/'candidates'/f'{tid}.yaml') for tid in report['task_ids']]
    assert all(s.scene.source=='infinigen' for s in scenarios)
    assert all(not s.termination.fail_on_wrong_grasp and not s.termination.succeed_when_holding_target for s in scenarios)
    assert scenarios[0].anchors==scenarios[1].anchors and scenarios[0].initial_states==scenarios[1].initial_states
    assert scenarios[0].placements!=scenarios[1].placements
    config=build_env_config(scenarios[0])
    assert len(config['objects'])==5 and config['scene']['type']=='Scene'
    assert config['robots'][0]['model']=='r1pro'
    assert report==bind_family(members,out/'assembly.json',op,compiled)
    with pytest.raises(ValueError,match='INCOMPLETE_TARGET_POSITION'):
        bind_family(members[:1],out/'assembly.json',op,tmp_path/'incomplete')


def test_missing_texture_dependency_is_a_failure_not_material_acceptance(usd_bundle):
    from pxr import Usd,UsdShade,Sdf
    scene,solve,_,_=usd_bundle;stage=Usd.Stage.Open(str(scene))
    shader=UsdShade.Shader.Define(stage,'/World/Looks/Texture')
    shader.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('missing_unit_texture.png'))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError,match='USD_TEXTURE_OR_ASSET_DEPENDENCY_MISSING'):ig.inspect_export(scene,solve)


def test_runtime_joint_labels_without_real_limits_cannot_pass():
    from rummagebench.sim.base import ArticulationInfo,ArticulationJointInfo
    from rummagebench.sim.omnigibson.infinigen_adapter import runtime_articulation_checks
    for limits in (None,(0,0),(float('nan'),1)):
        info=ArticulationInfo('unit_cabinet',[ArticulationJointInfo('hinge','revolute','root','door',limits=limits)],handle_link='door')
        assert runtime_articulation_checks(SimpleNamespace(articulation_info=lambda _:info),
                                          {'candidate_bindings':[{'name':'unit_cabinet'}]})


def test_source_audit_rejects_unpinned_checkout(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=UnitTest','-c','user.email=test@example.invalid',
                    'commit','--allow-empty','-qm','synthetic upstream unit fixture'],check=True)
    with pytest.raises(ValueError,match='UPSTREAM_VERSION_OR_SOURCE_MISMATCH'):ig.source_audit(tmp_path)


@pytest.mark.parametrize('disturb', [False, True])
def test_frozen_family_restores_common_topology_and_rejects_background_perturbation(usd_bundle,tmp_path,monkeypatch,disturb):
    """Mock physical behavior ONLY for unit tests; zero simulator acceptance."""
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly,freeze_family_background
    from rummagebench.authoring.tasks import native_occupancy
    from rummagebench.sim.omnigibson import state_io
    from rummagebench.core.scenario import ScenarioSpec
    scene,solve,assets,bindings=usd_bundle;out=tmp_path/'assembly'
    doc=prepare_assembly(make_spec(),scene,solve,bindings,assets,out)
    monkeypatch.setattr(state_io,'save_snapshot',lambda p,s,m:p.write_text('synthetic unit snapshot only'))
    monkeypatch.setattr(native_occupancy,'capture_native_occupancy',lambda b,ss,**kw:(
        {'slots':[{'slot_id':s.slot_id,'status':'confirmed_empty','geometry_complete':True} for s in ss.slots]},None))
    scenarios=[]
    robot=yaml.safe_load((REPO/'configs/tasks/robots/r1pro.yaml').read_text())['robot']
    for i in range(2):
        scenarios.append(ScenarioSpec.model_validate({'id':f'unit_member_{i}','instruction':'unit fixture only',
            'scene':{'model':'fixture_layout','source':'infinigen','assembly_manifest':str(out/'assembly.json')},
            'robot':{**robot,'init_anchor':'start'},'anchors':{'start':{'position':[0,0,0],'orientation':[0,0,0,1]}},
            'objects':[{'name':'search_target','category':'bowl','model':'unit_only'}],
            'target':{'entity':'search_target','category':'bowl'},
            'placements':[{'entity':'search_target','relation':'inside','receptacle':f'interactive_storage_{i}'}]}))
    class UnitBackend:
        def __init__(self):
            self.values={o['name']:[0.,0.,0.] for o in doc['objects']};self.target=[0.,0.,5.]
            self._env=SimpleNamespace(scene=SimpleNamespace(objects=[SimpleNamespace(name=n,dump_state=lambda serialized=False:[]) for n in self.values]))
            self.restores=[]
        def set_open(self,*a):return True
        def settle(self,*a):pass
        def validate_physics_state(self):pass
        def entity_pose(self,n):return list(self.target if n=='search_target' else self.values[n])
        def entity_aabb(self,n):return [[0.,0.,0.],[1.,1.,1.]]
        def capture_observe_state(self):return deepcopy({'values':self.values,'target':self.target,'sim':[]})
        def restore_observe_state(self,s):
            self.values=deepcopy(s['values']);self.target=deepcopy(s['target']);self.restores.append(deepcopy(s))
        def setup(self,scenario,**kw):
            kw['native_scene_capture'](self);self.target=kw['task_seed_pose'](self,scenario.objects[0]);kw['pre_placement_capture'](self)
        def _apply_placements(self,s,attempts):
            assert attempts==3
            self.target=[float(s.id[-1]),0.,.5]
            if disturb:self.values[next(iter(self.values))][0]+=1.
            return {'placements':[{'verified':True}]}
    backend=UnitBackend();checked=tmp_path/'family_check'
    if disturb:
        with pytest.raises(ValueError,match='TARGET_INSERTION_DISTURBED_BACKGROUND'):
            freeze_family_background(backend,scenarios,out/'assembly.json',checked)
        assert json.loads((checked/'family_preflight.json').read_text())['status']=='rejected_family_preflight'
    else:
        result=freeze_family_background(backend,scenarios,out/'assembly.json',checked)
        assert len(result['members'])==2 and result['evaluation_ready_episodes']==0
        assert backend.restores[0]==backend.restores[3] # shared parked target + background
        assert result['status']=='pending_simulation_acceptance'


@pytest.mark.parametrize("generator", [None, "<infinigen.assets.objects.tableware.BowlFactory object at unit_fixture>"])
def test_native_ambiguous_target_is_preserved_and_rejected_before_family_binding(usd_bundle,tmp_path,generator):
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    from rummagebench.authoring.tasks.household_binding import bind_family
    scene,solve,assets,bindings=usd_bundle
    state=json.loads(solve.read_text());state['objs']['prop']['generator']=generator;solve.write_text(json.dumps(state))
    out=tmp_path/'prepared';doc=prepare_assembly(make_spec(),scene,solve,bindings,assets,out)
    assert doc['target_category_confounds'] and any(o['source_id']=='prop' for o in doc['objects'])
    with pytest.raises(ValueError,match='NATIVE_TARGET_CATEGORY_AMBIGUOUS'):
        bind_family(family_members(make_spec()),out/'assembly.json',tmp_path/'absent_overlay',tmp_path/'compiled')


def test_extra_kitchen_storage_cannot_be_silently_excluded_from_search_set(usd_bundle,tmp_path):
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    scene,solve,assets,bindings=usd_bundle
    state=json.loads(solve.read_text());state['objs']['table']['tags']=['Semantics.Storage'];solve.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='UNBOUND_KITCHEN_STORAGE_SEARCH_SET_UNRESOLVED'):
        prepare_assembly(make_spec(),scene,solve,bindings,assets,tmp_path/'rejected')
    assert not (tmp_path/'rejected/assembly.json').exists()
