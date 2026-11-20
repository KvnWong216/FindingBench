"""Real upstream unit assets, never benchmark scenes or acceptance records."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from rummagebench.authoring.tasks.infinigen import CPU_ENV, cpu_sandbox_argv, source_audit


@pytest.mark.local_assets
def test_real_nested_cabinet_provenance_preserves_geometry_and_all_drawers(tmp_path):
    root = os.environ.get("FINDINGBENCH_INFINIGEN_ROOT")
    python = os.environ.get("FINDINGBENCH_INFINIGEN_CPU_PYTHON")
    if not root or not python:
        pytest.skip("real pinned upstream CPU environment required")
    source_audit(Path(root))
    script = '''
import bpy, json
from mathutils import Vector
from infinigen.assets.objects.shelves.kitchen_cabinet import KitchenCabinetBaseFactory
from infinigen.assets.objects.shelves.kitchen_space import KitchenSpaceFactory
from rummagebench.authoring.tasks.infinigen_components import CabinetProvenance,census_mesh,mesh_geometry_digest,CELL_ATTRIBUTE
from rummagebench.authoring.tasks.infinigen_component_binding import split_real_cells,face_signature
# A real upstream asset under unit-test dimensions, not an evaluation house.
vanilla=KitchenSpaceFactory(421,dimensions=Vector((.8,3.8,2.4)))(i=0)
original=mesh_geometry_digest(vanilla.data)
registry=CabinetProvenance(KitchenCabinetBaseFactory).install()
try: tagged=KitchenSpaceFactory(421,dimensions=Vector((.8,3.8,2.4)))(i=0)
finally: registry.restore()
assert mesh_geometry_digest(tagged.data)==original, 'instrumentation changed geometry or UV'
tagged.location=(2,3,1)
bpy.context.view_layer.update()
report=census_mesh(tagged,registry,'unit_counter')
assert len(report['cells'])>=8 and report['background_face_count']>0
assert {r['kind'] for r in report['cells']} >= {'drawer','door'}
drawers=[r for r in report['cells'] if r['kind']=='drawer']
assert sum(len(r['attachments']) for r in drawers)==sum(l['slot_type']=='drawer' for l in report['locations'])
assert len({l['location_id'] for l in report['locations']})==len(report['locations'])
assert sum(len(c['face_indices']) for c in report['cells'])+report['background_face_count']==report['face_count']
assert report['geometry_modified'] is False
split=split_real_cells(tagged,report)
assert split['geometry_uv_materials_preserved'] and len(split['cells'])==len(report['cells'])
assert all(c['local_aabb'] and c['matrix_world'] for c in split['cells'])
# Removing real face provenance must be rejected rather than guessing cells.
tagged.data.attributes.remove(tagged.data.attributes[CELL_ATTRIBUTE])
try:census_mesh(tagged,registry,'unit_counter')
except ValueError as exc: assert str(exc).startswith('NESTED_CABINET_PROVENANCE_LOST:')
else:raise AssertionError('missing source provenance accepted')
print(json.dumps({'unit_asset_only':True,'cells':len(report['cells']),'locations':len(report['locations'])}))
'''
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, **CPU_ENV, PYTHONPATH=os.pathsep.join([str(repo / "src"), root]))
    result = subprocess.run(cpu_sandbox_argv([python, "-c", script], tmp_path),
        cwd=root, env=env, capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stderr[-6000:]
    report = json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
    assert report['unit_asset_only'] is True


def test_complete_distribution_keeps_every_drawer_and_supersedes_old_prior():
    from types import SimpleNamespace
    from rummagebench.authoring.tasks.infinigen_component_binding import expand_search_distribution
    from rummagebench.authoring.tasks.household import TargetAsset
    spec = SimpleNamespace(family_id='old_two_cabinet_family', split_assignment=None,
        target=TargetAsset(category='tablespoon', model='unit_only', bbox_m=[.2,.03,.01]),
        appearance={}, robot_id='r1pro', max_planning_steps=48)
    sources=[]
    for i in range(2):
        cells=[{'component_id':f'cell_{i}_door','kind':'door','face_indices':[0,1],
                'attachments':[{'part_id':1}]},
               {'component_id':f'cell_{i}_drawers','kind':'drawer','face_indices':[2,3,4],
                'attachments':[{'part_id':1},{'part_id':2}]}]
        locations=[{'location_id':f'cell_{i}_door','component_id':f'cell_{i}_door',
                    'source_id':str(i),'slot_type':'cabinet_interior','coupled_source_component':False}]
        locations += [{'location_id':f'cell_{i}_drawers__drawer_{p}',
                       'component_id':f'cell_{i}_drawers','source_id':str(i),'slot_type':'drawer',
                       'part_id':p,'coupled_source_component':True} for p in (1,2)]
        sources.append({'source_id':str(i),'cells':cells,'locations':locations,
                        'face_count':6,'background_face_count':1})
    census={'status':'source_complete_pending_binding','source_roots':['0','1'],
            'sources':sources,'failures':[],'search_location_count':6}
    result=expand_search_distribution(spec,census)
    assert result['candidate_count']==6 and len(result['coupled_drawer_locations'])==4
    assert all(l['prior']==1/6 for l in result['locations'])
    assert result['supersedes_family_id']==spec.family_id and result['family_id']!=spec.family_id
    assert result['compiled_families']==0 and result['final_difficulty'] is None
    assert expand_search_distribution(spec,census)==result
    # Native open cabinet cells also stay in the user's whole-kitchen set.
    census['sources'][0]['cells'].append({'component_id':'open_cell','kind':'none',
        'face_indices':[5],'attachments':[]})
    census['sources'][0]['background_face_count']=0
    expanded=expand_search_distribution(spec,census)
    assert expanded['candidate_count']==7 and expanded['open_candidate_count']==1
    assert all(l['prior']==1/7 for l in expanded['locations'])
    # No successful reweighting after dropping a drawer, even if counts change.
    census['sources'][0]['locations'].pop()
    census['search_location_count']=5
    with pytest.raises(ValueError,match='DROPPED_CABINET_OR_DRAWER'):
        expand_search_distribution(spec,census)


def test_component_asset_matching_preserves_world_scale_and_refuses_pruning(tmp_path,monkeypatch):
    """Explicit unit-only metadata/asset bytes; never formal data inputs."""
    from types import SimpleNamespace
    from rummagebench.authoring.tasks.infinigen_component_binding import propose_component_assets
    from rummagebench.authoring.tasks.slots import SceneSlots
    slots=tmp_path/'unit_slots.yaml';slots.write_text('unit-fixture-only')
    monkeypatch.setattr(SceneSlots,'load',lambda p:SimpleNamespace(slots=[SimpleNamespace(
        room_type='kitchen',requires_open=True,articulated=True,slot_type='drawer',
        parent_category='unit_only',parent_model='unit_only')]))
    base=tmp_path/'objects/unit_only/unit_only'
    (base/'misc').mkdir(parents=True);(base/'usd').mkdir()
    (base/'usd/unit_only.usd').write_text('UNIT FIXTURE ONLY; not a USD stage')
    (base/'misc/metadata.json').write_text(json.dumps({'bbox_size':[1,1,1],
        'base_link_offset':[.1,0,0],'link_tags':{'door':['openable']}}))
    cell={'component_id':'unit_cell','kind':'drawer','local_aabb':[[0,0,0],[1,1,1]],
          'matrix_world':[[2,0,0,3],[0,2,0,4],[0,0,2,5],[0,0,0,1]],'attachments':[{'part_id':1}]}
    doc={'sources':[{'source_id':'unit_source','cells':[cell]}]}
    result=propose_component_assets(doc,[slots],tmp_path)
    assert result['bindings'][0]['scale']==[2,2,2]
    assert result['bindings'][0]['source_bbox_center_world']==[4,5,6]
    assert result['bindings'][0]['root_position_proposal']==[3.8,5,6]
    assert result['evaluation_ready_episodes']==0
    from copy import deepcopy
    from rummagebench.authoring.tasks.infinigen_component_binding import verify_component_asset_proposals
    verify_component_asset_proposals(doc,result)
    partial=deepcopy(result);partial['bindings'].clear()
    with pytest.raises(ValueError,match='ASSET_MAPPING_CHANGED_OR_INCOMPLETE'):
        verify_component_asset_proposals(doc,partial)
    # The explicitly different proposal profile may use another approved
    # container type, but may not claim to preserve source drawer topology.
    single=propose_component_assets(doc,[slots],tmp_path,profile='single-joint-containers')
    assert len(single['bindings'])==1 and single['source_topology_preservation_claimed'] is False
    metadata=base/'misc/metadata.json';multi=json.loads(metadata.read_text())
    multi['link_tags']['second_drawer']=['openable'];metadata.write_text(json.dumps(multi))
    no_single=propose_component_assets(doc,[slots],tmp_path,profile='single-joint-containers')
    assert no_single['status']=='rejected' and len(no_single['failures'])==1
    assert len(propose_component_assets(doc,[slots],tmp_path)['bindings'])==1
    cell['matrix_world'][0][0]=3
    failed=propose_component_assets(doc,[slots],tmp_path)
    assert failed['status']=='rejected' and len(failed['failures'])==1
    assert failed['candidate_pruning_allowed'] is False


def test_family_binding_refuses_implicit_first_drawer_and_slot_aliases():
    from types import SimpleNamespace
    from rummagebench.authoring.tasks.household_binding import resolve_family_slots
    slots=SimpleNamespace(slots=[SimpleNamespace(parent_entity='unit_drawer_bank',slot_id=i)
                                 for i in ('upper','lower')])
    bindings=[{'location_id':'a','name':'unit_drawer_bank'}]
    with pytest.raises(ValueError,match='UNIQUE_EXPLICIT_SLOT'):
        resolve_family_slots(bindings,slots)
    bindings=[{'location_id':i,'name':'unit_drawer_bank','slot_id':j}
              for i,j in [('a','upper'),('b','lower')]]
    resolved=resolve_family_slots(bindings,slots)
    assert resolved['a'].slot_id=='upper' and resolved['b'].slot_id=='lower'
    bindings[1]['slot_id']='upper'
    with pytest.raises(ValueError,match='CANNOT_ALIAS_ONE_SLOT'):
        resolve_family_slots(bindings,slots)


def test_component_import_manifest_cannot_compile_a_partial_search_family(monkeypatch,tmp_path):
    from rummagebench.authoring.tasks import household_binding
    monkeypatch.setattr(household_binding,'load_assembly_manifest',lambda p:
        {'component_context':{'complete_leaf_binding':'pending'}})
    with pytest.raises(ValueError,match='COMPLETE_KITCHEN_LEAF_BINDING_PENDING'):
        household_binding.bind_family([object()],tmp_path/'unit_assembly.json',
                                     tmp_path/'unit_overlay.json',tmp_path/'unit_output')


def test_native_preflight_uses_valid_scenario_before_native_only_capture(monkeypatch,tmp_path):
    """Unit stop before any simulator or asset import."""
    import importlib
    from types import SimpleNamespace
    from rummagebench.authoring.tasks.household import TargetAsset
    from rummagebench.sim.omnigibson import infinigen_adapter,backend
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    cli=importlib.import_module('household_pipeline')
    spec=SimpleNamespace(base_layout_id='unit_scene',family_id='unit_family',seed=0,
        robot_config_ref='configs/tasks/robots/r1pro.yaml',
        target=TargetAsset(category='tablespoon',model='unit_only',bbox_m=[.2,.03,.01]))
    monkeypatch.setattr(cli,'load_household_spec',lambda *a:spec)
    monkeypatch.setattr(cli,'REPO',repo)
    monkeypatch.setattr(infinigen_adapter,'load_assembly_manifest',lambda *a:
        {'scene_id':'unit_scene','family_id':'unit_family','pending_checks':['real_physics']})
    monkeypatch.setattr(infinigen_adapter,'scene_slots_from_assembly',lambda *a:None)
    class UnitBackend:
        def __init__(self,**kwargs):pass
        def setup(self,scenario,**kwargs):
            assert scenario.target.entity==spec.target.internal_entity
            assert scenario.objects[0].category==spec.target.category
            assert kwargs['native_scene_only'] and callable(kwargs['native_scene_capture'])
            raise RuntimeError('UNIT_STOP_BEFORE_SIM')
        def close(self):pass
    monkeypatch.setattr(backend,'OmniGibsonBackend',UnitBackend)
    start=tmp_path/'start.json';start.write_text(json.dumps({'position':[0,0,0],'orientation':[0,0,0,1]}))
    assembly=tmp_path/'unit_assembly.json';assembly.write_text('UNIT ONLY')
    args=SimpleNamespace(spec=tmp_path/'unit_spec.json',policy=tmp_path/'unit_policy.yaml',
        assembly=assembly,start=start,out=tmp_path/'output')
    with pytest.raises(RuntimeError,match='UNIT_STOP_BEFORE_SIM'):cli.gpu_infinigen_preflight(args)
    assert json.loads((args.out/'preflight.json').read_text())['status']=='failed_preflight'


def test_restore_failure_keeps_both_snapshots_and_bitwise_diagnostic(monkeypatch,tmp_path):
    """Unit-only backend/receipts. A failed restore never receives acceptance."""
    import importlib
    from types import SimpleNamespace
    import numpy as np
    from rummagebench.authoring.tasks.household import TargetAsset
    from rummagebench.authoring.tasks import native_occupancy
    from rummagebench.sim.omnigibson import infinigen_adapter,backend,state_io
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    cli=importlib.import_module('household_pipeline')
    spec=SimpleNamespace(base_layout_id='unit_scene',family_id='unit_family',seed=0,
        robot_config_ref='configs/tasks/robots/r1pro.yaml',
        target=TargetAsset(category='tablespoon',model='unit_only',bbox_m=[.2,.03,.01]))
    monkeypatch.setattr(cli,'load_household_spec',lambda *a:spec)
    monkeypatch.setattr(cli,'REPO',repo)
    monkeypatch.setattr(infinigen_adapter,'load_assembly_manifest',lambda *a:
        {'scene_id':'unit_scene','family_id':'unit_family','pending_checks':['real_physics'],'candidate_bindings':[]})
    monkeypatch.setattr(infinigen_adapter,'scene_slots_from_assembly',lambda *a:None)
    monkeypatch.setattr(infinigen_adapter,'runtime_articulation_checks',lambda *a:[])
    original=np.array([-0.,1.],dtype=np.float32)
    monkeypatch.setattr(native_occupancy,'capture_native_occupancy',lambda *a,**kw:
        ({'source_state_sha256':native_occupancy.state_hash(original),'objects':{}},original))
    def reject(*a):raise ValueError('native source state mismatch')
    monkeypatch.setattr(native_occupancy,'verify_source',reject)
    monkeypatch.setattr(state_io,'save_snapshot',lambda path,*a,**kw:Path(path).write_text('UNIT ONLY; no simulator snapshot'))
    class UnitBackend:
        def __init__(self,**kw):self._env=SimpleNamespace(scene=SimpleNamespace(objects=[]))
        def setup(self,scenario,**kw):kw['native_scene_capture'](self)
        def validate_physics_state(self):pass
        def settle(self,*a):pass
        def _validate_spawn_clearance(self):pass
        def restore_observe_state(self,*a):pass
        def dump_state(self):return np.array([0.,1.],dtype=np.float32)
        def close(self):pass
    monkeypatch.setattr(backend,'OmniGibsonBackend',UnitBackend)
    start=tmp_path/'start.json';start.write_text(json.dumps({'position':[0,0,0],'orientation':[0,0,0,1]}))
    assembly=tmp_path/'assembly.json';assembly.write_text('UNIT ONLY')
    args=SimpleNamespace(spec=tmp_path/'spec',policy=tmp_path/'policy',assembly=assembly,
                         start=start,out=tmp_path/'output',diagnose_restore_roundtrips=0)
    with pytest.raises(ValueError,match='native source state mismatch'):cli.gpu_infinigen_preflight(args)
    assert (args.out/'captured_native_state.pt').is_file()
    assert (args.out/'failed_restored_native_state.pt').is_file()
    assert not (args.out/'native_state.pt').exists()
    delta=json.loads((args.out/'restore_failure.json').read_text())
    assert delta['changed_scalar_count']==0 and delta['bitwise_changed_scalar_count']==1
    assert delta['before_sha256']!=delta['after_sha256']
    assert json.loads((args.out/'preflight.json').read_text())['status']=='failed_preflight'


@pytest.mark.parametrize('drift',[False,True])
def test_same_checkpoint_requires_exact_repeatable_observed_world(monkeypatch,tmp_path,drift):
    """Unit reconstruction A -> B; drifting B is refused without tolerance."""
    import importlib
    from types import SimpleNamespace
    import numpy as np
    from rummagebench.authoring.tasks.household import TargetAsset
    from rummagebench.authoring.tasks import native_occupancy as occupancy
    from rummagebench.sim.omnigibson import infinigen_adapter,backend,state_io
    from rummagebench.authoring.tasks.plan import sha256_bytes,canonical_json
    repo=Path(__file__).resolve().parents[2];monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    cli=importlib.import_module('household_pipeline')
    spec=SimpleNamespace(base_layout_id='unit_scene',family_id='unit_family',seed=0,
        robot_config_ref='configs/tasks/robots/r1pro.yaml',
        target=TargetAsset(category='tablespoon',model='unit_only',bbox_m=[.2,.03,.01]))
    monkeypatch.setattr(cli,'load_household_spec',lambda *a:spec);monkeypatch.setattr(cli,'REPO',repo)
    monkeypatch.setattr(infinigen_adapter,'load_assembly_manifest',lambda *a:
        {'scene_id':'unit_scene','family_id':'unit_family','pending_checks':['physical_family'],'candidate_bindings':[]})
    monkeypatch.setattr(infinigen_adapter,'scene_slots_from_assembly',lambda *a:None)
    monkeypatch.setattr(infinigen_adapter,'runtime_articulation_checks',lambda *a:[])
    def capture(b,*a,**kw):
        state=b.dump_state();doc={'source_state_sha256':occupancy.state_hash(state),'objects':{},'counts':{}}
        doc['inventory_sha256']=sha256_bytes(canonical_json(doc));return doc,state
    monkeypatch.setattr(occupancy,'capture_native_occupancy',capture)
    def verify(doc,b,*a):
        if occupancy.state_hash(b.dump_state())!=doc['source_state_sha256']:
            raise ValueError('native source state mismatch')
    monkeypatch.setattr(occupancy,'verify_source',verify)
    monkeypatch.setattr(state_io,'save_snapshot',lambda path,*a,**kw:Path(path).write_text('UNIT ONLY; not a snapshot'))
    loads=[]
    class UnitBackend:
        def __init__(self,**kw):
            self.value=0.;self._env=SimpleNamespace(scene=SimpleNamespace(objects=[]),
                get_obs=lambda:([{'robot_0':{'head':{'rgb':np.zeros((480,854,3),dtype=np.uint8)}}}],None))
        def setup(self,scenario,**kw):kw['native_scene_capture'](self)
        def validate_physics_state(self):pass
        def settle(self,*a):pass
        def _validate_spawn_clearance(self):pass
        def restore_observe_state(self,snapshot):
            assert np.array_equal(snapshot['sim'],np.array([0.],dtype=np.float32))
            loads.append(0.);self.value=2. if drift and len(loads)>1 else 1.
            snapshot['sim'][0]=9.  # An SDK may mutate its deserialization buffer.
        def dump_state(self):return np.array([self.value],dtype=np.float32)
        def close(self):pass
        def get_observation(self):return np.zeros((480,854,3),dtype=np.uint8)
        def render_snapshot(self,path):Path(path).write_text('UNIT ONLY; not an image')
        def _camera_prim(self,sensor_name=None):
            assert sensor_name=='head'
            return SimpleNamespace(GetPath=lambda:'/unit/head/Camera')
    monkeypatch.setattr(backend,'OmniGibsonBackend',UnitBackend)
    start=tmp_path/'start.json';start.write_text(json.dumps({'position':[0,0,0],'orientation':[0,0,0,1]}))
    assembly=tmp_path/'assembly.json';assembly.write_text('UNIT ONLY')
    args=SimpleNamespace(spec=tmp_path/'spec',policy=tmp_path/'policy',assembly=assembly,start=start,
                         out=tmp_path/'output',restore_contract='same-checkpoint',diagnose_restore_roundtrips=0)
    if drift:
        with pytest.raises(ValueError,match='native source state mismatch'):cli.gpu_infinigen_preflight(args)
        assert not (args.out/'native_checkpoint.pt').exists()
    else:
        result=cli.gpu_infinigen_preflight(args)
        assert len(loads)==3 and result['native_restore_repeats']==2
        recipe=result['native_restore_contract']
        assert recipe['checkpoint_input_state_sha256']!=recipe['observed_world_state_sha256']
        assert len(set(result['native_restore_observed_hashes']))==1
        inventory=json.loads((args.out/'inventory.json').read_text())
        assert inventory['version']==2 and inventory['snapshot_file']=='native_checkpoint.pt'
        assert result['status']=='pending_simulation_acceptance' and result['evaluation_ready_episodes']==0


def test_sdk_zero_exit_cannot_pass_missing_or_failed_preflight(monkeypatch,tmp_path):
    import importlib
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    cli=importlib.import_module('household_pipeline')
    assert cli.preflight_child_result(0,tmp_path)==2
    report=tmp_path/'preflight.json';report.write_text(json.dumps({'status':'failed_preflight'}))
    assert cli.preflight_child_result(0,tmp_path)==2
    report.write_text(json.dumps({'status':'pending_simulation_acceptance','native_restore_repeats':2,
        'camera':{'sensor_name':'robot:zed_link:Camera:0','prim':'/unit/zed_link/Camera','shape':[480,854,3]},
        'completed_checks':['native_inventory','native_restore_twice','camera_rgb_shape',
                            'loaded_articulation_query','spawn_clearance']}))
    assert cli.preflight_child_result(0,tmp_path)==0
    mismatch=json.loads(report.read_text())
    mismatch['camera']['prim']='/unit/left_realsense_link/Camera'
    report.write_text(json.dumps(mismatch))
    assert cli.preflight_child_result(0,tmp_path)==2
    assert cli.preflight_child_result(139,tmp_path)==139


def test_native_integrated_handles_reuse_existing_moving_link_interface():
    from types import SimpleNamespace
    from rummagebench.feasibility.ik_solver import Pose
    from rummagebench.sim.base import ArticulationInfo,ArticulationJointInfo
    from rummagebench.sim.omnigibson.infinigen_adapter import runtime_articulation_checks
    info=ArticulationInfo('unit_cabinet',[
        ArticulationJointInfo('hinge','revolute','base','door',limits=(0.,1.5))],handle_link=None)
    backend=SimpleNamespace(articulation_info=lambda entity:info,
        link_pose=lambda entity,link:Pose([.5 if link=='door' else 0.,0.,.7],[0,0,0,1]),
        link_aabb=lambda entity,link:([.4,-.3,.4],[.6,.3,1.]))
    assert runtime_articulation_checks(backend,{'candidate_bindings':[{'name':'unit_cabinet'}]})==[]
    proof=backend._findingbench_interaction_interfaces['unit_cabinet']
    assert proof['annotated_handle_link'] is None
    assert proof['targets'][0]['source']=='fallback_link_anchor'
    backend.link_pose=lambda *a:None
    assert runtime_articulation_checks(backend,{'candidate_bindings':[{'name':'unit_cabinet'}]})


def test_loaded_distribution_retains_all_native_regions_and_open_source_cells():
    """Unit topology only; no simulator completeness or physical acceptance."""
    from copy import deepcopy
    from rummagebench.authoring.tasks.infinigen_component_binding import propose_loaded_search_distribution
    original={'family_id':'source_family','candidate_count':2,'source_layout_group':'same_house',
              'split_assignment':{'label':'dev'},'target':{},'public_task':{},
              'locations':[{'requires_open':False,'location_id':'open','component_id':'open','slot_type':'cabinet_interior'}]}
    manifest={'manifest_sha256':'unit_hash','candidate_bindings':[{'name':'bank'}],
              'component_context':{'complete_search_distribution':original}}
    row={'entity':'bank','source_component':'source_bank',
         'loaded_fillable_links':['upper','lower'],
         'loaded_regions':[{'link':v,'meta_link_type':'fillable','visual_aabb':[[0,0,z],[1,1,z+.4]]}
                           for v,z in [('upper',.5),('lower',0.)]],
         'loaded_joints':[{'parent_link':'drawer','child_link':v,'joint_type':'fixed'} for v in ('upper','lower')]
                         +[{'parent_link':'base','child_link':'drawer','joint_type':'prismatic'}]}
    result=propose_loaded_search_distribution(manifest,{'rows':[row]})
    assert result['candidate_count']==3 and sum(l['prior'] for l in result['locations'])==1
    assert result['source_layout_group']=='same_house' and result['supersedes_source_candidate_count']==2
    assert result['physical_cavity_completeness_certified'] is False and result['final_difficulty'] is None
    assert result['evaluation_ready_episodes']==0
    assert {l['fillable_link'] for l in result['locations'] if 'fillable_link' in l}=={'upper','lower'}
    assert all(l['requires_open']=='pending_articulation_reveal_measurement'
               for l in result['locations'] if 'fillable_link' in l)
    partial=deepcopy(row);partial['loaded_regions'].pop()
    with pytest.raises(ValueError,match='GEOMETRY_COVERAGE_INCOMPLETE'):
        propose_loaded_search_distribution(manifest,{'rows':[partial]})
    with pytest.raises(ValueError,match='BODY_COVERAGE_INCOMPLETE'):
        propose_loaded_search_distribution(manifest,{'rows':[]})
