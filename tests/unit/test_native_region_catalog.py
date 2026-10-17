"""UNIT ONLY: fabricated receipts exercise integrity; no physical acceptance."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from rummagebench.authoring.tasks import infinigen_component_binding as binding
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.sim.omnigibson import infinigen_adapter as adapter


@pytest.fixture
def unit_catalog(tmp_path, monkeypatch):
    assembly=tmp_path/'assembly.json'
    assembly.write_text('UNIT ONLY SOURCE ASSEMBLY')
    meta=tmp_path/'cabinet'/'unit_model'/'misc'/'metadata.json'
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({'bbox_size':[1.,1.,1.]}))
    source={'scene_id':'unit_scene','manifest_sha256':'unit_manifest_hash','family_id':'unit_source_family',
        'external_asset_sources':{str(meta):sha256_path(meta)},'objects':[],
        'candidate_bindings':[{'name':'bank','source_bbox_center_world':[0.,0.,.5],
            'og_config':{'category':'cabinet','model':'unit_model','position':[0.,0.,0.],
                         'orientation':[0.,0.,0.,1.],'scale':[1.,1.,1.]}}],
        'component_context':{'complete_leaf_binding':'pending','complete_search_distribution':{
            'family_id':'unit_source_family','candidate_count':3,'source_layout_group':'unit_house',
            'split_assignment':{'label':'dev'},'target':{},'public_task':{},'locations':[]}}}
    topology={'rows':[{'entity':'bank','source_component':'source_bank',
        'loaded_fillable_links':['upper','lower'],
        'loaded_regions':[{'link':v,'meta_link_type':'fillable','visual_aabb':[[0,0,z],[1,1,z+.4]]}
                          for v,z in [('upper',.5),('lower',0.)]],
        'loaded_joints':[{'parent_link':'drawer','child_link':v,'joint_type':'fixed'} for v in ['upper','lower']]
                        +[{'parent_link':'base','child_link':'drawer','joint_type':'prismatic'}]}]}
    root=tmp_path/'unit_preflight';root.mkdir()
    (root/'loaded_component_topology.json').write_bytes(canonical_json(topology))
    (root/'loaded_native_search_proposal.json').write_bytes(canonical_json(binding.propose_loaded_search_distribution(source,topology)))
    evidence={'status':'pending_simulation_acceptance','failures':[],
        'assembly_manifest_sha256':sha256_path(assembly),'native_restore_repeats':2,
        'camera':{'sensor_name':'robot:zed_link:Camera:0','prim':'/unit/zed_link/Camera','shape':[480,854,3]},
        'completed_checks':['native_inventory','native_restore_twice','camera_rgb_shape','loaded_articulation_query','spawn_clearance'],
        'native_region_artifacts':{n:sha256_path(root/n) for n in ['loaded_component_topology.json','loaded_native_search_proposal.json']}}
    preflight=root/'preflight.json';preflight.write_bytes(canonical_json(evidence))
    monkeypatch.setattr(adapter,'load_assembly_manifest',lambda _:deepcopy(source))
    doc=binding.bind_native_region_catalog(assembly,preflight,tmp_path/'native.json')
    return doc,source,preflight,tmp_path


def test_all_regions_get_explicit_distinct_existing_slots(unit_catalog,monkeypatch):
    doc,source,_,root=unit_catalog
    assert binding.verify_native_region_catalog(doc)['candidate_count']==2
    assert doc['candidate_bindings']==source['candidate_bindings']
    assert doc['component_context']['complete_leaf_binding']=='explicit_native_links_pending_probe'
    monkeypatch.setattr(adapter,'load_assembly_manifest',lambda _:doc)
    slots=adapter.scene_slots_from_assembly(root/'native.json')
    assert len(slots.furniture)==1 and len(slots.slots)==2
    assert {s.link for s in slots.slots}=={'upper','lower'}
    assert len({s.slot_id for s in slots.slots})==2
    assert all(s.slot_type=='drawer' and 'pending_probe' in s.geometry.source for s in slots.slots)


@pytest.mark.parametrize('mutation',['drop','alias','geometry','family','background'])
def test_rehashing_cannot_authorize_changed_mapping(unit_catalog,mutation):
    doc,_,_,_=unit_catalog
    if mutation=='drop':doc['native_region_bindings'].pop()
    elif mutation=='alias':doc['native_region_bindings'][0]['slot_id']=doc['native_region_bindings'][1]['slot_id']
    elif mutation=='geometry':doc['native_region_bindings'][0]['native_region']['visual_aabb'][0][0]=-.1
    elif mutation=='family':doc['family_id']='forged_family'
    else:doc['objects'].append({'name':'omitted_background'})
    doc['manifest_sha256']=sha256_bytes(canonical_json({k:v for k,v in doc.items() if k!='manifest_sha256'}))
    with pytest.raises(ValueError,match='MAPPING_OR_SOURCE_CHANGED'):
        binding.verify_native_region_catalog(doc)


def test_replaced_camera_receipt_cannot_be_rehashed_as_valid_import(unit_catalog):
    doc,_,preflight,_=unit_catalog
    evidence=json.loads(preflight.read_text());evidence['camera']['prim']='/unit/left_realsense_link/Camera'
    preflight.write_bytes(canonical_json(evidence))
    doc['native_region_catalog']['inputs'][str(preflight)]=sha256_path(preflight)
    with pytest.raises(ValueError,match='COMPLETE_REAL_IMPORT_PREFLIGHT'):
        binding.verify_native_region_catalog(doc)


def test_changed_real_topology_receipt_invalidates_catalog(unit_catalog):
    doc,_,preflight,_=unit_catalog
    (preflight.parent/'loaded_component_topology.json').write_text('{}')
    with pytest.raises(ValueError,match='CATALOG_INPUT_CHANGED'):
        binding.verify_native_region_catalog(doc)


def test_native_family_document_preserves_source_split_and_expands_all_locations(unit_catalog,monkeypatch):
    from test_infinigen_pipeline import make_spec
    from rummagebench.authoring.tasks.household import HouseholdTaskSpec,family_members
    doc,_,_,root=unit_catalog
    source=make_spec()
    proposal=doc['component_context']['loaded_search_distribution']
    proposal['split_assignment']=source.split_assignment.model_dump(mode='json')
    proposal['source_layout_group']=source.split_assignment.source_layout_group
    doc['component_context']['source_spec_sha256']=sha256_bytes(source.canonical_bytes())
    unit_source=root/'source_spec';unit_source.write_bytes(source.canonical_bytes())
    lineage={'version':1,'paths':{'source_spec':str(unit_source)},'hashes':{'source_spec':sha256_path(unit_source)}}
    base=HouseholdTaskSpec.model_validate(binding.native_family_document(source,doc,lineage))
    members=family_members(base)
    assert len(members)==2 and len({m.task_id for m in members})==2
    assert {m.target_location for m in members}=={b['location_id'] for b in doc['native_region_bindings']}
    assert all(m.generator_version=='household-infinigen-native-3' and m.split_assignment==source.split_assignment
               and m.target==source.target and not m.measured_metrics and m.final_difficulty is None for m in members)
    # UNIT dependencies bypass actual source assets only to verify immutable derivation.
    monkeypatch.setattr(binding,'native_source_inputs',lambda *a:(source,doc,{}))
    changed=members[0].model_copy(deep=True);changed.locations.pop()
    with pytest.raises(ValueError,match='COMPLETE_SOURCE_DERIVATION'):
        binding.validate_native_family_spec(changed,root)


def test_native_lineage_rejects_source_replacement_before_parsing(unit_catalog):
    _,_,_,root=unit_catalog
    paths={k:str(root/k) for k in ('source_spec','source_policy','native_assembly','generation_receipt')}
    for ref in paths.values():Path(ref).write_text('UNIT ONLY')
    lineage={'version':1,'paths':paths,'hashes':{k:sha256_path(v) for k,v in paths.items()}}
    Path(paths['source_spec']).write_text('REPLACED UNIT INPUT')
    with pytest.raises(ValueError,match='LINEAGE_INPUT_CHANGED'):
        binding.native_source_inputs(lineage,root)
