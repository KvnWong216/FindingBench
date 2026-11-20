"""Evidence-directed storage repairs; no task/inventory simplification.

Offline proposals only. Every output still requires native build and a fresh
frozen-state reload. Unknown geometry/semantics are deferred, never accepted.
"""
from pathlib import Path
from functools import lru_cache,wraps
import copy, hashlib, json, re
import numpy as np
import yaml
import generate_dense_pilot as g
import constructed_storage as storage
from overnight_asset_queue import storage_scope_text,has_qualified_storage_location,native_policy_signature
from audit_dense_layouts import (authoring_volumes, authoring_native_links,
                                preflight_recipe, solve_layout)


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError, ValueError):return {} if default is None else default


def serialized_native_profiles(function):
    """Both native lanes update one profile index without losing each other."""
    @wraps(function)
    def update(*args,**kwargs):
        import fcntl
        path=g.ROOT/'build/native_support_geometry/index.lock';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return function(*args,**kwargs)
    return update


def proposal_coordinate_bug(report, broken_hashes=None):
    """Known faulty proposal code cannot establish that a layout is invalid."""
    source='src/rummagebench/sim/omnigibson/inside_volume.py'
    if broken_hashes is None:
        broken_hashes=set()
        for version in ('v4','v5'):
            path=g.ROOT/f'build/generator_runtime_expert_layout_{version}_20261109'/source
            if path.exists():broken_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return report.get('runtime_sources',{}).get(source) in broken_hashes


@lru_cache(maxsize=24)
def layout_for(scene):
    return g.Layout(g.ROOT/'runs/vla_difficulty_v2/_survey'/f'{scene}.json')


def layout_for_spec(spec):
    """Replay declared fixture translations in private authoring geometry.

    Keep old occupancy as a conservative extra obstacle; never infer freed
    floor from a translated entity AABB that can overlap other scene bodies.
    Native navigation and collision checks still use the actual edited scene.
    """
    base=layout_for(spec['scene']['model'])
    manifest=spec['scene'].get('clearance_manifest')
    doc=read(manifest) if manifest else {}
    changes=doc.get('expert_scene_translations',[]);removed=doc.get('removed_objects',[])
    if not changes and not removed:return base
    if changes and hashlib.sha256(base.path.read_bytes()).hexdigest()!=doc.get('expert_base_survey_sha256'):
        raise ValueError('SCENE_TRANSLATION_SURVEY_CHANGED')
    from scipy.ndimage import label,distance_transform_edt
    result=copy.copy(base);result.entities=copy.deepcopy(base.entities);result.free=base.free.copy()
    result._authoring_geometry_identity=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()
    result.doc=dict(base.doc,footprint_boxes=list(base.doc.get('footprint_boxes',[])))
    for name in removed:result.entities.pop(name,None)
    for change in changes:
        entity=result.entities[change['entity']];delta=np.asarray(change['delta'],float)
        entity['aabb']=(np.asarray(entity['aabb'])+delta).tolist()
        for region in [entity.get('support',{}),*entity.get('interiors',[])]:
            for key in ('lo','hi'):
                if key in region:region[key]=(np.asarray(region[key])+delta).tolist()
        lo,hi=np.asarray(entity['aabb'])[:,:2];n=result.map.size
        result.doc['footprint_boxes'].append([*lo,*hi])
        r0,c0=result.map.to_map(*(lo-result.radius-.1));r1,c1=result.map.to_map(*(hi+result.radius+.1))
        r0,c0=max(0,r0),max(0,c0);r1,c1=min(n,r1+1),min(n,c1+1)
        yy,xx=np.mgrid[r0:r1,c0:c1];xx=(xx-n/2)*.1;yy=(yy-n/2)*.1
        distance=np.hypot(np.maximum(np.maximum(lo[0]-xx,0),xx-hi[0]),np.maximum(np.maximum(lo[1]-yy,0),yy-hi[1]))
        result.free[r0:r1,c0:c1] &= distance>result.radius
    result.labels,result.nlabels=label(result.free);result.room_clear=distance_transform_edt(result.free)*.1
    result.points=np.argwhere(result.free&(result.room_clear>=.2));result.cache={};result._constructed_route_cache={}
    return result


def dimensions(spec):
    result={}
    hints={p['entity']:p['orientation_hint_xyzw'] for p in spec.get('placements',[]) if p.get('orientation_hint_xyzw') is not None}
    for obj in spec['objects']:
        size=read(g.ASSETS/'objects'/obj['category']/obj['model']/'misc/metadata.json')['bbox_size']
        scale=np.asarray(obj.get('scale',1.))
        shape=np.asarray(size)*scale
        if obj['name'] in hints:
            from scipy.spatial.transform import Rotation
            shape=np.abs(Rotation.from_quat(hints[obj['name']]).as_matrix())@shape
        result[obj['name']]=shape.tolist()
    return result


def clearance_xy_delta(spec,receptacle):
    """The SDK applies declared receptacle translations to explicit XY once."""
    manifest=spec.get('scene',{}).get('clearance_manifest')
    doc=read(manifest) if manifest else {}
    return sum((np.asarray(change['delta'],float)[:2] for change in doc.get('translations',[])
                if change['entity']==receptacle),np.zeros(2))


def authoring_xy_before_clearance(spec,receptacle,world_xy):
    """Serialize a surveyed world-space proposal in the SDK's pre-edit frame."""
    return (np.asarray(world_xy,float)-clearance_xy_delta(spec,receptacle)).tolist()


def repair_double_translated_surface_pose(row,evidence,out):
    """Correct a proven twice-translated proposal, without moving any asset."""
    source=Path(row['scenario']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    report=read(Path(evidence)/'validation.json')
    if report.get('scenario_sha256')!=sha:return None
    match=re.fullmatch(r'position_xy (\[[^\]]+\]) is outside the top surface of (\S+)',str(report.get('error','')))
    if not match:return None
    observed=np.asarray(json.loads(match[1]),float);name=match[2]
    original=yaml.safe_load(source.read_text());delta=clearance_xy_delta(original,name)
    if np.linalg.norm(delta)<1e-9:return None
    layout=layout_for_spec(original);body=layout.entities.get(name,{})
    if not body.get('aabb'):return None
    lo,hi=np.asarray(body['aabb'],float)[:,:2]
    group=[p for p in original['placements'] if p['receptacle']==name and p.get('position_xy') is not None]
    matches=[p for p in group if np.allclose(np.asarray(p['position_xy'])+delta,observed,atol=1e-6)
        and np.all(np.asarray(p['position_xy'])>=lo) and np.all(np.asarray(p['position_xy'])<=hi)]
    if len(matches)!=1:return None
    spec=copy.deepcopy(original);p=next(p for p in spec['placements'] if p['entity']==matches[0]['entity'])
    world_xy=list(p['position_xy']);p['position_xy']=authoring_xy_before_clearance(spec,name,world_xy)
    assert_preserved(original,spec)
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    journal=dict(method='correct_double_translated_authoring_xy',entity=p['entity'],receptacle=name,
        source_sha256=sha,evidence=str(evidence),declared_runtime_translation_xy=delta.tolist(),
        observed_rejected_runtime_xy=observed.tolist(),desired_survey_world_xy=world_xy,
        serialized_pre_clearance_xy=p['position_xy'],physical_validation_required=True,
        preserved=['all assets and geometry','instruction','goal','inventory','robot','spawn','relation and receptacle'])
    (path.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    return dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        coordinate_frame_repairs=row.get('coordinate_frame_repairs',[])+[journal]),journal


def include_task_receptacles(layout,spec,*,evidence=None,source_sha=None):
    """Add hash-verified rigid basket geometry missing from scene surveys."""
    from audit_dense_layouts import load_geometry_index,asset_digest,offline_interior_hints
    path=g.ROOT/'build/native_fillable_geometry/index.json'
    index=load_geometry_index(path).get('assets',{}) if path.exists() else {}
    capture={}
    if evidence and source_sha and read(Path(evidence)/'validation.json').get('scenario_sha256')==source_sha:
        capture=read(Path(evidence)/'geometry.private.json')
    needed={p['receptacle'] for p in spec['placements'] if p['relation']=='inside'}
    result=copy.copy(layout);result.entities=dict(layout.entities)
    for obj in spec['objects']:
        if obj['name'] not in needed or obj['name'] in result.entities:continue
        key=f"{obj['category']}/{obj['model']}";geometry=index.get(key)
        if not geometry:
            measured=capture.get(obj['name'],{}).get('volumes',{})
            if measured:
                interiors=[]
                for link,(lo,hi) in measured.items():
                    size=np.asarray(hi)-np.asarray(lo)
                    if obj['category'] in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray'):
                        size[2]=max(0.,2*size[2]-.02)
                    if np.isfinite(size).all() and (size>0).all():
                        interiors.append(dict(link=link,lo=[0.,0.,0.],hi=size.tolist()))
                if interiors:result.entities[obj['name']]=dict(category=obj['category'],model=obj['model'],interiors=interiors,
                    offline_interior_provenance=dict(asset=key,source_sha256=source_sha,
                        capture=str(Path(evidence)/'geometry.private.json'),physical_validation_required=True,
                        open_top_centroid_height=obj['category'] in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray')))
            continue
        asset=g.ASSETS/'objects'/obj['category']/obj['model']
        if (asset_digest(asset/'misc/metadata.json')!=geometry.get('metadata_sha256') or
            asset_digest(asset/'usd'/f"{obj['model']}.encrypted.usd")!=geometry.get('encrypted_asset_sha256')):continue
        scale=np.asarray(obj.get('scale',1.));scale=np.repeat(scale,3) if scale.ndim==0 else scale
        hints=offline_interior_hints(geometry,scale,[0.,0.,0.,1.])
        if obj['category'] in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray'):
            hints=copy.deepcopy(hints)
            for hint in hints:hint['extent_m'][2]=max(0.,2*hint['extent_m'][2]-.02)
        if hints:result.entities[obj['name']]=dict(category=obj['category'],model=obj['model'],interiors=[],
            offline_interior_hints=hints,offline_interior_provenance=dict(asset=key,scale=scale.tolist(),
                open_top_centroid_height=obj['category'] in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray'),
                metadata_sha256=geometry['metadata_sha256'],encrypted_asset_sha256=geometry['encrypted_asset_sha256'],physical_validation_required=True))
    return result


def blocking_storage(spec, evidence):
    """Use the first real rejection, not subsequent skipped placements."""
    log=Path(evidence)/'native_validation.log'
    if log.exists():
        text=log.read_text(errors='replace')
        match=re.search(r'\[placement\] end (\S+) placed=False',text)
        if match:
            placement=next((p for p in spec['placements'] if p['entity']==match[1]),None)
            if placement:return placement['receptacle']
    report=read(Path(evidence)/'restore_validation.json') or read(Path(evidence)/'validation.json')
    error=str(report.get('error',''))
    nonfinite_fixture=re.search(r'/articulated__([^/]+)/[^\s]+ orientation tensor\(\[nan',error)
    if nonfinite_fixture:
        name=nonfinite_fixture[1]
        if any(p['receptacle']==name for p in spec['placements']):return name
    # These failures occur after all placement calls returned. Looking only
    # for placed=False loses the actual source and prevents a causal repair.
    missing_open=re.fullmatch(r'(\S+) has no Open state',error)
    if missing_open:return missing_open[1]
    if error.startswith('placement failed verification after physics settled:'):
        failed_open=re.search(r"['\"]([^'\"]+) open=(?:True|False) \(declared (?:True|False)\)['\"]",error)
        if failed_open:return failed_open[1]
        failed_relation=re.search(r"['\"]([^'\"]+) (?:inside|on_top) ([^'\"]+)['\"]",error)
        if failed_relation:
            placement=next((p for p in spec['placements'] if p['entity']==failed_relation[1]),None)
            if placement and placement['receptacle']==failed_relation[2]:return placement['receptacle']
    if log.exists() and any(key in error for key in ('EXPERT_CONTAINER','EXPERT_LAYOUT','EXPERT_SUPPORT','native_timeout')):
        for line in reversed(text.splitlines()):
            if '[container-settling] ' in line and 'EXPERT_CONTAINER' in error:
                try:return json.loads(line.split('[container-settling] ',1)[1])['entity']
                except (ValueError,KeyError):pass
            if '[placement]' in line and ' start ' in line:
                match=re.search(r' start (\S+) (?:inside|on_top) (\S+) link=',line)
                if match:return match[2]
    match=re.search(r'(?:POST_SETTLE_RELATION_FAILED|OPEN_STATE_CHANGED):\s*(\S+)',error)
    if match:
        placement=next((p for p in spec['placements'] if p['entity']==match[1]),None)
        return placement['receptacle'] if placement else match[1]
    windows=read(Path(evidence)/'settling.private.json',[])
    if isinstance(windows,list) and windows:
        unstable=[name for name,row in windows[-1].get('objects',{}).items()
                  if row.get('linear_speed',0)>.01 or row.get('angular_speed',0)>.1]
        for name in unstable:
            placement=next((p for p in spec['placements'] if p['entity']==name),None)
            if placement:return placement['receptacle']
    return None


def repair_kind(entity, placements, instruction):
    category=entity.get('category','')
    if category=='fridge':return 'fridge'
    # Compiler-selected links are not task semantics. A generic supplies/storage
    # request must not become a drawer-only task because of that hidden choice.
    if re.search(r'\b(drawer|drawers)\b',storage_scope_text(instruction),re.I):return 'drawer'
    if re.search(r'\b(cabinet|cabinets|cupboard|cupboards|pantry)\b',storage_scope_text(instruction),re.I):return 'cabinet'
    if category in ('bottom_cabinet','bottom_cabinet_no_top','top_cabinet','cabinet','nightstand','dresser','wardrobe','metal_bottom_cabinet'):
        return 'cabinet'
    return None


def candidate_models(kind,room=None):
    preferred={'cabinet':[('bottom_cabinet','jrhgeu')],
               'drawer':[('bottom_cabinet_no_top','ncgmqr')],
               'fridge':[('fridge','hivvdf')]}[kind]
    categories={'cabinet':['bottom_cabinet','bottom_cabinet_no_top','metal_bottom_cabinet'],
                'drawer':['bottom_cabinet_no_top','bottom_cabinet','metal_bottom_cabinet','nightstand'], 'fridge':['fridge']}[kind]
    # A one-metre mirror cannot be screened only against low kitchen units.
    # Household living-room and entryway storage can also use a tall closet;
    # keep such furniture out of unrelated kitchen/grocery source repairs.
    if kind=='cabinet' and room and g.roomkind(room) in (
            'bedroom','bathroom','office','private_office','living_room','entryway',
            'storage_room','utility_room','garage'):
        categories+=['wardrobe']
    rest=[]
    for category in categories:
        for path in (g.ASSETS/'objects'/category).glob('*/misc/metadata.json'):
            size=read(path).get('bbox_size',[])
            max_height=2.3 if kind=='fridge' or category=='wardrobe' else 1.3
            if len(size)!=3 or not (.35<size[2]<max_height) or max(size[:2])>1.8:continue
            rest.append((float(np.prod(size[:2])),category,path.parent.parent.name))
    return preferred+[(c,m) for _,c,m in sorted(rest) if (c,m) not in preferred]


def native_drawer_links(category,model):
    """A fillable link is a drawer only if its ancestry moves prismatically."""
    from audit_dense_layouts import load_geometry_index,asset_digest
    path=g.ROOT/'build/native_fillable_geometry/index.json'
    if not path.exists():return set()
    geometry=load_geometry_index(path).get('assets',{}).get(f'{category}/{model}')
    if not geometry:return set()
    root=g.ASSETS/'objects'/category/model
    if (asset_digest(root/'misc/metadata.json')!=geometry.get('metadata_sha256') or
        asset_digest(root/'usd'/f'{model}.encrypted.usd')!=geometry.get('encrypted_asset_sha256')):return set()
    parents={}
    for joint in geometry.get('joints',[]):
        for body in joint.get('body1',[]):parents.setdefault(body,[]).append(joint)
    result=set()
    for volume in geometry.get('volumes',[]):
        paths={part['prim_path'].split('/visuals/')[0] for part in volume['components']}
        valid=bool(paths)
        for body in paths:
            seen=set();prismatic=False;chain_ok=True
            while body in parents:
                if body in seen or len(parents[body])!=1:chain_ok=False;break
                seen.add(body);joint=parents[body][0]
                prismatic |= joint['type']=='PhysicsPrismaticJoint'
                if len(joint.get('body0',[]))!=1:chain_ok=False;break
                body=joint['body0'][0]
            valid &= prismatic and chain_ok
        if valid:result.add(volume['link'])
    return result


def assert_preserved(original, repaired):
    for key in ('id','instruction','goal','target','robot','anchors'):
        if original.get(key)!=repaired.get(key):raise ValueError('Semantic field changed: '+key)
    for obj in original['objects']:
        if obj not in repaired['objects']:raise ValueError('Original asset changed: '+obj['name'])
    for placement in original['placements']:
        matches=[p for p in repaired['placements'] if p['entity']==placement['entity']]
        if len(matches)!=1 or matches[0]['relation']!=placement['relation']:
            raise ValueError('Original entity or relation lost: '+placement['entity'])


def authoring_recipe_identity(spec):
    """Ignore clearance journal filenames, retain every physical scene edit."""
    doc=copy.deepcopy(spec);manifest=doc.get('scene',{}).get('clearance_manifest')
    if manifest and Path(manifest).is_file():
        clearance=read(manifest)
        doc['scene']['clearance_manifest']={key:clearance.get(key,[] if key!='scene_model' else None)
            for key in ('scene_model','wall_warps','translations','removed_objects')}
    return json.dumps(doc,sort_keys=True,separators=(',',':'))


def assert_task_inventory_preserved(original,repaired,*,editable_fixture_names=()):
    """Permit replacing failed expert fixtures, never targets or distractors."""
    contract=copy.deepcopy(original)
    new={o['name']:o for o in repaired['objects']}
    for i,obj in enumerate(contract['objects']):
        if obj in repaired['objects']:continue
        name=obj['name'];replacement=new.get(name,{})
        if ((not name.startswith('expert_storage_') and name not in editable_fixture_names) or not obj.get('fixed_base')
                or not replacement.get('fixed_base') or name==original.get('target',{}).get('entity')):
            raise ValueError('Original task asset changed: '+name)
        contract['objects'][i]=copy.deepcopy(replacement)
    assert_preserved(contract,repaired)


def editable_authored_storage(spec,row,name):
    """Only replace our own unused support fixture, preserving every task item.

    A constructed source named search_storage is just as editable as one
    named expert_storage_1. Scene furniture, targets, reference surfaces and
    delivery surfaces are protected, independently of naming conventions.
    """
    obj=next((o for o in spec['objects'] if o['name']==name),{})
    if not obj.get('fixed_base') or not any(f['name']==name for f in row.get('inserted_furniture',[])):return False
    if name in row.get('target_entities',[]) or name==spec.get('target',{}).get('entity'):return False
    if any(p.get('entity')==name or p.get('receptacle')==name for p in spec.get('goal',[])):return False
    if any(p['receptacle']==name and p['relation']!='inside' for p in spec['placements']):return False
    return True


def apply_registered_editorial_revision(row,out):
    """Apply a separately authorized, exact-source data revision, never a retry.

    Semantic changes are forbidden by the normal layout-repair contract. This
    explicit registry is the only exception: retain the original bytes and a
    complete before/after journal, then validate the revised task from scratch.
    """
    source=Path(row['scenario']);raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    registered=read(g.ROOT/'build/expert_editorial_revisions/index.json').get('tasks',{}).get(row.get('parent_task_id'))
    if not registered:return None
    decision=next((d for d in [registered,*registered.get('previous_revisions',[])] if d['original_scenario_sha256']==sha),None)
    if decision is None:return None
    if decision.get('authorization')!='user_approved_find_and_deliver_rewrite':raise ValueError('EDITORIAL_REVISION_NOT_AUTHORIZED')
    original=yaml.safe_load(raw);spec=copy.deepcopy(original)
    if original['instruction']!=decision['before_instruction']:raise ValueError('EDITORIAL_INSTRUCTION_MISMATCH')
    spec['instruction']=decision['after_instruction'];changes=[]
    for change in decision['object_changes']:
        before,after=change['before'],change['after']
        if before['name']!=after['name'] or before not in spec['objects']:raise ValueError('EDITORIAL_OBJECT_MISMATCH')
        if {k:v for k,v in before.items() if k not in ('category','model')}!={k:v for k,v in after.items() if k not in ('category','model')}:
            raise ValueError('EDITORIAL_REVISION_MAY_ONLY_REBIND_ASSET')
        asset=g.ASSETS/'objects'/after['category']/after['model']
        meta=asset/'misc/metadata.json';usd=asset/'usd'/f"{after['model']}.encrypted.usd"
        if not meta.is_file() or not usd.is_file():raise ValueError('EDITORIAL_NATIVE_ASSET_MISSING')
        spec['objects'][spec['objects'].index(before)]=copy.deepcopy(after)
        changes.append(dict(change,native_bbox_m=read(meta)['bbox_size'],
            metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),
            encrypted_asset_sha256=hashlib.sha256(usd.read_bytes()).hexdigest()))
    binding_changes=decision.get('surface_binding_changes',[])
    for change in binding_changes:
        field=change['field'];before=change['before'];after=change['after']
        if field not in ('placements','goal') or before not in spec[field]:raise ValueError('EDITORIAL_SURFACE_BINDING_MISMATCH')
        relation='relation' if field=='placements' else 'kind'
        if before.get(relation)!='on_top' or after.get(relation)!='on_top' or before.get('entity')!=after.get('entity'):
            raise ValueError('EDITORIAL_SURFACE_BINDING_MUST_RETAIN_ON_TOP_AND_ENTITY')
        allowed={'receptacle','position_xy'} if field=='placements' else {'receptacle'}
        if {k:v for k,v in before.items() if k not in allowed}!={k:v for k,v in after.items() if k not in allowed}:
            raise ValueError('EDITORIAL_SURFACE_BINDING_CHANGED_OTHER_FIELDS')
        spec[field][spec[field].index(before)]=copy.deepcopy(after)
    objects={o['name']:o for o in spec['objects']};target=objects[spec['target']['entity']]
    spec['target']['category']=target['category']
    # This revision keeps sample matching well-defined rather than introducing
    # two equally valid targets or an unrelated reference on the delivery table.
    if 'comparison_sample' in objects:
        identity=lambda o:(o['category'],o['model'])
        if identity(objects['comparison_sample'])!=identity(target):raise ValueError('EDITORIAL_SAMPLE_TARGET_MISMATCH')
        if any(identity(o)==identity(target) for n,o in objects.items() if n not in (target['name'],'comparison_sample')):
            raise ValueError('EDITORIAL_AMBIGUOUS_MATCHING_TARGET')
    editable={'instruction','objects','target'}|({'placements','goal'} if binding_changes else set())
    for key in set(original)-editable:
        if original[key]!=spec[key]:raise ValueError('EDITORIAL_UNREGISTERED_FIELD_CHANGED: '+key)
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    archive=path.parent/'original_scenario.yaml'
    if archive.exists() and archive.read_bytes()!=raw:raise ValueError('EDITORIAL_ORIGINAL_ARCHIVE_CHANGED')
    archive.write_bytes(raw);path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    furniture=copy.deepcopy(row.get('inserted_furniture',[]))
    for f in furniture:
        obj=objects.get(f['name'])
        if obj is None or (obj['category'],obj['model'])==(f['category'],f['model']):continue
        size=np.asarray(read(g.ASSETS/'objects'/obj['category']/obj['model']/'misc/metadata.json')['bbox_size'])
        floor_z=float(f['aabb'][0][2]);xy=np.asarray(f['position_xy'])
        f.update(category=obj['category'],model=obj['model'],aabb=[[*list(xy-size[:2]/2),floor_z],[*list(xy+size[:2]/2),float(floor_z+size[2])]])
    doc=dict(status='editorial_revision_physics_pending',revision_id=decision['revision_id'],
        authorization=decision['authorization'],reason=decision['reason'],original_recipe=str(source),
        original_archive=str(archive.resolve()),original_scenario_sha256=sha,
        revised_scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        before_instruction=original['instruction'],after_instruction=spec['instruction'],
        object_changes=changes,surface_binding_changes=binding_changes,
        before_goal=original.get('goal'),after_goal=spec.get('goal'),
        before_target=original['target'],after_target=spec['target'],
        preserved=['find-and-deliver paradigm','goal entity and relation','scene','spawn','robot','skills',
                   'entity names and count','placement relations','target cardinality'],
        physical_validation_required=True,human_completion_verified=False)
    result=dict(row,scenario=str(path.resolve()),scenario_sha256=doc['revised_scenario_sha256'],
        instruction=spec['instruction'],target_key=target['category']+'/'+target['model'],inserted_furniture=furniture,
        previous_binding_sha256=row.get('binding_sha256'),
        binding_sha256=hashlib.sha256(json.dumps(spec['objects'],sort_keys=True).encode()).hexdigest(),
        editorial_revisions=row.get('editorial_revisions',[])+[doc],physics_status='pending',
        status='editorial_revision_physics_pending')
    if decision.get('destination_receptacle'):
        result['destination_receptacle']=decision['destination_receptacle']
        layout=storage.replay_layout(layout_for_spec(spec),furniture)
        route=layout.route(result['source_receptacle'],result['source_room'],
                           result['destination_receptacle'],result['destination_room'])
        if route is None and binding_changes:
            reference=next((p for p in spec['placements'] if p['entity']==row.get('reference_entity')
                            and p['relation']=='on_top' and p['receptacle']==result['destination_receptacle']),None)
            if reference is not None:
                # A broad countertop AABB's near edge can be <55 cm from a
                # valid approach while its actual reference point is reachable.
                # Route to a real point on the unchanged surface, using the
                # same free-cell map and connected-component/path checks.
                name=result['destination_receptacle'];surface=layout.entities[name]
                lo,hi=np.asarray(surface['aabb'],float);xy=(lo[:2]+hi[:2])/2
                point=copy.copy(layout);point.entities=dict(layout.entities);point.cache={}
                point.entities[name]=dict(surface,aabb=[[*xy,float(lo[2])],[*xy,float(hi[2])]])
                route=point.route(result['source_receptacle'],result['source_room'],name,result['destination_room'])
                doc['destination_route_surface_point_xy']=xy.tolist()
        if route is None:raise ValueError('EDITORIAL_SURFACE_DESTINATION_HAS_NO_ROUTE_PROPOSAL')
        result['route']=route
    if row.get('adaptation'):
        result['adaptation']=dict(row['adaptation'],after_instruction=spec['instruction'],
            target_asset=result['target_key'],target_assets=[objects[n]['category']+'/'+objects[n]['model'] for n in row['target_entities']],
            editorial_revision_id=decision['revision_id'])
    (path.parent/'editorial_revision.json').write_text(json.dumps(doc,indent=2))
    return result,doc


def storage_site_key(scene,name,entity):
    return hashlib.sha256(json.dumps([scene,name,entity.get('category'),entity.get('model'),
        np.round(np.asarray(entity['aabb'],float),6).tolist()],sort_keys=True).encode()).hexdigest()


@serialized_native_profiles
def learn_verified_storage_sites(row,evidence):
    """Proposal preference only; caller must first verify the full native pair."""
    spec=yaml.safe_load(Path(row['scenario']).read_text())
    layout=layout_for_spec(spec);spawned={o['name']:o for o in spec['objects']}
    path=g.ROOT/'build/verified_storage_sites/index.json';index=read(path,dict(sites={}))
    # Open shelving has the same evidence requirement as closed storage.
    # Preserve the original native pair and treat it only as model preference
    # for future inventories, never a certificate that a new layout is valid.
    surface_path=g.ROOT/'build/verified_surface_sites/index.json'
    surfaces=read(surface_path,dict(schema_version=1,assets={}))
    for name in {p['receptacle'] for p in spec['placements'] if p['relation']=='on_top'}:
        asset=spawned.get(name) or layout.entities.get(name)
        if not asset or asset.get('category') not in ('shelf','bookcase','desk','table','breakfast_table','coffee_table',
                'dining_table','countertop','bottom_cabinet','bottom_cabinet_no_top','metal_bottom_cabinet','dresser'):continue
        count=sum(p['relation']=='on_top' and p['receptacle']==name for p in spec['placements'])
        key=asset['category']+'/'+asset['model'];prior=surfaces['assets'].get(key,{})
        if count>=prior.get('validated_prop_count',0):
            surfaces['assets'][key]=dict(prior,recipe=row['scenario'],evidence=str(evidence),
                validated_prop_count=count,scope='Positive native example; new contents require independent native initialization and reload')
        group=[p for p in spec['placements'] if p['relation']=='on_top' and p['receptacle']==name]
        if len(group)>=2 and all(p.get('position_xy') is None for p in group):
            surfaces['assets'].setdefault(key,{}).update(native_sampling_evidence=dict(
                recipe=row['scenario'],evidence=str(evidence),validated_prop_count=count))
    if surfaces!=read(surface_path):
        surface_path.parent.mkdir(parents=True,exist_ok=True);tmp=surface_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(surfaces,indent=2));tmp.replace(surface_path)
    for name in {p['receptacle'] for p in spec['placements'] if p['relation']=='inside'}:
        asset=spawned.get(name) or layout.entities.get(name)
        if asset and asset.get('category') and asset.get('model'):
            key=asset['category']+'/'+asset['model']
            index.setdefault('asset_successes',{})[key]=dict(accepted_evidence=str(evidence),
                recipe_sha256=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest(),
                scope='packing preference only; new geometry, pose and contents require independent native validation')
        # Constructed fixtures have recipe-specific positions. Only surveyed
        # native fixtures can be preferred across different task recipes.
        if name in spawned or name not in layout.entities:continue
        key=storage_site_key(spec['scene']['model'],name,layout.entities[name])
        index['sites'][key]=dict(scene=spec['scene']['model'],receptacle=name,category=layout.entities[name]['category'],
            model=layout.entities[name]['model'],accepted_evidence=str(evidence),recipe_sha256=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest(),
            scope='same surveyed fixture pose; proposal preference only; every new task requires native initialization and reload')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(index,indent=2));tmp.replace(path)


@serialized_native_profiles
def learn_opened_furniture_footprints(row,evidence):
    """Keep observed opening envelopes, even when a later native gate fails."""
    recipe=Path(row['scenario']);report=read(Path(evidence)/'validation.json')
    if not recipe.is_file() or report.get('scenario_sha256')!=hashlib.sha256(recipe.read_bytes()).hexdigest():return
    log=Path(evidence)/'native_validation.log'
    if not log.exists() or proposal_coordinate_bug(report):return
    spec=yaml.safe_load(recipe.read_text());objects={o['name']:o for o in spec['objects']}
    furniture={f['name']:f for f in row.get('inserted_furniture',[])};opened=set();observed=[]
    for line in log.read_text(errors='replace').splitlines():
        try:
            if '[open-state-audit] ' in line:
                event=json.loads(line.split('[open-state-audit] ',1)[1])
                if event.get('stage')=='after_set':
                    if event.get('requested_open') and event.get('actual_open'):opened.add(event['entity'])
                    else:opened.discard(event['entity'])
            if '[expert-pose-result] ' not in line:continue
            event=json.loads(line.split('[expert-pose-result] ',1)[1]);name=event['entity']
            obj=objects.get(name,{});site=furniture.get(name)
            if not site or name not in opened or not obj.get('fixed_base') or not event.get('placed'):continue
            if not np.allclose(obj.get('scale',1.),1.):continue
            q=np.asarray(event.get('orientation',[]),float)
            if q.shape!=(4,) or np.linalg.norm(q[:3])>.001 or abs(abs(q[3])-1)>.001:continue
            bounds=np.asarray(event['aabb'],float)[:,:2]-np.asarray(site['position_xy'],float)
            if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[0]>=bounds[1]):continue
            bounds=np.array([np.floor(bounds[0]*100)/100,np.ceil(bounds[1]*100)/100])
            observed.append((obj,bounds))
        except (ValueError,KeyError,TypeError):continue
    if not observed:return
    from audit_dense_layouts import asset_digest
    path=g.ROOT/'build/native_support_geometry/index.json';index=read(path,dict(assets={},physical_validation_required=True))
    for obj,bounds in observed:
        key=obj['category']+'/'+obj['model'];asset=g.ASSETS/'objects'/obj['category']/obj['model']
        entry=index['assets'].setdefault(key,{})
        metadata_sha=asset_digest(asset/'misc/metadata.json')
        encrypted_sha=asset_digest(asset/'usd'/f"{obj['model']}.encrypted.usd")
        if entry.get('metadata_sha256')!=metadata_sha or entry.get('encrypted_asset_sha256')!=encrypted_sha:
            entry={};index['assets'][key]=entry
        prior=entry.get('opened_footprint_offsets_m')
        if prior:
            prior=np.asarray(prior,float);bounds=np.array([np.minimum(bounds[0],prior[0]),np.maximum(bounds[1],prior[1])])
        entry.update(metadata_sha256=metadata_sha,encrypted_asset_sha256=encrypted_sha,
            opened_footprint_offsets_m=bounds.tolist(),opening_envelope_evidence=str(log),
            opening_envelope_scope='Measured identity-yaw native opening; conservative authoring screen, not a contact certificate')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(index,indent=2));tmp.replace(path)


@serialized_native_profiles
def learn_native_support_profiles(row,evidence):
    """Transfer measured unit-scale fixture geometry as proposal evidence only.

    Surveyed fixtures may have scene-specific scale, so only unchanged native
    fixed assets whose explicit floor placement succeeded are eligible.
    """
    recipe=Path(row['scenario']);report=read(Path(evidence)/'validation.json')
    if not recipe.is_file() or not report.get('scenario_sha256'):return
    if report.get('scenario_sha256')!=hashlib.sha256(recipe.read_bytes()).hexdigest():return
    log=Path(evidence)/'native_validation.log'
    if not log.exists() or proposal_coordinate_bug(report):return
    spec=yaml.safe_load(recipe.read_text());poses={};planes={};patches={};planned=set();problems={}
    for line in log.read_text(errors='replace').splitlines():
        if '[expert-layout-problem] ' in line:
            try:
                problem=json.loads(line.split('[expert-layout-problem] ',1)[1]);problems.setdefault(problem['receptacle'],problem)
            except (ValueError,KeyError):pass
        if '[expert-layout-plan] ' in line:
            try:planned.add(json.loads(line.split('[expert-layout-plan] ',1)[1])['receptacle'])
            except (ValueError,KeyError):pass
        for marker,dest,key in [('[expert-pose-result] ',poses,'entity'),
                                ('[expert-support-plane] ',planes,'receptacle'),
                                ('[expert-support-patch] ',patches,'receptacle')]:
            if marker not in line:continue
            try:event=json.loads(line.split(marker,1)[1])
            except ValueError:continue
            if dest is poses:dest[event[key]]=event
            else:dest.setdefault(event[key],[]).append(event)
    path=g.ROOT/'build/native_support_geometry/index.json';index=read(path,dict(assets={},physical_validation_required=True))
    from audit_dense_layouts import asset_digest
    changed=False
    for obj in spec['objects']:
        name=obj['name'];pose=poses.get(name,{})
        if not obj.get('fixed_base') or not pose.get('placed') or name not in planes:continue
        if not np.allclose(obj.get('scale',1.),1.):continue
        q=np.asarray(pose.get('orientation',[1,0,0,0]))
        if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q[:3])>.001 or abs(abs(q[3])-1)>.001:continue
        placement=next((p for p in spec['placements'] if p['entity']==name),{})
        if placement.get('position_xy') is None or placement.get('relation')!='on_top':continue
        components={}
        for event in planes[name]:
            component=event['component'];entry={k:event[k] for k in ['status','reason','floor_minus_fillable_bottom_m','height_spread_m','measured_height_range_m'] if k in event}
            entry['support_search_complete']=name in planned
            measured=[p for p in patches.get(name,[]) if p['component'].startswith(component+'#support_patch:')]
            if measured:
                entry['patches']=[dict(extent_m=(np.asarray(p['hi'])-np.asarray(p['lo'])).tolist(),
                    floor_minus_fillable_bottom_m=p['floor_minus_fillable_bottom_m']) for p in measured]
            components[component]=entry
        asset=g.ASSETS/'objects'/obj['category']/obj['model']
        metadata_sha=asset_digest(asset/'misc/metadata.json');usd_sha=asset_digest(asset/'usd'/f"{obj['model']}.encrypted.usd")
        old_entry=index['assets'].get(f"{obj['category']}/{obj['model']}",{})
        if old_entry.get('metadata_sha256')!=metadata_sha or old_entry.get('encrypted_asset_sha256')!=usd_sha:old_entry={}
        index['assets'][f"{obj['category']}/{obj['model']}"]=dict(old_entry,
            metadata_sha256=metadata_sha,encrypted_asset_sha256=usd_sha,
            scope='native scale; world identity orientation; opened fixture; proposal screening only',
            components=components,evidence=str(log),recipe_sha256=report['scenario_sha256'],runtime_sources=report.get('runtime_sources',{}))
        entry=index['assets'][f"{obj['category']}/{obj['model']}"];problem=problems.get(name,{})
        # Native normalized convex domains include panels, actual support
        # heights and shared cavities that outer metadata boxes can miss.
        # Reuse only a unit-scale fixed fixture's FIRST (empty) problem.
        # Keep a stable profile; a subsequently accepted native pair may
        # replace an unaccepted proposal profile once, not on every log write.
        valid_domain=(problem.get('volumes') and all(np.asarray(v).shape==(3,) and np.isfinite(v).all() and (np.asarray(v)>0).all() for v in problem['volumes'].values())
            and all(np.isfinite(values).all() for field in ['center_halfspaces','obstacles'] for values in problem.get(field,{}).values()))
        if valid_domain and not any('#on:' in key for key in problem['volumes']):
            from track_full_asset_queue import accepted_native_pair
            positive=accepted_native_pair(recipe,Path(evidence))
            previous=entry.get('native_proposal_domains',{})
            if not previous or positive and not previous.get('accepted_native_pair'):
                supports={event['component']:{k:event[k] for k in ['status','floor_minus_fillable_bottom_m','height_spread_m','measured_height_range_m'] if k in event}
                          for event in [*planes[name],*patches.get(name,[])]}
                if all(key in supports and 'floor_minus_fillable_bottom_m' in supports[key] for key in problem['volumes']):
                    entry['native_proposal_domains']=dict(
                        volumes=problem['volumes'],native_links={key:key.split('#',1)[0] for key in problem['volumes']},
                        center_halfspaces=problem.get('center_halfspaces',{}),obstacles=problem.get('obstacles',{}),
                        overlap_groups=problem.get('overlap_groups',{}),supports=supports,
                        accepted_native_pair=positive,source_recipe_sha256=report['scenario_sha256'],evidence=str(log),
                        scope='Unit-scale identity-yaw empty native fixture; proposal constraints only, fresh native physics required')
        changed=True
    if changed:
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(index,indent=2));tmp.replace(path)


def compartment_assignment(entity, items, kind, *, native_size=False, allow_patch_support=False,allow_measured_empty=False,single_drawer=True,support_entities=()):
    """Screen dimensions, then jointly pack the complete unchanged inventory."""
    bins=authoring_volumes(entity);links=authoring_native_links(entity)
    support={};domains={}
    if native_size:
        profile=read(g.ROOT/'build/native_support_geometry/index.json').get('assets',{}).get(
            f"{entity['category']}/{entity['model']}",{})
        if profile:
            from audit_dense_layouts import asset_digest
            root=g.ASSETS/'objects'/entity['category']/entity['model']
            if (asset_digest(root/'misc/metadata.json')==profile.get('metadata_sha256') and
                asset_digest(root/'usd'/f"{entity['model']}.encrypted.usd")==profile.get('encrypted_asset_sha256')):
                support=profile.get('components',{})
                domains=profile.get('native_proposal_domains',{})
                if domains.get('volumes'):
                    bins=copy.deepcopy(domains['volumes']);links=dict(domains['native_links']);support=dict(domains['supports'])
                else:domains={}
        if support and not domains:
            filtered={}
            for key,size in bins.items():
                report=support.get(key.replace('#component:','#mesh:'))
                if report is None or report.get('status')!='measured_plane':
                    if allow_patch_support and report and report.get('patches'):
                        for i,patch in enumerate(report['patches']):
                            patch_key=key+f'#support_patch:{i}'
                            filtered[patch_key]=list(patch['extent_m']);links[patch_key]=links[key]
                            support[patch_key.replace('#component:','#mesh:')]=dict(
                                status='measured_plane',floor_minus_fillable_bottom_m=patch['floor_minus_fillable_bottom_m'])
                    elif allow_patch_support and (allow_measured_empty or not report or not report.get('support_search_complete')):
                        filtered[key]=list(size)
                    continue
                size=list(size);size[2]-=max(0.,report['floor_minus_fillable_bottom_m'])
                if size[2]>0:filtered[key]=size
            bins=filtered
    if kind=='drawer':
        verified=native_drawer_links(entity['category'],entity['model']) if entity.get('category') and entity.get('model') else None
        bins={k:v for k,v in bins.items() if links[k] in verified or any(
            link.startswith(f'meta__{links[k]}_') for link in verified)} if verified is not None else {
            k:v for k,v in bins.items() if 'drawer' in links[k]}
    # One drawer can expose several disconnected fillable components. Keep a
    # group in one native drawer link, not artificially in one mesh component.
    alternatives=[{key:value for key,value in bins.items() if links[key]==link}
                  for link in dict.fromkeys(links[key] for key in bins)] if kind=='drawer' and single_drawer else [bins]
    if support:
        known={key:value for key,value in bins.items() if support.get(key.replace('#component:','#mesh:'),{}).get('status')=='measured_plane'}
        if kind=='drawer' and single_drawer:
            alternatives.sort(key=lambda group:not all(key in known for key in group))
        elif known and len(known)!=len(bins):alternatives=[known,*alternatives]
    for allowed in alternatives:
        if not allowed:continue
        if any(not any(v[2]>=size[2]+.005 and np.all(np.sort(v[:2])>=np.sort(size[:2])+.005)
                       for v in allowed.values()) for size in items.values()):continue
        fixed={n:{key for key in allowed if size[2]/2>=max(0.,-support.get(
            key.replace('#component:','#mesh:'),{}).get('floor_minus_fillable_bottom_m',0.))+
            max(0.,support.get(key.replace('#component:','#mesh:'),{}).get('height_spread_m',0.))+.00001}
            for n,size in items.items()} if support else None
        constraints={field:{key:value for key,value in domains.get(field,{}).items() if key in allowed}
                     for field in ['center_halfspaces','obstacles','overlap_groups']} if domains else {}
        deferred=[name for name in items if fixed is not None and not fixed[name]]
        anchors=set(support_entities)-set(deferred)
        if deferred and not anchors:continue
        packed_items={name:size for name,size in items.items() if name not in deferred}
        packed_fixed={name:choices for name,choices in fixed.items() if name in packed_items} if fixed is not None else None
        if domains:
            # Six native components and both orientations can exceed the
            # old 1-M candidate guard before any search. Start with a
            # conservative 5-mm witness; use the native 4-M budget on the
            # fine grid when needed. A coarse miss is never infeasibility.
            solved=solve_layout(packed_items,allowed,packed_fixed,grid_m=.005,margin_m=.015,
                time_limit_s=1.,max_candidates=4000000,**constraints)
            if solved['status']!='feasible_model':
                coarse=dict(status=solved['status'],reason=solved.get('reason'))
                solved=solve_layout(packed_items,allowed,packed_fixed,grid_m=.001,margin_m=.015,
                    time_limit_s=1.,max_candidates=4000000,**constraints)
                solved=dict(solved,coarse_proposal_attempt=coarse)
        else:solved=solve_layout(packed_items,allowed,packed_fixed,margin_m=.015,time_limit_s=1.,**constraints)
        if solved['status']!='feasible_model':
            preferred=dict(status=solved['status'],reason=solved.get('reason'))
            # Match the reviewed live planner's bounded authoring-spacing
            # fallback. This does not alter contact, support or safety gates.
            solved=solve_layout(packed_items,allowed,packed_fixed,margin_m=.005,
                time_limit_s=1.,max_candidates=4000000,**constraints)
            solved=dict(solved,preferred_spacing_attempt=preferred,
                physical_validation_required=True)
        if solved['status']=='feasible_model':
            if deferred:
                assignment=dict(solved['assignment']);selected={}
                for name in deferred:
                    candidates=[support for support in sorted(anchors) if support in assignment and
                        items[support][2]+items[name][2]+.015<=allowed[assignment[support]][2] and
                        np.all(np.sort(items[name][:2])+.015<=np.sort(allowed[assignment[support]][:2])) and
                        min(items[support][:2])>=.025]
                    if not candidates:break
                    selected[name]=candidates[0];assignment[name]=assignment[candidates[0]]
                else:
                    solved=dict(solved,assignment=assignment,deferred_native_stacking=selected,
                        model_scope='Base-layer box witness only; deferred props require actual measured native support and fresh physical validation',
                        physical_validation_required=True)
                    return {n:links[b] for n,b in assignment.items()},solved
                continue
            return {n:links[b] for n,b in solved['assignment'].items()},solved
    return None


def flat_support_entities(spec,group):
    eligible={p['entity'] for p in group}
    categories={'cutting_board','chopping_board','baking_sheet','notebook','hardback','paperback_book','clipboard'}
    return {o['name'] for o in spec['objects'] if o['name'] in eligible and o.get('category') in categories}


def storage_inventory_signature(spec,group):
    objects={o['name']:o for o in spec['objects']}
    inventory=[{k:v for k,v in objects[p['entity']].items() if k!='name'} for p in group]
    return hashlib.sha256(json.dumps(sorted(inventory,key=lambda o:json.dumps(o,sort_keys=True)),sort_keys=True).encode()).hexdigest()


def observed_storage_model_rejection(row,evidence,spec,old,group,entity):
    """Record a task-local proposal preference, never a global asset ban.

    Only a hash-matched exhaustive live-layout rejection with no compiler link
    restriction qualifies. Timeouts and search-budget UNKNOWN are not evidence
    of insufficient geometry and must not penalize a native model.
    """
    report=read(Path(evidence)/'validation.json');source=Path(row['scenario'])
    if report.get('scenario_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():return None
    if proposal_coordinate_bug(report) or any(p.get('link') for p in group):return None
    log=Path(evidence)/'native_validation.log'
    if not log.exists():return None
    events=[]
    for line in log.read_text(errors='replace').splitlines():
        if '[expert-layout-plan] ' not in line:continue
        try:events.append(json.loads(line.split('[expert-layout-plan] ',1)[1]))
        except ValueError:pass
    if not events or events[-1].get('receptacle')!=old or events[-1].get('status')!='infeasible_model':return None
    return dict(inventory_sha256=storage_inventory_signature(spec,group),
        asset=entity['category']+'/'+entity['model'],reason=events[-1].get('reason'),
        native_policy_signature=native_policy_signature(report.get('runtime_sources',{})),
        evidence=str(log),scenario_sha256=report['scenario_sha256'],
        scope='same task and unchanged inventory; try another model first, never exclude the asset globally')


def declared_source_anchor(spec,layout,name):
    """Return a real surveyed ancestor for local authoring, never invent a pose."""
    seen=set();current=name
    while current not in layout.entities:
        if current in seen:return None
        seen.add(current)
        placement=next((p for p in spec['placements'] if p['entity']==current),None)
        if placement is None:return None
        current=placement['receptacle']
    return current


def reuse_existing_storage(original, row, base, old, group, room, kind, items, out, evidence, *, allow_partial=False,origin_xy=None,allowed_categories=None):
    """Fit stock jointly into existing storage before consuming floor space.

    A same-room storage choice is authoring, never assistance to an agent.
    Explicit spatial qualifiers have already been deferred by curate().
    """
    categories={'cabinet':{'bottom_cabinet','bottom_cabinet_no_top','top_cabinet','cabinet'},
                'drawer':{'bottom_cabinet','bottom_cabinet_no_top','nightstand','dresser'},
                'fridge':{'fridge'}}[kind]
    if kind=='cabinet' and g.roomkind(room) in ('bedroom','bathroom','office','private_office',
            'living_room','entryway','storage_room','utility_room','garage'):
        categories|={'wardrobe','metal_bottom_cabinet'}
    if allowed_categories is not None:categories &= set(allowed_categories)
    occupied={p['receptacle'] for p in original['placements'] if p['relation']=='inside' and p['receptacle']!=old}
    protected={row['destination_receptacle'],old}
    protected.update(p.get('entity') for p in original.get('goal',[]))
    protected.update(p.get('receptacle') for p in original.get('goal',[]))
    if any((p.get('receptacle')==old or p.get('entity')==old) and p.get('kind')!='on_top'
           for p in original.get('goal',[])):return None
    spawn=original['anchors'][original['robot']['init_anchor']]['position'];rr,cc=base.map.to_map(*spawn[:2])
    if not (0<=rr<base.free.shape[0] and 0<=cc<base.free.shape[1] and base.free[rr,cc]):return None
    candidates=[]
    anchor=declared_source_anchor(original,base,old)
    if anchor is None:return None
    old_entity=base.entities.get(old) or next((o for o in original['objects'] if o['name']==old),{})
    for name,entity in base.entities.items():
        if (name in protected or entity.get('category') not in categories or room not in entity.get('in_rooms',[])
                or original.get('initial_states',{}).get(name,{}).get('open') is True):continue
        origin=np.asarray(origin_xy) if origin_xy is not None else g.boxcenter(base.entities[anchor])[:2]
        distance=float(np.linalg.norm(g.boxcenter(entity)[:2]-origin))
        if distance<=4.:candidates.append((distance,name,entity))
    known=read(g.ROOT/'build/verified_storage_sites/index.json').get('sites',{})
    scene_name=original.get('scene',{}).get('model',getattr(base,'scene',''))
    for _,name,entity in sorted(candidates,key=lambda x:(storage_site_key(scene_name,x[1],x[2]) not in known,x[0],x[1])):
        if not storage.model_openable(entity['category'],entity['model']):continue
        destination_anchor=scene_route_anchor(original,row['destination_receptacle'],base.entities)
        route=base.route(name,room,destination_anchor,row['destination_room'])
        if route is None:continue
        sr,sc=base.map.to_map(*route['start'][:2])
        if base.labels[rr,cc]!=base.labels[sr,sc]:continue
        scope=dict(single_drawer=bool(re.search(r'\bdrawer\b',storage_scope_text(original['instruction']),re.I))) if kind=='drawer' else {}
        # Already occupied storage is usable only if the COMBINED contents
        # fit. Never test a new group against a fictitiously empty cupboard.
        existing=[p['entity'] for p in original['placements'] if p['relation']=='inside' and p['receptacle']==name]
        joint_items=dict(items)
        if existing:
            sizes=dimensions(original)
            joint_items.update({n:sizes[n] for n in existing})
        packed=compartment_assignment(entity,joint_items,kind,**scope)
        if packed is None:continue
        assignment,solution=packed;spec=copy.deepcopy(original)
        for p in spec['placements']:
            if p['entity'] in assignment:p.update(receptacle=name,link=assignment[p['entity']])
        # An offline pack is a size witness, not a language constraint on the
        # live solver. Its arbitrary link must not prevent the native planner
        # from choosing another physically supported shelf in this cabinet.
        released=clear_unstated_links(spec)
        if not storage.model_openable(old_entity['category'],old_entity['model']):
            spec.get('initial_states',{}).pop(old,None)
        spec.setdefault('initial_states',{})[name]={'open':False}
        pf=preflight_recipe(spec,base.entities,dimensions(spec))
        if pf['status']!='heuristic_fit' and not allow_partial:continue
        assert_preserved(original,spec)
        path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
        result=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            layout_preflight=pf,restoration_entities=list(dict.fromkeys(row.get('restoration_entities',[])+[name])))
        if row.get('source_receptacle')==old and any(n in assignment for n in row.get('target_entities',[original.get('target',{}).get('entity')])):
            result.update(source_receptacle=name,route=route)
        doc=dict(status='static_screened_physics_pending' if pf['status']=='heuristic_fit' else 'static_partial_not_native_ready',old_storage=old,reused_native_storage=name,
            existing_contents_retained=existing,joint_packed_entities=sorted(joint_items),
            released_compiler_links=released,
            original_recipe=row['scenario'],diagnosis_directory=str(evidence),solver=solution,
            preserved=['instruction','goal','target','robot','spawn','all original assets','furniture geometry','relation types'])
        (path.parent/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
        return result,doc
    return None


def prune_unused_authoring_fixtures(row,evidence,out):
    """Discard only unreferenced authoring scaffolds in a failed candidate.

    The original scene occupancy stays intact. These objects were added after
    the survey, so omitting them from replay restores the measured baseline;
    it never infers free floor by subtracting a scene object's bounding box.
    Original recipes and native asset files are retained.
    """
    from track_full_asset_queue import accepted_native_pair
    if accepted_native_pair(Path(row['scenario']),Path(evidence)):return None
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text())
    if has_qualified_storage_location(original['instruction']) or original.get('temporal'):return None
    protected={row.get('source_receptacle'),row.get('destination_receptacle'),row.get('reference_entity'),
               original.get('target',{}).get('entity'),*row.get('target_entities',[]),
               *original.get('initial_states',{})}
    protected.update(p['receptacle'] for p in original['placements'])
    protected.update(p.get('entity') for p in original.get('goal',[]))
    protected.update(p.get('receptacle') for p in original.get('goal',[]))
    authored={f['name'] for f in row.get('inserted_furniture',[])}
    candidates=[o['name'] for o in original['objects'] if o.get('fixed_base')
                and o['name'] in authored and o['name'] not in protected]
    if not candidates:return None
    base=layout_for_spec(original)
    removed=[name for name in candidates if name not in base.entities]
    if not removed:return None
    spec=copy.deepcopy(original)
    spec['objects']=[o for o in spec['objects'] if o['name'] not in removed]
    spec['placements']=[p for p in spec['placements'] if p['entity'] not in removed]
    contract=copy.deepcopy(original);contract['objects']=copy.deepcopy(spec['objects']);contract['placements']=copy.deepcopy(spec['placements'])
    assert_preserved(contract,spec)
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    journal=dict(method='remove_unreferenced_failed_authoring_fixtures',removed_entities=removed,
                 removed_objects=[o for o in original['objects'] if o['name'] in removed],
                 removed_placements=[p for p in original['placements'] if p['entity'] in removed],
                 original_recipe=str(source),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 reason='Unused added fixtures from abandoned repairs occupy floor but carry no props, goals, declared states or task binding.',
                 original_scene_occupancy_retained=True,physical_validation_required=True)
    (path.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    revised=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inserted_furniture=[f for f in row.get('inserted_furniture',[]) if f['name'] not in removed],
        restoration_entities=[n for n in row.get('restoration_entities',[]) if n not in removed],
        object_count=len(spec['objects']),authoring_fixture_cleanup=row.get('authoring_fixture_cleanup',[])+[journal])
    return revised,spec,journal


def curate(row, evidence, out, *, variant=0, old_storage=None, allow_partial=False,
           group_entities=None, allow_patch_support=False, prefer_constructed=False,exhaustive_models=False,allowed_models=()):
    """Replace one diagnosed storage group; preserve all semantic fields."""
    original=yaml.safe_load(Path(row['scenario']).read_text())
    old=old_storage or blocking_storage(original,evidence)
    group=[p for p in original['placements'] if p['receptacle']==old and p['relation']=='inside' and
           (group_entities is None or p['entity'] in group_entities)]
    if not group:
        return None,dict(status='deferred',reason='First blocker is not an inside-storage group',storage=old)
    # An automatic repair must not reinterpret an explicitly selected location.
    if has_qualified_storage_location(original['instruction']):
        return None,dict(status='deferred',reason='Explicit spatial qualifier requires manual mapping',storage=old)
    cleaned=prune_unused_authoring_fixtures(row,evidence,Path(out)/'_without_orphans')
    if cleaned is not None:
        revised,_,journal=cleaned
        result,info=curate(revised,evidence,out,variant=variant,old_storage=old,allow_partial=allow_partial,
            group_entities=group_entities,allow_patch_support=allow_patch_support,
            prefer_constructed=prefer_constructed,exhaustive_models=exhaustive_models,allowed_models=allowed_models)
        return result,dict(info,authoring_fixture_cleanup=journal)
    base=storage.replay_layout(layout_for_spec(original),row.get('inserted_furniture',[]))
    anchor=declared_source_anchor(original,base,old)
    if anchor is None:return None,dict(status='deferred',reason='Source has no declared surveyed ancestor',storage=old)
    old_entity=base.entities.get(old) or next((o for o in original['objects'] if o['name']==old),{})
    observed=read(Path(evidence)/'validation.json')
    native_missing_open=(observed.get('scenario_sha256')==hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest()
                         and observed.get('error')==f'{old} has no Open state')
    old_center=g.boxcenter(base.entities[anchor])
    room=(base.entities[anchor].get('in_rooms') or [row['source_room']])[0]
    kind=repair_kind(old_entity,group,original['instruction'])
    if kind is None:return None,dict(status='deferred',reason='No semantics-preserving storage profile',storage=old)
    sizes=dimensions(original);items={p['entity']:sizes[p['entity']] for p in group}
    experience=list(row.get('storage_model_experience',[]))
    rejection=observed_storage_model_rejection(row,evidence,original,old,group,old_entity)
    if rejection and not any((r.get('inventory_sha256'),r.get('asset'),r.get('native_policy_signature'))==
                            (rejection['inventory_sha256'],rejection['asset'],rejection.get('native_policy_signature')) for r in experience):experience.append(rejection)
    row=dict(row,storage_model_experience=experience)
    inventory_signature=storage_inventory_signature(original,group)
    policy_signature=row.get('proposal_native_policy_signature') or native_policy_signature(
        read(Path(evidence)/'validation.json').get('runtime_sources',{}))
    rejected_models={r['asset'] for r in experience if r.get('inventory_sha256')==inventory_signature
                     and r.get('native_policy_signature')==policy_signature}
    # An earlier failed intervention is not immutable scene furniture. Keeping
    # every abandoned expert cabinet permanently blocks subsequent authoring.
    # Replace only our own auxiliary fixture, preserving every task item.
    replace_auxiliary=(editable_authored_storage(original,row,old)
                       and (group_entities is None or {p['entity'] for p in group}=={p['entity'] for p in original['placements'] if p['receptacle']==old}))
    construction_base=storage.replay_layout(layout_for_spec(original),
        [f for f in row.get('inserted_furniture',[]) if f['name']!=old]) if replace_auxiliary else base
    if not prefer_constructed:
        reused=reuse_existing_storage(original,row,base,old,group,room,kind,items,out,evidence,allow_partial=allow_partial)
        if reused is not None:return reused
    feasible=[];models=candidate_models(kind,room)
    if allowed_models:models=[m for m in models if m in allowed_models]
    if prefer_constructed:
        profiles=read(g.ROOT/'build/native_support_geometry/index.json').get('assets',{})
        successes=read(g.ROOT/'build/verified_storage_sites/index.json').get('asset_successes',{})
        native_report=read(Path(evidence)/'validation.json')
        observed_native_failure=(native_report.get('scenario_sha256')==hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest()
                                 and native_report.get('status')=='failed'
                                 and not proposal_coordinate_bug(native_report))
        current_model=(old_entity.get('category'),old_entity.get('model'))
        geometry=read(g.ROOT/'build/native_fillable_geometry/index.json').get('assets',{})
        tallest=max(size[2] for size in items.values())
        def quality(model):
            values=[abs(c.get('floor_minus_fillable_bottom_m',1.)) for c in
                    profiles.get('/'.join(model),{}).get('components',{}).values() if c.get('status')=='measured_plane']
            # Try another fitting model before returning to the exact failed
            # fixture. This is only task-local ordering, never an infeasibility
            # claim or an asset blacklist, including when the solver was UNKNOWN.
            # Tall stock in a failed generic cupboard gets a door cabinet
            # before another shallow drawer stack. USD fillable hints can
            # overestimate clearance above a drawer; this is only ordering,
            # never a claim that a drawer is physically infeasible.
            tall_drawer_risk=(observed_native_failure and kind=='cabinet' and tallest>.22 and
                any(j['type']=='PhysicsPrismaticJoint' for j in geometry.get('/'.join(model),{}).get('joints',[])))
            return (bool(observed_native_failure and model==current_model),bool(tall_drawer_risk),
                    '/'.join(model) in rejected_models,'/'.join(model) not in successes,not bool(values),min(values) if values else 1.)
        models=sorted(models,key=quality)
    for allow_measured_empty in (False,True):
        added_in_pass=0
        for category,model in models:
            # USD joints alone do not guarantee that this SDK instantiated
            # Open. This exact task-local failure requires another fixture;
            # it is not a global blacklist or an exemption from Open checks.
            if native_missing_open and (category,model)==(old_entity.get('category'),old_entity.get('model')):continue
            if any((category,model)==candidate[:2] for candidate in feasible):continue
            if not storage.model_openable(category,model):continue
            entity=dict(category=category,model=model,interiors=[])
            storage.enrich_constructed_interiors({'candidate':entity})
            packed=compartment_assignment(entity,items,kind,native_size=True,allow_patch_support=allow_patch_support,
                                         allow_measured_empty=allow_measured_empty,
                                         support_entities=flat_support_entities(original,group),
                                         single_drawer=bool(re.search(r'\bdrawer\b',storage_scope_text(original['instruction']),re.I)))
            if packed is not None:
                assignment,solved=packed;feasible.append((category,model,assignment,solved));added_in_pass+=1
            if added_in_pass>=3 and not exhaustive_models:break
        # A geometrically fitting model may have no usable authoring site.
        # Keep a bounded second tier instead of repeatedly returning to the
        # same blocked scene cabinet when the first tier cannot be placed.
    if not feasible:
        if prefer_constructed:
            reused=reuse_existing_storage(original,row,base,old,group,room,kind,items,out,evidence,allow_partial=allow_partial)
            if reused is not None:return reused
        return None,dict(status='deferred',reason='No measured native compartment fits unchanged inventory',storage=old)
    from collections import Counter
    screens=Counter();previous=storage.MODELS.get(kind)
    try:
        start=0 if prefer_constructed else variant%len(feasible)
        for category,model,assignment,solution in feasible[start:]+feasible[:start]:
            storage.MODELS[kind]=(category,model)
            size=read(g.ASSETS/'objects'/category/model/'misc/metadata.json')['bbox_size']
            clearances=[max(size[:2]),max(size[:2])/2,0.] if prefer_constructed else [0.]
            for clearance,site in [(c,s) for c in clearances for s in range(0 if prefer_constructed else variant%2,8)]:
                name=old if replace_auxiliary else f'expert_storage_{variant}'
                if name in construction_base.entities:continue
                # Placement temporarily opens drawers/doors. Screening only
                # the closed bbox can put an open drawer over a neighbouring
                # floor tile or inside a wall. Reserve a conservative sweep;
                # native support/collision checks remain authoritative.
                proposal=storage.augment(construction_base,room,kind,name,variant=site,
                    opening_clearance_m=clearance)
                if proposal is None:screens['no_native_fixture_site']+=1;continue
                enhanced,source=proposal;spec=copy.deepcopy(original)
                spawn=spec['anchors'][spec['robot']['init_anchor']]['position'];rr,cc=enhanced.map.to_map(*spawn[:2])
                spawn_free=(0<=rr<enhanced.free.shape[0] and 0<=cc<enhanced.free.shape[1] and enhanced.free[rr,cc])
                # A useful new cabinet may occupy the previous start patch.
                # The user explicitly authorized changing the initial robot
                # position. Keep partial multi-source repairs conservative:
                # their later preservation contract does not authorize it.
                if not spawn_free and (allow_partial or len(row.get('spawn_repairs',[]))>=3):screens['spawn_occupied_no_authorized_local_retry']+=1;continue
                # Keep the new fixture local and retain a reachable delivery surface.
                if np.linalg.norm(g.boxcenter(enhanced.entities[name])[:2]-old_center[:2])>4.:screens['source_too_far']+=1;continue
                destination_anchor=scene_route_anchor(original,row['destination_receptacle'],enhanced.entities)
                route=enhanced.route(name,room,destination_anchor,row['destination_room'])
                if route is None:screens['no_source_delivery_route']+=1;continue
                sr,sc=enhanced.map.to_map(*route['start'][:2])
                if spawn_free and enhanced.labels[rr,cc]!=enhanced.labels[sr,sc]:screens['spawn_component_disconnected']+=1;continue
                other_sources={p['receptacle'] for p in spec['placements']
                               if p['relation']=='inside' and p['receptacle']!=old}
                if any(rec in enhanced.entities and enhanced.route(rec,
                       (enhanced.entities[rec].get('in_rooms') or [room])[0],
                       destination_anchor,row['destination_room']) is None for rec in other_sources):continue
                for p in spec['placements']:
                    if p['entity'] in assignment:p.update(receptacle=name,link=assignment[p['entity']])
                released=clear_unstated_links(spec)
                # Invalid unopenable source states cannot be restored. Valid
                # old furniture states are preserved even when now empty.
                if native_missing_open or not storage.model_openable(old_entity['category'],old_entity['model']):
                    if any((p.get('receptacle')==old and p.get('kind')=='inside') or
                           (p.get('entity')==old and p.get('kind') in ('open','closed')) for p in spec.get('goal',[])):
                        continue
                    spec['initial_states'].pop(old,None)
                spec['initial_states'][name]={'open':False}
                added=next(f for f in enhanced.inserted_furniture if f['name']==name)
                replacement=dict(name=name,category=category,model=model,fixed_base=True)
                if replace_auxiliary:
                    previous_obj=next(o for o in spec['objects'] if o['name']==old)
                    spec['objects'][spec['objects'].index(previous_obj)]=replacement
                    floor_placement=next(p for p in spec['placements'] if p['entity']==old)
                    floor_placement.update(receptacle=added['floor'],position_xy=added['position_xy'])
                else:
                    spec['objects'].append(replacement)
                    spec['placements'].append(dict(entity=name,relation='on_top',receptacle=added['floor'],position_xy=added['position_xy']))
                spawn_change=None
                if not spawn_free:
                    spawn_row=dict(row,source_receptacle=name,source_room=room)
                    spawn_change=nearby_spawn_proposal(spec,spawn_row,enhanced,[name])
                    if spawn_change is None:screens['no_local_spawn_preserving_grounding']+=1;continue
                    init=spec['robot']['init_anchor'];before_spawn=copy.deepcopy(spec['anchors'][init])
                    spec['anchors'][init]['position']=spawn_change['position'];route=spawn_change['route']
                pf=preflight_recipe(spec,enhanced.entities,dimensions(spec))
                if pf['status']!='heuristic_fit' and not allow_partial:screens['other_group_static_rejection']+=1;continue
                contract=copy.deepcopy(original)
                if replace_auxiliary:
                    previous_obj=next(o for o in contract['objects'] if o['name']==old)
                    contract['objects'][contract['objects'].index(previous_obj)]=copy.deepcopy(replacement)
                if spawn_change is not None:contract['anchors'][spec['robot']['init_anchor']]['position']=spawn_change['position']
                assert_preserved(contract,spec)
                path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
                result=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    inserted_furniture=enhanced.inserted_furniture,object_count=len(spec['objects']),layout_preflight=pf,
                    restoration_entities=list(dict.fromkeys(row.get('restoration_entities',[])+[name])))
                if row.get('source_receptacle')==old and any(n in assignment for n in row.get('target_entities',[original.get('target',{}).get('entity')])):
                    result.update(source_receptacle=name,route=route)
                if spawn_change is not None:
                    spawn_edit=dict(authorization='user_requested_change_initial_spawn_20261114',before=before_spawn,
                        after=copy.deepcopy(spec['anchors'][spec['robot']['init_anchor']]),
                        source_sha256=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest(),
                        reason='New source fixture occupies the old initial footprint; same-room local start preserves heading and task grounding',
                        screening=spawn_change['screening'],physical_validation_required=True)
                    result.update(spawn_repairs=row.get('spawn_repairs',[])+[spawn_edit],
                        spawn_repair_origin=row.get('spawn_repair_origin',before_spawn['position']),route=route)
                evidence_doc=dict(status='static_screened_physics_pending' if pf['status']=='heuristic_fit' else 'static_partial_not_native_ready',old_storage=old,new_storage=added,
                    released_compiler_links=released,
                    replaced_failed_auxiliary_fixture=replace_auxiliary,
                    previous_auxiliary_asset=dict(category=old_entity['category'],model=old_entity['model']) if replace_auxiliary else None,
                    conservative_opening_clearance_m=clearance,
                    opening_sweep_requires_native_measurement=bool(prefer_constructed and clearance<max(size[:2])),
                    original_recipe=row['scenario'],original_sha256=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest(),
                    diagnosis_directory=str(evidence),solver=solution,variant=variant,
                    task_local_model_experience=experience,
                    removed_uninstantiated_open_state=old if native_missing_open else None,
                    initial_spawn_change=spawn_edit if spawn_change is not None else None,
                    preserved=['instruction','goal','target','robot','spawn height and heading','all task items and counts','relation types'])
                (path.parent/'expert_intervention.json').write_text(json.dumps(evidence_doc,indent=2))
                return result,evidence_doc
    finally:
        if previous is None:storage.MODELS.pop(kind,None)
        else:storage.MODELS[kind]=previous
    if prefer_constructed and not exhaustive_models:
        return curate(row,evidence,out,variant=variant,old_storage=old,allow_partial=allow_partial,
            group_entities=group_entities,allow_patch_support=allow_patch_support,prefer_constructed=True,exhaustive_models=True)
    if prefer_constructed:
        reused=reuse_existing_storage(original,row,base,old,group,room,kind,items,out,evidence,allow_partial=allow_partial)
        if reused is not None:return reused
    return None,dict(status='deferred',reason='No connected, spawn-clear local site with whole-scene packing',storage=old,
                     candidate_models=[dict(category=c,model=m) for c,m,*_ in feasible],rejection_counts=dict(screens))


def discover_sources():
    """Bind failures to recipes by content hash, not matching filenames alone."""
    queue=read(g.ROOT/'runs/asset_full_queue_20261105/queue.json')
    wanted={r['id']:r for r in queue['tasks'] if r['status']!='ready_for_human_review'}
    by_instance={}
    for path in (g.ROOT/'scenarios').glob('*/manifest.json'):
        for row in read(path).get('tasks',[]):
            if row.get('parent_task_id') in wanted:by_instance.setdefault(row['id'],[]).append(row)
    hashes={};cases=[]
    for tid,task in wanted.items():
        reports=[]
        for iid,state in task.get('instances',{}).items():
            directory=g.ROOT/'runs'/state['run']/iid
            build=read(directory/'validation.json');restore=read(directory/'restore_validation.json')
            filename='restore_validation.json' if build.get('status')=='initialization_validated' and restore else 'validation.json'
            path=directory/filename;report=read(path)
            if report:reports.append((path.stat().st_mtime,directory,report,iid))
        case=dict(id=tid,record=None,evidence=None,error='No matching native report')
        for _,directory,report,iid in sorted(reports,reverse=True,key=lambda x:x[0]):
            for row in by_instance.get(iid,[]):
                path=Path(row['scenario'])
                if not path.is_absolute():path=g.ROOT/path
                if not path.exists():continue
                if path not in hashes:hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
                if hashes[path]!=report.get('scenario_sha256'):continue
                case.update(record=dict(row,scenario=str(path)),evidence=str(directory),
                            error=report.get('error',report.get('status')),source_sha256=hashes[path]);break
            if case['record']:break
        cases.append(case)
    return cases


def clear_unstated_links(spec):
    """Remove compiler-only compartment choices, preserving language scope."""
    text=storage_scope_text(spec.get('instruction',''))
    if has_qualified_storage_location(spec.get('instruction','')):
        return []
    # A singular drawer must retain a single compartment, not become a search
    # across several drawers. Such cases use the fixture-repair route instead.
    if re.search(r'\bdrawer\b',text,re.I):return []
    changed=[]
    for placement in spec['placements']:
        if placement['relation']=='inside' and placement.get('link'):
            changed.append(dict(entity=placement['entity'],previous_link=placement.pop('link')))
    return changed


def source_allows_multiple_fixtures(instruction):
    """Collective supplies/storage do not imply one compiler-picked cabinet."""
    text=storage_scope_text(instruction)
    if has_qualified_storage_location(instruction):return False
    return not bool(re.search(r'\b(drawer|cupboard|cabinet|basket|fridge|refrigerator|sideboard|trunk)\b',text,re.I))


def repair_covered_surface_binding(row,evidence,out):
    """Correct body identity at an unchanged physical counter location."""
    original=yaml.safe_load(Path(row['scenario']).read_text())
    if not original.get('scene',{}).get('model'):return None
    relations=[p for p in original['placements'] if p['relation']=='on_top']+[
        p for p in original.get('goal',[]) if p.get('kind')=='on_top']
    if not relations:return None
    layout=layout_for_spec(original);mapping={}
    for p in relations:
        name=p.get('receptacle')
        if name not in layout.entities:continue
        surface=g.exposed_surface_entity(layout.entities,name)
        if surface!=name:mapping[name]=surface
    if not mapping:return None
    spec=copy.deepcopy(original);changes=[];sizes=dimensions(spec)
    for field in ('placements','goal'):
        for p in spec[field]:
            relation=p.get('relation',p.get('kind'))
            old=p.get('receptacle')
            if relation!='on_top' or old not in mapping:continue
            p['receptacle']=mapping[old]
            changes.append(dict(field=field,entity=p['entity'],before=old,after=mapping[old]))
            if (field=='placements' and p.get('position_xy') is None and
                    sum(q['relation']=='on_top' and q['receptacle']==old for q in original['placements'])==1):
                lo,hi=np.asarray(layout.entities[p['receptacle']]['aabb'],float)
                a,b=np.asarray(layout.entities[old]['aabb'],float)
                lo[:2]=np.maximum(lo[:2],a[:2]);hi[:2]=np.minimum(hi[:2],b[:2])
                # Only a starting proposal. Native contact, stability and
                # independent reload still determine acceptance.
                if np.all(np.asarray(sizes[p['entity']][:2])+.05<=(hi-lo)[:2]):
                    p['position_xy']=authoring_xy_before_clearance(spec,p['receptacle'],((lo+hi)/2)[:2])
    result=copy.deepcopy(row)
    # An inside source retains its cabinet identity. Only surface sources move.
    for key,roomkey in [('source_receptacle','source_room'),('destination_receptacle','destination_room')]:
        old=result.get(key)
        if old in mapping and (key=='destination_receptacle' or any(
                p.get('receptacle')==old and p.get('relation')=='on_top' and p['entity'] in row['target_entities']
                for p in original['placements'])):result[key]=mapping[old]
    replay=storage.replay_layout(layout,row.get('inserted_furniture',[]))
    approach_patch=None
    if row.get('destination_receptacle') in mapping:
        # One countertop body may span several cupboards. Preserve the
        # original local destination in front of the spawn, rather than
        # selecting the first 64 approaches to the body's whole-house AABB.
        # The patch is a subset of the real slab; no scene geometry changes.
        old=row['destination_receptacle'];new=mapping[old]
        a,b=np.asarray(layout.entities[old]['aabb'],float)
        lo,hi=np.asarray(layout.entities[new]['aabb'],float)
        lo[:2]=np.maximum(lo[:2],a[:2]);hi[:2]=np.minimum(hi[:2],b[:2])
        replay=copy.copy(replay);replay.entities=dict(replay.entities);replay.cache={}
        replay.entities[new]=dict(replay.entities[new],aabb=[lo.tolist(),hi.tolist()])
        approach_patch=dict(contact_body=new,above_fixture=old,aabb=[lo.tolist(),hi.tolist()],
                            scope='private route proposal to a subset of the actual slab; no synthetic body')
    # A dynamic basket is not a scene-survey entity. Resolve both role
    # endpoints through the repaired support tree without changing the
    # actual basket identity exposed in the task metadata.
    source_anchor=scene_route_anchor(spec,result['source_receptacle'],replay.entities)
    destination_anchor=scene_route_anchor(spec,result['destination_receptacle'],replay.entities)
    if source_anchor is None or destination_anchor is None:return None
    route=replay.route(source_anchor,result['source_room'],destination_anchor,result['destination_room'])
    if route is None:return None
    spawn=spec['anchors'][spec['robot']['init_anchor']]['position']
    rr,cc=replay.map.to_map(*spawn[:2]);ar,ac=replay.map.to_map(*route['start'][:2])
    if not (0<=rr<replay.map.size and 0<=cc<replay.map.size and replay.free[rr,cc]
            and replay.labels[rr,cc]==replay.labels[ar,ac]):return None
    result['route']=route
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    journal=dict(status='surface_body_binding_corrected_physics_pending',changes=changes,
        destination_approach_patch=approach_patch,
        geometry={old:dict(before=layout.entities[old]['aabb'],after=layout.entities[new]['aabb']) for old,new in mapping.items()},
        source_survey=str(layout.path),source_survey_sha256=hashlib.sha256(layout.path.read_bytes()).hexdigest(),
        original_recipe=row['scenario'],original_scenario_sha256=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest(),
        reason='Native OnTop requires touching the named body; a separate measured slab covers the named cabinet.',
        preserved=['instruction','all object models and counts','target identities','relation kinds','spawn','robot and skills','physical delivery location'],
        physical_validation_required=True)
    result.update(scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),surface_binding_revision=journal)
    (path.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    return result,journal


def solve_surface_proposal(items,extent,*,time_limit_s=1.,obstacles=(),allow_yaw=False):
    """Keep the preferred spacing, then try denser physical proposals.

    Empty space between authored boxes is not a benchmark safety threshold.
    A 25-mm authoring margin can reject a surface that has room for every
    real prop. The native collision/contact/stability checks still decide
    whether any of these proposals is physically acceptable.
    """
    attempts=[]
    for margin in [.025,.015,.005]:
        # Obstacles disable the simple strip witness. Start with conservative
        # outward-rounded 5-mm candidates to avoid allocating a huge 1-mm
        # plane; a coarse miss must still try the finer domain.
        for grid in ([.005,.001] if obstacles else [.001]):
            result=solve_layout(items,{'surface':[*extent,6.]},margin_m=margin,grid_m=grid,
                time_limit_s=time_limit_s,allowed_yaws=None if allow_yaw else {name:[0] for name in items},
                **({'obstacles':{'surface':obstacles},'max_candidates':4_000_000} if obstacles else {}))
            attempts.append(dict(margin_m=margin,grid_m=grid,status=result['status'],reason=result.get('reason')))
            if result['status']=='feasible_model':
                return dict(result,authoring_spacing_attempts=attempts,physical_validation_required=True)
    return dict(result,authoring_spacing_attempts=attempts)


def plan_explicit_surfaces(spec, layout, *, replan_receptacles=(),dimension_overrides=None,allow_yaw=False):
    """Joint surface proposals use existing XY authoring, never acceptance."""
    groups={}
    for p in spec['placements']:
        if p['relation']=='on_top' and p['receptacle'] in layout.entities:
            if layout.entities[p['receptacle']].get('category') in {
                'breakfast_table','coffee_table','dining_table','table','desk','countertop','shelf','bookcase',
                'bottom_cabinet','bottom_cabinet_no_top','metal_bottom_cabinet','dresser'}:
                groups.setdefault(p['receptacle'],[]).append(p)
    sizes=dimensions(spec);sizes.update(dimension_overrides or {})
    objects={o['name']:o for o in spec['objects']};plans=[]
    for rec,placements in groups.items():
        if any(objects.get(p['entity'],{}).get('fixed_base') for p in placements):continue
        if rec not in replan_receptacles and any(p.get('position_xy') is not None for p in placements):continue
        lo,hi=np.asarray(layout.entities[rec]['aabb'],float)
        # Default XY keeps the declared orientation. Reviewed quarter-turn
        # proposals must also be serialized as actual initial quaternions.
        items={p['entity']:sizes[p['entity']] for p in placements}
        # A countertop's outer box may include a stove, faucet or other
        # raised scene body. Keep those actual measured bodies out of the
        # proposal, rather than repeatedly packing through their footprints.
        obstacles=[];height=max(v[2] for v in items.values())
        for other,body in layout.entities.items():
            if other==rec or other in items or not body.get('aabb'):continue
            low,high=np.asarray(body['aabb'],float)
            if (high[2]>hi[2]+.005 and low[2]<hi[2]+height+.025
                    and np.all(high[:2]>lo[:2]) and np.all(low[:2]<hi[:2])):
                origin=np.array([lo[0],lo[1],hi[2]])
                obstacles.append([*(low-origin).tolist(),*(high-origin).tolist()])
        solved=solve_surface_proposal(items,(hi-lo)[:2],time_limit_s=2.,obstacles=obstacles,**({'allow_yaw':True} if allow_yaw else {}))
        if solved['status']!='feasible_model':continue
        for p in placements:
            pose=solved['poses'][p['entity']]
            p['position_xy']=authoring_xy_before_clearance(spec,rec,lo[:2]+np.asarray(pose['center_xy_m']))
            if allow_yaw and pose.get('yaw_degrees',0):
                from scipy.spatial.transform import Rotation
                hint=p.get('orientation_hint_xyzw') or [0.,0.,0.,1.]
                p['orientation_hint_xyzw']=(Rotation.from_euler('z',pose['yaw_degrees'],degrees=True)*Rotation.from_quat(hint)).as_quat().tolist()
        plans.append(dict(receptacle=rec,entities=[p['entity'] for p in placements],solver=solved,
                          measured_scene_obstacles=obstacles,
                          physical_validation_required=True))
    return plans


def surface_attempt_key(spec,receptacle,report):
    """A proposal family is specific to inventory, fixture and native code."""
    objects={o['name']:o for o in spec['objects']}
    contents=sorted((objects[p['entity']] for p in spec['placements']
        if p['relation']=='on_top' and p['receptacle']==receptacle),key=lambda o:o['name'])
    fixture=next((p for p in spec['placements'] if p['entity']==receptacle),None)
    return hashlib.sha256(json.dumps(dict(fixture=objects.get(receptacle),site=fixture,contents=contents,
        native_policy=native_policy_signature(report.get('runtime_sources',{}))),sort_keys=True).encode()).hexdigest()


def repair_failed_surface_poses(row,evidence,out):
    """Repack a rejected surface group on its unchanged real support body."""
    source=Path(row['scenario']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    report=read(Path(evidence)/'validation.json');restored=read(Path(evidence)/'restore_validation.json')
    if restored.get('scenario_sha256')==sha and restored.get('status')=='failed':report=restored
    if report.get('scenario_sha256')!=sha:return None
    error=str(report.get('error',''));original=yaml.safe_load(source.read_text())
    failed=set(re.findall(r"(?:\['|', ')(\S+) on_top \S+",error))
    if error.startswith('POST_SETTLE_RELATION_FAILED: '):failed.add(error.split(': ',1)[1])
    receptacles={p['receptacle'] for p in original['placements']
                 if p['entity'] in failed and p['relation']=='on_top'}
    sampled={edit.get('attempt_key') for edit in row.get('native_surface_sampling_repairs',[])}
    # Explicit XY already failed before the recorded native-sampling probe.
    # If that probe also failed, try a different real support instead of
    # oscillating between the same outer-box pack and native sampling.
    receptacles={rec for rec in receptacles if surface_attempt_key(original,rec,report) not in sampled}
    if not receptacles:return None
    spec=copy.deepcopy(original)
    layout=include_task_receptacles(storage.replay_layout(layout_for_spec(spec),row.get('inserted_furniture',[])),
                                   spec,evidence=evidence,source_sha=sha)
    # Limit the repair to the observed failing support, preserving other
    # already-authored poses and every object, goal and language constraint.
    # Feed only the affected groups into the proposal helper.
    working=dict(spec,placements=[p for p in spec['placements'] if p['receptacle'] in receptacles])
    plans=plan_explicit_surfaces(working,layout,replan_receptacles=receptacles)
    if not plans or spec==original:return None
    preflight=preflight_recipe(spec,layout.entities,dimensions(spec))
    if preflight['status']!='heuristic_fit':return None
    assert_preserved(original,spec)
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    journal=dict(method='repack_failed_surface_on_same_native_body',source_sha256=sha,
                 original_recipe=str(source),evidence=str(evidence),plans=plans,
                 preserved=['instruction','goal','all original assets','source and destination bodies','spawn','relation kinds'],
                 physical_validation_required=True)
    (path.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    return dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                layout_preflight=preflight,surface_pose_repairs=row.get('surface_pose_repairs',[])+[journal]),journal


def repair_unstable_reference_surface(row,evidence,out):
    """Give a reload-unstable isolated reference an explicit centre proposal."""
    name=row.get('reference_entity')
    if not name:return None
    path=Path(row['scenario']);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    report=read(Path(evidence)/'restore_validation.json')
    restored_failure=(report.get('scenario_sha256')==sha and report.get('error')=='POST_SETTLE_RELATION_FAILED: '+name)
    build=read(Path(evidence)/'validation.json')
    log=Path(evidence)/'native_validation.log';first=[]
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith('[first-placement-failure] '):
                try:first.append(json.loads(line.partition('] ')[2]))
                except ValueError:pass
    initial_failure=(build.get('scenario_sha256')==sha and build.get('status')=='failed' and first
                     and first[0].get('entity')==name and first[0].get('relation')=='on_top')
    # A native sample can be supported by an OPEN drawer and lose that
    # support when the cupboard is closed. Its initial placement was True,
    # so only the final declared-relation verification identifies this case.
    post_close_failure=(build.get('scenario_sha256')==sha and build.get('status')=='failed'
                        and 'placement failed verification after physics settled:' in str(build.get('error',''))
                        and (name+' on_top ') in str(build.get('error','')))
    if not restored_failure and not initial_failure and not post_close_failure:return None
    original=yaml.safe_load(path.read_text());spec=copy.deepcopy(original)
    p=next((p for p in spec['placements'] if p['entity']==name and p['relation']=='on_top'),None)
    if p is None or any(q['entity']!=name and q['receptacle']==p['receptacle'] and q['relation']=='on_top' for q in spec['placements']):return None
    layout=layout_for_spec(spec);surface=layout.entities.get(p['receptacle'],{})
    if not surface.get('aabb'):return None
    lo,hi=np.asarray(surface['aabb'],float)
    if np.any(np.asarray(dimensions(spec)[name][:2])+.1>(hi-lo)[:2]):return None
    # A long countertop's bounding-box centre may lie above a stove, sink
    # rim or another raised body. After that exact point was rejected, use
    # another footprint-clear point on the SAME support body, rather than
    # returning the identical failed centre indefinitely.
    size=np.asarray(dimensions(spec)[name],float)
    delta=clearance_xy_delta(spec,p['receptacle'])
    center=((lo+hi)/2)[:2];previous=np.asarray(p['position_xy'],float)+delta if p.get('position_xy') is not None else center
    tried=[np.asarray(edit['after_position_xy'])+delta for edit in row.get('reference_pose_repairs',[])]
    if p.get('position_xy') is not None:tried.append(previous)
    options=[center]
    for delta in [.15,.3,.45,.6]:
        options.extend(center+np.asarray([dx,dy]) for dx,dy in
            [(delta,0),(-delta,0),(0,delta),(0,-delta),(delta,delta),(-delta,delta),(delta,-delta),(-delta,-delta)])
    viable=[]
    for candidate in options:
        if np.linalg.norm(candidate-previous)>.85 or any(np.linalg.norm(candidate-v)<.1 for v in tried):continue
        a=candidate-size[:2]/2-.025;b=candidate+size[:2]/2+.025
        if np.any(a<lo[:2]) or np.any(b>hi[:2]):continue
        blocked=False
        for other,body in layout.entities.items():
            if other==p['receptacle'] or not body.get('aabb'):continue
            low,high=np.asarray(body['aabb'],float)
            if (high[2]>hi[2]+.005 and low[2]<hi[2]+size[2]+.025 and
                    np.all(high[:2]>a) and np.all(low[:2]<b)):
                blocked=True;break
        if not blocked:viable.append(candidate)
    if not viable:return None
    previous_xy=copy.deepcopy(p.get('position_xy'))
    xy=authoring_xy_before_clearance(spec,p['receptacle'],viable[0])
    p['position_xy']=xy;assert_preserved(original,spec)
    target=Path(out)/row['id']/'scenario.yaml';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
    journal=dict(status='reference_surface_pose_proposal_physics_pending',entity=name,receptacle=p['receptacle'],position_xy=xy,
        original_recipe=str(path),original_scenario_sha256=sha,evidence=str(evidence),
        previous_position_xy=previous_xy,
        reason='Native initialization, final post-closing verification or independent reload rejected the isolated reference OnTop relation; propose another footprint-clear point on the same physical support body.',
        physical_validation_required=True,preserved=['instruction','goal','all assets','target identities','spawn','relations'])
    (target.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    edit=dict(entity=name,receptacle=p['receptacle'],before_position_xy=previous_xy,
              after_position_xy=xy,source_sha256=sha,physical_validation_required=True)
    return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                reference_pose_repairs=row.get('reference_pose_repairs',[])+[edit]),journal


def center_open_container_placements(spec,layout):
    """Give baskets a stable authoring pose on their declared support surface.

    AABB fit is only a proposal. Concave countertops and all physical contacts
    still have to pass native initialization and independent reload.
    """
    objects={o['name']:o for o in spec['objects']};sizes=dimensions(spec);changes=[]
    for p in spec['placements']:
        if p['relation']!='on_top' or p.get('position_xy') is not None:continue
        obj=objects.get(p['entity'],{})
        if obj.get('category') not in ('shopping_basket','wicker_basket'):continue
        surface=layout.entities.get(p['receptacle'],{})
        if not surface.get('aabb'):continue
        lo,hi=np.asarray(surface['aabb'],float)
        if np.any(np.asarray(sizes[p['entity']][:2])+.04>(hi-lo)[:2]):continue
        xy=authoring_xy_before_clearance(spec,p['receptacle'],((lo+hi)/2)[:2]);p['position_xy']=xy
        changes.append(dict(entity=p['entity'],receptacle=p['receptacle'],position_xy=xy,
                            orientation='identity via explicit XY placement',physical_validation_required=True))
    return changes


def apply_registered_support_rebinding(row,evidence,out,*,decision_override=None):
    """Apply an explicit expert support-asset decision, preserving task content.

    Default repair contracts stay strict. Only the registered, non-target
    receptacle changes native model, within its exact semantic category.
    """
    decision=decision_override or read(g.ROOT/'build/expert_support_overrides/index.json').get('tasks',{}).get(row.get('parent_task_id'))
    if not decision:return None
    original=yaml.safe_load(Path(row['scenario']).read_text())
    if hashlib.sha256(original['instruction'].encode()).hexdigest()!=decision['instruction_sha256']:return None
    name=decision['entity'];before=next((o for o in original['objects'] if o['name']==name),None)
    if before!=decision['original_object']:return None
    if name in row.get('target_entities',[]) or name==original.get('target',{}).get('entity'):raise ValueError('SUPPORT_REBINDING_CANNOT_CHANGE_TARGET')
    after=decision['replacement']
    if before['category']!=after['category'] or before['name']!=after['name']:raise ValueError('SUPPORT_REBINDING_CATEGORY_CHANGED')
    if {k:v for k,v in before.items() if k!='model'}!={k:v for k,v in after.items() if k!='model'}:raise ValueError('SUPPORT_REBINDING_MAY_ONLY_SELECT_NATIVE_MODEL')
    if re.search(r'\bdrawers?\b',storage_scope_text(original['instruction']),re.I) and not native_drawer_links(after['category'],after['model']):
        raise ValueError('SUPPORT_REBINDING_DOES_NOT_HAVE_REQUIRED_NATIVE_DRAWERS')
    spec=copy.deepcopy(original);spec['objects'][spec['objects'].index(before)]=copy.deepcopy(after)
    furniture=copy.deepcopy(row.get('inserted_furniture',[]));updated_fixture=False
    for fixture in furniture:
        if fixture['name']!=name:continue
        size=np.asarray(dimensions(spec)[name]);xy=np.asarray(fixture['position_xy']);floor_z=fixture['aabb'][0][2]
        fixture.update(model=after['model'],category=after['category'],aabb=[[*list(xy-size[:2]/2),floor_z],[*list(xy+size[:2]/2),float(floor_z+size[2])]])
        updated_fixture=True
    layout=include_task_receptacles(storage.replay_layout(layout_for_spec(spec),furniture),spec)
    centered=center_open_container_placements(spec,layout)
    pf=preflight_recipe(spec,layout.entities,dimensions(spec))
    if pf['status']!='heuristic_fit':return None
    route=None
    if updated_fixture:
        spawn=spec['anchors'][spec['robot']['init_anchor']]['position'];rr,cc=layout.map.to_map(*spawn[:2])
        if not (0<=rr<layout.map.size and 0<=cc<layout.map.size and layout.free[rr,cc]):return None
        route=layout.route(row['source_receptacle'],row['source_room'],row['destination_receptacle'],row['destination_room'])
        if route is None:return None
        ar,ac=layout.map.to_map(*route['start'][:2])
        if layout.labels[rr,cc]!=layout.labels[ar,ac]:return None
    # Validate the usual contract after substituting exactly the one recorded
    # support decision. This cannot remove/change any target or distractor.
    permitted=copy.deepcopy(original);permitted['objects'][permitted['objects'].index(before)]=copy.deepcopy(after)
    assert_preserved(permitted,spec)
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    doc=dict(status='static_screened_physics_pending',method='registered_native_support_rebinding',
        centered_container_placements=centered,
        decision=decision,original_recipe=row['scenario'],physical_validation_required=True,
        preserved=['instruction','goal','target','robot','spawn','all targets and distractors','object count','all relation types'],
        changed=['one native support model; original asset scales retained'])
    (path.parent/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
    result=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                inserted_furniture=furniture,layout_preflight=pf,support_asset_changes=row.get('support_asset_changes',[])+[decision])
    if route is not None:result['route']=route
    return result,doc


def repair_rigid_container_capacity(row,evidence,out):
    """Try a larger native, non-target basket only after a geometric miss.

    This is a recorded data-authoring asset choice. It never changes a task
    target, named colour/size, object count, relation type or native check.
    Missing geometry alone cannot justify replacing the current model.
    """
    source=Path(row['scenario']);spec=yaml.safe_load(source.read_text())
    if re.search(r'\b(?:red|green|blue|black|white|brown|yellow|small|large|round|square)\s+(?:\w+\s+){0,2}(?:basket|bowl|tray)\b',spec['instruction'],re.I):return None
    targets=set(row.get('target_entities',[]))|{spec.get('target',{}).get('entity')}
    destinations={goal.get('receptacle') for goal in spec.get('goal',[])}
    parents={p['receptacle'] for p in spec['placements'] if p['relation']=='inside'}
    eligible=[obj for obj in spec['objects'] if obj.get('category') in ('shopping_basket','wicker_basket')
              and obj['name'] in parents and obj['name'] not in targets|destinations and not obj.get('scale')]
    if not eligible:return None
    layout=include_task_receptacles(storage.replay_layout(layout_for_spec(spec),row.get('inserted_furniture',[])),spec)
    initial=preflight_recipe(spec,layout.entities,dimensions(spec))
    findings={f['receptacle']:f for f in initial.get('findings',[])}
    def capacity(body):
        volumes=copy.deepcopy(authoring_volumes(body))
        if body.get('offline_interior_provenance',{}).get('open_top_centroid_height'):
            for v in volumes.values():v[2]=(v[2]+.02)/2
        return sum(float(np.prod(v)) for v in volumes.values())
    for obj in eligible:
        name=obj['name'];category=obj['category'];finding=findings.get(name,{})
        if (category not in ('shopping_basket','wicker_basket') or name in targets|destinations or
                finding.get('status')=='heuristic_fit' or not finding.get('volume_extents_m') or obj.get('scale')):continue
        old_capacity=capacity(layout.entities.get(name,{}));candidates=[]
        for metadata in (g.ASSETS/'objects'/category).glob('*/misc/metadata.json'):
            model=metadata.parent.parent.name
            if model==obj['model']:continue
            body=dict(category=category,model=model,interiors=[]);storage.enrich_constructed_interiors({'candidate':body})
            amount=capacity(body)
            if amount>old_capacity+1e-8:candidates.append((amount,model))
        for _,model in sorted(candidates):
            decision=dict(revision_id='geometry_screened_container_'+hashlib.sha256((source.read_text()+name+model).encode()).hexdigest()[:16],
                authorization='user_approved_find_and_deliver_rewrite',entity=name,
                instruction_sha256=hashlib.sha256(spec['instruction'].encode()).hexdigest(),original_object=obj,
                replacement=dict(obj,model=model),original_scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reason='The original nominal layout has no current geometric witness. Try a larger same-category native basket with hash-verified internal geometry; keep every task item, target, count, instruction and relation. The new complete recipe must obtain its own geometric witness and fresh native initialization/reload.',
                original_geometric_finding=finding,decision_source='native_capacity_screen_not_agent_feedback')
            result=apply_registered_support_rebinding(row,evidence,Path(out)/model,decision_override=decision)
            if result is not None:return result
    return None


def observed_fixture_envelope(row,evidence,name):
    """Read a pose-bound open footprint from this exact failed native build."""
    recipe=Path(row['scenario'])
    if read(Path(evidence)/'validation.json').get('scenario_sha256')!=hashlib.sha256(recipe.read_bytes()).hexdigest():return None
    spec=yaml.safe_load(recipe.read_text())
    placement=next((p for p in spec['placements'] if p['entity']==name and p.get('position_xy') is not None),None)
    if not placement:return None
    for line in (Path(evidence)/'native_validation.log').read_text(errors='replace').splitlines():
        if '[expert-pose-result] ' not in line:continue
        try:event=json.loads(line.split('[expert-pose-result] ',1)[1])
        except ValueError:continue
        if event.get('entity')!=name:continue
        q=np.asarray(event['orientation'])
        if np.linalg.norm(q[:3])>.001 or abs(abs(q[3])-1)>.001:return None
        bounds=np.asarray(event['aabb'],float)[:,:2]-np.asarray(placement['position_xy'])
        if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):return None
        # Numerical pose variation allowance, not a relaxation of native checks.
        return (bounds+np.array([[-.02,-.02],[.02,.02]])).tolist()
    return None


def relocate_failed_fixture(row,evidence,out):
    """Repair the fixture that failed to stand on its floor, not its contents.

    Reuse the exact fixed native asset and all task objects. Only its authoring
    site changes; the original task and source/destination identities stay put.
    """
    original=yaml.safe_load(Path(row['scenario']).read_text())
    log=Path(evidence)/'native_validation.log'
    if not log.exists():return None
    match=re.search(r'\[placement\] end (\S+) placed=False',log.read_text(errors='replace'))
    if not match:return None
    name=match[1];inserted=row.get('inserted_furniture',[])
    fixture=next((f for f in inserted if f['name']==name),None)
    obj=next((o for o in original['objects'] if o['name']==name),None)
    placement=next((p for p in original['placements'] if p['entity']==name),None)
    if not fixture or not obj or not obj.get('fixed_base') or not placement or placement['relation']!='on_top':return None
    kind=repair_kind(fixture,[],original['instruction'])
    if kind is None:return None
    base=storage.replay_layout(layout_for_spec(original),[f for f in inserted if f['name']!=name])
    previous=storage.MODELS.get(kind);size=dimensions(original)[name]
    envelope=observed_fixture_envelope(row,evidence,name)
    try:
        storage.MODELS[kind]=(obj['category'],obj['model'])
        for site in range(8):
            proposal=storage.augment(base,fixture['room'],kind,name,variant=site,
                opening_clearance_m=0. if envelope is not None else max(size[:2]),footprint_offsets_m=envelope)
            if proposal is None:continue
            layout,_=proposal;added=next(f for f in layout.inserted_furniture if f['name']==name)
            if added['floor']==placement['receptacle'] and np.allclose(added['position_xy'],placement.get('position_xy'),atol=.001):continue
            spawn=original['anchors'][original['robot']['init_anchor']]['position'];rr,cc=layout.map.to_map(*spawn[:2])
            if not (0<=rr<layout.free.shape[0] and 0<=cc<layout.free.shape[1] and layout.free[rr,cc]):continue
            route=layout.route(name,fixture['room'],row['destination_receptacle'],row['destination_room'])
            if route is None:continue
            sr,sc=layout.map.to_map(*route['start'][:2])
            if layout.labels[rr,cc]!=layout.labels[sr,sc]:continue
            if np.linalg.norm(np.asarray(added['position_xy'])-np.asarray(fixture['position_xy']))>4.:continue
            spec=copy.deepcopy(original)
            p=next(p for p in spec['placements'] if p['entity']==name)
            p.update(receptacle=added['floor'],position_xy=added['position_xy'])
            pf=preflight_recipe(spec,layout.entities,dimensions(spec))
            if pf['status']!='heuristic_fit':continue
            assert_preserved(original,spec)
            path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
            result=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        inserted_furniture=layout.inserted_furniture,layout_preflight=pf)
            if row.get('source_receptacle')==name:result['route']=route
            doc=dict(status='static_screened_physics_pending',method='relocate_failed_fixture',
                entity=name,previous_fixture=fixture,new_fixture=added,original_recipe=row['scenario'],
                measured_open_footprint_offsets_m=envelope,
                physical_validation_required=True,preserved=['instruction','goal','target','robot','spawn','all assets','all relation types'])
            (path.parent/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
            return result,doc
    finally:
        if previous is None:storage.MODELS.pop(kind,None)
        else:storage.MODELS[kind]=previous
    return None,dict(status='deferred',method='relocate_failed_fixture',entity=name,
        reason='No connected opening-clear site for the same failed native fixture; do not replay its unchanged floor failure',
        physical_validation_required=True)


def scene_route_anchor(spec, entity, scene_entities):
    """Resolve a task-object destination through its authored support chain.

    This selects a private route-screening anchor, never a different delivery
    object or goal. The live task still uses the original basket/container.
    """
    parents={p['entity']:p['receptacle'] for p in spec.get('placements',[])}
    seen=set()
    while entity not in scene_entities:
        if entity in seen:raise ValueError('CYCLIC_TASK_ROUTE_SUPPORT_CHAIN')
        seen.add(entity)
        if entity not in parents:raise ValueError('TASK_ROUTE_SUPPORT_NOT_IN_SCENE: '+entity)
        entity=parents[entity]
    return entity


def nearby_spawn_proposal(spec,row,layout,contact_entities=()):
    """Screen a local start change, preserving heading and deictic grounding.

    These are authoring proposals. The unchanged native footprint, body,
    opening, relation and independent reload checks remain authoritative.
    """
    from collections import deque
    anchor=spec['anchors'][spec['robot']['init_anchor']]
    old=np.asarray(anchor['position'],float);q=np.asarray(anchor['orientation'],float)
    if len(q)!=4 or not np.isfinite(q).all() or not np.isfinite(old).all():return None
    # Do not interpret a tilted robot as an upright planar heading.
    if np.linalg.norm(q[:2])>1e-5 or abs(np.linalg.norm(q)-1.)>1e-4:return None
    yaw=2.*np.arctan2(q[2],q[3]);forward=np.array([np.cos(yaw),np.sin(yaw)])
    grid=layout.map;n=grid.size;resolution=grid.resolution
    def cell(xy):return tuple(int(round(float(v)/resolution+n/2)) for v in (xy[1],xy[0]))
    initial=cell(old[:2])
    if not (0<=initial[0]<n and 0<=initial[1]<n):return None
    room_id=grid.room_ins[initial]
    if not room_id:return None
    source=scene_route_anchor(spec,row['source_receptacle'],layout.entities)
    dest=scene_route_anchor(spec,row['destination_receptacle'],layout.entities)
    ends=layout.approaches(source,row['source_room'])
    destinations=layout.approaches(dest,row['destination_room'])
    if not ends or not destinations:return None
    destination_center=g.boxcenter(layout.entities[dest])[:2]
    front_required=bool(re.search(r'\bin front of you\b',spec['instruction'],re.I))
    tried=[edit['after']['position'][:2] for edit in row.get('spawn_repairs',[])]
    origin_xy=np.asarray(row.get('spawn_repair_origin',old.tolist())[:2],float)
    candidates=[]
    for r,c in layout.points:
        r,c=int(r),int(c)
        if grid.room_ins[r,c]!=room_id:continue
        xy=np.array([(c-n/2)*resolution,(r-n/2)*resolution])
        displacement=float(np.linalg.norm(xy-old[:2]))
        if not .3<=displacement<=1.2+1e-8 or np.linalg.norm(xy-origin_xy)>1.2+1e-8:continue
        if any(np.linalg.norm(xy-np.asarray(previous))<.15 for previous in tried):continue
        component=layout.labels[r,c]
        goals={p[1] for p in ends if layout.labels[p[1]]==component}
        if not goals or not any(layout.labels[p[1]]==component for p in destinations):continue
        vector=destination_center-xy;distance=float(np.linalg.norm(vector))
        if front_required and (distance<.1 or np.dot(vector,forward)<.5*distance):continue
        contact_clearances=[]
        for entity in contact_entities:
            lo,hi=np.asarray(layout.entities[entity]['aabb'])[:,:2]
            clearance=float(np.linalg.norm(np.maximum(np.maximum(lo-xy,0),xy-hi)))
            old_clearance=float(np.linalg.norm(np.maximum(np.maximum(lo-old[:2],0),old[:2]-hi)))
            if clearance<max(layout.radius+.30,old_clearance+.20):break
            contact_clearances.append(clearance)
        else:
            # Prefer modest backward/lateral moves with more free floor.
            score=displacement-.25*min(float(layout.room_clear[r,c]),.6)+.1*max(0.,float(np.dot(xy-old[:2],forward)))
            candidates.append((score,(r,c),xy,goals,contact_clearances))
    for _,start,xy,goals,clearances in sorted(candidates,key=lambda p:(p[0],p[1])):
        queue=deque([start]);parents={start:None};depth={start:0};goal=None
        while queue:
            current=queue.popleft()
            if current in goals:goal=current;break
            if depth[current]>=200:continue
            r,c=current
            for nxt in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                if (0<=nxt[0]<n and 0<=nxt[1]<n and layout.free[nxt] and nxt not in parents):
                    parents[nxt]=current;depth[nxt]=depth[current]+1;queue.append(nxt)
        if goal is None:continue
        cells=[];current=goal
        while current is not None:cells.append(current);current=parents[current]
        cells.reverse()
        route=dict(start=[float(xy[0]),float(xy[1]),float(yaw)],
            path_xy=[[(c-n/2)*resolution,(r-n/2)*resolution] for r,c in cells],
            length_m=round((len(cells)-1)*resolution,3),grid_m=resolution,base_radius_m=layout.radius,
            source_approach_count=len(ends),destination_approach_count=len(destinations),
            provenance='Local same-room spawn repair; conservative surveyed base path; native validation pending')
        return dict(position=[float(xy[0]),float(xy[1]),float(old[2])],route=route,
            screening=dict(room_id=int(room_id),room_name=grid.names.get(room_id),
                extra_floor_clearance_m=float(layout.room_clear[start]),
                contacted_fixture_standoffs_m=dict(zip(contact_entities,clearances)),
                same_component_as_source_and_destination=True,
                front_of_you_preserved=front_required,heading_unchanged=True))
    return None


def repair_robot_spawn(row,evidence,out):
    """User-authorized spawn-only correction for observed body interference."""
    source=Path(row['scenario']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    report=read(Path(evidence)/'validation.json')
    if report.get('scenario_sha256')!=sha or len(row.get('spawn_repairs',[]))>=3:return None
    error=str(report.get('error',''))
    invalid_spawn='invalid spawn' in error.lower()
    state_failure=('EXPERT_CONTAINER_JOINTS_NOT_SETTLED' in error or
                   'OPEN_STATE_CHANGED:' in error or ' open=' in error)
    if not invalid_spawn and not state_failure:return None
    contacts=[];entities=set();log=Path(evidence)/'native_validation.log'
    for line in log.read_text(errors='replace').splitlines() if log.exists() else []:
        if '[open-state-audit] ' not in line:continue
        try:event=json.loads(line.split('[open-state-audit] ',1)[1])
        except ValueError:continue
        for pair in event.get('contacts',[]):
            if any('controllable__' in part or '/robot_' in part for part in pair):
                contacts.append(pair);entities.add(event['entity'])
    if not invalid_spawn and not contacts:return None
    original=yaml.safe_load(source.read_text())
    layout=storage.replay_layout(layout_for_spec(original),row.get('inserted_furniture',[]))
    # Unknown contact geometry cannot justify a distance screen.
    if not entities.issubset(layout.entities):return None
    proposal=nearby_spawn_proposal(original,row,layout,sorted(entities))
    if proposal is None:return None
    spec=copy.deepcopy(original);name=spec['robot']['init_anchor'];before=copy.deepcopy(spec['anchors'][name])
    spec['anchors'][name]['position']=proposal['position']
    # This explicit, scoped exception changes only the authorized position.
    # The normal preservation contract still protects every other field.
    contract=copy.deepcopy(original);contract['anchors'][name]['position']=proposal['position']
    assert_preserved(contract,spec)
    edit=dict(authorization='user_requested_change_initial_spawn_20261114',before=before,
        after=copy.deepcopy(spec['anchors'][name]),source_sha256=sha,
        observed_robot_contacts=contacts,observed_error=report.get('error'),
        screening=proposal['screening'],physical_validation_required=True)
    directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True)
    target=directory/'scenario.yaml';target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
    journal=dict(method='local_initial_spawn_repair',original_recipe=str(source),evidence=str(evidence),edit=edit,
        preserved=['instruction','goal','target','robot','spawn height and orientation','all assets','all placements','all other anchors'],
        physical_validation_required=True)
    (directory/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        route=proposal['route'],spawn_repair_origin=row.get('spawn_repair_origin',before['position']),
        spawn_repairs=row.get('spawn_repairs',[])+[edit]),journal


def repair_explicit_shelf_sampling(row,evidence,out):
    """Use native geometry sampling when an outer-box shelf XY was rejected."""
    source=Path(row['scenario']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if read(Path(evidence)/'validation.json').get('scenario_sha256')!=sha:return None
    log=Path(evidence)/'native_validation.log'
    if not log.exists():return None
    failed=re.search(r'\[placement\] end (\S+) placed=False',log.read_text(errors='replace'))
    original=yaml.safe_load(source.read_text())
    placement=next((p for p in original['placements'] if failed and p['entity']==failed[1]),{})
    if placement.get('relation')!='on_top' or placement.get('position_xy') is None:return None
    old=placement['receptacle'];objects={o['name']:o for o in original['objects']}
    if objects.get(old,{}).get('category') not in ('shelf','bookcase'):return None
    attempt_key=surface_attempt_key(original,old,read(Path(evidence)/'validation.json'))
    if any(edit.get('attempt_key')==attempt_key for edit in row.get('native_surface_sampling_repairs',[])):return None
    group=[p for p in original['placements'] if p['relation']=='on_top' and p['receptacle']==old]
    if len(group)<2 or any(objects.get(p['entity'],{}).get('fixed_base') or
                           p['entity']==row.get('reference_entity') for p in group):return None
    spec=copy.deepcopy(original);changed=[]
    for p in spec['placements']:
        if p['receptacle']==old and p['relation']=='on_top' and p.get('position_xy') is not None:
            changed.append(dict(entity=p['entity'],previous_position_xy=p.pop('position_xy')))
    assert_preserved(original,spec)
    target=Path(out)/row['id']/'scenario.yaml';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
    journal=dict(method='native_on_top_sampling_on_unchanged_shelf',original_recipe=str(source),
        source_sha256=sha,evidence=str(evidence),receptacle=old,changes=changed,attempt_key=attempt_key,
        reason='Outer shelf bounding boxes do not certify a flat support slab. Let the existing native OnTop sampler propose physically supported coordinates on the same asset.',
        preserved=['instruction','goal','target','all assets and counts','robot','spawn','every relation and receptacle'],
        physical_validation_required=True)
    (target.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
    return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                native_surface_sampling_repairs=row.get('native_surface_sampling_repairs',[])+[journal]),journal


def repair_open_surface_source(row,evidence,out,variant,*,preferred_assets=(),reviewed_surface_entity=None,prefer_constructed=False,
                               allowed_assets=(),reviewed_surface_categories=()):
    """Jointly fit failed surface stock on another native source of its kind."""
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text())
    native_report=read(Path(evidence)/'validation.json')
    if native_report.get('scenario_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():return None
    log=Path(evidence)/'native_validation.log'
    if not log.exists():return None
    failed=re.search(r'\[placement\] end (\S+) placed=False',log.read_text(errors='replace'))
    initial_goal_overlap=native_report.get('error')=='INVALID_INITIAL_GOAL'
    entity=reviewed_surface_entity or (failed[1] if failed else original.get('target',{}).get('entity') if initial_goal_overlap else None)
    if entity is None and native_report.get('error')=='native_timeout':
        active=re.findall(r'\[placement\].*? start (\S+) (inside|on_top) (\S+) link=',log.read_text(errors='replace'))
        if active and active[-1][1]=='on_top':entity=active[-1][0]
    if entity is None:
        # Placement may initially succeed and lose OnTop only after closing
        # and settling. Diagnose the final native predicate, too.
        final=re.search(r"(?:\['|', ')(\S+) on_top (\S+)",str(native_report.get('error','')))
        if final:entity=final[1]
    placement=next((p for p in original['placements'] if p['entity']==entity),{})
    if placement.get('relation')!='on_top':return None
    old=placement['receptacle']
    by_entity={p['entity']:p for p in original['placements']}
    source_ancestors=set();cursor=row['source_receptacle']
    while cursor not in source_ancestors:
        source_ancestors.add(cursor)
        if cursor not in by_entity:break
        cursor=by_entity[cursor]['receptacle']
    source_follows_surface=old in source_ancestors
    if initial_goal_overlap:
        # A goal item inside the selected basket can already count as being
        # on the basket's supporting delivery surface. Move that whole source
        # subtree, preserving its internal relations and the goal itself.
        def goal_follows_selected_source(goal):
            if goal.get('kind')!='on_top' or goal.get('receptacle')!=old:return False
            cursor=goal.get('entity');seen=set()
            while cursor not in seen:
                if cursor==placement['entity']:return True
                seen.add(cursor);parent=by_entity.get(cursor,{})
                if parent.get('relation') not in ('inside','on_top'):break
                cursor=parent.get('receptacle')
            return False
        if not any(goal_follows_selected_source(goal) for goal in original.get('goal',[])):return None
    if (old==row['destination_receptacle'] and not initial_goal_overlap) or has_qualified_storage_location(original['instruction']):return None
    base=storage.replay_layout(layout_for_spec(original),row.get('inserted_furniture',[]));entity=base.entities.get(old,{})
    category=entity.get('category')
    if (category=='countertop' and re.search(r'\b(counter|countertop|worktop)\b',storage_scope_text(original['instruction']),re.I)
            and not (initial_goal_overlap and reviewed_surface_entity is not None and reviewed_surface_categories)):
        return None
    profiles={'shelf':('shelf',('shelf','bookcase')),'bookcase':('shelf',('shelf','bookcase')),
              'coffee_table':('table',('coffee_table','breakfast_table','dining_table')),
              'breakfast_table':('table',('breakfast_table','dining_table')),
              'dining_table':('table',('dining_table','breakfast_table')),
              'desk':('desk',('desk',)),
              'bottom_cabinet':('table',('bottom_cabinet','metal_bottom_cabinet')),
              'bottom_cabinet_no_top':('table',('bottom_cabinet_no_top','bottom_cabinet','metal_bottom_cabinet')),
              'metal_bottom_cabinet':('table',('metal_bottom_cabinet','bottom_cabinet')),
              'dresser':('table',('dresser',)),
              'countertop':('table',('countertop','breakfast_table','dining_table'))}
    if category not in profiles:return None
    kind,categories=profiles[category]
    if reviewed_surface_categories:
        # Explicit expert data revision, never an automatic fallback: the
        # selected prop's unmentioned support may use another native class.
        if reviewed_surface_entity is None:return None
        kind,categories='table',tuple(reviewed_surface_categories)
    objects={o['name']:o for o in original['objects']}
    # A generic supply-storage task need not inherit the compiler's shelf
    # choice. Several native bookcases expose only narrow top frame rails;
    # their whole outer box is not a support slab. After a native rejection,
    # an authored, unqualified open source can use a real flat work table.
    # Explicit shelf/bookcase instructions and native scene fixtures retain
    # their furniture class. Every original prop/relation/goal is retained.
    prefer_flat_table=(category in ('shelf','bookcase') and objects.get(old,{}).get('fixed_base') and
        not re.search(r'\b(shelf|shelves|bookshelf|bookshelves|bookcase|rack)\b',storage_scope_text(original['instruction']),re.I))
    if prefer_flat_table:categories=('breakfast_table','dining_table','coffee_table',*categories)
    if re.search(r'\bcoffee[ -]table\b',storage_scope_text(original['instruction']),re.I):
        kind,categories='table',('coffee_table',)
    group=[p for p in original['placements'] if p['receptacle']==old and p['relation']=='on_top']
    if initial_goal_overlap:
        # The compiler can accidentally bind source stock and a matching
        # sample to the delivery surface. Keep that physical sample and the
        # goal in place; only the searched stock needs another support.
        references={row.get('reference_entity'),'comparison_sample'}
        group=[p for p in group if p['entity'] not in references]
    if not group or any(p['entity'] in {row.get('reference_entity'),'comparison_sample'} for p in group):return None
    if any(p['receptacle']==old and p['relation']=='inside' for p in original['placements']):return None
    fixed={o['name'] for o in original['objects'] if o.get('fixed_base')}
    if any(p['entity'] in fixed for p in group):return None
    replace=(old in fixed and any(f['name']==old for f in row.get('inserted_furniture',[])) and
             not any(p.get('entity')==old or p.get('receptacle')==old for p in original.get('goal',[])))
    construction=storage.replay_layout(layout_for_spec(original),[f for f in row.get('inserted_furniture',[]) if not (replace and f['name']==old)])
    sizes=dimensions(original);items={p['entity']:sizes[p['entity']] for p in group};models=[]
    # Prefer an existing suitable surface in the same room before adding
    # furniture to a crowded room. Preserve explicit furniture semantics.
    occupied={p['receptacle'] for p in original['placements']}
    spawn=original['anchors'][original['robot']['init_anchor']]['position']
    sr,sc=base.map.to_map(*spawn[:2])
    for name,body in base.entities.items():
        if prefer_constructed:break
        if (name in occupied or name==row['destination_receptacle'] or body.get('category') not in categories
                or row['source_room'] not in body.get('in_rooms',[]) or not body.get('aabb')):continue
        if np.linalg.norm(g.boxcenter(body)[:2]-g.boxcenter(entity)[:2])>4.:continue
        destination=scene_route_anchor(original,row['destination_receptacle'],base.entities)
        route_source=name if source_follows_surface else scene_route_anchor(original,row['source_receptacle'],base.entities)
        route=base.route(route_source,row['source_room'],destination,row['destination_room'])
        if route is None:continue
        ar,ac=base.map.to_map(*route['start'][:2])
        if not (0<=sr<base.map.size and 0<=sc<base.map.size and base.free[sr,sc]
                and base.labels[sr,sc]==base.labels[ar,ac]):continue
        spec=copy.deepcopy(original)
        for p in spec['placements']:
            if p['entity'] in items:p['receptacle']=name;p.pop('position_xy',None)
        working=dict(spec,placements=[p for p in spec['placements'] if p['entity'] in items])
        plans=plan_explicit_surfaces(working,base,replan_receptacles={name})
        if not plans:continue
        measured=include_task_receptacles(base,spec,evidence=evidence,source_sha=hashlib.sha256(source.read_bytes()).hexdigest())
        pf=preflight_recipe(spec,measured.entities,dimensions(spec))
        if pf['status']!='heuristic_fit':continue
        assert_preserved(original,spec)
        path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
        journal=dict(method='reuse_existing_native_source_surface',old_storage=old,new_storage=name,
            source_room=row['source_room'],plans=plans,original_recipe=str(source),
            physical_validation_required=True,preserved=['instruction','goal','all original assets','spawn','source room','relation kinds'])
        (path.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
        return dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_receptacle=name if row['source_receptacle']==old else row['source_receptacle'],route=route,layout_preflight=pf,
            restoration_entities=list(dict.fromkeys(row.get('restoration_entities',[])+[name]))),journal
    for category in categories:
        for path in (g.ASSETS/'objects'/category).glob('*/misc/metadata.json'):
            size=read(path).get('bbox_size',[])
            if len(size)!=3 or not (.35<size[2]<1.25) or max(size[:2])>2.6:continue
            if any(size[0]<v[0]+.01 or size[1]<v[1]+.01 for v in items.values()):continue
            models.append((float(np.prod(size[:2])),category,path.parent.parent.name,size))
    # Positive native examples are preferences, not certificates for these
    # contents. Prefer a different proven fixture to repeatedly proposing the
    # same failed model solely because its outer AABB is small.
    from track_full_asset_queue import accepted_native_pair
    proven=set()
    preferred_native_sampling=set()
    for asset,record in read(g.ROOT/'build/verified_surface_sites/index.json').get('assets',{}).items():
        if accepted_native_pair(record['recipe'],record['evidence']):proven.add(asset)
        native=record.get('native_sampling_evidence',{})
        if native and accepted_native_pair(native['recipe'],native['evidence']):preferred_native_sampling.add(asset)
    current_asset=(entity.get('category'),entity.get('model'))
    if allowed_assets:models=[m for m in models if m[1]+'/'+m[2] in allowed_assets]
    models.sort(key=lambda model:(bool(preferred_assets and model[1]+'/'+model[2] not in preferred_assets),
        bool(prefer_flat_table and model[1] in ('shelf','bookcase')),
        (model[1],model[2])==current_asset,
        model[1]+'/'+model[2] not in proven,model[0],model[1],model[2]))
    previous=storage.MODELS.get(kind);room=row['source_room']
    try:
        for _,category,model,size in models:
            solved=solve_surface_proposal(items,size[:2])
            if solved['status']!='feasible_model':continue
            storage.MODELS[kind]=(category,model)
            for site in range(8):
                name=old if replace else f'expert_surface_{variant}'
                proposed=storage.augment(construction,room,kind,name,variant=site)
                if proposed is None:continue
                layout,_=proposed;added=next(f for f in layout.inserted_furniture if f['name']==name)
                if np.linalg.norm(np.asarray(added['position_xy'])-g.boxcenter(entity)[:2])>4.:continue
                spawn=original['anchors'][original['robot']['init_anchor']]['position'];rr,cc=layout.map.to_map(*spawn[:2])
                if not (0<=rr<layout.map.size and 0<=cc<layout.map.size and layout.free[rr,cc]):continue
                destination_anchor=scene_route_anchor(original,row['destination_receptacle'],layout.entities)
                route_source=name if source_follows_surface else scene_route_anchor(original,row['source_receptacle'],layout.entities)
                route=layout.route(route_source,room,destination_anchor,row['destination_room'])
                if route is None:continue
                ar,ac=layout.map.to_map(*route['start'][:2])
                if layout.labels[rr,cc]!=layout.labels[ar,ac]:continue
                spec=copy.deepcopy(original);replacement=dict(name=name,category=category,model=model,fixed_base=True)
                if replace:
                    obj=next(o for o in spec['objects'] if o['name']==old);spec['objects'][spec['objects'].index(obj)]=replacement
                    floor=next(p for p in spec['placements'] if p['entity']==old);floor.update(receptacle=added['floor'],position_xy=added['position_xy'])
                else:
                    spec['objects'].append(replacement);spec['placements'].append(dict(entity=name,relation='on_top',receptacle=added['floor'],position_xy=added['position_xy']))
                lo=np.asarray(added['aabb'][0])[:2]
                for p in spec['placements']:
                    if p['entity'] in items:
                        p['receptacle']=name
                        if category+'/'+model in preferred_native_sampling:p.pop('position_xy',None)
                        else:p['position_xy']=(lo+np.asarray(solved['poses'][p['entity']]['center_xy_m'])).tolist()
                measured=include_task_receptacles(layout,spec,evidence=evidence,source_sha=hashlib.sha256(source.read_bytes()).hexdigest())
                pf=preflight_recipe(spec,measured.entities,dimensions(spec))
                if pf['status']!='heuristic_fit':continue
                contract=copy.deepcopy(original)
                if replace:
                    obj=next(o for o in contract['objects'] if o['name']==old);contract['objects'][contract['objects'].index(obj)]=replacement
                assert_preserved(contract,spec)
                if spec==original:continue
                path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
                doc=dict(status='static_screened_physics_pending',method='joint_native_open_surface_layout',source_kind=kind,
                    initial_goal_overlap_repaired=initial_goal_overlap,
                    original_recipe=str(source),old_storage=old,new_storage=added,replaced_auxiliary_fixture=replace,
                    reviewed_surface_entity=reviewed_surface_entity,
                    reviewed_surface_categories=list(reviewed_surface_categories),
                    delivery_entity=row['destination_receptacle'],private_route_destination_anchor=destination_anchor,
                    solver=solved,physical_validation_required=True,
                    coordinate_proposal='native_OnTop_sampling' if category+'/'+model in preferred_native_sampling else 'offline_rectangle_XY',
                    preserved=['instruction','goal','target','robot','spawn','all task items and counts','all placement relations'])
                (path.parent/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
                return dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    inserted_furniture=layout.inserted_furniture,object_count=len(spec['objects']),source_receptacle=name if row['source_receptacle']==old else row['source_receptacle'],
                    route=route,layout_preflight=pf,restoration_entities=list(dict.fromkeys(row.get('restoration_entities',[])+[name]))),doc
    finally:
        if previous is None:storage.MODELS.pop(kind,None)
        else:storage.MODELS[kind]=previous
    return None


def apply_registered_scene_contact_repair(row,evidence,out):
    """Apply an evidence-bound scene edit; task objects and predicates stay put."""
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text());scene=original['scene']['model']
    registered=read(g.ROOT/'build/expert_scene_contact_repairs/index.json').get('scenes',{}).get(scene)
    if not registered:return None
    prior=read(original['scene'].get('clearance_manifest','')) or dict(scene_model=scene,wall_warps=[],translations=[],removed_objects=[])
    protected={o['name'] for o in original['objects']}|set(original.get('initial_states',{}))
    for p in [*original['placements'],*original.get('goal',[])]:protected.update(p[k] for k in ('entity','receptacle') if p.get(k))
    decision=None
    for candidate in [registered,*registered.get('additional_rules',[])]:
        name=candidate['source_entity']
        relevant=(original.get('initial_states',{}).get(name)==candidate['required_declared_state']
                  if candidate.get('required_declared_state') is not None else
                  any(p['receptacle']==name and p['relation']=='inside' for p in original['placements']))
        if not relevant or protected.intersection(candidate['removed_objects']):continue
        if any(d.get('revision_id')==candidate['revision_id'] for d in row.get('scene_contact_repairs',[])):continue
        required=candidate.get('required_translation')
        if required and not any(t['entity']==required['entity'] and
                np.allclose(t['delta'],required['delta'],atol=1e-8,rtol=0) for t in prior.get('translations',[])):continue
        if candidate.get('positive_native_pairs'):
            from track_full_asset_queue import accepted_native_pair
            if not all(accepted_native_pair(g.ROOT/pair['recipe'],g.ROOT/pair['evidence'])
                       for pair in candidate['positive_native_pairs']):continue
        decision=candidate;break
    if decision is None:return None
    name=decision['source_entity']
    base=layout_for(scene)
    if hashlib.sha256(base.path.read_bytes()).hexdigest()!=decision['base_survey_sha256']:raise ValueError('CONTACT_REPAIR_SURVEY_CHANGED')
    entity=base.entities[name]
    if entity['category']+'/'+entity['model']!=decision['source_asset']:raise ValueError('CONTACT_REPAIR_SOURCE_CHANGED')
    evidence_path=Path(decision['native_failure_evidence'])
    if not evidence_path.is_file() or hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=decision['native_failure_evidence_sha256']:
        raise ValueError('CONTACT_REPAIR_EVIDENCE_CHANGED')
    moved={c['entity'] for c in decision['translations']}
    if any(n not in base.entities for n in moved|set(decision['removed_objects'])):raise ValueError('CONTACT_REPAIR_ENTITY_MISSING')
    manifest=copy.deepcopy(prior)
    removed=set(decision['removed_objects'])
    manifest['translations']=[c for c in manifest.get('translations',[]) if c['entity'] not in removed and not (
        c['entity'] in moved and (c.get('expert_scene_nudge') or c.get('expert_scene_contact_repair')))]
    manifest['translations'] += [dict(c,expert_scene_contact_repair=True) for c in decision['translations']]
    manifest['removed_objects']=sorted(set(manifest.get('removed_objects',[]))|set(decision['removed_objects']))
    manifest['expert_scene_translations']=[c for c in manifest.get('expert_scene_translations',[]) if c['entity'] not in moved|removed]+decision['translations']
    manifest.update(expert_base_survey_sha256=decision['base_survey_sha256'],status='expert_contact_repair_physics_pending',
                    contact_repair_revision=decision['revision_id'])
    directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True)
    clearance=directory/('clearance_'+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]+'.json')
    clearance.write_text(json.dumps(manifest,indent=2));spec=copy.deepcopy(original);spec['scene']['clearance_manifest']=str(clearance.resolve())
    released=clear_unstated_links(spec)
    layout=storage.replay_layout(layout_for_spec(spec),row.get('inserted_furniture',[]))
    spawn=spec['anchors'][spec['robot']['init_anchor']]['position'];rr,cc=layout.map.to_map(*spawn[:2])
    if not (0<=rr<layout.map.size and 0<=cc<layout.map.size and layout.free[rr,cc]):return None
    route=layout.route(row['source_receptacle'],row['source_room'],row['destination_receptacle'],row['destination_room'])
    if route is None:return None
    ar,ac=layout.map.to_map(*route['start'][:2])
    if layout.labels[rr,cc]!=layout.labels[ar,ac]:return None
    pf=preflight_recipe(spec,layout.entities,dimensions(spec))
    if pf['status']!='heuristic_fit':return None
    assert_preserved(original,spec)
    path=directory/'scenario.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    doc=dict(status='static_screened_physics_pending',method='registered_native_contact_layout_repair',decision=decision,released_compiler_links=released,
        original_recipe=str(source),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        physical_validation_required=True,preserved=['instruction','goal','target','robot','spawn','all task assets and counts','all placement relations'])
    (directory/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
    return dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),layout_preflight=pf,route=route,
                scene_contact_repairs=row.get('scene_contact_repairs',[])+[dict(revision_id=decision['revision_id'],clearance_sha256=hashlib.sha256(clearance.read_bytes()).hexdigest())]),doc


def nudge_interfering_scene_storage(row,evidence,out):
    """Move a wall-contacting scene cabinet a few centimetres, with provenance.

    This is scene authoring, never a joint lock or collision exemption. Keep
    its native model, all contents, reference surface and delivery identity.
    """
    # A recorded edit elsewhere in the house does not resolve a newly
    # observed contact on this cabinet. Every proposal below is still tied
    # to this exact recipe/report and bounded per affected entity.
    recipe=Path(row['scenario']);sha=hashlib.sha256(recipe.read_bytes()).hexdigest()
    report=read(Path(evidence)/'validation.json')
    error=str(report.get('error',''))
    state_failure=('EXPERT_CONTAINER_JOINTS_NOT_SETTLED' in error or 'OPEN_STATE_CHANGED:' in error or ' open=' in error)
    if report.get('scenario_sha256')!=sha or not state_failure:return None
    original=yaml.safe_load(recipe.read_text());name=blocking_storage(original,evidence)
    if sum(edit.get('entity')==name for edit in row.get('scene_storage_nudges',[]))>=3:return None
    if name in {o['name'] for o in original['objects']}:return None
    base=layout_for(original['scene']['model']);entity=base.entities.get(name)
    if not entity or not entity.get('fixed_base') or not entity.get('openable'):return None
    contacts=[];opening_travel=0.
    for line in (Path(evidence)/'native_validation.log').read_text(errors='replace').splitlines():
        if '[authoring-opening] ' in line:
            try:
                opening=json.loads(line.split('[authoring-opening] ',1)[1])
                if opening.get('entity')==name:opening_travel=max(opening_travel,max((abs(t['position']) for t in opening.get('targets',[])),default=0.))
            except (ValueError,KeyError,TypeError):pass
        if '[open-state-audit] ' not in line:continue
        try:event=json.loads(line.split('[open-state-audit] ',1)[1])
        except ValueError:continue
        if event.get('entity')!=name:continue
        for pair in event.get('contacts',[]):
            # Native articulated prim names may be prefixed articulated__.
            belongs=[('/'+name+'/' in p or '/articulated__'+name+'/' in p) for p in pair]
            if sum(belongs)==1:contacts.append(pair)
    if not contacts:return None
    center=g.boxcenter(entity);spawn=np.asarray(original['anchors'][original['robot']['init_anchor']]['position'])
    direction=spawn[:2]-center[:2];length=np.linalg.norm(direction)
    if length<.01:return None
    direction/=length;normal=np.array([-direction[1],direction[0]])
    # A drawer may extend through the wall despite a collision-free closed
    # cabinet. Use the observed native prismatic travel, not just an 8-cm
    # closed-box nudge. A thin surveyed wall supplies a geometric normal.
    candidates=[]
    geometry=read(g.ROOT/'build/native_fillable_geometry/index.json').get('assets',{}).get(
        entity['category']+'/'+entity['model'],{})
    prismatic=any(j.get('type')=='PhysicsPrismaticJoint' for j in geometry.get('joints',[]))
    for other,wall in base.entities.items():
        if wall.get('category')!='walls' or not any(any('/'+other+'/' in path for path in pair) for pair in contacts):continue
        lo,hi=np.asarray(wall['aabb']);width=hi[:2]-lo[:2];axis=int(np.argmin(width))
        if width[axis]>.3 or width[1-axis]<3*width[axis]:continue
        sign=1. if center[axis]>(lo[axis]+hi[axis])/2 else -1.
        away=np.zeros(2);away[axis]=sign
        source_lo,source_hi=np.asarray(entity['aabb'])
        penetration=max(0.,float(hi[axis]-source_lo[axis] if sign>0 else source_hi[axis]-lo[axis]))
        travel=opening_travel if prismatic and 0.<opening_travel<.6 else min(.5,float(max(source_hi[:2]-source_lo[:2])))
        required=travel+penetration+.03
        for extra in [0.,.08,.16]:
            if required+extra<=.65:candidates.append(away*(required+extra))
    robot_contact=any(any('robot_' in part or 'controllable__' in part for part in pair) for pair in contacts)
    # Robot interference is now handled by the explicitly authorized local
    # spawn repair above. Do not silently substitute a furniture translation
    # when no suitable local start was found.
    if robot_contact:return None
    axes=[direction,normal,-normal,-direction]
    candidates.extend(axis*distance for axis in axes for distance in [.08,.16,.24,.32])
    tried={tuple(np.round(x['delta'][:2],6)) for x in row.get('scene_storage_nudges',[]) if x['entity']==name}
    old_manifest=Path(original['scene']['clearance_manifest']) if original['scene'].get('clearance_manifest') else None
    prior=read(old_manifest) if old_manifest else dict(scene_model=original['scene']['model'],wall_warps=[],translations=[],removed_objects=[])
    for displacement in candidates:
            delta=[*list(displacement),0.]
            if tuple(np.round(delta[:2],6)) in tried:continue
            # The cabinet centre must remain in its original semantic room.
            rr,cc=base.map.to_map(*(center[:2]+delta[:2]));ids=[i for i,r in base.map.names.items() if r in entity.get('in_rooms',[])]
            if not (0<=rr<base.map.size and 0<=cc<base.map.size and base.map.room_ins[rr,cc] in ids):continue
            changes=[c for c in prior.get('expert_scene_translations',[]) if c['entity']!=name]+[dict(entity=name,delta=delta)]
            manifest=copy.deepcopy(prior)
            manifest['translations']=[c for c in manifest.get('translations',[]) if not (c.get('expert_scene_nudge') and c['entity']==name)]
            manifest['translations'].append(dict(entity=name,delta=delta,expert_scene_nudge=True))
            manifest.update(expert_scene_translations=changes,expert_base_survey_sha256=hashlib.sha256(base.path.read_bytes()).hexdigest(),
                status='expert_authoring_physics_pending',derived_from=str(old_manifest) if old_manifest else None)
            directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True)
            clearance=directory/('clearance_'+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]+'.json')
            clearance.write_text(json.dumps(manifest,indent=2));spec=copy.deepcopy(original);spec['scene']['clearance_manifest']=str(clearance.resolve())
            layout=storage.replay_layout(layout_for_spec(spec),row.get('inserted_furniture',[]))
            sr,sc=layout.map.to_map(*spawn[:2])
            if not (0<=sr<layout.map.size and 0<=sc<layout.map.size and layout.free[sr,sc]):continue
            room=(entity.get('in_rooms') or [row['source_room']])[0]
            route=layout.route(name,room,row['destination_receptacle'],row['destination_room'])
            if route is None:continue
            ar,ac=layout.map.to_map(*route['start'][:2])
            if layout.labels[sr,sc]!=layout.labels[ar,ac]:continue
            pf=preflight_recipe(spec,layout.entities,dimensions(spec))
            if pf['status']!='heuristic_fit':continue
            assert_preserved(original,spec)
            path=directory/'scenario.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
            edit=dict(entity=name,delta=delta,observed_native_opening_travel_m=opening_travel if prismatic else None,
                observed_external_contacts=contacts,source_sha256=sha,
                clearance_sha256=hashlib.sha256(clearance.read_bytes()).hexdigest(),physical_validation_required=True)
            doc=dict(status='static_screened_physics_pending',method='translate_contacting_scene_storage',
                edit=edit,original_recipe=str(recipe),diagnosis_directory=str(evidence),
                preserved=['instruction','goal','target','robot','spawn','all native assets','entity identities','all placement relations'])
            (directory/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
            return dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        layout_preflight=pf,scene_storage_nudges=row.get('scene_storage_nudges',[])+[edit],route=route),doc
    return None


def repair_native_blocking_subset(row,evidence,out,variant):
    """Move only a diagnosed nonfitting subset when language allows sources."""
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text())
    if not source_allows_multiple_fixtures(original['instruction']):return None
    report=read(Path(evidence)/'validation.json')
    if report.get('scenario_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest() or 'EXPERT_LAYOUT_UNRESOLVED: infeasible_model' not in str(report.get('error','')):return None
    log=Path(evidence)/'native_validation.log';plans=[]
    if not log.exists():return None
    for line in log.read_text(errors='replace').splitlines():
        if '[expert-layout-plan] ' in line:
            try:plans.append(json.loads(line.split('[expert-layout-plan] ',1)[1]))
            except ValueError:pass
    if not plans:return None
    plan=plans[-1];blocked=set(plan.get('blocking_entities',[]));old=plan.get('receptacle')
    group={p['entity'] for p in original['placements'] if p['relation']=='inside' and p['receptacle']==old}
    if not blocked or not blocked<group:return None
    result,doc=curate(row,evidence,out,variant=variant,old_storage=old,group_entities=blocked,
                      allow_patch_support=True,prefer_constructed=True)
    if result is None:return None
    return result,dict(doc,native_blocking_subset=sorted(blocked),native_solver_evidence=plan,
                       reason='Preserve all objects; collective source description permits a neighbouring physical fixture')


def repair_scene_fixture_in_place(row,evidence,out,*,allow_unanchored_scene_source=False,
                                  allow_shared_delivery_replacement=False,preferred_assets=()):
    """Replace a diagnosed scene source with a fitting native fixture in situ.

    Adding a second cabinet in a small bathroom can consume its only approach
    patch. Replace the source furniture at its existing site instead. The
    physical source site, props, instruction and goals are preserved; the removed
    scene instance is replaced by a real collidable native asset, never a
    collision exemption. A private replacement alias avoids a name collision
    before the old instance is removed. All native checks and fresh reload remain required.
    """
    recipe=Path(row['scenario']);sha=hashlib.sha256(recipe.read_bytes()).hexdigest()
    report=read(Path(evidence)/'validation.json')
    if report.get('scenario_sha256')!=sha or report.get('status')!='failed':return None
    original=yaml.safe_load(recipe.read_text());name=blocking_storage(original,evidence)
    if not name:return None
    authored=name in {o['name'] for o in original['objects']}
    if authored and not editable_authored_storage(original,row,name):return None
    # In-place identity/location preservation permits "left cabinet" or
    # "lower cupboards". An ordinal drawer needs a manually verified mapping
    # of the new fixture's particular moving links, so leave it untouched.
    if re.search(r'\b(upper|lower|top|bottom|first|second|third)\s+(?:\w+\s+)?drawer\b',
                 storage_scope_text(original['instruction']),re.I):return None
    shared_delivery=(name==row['destination_receptacle'] or
        any(p.get('entity')==name or p.get('receptacle')==name for p in original.get('goal',[])) or
        any(p['relation']=='on_top' and p['receptacle']==name for p in original['placements']))
    if shared_delivery and not allow_shared_delivery_replacement:return None
    if any(p.get('entity')==name for p in original.get('goal',[])):return None
    base=storage.replay_layout(layout_for_spec(original),row.get('inserted_furniture',[]))
    entity=base.entities.get(name,{})
    # Surveyed fixtures carry fixed_base, whereas replayed authored fixtures
    # derive that property from the actual recipe object, not the AABB cache.
    fixed = entity.get('fixed_base') or any(o['name']==name and o.get('fixed_base')
                                          for o in original['objects'])
    if not entity.get('aabb'):return None
    replace_dynamic_fixture=not fixed
    if replace_dynamic_fixture:
        if (not allow_unanchored_scene_source or authored or entity.get('graspable') is not False or
                entity.get('category') not in ('bottom_cabinet','bottom_cabinet_no_top','metal_bottom_cabinet','wardrobe','nightstand','dresser')):return None
        # Expert-only scene reauthoring: replace the whole failed furniture
        # with another real anchored fixture, never freeze its task contents.
        # Do not strand native decorations resting on the old cabinet top.
        a,b=np.asarray(entity['aabb'],float)
        for other,body in base.entities.items():
            if other==name or not body.get('aabb'):continue
            low,high=np.asarray(body['aabb'],float);center=(low+high)/2
            if abs(low[2]-b[2])<.04 and np.all(center[:2]>=a[:2]) and np.all(center[:2]<=b[:2]):return None
    group=[p for p in original['placements'] if p['relation']=='inside' and p['receptacle']==name]
    if not group:return None
    kind=repair_kind(entity,group,original['instruction'])
    if kind is None:return None
    lo,hi=np.asarray(entity['aabb'],float);extent=hi-lo;xy=((lo+hi)/2)[:2]
    room=(entity.get('in_rooms') or [row['source_room']])[0]
    sizes=dimensions(original);items={p['entity']:sizes[p['entity']] for p in group}
    floors=[(n,f) for n,f in base.entities.items() if f.get('category')=='floors' and f.get('aabb')
            and abs(float(f['aabb'][1][2])-lo[2])<.12]
    candidates=[]
    for category,model in candidate_models(kind,room):
        if (category,model)==(entity['category'],entity['model']) or not storage.model_openable(category,model):continue
        size=np.asarray(read(g.ASSETS/'objects'/category/model/'misc/metadata.json').get('bbox_size',[]),float)
        if len(size)!=3 or not np.isfinite(size).all() or np.any(size<=0):continue
        # Keep scene-asset substitutions close to their original envelope.
        # For our own unqualified support fixture the floor, other bodies,
        # robot footprint and route checks below determine whether it fits;
        # the failed fixture's arbitrary width is not a physical constraint.
        auxiliary_unqualified=(authored and not has_qualified_storage_location(original['instruction']))
        if not auxiliary_unqualified and np.any(size[:2]>extent[:2]+.12):continue
        # An auxiliary cabinet has no semantic height contract unless the
        # instruction selects a particular spatial level. A smaller real
        # cabinet at the same site can replace an unusable tall fixture.
        height_reduction=(authored and kind=='cabinet' and size[2]<=extent[2]
                          and not has_qualified_storage_location(original['instruction']))
        height_expansion=(authored and kind=='cabinet' and size[2]<=1.3
                          and not has_qualified_storage_location(original['instruction']))
        if abs(size[2]-extent[2])>.2 and not (height_reduction or height_expansion):continue
        proposed=dict(category=category,model=model,interiors=[])
        storage.enrich_constructed_interiors({'candidate':proposed})
        packed=compartment_assignment(proposed,items,kind,native_size=True,allow_patch_support=True,
            support_entities=flat_support_entities(original,group),
            single_drawer=bool(re.search(r'\bdrawer\b',storage_scope_text(original['instruction']),re.I)))
        if packed is None:continue
        candidates.append((float(np.linalg.norm(size-extent)),category,model,size,packed))
    replacement_name=name if authored else 'replacement_'+name
    if not authored and (replacement_name in base.entities or replacement_name in {o['name'] for o in original['objects']}):return None
    preferred={tuple(asset):i for i,asset in enumerate(preferred_assets)}
    for _,category,model,size,(assignment,solved) in sorted(candidates,
            key=lambda x:(preferred.get((x[1],x[2]),len(preferred)),*x[:3])):
        for floor,floor_entity in floors:
            if not storage.footprint_fits_floor(xy,size,floor_entity):continue
            z=float(floor_entity['aabb'][1][2]);newlo=[*list(xy-size[:2]/2),z];newhi=[*list(xy+size[:2]/2),z+float(size[2])]
            if any(n not in (name,floor) and b.get('category')!='floors' and b.get('aabb') and
                   np.all(np.minimum(newhi,b['aabb'][1])-np.maximum(newlo,b['aabb'][0])>.01)
                   for n,b in base.entities.items()):continue
            inserted=dict(name=replacement_name,category=category,model=model,floor=floor,position_xy=xy.tolist(),
                          aabb=[newlo,newhi],room=room)
            furniture=[f for f in row.get('inserted_furniture',[]) if f['name']!=name]+[inserted]
            layout=storage.replay_layout(layout_for_spec(original),furniture)
            destination=scene_route_anchor(original,row['destination_receptacle'],layout.entities)
            if shared_delivery and destination==name:destination=replacement_name
            route=layout.route(replacement_name,room,destination,row['destination_room'])
            if route is None:continue
            spawn=original['anchors'][original['robot']['init_anchor']]['position']
            rr,cc=layout.map.to_map(*spawn[:2]);ar,ac=layout.map.to_map(*route['start'][:2])
            spawn_free=(0<=rr<layout.map.size and 0<=cc<layout.map.size and layout.free[rr,cc]
                    and layout.labels[rr,cc]==layout.labels[ar,ac])
            spawn_change=None
            if not spawn_free:
                if len(row.get('spawn_repairs',[]))>=3:continue
                spawn_row=dict(row,source_receptacle=replacement_name,source_room=room,
                    destination_receptacle=destination)
                spawn_change=nearby_spawn_proposal(original,spawn_row,layout,[replacement_name])
                if spawn_change is None:continue
                route=spawn_change['route']
            spec=copy.deepcopy(original)
            spawn_edit=None
            if spawn_change is not None:
                anchor=spec['robot']['init_anchor'];before=copy.deepcopy(spec['anchors'][anchor])
                spec['anchors'][anchor]['position']=spawn_change['position']
                spawn_edit=dict(authorization='user_requested_change_initial_spawn_20261114',
                    before=before,after=copy.deepcopy(spec['anchors'][anchor]),source_sha256=sha,
                    reason='Same-site replacement occupies or disconnects the original start footprint',
                    screening=spawn_change['screening'],physical_validation_required=True)
            prior=read(original['scene'].get('clearance_manifest','')) or dict(scene_model=original['scene']['model'],wall_warps=[],translations=[],removed_objects=[])
            clearance=copy.deepcopy(prior)
            if not authored:clearance['removed_objects']=sorted(set(clearance.get('removed_objects',[]))|{name})
            replacement=dict(entity=name,replacement_entity=replacement_name,before_asset=entity['category']+'/'+entity['model'],after_asset=category+'/'+model,
                before_aabb=entity['aabb'],after_aabb=inserted['aabb'],source_sha256=sha,evidence=str(evidence),
                reason='Replace failed source at its existing site; adding another fixture consumes its approach space',
                physical_validation_required=True)
            clearance.setdefault('expert_fixture_replacements',[]).append(replacement)
            directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True)
            clearance_path=directory/('clearance_'+hashlib.sha256(json.dumps(clearance,sort_keys=True).encode()).hexdigest()[:16]+'.json')
            clearance_path.write_text(json.dumps(clearance,indent=2));spec['scene']['clearance_manifest']=str(clearance_path.resolve())
            replacement_object=dict(name=replacement_name,category=category,model=model,fixed_base=True)
            if authored:
                index=next(i for i,o in enumerate(spec['objects']) if o['name']==name);spec['objects'][index]=replacement_object
                position=next(p for p in spec['placements'] if p['entity']==name)
                position.update(receptacle=floor,position_xy=xy.tolist())
            else:
                spec['objects'].append(replacement_object)
                spec['placements'].append(dict(entity=replacement_name,relation='on_top',receptacle=floor,position_xy=xy.tolist()))
            for p in spec['placements']:
                if p['entity'] in assignment:p.update(receptacle=replacement_name,link=assignment[p['entity']])
                elif shared_delivery and p['receptacle']==name:
                    p['receptacle']=replacement_name;p.pop('link',None)
            if shared_delivery:
                for goal in spec.get('goal',[]):
                    if goal.get('receptacle')==name:goal['receptacle']=replacement_name
            if name in spec.get('initial_states',{}):spec['initial_states'][replacement_name]=spec['initial_states'].pop(name)
            clear_unstated_links(spec)
            pf=preflight_recipe(spec,layout.entities,dimensions(spec))
            if pf['status']!='heuristic_fit':continue
            contract=copy.deepcopy(original)
            if spawn_change is not None:
                contract['anchors'][spec['robot']['init_anchor']]['position']=spawn_change['position']
            if shared_delivery:
                # The alias denotes the replacement at the same physical
                # site; verify semantic equality after canonicalizing it.
                for goal in contract.get('goal',[]):
                    if goal.get('receptacle')==name:goal['receptacle']=replacement_name
            assert_task_inventory_preserved(contract,spec,editable_fixture_names={name} if authored else ())
            target=directory/'scenario.yaml';target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
            journal=dict(method='replace_native_scene_source_at_same_site',edit=replacement,
                expert_replaced_unanchored_scene_fixture=replace_dynamic_fixture,
                initial_spawn_change=spawn_edit,
                shared_delivery_alias=(dict(before=name,after=replacement_name,
                    physical_site_unchanged=True) if shared_delivery else None),
                initial_closed_state_retained=original.get('initial_states',{}).get(name),solver=solved,
                preserved=['instruction','goal','physical source site and room','initial source open/closed state','target','all task props','robot','spawn height and heading','all original relation kinds'],
                physical_validation_required=True)
            (directory/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
            return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                inserted_furniture=furniture,object_count=len(spec['objects']),layout_preflight=pf,route=route,
                source_receptacle=replacement_name if row['source_receptacle']==name else row['source_receptacle'],
                destination_receptacle=replacement_name if shared_delivery and row['destination_receptacle']==name else row['destination_receptacle'],
                restoration_entities=[replacement_name if n==name else n for n in row.get('restoration_entities',[])],
                spawn_repairs=row.get('spawn_repairs',[])+([spawn_edit] if spawn_edit else []),
                spawn_repair_origin=row.get('spawn_repair_origin',spawn),
                scene_fixture_replacements=row.get('scene_fixture_replacements',[])+[replacement]),journal
    return None


def anchor_native_support_fixture(row,evidence,out,entity):
    """Explicit scene revision: same native support, same site, anchored base.

    Never use this for a task prop or container. Some native furniture has a
    free base; replacing that support with an authored anchored copy removes
    an irrelevant moving-table interaction without freezing any searched or
    manipulated object. No geometry, task or acceptance threshold changes.
    """
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text())
    if entity in {o['name'] for o in original['objects']}:return None
    base=layout_for_spec(original);body=base.entities.get(entity,{})
    if (body.get('fixed_base') is not False or body.get('graspable') is not False
            or body.get('category') not in ('breakfast_table','dining_table','desk','coffee_table')
            or not body.get('aabb')):return None
    lo,hi=np.asarray(body['aabb'],float);size=np.asarray(read(g.ASSETS/'objects'/body['category']/body['model']/'misc/metadata.json').get('bbox_size',[]))
    if size.shape!=(3,) or np.max(np.abs(size-(hi-lo)))>.005:return None
    # This bounded adapter supports only the same upright native envelope.
    # A rotated/scaled scene instance needs an explicit full pose mapping.
    manifest=copy.deepcopy(read(original['scene']['clearance_manifest']) if original['scene'].get('clearance_manifest') else {})
    if any(x['entity']==entity for x in manifest.get('translations',[])):return None
    floors=[(n,b) for n,b in base.entities.items() if b.get('category')=='floors' and b.get('aabb')
        and abs(float(b['aabb'][1][2])-lo[2])<.03
        and storage.footprint_fits_floor(((lo+hi)/2)[:2],size,b)]
    if not floors:return None
    floor,_=min(floors,key=lambda p:np.prod(np.asarray(p[1]['aabb'][1])[:2]-p[1]['aabb'][0][:2]))
    name='anchored_'+entity
    if name in base.entities:return None
    directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True)
    manifest['removed_objects']=list(dict.fromkeys(manifest.get('removed_objects',[])+[entity]))
    manifest.update(scene_model=original['scene']['model'],status='explicit_native_furniture_anchor_physics_pending',
        expert_anchor_revision=dict(entity=entity,replacement=name,same_native_asset=True,same_site=True,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),evidence=str(evidence)))
    clearance=directory/'anchored_support_clearance.json';clearance.write_text(json.dumps(manifest,indent=2)+'\n')
    spec=copy.deepcopy(original);spec['scene']['clearance_manifest']=str(clearance.resolve())
    spec['objects'].append(dict(name=name,category=body['category'],model=body['model'],fixed_base=True))
    for p in spec['placements']:
        if p['receptacle']==entity:p['receptacle']=name
    for goal in spec.get('goal',[]):
        if goal.get('receptacle')==entity:goal['receptacle']=name
    spec['placements'].append(dict(entity=name,relation='on_top',receptacle=floor,position_xy=((lo+hi)/2)[:2].tolist()))
    furniture=row.get('inserted_furniture',[])+[dict(name=name,category=body['category'],model=body['model'],
        floor=floor,position_xy=((lo+hi)/2)[:2].tolist(),aabb=[lo.tolist(),hi.tolist()],room=row['source_room'])]
    layout=storage.replay_layout(layout_for_spec(spec),furniture)
    source_name=name if row['source_receptacle']==entity else row['source_receptacle']
    dest_name=name if row['destination_receptacle']==entity else row['destination_receptacle']
    route=layout.route(scene_route_anchor(spec,source_name,layout.entities),row['source_room'],
        scene_route_anchor(spec,dest_name,layout.entities),row['destination_room'])
    if route is None:return None
    measured=include_task_receptacles(layout,spec,evidence=evidence,source_sha=hashlib.sha256(source.read_bytes()).hexdigest())
    pf=preflight_recipe(spec,measured.entities,dimensions(spec))
    if pf['status']!='heuristic_fit':return None
    contract=copy.deepcopy(original)
    for goal in contract.get('goal',[]):
        if goal.get('receptacle')==entity:goal['receptacle']=name
    assert_preserved(contract,spec)
    target=directory/'scenario.yaml';target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
    journal=dict(method='same_native_support_fixture_anchored_at_same_site',old_entity=entity,new_entity=name,
        original_recipe=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        same_native_asset=body['category']+'/'+body['model'],same_native_envelope=[lo.tolist(),hi.tolist()],
        reason='Review an anchored copy of freely based native support furniture; all searched and manipulated props remain dynamic.',
        physical_validation_required=True)
    (directory/'expert_intervention.json').write_text(json.dumps(journal,indent=2)+'\n')
    return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        inserted_furniture=furniture,source_receptacle=source_name,destination_receptacle=dest_name,route=route,
        layout_preflight=pf,object_count=len(spec['objects']),source_layout_revisions=row.get('source_layout_revisions',[])+[journal]),journal


def replace_authored_reference_surface(row,evidence,out,asset):
    """Reviewed native fixture exchange at the unchanged reference/goal site."""
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text());name=row['destination_receptacle']
    obj=next((o for o in original['objects'] if o['name']==name),None)
    fixture=next((f for f in row.get('inserted_furniture',[]) if f['name']==name),None)
    if (not obj or not obj.get('fixed_base') or not fixture or name in row['target_entities']
            or any(g.get('entity')==name for g in original.get('goal',[]))):return None
    category,model=asset.split('/');size=np.asarray(read(g.ASSETS/'objects'/category/model/'misc/metadata.json').get('bbox_size',[]))
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):return None
    spec=copy.deepcopy(original);replacement=dict(obj,category=category,model=model)
    spec['objects'][spec['objects'].index(obj)]=replacement
    xy=np.asarray(fixture['position_xy']);bottom=float(fixture['aabb'][0][2])
    changed=copy.deepcopy(fixture);changed.update(category=category,model=model,
        aabb=[[*list(xy-size[:2]/2),bottom],[*list(xy+size[:2]/2),bottom+float(size[2])]])
    furniture=[changed if f['name']==name else f for f in row['inserted_furniture']]
    base=layout_for_spec(original);floor=base.entities.get(fixture['floor'],{})
    if not floor or not storage.footprint_fits_floor(xy,size,floor):return None
    layout=storage.replay_layout(base,furniture)
    for other,body in layout.entities.items():
        if other in (name,fixture['floor']) or body.get('category')=='floors' or not body.get('aabb'):continue
        if np.all(np.minimum(changed['aabb'][1],body['aabb'][1])-np.maximum(changed['aabb'][0],body['aabb'][0])>.01):return None
    group=[p for p in spec['placements'] if p['relation']=='on_top' and p['receptacle']==name]
    working=dict(spec,placements=group)
    plans=plan_explicit_surfaces(working,layout,replan_receptacles={name})
    if not plans:return None
    source_anchor=scene_route_anchor(spec,row['source_receptacle'],layout.entities)
    route=layout.route(source_anchor,row['source_room'],name,row['destination_room'])
    if route is None:return None
    spawn=spec['anchors'][spec['robot']['init_anchor']]['position'];r,c=layout.map.to_map(*spawn[:2]);sr,sc=layout.map.to_map(*route['start'][:2])
    if not (0<=r<layout.map.size and 0<=c<layout.map.size and layout.free[r,c]
            and layout.labels[r,c]==layout.labels[sr,sc]):return None
    measured=include_task_receptacles(layout,spec,evidence=evidence,source_sha=hashlib.sha256(source.read_bytes()).hexdigest())
    pf=preflight_recipe(spec,measured.entities,dimensions(spec))
    if pf['status']!='heuristic_fit':return None
    permitted=copy.deepcopy(original);permitted['objects'][permitted['objects'].index(obj)]=replacement
    assert_preserved(permitted,spec)
    directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True);target=directory/'scenario.yaml'
    target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
    journal=dict(method='reviewed_reference_fixture_capacity_exchange',original_recipe=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),before_fixture=fixture,after_fixture=changed,
        plans=plans,reason='Use a real native fixture matching the cabinet-top instruction and enough real surface for all distinct matching samples. Preserve the exact delivery site, reference assets, task/goal and robot spawn.',
        physical_validation_required=True)
    (directory/'expert_intervention.json').write_text(json.dumps(journal,indent=2)+'\n')
    return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        inserted_furniture=furniture,route=route,layout_preflight=pf,source_layout_revisions=row.get('source_layout_revisions',[])+[journal]),journal


def flat_stock_stack_proposals(objects,items,targets):
    """Two real-object piles; envelopes reserve their full swept footprints."""
    # Open boxes, trays and game sets are not solid planar supports: smaller
    # objects can fall into their cavities or between their pieces. Nested
    # storage requires a real Inside relation, not a fabricated OnTop pile.
    flat_categories={'hardback','paperback_book','textbook','catalog','comic_book','magazine','notebook','postcard',
        'folder','cd','dvd','phonograph_record','video_game','cutting_board','chopping_board'}
    flat=[name for name in items if objects[name]['category'] in flat_categories]
    if len(flat)<3:return [],{}
    flat.sort(key=lambda n:-float(np.prod(items[n][:2])))
    stacks=[flat[::2],flat[1::2]];envelopes={}
    for pile in stacks:
        for name in targets:
            if name in pile[2:]:pile.remove(name);pile.insert(1,name)
        envelopes[pile[0]]=[max(items[n][0] for n in pile),max(items[n][1] for n in pile),sum(items[n][2] for n in pile)]
    return stacks,envelopes


def fitting_work_surface_models(items,limit=3,*,allow_yaw=False,allowed_assets=()):
    """Reject impossible stock/table sizes before searching any floor sites."""
    candidates=[];area=sum(float(np.prod(v[:2])) for v in items.values())
    for category in ('breakfast_table','desk'):
        for path in (g.ASSETS/'objects'/category).glob('*/misc/metadata.json'):
            if allowed_assets and category+'/'+path.parent.parent.name not in allowed_assets:continue
            size=read(path).get('bbox_size',[])
            if (len(size)!=3 or not .65<=size[2]<=.95 or min(size[:2])<.55 or max(size[:2])>1.65
                    or np.prod(size[:2])<area or any(not (np.all(np.asarray(v[:2])+.01<=size[:2])
                        or allow_yaw and np.all(np.asarray(v[:2][::-1])+.01<=size[:2])) for v in items.values())):continue
            candidates.append((float(np.prod(size[:2])),category,path.parent.parent.name,size))
    from track_full_asset_queue import accepted_native_pair
    proven={key for key,record in read(g.ROOT/'build/verified_surface_sites/index.json').get('assets',{}).items()
        if record.get('validated_prop_count',0)>=2 and record.get('recipe') and record.get('evidence')
        and accepted_native_pair(record['recipe'],record['evidence'])}
    fitted=[]
    for _,category,model,size in sorted(candidates,key=lambda p:(p[1]+'/'+p[2] not in proven,*p[:3])):
        if solve_surface_proposal(items,size[:2],**({'allow_yaw':True} if allow_yaw else {}))['status']!='feasible_model':continue
        fitted.append((category,model))
        if len(fitted)>=limit:break
    return fitted


def work_surface_spawn_proposal(spec,row,layout,source_name):
    """Reviewed initial placement with an unambiguous forward delivery cue.

    Authoring only: stay in the original room within 1.2 m, retain height,
    require a free robot footprint and connected source/delivery approaches.
    A heading change may face the delivery table while leaving the newly
    authored source table outside its forward cone. Native gates still run.
    """
    anchor=spec['anchors'][spec['robot']['init_anchor']]
    old=np.asarray(anchor['position'],float);grid=layout.map;n=grid.size
    r,c=grid.to_map(*old[:2])
    if not (0<=r<n and 0<=c<n) or not grid.room_ins[r,c]:return None
    room_id=grid.room_ins[r,c]
    dest=scene_route_anchor(spec,row['destination_receptacle'],layout.entities)
    source_center=g.boxcenter(layout.entities[source_name])[:2]
    dest_center=g.boxcenter(layout.entities[dest])[:2]
    source_ends=layout.approaches(source_name,row['source_room'])
    dest_ends=layout.approaches(dest,row['destination_room'])
    components={layout.labels[p[1]] for p in source_ends}&{layout.labels[p[1]] for p in dest_ends}
    old_q=np.asarray(anchor.get('orientation',[0.,0.,0.,1.]),float)
    old_yaw=2*np.arctan2(old_q[2],old_q[3]);front=bool(re.search(r'\bin front of you\b',spec['instruction'],re.I))
    candidates=[]
    for rr,cc in [(r,c),*layout.points]:
        rr,cc=int(rr),int(cc)
        if grid.room_ins[rr,cc]!=room_id or not layout.free[rr,cc] or layout.labels[rr,cc] not in components:continue
        xy=old[:2].copy() if (rr,cc)==(r,c) else np.array([(cc-n/2)*grid.resolution,(rr-n/2)*grid.resolution])
        displacement=float(np.linalg.norm(xy-old[:2]))
        if displacement>1.2+1e-8:continue
        to_dest=dest_center-xy;to_source=source_center-xy
        if min(np.linalg.norm(to_dest),np.linalg.norm(to_source))<.1:continue
        for yaw in [old_yaw,float(np.arctan2(to_dest[1],to_dest[0]))]:
            forward=np.array([np.cos(yaw),np.sin(yaw)])
            if front and (np.dot(to_dest,forward)<.5*np.linalg.norm(to_dest)
                          or np.dot(to_source,forward)>=.5*np.linalg.norm(to_source)):continue
            angle=abs(float(np.arctan2(np.sin(yaw-old_yaw),np.cos(yaw-old_yaw))))
            candidates.append((displacement+.15*angle,rr,cc,yaw,xy))
    if not candidates:return None
    _,_,_,yaw,xy=min(candidates,key=lambda p:p[:4])
    after=copy.deepcopy(anchor);after['position']=[float(xy[0]),float(xy[1]),float(old[2])]
    after['orientation']=[0.,0.,float(np.sin(yaw/2)),float(np.cos(yaw/2))]
    return dict(before=copy.deepcopy(anchor),after=after,reason='Keep a unique forward delivery table after explicitly reauthoring generic source storage on a work surface.',
        displacement_m=float(np.linalg.norm(xy-old[:2])),same_initial_room=True,physical_validation_required=True)


def generic_source_stock_placements(spec,row,*,include_open_container_contents=False):
    """Source props and real OnTop piles, never delivery/reference props."""
    objects={o['name']:o for o in spec['objects']};references={row.get('reference_entity'),'comparison_sample'}
    selected=set()
    source_body=objects.get(row['source_receptacle'],{})
    open_source=(include_open_container_contents and row['source_receptacle'] not in row.get('target_entities',[])
        and source_body.get('category') in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray'))
    if open_source:selected.add(row['source_receptacle'])
    for p in spec['placements']:
        obj=objects.get(p['entity'],{})
        if not obj or obj.get('fixed_base') or p['entity'] in references:continue
        parent=objects.get(p['receptacle'],{})
        fixed_parent=not parent or parent.get('fixed_base')
        if (p['relation']=='inside' and (fixed_parent or open_source and p['receptacle']==row['source_receptacle']) or
            p['relation']=='on_top' and fixed_parent and p['receptacle']==row['source_receptacle']
            and p['receptacle']!=row['destination_receptacle']):selected.add(p['entity'])
    while True:
        children={p['entity'] for p in spec['placements'] if p['relation']=='on_top'
            and p['receptacle'] in selected and p['entity'] in objects
            and not objects[p['entity']].get('fixed_base') and p['entity'] not in references}
        if children<=selected:break
        selected|=children
    return [p for p in spec['placements'] if p['entity'] in selected]


def reauthor_generic_storage_on_work_surface(row,evidence,out,*,expert_approved=False,stack_flat_stock=False,allow_reviewed_spawn=False,
                                            prefer_constructed=False,include_open_container_contents=False,allow_surface_yaw=False,allow_furniture_yaw=False,
                                            allowed_surface_assets=()):
    """Explicit data revision for oversized stock in unqualified storage.

    This is never an automatic failure fallback. A reviewer may choose an
    ordinary open work surface when the task does not name a cupboard,
    drawer or shelf. Keep the exact task, goal, native inventory and source
    room. Initial support relations change and are fully journaled; every
    physical/native/reload check must run on the new data.
    """
    if not expert_approved:return None
    source=Path(row['scenario']);original=yaml.safe_load(source.read_text())
    scope=storage_scope_text(original['instruction'])
    for name in row['target_entities']:
        target_object=next(o for o in original['objects'] if o['name']==name)
        noun=target_object['category'].replace('_',' ')
        scope=re.sub(r'\b'+re.escape(noun)+r'\b','',scope,count=1,flags=re.I)
    source_clauses=list(re.finditer(r'\b(?:among|from)\b',scope,re.I))
    if source_clauses:
        # Inline matching examples may mention a cabinet that belongs to
        # the reference/delivery clause. Keep the actual source clause.
        scope=scope[source_clauses[-1].end():]
        scope=re.split(r'\b(?:to|onto)\s+(?:the|a|my|our)\b',scope,maxsplit=1,flags=re.I)[0]
    if (has_qualified_storage_location(original['instruction']) or
            re.search(r'\b(cabinet|cupboard|drawer|fridge|refrigerator|shelf|shelves|bookcase|basket|trunk)s?\b',scope,re.I)):
        return None
    objects={o['name']:o for o in original['objects']}
    group=generic_source_stock_placements(original,row,include_open_container_contents=include_open_container_contents)
    if not group or not set(row['target_entities']).issubset({p['entity'] for p in group}):return None
    parents={p['receptacle'] for p in group}
    # A fixed delivery fixture stays at its original site. Never relocate an
    # actual movable goal receptacle together with the searched source stock.
    if {g.get('receptacle') for g in original.get('goal',[])}&{p['entity'] for p in group}:return None
    removed={name for name in {f['name'] for f in row.get('inserted_furniture',[])} if objects.get(name,{}).get('fixed_base')
        and any(f['name']==name for f in row.get('inserted_furniture',[]))
        and not any(g.get('entity')==name or g.get('receptacle')==name for g in original.get('goal',[]))
        and all(p['entity'] in {q['entity'] for q in group} for p in original['placements'] if p['receptacle']==name)}
    furniture=[f for f in row.get('inserted_furniture',[]) if f['name'] not in removed]
    base=storage.replay_layout(layout_for_spec(original),furniture)
    sizes=dimensions(original);items={p['entity']:sizes[p['entity']] for p in group}
    baseline=storage.replay_layout(layout_for_spec(original),row.get('inserted_furniture',[]))
    actual_parent=next(p['receptacle'] for p in group if p['entity']==row['target_entities'][0])
    old_anchor=scene_route_anchor(original,actual_parent,baseline.entities)
    old_center=g.boxcenter(baseline.entities[old_anchor])[:2]
    room=row['source_room'];occupied={p['receptacle'] for p in original['placements']}-parents
    proposals=[]
    for name,body in base.entities.items():
        if prefer_constructed:break
        if (name in occupied or name==row['destination_receptacle'] or room not in body.get('in_rooms',[])
                or body.get('category') not in ('desk','breakfast_table','dining_table','coffee_table','table','countertop')
                or not body.get('aabb') or np.linalg.norm(g.boxcenter(body)[:2]-old_center)>4.):continue
        proposals.append((base,name,furniture))
    packing_items=copy.deepcopy(items)
    if stack_flat_stock:
        flat_stacks,envelopes=flat_stock_stack_proposals(objects,items,row['target_entities'])
        for pile in flat_stacks:
            for child in pile[1:]:packing_items.pop(child,None)
        packing_items.update(envelopes)
    def candidate_surfaces():
        # Try an existing surface before spending CPU on any new floor site.
        yield from proposals
        for asset in fitting_work_surface_models(packing_items,allow_yaw=allow_surface_yaw,allowed_assets=allowed_surface_assets):
            for pose_index in range(16 if allow_furniture_yaw else 8):
                yaw,site=(90 if pose_index>=8 else 0),pose_index%8
                previous=storage.MODELS.get('table')
                try:
                    storage.MODELS['table']=asset
                    proposal=storage.augment(base,room,'table','search_work_surface',variant=site,**({'yaw_degrees':yaw} if yaw else {}))
                finally:
                    if previous is None:storage.MODELS.pop('table',None)
                    else:storage.MODELS['table']=previous
                if proposal is None:continue
                layout,_=proposal;name='search_work_surface'
                if np.linalg.norm(g.boxcenter(layout.entities[name])[:2]-old_center)<=4.:
                    yield layout,name,layout.inserted_furniture
    for layout,name,furniture in candidate_surfaces():
        spec=copy.deepcopy(original);moved=[];spawn_change=None
        if allow_reviewed_spawn:
            spawn_change=work_surface_spawn_proposal(spec,row,layout,name)
            if spawn_change is None:continue
            spec['anchors'][spec['robot']['init_anchor']]=spawn_change['after']
        if room==row['destination_room'] and re.search(r'\bin front of you\b',spec['instruction'],re.I):
            anchor=spec['anchors'][spec['robot']['init_anchor']]
            q=anchor.get('orientation',[0.,0.,0.,1.]);yaw=2*np.arctan2(q[2],q[3])
            forward=np.array([np.cos(yaw),np.sin(yaw)])
            vector=g.boxcenter(layout.entities[name])[:2]-np.asarray(anchor['position'][:2])
            # Do not introduce a second candidate work surface in the initial
            # forward cone selected by the unchanged delivery instruction.
            if np.dot(vector,forward)>=.5*np.linalg.norm(vector):continue
        for p in spec['placements']:
            if p['entity'] not in items:continue
            before=copy.deepcopy(p);p.update(relation='on_top',receptacle=name)
            p.pop('link',None);p.pop('position_xy',None);moved.append(dict(before=before,after=p))
        stacks=[];overrides={}
        if stack_flat_stock:
            stacks,overrides=flat_stock_stack_proposals(objects,items,row['target_entities'])
            for pile in stacks:
                for below,above in zip(pile,pile[1:]):
                    next(p for p in spec['placements'] if p['entity']==above)['receptacle']=below
        spec['objects']=[o for o in spec['objects'] if o['name'] not in removed]
        spec['placements']=[p for p in spec['placements'] if p['entity'] not in removed]
        for old in removed:spec.get('initial_states',{}).pop(old,None)
        if name not in base.entities:
            added=next(f for f in furniture if f['name']==name)
            spec['objects'].append(dict(name=name,category=added['category'],model=added['model'],fixed_base=True))
            spec['placements'].append(dict(entity=name,relation='on_top',receptacle=added['floor'],position_xy=added['position_xy']))
            if added.get('orientation_hint_xyzw') is not None:
                spec['placements'][-1]['orientation_hint_xyzw']=list(added['orientation_hint_xyzw'])
        working=dict(spec,placements=[p for p in spec['placements'] if p['entity'] in items and p['receptacle']==name])
        plans=plan_explicit_surfaces(working,layout,replan_receptacles={name},dimension_overrides=overrides,allow_yaw=allow_surface_yaw)
        if not plans:continue
        dest=scene_route_anchor(spec,row['destination_receptacle'],layout.entities)
        route=layout.route(name,room,dest,row['destination_room'])
        if route is None:continue
        spawn=spec['anchors'][spec['robot']['init_anchor']]['position']
        rr,cc=layout.map.to_map(*spawn[:2]);ar,ac=layout.map.to_map(*route['start'][:2])
        if not (0<=rr<layout.map.size and 0<=cc<layout.map.size and layout.free[rr,cc]
                and layout.labels[rr,cc]==layout.labels[ar,ac]):continue
        pf=preflight_recipe(spec,layout.entities,dimensions(spec))
        if pf['status']!='heuristic_fit':continue
        for key in ('instruction','goal','target','robot','anchors'):
            if key=='anchors' and spawn_change is not None:
                expected=copy.deepcopy(original['anchors']);expected[spec['robot']['init_anchor']]=spawn_change['after']
                if spec['anchors']!=expected:raise ValueError('UNREVIEWED_ANCHOR_CHANGED')
                continue
            if spec.get(key)!=original.get(key):raise ValueError('OPEN_SOURCE_SEMANTIC_CONTRACT_CHANGED:'+key)
        if any(o not in spec['objects'] for o in original['objects'] if not o.get('fixed_base')):
            raise ValueError('OPEN_SOURCE_TASK_INVENTORY_CHANGED')
        target=Path(out)/row['id']/'scenario.yaml';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
        journal=dict(method='expert_reauthored_generic_storage_on_open_work_surface',
            source_recipe=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            evidence=str(evidence),relation_changes=moved,removed_unused_authored_fixtures=sorted(removed),
            flat_stock_stacks=stacks,stack_footprint_envelopes_m=overrides,
            source_surface=name,plans=plans,source_room=room,
            reviewed_initial_placement=spawn_change,
            reviewed_open_source_relayout=include_open_container_contents,
            reviewed_surface_yaw_proposals=allow_surface_yaw,
            reviewed_furniture_yaw_proposals=allow_furniture_yaw,
            reviewed_surface_asset_bank=list(allowed_surface_assets),
            reviewed_source_scope=scope,
            reason='The task names unqualified source stock. Reauthor its initial arrangement on a real work surface, retaining the exact instruction, goal, native props and counts; journal every support relation and initial-pose change and rerun all native physical checks.',
            physical_validation_required=True,human_completion_verified=False)
        (target.parent/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
        return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            inserted_furniture=furniture,object_count=len(spec['objects']),layout_preflight=pf,route=route,
            source_receptacle=name,restoration_entities=[n for n in row.get('restoration_entities',[]) if n not in removed],
            source_layout_revisions=row.get('source_layout_revisions',[])+[journal]),journal
    return None


def repair_thin_drawer_contents_with_native_tray(row,evidence,out):
    """Use a real organizer for thin props below an inset volume boundary.

    Some native drawers' fillable volumes start above their collision floors.
    Millimetre-thick combs/pins then have no legal resting centre at all. Add
    one ordinary rigid tray, rather than floating props or changing Inside.
    Native support, contact, membership, settling and reload must all pass.
    """
    source=Path(row['scenario']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    report=read(Path(evidence)/'validation.json');log=Path(evidence)/'native_validation.log'
    if (report.get('scenario_sha256')!=sha or 'EXPERT_LAYOUT_UNRESOLVED: infeasible_model' not in str(report.get('error',''))
            or not log.exists()):return None
    problems=[]
    for line in log.read_text(errors='replace').splitlines():
        if '[expert-layout-problem] ' in line:
            try:problems.append(json.loads(line.split('[expert-layout-problem] ',1)[1]))
            except ValueError:pass
    if not problems:return None
    problem=problems[-1];items=problem.get('items',{});fixed=problem.get('fixed',{})
    if not items or set(fixed)!=set(items) or any(fixed.values()) or max(v[2] for v in items.values())>.04:return None
    receptacle=problem['receptacle'];original=yaml.safe_load(source.read_text())
    group={p['entity'] for p in original['placements'] if p['relation']=='inside' and p['receptacle']==receptacle}
    if not set(items).issubset(group) or any(e['receptacle']==receptacle for e in row.get('drawer_liner_repairs',[])):return None
    selected_links={p.get('link') for p in original['placements'] if p['entity'] in items}
    if len(selected_links)>1:return None
    selected_link=next(iter(selected_links),None)
    compatible_volumes={key:v for key,v in problem.get('volumes',{}).items()
        if not selected_link or key.split('#',1)[0]==selected_link
        or key.split('#',1)[0].startswith(f'meta__{selected_link}_')}
    if not compatible_volumes:return None
    name='drawer_support_'+hashlib.sha256(receptacle.encode()).hexdigest()[:8]
    if name in {o['name'] for o in original['objects']}:return None
    layout=include_task_receptacles(storage.replay_layout(layout_for_spec(original),row.get('inserted_furniture',[])),
        original,evidence=evidence,source_sha=sha)
    models=[]
    categories=['notebook','tray']
    if str(row.get('source_room','')).startswith(('kitchen_','dining_room_')):categories=['cutting_board','notebook','tray']
    for preference,category in enumerate(categories):
        for path in (g.ASSETS/'objects'/category).glob('*/misc/metadata.json'):
            size=read(path).get('bbox_size',[])
            if len(size)!=3 or not .008<=size[2]<=.05:continue
            if not any(np.all(np.asarray(size)+.03<=np.asarray(v)) for v in compatible_volumes.values()):continue
            if min(size[:2])<.025:continue
            full_cover=all(np.all(np.sort(size[:2])>=np.sort(v[:2])+.015) for v in items.values())
            coverage=min(float(np.min(np.sort(size[:2])/(np.sort(v[:2])+.015))) for v in items.values())
            # The live planner already permits a flat object's overhang:
            # its centre is constrained to the measured real support patch,
            # while its full box stays in the native cavity. Do not impose
            # a contradictory full-footprint rule only in offline authoring.
            models.append((preference,not full_cover,
                float(np.prod(size[:2])) if full_cover else -coverage,
                category,path.parent.parent.name,size))
    for _,_,_,category,model,size in sorted(models):
        spec=copy.deepcopy(original);spec['objects'].append(dict(name=name,category=category,model=model,fixed_base=False))
        spec['placements'].append(dict(entity=name,relation='inside',receptacle=receptacle))
        if selected_link:spec['placements'][-1]['link']=selected_link
        pf=preflight_recipe(spec,layout.entities,dimensions(spec))
        if pf['status']!='heuristic_fit':continue
        assert_preserved(original,spec)
        directory=Path(out)/row['id'];directory.mkdir(parents=True,exist_ok=True)
        target=directory/'scenario.yaml';target.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(target)
        edit=dict(receptacle=receptacle,entity=name,asset=category+'/'+model,extent_m=size,
            original_compartment_link=selected_link,
            support_policy='Measured support constrains the child centre; the unchanged native cavity constrains its whole box. Overhang is a proposal only; native stability/contact/membership/reload must pass.',
            observed_unplaceable_thin_entities=sorted(items),native_empty_fixed_domains=fixed,
            source_sha256=sha,evidence=str(evidence),physical_validation_required=True)
        journal=dict(method='real_native_organizer_for_inset_drawer_floor',edit=edit,
            reason='All thin props have empty native centre domains on the real drawer floor. A collidable rigid organizer provides real support above that floor; native acceptance remains unchanged.',
            preserved=['instruction','goal','all original props and counts','source fixture','all original placements','robot','spawn'],physical_validation_required=True)
        (directory/'expert_intervention.json').write_text(json.dumps(journal,indent=2))
        return dict(row,scenario=str(target.resolve()),scenario_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            object_count=len(spec['objects']),distractor_count=row.get('distractor_count',0)+1,layout_preflight=pf,
            restoration_entities=list(dict.fromkeys(row.get('restoration_entities',[])+[name])),
            drawer_liner_repairs=row.get('drawer_liner_repairs',[])+[edit]),journal
    return None


def expert_proposal(row, evidence, out, *, policy, variant=10):
    """Apply strict repairs, or first establish a registered editorial baseline."""
    coordinate=repair_double_translated_surface_pose(row,evidence,out)
    if coordinate is not None:return coordinate[0],dict(coordinate[1],expert_policy=policy)
    corrected=repair_covered_surface_binding(row,evidence,out)
    if corrected is not None:
        resized=apply_registered_support_rebinding(corrected[0],evidence,Path(out)/'_capacity')
        if resized is None:resized=repair_rigid_container_capacity(corrected[0],evidence,Path(out)/'_capacity')
        if resized is not None:return resized[0],dict(resized[1],surface_binding_correction=corrected[1],expert_policy=policy)
        return corrected[0],dict(corrected[1],expert_policy=policy)
    reference=repair_unstable_reference_surface(row,evidence,out)
    if reference is not None:return reference[0],dict(reference[1],expert_policy=policy)
    editorial=apply_registered_editorial_revision(row,Path(out)/'_editorial')
    if editorial is not None:
        revised,journal=editorial
        result,info=expert_proposal(revised,evidence,out,policy=policy,variant=variant)
        return result,dict(info,editorial_revision=journal)
    rebound=apply_registered_support_rebinding(row,evidence,out)
    if rebound is not None:return rebound[0],dict(rebound[1],expert_policy=policy)
    capacity=repair_rigid_container_capacity(row,evidence,out)
    if capacity is not None:return capacity[0],dict(capacity[1],expert_policy=policy)
    contact_repair=apply_registered_scene_contact_repair(row,evidence,out)
    if contact_repair is not None:return contact_repair[0],dict(contact_repair[1],expert_policy=policy)
    spawn=repair_robot_spawn(row,evidence,out)
    if spawn is not None:return spawn[0],dict(spawn[1],expert_policy=policy)
    sampled_shelf=repair_explicit_shelf_sampling(row,evidence,out)
    if sampled_shelf is not None:return sampled_shelf[0],dict(sampled_shelf[1],expert_policy=policy)
    repacked_surface=repair_failed_surface_poses(row,evidence,out)
    if repacked_surface is not None:return repacked_surface[0],dict(repacked_surface[1],expert_policy=policy)
    surface_repair=repair_open_surface_source(row,evidence,out,variant)
    if surface_repair is not None:return surface_repair[0],dict(surface_repair[1],expert_policy=policy)
    nudged=nudge_interfering_scene_storage(row,evidence,out)
    if nudged is not None:return nudged[0],dict(nudged[1],expert_policy=policy)
    relocated=relocate_failed_fixture(row,evidence,out)
    if relocated is not None:
        if relocated[0] is not None:return relocated[0],dict(relocated[1],expert_policy=policy)
        # A failed large auxiliary cabinet need not retain its model forever.
        # A floor-placement failure identifies the fixture itself, whereas
        # blocking_storage() ordinarily returns its supporting floor. Route
        # its contents to the existing bounded native-model/site search.
        original=yaml.safe_load(Path(row['scenario']).read_text());failed_fixture=relocated[1].get('entity')
        if failed_fixture and editable_authored_storage(original,row,failed_fixture):
            result,info=curate(row,evidence,out,variant=variant,old_storage=failed_fixture,
                               allow_patch_support=True,prefer_constructed=True)
            return result,dict(info,expert_policy=policy,prior_fixture_relocation=relocated[1])
        return None,dict(relocated[1],expert_policy=policy)
    partitioned=repair_native_blocking_subset(row,evidence,out,variant)
    if partitioned is not None:return partitioned[0],dict(partitioned[1],expert_policy=policy)
    liner=repair_thin_drawer_contents_with_native_tray(row,evidence,out)
    if liner is not None:return liner[0],dict(liner[1],expert_policy=policy)
    in_place=repair_scene_fixture_in_place(row,evidence,out)
    if in_place is not None:return in_place[0],dict(in_place[1],expert_policy=policy)
    attempted_fixture_repair=None
    if policy.get('method')=='relocate_storage':
        result,info=curate(row,evidence,out,variant=variant,allow_patch_support=True,prefer_constructed=True)
        if result is not None:return result,dict(info,expert_policy=policy)
        attempted_fixture_repair=info
        observed=read(Path(evidence)/'validation.json')
        if ('EXPERT_CONTAINER_JOINTS_NOT_SETTLED' in str(observed.get('error','')) and
                observed.get('scenario_sha256')==hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest()):
            return None,dict(status='deferred',reason='Removing compiler links cannot repair observed furniture joint interference; a changed physical source is required',
                             fixture_repair=info,expert_policy=policy,physical_validation_required=True)
    original=yaml.safe_load(Path(row['scenario']).read_text());spec=copy.deepcopy(original)
    changes=clear_unstated_links(spec) if policy.get('clear_unstated_links') else []
    layout=include_task_receptacles(storage.replay_layout(layout_for_spec(spec),row.get('inserted_furniture',[])),spec,
        evidence=evidence,source_sha=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest())
    centered=center_open_container_placements(spec,layout)
    surfaces=plan_explicit_surfaces(spec,layout) if policy.get('plan_surfaces') else []
    preflight=preflight_recipe(spec,layout.entities,dimensions(spec))
    if preflight['status']!='heuristic_fit':
        result,info=curate(row,evidence,out,variant=variant,allow_patch_support=True,prefer_constructed=True)
        if result is not None:return result,dict(info,expert_policy=policy)
        # A task with two invalid sources cannot be repaired if every
        # intermediate change is required to fix both sources at once.
        # Chain partial authoring changes privately; launch native validation
        # only after the complete scene passes the unchanged static gate.
        original_contract=copy.deepcopy(original);current=dict(row);steps=[]
        for index in range(4):
            bad=next((f for f in preflight['findings'] if f['status']!='heuristic_fit'),None)
            if bad is None:break
            stage=Path(out)/'_working'/f'step_{index}';initial=stage/row['id']/'scenario.yaml'
            initial.parent.mkdir(parents=True,exist_ok=True);initial.write_text(yaml.safe_dump(spec,sort_keys=False))
            current.update(scenario=str(initial.resolve()),scenario_sha256=hashlib.sha256(initial.read_bytes()).hexdigest())
            repaired,step=curate(current,evidence,stage/'repaired',variant=variant+index,
                                 old_storage=bad['receptacle'],allow_partial=True,allow_patch_support=True,prefer_constructed=True)
            # Plural drawers/cupboards and generic storage may span fixtures.
            # Never split a named single drawer, cabinet or basket into several.
            if repaired is None and source_allows_multiple_fixtures(original['instruction']):
                group=[p['entity'] for p in spec['placements'] if p['receptacle']==bad['receptacle']]
                sizes=dimensions(spec);ordered=sorted(group,key=lambda n:-sizes[n][0]*sizes[n][1])
                for count in dict.fromkeys([len(group)//2,max(1,len(group)//3),1]):
                    if not 0<count<len(group):continue
                    repaired,step=curate(current,evidence,stage/f'partition_{count}',variant=variant+index,
                        old_storage=bad['receptacle'],allow_partial=True,allow_patch_support=True,prefer_constructed=True,group_entities=set(ordered[:count]))
                    if repaired is not None:
                        step['partitioned_original_group']=group;step['moved_entities']=ordered[:count];break
            steps.append(step)
            if repaired is None:break
            current=repaired;spec=yaml.safe_load(Path(current['scenario']).read_text())
            preflight=current['layout_preflight']
            if preflight['status']=='heuristic_fit':
                editable={f['name'] for f in row.get('inserted_furniture',[])
                          if editable_authored_storage(original_contract,row,f['name'])}
                assert_task_inventory_preserved(original_contract,spec,editable_fixture_names=editable)
                path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
                current.update(scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                doc=dict(status='static_screened_physics_pending',expert_policy=policy,partial_authoring_steps=steps,
                         original_recipe=row['scenario'],removed_compiler_links=changes,physical_validation_required=True)
                (path.parent/'expert_intervention.json').write_text(json.dumps(doc,indent=2))
                return current,doc
        return None,dict(status='deferred',reason='Per-case multi-source repair remains unresolved; no native launch',
                         preflight=preflight,fixture_repair=info,partial_authoring_steps=steps,expert_policy=policy)
    assert_preserved(original,spec)
    if spec==original and attempted_fixture_repair is not None:
        return None,dict(status='deferred',reason='No changed, whole-scene-valid fixture proposal',
                         fixture_repair=attempted_fixture_repair,expert_policy=policy,physical_validation_required=True)
    path=Path(out)/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(spec,sort_keys=False));g.load_scenario(path)
    result=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),layout_preflight=preflight)
    info=dict(status='static_screened_physics_pending',expert_policy=policy,removed_compiler_links=changes,
              centered_container_placements=centered,
              explicit_surface_plans=surfaces,original_recipe=row['scenario'],
              original_sha256=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest(),
              diagnosis_directory=str(evidence),preserved=['instruction','goal','target','robot','spawn','all original assets','relation types'])
    (path.parent/'expert_intervention.json').write_text(json.dumps(info,indent=2))
    return result,info


def request(payload):
    action=payload['action']
    if action=='index':return dict(cases=discover_sources())
    case=payload['case'];out=Path(payload['out']);out.mkdir(parents=True,exist_ok=True)
    row=case.get('record')
    if row is None:return dict(status='deferred',reason='No hash-matched source recipe')
    if payload.get('native_policy_signature'):row=dict(row,proposal_native_policy_signature=payload['native_policy_signature'])
    if action=='signature':
        learn_opened_furniture_footprints(row,case['evidence'])
        learn_native_support_profiles(row,case['evidence'])
        from track_full_asset_queue import accepted_native_pair
        if accepted_native_pair(row['scenario'],Path(case['evidence'])):
            learn_verified_storage_sites(row,case['evidence'])
            return dict(signature=None,reason='Strictly accepted layout; retain as positive authoring evidence, never blacklist')
        build=read(Path(case['evidence'])/'validation.json') if case.get('evidence') else {}
        from overnight_asset_queue import infrastructure_error
        if infrastructure_error(build.get('error','')):
            return dict(signature=None,reason='Proposal/runtime infrastructure failure; no evidence to blacklist source geometry')
        if 'method not intended for entity prim with no joints' in str(build.get('error','')):
            return dict(signature=None,reason='Jointless-container diagnostic bug; no evidence to blacklist source geometry')
        if proposal_coordinate_bug(build):
            return dict(signature=None,reason='Known duplicate bbox/root conversion; source geometry not independently rejected')
        from audit_dense_layouts import source_layout_signature
        return dict(signature=source_layout_signature(yaml.safe_load(Path(row['scenario']).read_text()),row['target_entities'][0]))
    existing=next((r for r in read(out/'manifest.json').get('tasks',[]) if r['id']==row['id']),None)
    if existing:return dict(status='proposed',record=existing,reused_immutable_candidate=True)
    if action=='expert':
        result,info=expert_proposal(row,case['evidence'],out,policy=case['expert_policy'],variant=payload.get('variant',10))
        if result is None:return info
        before=yaml.safe_load(Path(row['scenario']).read_text())
        after=yaml.safe_load(Path(result['scenario']).read_text())
        if authoring_recipe_identity(before)==authoring_recipe_identity(after):
            return dict(status='deferred',reason='No authored change from observed failed recipe; skip identical native replay',evidence=info)
    elif action=='curate':
        result,info=curate(row,case['evidence'],out,variant=payload.get('variant',0))
        if result is None:return info
    elif action=='direct':
        import shutil
        spec=yaml.safe_load(Path(row['scenario']).read_text())
        log=Path(case['evidence'])/'native_validation.log'
        failed=re.search(r'\[placement\] end (\S+) placed=False',log.read_text(errors='replace')) if log.exists() else None
        fixed={o['name'] for o in spec['objects'] if o.get('fixed_base')}
        if failed and failed[1] in fixed and any(p['entity']==failed[1] and p['relation']=='on_top' for p in spec['placements']):
            return dict(status='deferred',reason='Observed fixed-fixture floor failure needs a site repair, not an unchanged interior-solver replay')
        sha=hashlib.sha256(Path(row['scenario']).read_bytes()).hexdigest()
        layout=include_task_receptacles(storage.replay_layout(layout_for_spec(spec),row.get('inserted_furniture',[])),
                                       spec,evidence=case['evidence'],source_sha=sha)
        preflight=preflight_recipe(spec,layout.entities,dimensions(spec))
        observed=read(Path(case['evidence'])/'validation.json')
        restore=read(Path(case['evidence'])/'restore_validation.json')
        placed=observed.get('build',{}).get('placements',[])
        convergence_error=(str(observed.get('error','')).startswith('LAYOUT_NOT_CONVERGED:') or
                           (observed.get('status')=='initialization_validated' and restore.get('scenario_sha256')==sha
                            and str(restore.get('error','')).startswith('LAYOUT_NOT_CONVERGED:')))
        prior_native_placement=(observed.get('scenario_sha256')==sha and convergence_error
            and len(placed)==len(spec['placements']) and
            {p['entity'] for p in placed}=={p['entity'] for p in spec['placements']}
            and all(p.get('placed') is True and p.get('verified') is True for p in placed))
        if preflight['status']!='heuristic_fit' and not prior_native_placement:
            return dict(status='deferred',reason='Original layout fails corrected static geometry gate',preflight=preflight)
        path=out/row['id']/'scenario.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row['scenario'],path)
        result=dict(row,scenario=str(path.resolve()),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        info=dict(status='physical_revalidation_pending',change='Unchanged recipe; reviewed native runtime counterfactual',
                  preflight=preflight,prior_verified_native_placements=prior_native_placement,
                  reason='A single-layer proposal rejection cannot refute observed native 3D placement; every original physical gate and fresh reload must run again.',
                  physical_validation_required=True)
    else:raise ValueError('Unknown expert request: '+action)
    manifest=read(out/'manifest.json',dict(tasks=[]));manifest['tasks'].append(result)
    temporary=(out/'manifest.json').with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(out/'manifest.json')
    return dict(status='proposed',record=result,evidence=info)


if __name__=='__main__':
    import argparse,sys,contextlib,traceback
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--serve',action='store_true')
    parser.add_argument('--request',type=Path);parser.add_argument('--response',type=Path);args=parser.parse_args()
    def handle(payload):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if payload.get('action')=='signature':
                    # Two native lanes can finish together. Serialize only
                    # shared knowledge updates, never the native simulations.
                    import fcntl
                    with (g.ROOT/'build/expert_knowledge.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        return request(payload)
                return request(payload)
        except Exception as exc:
            traceback.print_exc();return dict(status='error',error=type(exc).__name__+': '+str(exc))
    if args.serve:
        for line in sys.stdin:
            result=handle(json.loads(line));print(json.dumps(result),flush=True)
    else:
        result=handle(read(args.request))
        if args.response:args.response.write_text(json.dumps(result,indent=2))
        else:print(json.dumps(result,indent=2))
