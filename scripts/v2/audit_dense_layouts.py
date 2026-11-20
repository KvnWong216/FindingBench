"""Audit every authored recipe; propose compartment repairs without weakening checks.

Static packing checks are heuristics, not physical certification. Originals and
accepted episodes are never overwritten. Proposed repairs require native replay.
"""
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
import ast, copy, hashlib, json, math, re, time
import yaml

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets/objects')
RUNS=['asset_search_pipeline','asset_search_remaining_v2_20261104','asset_search_repairs_v2_20261104',
      'asset_remaining_repairs_20261105','asset_search_full_20261105', 'asset_search_gpu_new_20261106','manual_dense_repairs_20261106/candidate']

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

@lru_cache(maxsize=512)
def _cached_asset_digest(path, identity):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def asset_digest(path):
    """Avoid rehashing shared assets for every house; invalidate file changes."""
    path=Path(path);s=path.stat()
    return _cached_asset_digest(str(path),(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))

@lru_cache(maxsize=8)
def _geometry_index(path, identity):
    return json.loads(Path(path).read_text())

def load_geometry_index(path):
    path=Path(path);s=path.stat()
    return _geometry_index(str(path),(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))

def fillable_geometry_from_stage(stage):
    """Extract actual mesh vertices in asset-root coordinates, without physics.

    These are authoring inputs, not certified empty space. Preserve disconnected
    components; a union AABB may contain solid partitions. Scene scale, pose and
    articulated transforms must be applied before using this for layout checks.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    root=stage.GetDefaultPrim()
    if not root or not root.IsValid():raise ValueError('Missing default asset prim')
    cache=UsdGeom.XformCache()
    inverse=cache.GetLocalToWorldTransform(root).GetInverse()
    volumes=[]
    for link in root.GetChildren():
        name=link.GetName()
        if not (name.startswith('meta__') and 'fillable' in name and name.endswith('_link')):continue
        components=[]
        for prim in Usd.PrimRange(link,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or '/visuals/' not in str(prim.GetPath()):continue
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
                raise ValueError('Invalid native fillable mesh points')
            transform=np.asarray(cache.GetLocalToWorldTransform(prim)*inverse,dtype=float)
            points=points@transform[:3,:3]+transform[3,:3]
            components.append(dict(prim_path=str(prim.GetPath()),vertices=points.tolist(),
                lo=points.min(axis=0).tolist(),hi=points.max(axis=0).tolist()))
        if components:volumes.append(dict(link=name,components=components))
    joints=[dict(path=str(p.GetPath()),type=p.GetTypeName(),
        body0=[str(v) for v in p.GetRelationship('physics:body0').GetTargets()],
        body1=[str(v) for v in p.GetRelationship('physics:body1').GetTargets()])
        for p in stage.Traverse() if p.GetTypeName().endswith('Joint')]
    return dict(coordinates='asset_root_authored_pose',volumes=volumes,joints=joints,
                physical_validation_required=True)

def extract_fillable_geometry(asset_directory):
    """Read licensed native USD through the SDK, cleaning decrypted temporaries."""
    import tempfile
    from omnigibson.utils.asset_utils import decrypt_file
    from omnigibson.objects.dataset_object import MODEL_USD_HOTFIXES
    from pxr import Usd
    directory=Path(asset_directory);model=directory.name
    if model in MODEL_USD_HOTFIXES:raise ValueError('Native model hotfix requires live geometry audit')
    source=directory/'usd'/f'{model}.encrypted.usd'
    with tempfile.TemporaryDirectory(prefix='findingbench-volume-audit-') as temporary:
        path=Path(temporary)/f'{model}.usd'
        decrypt_file(str(source),str(path))
        stage=Usd.Stage.Open(str(path))
        result=fillable_geometry_from_stage(stage)
        del stage
    result.update(category=directory.parent.name,model=model,
        encrypted_asset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256((directory/'misc/metadata.json').read_bytes()).hexdigest())
    return result

def offline_interior_hints(geometry, scale, orientation):
    """Conservative authoring size hints for upright, translational storage.

    Never synthesize world positions or mark a task physically feasible. Volumes
    attached through rotating joints and tilted objects remain unknown. Keep
    disconnected components separate; a hinged door does not rotate a fixed cavity.
    """
    import numpy as np
    scale=np.asarray(scale,dtype=float);q=np.asarray(orientation,dtype=float)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or (scale<=0).any():return []
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.001:return []
    if np.linalg.norm(q[:2])>.001:return []
    joints=geometry.get('joints',[])
    translational=all(j['type'] in ('PhysicsPrismaticJoint','PhysicsFixedJoint') for j in joints)
    def stationary_orientation(volume):
        if translational:return True
        parents={};bodies=set();children=set()
        for joint in joints:
            for body in joint.get('body1',[]):parents.setdefault(body,[]).append(joint);children.add(body)
            bodies.update(joint.get('body0',[]))
        roots=bodies-children
        if len(roots)!=1:return False
        path=volume['components'][0].get('prim_path','').split('/visuals/')[0]
        seen=set()
        while path not in roots:
            if path in seen or len(parents.get(path,[]))!=1:return False
            seen.add(path);joint=parents[path][0]
            if joint['type'] not in ('PhysicsPrismaticJoint','PhysicsFixedJoint') or len(joint.get('body0',[]))!=1:return False
            path=joint['body0'][0]
        return True
    result=[]
    for volume in geometry.get('volumes',[]):
        if not volume['components']:continue
        if not stationary_orientation(volume):continue
        parts=[]
        for i,component in enumerate(volume['components']):
            points=np.asarray(component['vertices'],dtype=float)
            if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
                parts=[];break
            extent=np.ptp(points*scale,axis=0)+.002
            part=dict(link=volume['link'],extent_m=extent.tolist())
            if len(volume['components'])>1:part['component']=i
            parts.append(part)
        result.extend(parts)
    return result

def _volume_key(volume):
    return volume['link'] if 'component' not in volume else f"{volume['link']}#component:{volume['component']}"

def authoring_volumes(entity):
    """Use surveyed sizes, splitting any union AABB at verified native partitions."""
    interiors=entity.get('interiors',[])
    if interiors:
        result={v['link']:[b-a for a,b in zip(v['lo'],v['hi'])] for v in interiors}
        hints=entity.get('offline_interior_hints',[])
        split={link for link,count in Counter(v['link'] for v in hints).items() if count>1}
        for link in split:
            for key in list(result):
                if key==link or link.startswith(f'meta__{key}_'):del result[key]
            result.update({_volume_key(v):v['extent_m'] for v in hints if v['link']==link})
        return result
    return {_volume_key(v):v['extent_m'] for v in entity.get('offline_interior_hints',[])}

def authoring_native_links(entity):
    if entity.get('interiors'):
        keys=authoring_volumes(entity)
        hinted={_volume_key(v):v['link'] for v in entity.get('offline_interior_hints',[])}
        return {key:hinted.get(key,key) for key in keys}
    return {_volume_key(v):v['link'] for v in entity.get('offline_interior_hints',[])}

def enrich_offline_interiors(entities, scene, index, assets):
    """Attach hash-checked asset hints only to matching native scene objects."""
    init=scene.get('objects_info',{}).get('init_info',{})
    states=scene.get('state',{}).get('registry',{}).get('object_registry',{})
    verified={};count=0
    for name,row in entities.items():
        args=init.get(name,{}).get('args',{})
        key=f"{row.get('category')}/{row.get('model')}"
        geometry=index.get('assets',{}).get(key)
        if not geometry or any(args.get(k)!=row.get(k) for k in ('category','model')):continue
        if row.get('interiors') and not any(len(v.get('components',[]))>1 for v in geometry.get('volumes',[])):continue
        if key not in verified:
            directory=Path(assets)/key
            try:
                verified[key]=all(asset_digest(directory/file)==geometry.get(field)
                    for file,field in [(f"usd/{row['model']}.encrypted.usd",'encrypted_asset_sha256'),
                                       ('misc/metadata.json','metadata_sha256')])
            except OSError:verified[key]=False
        if not verified[key]:continue
        pose=states.get(name,{}).get('root_link',{}).get('ori')
        if pose is None or args.get('scale') is None:continue
        hints=offline_interior_hints(geometry,args['scale'],pose)
        if not hints:continue
        row['offline_interior_hints']=hints
        row['offline_interior_provenance']=dict(asset=key,scale=args['scale'],
            encrypted_asset_sha256=geometry['encrypted_asset_sha256'],
            metadata_sha256=geometry['metadata_sha256'],root_orientation=pose,
            coordinates='size_only_not_world_positions',physical_validation_required=True)
        count+=1
    return count

def upright_fit(size,extent,margin=.005):
    return size[2]+margin<=extent[2] and (
        size[0]+margin<=extent[0] and size[1]+margin<=extent[1] or
        size[1]+margin<=extent[0] and size[0]+margin<=extent[1])

# Shared pure-CUDA authoring solver is also available to the native builder.
from rummagebench.sim.omnigibson.inside_volume import solve_layout, _solve_layout_gpu_cached


def allocate(items,volumes,fixed=None):
    result=solve_layout(items,volumes,fixed)
    return result['assignment'] if result['status']=='feasible_model' else None


def layered_open_container_proposal(items,volumes,*,margin_m=.005):
    """Construct disjoint 3D box layers, never certify physical support.

    Only open containers use this authoring fallback: their existing native
    Inside sampler handles 3D placements. Closed-cabinet live planning remains
    unchanged. Each layer uses the GPU packing solver, not random sampling.
    Unknown searches and insufficient height return no witness.
    """
    if len(volumes)!=1 or not 1<=len(items)<=12:return None
    bin_name,extent=next(iter(volumes.items()));remaining=sorted(items,key=lambda n:(-items[n][0]*items[n][1],n))
    height=0.;poses={};layers=[]
    while remaining:
        layer=[];witness=None
        for name in remaining:
            trial={n:items[n] for n in [*layer,name]}
            candidate=solve_layout(trial,volumes,margin_m=margin_m,time_limit_s=.2)
            if candidate.get('status')=='feasible_model':layer.append(name);witness=candidate
        if not layer:return None
        thickness=max(items[n][2] for n in layer)
        if height+thickness+margin_m>extent[2]:return None
        for name in layer:
            pose=dict(witness['poses'][name]);pose.update(lower_z_m=height,height_m=items[name][2])
            poses[name]=pose
        layers.append(dict(entities=layer,lower_z_m=height,height_m=thickness))
        height+=thickness+margin_m;remaining=[n for n in remaining if n not in layer]
    return dict(status='feasible_model',backend='torch_cuda_greedy_box_layers',poses=poses,
        assignment={n:bin_name for n in items},layers=layers,
        model='nominal upright boxes in disjoint horizontal layers; support and stability unverified',
        physical_validation_required=True)


def preflight_recipe(spec, entities, dimensions):
    """Joint upright packing proposal, never a physical feasibility certificate.

    Use surveyed or hash-checked native geometry. Individual size mismatches
    are rejected before sampling; missing geometry must be resolved by the
    compiler. All explicit link selections and location constraints are retained.
    """
    groups=defaultdict(list)
    for p in spec.get('placements',[]):
        if p['relation']=='inside':groups[p['receptacle']].append(p)
    findings=[];assignments={}
    specific=bool(re.search(r'\b(drawer|upper|lower|topmost|bottommost|leftmost|rightmost)\b',
                            spec.get('instruction',''),re.I))
    for rec,placements in groups.items():
        entity=entities.get(rec,{})
        volumes=authoring_volumes(entity)
        native_links=authoring_native_links(entity)
        evidence=dict(receptacle=rec,volume_extents_m=volumes,native_links=native_links,
            geometry_source='survey_with_native_components' if entity.get('interiors') and any('#component:' in k for k in volumes) else 'survey' if entity.get('interiors') else
                'offline_asset_hint' if entity.get('offline_interior_hints') else 'unknown',
            geometry_provenance=entity.get('offline_interior_provenance',{}))
        fixed={p['entity']:p['link'] for p in placements if p.get('link')}
        # Runtime accepts either a real meta-link or its owning drawer link.
        # Resolve an owner only when exactly one surveyed/hinted volume matches;
        # preserve the recipe's explicit link and never redirect compartments.
        for entity,link in list(fixed.items()):
            matches={name for name in native_links.values() if name==link or
                     (name.startswith(f'meta__{link}_') and 'fillable' in name)}
            if len(matches)==1:fixed[entity]={key for key,native in native_links.items() if native in matches}
        if (not volumes or any(p['entity'] not in dimensions for p in placements)
            or any(not isinstance(keys,set) or not keys.issubset(volumes) for keys in fixed.values())):
            findings.append(dict(evidence,status='unknown',reason='Missing dimensions or matching surveyed links'))
            continue
        items={p['entity']:dimensions[p['entity']] for p in placements}
        oversized=[name for name,size in items.items() if not any(
            upright_fit(size,extent) for key,extent in volumes.items()
            if name not in fixed or key in fixed[name])]
        if oversized:
            findings.append(dict(evidence,status='rejected_candidate',
                reason='Pre-sampling dimension gate: no allowed compartment fits the upright object with clearance',
                entities=oversized,object_extents_m={name:items[name] for name in oversized}))
            continue
        solved=solve_layout(items,volumes,fixed)
        if (solved['status']!='feasible_model' and len(volumes)==1 and
                entities.get(rec,{}).get('category') in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray')):
            # The earlier one-layer model cannot refute ordinary stacking of
            # bread or produce. Require a constructive non-overlapping 3D
            # witness before spending native work; never bypass native gates.
            physical_volumes=copy.deepcopy(volumes)
            if entities.get(rec,{}).get('offline_interior_provenance',{}).get('open_top_centroid_height'):
                for size in physical_volumes.values():size[2]=(size[2]+.02)/2
            layered=layered_open_container_proposal(items,physical_volumes)
            if layered is not None:solved=layered
        plan=solved.get('assignment')
        if solved['status']!='feasible_model':
            findings.append(dict(evidence,status='rejected_candidate' if solved['status']=='infeasible_model' else 'unknown',
                native_sampling_required=False,solver=solved,
                reason='Finite layout model infeasible' if solved['status']=='infeasible_model' else 'Layout solver undecided; no native sampling',entities=list(items)))
        else:
            findings.append(dict(evidence,status='heuristic_fit',solver=solved,assignment={name:native_links[key] for name,key in plan.items()},component_allocation=plan))
            if not specific:
                assignments.update({name:native_links[key] for name,key in plan.items() if name not in fixed})
    rejected=any(f['status']=='rejected_candidate' for f in findings)
    return dict(status='rejected_candidate' if rejected else 'unknown' if any(
        f['status']=='unknown' for f in findings) else 'heuristic_fit',findings=findings,
        prefer_alternative_layout=any(f.get('native_sampling_required') for f in findings),
        assignments={} if rejected else assignments,physical_validation_required=True)

def audit():
    tasks={}
    for catalog in ('asset_search_v1','asset_search_remaining_v2_20261104'):
        for task in read(ROOT/'scenarios'/catalog/'manifest.json').get('tasks',[]):tasks[task['id']]=task
    # Prefer a manually repaired recipe only after both native gates pass.
    for task in read(ROOT/'scenarios/manual_dense_repairs_20261106/manifest.json').get('tasks',[]):
        d=ROOT/'runs/manual_dense_repairs_20261106/candidate'/task['id']
        a,b=read(d/'validation.json'),read(d/'restore_validation.json')
        sha=hashlib.sha256(Path(task['scenario']).read_bytes()).hexdigest()
        if (a.get('status')=='initialization_validated' and b.get('status')=='restore_validated'
            and a.get('scenario_sha256')==b.get('scenario_sha256')==sha):tasks[task['id']]=task
    rows=[];proposals=[]
    for task in tasks.values():
        path=ROOT/task['scenario'];spec=yaml.safe_load(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest()
        matches=[];accepted=False
        for run in RUNS:
            directory=ROOT/'runs'/run/task['id'];a=read(directory/'validation.json');b=read(directory/'restore_validation.json')
            if a.get('scenario_sha256')!=sha:continue
            matches.append((directory.stat().st_mtime,directory,a,b))
            frozen=directory/'frozen_member.private.json'
            if (a.get('status')=='initialization_validated' and b.get('status')=='restore_validated'
                and a.get('fresh_native_setup') and b.get('fresh_native_setup')
                and b.get('scenario_sha256')==sha and frozen.exists()
                and b.get('frozen_manifest_sha256')==hashlib.sha256(frozen.read_bytes()).hexdigest()):accepted=True
        latest=max(matches,key=lambda x:x[0]) if matches else None
        row={'id':task['id'],'parent':task['parent_task_id'],'scene':spec['scene']['model'],
             'scenario':str(path),'scenario_sha256':sha,'status':'accepted' if accepted else 'unattempted',
             'findings':[],'placement_evidence':{},'unstable_objects':[]}
        dims={}
        for obj in spec.get('objects',[]):
            meta=read(ASSETS/obj['category']/obj['model']/'misc/metadata.json')
            size=meta.get('bbox_size');scale=obj.get('scale',1.)
            if size:
                scale=scale if isinstance(scale,list) else [scale]*3
                dims[obj['name']]=[x*y for x,y in zip(size,scale)]
        volumes={};counts=Counter()
        if latest:
            _,directory,a,b=latest;report=b if a.get('status')=='initialization_validated' and b else a
            row['evidence_run']=str(directory);row['error']=report.get('error')
            if not accepted:
                row['status']='restore_pending' if a.get('status')=='initialization_validated' and not b else 'failed'
            for window in read(directory/'settling.private.json')[-1:] if isinstance(read(directory/'settling.private.json'),list) else []:
                row['unstable_objects']=[{'entity':k,**v} for k,v in window.get('objects',{}).items()
                    if v.get('linear_speed',0)>.01 or v.get('angular_speed',0)>.1
                    or (v.get('displacement_m') or 0)>.001 or (v.get('rotation_rad') or 0)>.01]
            logfile=directory/'native_validation.log'
            if logfile.exists():
                for line in logfile.read_text(errors='replace').splitlines():
                    if '[placement-volume] ' in line:
                        try:
                            header,payload=line.split('[placement-volume] ',1)[1].split(': ',1)
                            name,rec=header.split(' in ',1);data=ast.literal_eval(payload)
                            volumes[rec]={v['link']:v['extent_m'] for v in data['volumes']}
                            dims[name]=data['object_extent_m']
                        except (ValueError,KeyError,SyntaxError):pass
                    if '[inside-sampling] ' in line:
                        try:counts.update(json.loads(line.split('[inside-sampling] ',1)[1]).get('counts',{}))
                        except ValueError:pass
            row['sampling_counts']=dict(counts)
        groups=defaultdict(list)
        for placement in spec.get('placements',[]):
            if placement['relation']=='inside':groups[placement['receptacle']].append(placement)
        assignments={};safe=True
        for rec,group in groups.items():
            available=volumes.get(rec,{})
            if not available:
                row['findings'].append({'kind':'volume_evidence_missing','receptacle':rec});safe=False;continue
            row['placement_evidence'][rec]=available
            names={p['entity']:dims[p['entity']] for p in group if p['entity'] in dims}
            if len(names)!=len(group):safe=False;continue
            no_fit=[name for name,size in names.items() if not any(upright_fit(size,v) for v in available.values())]
            if no_fit:
                row['findings'].append({'kind':'upright_dimension_mismatch','receptacle':rec,'entities':no_fit,
                    'note':'Heuristic: metadata bbox upright/yaw fit, not proof against all possible poses.'});safe=False
            fixed={}
            for p in group:
                if not p.get('link'):continue
                link=p['link']
                matches=[k for k in available if k==link or k.startswith(f'meta__{link}_')]
                # Multiple native volumes behind a physical link are ambiguous.
                if len(matches)!=1:
                    safe=False;break
                fixed[p['entity']]=matches[0]
            else:
                plan=allocate(names,available,fixed)
                if plan is None:
                    row['findings'].append({'kind':'compartment_packing_needs_manual_review','receptacle':rec});safe=False
                elif len(available)>1 and any(p.get('link') is None for p in group):
                    unpinned={p['entity'] for p in group if not p.get('link')}
                    assignments.update({k:v for k,v in plan.items() if k in unpinned})
                    row['findings'].append({'kind':'explicit_compartment_candidate','receptacle':rec,'assignment':plan})
                continue
            row['findings'].append({'kind':'ambiguous_selected_link','receptacle':rec})
        # Do not reinterpret location-specific instructions or alter successful tasks.
        specific=bool(re.search(r'\b(drawer|upper|lower|topmost|bottommost|leftmost|rightmost)\b',spec['instruction'],re.I))
        if row['status']=='failed' and safe and assignments and not specific:
            revised=copy.deepcopy(spec)
            for p in revised['placements']:
                if p['entity'] in assignments:p['link']=assignments[p['entity']]
            if revised!=spec:
                proposals.append((task,revised,{'source_sha256':sha,'changes':assignments,
                    'status':'unvalidated_repair_candidate','unchanged':['assets','instruction','goal','receptacles','physics thresholds']}))
                row['repair_candidate']=True
        rows.append(row)
    out=ROOT/'runs/dense_layout_audit_20261106';out.mkdir(parents=True,exist_ok=True)
    summary={'updated_unix':time.time(),'physical_recipes':len(rows),'semantic_tasks':len({x['parent'] for x in rows}),
             'status_counts':dict(Counter(x['status'] for x in rows)),
             'finding_case_counts':dict(Counter(k for row in rows for k in {f['kind'] for f in row['findings']})),
             'unvalidated_compartment_repair_candidates':len(proposals),
             'note':'Evidence missing and heuristic fit warnings are not certified physical failures.'}
    (out/'audit.json').write_text(json.dumps({'summary':summary,'tasks':rows},indent=2))
    catalog=ROOT/'scenarios/dense_compartment_repairs_20261106';catalog.mkdir(exist_ok=True);manifest=[]
    for task,spec,evidence in proposals:
        d=catalog/task['id'];d.mkdir(exist_ok=True)
        # Freeze each proposed recipe once queued; later audits cannot mutate an
        # in-flight or accepted candidate. Further edits require explicit review.
        if not (d/'scenario.yaml').exists():
            (d/'scenario.yaml').write_text(yaml.safe_dump(spec,sort_keys=False))
            (d/'repair.json').write_text(json.dumps(evidence,indent=2))
        member=dict(task,scenario=str(d/'scenario.yaml'));manifest.append(member)
    (catalog/'manifest.json').write_text(json.dumps({'tasks':manifest,'audit':str(out/'audit.json')},indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

def source_layout_signature(spec, target):
    placement=next(p for p in spec['placements'] if p['entity']==target)
    source=placement['receptacle']
    obj=next((o for o in spec.get('objects',[]) if o['name']==source),None)
    location=next((p for p in spec['placements'] if p['entity']==source),None)
    return dict(scene=spec['scene']['model'],source=source,
                asset=[obj['category'],obj['model']] if obj else None,location=location,
                relation=placement['relation'],link=placement.get('link'))


def classify_failure(error):
    """Classify observed symptoms; do not infer a physical cause from a timeout."""
    error=str(error or '')
    if 'no Open state' in error or 'capability mismatch' in error:
        return 'asset_capability', 'missing_required_native_state'
    if 'LAYOUT_NOT_CONVERGED' in error:return 'stability', 'settling_not_converged'
    if 'POST_SETTLE_RELATION_FAILED' in error:return 'placement', 'post_settle_relation_lost'
    if 'OPEN_STATE_CHANGED' in error:return 'articulation', 'declared_open_state_changed'
    if 'nan' in error and 'quaternion' in error:return 'physics', 'nonfinite_articulation_pose'
    if 'invalid spawn' in error:return 'spawn', 'robot_obstacle_overlap'
    if 'native_process_exited_without_current_report' in error:return 'infrastructure', 'process_exit_without_report'
    if 'native_timeout' in error:return 'timeout', 'cause_unresolved'
    if 'placement failed verification' in error:
        return 'placement', 'post_physics_relation_failure'
    if 'CUDA out of memory' in error or 'No space left' in error:return 'infrastructure', 'resource_exhaustion'
    return 'unclassified', 'needs_trace_review'


def expert_audit():
    """Read-only triage of current failed semantic tasks, preserving all evidence."""
    queue=read(ROOT/'runs/asset_full_queue_20261105/queue.json')
    rows=[]
    for task in queue['tasks']:
        if task['status']!='awaiting_physical_repair':continue
        for iid,instance in task['instances'].items():
            directory=ROOT/'runs'/instance['run']/iid
            build=read(directory/'validation.json');restore=read(directory/'restore_validation.json')
            phase='restore' if build.get('status')=='initialization_validated' else 'build'
            report=restore if phase=='restore' else build
            if not report:continue
            family,subtype=classify_failure(report.get('error'))
            evidence=[];sampling=Counter();first_failed=None
            logfile=directory/'native_validation.log'
            if logfile.exists():
                for line in logfile.read_text(errors='replace').splitlines():
                    if '[placement-volume] ' in line:
                        try:
                            header,payload=line.split('[placement-volume] ',1)[1].split(': ',1)
                            obj,rec=header.split(' in ',1);data=ast.literal_eval(payload)
                            if data['volumes'] and all(v.get('size_fit') is False for v in data['volumes']):
                                item=dict(object=obj,receptacle=rec,size=data['object_extent_m'],volumes=data['volumes'])
                                if item not in evidence:evidence.append(item)
                        except (ValueError,KeyError,SyntaxError):pass
                    if '[placement] end ' in line and 'placed=False' in line and first_failed is None:
                        first_failed=line.strip()
                    if '[inside-sampling] ' in line:
                        try:sampling.update(json.loads(line.split('[inside-sampling] ',1)[1]).get('counts',{}))
                        except ValueError:pass
            if family=='placement' and evidence:subtype='upright_fit_warning'
            elif family=='placement' and first_failed:subtype='sampler_exhausted_or_rejected'
            rows.append(dict(task_id=task['id'],instance=iid,phase=phase,family=family,subtype=subtype,
                error=report.get('error'),evidence_directory=str(directory),scenario_sha256=report.get('scenario_sha256'),
                first_failed_placement=first_failed,dimension_evidence=evidence,sampling_counts=dict(sampling),
                confidence='observed_symptom_only',note='Upright bbox mismatch does not prove impossibility at every pose.'))
    groups=defaultdict(list)
    for row in rows:groups[(row['phase'],row['family'],row['subtype'])].append(row)
    summary=dict(updated=time.time(),queue_snapshot=queue['summary'],failed_semantic_tasks=sum(t['status']=='awaiting_physical_repair' for t in queue['tasks']),
        evidence_instances=len(rows),groups=[dict(phase=k[0],family=k[1],subtype=k[2],instances=len(v),
            distinct_tasks=len({r['task_id'] for r in v}),representatives=[r['instance'] for r in v[:3]])
            for k,v in sorted(groups.items(),key=lambda kv:-len(kv[1]))],
        note='Groups may overlap at semantic-task level; counts are not additive across groups.')
    out=ROOT/'runs/dense_layout_audit_20261106'
    out.mkdir(exist_ok=True)
    (out/'expert_triage.json').write_text(json.dumps(dict(summary=summary,instances=rows),indent=2))
    (out/'expert_triage_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def benchmark_layout_gate():
    """Replay frozen real-case dimensions; report model decisions, not SR."""
    import numpy as np
    import torch
    source=ROOT/'runs/failure_case_study_20261109/evidence.json'
    evidence=read(source);records=[]
    # Separate one-time Torch/CUDA initialization from measured model solves.
    warm=time.perf_counter()
    solve_layout({'warm':[.1,.1,.1]},{'warm_bin':[.3,.3,.3]},time_limit_s=1.)
    torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm
    _solve_layout_gpu_cached.cache_clear()
    for case in evidence['cases']:
        dimensions={}
        for obj in case['objects']:
            metadata=read(ASSETS/obj['category']/obj['model']/'misc/metadata.json')
            size=metadata.get('bbox_size');scale=obj.get('scale',1.)
            if size:
                scale=scale if isinstance(scale,list) else [scale]*3
                dimensions[obj['name']]=[a*b for a,b in zip(size,scale)]
        groups=defaultdict(list)
        for placement in case['placements']:
            if placement['relation']=='inside':groups[placement['receptacle']].append(placement)
        findings={f['receptacle']:f for f in (case.get('preflight') or {}).get('findings',[])}
        for rec,placements in groups.items():
            prior=findings.get(rec,{})
            volumes=prior.get('volume_extents_m',{});native=prior.get('native_links',{k:k for k in volumes})
            fixed={p['entity']:{k for k,v in native.items() if v==p['link'] or v.startswith('meta__'+p['link']+'_')} for p in placements if p.get('link')}
            record=dict(task=case['task'],receptacle=rec,items=len(placements),bins=len(volumes),reference_failure=case['error'])
            if not volumes or any(p['entity'] not in dimensions for p in placements) or any(not v for v in fixed.values()):
                records.append(dict(record,status='unknown_geometry',wall_seconds=0));continue
            started=time.perf_counter();hits=_solve_layout_gpu_cached.cache_info().hits
            result=solve_layout({p['entity']:dimensions[p['entity']] for p in placements},volumes,fixed,time_limit_s=1.)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            records.append(dict(record,**result,wall_seconds=elapsed,cache_hit=_solve_layout_gpu_cached.cache_info().hits>hits))
    measured=[r['wall_seconds'] for r in records if r['status']!='unknown_geometry']
    summary=dict(cases=len(evidence['cases']),container_groups=len(records),status_counts=dict(Counter(r['status'] for r in records)),
        cuda_device=torch.cuda.get_device_name(0),initialization_seconds=warm_seconds,
        model_solve_median_seconds=float(np.median(measured)),model_solve_p95_seconds=float(np.percentile(measured,95)),
        total_measured_wall_seconds=sum(measured),
        note='Frozen single-layer bbox model replay; no claims about mesh feasibility, native success rate or end-to-end speedup.')
    out=ROOT/'runs/failure_case_study_20261109/gpu_layout_gate_benchmark.json'
    out.write_text(json.dumps(dict(summary=summary,groups=records),indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    import sys
    if '--solver-benchmark' in sys.argv:benchmark_layout_gate()
    elif '--expert' in sys.argv:expert_audit()
    else:audit()

