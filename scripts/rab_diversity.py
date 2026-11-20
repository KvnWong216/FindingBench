#!/usr/bin/env python3
"""Inspect real composed RAB USD geometry, placement, appearance and dynamics.

CPU USD evidence only. Asset names and different file/layout IDs never count
automatically as different categories, house types or physical validation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from functools import lru_cache


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@lru_cache(maxsize=4096)
def file_signature(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def opinion(value):
    from pxr import Sdf
    if isinstance(value,Sdf.AssetPath):
        path=Path(value.resolvedPath or value.path)
        return dict(asset_sha256=file_signature(str(path))) if path.is_file() else dict(unresolved_asset=value.path)
    return str(value)


def geometry(stage):
    import numpy as np
    from pxr import UsdGeom,UsdPhysics
    cache=UsdGeom.XformCache();shapes=[];placement=[];appearance=[];architecture=[]
    architecture_world=[];mesh_count=0
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        shell=path.startswith('/World/Background/Shell/') and (
            path.count('/')==4 and re.search(
                r'/(?:[NP][XY][a-z]?_|DrFr|SkN|SkP|[^/]*(?:Wall|Floor|Ceiling|Return))',
                path.removeprefix('/World/Background/Shell'),re.I))
        if prim.IsA(UsdGeom.Cube) and shell:
            shape=signature(('Cube',UsdGeom.Cube(prim).GetSizeAttr().Get()))
            matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).round(6).tolist()
            architecture.append(shape)
            architecture_world.append((shape,matrix))
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim);points=mesh.GetPointsAttr().Get()
            if points is None:continue
            vertices=np.asarray(points,dtype='<f8').round(6)
            if not np.isfinite(vertices).all():raise ValueError('NONFINITE_RAB_GEOMETRY:'+str(prim.GetPath()))
            parts=[vertices.tobytes()]
            for attr in (mesh.GetFaceVertexCountsAttr(),mesh.GetFaceVertexIndicesAttr()):
                value=attr.Get();parts.append(np.asarray(value if value is not None else [],dtype='<i8').tobytes())
            shape=hashlib.sha256(b''.join(parts)).hexdigest()
            matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).round(6).tolist()
            shapes.append(shape);placement.append((shape,matrix));mesh_count+=1
            # Identify architecture conservatively, independently of furniture
            # placement. Record exact paths so the grouping can be reviewed.
            if shell or re.search(r'(^|[/_])(wall|walls|floor|floors|ceiling|roof|window|windows)([/_0-9]|$)',path,re.I):
                architecture.append(shape);architecture_world.append((shape,matrix))
        if prim.GetTypeName() in ('Material','Shader'):
            props=[]
            for attr in prim.GetAttributes():
                if attr.GetName().startswith(('inputs:','info:')):
                    props.append((attr.GetName(),opinion(attr.Get())))
            appearance.append(sorted(props))
    def multiset(seq):return sorted(seq,key=lambda v:json.dumps(v,sort_keys=True))
    return dict(meshes=mesh_count,geometry_multiset_signature=signature(multiset(shapes)),
        furniture_and_architecture_placement_signature=signature(multiset(placement)),
        material_opinion_signature=signature(multiset(appearance)),
        architectural_mesh_count=len(architecture),
        architectural_geometry_signature=signature(multiset(architecture)) if architecture else None,
        architectural_world_geometry_signature=signature(multiset(architecture_world)) if architecture else None)


def asset_record(row,namespace,bundle):
    from pxr import Usd,UsdGeom,UsdPhysics
    source=namespace.resolve(row['logical'])
    relocated=bundle.clone(source);stage=Usd.Stage.Open(str(relocated))
    if stage is None:raise ValueError('RAB_ASSET_STAGE_OPEN_FAILED:'+row['id'])
    root=stage.GetDefaultPrim()
    if not root:raise ValueError('RAB_ASSET_DEFAULT_PRIM_MISSING:'+row['id'])
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=False)
    bounds=bbox.ComputeWorldBound(root).ComputeAlignedRange()
    dims=[float(v) for v in bounds.GetSize()]
    if not all(math.isfinite(v) and v>=0 for v in dims):raise ValueError('RAB_ASSET_BOUNDS_INVALID:'+row['id'])
    bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    body_records=[dict(path=str(p.GetPath()),enabled=p.GetAttribute('physics:rigidBodyEnabled').Get() is not False,
        kinematic=p.GetAttribute('physics:kinematicEnabled').Get() is True) for p in bodies]
    colliders=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
               and p.GetAttribute('physics:collisionEnabled').Get() is not False]
    eligible=len(bodies)==1 and body_records[0]['enabled'] and not body_records[0]['kinematic'] and bool(colliders)
    return dict(row,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        default_prim=str(root.GetPath()),dimensions_m=dims,rigid_bodies=body_records,colliders=colliders,
        dynamic_single_body_candidate=eligible,native_physx_validated=False,
        **geometry(stage))


def discover_entries(config):
    """Include model entries outside all_assets; inventory isn't just 82 IDs.

    A raw model is only a candidate: it may have no physics or default prim.
    Keep the full resource list for root/scene-family review separately.
    """
    prefix='libraries/current_assets/all_assets/'
    rows=[dict(id=Path(k[len(prefix):]).parts[0],kind='registered_wrapper',
        logical='/home/lingjun/robot_agent_arena/api-protocol/'+k[len('libraries/current_assets/'):])
        for k in config['resources'] if k.startswith(prefix) and k.endswith('/asset.usda')
        and len(Path(k[len(prefix):]).parts)==2]
    seen={str(Path(config['resources'][k]).resolve()) for k in config['resources']
          if k.startswith(prefix) and k.endswith('/asset.usda') and len(Path(k[len(prefix):]).parts)==2}
    inventory=[]
    for location,value in config['resources'].items():
        if not location.startswith('profiles/room/'):continue
        path=Path(value)
        if path.suffix.lower() not in ('.usd','.usda','.usdc','.usdz'):continue
        inventory.append(dict(location=location,path=str(path),source_sha256=file_signature(str(path))))
        named_model=('/usdz/' in location or '/newcans/' in location or '/photo_assets/' in location
            or '/resources/shared_assets/' in location or '/interior_resources_v1/' in location)
        if not named_model or str(path.resolve()) in seen:continue
        seen.add(str(path.resolve()))
        rows.append(dict(id='source_'+signature(location)[:12],kind='additional_source_model',
                         source_location=location,logical=str(path.resolve())))
    return rows,inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.repo.resolve()/'src'))
    from pxr import Usd
    from rab_assets import AssetNamespace,LayerBundle
    args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.runtime/'runtime.json').read_text());namespace=AssetNamespace.from_runtime(args.runtime)
    bundle=LayerBundle(namespace,args.output/'relocated_layers')
    rows,source_inventory=discover_entries(config)
    assets=[];layouts=[];errors=[];unclassified=[];started=time.time()
    def save(status):
        (args.output/'progress.json').write_text(json.dumps(dict(status=status,assets=len(assets),layouts=len(layouts),errors=errors,updated_at=time.time()),indent=2))
    for row in rows:
        try:assets.append(asset_record(row,namespace,bundle))
        except Exception as exc:
            record=dict(kind='asset',id=row['id'],logical=row['logical'],error=str(exc))
            (errors if row['kind']=='registered_wrapper' else unclassified).append(record)
        save('inspecting_assets')
    # Only original canonical layout resources; aliases do not add diversity.
    prefix='libraries/original_absolute/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/'
    candidates=[k for k in config['resources'] if k.startswith(prefix) and Path(k).suffix.lower() in ('.usd','.usda','.usdc')]
    for location in sorted(candidates):
        try:
            source=Path(config['resources'][location]);relocated=bundle.clone(source);stage=Usd.Stage.Open(str(relocated))
            if stage is None:raise ValueError('RAB_LAYOUT_STAGE_OPEN_FAILED')
            from rab_assets import joints
            layouts.append(dict(id=Path(location).stem,source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                joints_schema_only=joints(stage),native_physx_validated=False,**geometry(stage)))
        except Exception as exc:errors.append(dict(kind='layout',location=location,error=str(exc)))
        save('inspecting_layouts')
    def counts(field):return dict(Counter(r[field] for r in layouts if r.get(field)))
    model_groups={}
    for row in assets:model_groups.setdefault(row['geometry_multiset_signature'],[]).append(row['id'])
    report=dict(schema='findingbench.rab_diversity.v1',started_at=started,finished_at=time.time(),
        status='cpu_usd_audit_passed' if not errors and len(layouts)==50 else 'incomplete',
        asset_wrappers=sum(r['kind']=='registered_wrapper' for r in assets),
        additional_source_models=sum(r['kind']=='additional_source_model' for r in assets),
        source_model_entries_requiring_root_or_capability_review=unclassified,
        source_native_file_inventory=source_inventory,
        dynamic_single_body_candidates=sum(r['dynamic_single_body_candidate'] for r in assets),
        asset_geometry_groups=model_groups,
        layouts=len(layouts),architectural_geometry_groups=counts('architectural_geometry_signature'),
        architectural_world_geometry_groups=counts('architectural_world_geometry_signature'),
        full_placement_groups=counts('furniture_and_architecture_placement_signature'),
        material_groups=counts('material_opinion_signature'),assets=assets,layout_records=layouts,errors=errors,
        caveats=['Geometry groups are measured signatures, not semantic house-type labels.',
                 'Joint schemas are not native opening/closing admission.',
                 'Dynamic body schemas are not settled placement or graspability validation.'],
        resource_relocation=bundle.receipt())
    (args.output/'diversity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    save(report['status']);print(json.dumps({k:report[k] for k in ('status','asset_wrappers','dynamic_single_body_candidates','layouts')}))
    return 0 if report['status']=='cpu_usd_audit_passed' else 2


if __name__=='__main__':raise SystemExit(main())
