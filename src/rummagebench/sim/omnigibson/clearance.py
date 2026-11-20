"""Declared authoring edits to wall geometry; never changes skill collision rules."""
from pathlib import Path
import json
import numpy as np


def warp_points(points, axis, source, target):
    """Continuous, monotone wall deformation, identity outside authored bounds."""
    source=np.asarray(source,float);target=np.asarray(target,float)
    if axis not in (0,1) or len(source)<2 or source.shape!=target.shape:
        raise ValueError('Invalid wall warp')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Nonfinite warp')
    if not (np.diff(source)>0).all() or not (np.diff(target)>0).all():raise ValueError('Warp must preserve order')
    if not np.allclose(source[[0,-1]],target[[0,-1]]):raise ValueError('Wall outer bounds must remain fixed')
    out=np.array(points,dtype=float,copy=True);x=out[:,axis];inside=(x>=source[0])&(x<=source[-1])
    out[inside,axis]=np.interp(x[inside],source,target)
    return out


def apply_clearance(backend, manifest_path):
    from pxr import UsdGeom,Usd,Gf,Vt
    from rummagebench.sim.omnigibson.usd_collision import iter_prims
    path=Path(manifest_path)
    doc=json.loads(path.read_text())
    if doc['scene_model']!=backend._scenario.scene.model:raise ValueError('Clearance layout mismatch')
    objects={o.name:o for o in backend._env.scene.objects}
    protected={o.name for o in backend._env.scene.robots}
    for scope in getattr(backend._env.task,'object_scopes',[]) or []:
        protected.update(getattr(o,'name','') for o in scope.values() if o is not None)
    spec=backend._scenario
    protected.update(spec.initial_states)
    protected.update(o.name for o in spec.objects)
    if spec.target is not None:protected.add(spec.target.entity)
    for placement in spec.placements:protected.update([placement.entity,placement.receptacle])
    removed=doc.get('removed_objects',[])
    if protected.intersection(removed):raise ValueError('Clearance cannot remove task-bound objects')
    for name in removed:
        if name not in objects:raise ValueError(f'Unknown removal: {name}')
    # Prepare all mesh edits before modifying anything. Native visual AND collision
    # meshes are warped with the same world-space map; no invisible collision bypass.
    edits={}
    for change in doc.get('wall_warps',[]):
        obj=objects[change['entity']]
        if obj.category!='walls':raise ValueError('Only wall assets may be deformed')
        for prim in iter_prims(obj.prim):
            if not prim.IsA(UsdGeom.Gprim):continue
            if not prim.IsA(UsdGeom.Mesh):raise ValueError(f'Unsupported wall primitive: {prim.GetTypeName()}')
            mesh=UsdGeom.Mesh(prim);points=edits.get(str(prim.GetPath()),(None,np.asarray(mesh.GetPointsAttr().Get(),float)))[1]
            if not len(points):continue
            mat=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            M=np.asarray(mat);world=np.c_[points,np.ones(len(points))]@M
            warped=warp_points(world[:,:3],change['axis'],change['source'],change['target'])
            if np.allclose(warped,world[:,:3]):continue
            local=(np.c_[warped,np.ones(len(points))]@np.linalg.inv(M))[:,:3]
            edits[str(prim.GetPath())]=(mesh,local)
    from rummagebench.sim.omnigibson.state_io import dump_state,load_state
    original_state=dump_state(backend._sim)
    backend._sim.stop()
    with backend._sim.editing_usd():
        for mesh,points in edits.values():
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)),Gf.Vec3f(*points.max(0))])
    backend._sim.play()
    load_state(backend._sim,original_state,settle_steps=0)
    for name in removed:
        backend._env.scene.remove_object(objects[name])
    for change in doc.get('translations',[]):
        obj=objects[change['entity']];pos,ori=obj.get_position_orientation()
        import torch
        obj.set_position_orientation(pos+torch.tensor(change['delta'],dtype=pos.dtype,device=pos.device),ori)
        # Legacy authoring anchors / explicit on-top placements follow moved
        # receptacles; these coordinates remain evaluator-private.
        if change['entity'] in spec.anchors:
            anchor=spec.anchors[change['entity']]
            anchor.position=[v+d for v,d in zip(anchor.position,change['delta'])]
        for placement in spec.placements:
            if placement.receptacle==change['entity'] and placement.position_xy is not None:
                placement.position_xy=[v+d for v,d in zip(placement.position_xy,change['delta'][:2])]

    backend._collision_body_cache=None
    backend._navigation_native_cache={}
    # Geometry is persistent USD state, object poses enter the normal reset snapshot.
    return dict(manifest=str(path),modified_meshes=len(edits),translations=doc.get('translations',[]),removed_objects=removed,
                scope='Authored geometry; no route supplied to player',validation='pending')
