"""Local, evaluator-only collision geometry shared by Gen and OmniGibson.

This is a placement model, never a physics certificate. Only supported USD
collision representations are exported; visual meshes and envelope fallbacks
are forbidden. Licensed geometry stays in the ignored build directory.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.native_occupancy import plain
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path


def digest(doc):
    return sha256_bytes(canonical_json({k: v for k, v in doc.items() if k != 'sha256'}))


def native_scene_snapshot(source, saved):
    """Keep the original inventory, actual loaded state and runtime versions."""
    result=deepcopy(source)
    names=set(source['objects_info']['init_info'])
    native=saved['state']['registry']['object_registry']
    if not names <= set(native):
        raise ValueError('COLLISION_NATIVE_STATE_INCOMPLETE')
    result['state']=deepcopy(saved['state'])
    result['state']['registry']['object_registry']={name:deepcopy(native[name]) for name in sorted(names)}
    result['versions']=deepcopy(saved['versions'])
    return result


def checked_mesh(vertices, faces):
    v, f = np.asarray(vertices, float), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('COLLISION_MESH_INVALID')
    return v, f.astype(np.int64)


def hull_mesh(vertices):
    """Outward triangle faces, including concavities filled by convexHull."""
    v = np.asarray(vertices, float)
    hull = ConvexHull(v)
    faces = hull.simplices.copy()
    normal = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    flip = np.sum(normal * hull.equations[:, :3], axis=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return checked_mesh(v, faces)


def supporting_bottom_faces(vertices, faces):
    """Find real downward faces whose plane supports the *whole* collider.

    Gen's canonical 1%-height band can contain no complete triangle on curved
    native assets. Use existing exterior triangles, never an invented flat
    bottom or a box. A solver may align this slightly tilted plane; later
    native physics still decides whether that pose is stable.
    """
    v,f=checked_mesh(vertices,faces)
    normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    lengths=np.linalg.norm(normals,axis=1)
    scale=float(np.linalg.norm(np.ptp(v,axis=0)))
    valid=lengths>max(scale*scale*1e-12,1e-18)
    normals[valid]/=lengths[valid,None]
    candidates=np.flatnonzero(valid & (normals[:,2]<-.5))
    # Most nearly horizontal first. Deterministic ties preserve mesh ordering.
    candidates=candidates[np.argsort(normals[candidates,2],kind='stable')]
    tolerance=max(scale*1e-7,1e-9)
    for i in candidates:
        if np.max((v-v[f[i,0]])@normals[i])<=tolerance:
            mask=np.zeros(len(f),dtype=bool);mask[i]=True
            return mask
    raise ValueError('COLLISION_NO_REAL_SUPPORTING_BOTTOM_FACE')


def world_to_local(vertices, center, quaternion, *, factory=False):
    """Keep the existing layout front convention: OG yaw = Gen yaw + 90°."""
    v = (np.asarray(vertices) - center) @ Rotation.from_quat(quaternion).as_matrix()
    if factory:
        v = v @ Rotation.from_euler('z', 90, degrees=True).as_matrix().T
    return v


def layout_pose(matrix):
    """Reject solver transforms that Placement's yaw-only schema cannot carry."""
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():
        raise ValueError('COLLISION_LAYOUT_TRANSFORM_INVALID')
    yaw=np.arctan2(matrix[1,0],matrix[0,0])
    expected=np.eye(4)
    expected[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    expected[:3,3]=matrix[:3,3]
    if not np.allclose(matrix,expected,atol=1e-6,rtol=0):
        raise ValueError('COLLISION_LAYOUT_TILT_OR_SCALE_UNSUPPORTED')
    return matrix[:3,3].tolist(),float(np.degrees(yaw)+180)


def native_layout_pose(matrix):
    """Carry the whole rigid pose across the Gen/BEHAVIOR front convention.

    Tilts are legitimate proposals on a non-axis-aligned support face. Physics
    decides their stability. Scale, reflection and shear cannot be reproduced
    by a native object pose and must reject.
    """
    matrix=np.asarray(matrix,float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-7,rtol=0)):
        raise ValueError('COLLISION_LAYOUT_TRANSFORM_INVALID')
    rot=matrix[:3,:3]
    if (not np.allclose(rot.T@rot,np.eye(3),atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.det(rot),1.,atol=1e-6,rtol=0)):
        raise ValueError('COLLISION_LAYOUT_SCALE_OR_SHEAR_UNSUPPORTED')
    native=Rotation.from_matrix(rot)*Rotation.from_euler('z',90,degrees=True)
    return dict(position=matrix[:3,3].tolist(),orientation=native.as_quat().tolist())


def rigid_rotation_update(matrix, axis, angle):
    """Apply Gen's world-axis rotation without accumulating matrix scale.

    The upstream float32 matrix→scale→matrix round trip changes unit scale
    after repeated rotations. Keep rotation in SO(3); do not modify vertices
    or relax the final rigid-transform validation.
    """
    native_layout_pose(matrix)  # reject any already non-rigid transform
    axis=np.asarray(axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isfinite(angle) or np.linalg.norm(axis)<1e-12:
        raise ValueError('COLLISION_ROTATION_AXIS_INVALID')
    return (Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle)
            * Rotation.from_matrix(np.asarray(matrix)[:3,:3])).as_quat()


def collision_mesh(obj):
    """Export every enabled collider in live world coordinates, or reject.

    USD's convexHull point hull is explicit (PhysX cooking/contact margins
    remain native validation work). Cooked decomposition, SDF and primitives
    requiring tessellation are unsupported rather than replaced by boxes.
    """
    from pxr import UsdGeom, UsdPhysics
    from rummagebench.sim.omnigibson.usd_collision import (
        iter_prims, compose_link_matrix, box_corners, _BOX_TRIS)
    vertices, faces, inventory = [], [], []
    offset = 0
    for prim in iter_prims(obj.prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        body = prim
        while body != obj.prim and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        base = np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0.), float)
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            link = obj.links.get(body.GetName())
            if link is None:
                raise ValueError(f'COLLISION_LIVE_LINK_MISSING:{body.GetPath()}')
            pos, ori = plain(link.get_position_orientation())
            if not np.isfinite([*pos, *ori]).all() or not .999 < np.linalg.norm(ori) < 1.001:
                raise ValueError('COLLISION_LIVE_POSE_INVALID')
            live = compose_link_matrix(base, pos, ori)
        else:
            live = base
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.), float) @ np.linalg.inv(base) @ live
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or 'none')
        if prim.IsA(UsdGeom.Cube) and approximation == 'none':
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            v, f = box_corners(np.ones(3) * size / 2), np.asarray(_BOX_TRIS)
        elif prim.IsA(UsdGeom.Mesh) and approximation in ('none', 'convexHull'):
            mesh = UsdGeom.Mesh(prim)
            v = np.asarray(mesh.GetPointsAttr().Get(), float)
            if approximation == 'convexHull':
                v, f = hull_mesh(v)
            else:
                counts = list(mesh.GetFaceVertexCountsAttr().Get())
                if not counts or any(n != 3 for n in counts):
                    raise ValueError(f'COLLISION_TRIANGULATION_REQUIRED:{prim.GetPath()}')
                f = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        else:
            raise ValueError(f'COLLISION_REPRESENTATION_UNSUPPORTED:{prim.GetTypeName()}:{approximation}:{prim.GetPath()}')
        v, f = checked_mesh(v, f)
        v = (np.c_[v, np.ones(len(v))] @ matrix)[:, :3]
        if np.linalg.det(matrix[:3, :3]) < 0:
            f = f[:, [0, 2, 1]]
        vertices.extend(v.tolist()); faces.extend((f + offset).tolist()); offset += len(v)
        inventory.append(dict(prim=str(prim.GetPath()), kind=prim.GetTypeName(),
                              approximation=approximation, vertices=len(v), triangles=len(f)))
    if not inventory:
        if getattr(obj,'visual_only',False) is True:
            return dict(vertices=[],faces=[],colliders=[],noncolliding_visual_only=True)
        raise ValueError(f'COLLISION_GEOMETRY_UNAVAILABLE:{obj.name}')
    checked_mesh(vertices, faces)
    return dict(vertices=vertices, faces=faces, colliders=inventory)


def load_geometry(path, *, scene=None, assets=None):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    if doc.get('version') != 1 or doc.get('status') != 'geometry_exported_pending_physics' or doc.get('sha256') != digest(doc):
        raise ValueError('COLLISION_EXPORT_INVALID')
    if scene is not None and doc['scene'] != scene:
        raise ValueError('COLLISION_SCENE_MISMATCH')
    for p, expected in doc['inputs'].items():
        if not expected or sha256_path(p) != expected:
            raise ValueError(f'COLLISION_INPUT_CHANGED:{p}')
    snapshot = json.loads(Path(doc['scene_snapshot']).read_text())
    if snapshot['state']['registry'].get('system_registry'):
        raise ValueError('COLLISION_SYSTEM_STATE_RESTORE_UNSUPPORTED')
    names = set(snapshot['objects_info']['init_info'])
    if set(doc['native']) != names or set(snapshot['state']['registry']['object_registry']) != names:
        raise ValueError('COLLISION_NATIVE_COVERAGE_INCOMPLETE')
    if doc['inputs'].get(doc['scene_snapshot']) != sha256_path(doc['scene_snapshot']):
        raise ValueError('COLLISION_SNAPSHOT_UNBOUND')
    # A measured approved library may serve several smaller sampling pools.
    # Every requested record must match exactly; extra measured prototypes are
    # not introduced into the request's whitelist or solver factories.
    if assets is not None and (not assets or any(doc['asset_records'].get(k)!=v for k,v in assets.items())):
        raise ValueError('COLLISION_ASSET_POOL_MISMATCH')
    if set(doc['assets']) != set(doc['asset_records']):
        raise ValueError('COLLISION_ASSET_COVERAGE_INCOMPLETE')
    for name,mesh in [*doc['native'].items(), *doc['assets'].items()]:
        if mesh.get('noncolliding_visual_only') is True:
            args=snapshot['objects_info']['init_info'].get(name,{}).get('args',{})
            if (args.get('visual_only') is not True or mesh['vertices'] or mesh['faces'] or mesh['colliders']):
                raise ValueError('COLLISION_VISUAL_ONLY_EVIDENCE_INVALID')
            continue
        checked_mesh(mesh['vertices'], mesh['faces'])
        if not mesh['colliders']:
            raise ValueError('COLLISION_INVENTORY_MISSING')
        if any((c['kind'],c['approximation']) not in
               {('Mesh','none'),('Mesh','convexHull'),('Cube','none')} for c in mesh['colliders']):
            raise ValueError('COLLISION_REPRESENTATION_UNSUPPORTED')
    return doc


def export_geometry(backend, context, records, output):
    """Save loaded native state first, then import isolated asset prototypes."""
    from omnigibson.objects import DatasetObject
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(context['scene_json']).read_text())
    saved = plain(backend._env.scene.save(as_dict=True))
    names = context['retained_native_objects']
    # Original init_info retains every native body and all scene metadata;
    # state comes from this actual scene, excluding the authoring robot.
    source=native_scene_snapshot(source,saved)
    snapshot = output / 'native_scene.json'
    atomic_write_json(snapshot, source)
    doc = dict(version=1, status='running', scene=context['scene'], scene_snapshot=str(snapshot),
               authoring_seed=backend._seed,
               inputs={**context['inputs'], str(snapshot): sha256_path(snapshot)},
               native={}, assets={}, asset_records=records, failures={}, evaluation_ready_episodes=0,
               scope='live USD collision mesh/hull placement model; cooking margins, stability and skills unverified')
    for a in records.values():
        doc['inputs'].update(a['sources'])
    try:
        from rummagebench.sim.omnigibson.behavior_adapter import asset_record
        root = Path(context['scene_json']).parents[3]
        for info in source['objects_info']['init_info'].values():
            args = info['args']
            doc['inputs'].update(asset_record(root,args['category'],args['model'])['sources'])
        if saved['state']['registry'].get('system_registry'):
            raise ValueError('COLLISION_SYSTEM_STATE_RESTORE_UNSUPPORTED')
        for name in names:
            try:
                doc['native'][name] = collision_mesh(resolve_object(backend._env.scene, name))
            except Exception as exc:
                doc['failures'][name] = str(exc)
        backend._sim.stop()
        prototypes = {}
        for i, (key, record) in enumerate(sorted(records.items())):
            obj = DatasetObject(name=f'geometry_prototype_{i}', category=record['category'], model=record['model'],
                                fixed_base=False, scale=[1., 1., 1.])
            backend._env.scene.add_object(obj)
            obj.set_position_orientation(position=[100. + i * 3, 100., 10.], orientation=[0., 0., 0., 1.])
            prototypes[key] = obj
        backend._sim.play(); backend._sim.step()
        for key, obj in prototypes.items():
            try:
                mesh = collision_mesh(obj)
                pos, ori = plain(obj.get_position_orientation())
                center = np.asarray(pos) + Rotation.from_quat(ori).apply(records[key]['base_link_offset'])
                mesh['vertices'] = world_to_local(mesh['vertices'], center, ori, factory=True).tolist()
                doc['assets'][key] = mesh
            except Exception as exc:
                doc['failures'][key] = str(exc)
        doc['status'] = 'rejected' if doc['failures'] else 'geometry_exported_pending_physics'
    except Exception as exc:
        doc['status'] = 'rejected'; doc['failures']['export'] = str(exc)
    doc['sha256'] = digest(doc)
    atomic_write_json(output / 'collision_geometry.json', doc)
    if doc['status'] != 'geometry_exported_pending_physics':
        raise ValueError('COLLISION_EXPORT_REJECTED:' + json.dumps(doc['failures']))
    return doc


def restore_native_state(env, manifest):
    """Rebind the simulator to the exact source state used for Gen geometry.

    Per-object native restore avoids overwriting newly imported daily objects.
    No steps occur between restore and geometry comparison. Future dynamics
    are still checked by the existing realizer.
    """
    from omnigibson.utils.python_utils import recursively_convert_to_torch
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    path = manifest['living_context']['geometry']
    doc = load_geometry(path, scene=manifest['base_scene'])
    saved = json.loads(Path(doc['scene_snapshot']).read_text())
    for name, state in saved['state']['registry']['object_registry'].items():
        resolve_object(env.scene, name).load_state(recursively_convert_to_torch(state), serialized=False)
    comparisons = {}
    for name, expected in doc['native'].items():
        actual = collision_mesh(resolve_object(env.scene, name))
        if expected.get('noncolliding_visual_only'):
            if actual!=expected:raise ValueError(f'COLLISION_VISUAL_ONLY_CHANGED:{name}')
            comparisons[name]=0.
            continue
        if actual['colliders'] != expected['colliders'] or actual['faces'] != expected['faces']:
            raise ValueError(f'COLLISION_RESTORE_TOPOLOGY_MISMATCH:{name}')
        delta = float(np.max(np.abs(np.asarray(actual['vertices']) - expected['vertices'])))
        comparisons[name] = delta
        if delta > 1e-5:
            raise ValueError(f'COLLISION_RESTORE_POSE_MISMATCH:{name}:{delta}')
    for item in manifest['objects']:
        p=item['placement'];key=f"{p['category']}/{p['model_id']}"
        obj=resolve_object(env.scene,p['name'])
        actual=collision_mesh(obj);pos,ori=plain(obj.get_position_orientation())
        center=np.asarray(pos)+Rotation.from_quat(ori).apply(doc['asset_records'][key]['base_link_offset'])
        vertices=world_to_local(actual['vertices'],center,ori,factory=True)
        expected=doc['assets'][key]
        if (actual['faces']!=expected['faces'] or vertices.shape!=np.asarray(expected['vertices']).shape
                or not np.allclose(vertices,expected['vertices'],atol=1e-5,rtol=0)):
            raise ValueError(f'COLLISION_ASSET_SHAPE_MISMATCH:{p["name"]}')
    return dict(status='native_state_and_collision_geometry_restored', geometry_sha256=doc['sha256'],
                max_vertex_error_m=max(comparisons.values()), objects=len(comparisons),
                evaluation_ready_episodes=0)
