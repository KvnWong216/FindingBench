"""Native BEHAVIOR asset placements for the existing OmniGibson builder.

CPU imports only. Furniture keeps its original joints, collision and object
states. Layout coordinates use a -Y front; BEHAVIOR assets use +X. No mesh
conversion, procedural replacements, or nonuniform scaling is involved.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from rummagebench.authoring.layout.asset_catalog import AssetCatalog, AssetCategory, AssetEntry
from rummagebench.authoring.layout.generator import LayoutGenerator
from rummagebench.authoring.layout.spec import LayoutConfig, Placement
from rummagebench.authoring.layout.validator import LayoutValidator
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.feasibility.ik_solver import quat_rotate


def asset_record(root: Path, category: str, model: str) -> dict:
    if not all(re.fullmatch(r'[a-zA-Z0-9_]+', s) for s in (category, model)):
        raise ValueError('exact category/model identifiers required')
    directory = Path(root).resolve() / 'objects' / category / model
    metadata = directory / 'misc/metadata.json'
    usd = directory / 'usd' / f'{model}.encrypted.usd'
    if not usd.is_file():
        usd = directory / 'usd' / f'{model}.usd'
    if not metadata.is_file() or not usd.is_file():
        raise ValueError(f'BEHAVIOR_ASSET_MISSING:{category}/{model}')
    data = json.loads(metadata.read_text())
    size = np.asarray(data.get('bbox_size'), dtype=float)
    offset = np.asarray(data.get('base_link_offset'), dtype=float)
    if (size.shape != (3,) or offset.shape != (3,)
            or not np.isfinite(np.r_[size, offset]).all() or np.any(size <= 0)):
        raise ValueError(f'BEHAVIOR_ASSET_GEOMETRY_UNKNOWN:{category}/{model}')
    return dict(category=category, model=model, bbox_size=size.tolist(),
                base_link_offset=offset.tolist(), openable_joint_ids=data.get('openable_joint_ids', []),
                sources={str(metadata): sha256_path(metadata), str(usd): sha256_path(usd)})


def object_config(placement: Placement, asset: dict, native_pose=None) -> dict:
    # The layout sampler reserves a strip along local -Y. Rotate the native
    # +X front into that direction; its X/Y extents were swapped for sampling.
    yaw = math.radians(placement.orientation_deg - 90)
    quat = [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
    if native_pose is not None:
        if set(native_pose)!={'position','orientation'} or native_pose['position']!=placement.position:
            raise ValueError('NATIVE_PLACEMENT_POSE_MISMATCH')
        quat=np.asarray(native_pose['orientation'],float)
        if quat.shape!=(4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat),1.,atol=1e-6,rtol=0):
            raise ValueError('NATIVE_PLACEMENT_QUATERNION_INVALID')
        quat=(quat/np.linalg.norm(quat)).tolist()
    root = np.asarray(placement.position) - quat_rotate(quat, asset['base_link_offset'])
    return dict(type='DatasetObject', name=placement.name, category=asset['category'],
                model=asset['model'], fixed_base=placement.role != 'clutter',
                position=root.tolist(), orientation=quat, pose_frame='world', scale=[1., 1., 1.])


def prepare_layout(config: LayoutConfig, pool: dict, assets_root: Path, seed: int) -> dict:
    records, categories = {}, {}
    for role, choices in pool.items():
        categories[role] = []
        for choice in choices:
            a = asset_record(assets_root, choice['category'], choice['model'])
            key = f"{a['category']}/{a['model']}"
            if role == 'storage' and not a['openable_joint_ids']:
                raise ValueError(f'STORAGE_HAS_NO_NATIVE_OPENABLE_JOINT:{key}')
            records[key] = a
            x, y, z = a['bbox_size']
            categories[role].append(AssetCategory(category=a['category'], models=[
                AssetEntry(model_id=a['model'], aabb_size=[y, x, z])]))
    catalog = AssetCatalog(categories)
    layout = LayoutGenerator(catalog, config).generate(seed)
    checks = LayoutValidator(config).validate(layout, catalog)
    if not checks['structural_valid']:
        raise ValueError('LAYOUT_REJECTED:' + json.dumps({
            'rejected': layout.rejected,
            'checks': {k: v for k, v in checks['checks'].items() if v['status'] == 'fail'}}))
    objects = []
    for p in layout.placements:
        asset = records[f'{p.category}/{p.model_id}']
        objects.append(dict(placement=p.model_dump(), og_config=object_config(p, asset)))
    doc = dict(version=1, source='behavior_native_layout', status='pending_simulation',
               base_scene=config.base_scene, layout_seed=seed, config=config.model_dump(),
               assets=records, objects=objects, rejected=layout.rejected,
               structural_checks=checks['checks'], evaluation_ready_episodes=0)
    doc['manifest_sha256'] = sha256_bytes(canonical_json(doc))
    return doc


def load_manifest(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if (doc.get('version') not in (1, 2) or doc.get('source') != 'behavior_native_layout'
            or doc.get('status') != 'pending_simulation'):
        raise ValueError('BEHAVIOR_ASSEMBLY_SCHEMA_INVALID')
    if doc.get('manifest_sha256') != sha256_bytes(canonical_json({
            k: v for k, v in doc.items() if k != 'manifest_sha256'})):
        raise ValueError('BEHAVIOR_ASSEMBLY_HASH_MISMATCH')
    names = set()
    native_poses = {}
    if doc['version'] == 2:
        context = doc.get('living_context')
        if not context or doc['base_scene'] == '__empty__':
            raise ValueError('NATIVE_HOME_CONTEXT_REQUIRED')
        required_inputs = {**context['native']['inputs'],
            context['solver_request']:sha256_path(context['solver_request']),
            context['solver_result']:sha256_path(context['solver_result'])}
        if any(context['inputs'].get(p) != digest or not digest for p,digest in required_inputs.items()):
            raise ValueError('NATIVE_HOME_INPUT_BINDING_INCOMPLETE')
        for source, digest in context['inputs'].items():
            if not digest or sha256_path(source) != digest:
                raise ValueError(f'NATIVE_HOME_INPUT_CHANGED:{source}')
        request = json.loads(Path(context['solver_request']).read_text())
        result = json.loads(Path(context['solver_result']).read_text())
        if 'geometry' in request or 'geometry' in context:
            from rummagebench.authoring.layout.indoor_geometry import load_geometry
            geometry = context.get('geometry')
            if not geometry or request.get('geometry') != geometry:
                raise ValueError('COLLISION_MANIFEST_BINDING_MISSING')
            geometry_hash = sha256_path(geometry)
            if any(h != geometry_hash for h in (context['inputs'].get(geometry),
                    request.get('geometry_sha256'),result.get('geometry_sha256'))):
                raise ValueError('COLLISION_MANIFEST_BINDING_MISMATCH')
            exported = load_geometry(geometry,scene=doc['base_scene'],assets=request['assets'])
            if context['native']['scene_json'] != exported['scene_snapshot']:
                raise ValueError('COLLISION_SCENE_STATE_MISMATCH')
            native_poses=result.get('native_poses',{})
            if set(native_poses)!={p['name'] for p in result['placements']}:
                raise ValueError('COLLISION_NATIVE_POSES_INCOMPLETE')
        from rummagebench.authoring.layout.indoor_background import verify_result
        verify_result(request, result)
        if (result.get('status') != 'constraints_satisfied'
                or result.get('request_sha256') != sha256_path(context['solver_request'])
                or not result.get('violations') or any(v != 0 for v in result['violations'].values())
                or context['native'] != request['native_context']
                or doc['base_scene'] != context['native']['scene']
                or [o['placement'] for o in doc['objects']] != result['placements']):
            raise ValueError('NATIVE_HOME_SOLVER_BINDING_MISMATCH')
    for a in doc['assets'].values():
        for source, digest in a['sources'].items():
            if sha256_path(source) != digest or not digest:
                raise ValueError(f'BEHAVIOR_ASSET_CHANGED:{source}')
        # Re-read measured metadata, rather than trusting a modified transform
        # together with a freshly recomputed manifest hash.
        metadata = next(Path(p) for p in a['sources'] if p.endswith('/misc/metadata.json'))
        if a != asset_record(metadata.parents[4], a['category'], a['model']):
            raise ValueError('BEHAVIOR_ASSET_METADATA_MISMATCH')
    for item in doc['objects']:
        p = Placement.model_validate(item['placement'])
        a = doc['assets'][f'{p.category}/{p.model_id}']
        x, y, z = a['bbox_size']
        if (p.name in names or p.aabb_size != [y, x, z]
                or item['og_config'] != object_config(p, a, native_poses.get(p.name))):
            raise ValueError('BEHAVIOR_ASSEMBLY_TRANSFORM_MISMATCH')
        names.add(p.name)
    if not names:
        raise ValueError('BEHAVIOR_ASSEMBLY_EMPTY')
    return doc


def assembly_env_config(path: Path, scene_template: dict, scene_model: str) -> dict:
    doc = load_manifest(path)
    if scene_model != doc['base_scene']:
        raise ValueError('BEHAVIOR_BASE_SCENE_MISMATCH')
    if scene_model == '__empty__':
        # Explicit integration fixture, not a complete household. OG owns the
        # ground plane; every instantiated object is a real DatasetObject.
        scene = dict(type='Scene', use_floor_plane=True, use_skybox=True, include_robots=False)
    else:
        # Preserve the complete native house, including non-search rooms.
        scene = dict(scene_template, scene_model=scene_model)
        if doc['version'] == 2:
            scene['scene_file'] = doc['living_context']['native']['scene_json']
    return dict(scene=scene, objects=[item['og_config'] for item in doc['objects']])
