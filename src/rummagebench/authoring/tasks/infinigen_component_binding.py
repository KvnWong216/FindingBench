"""Split genuine upstream cabinet faces for existing interactive-asset binding.

The output is a source-derived assembly input, not an accepted task or physics
certificate. The complete drawer distribution is retained even if one asset
or OPEN call would reveal several drawers at once.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path


def face_signature(obj):
    """World-space face/UV/material multiset, invariant to Blender separation."""
    mesh = obj.data
    faces = []
    for polygon in mesh.polygons:
        loops = []
        for loop_index in polygon.loop_indices:
            vertex = mesh.loops[loop_index].vertex_index
            xyz = list(obj.matrix_world @ mesh.vertices[vertex].co)
            uvs = [(layer.name, list(layer.data[loop_index].uv)) for layer in mesh.uv_layers]
            loops.append([np.round(xyz, 6).tolist(), uvs])
        # Same winding, invariant only to starting vertex, not reversal.
        rotations = [loops[i:] + loops[:i] for i in range(len(loops))]
        material = mesh.materials[polygon.material_index].name if mesh.materials else None
        faces.append(sha256_bytes(canonical_json([min(rotations), material])))
    return Counter(faces)


def split_real_cells(obj, source_census):
    """Use Blender's separation operator; never reconstruct source geometry."""
    import bpy
    from rummagebench.authoring.tasks.infinigen_components import CELL_ATTRIBUTE, mesh_geometry_digest

    if obj.modifiers or mesh_geometry_digest(obj.data) != source_census["geometry_sha256"]:
        raise ValueError("COMPONENT_SPLIT_SOURCE_GEOMETRY_CHANGED_OR_UNAPPLIED")
    before = face_signature(obj)
    original_matrix = obj.matrix_world.copy()
    cells = source_census["cells"]
    ids = [c["cell_id"] for c in cells]
    attr = obj.data.attributes.get(CELL_ATTRIBUTE)
    if attr is None or len(ids) != len(set(ids)):
        raise ValueError("COMPONENT_SPLIT_PROVENANCE_INVALID")
    if set(d.value for d in attr.data) - {0, *ids}:
        raise ValueError("COMPONENT_SPLIT_INCOMPLETE_CENSUS")
    parts = []
    for cell in cells:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        attr = obj.data.attributes[CELL_ATTRIBUTE]
        values = np.empty(len(obj.data.polygons), dtype=np.int32)
        attr.data.foreach_get('value', values)
        selected = values == cell['cell_id']
        selected_count = int(selected.sum())
        if selected_count != len(cell["face_indices"]) or not selected_count:
            raise ValueError("COMPONENT_SPLIT_FACE_COUNT_MISMATCH")
        # Polygon selection is authoritative; clear stale vertex/edge state.
        for v in obj.data.vertices: v.select = False
        for e in obj.data.edges: e.select = False
        # Selection changes can reallocate Blender attribute storage. Read
        # provenance first, then write the entire selection in one operation.
        obj.data.polygons.foreach_set('select', selected)
        known = set(bpy.data.objects.keys())
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        added = [o for o in bpy.data.objects if o.name not in known]
        if len(added) != 1 or len(added[0].data.polygons) != selected_count:
            raise ValueError("COMPONENT_SPLIT_OPERATOR_COVERAGE_MISMATCH")
        part = added[0]
        part.name = cell["component_id"]
        if part.name != cell["component_id"] or not np.array_equal(part.matrix_world, original_matrix):
            raise ValueError("COMPONENT_SPLIT_NAME_OR_POSE_CONFLICT")
        xyz = np.asarray([v.co for v in part.data.vertices], float)
        parts.append(dict(cell, blender_name=part.name,
            local_aabb=[xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
            matrix_world=[list(row) for row in part.matrix_world]))
    after = face_signature(obj)
    for part in parts:
        after += face_signature(bpy.data.objects[part["blender_name"]])
    if after != before:
        raise ValueError("COMPONENT_SPLIT_CHANGED_WORLD_GEOMETRY_UV_OR_MATERIALS")
    return {"source_id": source_census["source_id"], "cells": parts,
            "background_blender_name": obj.name,
            "world_faces_sha256": sha256_bytes(canonical_json(sorted(before.items()))),
            "geometry_uv_materials_preserved": True}


def expand_search_distribution(spec, census):
    """Derive an entirely new uniform prior over ALL actual closed cavities."""
    if census.get("status") != "source_complete_pending_binding" or census.get("failures"):
        raise ValueError("FULL_KITCHEN_COMPONENT_CENSUS_REQUIRED")
    roots = census["source_roots"]
    if (not roots or len(roots) != len(set(roots)) or len(census['sources']) != len(roots)
            or set(roots) != {s["source_id"] for s in census["sources"]}):
        raise ValueError("COMPONENT_CENSUS_SOURCE_ROOT_COVERAGE_MISMATCH")
    leaves = [dict(l) for s in census["sources"] for l in s["locations"]]
    if (not leaves or len(leaves) != census["search_location_count"]
            or len({l["location_id"] for l in leaves}) != len(leaves)):
        raise ValueError("COMPONENT_CENSUS_LOCATION_COVERAGE_MISMATCH")
    for source in census["sources"]:
        faces = [i for c in source["cells"] for i in c["face_indices"]]
        if (len(faces) != len(set(faces)) or any(not isinstance(i,int) or not 0<=i<source['face_count'] for i in faces)
                or not 0<=source['background_face_count']<=source['face_count']
                or len(faces) + source["background_face_count"] != source["face_count"]):
            raise ValueError("COMPONENT_CENSUS_FACE_PARTITION_INVALID")
        by_cell = {c["component_id"]: c for c in source["cells"]}
        expected = []
        for cell in source["cells"]:
            if cell['kind'] == 'door': expected.append(cell['component_id'])
            elif cell['kind'] == 'drawer':
                expected.extend(cell['component_id'] + f"__drawer_{a['part_id']}" for a in cell['attachments'])
            elif cell['kind'] != 'none': raise ValueError('COMPONENT_CENSUS_UNKNOWN_KIND')
        actual = source['locations']
        if (set(expected) != {l['location_id'] for l in actual}
                or any(l['component_id'] not in by_cell or l['source_id'] != source['source_id'] for l in actual)):
            raise ValueError('COMPONENT_CENSUS_DROPPED_CABINET_OR_DRAWER')
        # The user's whole-kitchen scope also includes native open cells.
        # They may fail initial invisibility; retain them until that is
        # measured, rather than shrinking and reusing the closed-only prior.
        for cell in source['cells']:
            if cell['kind'] == 'none':
                leaves.append({'location_id':cell['component_id'], 'component_id':cell['component_id'],
                    'source_id':source['source_id'], 'slot_type':'cabinet_interior',
                    'requires_open':False, 'coupled_source_component':False,
                    'initial_visibility_status':'pending_natural_occlusion_measurement'})
    leaves.sort(key=lambda l: l['location_id'])
    n = len(leaves)
    for leaf in leaves:
        leaf['prior'] = 1 / n
        leaf.setdefault('requires_open', True)
    groups = {}
    for leaf in leaves: groups.setdefault(leaf['component_id'], []).append(leaf)
    group_order = sorted(groups, key=lambda key: (
        (2 if groups[key][0]['requires_open'] else 1) / (len(groups[key])/n), key))
    cumulative = expected = 0.
    for key in group_order:
        cumulative += 2 if groups[key][0]['requires_open'] else 1
        expected += len(groups[key])/n * (cumulative + 2)
    identity = sha256_bytes(canonical_json({'supersedes_family': spec.family_id, 'census': census}))
    # This calculation only diagnoses feasibility of the old proposal model.
    # It does not label real search difficulty or assume independent drawers.
    return {'version': 1, 'status': 'candidate_pending_interactive_binding',
        'family_id': 'family_' + identity[:20], 'supersedes_family_id': spec.family_id,
        'source_layout_group': spec.split_assignment.source_layout_group if spec.split_assignment else None,
        'split_assignment': spec.split_assignment.model_dump(mode='json') if spec.split_assignment else None,
        'target': spec.target.model_dump(mode='json'), 'appearance_request': spec.appearance,
        'robot_id': spec.robot_id,
        'public_task': {'instruction': f"Find the {spec.target.category.replace('_', ' ')} in any kitchen cabinet or drawer and pick it up.",
                        'max_planning_steps': spec.max_planning_steps},
        'locations': leaves, 'target_prior': 'uniform_over_complete_source_cavities',
        'candidate_count': n, 'candidate_distribution_changed': True,
        'open_candidate_count': sum(not l['requires_open'] for l in leaves),
        'source_census_sha256': sha256_bytes(canonical_json(census)),
        'declared_independent_OPEN_OBSERVE_model_only': {
            'expected_operations': n + 3, 'worst_case_operations': 2*n + 2,
            'fits_public_horizon_without_movement': 2*n + 2 <= spec.max_planning_steps,
            'applicability': 'pending; multiple drawers can share one asset or reveal',
            'real_scene_difficulty_certified': False},
        'declared_component_reveal_model_only': {
            'scope':'zero_travel_per_component_perfect_reveal; OPEN_OBSERVE_closed_OBSERVE_open',
            'order_on_negative_results':group_order,
            'expected_operations':expected, 'worst_case_operations':cumulative+2,
            'applicability':'pending real reveal sets; no physical difficulty or bound claim',
            'real_scene_difficulty_certified':False},
        'coupled_drawer_locations': [l['location_id'] for l in leaves if l['coupled_source_component']],
        'final_difficulty': None, 'compiled_families': 0, 'evaluation_ready_episodes': 0,
        'pending': ['complete_real_asset_bindings', 'per_drawer_skill_and_reveal_protocol',
                    'open_cell_visibility_and_natural_occlusion',
                    'physical_background_freeze', 'full_family_reference_search_measurement']}


def run_split_worker(input_folder: Path, output: Path):
    """CPU bpy worker; only accepts a complete, hash-checked source census."""
    import bpy
    from rummagebench.authoring.level1.snapshot import atomic_write_json
    raw = json.loads((input_folder / 'solve_state.json').read_text())
    geometry = json.loads((input_folder / 'findingbench_geometry.json').read_text())
    if geometry['solve_state_sha256'] != sha256_path(input_folder / 'solve_state.json'):
        raise ValueError('COMPONENT_SPLIT_SOLVE_STATE_HASH_MISMATCH')
    census = geometry.get('cabinet_component_census')
    if not census or census['status'] != 'source_complete_pending_binding':
        raise ValueError('FULL_KITCHEN_COMPONENT_CENSUS_REQUIRED')
    output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(input_folder / 'scene.blend'))
    split = []
    for source in census['sources']:
        obj = bpy.data.objects.get(source['blender_name'])
        if obj is None: raise ValueError('COMPONENT_SOURCE_MESH_MISSING')
        split.append(split_real_cells(obj, source))
    # Keep the original solve_state immutable. The derived relation graph
    # records exact mesh components, not invented solver placements.
    for source in split:
        parent = raw['objs'][source['source_id']]
        parent['tags'] = [t for t in parent.get('tags', []) if t not in ('Semantics(storage)', 'Semantics.Storage')]
        parent['findingbench_decomposed_components'] = [c['component_id'] for c in source['cells']]
        for cell in source['cells']:
            if cell['component_id'] in raw['objs']:
                raise ValueError('COMPONENT_SOURCE_ID_CONFLICT')
            tags = ['Semantics(object)', 'Semantics(furniture)', 'FromGenerator(KitchenCabinetFactory)']
            if cell['kind'] != 'none': tags.append('Semantics(storage)')
            raw['objs'][cell['component_id']] = {'obj':cell['blender_name'], 'active':True,
                'generator':'actual_KitchenCabinetFactory_component', 'tags':tags,
                'relations':[{'target_name':source['source_id'],
                              'relation':{'relation_type':'SourceComponentOf'}}],
                'findingbench_component':{'parent':source['source_id'], 'kind':cell['kind'],
                    'cell_id':cell['cell_id'], 'factory_seed':cell['factory_seed'],
                    'attachment_count':len(cell['attachments'])}}
    atomic_write_json(output / 'solve_state.json', raw)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'scene.blend'))
    atomic_write_json(output / 'components.json', {'version': 1, 'sources': split, 'census': census,
        'status': 'split_source_geometry_pending_interactive_binding',
        'source_hashes': {str((input_folder / n).resolve()): sha256_path(input_folder / n)
                          for n in ('scene.blend', 'solve_state.json', 'findingbench_geometry.json')},
        'scene_sha256': sha256_path(output / 'scene.blend'),
        'derived_solve_state_sha256':sha256_path(output / 'solve_state.json'), 'evaluation_ready_episodes': 0})


def propose_component_assets(component_doc, approved_slots, assets_root, profile='source-kind'):
    """Use the same exact easy_v1 models; require approved slot type as well.

    Fits are proposals. Drawer counts, fillable links and shared OPEN semantics
    must be checked on the actual imported asset, never inferred from size.
    """
    from rummagebench.authoring.tasks.slots import SceneSlots
    from rummagebench.sim.omnigibson.infinigen_adapter import replacement_root_position
    if profile not in ('source-kind','single-joint-containers'):
        raise ValueError('UNSUPPORTED_COMPONENT_BINDING_PROFILE')
    allowed = {}
    for path in approved_slots:
        for slot in SceneSlots.load(path).slots:
            if slot.room_type == 'kitchen' and slot.requires_open and slot.articulated:
                allowed.setdefault(slot.slot_type, set()).add((slot.parent_category, slot.parent_model))
    if profile=='single-joint-containers':
        all_models=allowed.get('cabinet_interior',set())|allowed.get('drawer',set())
        allowed={'cabinet_interior':all_models,'drawer':all_models}
    pool, rejected, hashes = {}, [], {}
    for kind in ('cabinet_interior', 'drawer'):
        pool[kind] = []
        for category, model in sorted(allowed.get(kind, set())):
            base = Path(assets_root) / 'objects' / category / model
            meta_path = base / 'misc/metadata.json'
            usd = base / 'usd' / f'{model}.usd'
            if not usd.is_file(): usd = usd.with_name(f'{model}.encrypted.usd')
            try:
                meta = json.loads(meta_path.read_text())
                native = np.asarray(meta['bbox_size'], float)
                if (native.shape != (3,) or not np.isfinite(native).all() or min(native) <= 0
                        or not usd.is_file() or not any('openable' in t for t in meta.get('link_tags', {}).values())):
                    raise ValueError('REAL_APPROVED_OPENABLE_ASSET_REQUIRED')
                replacement_root_position([0.,0.,0.], [1.,1.,1.], [0.,0.,0.,1.], meta)
                if profile=='single-joint-containers' and sum('openable' in tags for tags in meta.get('link_tags',{}).values())!=1:
                    continue  # Proposal filter only; live joints and sweeps must still pass.
                pool[kind].append((category, model, native, meta_path, usd))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                rejected.append({'category':category, 'model':model, 'slot_type':kind, 'reason':str(exc)})
    bindings, failures, open_cells = [], [], []
    for source in component_doc['sources']:
        for cell in source['cells']:
            if cell['kind'] == 'none':
                open_cells.append({'component_id':cell['component_id'],
                    'status':'pending_visibility_and_occlusion_scope', 'reason':'native open cabinet cell retained'})
                continue
            kind = 'drawer' if cell['kind'] == 'drawer' else 'cabinet_interior'
            box = np.asarray(cell['local_aabb'], float)
            if box.shape != (2,3) or not np.isfinite(box).all() or np.any(box[1] <= box[0]):
                raise ValueError('COMPONENT_REAL_LOCAL_BOUNDS_INVALID')
            matrix = np.asarray(cell['matrix_world'], float)
            if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0,0,0,1]):
                raise ValueError('COMPONENT_REAL_WORLD_TRANSFORM_INVALID')
            world_scale = np.linalg.norm(matrix[:3,:3], axis=0)
            if min(world_scale) <= 0: raise ValueError('COMPONENT_WORLD_SCALE_INVALID')
            rotation = matrix[:3,:3] / world_scale
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation),1,atol=1e-6):
                raise ValueError('COMPONENT_SHEAR_OR_REFLECTION_UNSUPPORTED')
            dims = (box[1] - box[0]) * world_scale
            choices=[]
            prefer_upper=float(box[0,2])>1.1
            for category, model, native, meta_path, usd in pool[kind]:
                scale=dims/native
                if np.all((scale >= .5) & (scale <= 2.)):
                    category_cost=(int((category=='top_cabinet')!=prefer_upper)
                                   if profile=='single-joint-containers' else 0)
                    choices.append((category_cost+float(np.abs(np.log(scale)).sum()),category,model,scale,meta_path,usd))
            if not choices:
                failures.append({'component_id':cell['component_id'], 'slot_type':kind,
                                 'dimensions_m':dims.tolist(), 'reason':'NO_APPROVED_INTERACTIVE_COMPONENT_MODEL_FITS'})
                continue
            _, category, model, scale, meta_path, usd = min(choices, key=lambda c:c[:3])
            from scipy.spatial.transform import Rotation
            quaternion = Rotation.from_matrix(rotation).as_quat().tolist()
            center = (matrix @ np.r_[(box[0]+box[1])/2,1])[:3].tolist()
            meta = json.loads(meta_path.read_text())
            bindings.append({'component_id':cell['component_id'], 'source_id':source['source_id'],
                             'category':category, 'model':model, 'scale':scale.tolist(),
                             'source_bbox_center_world':center, 'orientation':quaternion,
                             'root_position_proposal':replacement_root_position(center,scale,quaternion,meta),
                             'slot_type':kind, 'native_attachment_count':len(cell['attachments']),
                             'status':'pending_real_cavity_and_skill_binding'})
            for path in (meta_path,usd): hashes[str(path.resolve())]=sha256_path(path)
    return {'version':1,'status':'rejected' if failures else 'candidate_pending_physics',
            'binding_profile':profile,'source_topology_preservation_claimed':False,
            'bindings':bindings,'failures':failures,'rejected_assets':rejected,'asset_hashes':hashes,
            'approved_slot_hashes':{str(Path(p).resolve()):sha256_path(p) for p in approved_slots},
            'open_cells':open_cells,'target_conditioned':False,
            'candidate_pruning_allowed':False,'physical_acceptance':'pending','evaluation_ready_episodes':0}


def verify_component_preparation(path):
    """Check protected real split inputs before GPU export; no acceptance grant."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if (report.get('stage') != 'prepare_infinigen_components' or report.get('returncode') != 0
            or report.get('status') != 'source_components_prepared_pending_binding'
            or report.get('evaluation_ready_episodes') != 0):
        raise ValueError('COMPLETE_REAL_COMPONENT_PREPARATION_REQUIRED')
    folder = path.parent / 'geometry'
    required = {str((folder / n).resolve()) for n in ('scene.blend','solve_state.json','components.json')}
    if set(report.get('artifacts',{})) != required:
        raise ValueError('COMPONENT_PREPARATION_ARTIFACT_COVERAGE_INVALID')
    for ref,digest in report['artifacts'].items():
        if sha256_path(ref) != digest: raise ValueError('COMPONENT_PREPARATION_ARTIFACT_CHANGED')
    doc = json.loads((folder / 'components.json').read_text())
    for ref,digest in doc['source_hashes'].items():
        if sha256_path(ref) != digest: raise ValueError('COMPONENT_PREPARATION_SOURCE_CHANGED')
    if (doc['scene_sha256'] != sha256_path(folder / 'scene.blend')
            or doc['derived_solve_state_sha256'] != sha256_path(folder / 'solve_state.json')
            or report['family_proposal']['source_census_sha256'] != sha256_bytes(canonical_json(doc['census']))
            or any(s.get('geometry_uv_materials_preserved') is not True for s in doc['sources'])
            or {s['source_id'] for s in doc['sources']} != set(doc['census']['source_roots'])):
        raise ValueError('COMPONENT_PREPARATION_SOURCE_COVERAGE_OR_HASH_INVALID')
    proposals = report['asset_proposals']
    for field in ('asset_hashes','approved_slot_hashes'):
        for ref,digest in proposals[field].items():
            if sha256_path(ref) != digest: raise ValueError('COMPONENT_APPROVED_ASSET_INPUT_CHANGED')
    verify_component_asset_proposals(doc,proposals)
    return folder,report


def verify_component_asset_proposals(doc,proposals):
    # Asset hashes alone do not protect the chosen mapping. Recompute the
    # complete deterministic proposal from the exact split geometry and
    # approved inputs so a missing/replaced body cannot pass import checks.
    roots={str(Path(ref).parents[4]) for ref in proposals['asset_hashes']
           if Path(ref).parent.name in ('misc','usd')}
    if len(roots)!=1:
        raise ValueError('COMPONENT_ASSET_ROOT_UNRESOLVED')
    expected_assets=propose_component_assets(doc,list(proposals['approved_slot_hashes']),
        next(iter(roots)),proposals.get('binding_profile','source-kind'))
    # Older protected preparations predate the explicit profile labels.
    actual_assets=dict(proposals)
    actual_assets.setdefault('binding_profile','source-kind')
    actual_assets.setdefault('source_topology_preservation_claimed',False)
    if actual_assets!=expected_assets:
        raise ValueError('COMPONENT_ASSET_MAPPING_CHANGED_OR_INCOMPLETE')


@dataclass(frozen=True)
class _AssetLocation:
    location_id: str


@dataclass(frozen=True)
class _ComponentAssemblyInputs:
    """Adapt real asset bodies to the existing builder; contains no target prior."""
    source_spec: object
    family_id: str
    locations: tuple[_AssetLocation, ...]

    def __getattr__(self, name):
        return getattr(self.source_spec, name)


def prepare_component_assembly(spec, preparation, export_receipt, assets_root, output):
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    folder,report = verify_component_preparation(preparation)
    proposal = report['family_proposal']
    if spec.family_id != proposal['supersedes_family_id']:
        raise ValueError('COMPONENT_ASSEMBLY_SOURCE_SPEC_MISMATCH')
    assets = report['asset_proposals']
    if assets['status'] != 'candidate_pending_physics' or assets['failures']:
        raise ValueError('ALL_CLOSED_COMPONENT_ASSETS_REQUIRED_NO_PRUNING')
    exported = json.loads(Path(export_receipt).read_text())
    if (exported['status'] != 'exported_pending_simulation' or exported['gpu_executed'] is not True
            or exported['physical_gpu'] not in (4,5)
            or exported['input_blend_sha256'] != sha256_path(folder / 'scene.blend')
            or exported['solve_state_sha256'] != sha256_path(folder / 'solve_state.json')):
        raise ValueError('COMPONENT_ASSEMBLY_REQUIRES_MATCHING_REAL_EXPORT')
    source = Path(exported['scene'])
    if sha256_path(source) != exported['inspection']['scene_sha256']:
        raise ValueError('COMPONENT_EXPORTED_SCENE_CHANGED')
    bindings=[{'location_id':b['component_id'], 'source_id':b['component_id'],
               'category':b['category'], 'model':b['model']} for b in assets['bindings']]
    context = _ComponentAssemblyInputs(spec,proposal['family_id'],
                                      tuple(_AssetLocation(b['location_id']) for b in bindings))
    return prepare_assembly(context, source, folder / 'solve_state.json', bindings, assets_root, output,
        component_context={'preparation':str(Path(preparation).resolve()),
            'preparation_sha256':sha256_path(preparation), 'scope':'all_real_asset_bodies_preflight_only',
            'complete_search_distribution':proposal, 'complete_leaf_binding':'pending',
            'source_spec_sha256':sha256_bytes(spec.canonical_bytes())})


def inspect_loaded_component_topology(backend, manifest):
    """Real imported joints/fillable links, never inferred from room categories."""
    from dataclasses import asdict
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    from rummagebench.authoring.tasks.native_occupancy import checked_box, plain
    from omnigibson.object_states.contains import m as contains_macros
    context=manifest.get('component_context')
    if not context:raise ValueError('PROTECTED_COMPONENT_CONTEXT_REQUIRED')
    folder,prepared=verify_component_preparation(context['preparation'])
    doc=json.loads((folder/'components.json').read_text())
    cells={c['component_id']:c for s in doc['sources'] for c in s['cells']}
    rows=[]
    for binding in manifest['candidate_bindings']:
        obj=resolve_object(backend._env.scene,binding['name'])
        art=backend.articulation_info(obj.name)
        if art is None:raise ValueError('LOADED_COMPONENT_ARTICULATION_UNAVAILABLE')
        volumes={name:link for name,link in obj.links.items()
                 if link.is_meta_link and link.meta_link_type in contains_macros.CONTAINER_META_LINK_TYPES}
        fillable=sorted(volumes)
        regions=[{'link':name,'meta_link_type':volumes[name].meta_link_type,
                  'visual_aabb':checked_box(plain(volumes[name].visual_aabb)),
                  'geometry_use':'proposal_envelope_only; native volume membership required for placement'}
                 for name in fillable]
        cell=cells[binding['source_id']]
        count=len(cell['attachments']) if cell['kind']=='drawer' else 1
        movable=[j for j in art.joints if j.joint_type in ('prismatic','revolute')]
        rows.append({'entity':obj.name,'source_component':binding['source_id'],'source_kind':cell['kind'],
            'source_cavity_count':count,'loaded_fillable_links':fillable,'loaded_regions':regions,
            'loaded_joints':[asdict(j) for j in art.joints],
            'native_topology_matches_source_kind':bool(movable) and all(j.joint_type==('prismatic' if cell['kind']=='drawer' else 'revolute')
                                                       for j in movable),
            'native_cavity_cardinality_matches_source':len(fillable)==count,
            'native_world_aabb':backend.entity_aabb(obj.name),
            'expected_source_bbox_center':binding['source_bbox_center_world'],
            'handle_link':art.handle_link,'leaf_binding':'pending_actual_cavity_and_reveal_measurement'})
    return {'scope':'loaded_component_topology_diagnostic_only','rows':rows,
            'complete_source_search_locations':prepared['family_proposal']['candidate_count'],
            'loaded_closed_fillable_region_count':sum(len(r['loaded_fillable_links']) for r in rows),
            'unbound_native_open_source_components':[c['component_id'] for c in cells.values() if c['kind']=='none'],
            'physical_distribution_certified':False,'evaluation_ready_episodes':0}


def propose_loaded_search_distribution(manifest, topology):
    """Explicitly supersede the source prior after real asset topology changes.

    These are observed annotated regions, not proof that all physical cavities
    are annotated. Open source cells and unresolved bindings remain included.
    """
    original=manifest['component_context']['complete_search_distribution']
    locations=[]
    expected={b['name'] for b in manifest['candidate_bindings']}
    if len(topology['rows'])!=len(expected) or {r['entity'] for r in topology['rows']}!=expected:
        raise ValueError('LOADED_NATIVE_SEARCH_BODY_COVERAGE_INCOMPLETE')
    for row in topology['rows']:
        if not row['loaded_fillable_links']:
            raise ValueError('NATIVE_COMPONENT_CAVITIES_UNRESOLVED_NO_PRUNING')
        joints=row['loaded_joints']
        if len(row['loaded_fillable_links'])!=len(set(row['loaded_fillable_links'])):
            raise ValueError('NATIVE_FILLABLE_REGION_DUPLICATE')
        regions={v['link']:v for v in row.get('loaded_regions',[])}
        if set(regions)!=set(row['loaded_fillable_links']):
            raise ValueError('NATIVE_FILLABLE_GEOMETRY_COVERAGE_INCOMPLETE')
        for volume in row['loaded_fillable_links']:
            parents={j['parent_link'] for j in joints if j['child_link']==volume and j['joint_type']=='fixed'}
            if len(parents)!=1:
                raise ValueError('NATIVE_FILLABLE_REGION_PARENT_UNRESOLVED')
            parent=next(iter(parents))
            moving=[j for j in joints if j['child_link']==parent and j['joint_type'] in ('prismatic','revolute')]
            kind='drawer' if len(moving)==1 and moving[0]['joint_type']=='prismatic' else 'cabinet_interior'
            locations.append({'location_id':row['entity']+'__'+volume,'entity':row['entity'],
                'source_component':row['source_component'],'fillable_link':volume,'placement_parent_link':parent,
                'slot_type':kind,'requires_open':'pending_articulation_reveal_measurement',
                'native_region':regions[volume],'region_source':'actual_loaded_fillable_link',
                'cavity_occupancy_and_target_fit':'pending','joint_reveal_set':'pending_measurement'})
    # Native open cells are never removed to make the closed-only pipeline pass.
    for source in original['locations']:
        if not source['requires_open']:
            locations.append({'location_id':source['location_id'],'source_component':source['component_id'],
                'slot_type':source['slot_type'],'requires_open':False,'region_source':'actual_source_open_cell',
                'cavity_occupancy_and_target_fit':'pending_geometry_binding','joint_reveal_set':'pending_measurement'})
    locations.sort(key=lambda l:l['location_id'])
    if not locations or len({l['location_id'] for l in locations})!=len(locations):
        raise ValueError('NATIVE_SEARCH_LOCATIONS_DUPLICATE_OR_EMPTY')
    for location in locations:location['prior']=1/len(locations)
    inputs={'source_family':original['family_id'],'assembly_sha256':manifest['manifest_sha256'],
            'locations':locations}
    return {'version':1,'status':'candidate_pending_physical_distribution_validation',
        'scope':'actual_annotated_native_regions_and_retained_source_open_cells',
        'family_id':'family_'+sha256_bytes(canonical_json(inputs))[:20],
        'supersedes_family_id':original['family_id'],'supersedes_source_candidate_count':original['candidate_count'],
        'candidate_count':len(locations),'locations':locations,'prior':'uniform_over_entire_new_candidate_set',
        'candidate_distribution_changed':True,'source_layout_group':original['source_layout_group'],
        'split_assignment':original['split_assignment'],'target':original['target'],
        'public_task':original['public_task'],'final_difficulty':None,
        'physical_cavity_completeness_certified':False,'evaluation_ready_episodes':0,
        'pending':['unannotated_cavity_audit','physical_background_stability','per_region_binding',
                   'occupancy_and_fit','complete_family_freeze','reveal_sets_and_measured_search_cost']}


def native_family_document(source, manifest, lineage):
    """Deterministic, versioned derivative of the protected source proposal."""
    doc=source.model_dump(mode='json')
    proposal=manifest['component_context']['loaded_search_distribution']
    from rummagebench.authoring.tasks.household import native_direct_tier
    from rummagebench.authoring.tasks.tiers import tier_spec_path
    contract=native_direct_tier(proposal['candidate_count'],source.max_planning_steps)
    if (manifest['component_context']['source_spec_sha256']!=sha256_bytes(source.canonical_bytes())
            or proposal['source_layout_group']!=source.split_assignment.source_layout_group
            or proposal['split_assignment']!=source.split_assignment.model_dump(mode='json')):
        raise ValueError('NATIVE_FAMILY_SOURCE_OR_SPLIT_MISMATCH')
    doc.update(generator_version='household-infinigen-native-3',family_id=proposal['family_id'],
        locations=[{'location_id':b['location_id'],'slot_type':b['slot_type'],'relation':'inside',
                    'prior':b['prior'],'initial_open':False,'source_entity':b['source_component'],
                    'assembly_entity':b['name']} for b in manifest['native_region_bindings']],
        target_location=manifest['native_region_bindings'][0]['location_id'],
        generation_constraints={'source_generation_constraints':source.generation_constraints,
            'native_region_lineage':lineage,'candidate_distribution':'complete_observed_native_regions'},
        proxy_metrics={'scope':'native_region_candidate_only','physical_feasibility_verified':False},
        proxy_difficulty={},measured_metrics={},final_difficulty=None,status='candidate',reasons=[],
        input_hashes={**source.input_hashes,**{'native_lineage_'+k:sha256_path(v)
                      for k,v in lineage['paths'].items()},
                      'native_execution_config':sha256_path(tier_spec_path('household_direct_v1')),
                      'native_execution_contract':sha256_bytes(canonical_json(contract.model_dump(mode='json')))})
    return doc


def native_source_inputs(lineage,repo):
    """Verify original generation against the original solve state, never natively changed counts."""
    from rummagebench.authoring.tasks.household import HouseholdTaskSpec,validate_spec_assets
    from rummagebench.authoring.tasks.household_dataset import verify_generation_admission
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
    if (set(lineage)!={'version','paths','hashes'} or lineage['version']!=1
            or set(lineage['paths'])!={'source_spec','source_policy','native_assembly','generation_receipt'}
            or set(lineage['hashes'])!=set(lineage['paths'])):
        raise ValueError('NATIVE_FAMILY_LINEAGE_SCHEMA_INVALID')
    for key,ref in lineage['paths'].items():
        if not Path(ref).is_absolute() or sha256_path(ref)!=lineage['hashes'][key]:
            raise ValueError('NATIVE_FAMILY_LINEAGE_INPUT_CHANGED')
    paths={k:Path(v) for k,v in lineage['paths'].items()}
    source=HouseholdTaskSpec.model_validate_json(paths['source_spec'].read_text())
    if source.generator_version=='household-infinigen-native-3':
        raise ValueError('NATIVE_FAMILY_RECURSIVE_SOURCE_FORBIDDEN')
    validate_spec_assets(source,repo,paths['source_policy'])
    manifest=load_assembly_manifest(paths['native_assembly'])
    if not manifest.get('native_region_catalog'):
        raise ValueError('NATIVE_FAMILY_COMPLETE_REGION_CATALOG_REQUIRED')
    folder,prepared=verify_component_preparation(manifest['component_context']['preparation'])
    if prepared['source_receipt_sha256']!=sha256_path(paths['generation_receipt']):
        raise ValueError('NATIVE_FAMILY_SOURCE_GENERATION_RECEIPT_MISMATCH')
    components=json.loads((folder/'components.json').read_text())
    solves=[Path(p) for p in components['source_hashes'] if Path(p).name=='solve_state.json']
    if len(solves)!=1:raise ValueError('NATIVE_FAMILY_ORIGINAL_SOLVE_REQUIRED')
    admission=verify_generation_admission(source,paths['generation_receipt'],solves[0])
    return source,manifest,{'version':2,'source_generation':admission,
        'source_spec_sha256':lineage['hashes']['source_spec'],
        'native_assembly_sha256':lineage['hashes']['native_assembly'],
        'scope':'original_generation_proxy_admitted; complete_native_distribution_pending_physical_validation'}


def derive_native_family(source_spec,source_policy,assembly,generation_receipt,repo):
    from rummagebench.authoring.tasks.household import HouseholdTaskSpec,family_members
    paths={k:str(Path(v).resolve()) for k,v in dict(source_spec=source_spec,source_policy=source_policy,
        native_assembly=assembly,generation_receipt=generation_receipt).items()}
    lineage={'version':1,'paths':paths,'hashes':{k:sha256_path(v) for k,v in paths.items()}}
    source,manifest,_=native_source_inputs(lineage,repo)
    return family_members(HouseholdTaskSpec.model_validate(native_family_document(source,manifest,lineage)))


def validate_native_family_spec(spec,repo):
    from rummagebench.authoring.tasks.household import HouseholdTaskSpec,family_members
    from rummagebench.authoring.tasks.priors import PlacementPriors
    import yaml
    lineage=spec.generation_constraints['native_region_lineage']
    source,manifest,_=native_source_inputs(lineage,repo)
    base=HouseholdTaskSpec.model_validate(native_family_document(source,manifest,lineage))
    expected={s.target_location:s for s in family_members(base)}
    if expected.get(spec.target_location)!=spec:
        raise ValueError('NATIVE_FAMILY_SPEC_DIFFERS_FROM_COMPLETE_SOURCE_DERIVATION')
    policy=yaml.safe_load(Path(lineage['paths']['source_policy']).read_text())
    priors=PlacementPriors.load(repo/policy['placement_priors'])
    if any(not priors.is_approved(spec.target.category,s.slot_type,s.room_type) for s in spec.locations):
        raise ValueError('NATIVE_FAMILY_TARGET_PRIOR_UNAPPROVED')


def verify_native_region_catalog(doc):
    """Recompute all region bindings from the immutable real import evidence."""
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
    from rummagebench.sim.omnigibson.observation import head_rgb_metadata_matches
    catalog=doc['native_region_catalog']
    paths=catalog['inputs']
    for ref,digest in paths.items():
        if not Path(ref).is_file() or sha256_path(ref)!=digest:
            raise ValueError('NATIVE_REGION_CATALOG_INPUT_CHANGED')
    source_path=Path(catalog['source_assembly'])
    if str(source_path) not in paths or source_path==Path(catalog['preflight']):
        raise ValueError('NATIVE_REGION_CATALOG_SOURCE_LEDGER_INVALID')
    source=load_assembly_manifest(source_path)
    if source.get('native_region_catalog'):
        raise ValueError('NATIVE_REGION_CATALOG_RECURSIVE_SOURCE_FORBIDDEN')
    root=Path(catalog['preflight']).parent
    required={str(source_path),str(root/'preflight.json'),str(root/'loaded_component_topology.json'),
              str(root/'loaded_native_search_proposal.json')}
    if set(paths)!=required:
        raise ValueError('NATIVE_REGION_CATALOG_LEDGER_INCOMPLETE')
    evidence=json.loads((root/'preflight.json').read_text())
    if (evidence.get('status')!='pending_simulation_acceptance' or evidence.get('failures')
            or evidence.get('error') or evidence.get('assembly_manifest_sha256')!=sha256_path(source_path)
            or evidence.get('native_restore_repeats')!=2
            or not head_rgb_metadata_matches(evidence.get('camera'))
            or not {'native_inventory','native_restore_twice','camera_rgb_shape','loaded_articulation_query',
                    'spawn_clearance'}<=set(evidence.get('completed_checks',[]))):
        raise ValueError('NATIVE_REGION_CATALOG_REQUIRES_COMPLETE_REAL_IMPORT_PREFLIGHT')
    for name in ('loaded_component_topology.json','loaded_native_search_proposal.json'):
        if evidence.get('native_region_artifacts',{}).get(name)!=sha256_path(root/name):
            raise ValueError('NATIVE_REGION_CATALOG_TOPOLOGY_RECEIPT_MISMATCH')
    topology=json.loads((root/'loaded_component_topology.json').read_text())
    proposal=propose_loaded_search_distribution(source,topology)
    if proposal!=json.loads((root/'loaded_native_search_proposal.json').read_text()):
        raise ValueError('NATIVE_REGION_CATALOG_DISTRIBUTION_MISMATCH')
    if any(l['region_source']!='actual_loaded_fillable_link' for l in proposal['locations']):
        raise ValueError('SOURCE_OPEN_REGIONS_REQUIRE_EXPLICIT_GEOMETRY_BINDING_NO_PRUNING')
    bindings=[{**l,'name':l['entity'],'slot_id':f"{source['scene_id']}/{l['entity']}/{l['fillable_link']}"}
              for l in proposal['locations']]
    expected={**source,'family_id':proposal['family_id'],
        'native_region_catalog':catalog,'native_region_bindings':bindings,
        'component_context':{**source['component_context'],
            'complete_leaf_binding':'explicit_native_links_pending_probe',
            'loaded_search_distribution':proposal}}
    expected.pop('manifest_sha256',None)
    actual={k:v for k,v in doc.items() if k!='manifest_sha256'}
    if actual!=expected:
        raise ValueError('NATIVE_REGION_CATALOG_MAPPING_OR_SOURCE_CHANGED')
    return proposal


def bind_native_region_catalog(assembly, preflight, output):
    """Bind every observed region to existing slots; never grants physical fit.

    A new manifest sits beside the immutable source so its existing relative
    USD references remain unchanged. Real body bindings and scene geometry are
    retained; the location mapping is a separate, complete per-volume list.
    """
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
    assembly,preflight,output=(Path(p).resolve() for p in (assembly,preflight,output))
    source=load_assembly_manifest(assembly)
    if output.parent!=assembly.parent or output.exists() or source.get('native_region_catalog'):
        raise ValueError('NATIVE_REGION_MANIFEST_REQUIRES_NEW_SIBLING_PATH')
    proposal=json.loads((preflight.parent/'loaded_native_search_proposal.json').read_text())
    refs=[assembly,preflight,preflight.parent/'loaded_component_topology.json',
          preflight.parent/'loaded_native_search_proposal.json']
    catalog={'version':1,'source_assembly':str(assembly),'preflight':str(preflight),
             'inputs':{str(p):sha256_path(p) for p in refs},
             'status':'explicit_links_pending_physical_probe','evaluation_ready_episodes':0}
    doc={**source,'family_id':proposal['family_id'],'native_region_catalog':catalog,
         'native_region_bindings':[{**l,'name':l['entity'],
             'slot_id':f"{source['scene_id']}/{l['entity']}/{l['fillable_link']}"} for l in proposal['locations']],
         'component_context':{**source['component_context'],
            'complete_leaf_binding':'explicit_native_links_pending_probe','loaded_search_distribution':proposal}}
    verify_native_region_catalog(doc)
    doc.pop('manifest_sha256',None)
    doc['manifest_sha256']=sha256_bytes(canonical_json(doc))
    atomic_write_json(output,doc)
    return doc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_split_worker(args.input_folder, args.out)
