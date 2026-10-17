"""InfiniGen's actual constraint solver over a preserved native BEHAVIOR home.

Legacy requests use explicit metadata planning envelopes. Geometry-bound
requests use live collision exports and native rigid poses. OmniGibson loads
the complete native home and DatasetObjects. Neither mode generates new house
architecture, and neither a Gen solve nor an envelope is a physics certificate.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.feasibility.ik_solver import quat_rotate
from rummagebench.sim.omnigibson.behavior_adapter import asset_record, object_config
from rummagebench.authoring.layout.spec import Placement

STRUCTURAL = {'walls', 'floors', 'ceilings'}
SUPPORTS = {'countertop', 'breakfast_table', 'coffee_table', 'console_table', 'dining_table'}


def exclude_covered_support_envelopes(records, *, max_gap=.05, tolerance=.001):
    """Do not propose props inside stacked native support envelopes.

    This classifies the *planning envelopes*, not actual empty space. Keep
    every native body and its obstacle envelope; only suppress the duplicate
    lower support constraint when another approved support fully covers it.
    Partial overlaps and arbitrary occupants cannot justify this exclusion.
    """
    supports=[o for o in records if o['support']]
    skipped=[]
    for lower in supports:
        size=np.asarray(lower['size']);center=np.asarray(lower['center'])
        corners=np.array([center+quat_rotate(lower['quaternion'],[x*size[0]/2,y*size[1]/2,size[2]/2])
                          for x in (-1,1) for y in (-1,1)])
        for upper in supports:
            if upper is lower:continue
            q=np.asarray(upper['quaternion']);inv=[-q[0],-q[1],-q[2],q[3]]
            local=np.array([quat_rotate(inv,c-np.asarray(upper['center'])) for c in corners])
            half=np.asarray(upper['size'])/2
            gap=-half[2]-local[:,2]
            if (np.all(np.abs(local[:,:2])<=half[:2]+tolerance)
                    and np.all(gap>=-tolerance) and np.all(gap<=max_gap)
                    and upper['center'][2]>lower['center'][2]+tolerance):
                lower['support']=False
                skipped.append(dict(entity=lower['name'],reason='covered_by_native_support_envelope',
                    covering_entity=upper['name'],gap_m=[float(gap.min()),float(gap.max())],
                    scope='planning-envelope classification; live support geometry still pending'))
                break
    return skipped


def _rectangle_has_space(half, obstacles, clearance=.005):
    """Exact rectangle-union test within the declared conservative XY model."""
    lo,hi=-np.asarray(half),np.asarray(half)
    if np.any(hi<lo):return False
    obstacles=[(a-clearance,b+clearance) for a,b in obstacles]
    axes=[]
    for axis in (0,1):
        cuts=sorted({float(lo[axis]),float(hi[axis]),*[float(np.clip(r[i][axis],lo[axis],hi[axis]))
                   for r in obstacles for i in (0,1)]})
        axes.append(cuts+[(a+b)/2 for a,b in zip(cuts,cuts[1:])])
    for x in axes[0]:
        for y in axes[1]:
            p=np.array([x,y])
            if not any(np.all(p>=a) and np.all(p<=b) for a,b in obstacles):return True
    return False


def qualify_supports_for_pool(context, assets):
    """Supplement only where this asset pool fits the native envelope model.

    Existing occupants are kept, including on surfaces receiving no additions.
    A negative result means no proposal under this conservative model; it is
    NOT confirmed physical fullness, nor removal of a search location.
    """
    for support in context['objects']:
        if not support['support']:continue
        q=np.asarray(support['quaternion']);inv=[-q[0],-q[1],-q[2],q[3]]
        top=support['size'][2]/2;native=[]
        for other in context['objects']:
            if other is support:continue
            center=quat_rotate(inv,np.asarray(other['center'])-support['center'])
            # Conservative projection of the native oriented envelope into
            # the support frame, using all corners instead of room labels.
            corners=np.array([center+quat_rotate(inv,quat_rotate(other['quaternion'],
                np.asarray(other['size'])*np.array([x,y,z])/2))
                for x in (-1,1) for y in (-1,1) for z in (-1,1)])
            native.append((other['name'],corners.min(axis=0),corners.max(axis=0)))
        feasible=False;blockers=set()
        for asset in assets.values():
            size=np.asarray(asset['bbox_size'])
            relevant=[(n,a,b) for n,a,b in native if b[2]>top+.002 and a[2]<top+size[2]+.002]
            for yaw in (0.,np.pi/4,np.pi/2,3*np.pi/4):
                c,s=abs(np.cos(yaw)),abs(np.sin(yaw))
                half=np.array([c*size[0]+s*size[1],s*size[0]+c*size[1]])/2
                domain=np.asarray(support['size'][:2])/2-half-.005
                rects=[]
                for n,a,b in relevant:
                    lower,upper=a[:2]-half,b[:2]+half
                    if np.all(upper>=-domain) and np.all(lower<=domain):
                        rects.append((lower,upper));blockers.add(n)
                if _rectangle_has_space(domain,rects):feasible=True;break
            if feasible:break
        # Tiny empty tables are left to the original solver. This exclusion
        # requires existing native obstruction, not just inconvenient size.
        if not feasible and blockers:
            support['support']=False
            context['skipped_supports'].append(dict(entity=support['name'],
                reason='no_pool_fit_in_native_occupied_envelope',blocking_entities=sorted(blockers),
                tested_assets=sorted(assets),tested_yaw_degrees=[0,45,90,135],clearance_m=.005,
                scope='conservative proposal model only; native occupants retained, physical occupancy unverified'))


def verify_result(request: dict, result: dict):
    from collections import Counter
    supports = {o['name'] for o in request['native_context']['objects'] if o['support']}
    expected = {f'daily_items_{name}' for name in supports}
    brief = None
    if 'living_brief' in request:
        from .living_brief import compile_brief, allocate_counts
        brief = compile_brief(request['living_brief'], request['native_context'])
        if request.get('count_allocation') != allocate_counts(brief, request['seed']):
            raise ValueError('BRIEF_ALLOCATION_MISMATCH')
    if (not supports or result.get('status') != 'constraints_satisfied'
            or set(result.get('violations', {})) != expected
            or any(v != 0 for v in result['violations'].values())):
        raise ValueError('INCOMPLETE_HOME_CONSTRAINT_EVIDENCE')
    placements = result.get('placements', [])
    counts = Counter(p['support_name'] for p in placements)
    if brief is not None:
        valid = (set(counts) <= supports and len(placements) == brief['total_count']
                 and all(s['minimum'] <= counts[name] <= s['maximum']
                         for name, s in brief['surfaces'].items()))
        valid = valid and all(counts[name] == n for name, n in request['count_allocation'].items())
    else:
        valid = counts == {name: request['per_surface'] for name in supports}
    if not valid:
        raise ValueError('INCOMPLETE_LIVING_BACKGROUND')
    names = [p['name'] for p in placements]
    if len(set(names)) != len(names) or set(names) & set(request['native_context']['retained_native_objects']):
        raise ValueError('LIVING_BACKGROUND_NAME_CONFLICT')
    if 'geometry' in request and set(result.get('native_poses',{})) != set(names):
        raise ValueError('COLLISION_NATIVE_POSES_INCOMPLETE')
    for row in placements:
        p = Placement.model_validate(row)
        if p.role != 'clutter' or f'{p.category}/{p.model_id}' not in request['assets']:
            raise ValueError('LIVING_BACKGROUND_ASSET_NOT_APPROVED')
        if 'geometry' in request:
            object_config(p,request['assets'][f'{p.category}/{p.model_id}'],result['native_poses'][p.name])


def native_start(root: Path, scene: str, room: str):
    """Same native segmentation convention as RoomMap, without a cv2 dependency."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    paths = [root/'scenes'/scene/'layout'/name for name in
             ('floor_insseg_0.png','floor_semseg_0.png','floor_trav_0.png')]
    ins, sem, trav = [np.array(Image.open(path)) for path in paths]
    categories_path = root/'metadata/room_categories.txt'
    categories = categories_path.read_text().splitlines()
    types = {}
    names = {}
    for i in sorted(int(v) for v in np.unique(ins) if v):
        row,col = np.argwhere(ins==i)[0]
        category = categories[int(sem[row,col])-1]
        rank = types.get(category,0); types[category] = rank+1
        names[i] = f'{category}_{rank}'
    ids = [i for i,name in names.items() if name == room]
    if not ids: raise ValueError('NATIVE_ROOM_MISSING')
    mask = np.isin(ins,ids) & (trav>0)
    occupied_rows, occupied_cols = np.nonzero(mask)
    if not len(occupied_rows): raise ValueError('NO_NATIVE_START_CLEARANCE_PROPOSAL')
    r0, r1 = int(occupied_rows.min()), int(occupied_rows.max()) + 1
    c0, c1 = int(occupied_cols.min()), int(occupied_cols.max()) + 1
    # All pixels outside this crop are false. Padding it with false pixels
    # gives exactly the same distances and row-major tie break as the full
    # house map, without an expensive full-house distance transform per task.
    dist = distance_transform_edt(np.pad(mask[r0:r1,c0:c1],1))[1:-1,1:-1]*.01
    row,col = np.unravel_index(int(dist.argmax()),dist.shape)
    if dist[row,col] < .60: raise ValueError('NO_NATIVE_START_CLEARANCE_PROPOSAL')
    return [float((col+c0-ins.shape[1]/2)*.01),float((row+r0-ins.shape[0]/2)*.01)], {
        str(p):sha256_path(p) for p in [*paths,categories_path]}


def native_context(root: Path, scene: str, *, scene_json=None) -> dict:
    source = Path(scene_json) if scene_json is not None else root/'scenes'/scene/'json'/f'{scene}_best.json'
    doc = json.loads(source.read_text())
    if not (np.allclose(doc['state']['pos'], [0, 0, 0])
            and np.allclose(doc['state']['ori'], [0, 0, 0, 1])):
        raise ValueError('NONIDENTITY_SCENE_FRAME_NOT_SUPPORTED')
    states = doc['state']['registry']['object_registry']
    records, inputs, skipped = [], {str(source): sha256_path(source)}, []
    for name, info in sorted(doc['objects_info']['init_info'].items()):
        args = info['args']
        cat, model = args['category'], args['model']
        if cat in STRUCTURAL:
            continue
        metadata = root/'objects'/cat/model/'misc/metadata.json'
        data = json.loads(metadata.read_text())
        inputs[str(metadata)] = sha256_path(metadata)
        size = np.asarray(data['bbox_size'], float)*np.asarray(args.get('scale', [1, 1, 1]), float)
        pose = states[name]['root_link']
        q = np.asarray(pose['ori'], float); q /= np.linalg.norm(q)
        offset = np.asarray(data['base_link_offset'])*np.asarray(args.get('scale', [1, 1, 1]))
        center = np.asarray(pose['pos']) + quat_rotate(q, offset)
        if not np.isfinite(np.r_[size, center, q]).all() or np.any(size <= 0):
            raise ValueError(f'NATIVE_GEOMETRY_UNKNOWN:{name}')
        # Keep an oriented conservative envelope for all non-structural
        # furniture and existing occupants. No native object is deleted.
        yaw = math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
        upright = np.allclose(quat_rotate(q, [0, 0, 1]), [0, 0, 1], atol=.001)
        support = cat in SUPPORTS and upright and min(size[:2]) >= .25
        if cat in SUPPORTS and not support:
            skipped.append(dict(entity=name, reason='support_too_narrow_or_tilted'))
        records.append(dict(name=name, category=cat, model=model, center=center.tolist(),
                            size=size.tolist(), quaternion=q.tolist(), yaw=yaw,
                            rooms=args.get('in_rooms') or [], support=bool(support),
                            openable=bool(data.get('openable_joint_ids'))))
    skipped.extend(exclude_covered_support_envelopes(records))
    return dict(scene=scene, scene_json=str(source), inputs=inputs, objects=records,
                retained_native_objects=sorted(doc['objects_info']['init_info']),
                skipped_supports=skipped)


def prepare_home(root: Path, scene: str, pool: list, seed: int, out: Path,
                 upstream: Path, python: str, per_surface=2, steps=220, timeout=300,
                 living_brief=None, geometry=None) -> dict:
    from rummagebench.authoring.tasks.infinigen import CPU_ENV, cpu_sandbox_argv, source_audit
    if not 1 <= timeout <= 1800:
        raise ValueError('BOUNDED_HOME_SOLVE_TIMEOUT_REQUIRED')
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('OUTPUT_NOT_EMPTY')
    out.mkdir(parents=True, exist_ok=True)
    records = {f"{a['category']}/{a['model']}": asset_record(root, **a) for a in pool}
    geometry_doc = None
    if geometry is not None:
        from .indoor_geometry import load_geometry
        geometry = str(Path(geometry).resolve())
        geometry_doc = load_geometry(geometry, scene=scene, assets=records)
    context = native_context(root.resolve(), scene,
        scene_json=geometry_doc['scene_snapshot'] if geometry_doc else None)
    if geometry_doc:
        context['inputs'].update(geometry_doc['inputs'])
        context['inputs'][geometry] = sha256_path(geometry)
    if not any(o['support'] for o in context['objects']):
        raise ValueError('NO_APPROVED_NATIVE_SUPPORT_SURFACES')
    # Envelope occupancy is an explicitly conservative legacy proposal filter.
    # With real geometry let Gen test that geometry, including table cavities.
    if not geometry_doc:
        qualify_supports_for_pool(context,records)
    if not any(o['support'] for o in context['objects']):
        raise ValueError('NO_POOL_FIT_ON_NATIVE_SUPPORT_ENVELOPES')
    request = dict(version=1, seed=seed, per_surface=per_surface, steps=steps,
                   native_context=context, assets=records, upstream=source_audit(upstream))
    if geometry_doc:
        request.update(geometry=geometry, geometry_sha256=sha256_path(geometry))
    if living_brief is not None:
        from .living_brief import compile_brief, allocate_counts
        brief = compile_brief(living_brief, context)
        request.update(version=3, living_brief=brief, count_allocation=allocate_counts(brief, seed))
    request_path = out/'solver_request.json'
    atomic_write_json(request_path, request)
    argv = [python, '-m', __name__, 'solve', str(request_path), str(out/'solver_result.json')]
    env = dict(os.environ, **CPU_ENV)
    env['XDG_CONFIG_HOME'] = '/tmp/findingbench_config'
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), str(upstream.resolve())])
    with (out/'solver.log').open('w') as log:
        run = subprocess.run(cpu_sandbox_argv(argv, out), env=env, stdout=log,
                             stderr=subprocess.STDOUT, timeout=timeout, check=False, cwd=upstream)
    receipt_path = out/'solver_result.json'
    if run.returncode != 0 or not receipt_path.exists():
        raise ValueError(f'INFINIGEN_HOME_SOLVE_FAILED: see {out / "solver.log"}')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'constraints_satisfied' or receipt['request_sha256'] != sha256_path(request_path):
        raise ValueError('INFINIGEN_HOME_CONSTRAINTS_NOT_SATISFIED')
    verify_result(request, receipt)
    items = []
    for row in receipt['placements']:
        p = Placement.model_validate(row)
        a = records[f'{p.category}/{p.model_id}']
        items.append(dict(placement=p.model_dump(), og_config=object_config(p, a,
            receipt.get('native_poses',{}).get(p.name))))
    # verify_result enforces either legacy exact counts or the full brief.
    manifest = dict(version=2, source='behavior_native_layout', status='pending_simulation',
        base_scene=scene, layout_seed=seed, assets=records, objects=items, rejected=[],
        living_context=dict(native=context, solver_request=str(request_path), solver_result=str(receipt_path),
                            inputs={**context['inputs'], str(request_path):sha256_path(request_path),
                                    str(receipt_path):sha256_path(receipt_path)}),
        evaluation_ready_episodes=0)
    if geometry_doc:
        manifest['living_context']['geometry'] = geometry
    manifest['manifest_sha256'] = sha256_bytes(canonical_json(manifest))
    atomic_write_json(out/'assembly.json', manifest)
    return manifest


def solve(request_path: Path, result_path: Path):
    """Run inside the installed, CPU-isolated InfiniGen/bpy environment."""
    import bpy
    import gin
    import logging
    logging.basicConfig(level=logging.INFO)
    from infinigen.core import init as ig_init
    original_require = ig_init.require_blender_addon
    def local_addon_only(addon, fail='fatal', allow_online=False):
        import addon_utils
        installed = {a.__name__ for a in addon_utils.modules(refresh=True)}
        if addon not in installed and 'bl_ext.blender_org.'+addon not in installed:
            if fail == 'fatal': raise ValueError(f'LOCAL_ADDON_MISSING:{addon}')
            return False
        return original_require(addon, fail=fail, allow_online=False)
    ig_init.require_blender_addon = local_addon_only
    from infinigen.core import tags as t, tagging
    from infinigen.core.constraints import constraint_language as cl, reasoning as r, usage_lookup
    from infinigen.core.constraints.example_solver.solve import Solver
    from infinigen.core.constraints.example_solver.state_def import State, ObjectState
    from infinigen.core.constraints.evaluator.evaluate import evaluate_problem
    from infinigen.core.placement.factory import AssetFactory
    from infinigen.core.util import blender as butil
    from infinigen.core.util.math import FixedSeed
    from infinigen_examples.constraints import util as cu

    req = json.loads(request_path.read_text())
    geometry_doc = None
    if 'geometry' in req:
        from .indoor_geometry import load_geometry, world_to_local
        if sha256_path(req['geometry']) != req['geometry_sha256']:
            raise ValueError('COLLISION_REQUEST_BINDING_MISMATCH')
        geometry_doc = load_geometry(req['geometry'],scene=req['native_context']['scene'],assets=req['assets'])
    if not 1 <= req['per_surface'] <= 6 or not 10 <= req['steps'] <= 2000:
        raise ValueError('BOUNDED_HOME_SOLVE_REQUIRED')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    for key, value in dict(initial_temp=3., final_temp=.001, finetune_pct=.15,
                           max_invalid_candidates=5, print_report_freq=0).items():
        gin.bind_parameter('SimulatedAnnealingSolver.'+key, value)
    def box(name, size):
        obj = butil.spawn_cube(size=1)
        obj.name = name; obj.dimensions = size
        butil.apply_transform(obj, loc=False, rot=False, scale=True)
        from infinigen.core.constraints.example_solver.geometry import parse_scene
        parse_scene.preprocess_obj(obj)
        tagging.tag_canonical_surfaces(obj)
        return obj
    def collision_object(name, mesh, vertices=None):
        data = bpy.data.meshes.new(name+'_collision')
        data.from_pydata(mesh['vertices'] if vertices is None else vertices, [], mesh['faces'])
        data.update()
        obj = bpy.data.objects.new(name,data)
        bpy.context.collection.objects.link(obj)
        from infinigen.core.constraints.example_solver.geometry import parse_scene
        parse_scene.preprocess_obj(obj)
        tagging.tag_canonical_surfaces(obj)
        return obj
    if geometry_doc:
        from infinigen.core.constraints.constraint_language import util as iu
        from mathutils import Quaternion
        from .indoor_geometry import rigid_rotation_update
        original_rotate=iu.rotate
        def native_rigid_rotate(scene,name,axis,angle):
            obj=bpy.data.objects[name]
            if not obj.get('findingbench_native_rigid',False):
                return original_rotate(scene,name,axis,angle)
            q=rigid_rotation_update(np.asarray(obj.matrix_world,float),axis,angle)
            # Euler mode is also what upstream pose_backup / restore use.
            # Assign rotation, not matrix_world: Blender matrix decomposition
            # must not create a slightly non-unit scale on every rotation.
            obj.rotation_mode='XYZ'
            obj.rotation_euler=Quaternion([q[3],*q[:3]]).to_euler('XYZ')
            if scene:iu.sync_trimesh(scene,name)
        iu.rotate=native_rigid_rotate
        original_tag_surfaces=tagging.tag_canonical_surfaces
        def native_tag_surfaces(obj,*args,**kwargs):
            original_tag_surfaces(obj,*args,**kwargs)
            if (obj.get('findingbench_native_rigid',False)
                    and not tagging.tagged_face_mask(obj,{t.Subpart.Bottom}).any()):
                from .indoor_geometry import supporting_bottom_faces
                mask=supporting_bottom_faces(
                    [list(v.co) for v in obj.data.vertices],
                    [list(p.vertices) for p in obj.data.polygons])
                tagging.tag_object(obj,t.Subpart.Bottom,mask=mask)
                logging.info('Native bottom contact face: %s faces=%s',obj.name,np.flatnonzero(mask).tolist())
        # Addition retags placeholders after factory creation. Scope this
        # adapter to our measured native props, including that later call.
        tagging.tag_canonical_surfaces=native_tag_surfaces
    factories = []
    for index, (key, record) in enumerate(sorted(req['assets'].items())):
        def create_placeholder(self, **kwargs):
            if geometry_doc:
                key = f"{self.record['category']}/{self.record['model']}"
                obj=collision_object('native_collision_asset',geometry_doc['assets'][key])
                obj['findingbench_native_rigid']=True
                return obj
            x, y, z = self.record['bbox_size']
            return box('native_asset_planning_envelope', [y, x, z])
        cls = type('NativeDailyItem'+str(index), (AssetFactory,),
                   dict(record=record, create_placeholder=create_placeholder))
        factories.append(cls)
    usage_lookup.initialize_from_dict({t.Semantics.Object:factories, t.Semantics.Dishware:factories,
        t.Semantics.RealPlaceholder:factories, t.Semantics.SingleGenerator:[],
        t.Semantics.AssetAsPlaceholder:[], t.Semantics.PlaceholderBBox:[]})
    state = State()
    from infinigen.core.constraints.example_solver.geometry import parse_scene
    for row in req['native_context']['objects']:
        if geometry_doc:
            mesh = geometry_doc['native'][row['name']]
            if mesh.get('noncolliding_visual_only'):
                if row['support']:raise ValueError('NONCOLLIDING_SUPPORT_UNSUPPORTED')
                continue
            vertices = world_to_local(mesh['vertices'], row['center'], row['quaternion'])
            obj = collision_object(row['name'],mesh,vertices.tolist())
        else:
            obj = box(row['name'], row['size'])
        obj.location = row['center']; obj.rotation_mode = 'QUATERNION'
        q = row['quaternion']; obj.rotation_quaternion = [q[3], *q[:3]]
        tags = {t.Semantics.Object, -t.Semantics.Room, t.SpecificObject(row['name'])}
        if row['support']: tags.add(t.Semantics.Table)
        state.objs[row['name']] = ObjectState(obj=obj, tags=tags)
        parse_scene.add_to_scene(state.trimesh_scene, obj)
    if geometry_doc:
        # Architecture stays real triangle geometry; a whole-room bounding
        # box would erroneously fill its interior. It cannot receive props.
        included = {o['name'] for o in req['native_context']['objects']}
        for name in sorted(set(geometry_doc['native'])-included):
            if geometry_doc['native'][name].get('noncolliding_visual_only'):continue
            obj = collision_object(name,geometry_doc['native'][name])
            state.objs[name] = ObjectState(obj=obj,tags={t.Semantics.Object,-t.Semantics.Room,t.SpecificObject(name)})
            parse_scene.add_to_scene(state.trimesh_scene,obj)
    furniture = cl.scene()[{t.Semantics.Object, -t.Semantics.Room, t.Semantics.Table}]
    props = cl.scene()[{t.Semantics.Object, -t.Semantics.Room, t.Semantics.Dishware}].related_to(furniture, cu.ontop)
    # Same native DSL relations and top-area fullness term used by
    # infinigen_examples.constraints.home.home_furniture_constraints().
    # Existing architectural / furniture arrangement is a fixed native input.
    # Materialize each support's bound: a single quantified bound without
    # upstream greedy variable assignments would count all props together.
    constraints = {f'daily_items_{row["name"]}':
        props.related_to(furniture[t.SpecificObject(row['name'])]).count().equals(req['per_surface'])
        for row in req['native_context']['objects'] if row['support']}
    scores = {
        'obj_ontop_nonstorage_fullness': furniture.mean(lambda f:
            props.related_to(f).volume(dims=(0, 1)).safediv(f.volume(dims=(0, 1)))
                .sub(.12).abs().minimize(weight=10)),
        'daily_item_separation': furniture.mean(lambda f:
            cl.min_distance_internal(props.related_to(f)).maximize(weight=.1))}
    if 'living_brief' in req:
        from .living_brief import compile_brief, allocate_counts
        brief = compile_brief(req['living_brief'], req['native_context'])
        allocation = allocate_counts(brief, req['seed'])
        if req.get('count_allocation') != allocation:
            raise ValueError('BRIEF_ALLOCATION_MISMATCH')
        constraints = {}
        scores = {}
        for name, surface in brief['surfaces'].items():
            support = furniture[t.SpecificObject(name)]
            count = props.related_to(support).count()
            constraints[f'daily_items_{name}'] = count.equals(allocation[name])
            scores[f'fullness_preference_{name}'] = support.mean(lambda f, s=surface:
                props.related_to(f).volume(dims=(0, 1)).safediv(f.volume(dims=(0, 1)))
                    .sub(s['fullness']).abs().minimize(weight=10.))
        # No spread-out reward: clutter is permitted if the geometric solver
        # and subsequent native physical checks accept it. Clustering is not
        # itself certified by these count/fullness preferences.
    problem = cl.Problem(constraints=constraints, score_terms=scores)
    with FixedSeed(req['seed']):
        solver = Solver(output_folder=result_path.parent); solver.state = state
        try:
            solver.solve_objects(problem, r.Domain({t.Semantics.Object, -t.Semantics.Room, t.Semantics.Dishware}), {},
                n_steps=req['steps'], desc='native_living_background', abort_unsatisfied=True)
        finally:
            # Preserve actual transforms even when constraint or conversion
            # validation rejects; this receipt cannot authorize a candidate.
            atomic_write_json(result_path.parent/'solver_transforms.json',dict(
                status='diagnostic_only',request_sha256=sha256_path(request_path),
                objects=[dict(name=n,asset=f"{s.generator.record['category']}/{s.generator.record['model']}",
                    matrix=np.asarray(s.obj.matrix_world,float).tolist(),scale=list(s.obj.scale))
                    for n,s in state.objs.items() if s.generator is not None]))
    evaluated = evaluate_problem(problem, state)
    violations = {k:float(v) for k,v in evaluated.violations.items()}
    placements = []
    native_poses = {}
    for name, objstate in state.objs.items():
        if objstate.generator is None: continue
        obj = objstate.obj; record = objstate.generator.record
        parent = next(rel.target_name for rel in objstate.relations if isinstance(rel.relation, cl.StableAgainst))
        x, y, z = record['bbox_size']
        position,orientation = list(obj.location),math.degrees(obj.rotation_euler.z)+180
        name='daily_'+str(len(placements)).zfill(3)
        if geometry_doc:
            from .indoor_geometry import native_layout_pose
            native_poses[name]=native_layout_pose(np.asarray(obj.matrix_world,float))
            position=native_poses[name]['position']
            # Legacy yaw is descriptive only in this mode; native_poses is
            # authoritative and hash-bound through result and assembly.
        placements.append(dict(name=name, category=record['category'],
            model_id=record['model'], position=position,
            orientation_deg=orientation,
            aabb_size=[y, x, z], role='clutter', support_name=parent))
    atomic_write_json(result_path, dict(status='constraints_satisfied' if all(v==0 for v in violations.values()) else 'rejected',
        request_sha256=sha256_path(request_path), violations=violations, placements=placements,
        upstream_api='infinigen.core.constraints.example_solver.solve.Solver.solve_objects',
        geometry_sha256=req.get('geometry_sha256'),
        native_poses=native_poses,
        scope=('native_house_fixed; live_collision_geometry_for_layout; physical_checks_pending' if geometry_doc
               else 'native_house_fixed; metadata_envelopes_for_daily_object_layout; physical_checks_pending')))


if __name__ == '__main__':
    if sys.argv[1] != 'solve': raise SystemExit('expected solve')
    solve(Path(sys.argv[2]), Path(sys.argv[3]))
