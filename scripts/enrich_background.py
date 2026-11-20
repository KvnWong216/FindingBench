#!/usr/bin/env python3
"""Add reviewed sparse native props to a separate frozen task variant.

Restore the accepted parent first. Never resample its task inventory. New
objects remain dynamic; native contact, stability and independent reload
are required before a variant is published. This is authoring, not a skill.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('FINDINGBENCH_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def xy_gap(a, b):
    return math.hypot(*(max(0., a[0][i]-b[1][i], b[0][i]-a[1][i]) for i in (0, 1)))


def checked_box(box):
    if (len(box) != 2 or any(len(row) != 3 for row in box)
            or not all(math.isfinite(v) for row in box for v in row)
            or any(box[1][i] <= box[0][i] for i in range(3))):
        raise ValueError('INVALID_GEOMETRY')
    return box


def scene_records(backend):
    """Keep every collider, including robots without room membership."""
    return {obj.name: dict(category=obj.category, model=getattr(obj, 'model', None),
        n_joints=obj.n_joints, rooms=list(getattr(obj, 'in_rooms', None) or []),
        aabb=backend.entity_aabb(obj.name)) for obj in backend._env.scene.objects}


def check_spot(box, support, protected, obstacles, *, edge=.08, separation=1.):
    """Conservative proposal filter; physical validation still decides support."""
    checked_box(box); checked_box(support)
    if not all(support[0][i]+edge <= box[0][i] and box[1][i] <= support[1][i]-edge for i in (0, 1)):
        raise ValueError('SUPPORT_EDGE_CLEARANCE')
    if box[0][2] < support[1][2]-.005 or box[1][2]-support[1][2] > .30:
        raise ValueError('BACKGROUND_HEIGHT_LIMIT')
    if any(xy_gap(box, checked_box(other)) < separation for other in protected):
        raise ValueError('TASK_WORKSPACE_CLEARANCE')
    for other in obstacles:
        checked_box(other)
        if other[1][2] <= support[1][2]+.001:
            continue  # furniture below the supporting surface
        if all(box[0][i]-.03 < other[1][i] and other[0][i] < box[1][i]+.03 for i in range(3)):
            raise ValueError('BACKGROUND_NEIGHBOUR_COLLISION')


def plan_props(support, props, protected, obstacles):
    """At most two props and eight percent of one reviewed surface."""
    checked_box(support)
    if not 1 <= len(props) <= 2:
        raise ValueError('SPARSE_OBJECT_LIMIT')
    for prop in props:
        size = prop['size']
        if len(size) != 3 or any(not math.isfinite(v) or v <= 0 or v > .30 for v in size):
            raise ValueError('BACKGROUND_ASSET_SIZE')
    surface_area = math.prod(support[1][i]-support[0][i] for i in (0, 1))
    area = sum(p['size'][0]*p['size'][1] for p in props)
    if area/surface_area > .08:
        raise ValueError('SPARSE_COVERAGE_LIMIT')
    boxes = []
    for prop in props:
        size = prop['size']; x, y = prop['position_xy']
        z = support[1][2]+.01
        box = [[x-size[0]/2, y-size[1]/2, z], [x+size[0]/2, y+size[1]/2, z+size[2]]]
        check_spot(box, support, protected, [*obstacles, *boxes])
        boxes.append(box)
    return dict(proposal_boxes=boxes, footprint_fraction=area/surface_area)


def world_poses(backend):
    result = {}
    cv = lambda x: [float(v) for v in x]
    for obj in backend._env.scene.objects:
        result[obj.name] = dict(links={name: [cv(v) for v in link.get_position_orientation()]
                                     for name, link in obj.links.items()},
                                joints=cv(obj.get_joint_positions()) if obj.n_joints else [])
    return result


def compare_poses(before, after):
    import numpy as np
    metrics = dict(max_displacement_m=0., max_rotation_rad=0., max_joint_displacement=0.)
    for name, value in before.items():
        current = after[name]
        if set(value['links']) != set(current['links']):
            raise ValueError('ORIGINAL_LINK_TOPOLOGY_CHANGED')
        for link, (p, q) in value['links'].items():
            p2, q2 = current['links'][link]
            displacement = float(np.linalg.norm(np.asarray(p2)-p))
            q, q2 = np.asarray(q), np.asarray(q2)
            angle = float(2*np.arccos(np.clip(abs(np.dot(q, q2))/(np.linalg.norm(q)*np.linalg.norm(q2)), 0, 1)))
            metrics['max_displacement_m'] = max(metrics['max_displacement_m'], displacement)
            metrics['max_rotation_rad'] = max(metrics['max_rotation_rad'], angle)
        if value['joints']:
            metrics['max_joint_displacement'] = max(metrics['max_joint_displacement'],
                float(np.max(abs(np.asarray(current['joints'])-value['joints']))))
    if (metrics['max_displacement_m'] > .001 or metrics['max_rotation_rad'] > .01
            or metrics['max_joint_displacement'] > .005):
        raise ValueError('ORIGINAL_WORLD_MOVED: '+json.dumps(metrics))
    return metrics


def native(args, task, plan):
    if plan.get('furniture_policy'):
        from furnish_native import run
        return run(args, task, plan, ROOT)
    import torch
    from rummagebench.core.scenario import load_scenario, ScenarioSpec
    from rummagebench.adapters.python_api import prepare_renderer_grounding, install_renderer_grounding
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.authoring.tasks.household_state import export_native_member, load_frozen_member
    from rummagebench.skills.move import navigation_aabbs
    # This observer only measures native transforms; the parent's frozen
    # backend and placement/restore implementation remain pinned.
    import importlib.util
    observer_path = Path(getattr(args, 'observer', None) or ROOT/'scripts/v2/validate_dense_pilot.py').resolve()
    module_spec = importlib.util.spec_from_file_location('background_motion_observer', observer_path)
    observer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(observer)
    observe_native_motion = observer.observe_native_motion
    from PIL import Image
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    source = load_scenario(task['play_scenario']); prepare_renderer_grounding(source)
    frozen = load_frozen_member(Path(source.scene.frozen_member_manifest), source)
    backend = OmniGibsonBackend(seed=task.get('seed', 0))
    report = dict(status='failed', parent_task=task, plan=plan, full_skill_completion='unverified',
                  action_error_rate='unmeasured; requires paired execution trials',
                  process_id=os.getpid(), tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    auto_inputs = []
    automatic = bool(plan.get('auto_policy'))
    try:
        spec = load_scenario(output/'play.yaml') if args.restore else source
        prepare_renderer_grounding(spec)
        backend.setup(spec); install_renderer_grounding(backend)
        task_names = {obj.name for obj in source.objects}
        task_names.update(p.receptacle for p in source.placements if not p.receptacle.startswith('floors_'))
        task_names.update(p.receptacle for p in source.goal if getattr(p, 'receptacle', None))
        task_names.update(source.initial_states)
        task_names.update(backend.robot_entity_names())
        if not args.restore and plan.get('auto_policy'):
            import background_planner as planner
            policy_path = Path(plan['auto_policy'])
            policy = json.loads(policy_path.read_text())
            if policy['version'] != planner.VERSION:
                raise ValueError('BACKGROUND_POLICY_VERSION_MISMATCH')
            avoided = set(plan.get('avoid_assets', []))
            for category in policy['assets']:
                policy['assets'][category] = [a for a in policy['assets'][category]
                    if f'{a["category"]}/{a["model"]}' not in avoided]
            records = scene_records(backend)
            for name in plan.get('avoid_supports', []):
                if name in records:
                    records[name]['n_joints'] = max(1, records[name]['n_joints'])
            (output/'scene_geometry.private.json').write_text(json.dumps(records))
            document = dict(source.model_dump(mode='json'), parent_task_id=task['parent_task_id'])
            candidates = planner.propose(document, records, task_names, policy, variant=plan.get('variant', 0))
            rejected = []
            for candidate in candidates:
                rec = records[candidate['support']]
                neighbours = [r['aabb'] for name, r in records.items()
                              if name != candidate['support'] and r['category'] not in planner.STRUCTURAL]
                try:
                    plan_props(rec['aabb'], candidate['props'], [records[n]['aabb'] for n in task_names], neighbours)
                    plan = candidate
                    break
                except ValueError as error:
                    rejected.append(dict(support=candidate['support'], error=str(error)))
            else:
                raise ValueError('NO_PREFLIGHT_VALID_BACKGROUND_PROPOSAL:'+json.dumps(rejected))
            report['proposal_rejections'] = rejected
            report['plan'] = plan
            report['auto_policy_sha256'] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            auto_inputs = [policy_path, Path(planner.__file__).resolve()]
            (output/'resolved_plan.json').write_text(json.dumps(plan, indent=2))
            auto_inputs.append(output/'resolved_plan.json')
        if args.restore:
            original = json.loads((output/'original_world_poses.private.json').read_text())
            expected = json.loads((output/'exported_world_poses.private.json').read_text())
            report['reload_world_preservation'] = compare_poses(expected, world_poses(backend))
        else:
            scene = backend._env.scene
            original = world_poses(backend)
            (output/'original_world_poses.private.json').write_text(json.dumps(original))
            Image.fromarray(backend.capture_panorama_frame().rgb).save(output/'before.png')
            support = scene.object_registry('name', plan['support'])
            from background_planner import SUPPORTS
            if support.category not in SUPPORTS or support.n_joints:
                raise ValueError('REVIEWED_NON_ARTICULATED_SURFACE_REQUIRED')
            task_categories = {obj.category for obj in source.objects}
            protected_names = task_names
            if plan['support'] in protected_names:
                raise ValueError('TASK_SURFACE_CANNOT_BE_DECORATED')
            if plan['room'] not in (support.in_rooms or []):
                raise ValueError('REVIEWED_ROOM_BINDING_MISMATCH')
            protected = [backend.entity_aabb(name) for name in sorted(protected_names)]
            boxes = {obj.name: backend.entity_aabb(obj.name) for obj in scene.objects}
            # Whole-room wall/floor envelopes contain empty interior space;
            # they are not solid boxes. Native contacts/stability remain the
            # final physical gate; furniture and existing props filter here.
            neighbour_boxes = {obj.name: boxes[obj.name] for obj in scene.objects
                               if obj.name != support.name and obj.category not in ('walls', 'floors', 'ceilings')}
            (output/'geometry.private.json').write_text(json.dumps(dict(
                support=boxes[support.name], protected={name: boxes[name] for name in protected_names},
                neighbours=neighbour_boxes), indent=2))
            props = deepcopy(plan['props'])
            assets = Path(os.environ['OMNIGIBSON_DATA_PATH'])/'behavior-1k-assets/objects'
            for prop in props:
                if prop['category'] in task_categories:
                    raise ValueError('TASK_CATEGORY_CANNOT_BE_ADDED')
                metadata = assets/prop['category']/prop['model']/'misc/metadata.json'
                if prop.get('metadata_sha256') and hashlib.sha256(metadata.read_bytes()).hexdigest() != prop['metadata_sha256']:
                    raise ValueError('BACKGROUND_ASSET_METADATA_CHANGED')
                prop['size'] = json.loads(metadata.read_text())['bbox_size']
            report['geometric_proposal'] = plan_props(boxes[support.name], props, protected,
                list(neighbour_boxes.values()))
            report['minimum_task_workspace_gap_m'] = min(xy_gap(box, other)
                for box in report['geometric_proposal']['proposal_boxes'] for other in protected)
            baseline = deepcopy(backend._sim.dump_state(serialized=False))
            # Pair against the same import stop/play cycle and elapsed native
            # steps without props. Passive scene doors can move their joints
            # after a restore even in the unchanged parent; that must not be
            # mistaken for an effect of decoration or silently ignored.
            backend._sim.stop(); backend._sim.play(); backend._sim.step()
            backend._sim.load_state(deepcopy(baseline), serialized=False)
            for _ in props:
                backend.settle(20)
            backend.settle(60)
            control_output = output/'unchanged_parent_control'
            control_output.mkdir(exist_ok=True)
            report['control_motion'] = observe_native_motion(backend, source, control_output)
            if not report['control_motion']['converged']:
                raise ValueError('UNCHANGED_PARENT_NOT_STABLE')
            control_poses = world_poses(backend)
            (output/'control_world_poses.private.json').write_text(json.dumps(control_poses))
            backend._sim.stop()
            from omnigibson.objects import DatasetObject
            added = []
            for index, prop in enumerate(props):
                obj = DatasetObject(name=f'background_{index:02d}', category=prop['category'], model=prop['model'], fixed_base=False)
                scene.add_object(obj)
                obj.set_position_orientation(position=[100.+index*10, 100., 5.], orientation=[0, 0, 0, 1])
                added.append(obj)
            backend._sim.play(); backend._sim.step()
            # SDK named-state composition, never edit serialized tensor offsets.
            combined = deepcopy(backend._sim.dump_state(serialized=False))
            old = baseline[0]['registry']['object_registry']
            actual = combined[0]['registry']['object_registry']
            if set(actual) != set(old) | {obj.name for obj in added}:
                raise ValueError('ADDITIVE_OBJECT_TOPOLOGY_MISMATCH')
            combined[0]['registry']['object_registry'] = {
                name: deepcopy(old[name] if name in old else state) for name, state in actual.items()}
            backend._sim.load_state(combined, serialized=False)
            doc = source.model_dump(mode='json', exclude_none=True)
            doc['id'] = source.id+('_background_v2' if automatic else '_background_v1')
            doc['scene'].pop('frozen_member_manifest')
            for obj, prop in zip(added, props):
                doc['objects'].append(dict(name=obj.name, category=prop['category'], model=prop['model'], fixed_base=False))
                doc['placements'].append(dict(entity=obj.name, relation='on_top', receptacle=support.name,
                                              position_xy=prop['position_xy']))
            spec = ScenarioSpec.model_validate(doc); backend._scenario = spec
            # Formal content preservation: the only recipe changes are its ID
            # and additive native objects/placements. No goal or spawn edits.
            parent_doc = source.model_dump(mode='json', exclude_none=True)
            derived_doc = spec.model_dump(mode='json', exclude_none=True)
            derived_doc['id'] = parent_doc['id']
            derived_doc['scene']['frozen_member_manifest'] = parent_doc['scene']['frozen_member_manifest']
            derived_doc['objects'] = derived_doc['objects'][:-len(added)]
            derived_doc['placements'] = derived_doc['placements'][:-len(added)]
            if derived_doc != parent_doc:
                raise ValueError('TASK_CONTENT_CHANGED_DURING_ENRICHMENT')
            report['task_content_preserved'] = True
            for obj, prop in zip(added, props):
                obj.sleep_threshold = 0.
                if not backend._place_at_xy(obj, support, prop['position_xy']):
                    raise ValueError('NATIVE_BACKGROUND_SUPPORT_FAILED: '+obj.name)
                box = backend.entity_aabb(obj.name)
                report.setdefault('placement_checks', []).append(dict(
                    entity=obj.name, category=obj.category, model=obj.model,
                    aabb=box, support=support.name, support_aabb=boxes[support.name]))
                check_spot(box, boxes[support.name], protected,
                           [*neighbour_boxes.values(), *[backend.entity_aabb(other.name)
                            for other in added if other is not obj and added.index(other) < added.index(obj)]])
            report['added'] = [dict(name=obj.name, category=obj.category, model=obj.model,
                aabb=backend.entity_aabb(obj.name), navigation_aabbs=navigation_aabbs(backend, obj.name)) for obj in added]
            actual_coverage = sum(math.prod(row['aabb'][1][i]-row['aabb'][0][i] for i in (0, 1))
                                  for row in report['added'])/math.prod(boxes[support.name][1][i]-boxes[support.name][0][i] for i in (0, 1))
            report['actual_surface_coverage_fraction'] = actual_coverage
            if actual_coverage > .08:
                raise ValueError('NATIVE_SPARSE_COVERAGE_LIMIT')
            # Exact production MOVE representation: all added colliders must lie
            # above its base-height band. No new base obstacles at any XY/yaw.
            if any(lo[2] <= .5 and hi[2] >= .30 for obj in report['added'] for lo, hi in obj['navigation_aabbs']):
                raise ValueError('NEW_NAVIGATION_BASE_OBSTACLE')
            report['navigation'] = 'no added collider intersects the production MOVE base-height band'
        # Use the standard validator's settling period on creation, control
        # and reload. The initial reload pose was checked before these steps.
        backend.settle(60)
        motion = observe_native_motion(backend, spec, output)
        if not motion['converged']:
            raise ValueError('NATIVE_MOTION_NOT_CONVERGED')
        report['motion'] = motion
        current = world_poses(backend)
        (output/'observed_world_poses.private.json').write_text(json.dumps(current))
        report['task_world_preservation'] = compare_poses(
            {name: original[name] for name in task_names}, current)
        report['task_world_entities'] = sorted(task_names)
        if not args.restore:
            report['paired_world_preservation'] = compare_poses(control_poses, current)
        backend.validate_physics_state(); backend._validate_spawn_clearance()
        from omnigibson.object_states import Inside, OnTop, Open
        for placement in spec.placements:
            obj = backend._env.scene.object_registry('name', placement.entity)
            rec = backend._env.scene.object_registry('name', placement.receptacle)
            if not backend._verify_relation(obj, rec, placement.relation, Inside, OnTop) or not backend._in_link(obj, rec, placement.link):
                raise ValueError('RELATION_CHANGED: '+placement.entity)
        for name, state in spec.initial_states.items():
            if state.open is not None and bool(backend._env.scene.object_registry('name', name).states[Open].get_value()) != state.open:
                raise ValueError('INITIAL_OPEN_STATE_CHANGED: '+name)
        from types import SimpleNamespace
        from rummagebench.evaluation.goal_checker import check_goal
        goal = check_goal(SimpleNamespace(held_object=None), spec, backend)
        if goal.satisfied or goal.fault:
            raise ValueError('INVALID_INITIAL_GOAL')
        if not args.restore:
            (output/'exported_world_poses.private.json').write_text(json.dumps(current))
            deps = dict(frozen['dependencies'])
            for path in (Path(__file__).resolve(), args.plan, observer_path, *auto_inputs):
                archived = output/'authoring_sources'/path.name
                archived.parent.mkdir(exist_ok=True)
                archived.write_bytes(path.read_bytes())
                deps[str(archived)] = hashlib.sha256(archived.read_bytes()).hexdigest()
            export_native_member(backend, spec, output,
                dict(source='additive_sparse_background_v1', parent_frozen_sha256=frozen['manifest_sha256']), deps)
        Image.fromarray(backend.capture_panorama_frame().rgb).save(output/('restored.png' if args.restore else 'after.png'))
        report['status'] = 'restore_validated' if args.restore else 'initialization_validated'
        if args.restore:
            variant = dict(task, id=spec.id, play_scenario=str(output/'play.yaml'), scenario=str(output/'play.yaml'),
                           stage='ready_for_human_review', parent_variant=task['id'])
            # Observing stability advances physics. Restore the exact exported
            # state again before baking a derivative inspection cache.
            from rummagebench.authoring.tasks.household_state import restore_frozen_member
            restore_frozen_member(backend, Path(spec.scene.frozen_member_manifest), spec)
            report['cache_restore_pose_match'] = compare_poses(expected, world_poses(backend))
            try:
                from spectator_cache import export, lookup
                from spectator import physical_fingerprint
                parent_cache = lookup(ROOT/'build/spectator_cache', task)
                if parent_cache is None:
                    raise ValueError('PARENT_CAMERA_CACHE_UNAVAILABLE')
                camera = SimpleNamespace(home=parent_cache[1]['home'])
                fingerprint = physical_fingerprint(backend)
                tick = backend._sim.current_time_step_index
                cache_root, _ = export(ROOT/'build/spectator_cache', variant, backend, camera,
                    dict(physics_tick=tick, initial_physical_sha256=fingerprint))
                if physical_fingerprint(backend) != fingerprint or backend._sim.current_time_step_index != tick:
                    raise ValueError('CACHE_EXPORT_CHANGED_NATIVE_WORLD')
                report['cache'] = str(cache_root)
            except Exception as error:
                report['cache_error'] = str(error)
            (output/'human_ready.json').write_text(json.dumps(dict(tasks=[variant]), indent=2))
    except Exception as error:
        import traceback
        traceback.print_exc(); report['error'] = str(error); report['status'] = 'failed'
        report['exception_type'] = type(error).__name__
    finally:
        (output/('restore_validation.json' if args.restore else 'validation.json')).write_text(json.dumps(report, indent=2))
    print('BACKGROUND_RESULT', report['status'], report.get('error', ''), flush=True)
    # Kit shutdown can hang; all evidence is on disk before exiting.
    os._exit(0 if report['status'] != 'failed' else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--asset-run', type=Path, default=ROOT/'build/search_1k_20261115')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--observer', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.plan = args.plan.resolve(); args.output = args.output.resolve()
    from spectator import select_task
    task = select_task(args.asset_run.resolve(), args.task)
    plan = json.loads(args.plan.read_text())
    if plan['parent_task_id'] != task['parent_task_id'] or not plan.get('semantic_review'):
        parser.error('A per-task semantic review is required; no blind decoration of all tasks')
    runtime = Path(task['runtime_root'])
    for key in list(os.environ):
        if key.startswith('YGBENCH_'):
            del os.environ[key]
    os.environ.update(task.get('runtime_environment', {}))
    os.environ.update(RUMMAGEBENCH_ROOT=str(runtime), OMNIGIBSON_HEADLESS='True')
    if 'OMNIGIBSON_DATA_PATH' not in os.environ:
        for parent in ROOT.parents:
            if (parent/'behavior-datasets/behavior-1k-assets').is_dir():
                os.environ['OMNIGIBSON_DATA_PATH'] = str(parent/'behavior-datasets')
                break
    os.environ.setdefault('CXX', str(Path(sys.prefix)/'bin/x86_64-conda-linux-gnu-c++'))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    sys.path.insert(0, str(runtime/'src')); os.chdir(runtime)
    native(args, task, plan)


if __name__ == '__main__':
    main()
