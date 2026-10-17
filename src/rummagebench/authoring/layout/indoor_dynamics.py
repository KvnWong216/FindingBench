"""Bounded measurements and explicitly requested solver experiments.

The trace itself only steps/reads and never upgrades a rejected background.
The separate iteration experiment requires an explicit diagnostic CLI option.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.native_occupancy import plain
from rummagebench.authoring.tasks.plan import sha256_path
from rummagebench.sim.omnigibson.entity_resolver import resolve_object
from rummagebench.sim.omnigibson.probe_support import relation_holds


def apply_velocity_iteration_experiment(backend, manifest, output, iterations):
    """Explicit diagnostic override, never a production physics default.

    Only added dynamic props; native occupants, robots, target, masses,
    damping, contact parameters and sleep behavior are untouched.
    """
    if type(iterations) is not int or iterations not in (1, 4):
        raise ValueError('UNSUPPORTED_DIAGNOSTIC_VELOCITY_ITERATIONS')
    objects = [resolve_object(backend._env.scene, i['placement']['name'])
               for i in manifest['objects'] if i['placement']['role']=='clutter']
    if any(o.kinematic_only for o in objects):
        raise ValueError('ITERATION_EXPERIMENT_REQUIRES_DYNAMIC_RIGID_PROPS')
    # Native shakers have multiple links. OG's entity-level property selects
    # the articulation solver when appropriate, or all rigid links otherwise.
    records = {o.name:dict(before=int(o.solver_velocity_iteration_count),articulated=bool(o.articulated)) for o in objects}
    # SDK property setters own the USD-edit context (including rigid links).
    # Opening another context here is forbidden by this installed SDK.
    for obj in objects: obj.solver_velocity_iteration_count=iterations
    for obj in objects:
        records[obj.name]['after']=int(obj.solver_velocity_iteration_count)
        if records[obj.name]['after'] != iterations: raise ValueError('VELOCITY_ITERATION_OVERRIDE_FAILED')
    report = dict(scope='diagnostic physics override; not a dataset profile',
                  assembly_sha256=manifest['manifest_sha256'], iterations=iterations, objects=records)
    atomic_write_json(output/'background_physics_experiment.json', report)
    return report


def summarize_trace(rows):
    times = np.asarray([r['sim_time'] for r in rows], float)
    if len(rows) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('DYNAMICS_TIME_MUST_ADVANCE')
    summaries = {}
    names = set(rows[0]['objects'])
    if any(set(r['objects']) != names for r in rows):
        raise ValueError('DYNAMICS_ENTITY_SET_CHANGED')
    for name in sorted(names):
        samples = [r['objects'][name] for r in rows]
        poses = np.asarray([s['pose'] for s in samples], float)
        lin = np.asarray([s['linear_velocity'] for s in samples], float)
        ang = np.asarray([s['angular_velocity'] for s in samples], float)
        if (poses.shape != (len(rows), 7) or lin.shape != (len(rows), 3)
                or ang.shape != lin.shape or not np.isfinite(np.c_[poses, lin, ang]).all()):
            raise ValueError('INVALID_DYNAMICS_SAMPLES')
        q = poses[:, 3:]
        norms = np.linalg.norm(q, axis=1)
        if not np.allclose(norms, 1., atol=1e-3): raise ValueError('INVALID_DYNAMICS_QUATERNION')
        q = q / norms[:, None]
        # atan2 avoids arccos precision loss for tiny float32 rotations.
        def angles(a, b):
            aligned = b * np.where(np.sum(a*b, axis=1) < 0, -1., 1.)[:, None]
            return 4*np.arctan2(np.linalg.norm(a-aligned, axis=1), np.linalg.norm(a+aligned, axis=1))
        rotation = angles(np.broadcast_to(q[0], q.shape), q)
        angular_fd = angles(q[:-1], q[1:]) / np.diff(times)
        linear_fd = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / np.diff(times)
        spin, speed = np.linalg.norm(ang, axis=1), np.linalg.norm(lin, axis=1)
        valid = [bool(s['physics_handle_valid']) for s in samples]
        raw_deltas = [float(np.max(np.abs(np.asarray(s['physx_velocity'])-np.r_[s['linear_velocity'],s['angular_velocity']])))
                      for s in samples if s['physx_velocity'] is not None]
        summaries[name] = dict(samples=len(samples), duration_seconds=float(times[-1]-times[0]),
            max_position_drift_m=float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max()),
            max_rotation_drift_deg=float(np.degrees(rotation).max()),
            angular_speed_max_rad_s=float(spin.max()), angular_speed_p95_rad_s=float(np.quantile(spin, .95)),
            angular_exceedances=int((spin > .02).sum()), linear_exceedances=int((speed > .01).sum()),
            finite_difference_angular_max_rad_s=float(angular_fd.max()),
            finite_difference_linear_max_m_s=float(linear_fd.max()),
            invalid_handle_samples=valid.count(False), asleep_samples=sum(bool(s['asleep']) for s in samples),
            support_lost_samples=sum(not s['supported'] for s in samples),
            getter_physx_max_abs_difference=max(raw_deltas) if raw_deltas else None)
    return dict(duration_seconds=float(times[-1]-times[0]), objects=summaries,
                scope='diagnostic only; finite differences are not a replacement acceptance metric')


def diagnose_background(backend, manifest, output: Path, *, steps=360, render=True):
    if type(steps) is not int or not 2 <= steps <= 600: raise ValueError('BOUNDED_DYNAMICS_TRACE_REQUIRED')
    placements = [i['placement'] for i in manifest['objects'] if i['placement']['role']=='clutter']
    objects = {p['name']: resolve_object(backend._env.scene, p['name']) for p in placements}
    report = dict(status='running', assembly_sha256=manifest['manifest_sha256'], steps=steps,
                  physics_dt=float(backend._sim.get_physics_dt()), sim_step_dt=float(backend._sim.get_sim_step_dt()),
                  evaluation_ready_episodes=0, scope='post-rejection diagnostic; verdict remains rejected')
    report['body_parameters'] = {}
    for name, obj in objects.items():
        parameters = {}
        for key in ('mass','sleep_threshold','stabilization_threshold',
                    'solver_position_iteration_count','solver_velocity_iteration_count'):
            try: parameters[key] = plain(getattr(obj.root_link, key))
            except Exception as exc: parameters[key] = dict(unavailable=f'{type(exc).__name__}:{exc}')
        report['body_parameters'][name] = parameters
        # Root-link settings can differ from effective articulation settings.
        report['body_parameters'][name]['entity_parameters'] = dict(
            articulated=bool(obj.articulated),solver_position_iteration_count=int(obj.solver_position_iteration_count),
            solver_velocity_iteration_count=int(obj.solver_velocity_iteration_count),
            sleep_threshold=float(obj.sleep_threshold), stabilization_threshold=float(obj.stabilization_threshold))
    path = output/'background_dynamics.json'
    trace = output/'background_dynamics.jsonl'
    atomic_write_json(path, report)
    rows=[]
    try:
        with trace.open('x') as stream:
            for index in range(steps+1):
                if index: backend.settle(1)
                row = dict(step=index, sim_time=float(backend._sim.current_time), objects={})
                for p in placements:
                    obj = objects[p['name']]
                    position, orientation = plain(obj.get_position_orientation())
                    view = obj.root_link._rigid_prim_view
                    valid = bool(view.is_physics_handle_valid())
                    row['objects'][p['name']] = dict(pose=[*position, *orientation],
                        linear_velocity=plain(obj.get_linear_velocity()), angular_velocity=plain(obj.get_angular_velocity()),
                        physics_handle_valid=valid,
                        physx_pose=plain(view._physics_view.get_transforms()[0]) if valid else None,
                        physx_velocity=plain(view._physics_view.get_velocities()[0]) if valid else None,
                        asleep=bool(obj.is_asleep), supported=relation_holds(backend,p['name'],p['support_name'],'on_top'))
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); rows.append(row)
        report.update(status='diagnostic_completed_not_certified', summary=summarize_trace(rows))
        # Render only after time-series capture, so rendering cannot influence
        # the recorded measurements. Image failure cannot erase physics data.
        try:
            if render:
                backend.render_snapshot(str(output/'background_diagnostic.png'))
                report['image_sha256']=sha256_path(output/'background_diagnostic.png')
        except Exception as exc:
            report['image_error']=f'{type(exc).__name__}:{exc}'
    except Exception as exc:
        report.update(status='diagnostic_failed', error=f'{type(exc).__name__}:{exc}')
        raise
    finally:
        if trace.exists(): report['trace_sha256']=sha256_path(trace)
        atomic_write_json(path, report)
    return report


def motion_controls(backend, manifest, output, *, physics_steps=False):
    """Destructive diagnostic in a restored private state; never a data episode.

    All added props are observed. The first bowl and shaker receive independent
    slide / tipped-drop interventions, each starting from the same checkpoint.
    No physics parameters, sleeping flags or acceptance thresholds are changed.
    """
    import torch
    from scipy.spatial.transform import Rotation
    selected = {}
    for item in manifest['objects']:
        p = item['placement']
        if p['role'] != 'clutter': continue
        obj = resolve_object(backend._env.scene, p['name'])
        if obj.category in ('bowl', 'salt_shaker'):
            selected.setdefault(obj.category, obj)
    if set(selected) != {'bowl', 'salt_shaker'}:
        raise ValueError('MOTION_CONTROLS_REQUIRE_BOWL_AND_SHAKER')
    baseline = backend.capture_observe_state()
    report = dict(status='running', scope='intervened diagnostic; never dataset evidence',
                  assembly_sha256=manifest['manifest_sha256'], evaluation_ready_episodes=0, cases={})
    output.mkdir(parents=True, exist_ok=False)
    path = output/'controls.json'
    atomic_write_json(path, report)
    try:
        for category, obj in selected.items():
            for mode in ('stationary', 'slide', 'tipped_drop'):
                backend.restore_observe_state(baseline)
                intervention = dict(entity=obj.name, category=category, mode=mode)
                if mode == 'slide':
                    obj.set_linear_velocity(torch.tensor([1., 0., 0.]))
                    intervention['linear_velocity_m_s'] = [1., 0., 0.]
                elif mode == 'tipped_drop':
                    pos, quat = plain(obj.get_position_orientation())
                    pos[2] += .15
                    quat = (Rotation.from_rotvec([.8, 0., 0.])*Rotation.from_quat(quat)).as_quat().tolist()
                    obj.set_position_orientation(pos, quat)
                    intervention.update(lift_m=.15, rotation_rad=.8)
                folder = output/f'{category}_{mode}'
                folder.mkdir()
                if physics_steps:
                    from .indoor_stability import measure_background_motion
                    placements = [i['placement'] for i in manifest['objects'] if i['placement']['role']=='clutter']
                    measured, trace = measure_background_motion(backend, placements)
                    atomic_write_json(folder/'physics_motion.json', trace)
                    report['cases'][folder.name] = dict(intervention=intervention, measurements=measured,
                        trace_sha256=sha256_path(folder/'physics_motion.json'))
                else:
                    result = diagnose_background(backend, manifest, folder, steps=90, render=False)
                    report['cases'][folder.name] = dict(intervention=intervention,
                        summary=result['summary'], trace_sha256=result['trace_sha256'])
                atomic_write_json(path, report)
        report['status'] = 'controls_completed_not_certified'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}:{exc}')
        raise
    finally:
        backend.restore_observe_state(baseline)
        atomic_write_json(path, report)
    return report
