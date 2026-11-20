"""Native physical realization of Gen proposals before placing the target.

Only original approved support relations are retried. No native furniture is
moved explicitly, no asset is replaced, and failed checks stay failed. This
single-background stage is not a matched-family or door-clearance certificate.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import numpy as np

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.household_physics import placement_stability
from rummagebench.authoring.tasks.native_occupancy import plain
from rummagebench.authoring.tasks.plan import sha256_path
from rummagebench.feasibility.ik_solver import quat_rotate
from rummagebench.sim.omnigibson.entity_resolver import resolve_object
from rummagebench.sim.omnigibson.probe_support import place_entity, relation_holds


def motion_accepted(measurement, policy='strict-v3'):
    """Execution acceptance keeps support/pose bounds; peak speed is diagnostic.

    Never change the recorded strict verdict or the simulator's dynamics.
    """
    if policy=='strict-v3':return measurement['passed']
    if policy!='execution-v1':raise ValueError('UNKNOWN_BACKGROUND_ACCEPTANCE_POLICY')
    limits=measurement['limits']
    return (np.isfinite([measurement[k] for k in ('position_drift_m','rotation_drift_deg',
                'linear_speed_m_s','angular_speed_rad_s')]).all()
            and measurement['position_drift_m']<=limits['position_m']
            and measurement['rotation_drift_deg']<=limits['rotation_deg']
            and all(measurement.get('support_samples',[True])))


def parking_position(manifest):
    corners=[]
    for o in manifest['living_context']['native']['objects']:
        for signs in product((-1,1),repeat=3):
            corners.append(np.asarray(o['center'])+quat_rotate(o['quaternion'],np.asarray(o['size'])*signs/2))
    return (np.max(corners,axis=0)+5.).tolist()


def _measurement(backend, placements, *, profile='legacy'):
    if profile == 'observed-motion-v3':
        from .indoor_stability import measure_background_motion
        measurements, trace = measure_background_motion(backend, placements)
        for name, result in measurements.items():
            result['samples'] = dict(physics_dt=trace['physics_dt'],
                time=[s['time'] for s in trace['samples']],
                bodies=[s['bodies'][name] for s in trace['samples']])
        return measurements
    if profile != 'legacy': raise ValueError('UNKNOWN_BACKGROUND_STABILITY_PROFILE')
    objects={p['name']:resolve_object(backend._env.scene,p['name']) for p in placements}
    poses={name:[] for name in objects}
    linear={name:[] for name in objects}; angular={name:[] for name in objects}
    # Same 24 normal simulator steps per window, but inspect every step.
    # The old three-pose/final-velocity sample could miss intermediate spikes.
    for _ in range(24):
        backend.settle(1)
        for name,obj in objects.items():
            pos,ori=plain(obj.get_position_orientation());poses[name].append([*pos,*ori])
            linear[name].append(plain(obj.get_linear_velocity()))
            angular[name].append(plain(obj.get_angular_velocity()))
    results={}
    for name in objects:
        # Norm maxima enforce the unchanged limits at every sampled step.
        lv=np.asarray(linear[name]); av=np.asarray(angular[name])
        result=placement_stability(poses[name],lv[np.linalg.norm(lv,axis=1).argmax()],
                                   av[np.linalg.norm(av,axis=1).argmax()])
        result['samples']=dict(poses=poses[name],linear_velocity=linear[name],angular_velocity=angular[name],
                               simulator_steps_between_samples=1)
        results[name]=result
    return results


def _settled_measurement(backend, placements, *, profile='legacy'):
    """Bounded settling before resampling; never lower stability limits."""
    history=[]
    for window in range(3):
        measures=(_measurement(backend,placements) if profile=='legacy'
                  else _measurement(backend,placements,profile=profile))
        supports={p['name']:relation_holds(backend,p['name'],p['support_name'],'on_top')
                  for p in placements}
        history.append(dict(window=window,
            unstable=[n for n,m in measures.items() if not m['passed']],
            unsupported=[n for n,ok in supports.items() if not ok]))
        if all(m['passed'] for m in measures.values()) and all(supports.values()):break
    return measures,supports,history


def realize_background(backend, manifest: dict, output: Path, *, max_passes=3, diagnostic_steps=0,
                       stability_profile='legacy', acceptance_policy='strict-v3'):
    from rummagebench.sim.omnigibson.state_io import seed_everything, save_snapshot
    if not 1<=max_passes<=3:raise ValueError('BOUNDED_BACKGROUND_REALIZATION_REQUIRED')
    if diagnostic_steps not in (0, 360):raise ValueError('BOUNDED_DYNAMICS_TRACE_REQUIRED')
    if stability_profile not in ('legacy', 'observed-motion-v3'):
        raise ValueError('UNKNOWN_BACKGROUND_STABILITY_PROFILE')
    if acceptance_policy not in ('strict-v3','execution-v1'):
        raise ValueError('UNKNOWN_BACKGROUND_ACCEPTANCE_POLICY')
    # Reset the authoring stream after asset import. Hidden target location is
    # absent from this input and its body is parked outside native geometry.
    seed_everything(manifest['layout_seed'])
    placements=[o['placement'] for o in manifest['objects'] if o['placement']['role']=='clutter']
    report=dict(status='running',assembly_sha256=manifest['manifest_sha256'],repairs=[],settling=[],
                acceptance_policy=acceptance_policy,
                stability_sampling=('24-normal-steps-all-velocities-v2' if stability_profile=='legacy'
                                    else 'observed-motion-physics-steps-v3'),
                stability_profile=stability_profile,
                scope='native support and short-horizon stability; family/collision-sweep checks pending')
    path=output/'background_realization.json'
    atomic_write_json(path,report)
    try:
        for pass_index in range(max_passes+1):
            measures,supports,history=(_settled_measurement(backend,placements) if stability_profile=='legacy'
                else _settled_measurement(backend,placements,profile=stability_profile))
            report['settling'].append(dict(pass_index=pass_index,windows=history))
            bad=[p for p in placements if not motion_accepted(measures[p['name']],acceptance_policy)
                 or not supports[p['name']]]
            report['measurements']=measures
            report['support_checks']=supports
            report['failed_entities']=[p['name'] for p in bad]
            report['motion_diagnostic_entities']=[n for n,m in measures.items() if not m['passed']]
            atomic_write_json(path,report)
            if not bad:break
            if pass_index==max_passes:raise ValueError('BACKGROUND_NATIVE_REALIZATION_FAILED:'+','.join(report['failed_entities']))
            # A velocity threshold alone is not evidence that a valid support
            # location should be replaced. Repeated sampling can destabilize
            # neighboring props; reject persistent dynamics instead.
            movable=[p for p in bad if not supports[p['name']]
                or measures[p['name']].get('position_drift_m',0)>.005
                or measures[p['name']].get('rotation_drift_deg',0)>2.]
            if not movable:raise ValueError('BACKGROUND_PERSISTENT_DYNAMICS:'+','.join(report['failed_entities']))
            for p in movable:
                before=plain(resolve_object(backend._env.scene,p['name']).get_position_orientation())
                ok=place_entity(backend,p['name'],p['support_name'],'on_top',None)
                after=plain(resolve_object(backend._env.scene,p['name']).get_position_orientation())
                report['repairs'].append(dict(entity=p['name'],support=p['support_name'],
                    pass_index=pass_index,sampler_returned=ok,before=before,after=after,
                    native_sampler_max_trials=5))
                atomic_write_json(path,report)
        report['actual_poses']={p['name']:plain(resolve_object(backend._env.scene,p['name']).get_position_orientation())
                                for p in placements}
        # Local reconstruction checkpoint, explicitly not a certified frozen
        # family. Later target placement is checked again by the normal smoke.
        checkpoint=output/'realized_background_checkpoint.pt'
        save_snapshot(checkpoint,backend.dump_state(),
                      meta={'assembly_sha256':manifest['manifest_sha256'],'scope':'background_with_parked_target'})
        report['checkpoint_sha256']=sha256_path(checkpoint)
        report['status']=('support_and_pose_passed_motion_diagnostic' if report['motion_diagnostic_entities']
                          else 'support_and_stability_passed_pending_family')
        atomic_write_json(path,report)
        return report
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}:{exc}')
        atomic_write_json(path,report)
        if diagnostic_steps:
            from .indoor_dynamics import diagnose_background
            try:
                diagnose_background(backend,manifest,output,steps=diagnostic_steps)
                report['diagnostics_sha256']=sha256_path(output/'background_dynamics.json')
            except Exception as diagnostic_error:
                report['diagnostic_error']=f'{type(diagnostic_error).__name__}:{diagnostic_error}'
            atomic_write_json(path,report)
        raise


def verify_after_target(backend, manifest, before, output, *, require_matched_background=True):
    """Check current physical support/motion and optionally matched poses.

    Environment production records insertion displacements but accepts a settled
    supported arrangement. Family callers retain strict pose matching by default.
    Native-state and door-clearance certification are separate.
    """
    if (before.get('status') not in ('support_and_stability_passed_pending_family','support_and_pose_passed_motion_diagnostic')
            or before.get('assembly_sha256')!=manifest['manifest_sha256']):
        raise ValueError('BACKGROUND_REALIZATION_BINDING_REQUIRED')
    measurements={}; supports={}; actual={}
    policy=before.get('acceptance_policy','strict-v3')
    if not require_matched_background and before.get('stability_profile')!='observed-motion-v3':
        raise ValueError('ENVIRONMENT_READINESS_REQUIRES_OBSERVED_MOTION')
    placements=[i['placement'] for i in manifest['objects'] if i['placement']['role']=='clutter']
    motion=None
    if before.get('stability_profile')=='observed-motion-v3':
        motion=_measurement(backend,placements,profile='observed-motion-v3')
    for item in manifest['objects']:
        p=item['placement']
        if p['role']!='clutter':continue
        name=p['name'];obj=resolve_object(backend._env.scene,name)
        pos,ori=plain(obj.get_position_orientation());actual[name]=[pos,ori]
        old_pos,old_ori=before['actual_poses'][name]
        measurements[name]=placement_stability([[*old_pos,*old_ori],[*pos,*ori]],
            [0.,0.,0.] if motion is not None else plain(obj.get_linear_velocity()),
            [0.,0.,0.] if motion is not None else plain(obj.get_angular_velocity()))
        if motion is not None:
            measurements[name]['passed'] &= motion_accepted(motion[name],policy)
            measurements[name]['motion_observation']=motion[name]
        supports[name]=relation_holds(backend,name,p['support_name'],'on_top')
    failed=[n for n in measurements if not supports[n] or
            (not measurements[n]['passed'] if require_matched_background else not motion_accepted(motion[n],policy))]
    causes={}
    for name in failed:
        m=measurements[name];limits=m['limits'];reasons=[]
        if not supports[name]:reasons.append('support_relation_lost')
        if m['position_drift_m']>limits['position_m'] or m['rotation_drift_deg']>limits['rotation_deg']:
            reasons.append('pose_changed')
        if m['linear_speed_m_s']>limits['linear_m_s']:reasons.append('linear_velocity')
        if m['angular_speed_rad_s']>limits['angular_rad_s']:reasons.append('angular_velocity')
        if motion is not None and not motion[name]['passed']:reasons.append('observed_motion_or_support_loss')
        causes[name]=reasons
    report=dict(status='failed' if failed else ('daily_props_unchanged_within_tolerance'
                if require_matched_background else 'daily_props_physically_ready'),
        require_matched_background=require_matched_background,
        background_pose_changed_entities=[n for n,m in measurements.items()
            if m['position_drift_m']>m['limits']['position_m'] or m['rotation_drift_deg']>m['limits']['rotation_deg']],
        acceptance_policy=policy,
        motion_diagnostic_entities=[n for n,m in (motion or {}).items() if not m['passed']],
        assembly_sha256=manifest['manifest_sha256'],measurements=measurements,
        support_checks=supports,actual_poses=actual,failed_entities=failed,
        failure_causes=causes,
        scope='daily props only; native states, family matching and door sweeps pending')
    atomic_write_json(output/'background_after_target.json',report)
    if failed:
        moved=any(set(reasons)&{'support_relation_lost','pose_changed'} for reasons in causes.values())
        code='TARGET_INSERTION_DISTURBED_BACKGROUND' if moved else 'BACKGROUND_UNSTABLE_AFTER_TARGET'
        raise ValueError(code+':'+','.join(failed))
    return report


def verify_background_restore(backend, manifest, output, *, acceptance_policy='strict-v3'):
    """Round-trip a real perturbation through a portable private checkpoint.

    Observe every added collision link again after restoration. Native poses
    and joints are compared immediately; this is a reconstruction gate, not a
    door-clearance or matched-target-family certificate.
    """
    from rummagebench.sim.omnigibson.state_io import save_snapshot, load_snapshot
    names=manifest['living_context']['native']['retained_native_objects'] + [
        i['placement']['name'] for i in manifest['objects'] if i['placement']['role']=='clutter']
    objects={n:resolve_object(backend._env.scene,n) for n in names}
    def records():
        return {n:dict(pose=plain(o.get_position_orientation()),
                       joints=plain(o.get_joint_positions()) if o.n_joints else []) for n,o in objects.items()}
    before=records();snapshot=backend.capture_observe_state()
    path=output/'restorable_background.pt'
    save_snapshot(path,snapshot,meta=dict(assembly_sha256=manifest['manifest_sha256'],
        format='observe-state-with-spatial-cache-v1',scope='background_with_parked_target'))
    # Exercise disk serialization as well as in-memory restore. Move one real
    # added prop; no step or native relation query is used to author a new state.
    placements=[i['placement'] for i in manifest['objects'] if i['placement']['role']=='clutter']
    if not placements: raise ValueError('RESTORE_AUDIT_REQUIRES_ADDED_PROP')
    prop=objects[placements[0]['name']]
    pos,ori=plain(prop.get_position_orientation());pos[2]+=.1
    prop.set_position_orientation(pos,ori)
    restored,meta=load_snapshot(path)
    backend.restore_observe_state(restored)
    after=records();changed=[];max_position=0.;max_rotation=0.;max_joint=0.
    for name in names:
        p,q=before[name]['pose'];r,s=after[name]['pose']
        check=placement_stability([[*p,*q],[*r,*s]],[0.,0.,0.],[0.,0.,0.])
        joints=np.asarray(before[name]['joints']);now=np.asarray(after[name]['joints'])
        delta=float(np.max(np.abs(joints-now))) if joints.size and joints.shape==now.shape else (0. if joints.shape==now.shape else float('inf'))
        max_position=max(max_position,check['position_drift_m']);max_rotation=max(max_rotation,check['rotation_drift_deg']);max_joint=max(max_joint,delta)
        if check['position_drift_m']>1e-5 or check['rotation_drift_deg']>1e-3 or delta>1e-5:changed.append(name)
    motion=_measurement(backend,placements,profile='observed-motion-v3')
    failed=[n for n,m in motion.items() if not motion_accepted(m,acceptance_policy)]
    report=dict(status='restore_verified_pending_family' if not changed and not failed else 'failed',
        acceptance_policy=acceptance_policy,
        motion_diagnostic_entities=[n for n,m in motion.items() if not m['passed']],
        assembly_sha256=manifest['manifest_sha256'],checkpoint_sha256=sha256_path(path),
        checkpoint_format=meta['format'],restored_entities=len(names),changed_entities=changed,
        unstable_entities=failed,max_position_error_m=max_position,max_rotation_error_deg=max_rotation,
        max_joint_error=max_joint,measurements=motion,evaluation_ready_episodes=0,
        scope='private reconstruction and 3-second prop observation; family and clearance pending')
    atomic_write_json(output/'background_restore.json',report)
    if changed or failed:raise ValueError('BACKGROUND_RESTORE_FAILED:'+','.join(changed+failed))
    return report
