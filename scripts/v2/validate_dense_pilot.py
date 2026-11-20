"""Private physical initialization audit. Does not claim task solvability.

One native scene load per process, same object topology, clean pre-placement
state per recipe. --watch retries edited recipes without restarting Kit.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import time
import traceback


def root_speed(obj, kind):
    # A fixed base has no dynamic velocity; pose convergence is still checked.
    root = obj.root_link
    getter = getattr(root, 'get_' + kind + '_velocity', None)
    if callable(getter):
        return float(getter().norm())
    if root.__class__.__name__ == 'RigidKinematicPrim':
        return 0.0
    raise TypeError('Unsupported root velocity API: ' + root.__class__.__name__)


def pose_motion_metrics(poses,dt):
    """Measure actual native motion, including between-frame oscillations."""
    import numpy as np
    values=np.asarray(poses,float)
    if values.ndim!=2 or values.shape[1]!=7 or len(values)<2 or not np.isfinite(values).all() or not np.isfinite(dt) or not dt>0:
        raise ValueError('INVALID_NATIVE_POSE_TRAJECTORY')
    positions=values[:,:3];quaternions=values[:,3:]
    lengths=np.linalg.norm(quaternions,axis=1)
    if np.any(lengths<1e-12):raise ValueError('INVALID_NATIVE_POSE_QUATERNION')
    quaternions=quaternions/lengths[:,None]
    relative=2*np.arccos(np.clip(np.abs(quaternions@quaternions[0]),0,1))
    increments=2*np.arccos(np.clip(np.abs((quaternions[1:]*quaternions[:-1]).sum(1)),0,1))
    return dict(max_displacement_m=float(np.linalg.norm(positions-positions[0],axis=1).max()),
        max_rotation_rad=float(relative.max()),
        max_linear_speed=float((np.linalg.norm(np.diff(positions,axis=0),axis=1)/dt).max()),
        max_angular_speed=float((increments/dt).max()))

def observe_native_motion(backend,spec,destination,steps=240):
    """Direct 120-Hz motion check; never move, freeze or zero any actor.

    The same .01 m/s, .1 rad/s, 1-mm and .01-rad tolerances apply to motion
    measured from independent native transforms, rather than a residual solver
    velocity buffer. All native predicates still run after this observation.
    """
    import numpy as np
    sim=backend._sim;dt=float(sim.get_physics_dt())
    names=sorted({p.entity for p in spec.placements})
    objects={name:backend._env.scene.object_registry('name',name) for name in names}
    kinematic={name for name,obj in objects.items() if
               getattr(obj,'kinematic_only',False) or obj.root_link.__class__.__name__=='RigidKinematicPrim'}
    views={name:sim.physics_sim_view.create_rigid_body_view(obj.root_link.prim_path)
           for name,obj in objects.items() if name not in kinematic}
    if any(view.count!=1 for view in views.values()):raise ValueError('NATIVE_MOTION_VIEW_IDENTITY_MISMATCH')
    relevant={p.receptacle for p in spec.placements}|set(spec.initial_states)
    joint_objects={}
    for name in relevant:
        obj=backend._env.scene.object_registry('name',name)
        if getattr(obj,'n_joints',0):joint_objects[name]=obj
    def array(value):return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.asarray(value).copy()
    frames=[]
    for index in range(steps+1):
        rows={name:dict(pose=array(view.get_transforms())[0].tolist(),reported_velocity=array(view.get_velocities())[0].tolist())
              for name,view in views.items()}
        for name in kinematic:
            position,orientation=objects[name].get_position_orientation()
            rows[name]=dict(pose=[*array(position).tolist(),*array(orientation).tolist()],reported_velocity=None,
                            source='native_fixed_geometry_transform_no_dynamic_velocity_interface')
        joints={name:array(obj.get_joint_positions()).tolist() for name,obj in joint_objects.items()}
        frames.append(dict(step=index,objects=rows,joint_positions=joints))
        if index<steps:sim.step_physics()
    metrics={};converged=True
    for name in names:
        value=pose_motion_metrics([frame['objects'][name]['pose'] for frame in frames],dt)
        reported=None if name in kinematic else np.asarray([frame['objects'][name]['reported_velocity'] for frame in frames])
        value.update(native_reported_linear_speed_max=None if reported is None else float(np.linalg.norm(reported[:,:3],axis=1).max()),
                     native_reported_angular_speed_max=None if reported is None else float(np.linalg.norm(reported[:,3:],axis=1).max()),
                     motion_source='native_fixed_geometry_transform' if name in kinematic else 'independent_native_rigid_body_view')
        value['converged']=bool((reported is None or np.isfinite(reported).all()) and value['max_displacement_m']<=.001
            and value['max_rotation_rad']<=.01 and value['max_linear_speed']<=.01 and value['max_angular_speed']<=.1)
        converged &= value['converged'];metrics[name]=value
    joint_metrics={}
    for name,obj in joint_objects.items():
        positions=np.asarray([frame['joint_positions'][name] for frame in frames])
        ranges=np.asarray([max(1e-6,abs(float(j.upper_limit)-float(j.lower_limit))) for j in obj.joints.values()])
        if positions.shape[1:]!=ranges.shape:raise ValueError('NATIVE_MOTION_JOINT_TOPOLOGY_UNSUPPORTED')
        excursion=float((abs(positions-positions[0])/ranges).max())
        joint_metrics[name]=dict(max_normalized_excursion=excursion,converged=bool(np.isfinite(positions).all() and excursion<=.005))
        converged &= joint_metrics[name]['converged']
    report=dict(method='independent_native_transforms_every_physics_substep',converged=bool(converged),
        physics_steps=steps,physics_dt=dt,nominal_duration_s=steps*dt,
        thresholds=dict(linear_speed_max=.01,angular_speed_max=.1,displacement_max_m=.001,
                        rotation_max_rad=.01,normalized_joint_excursion_max=.005),
        objects=metrics,joints=joint_metrics,frames=frames)
    (destination/'native_motion_convergence.private.json').write_text(json.dumps(report,indent=2))
    return {k:v for k,v in report.items() if k!='frames'}


def observe_convergence(backend,spec,destination):
    """Check raw convergence, then independent motion without idle renders.

    A residual velocity mismatch need not consume all 60 render windows before
    we run the existing 240-substep native-transform check. A failed early
    observation continues the original full wait and final check. No actor is
    moved/frozen and no tolerance, contact, membership or spawn gate changes.
    """
    stability=[];quiet=0;previous={};native_motion=None;early_attempt=None
    for window in range(61):
        rows={}
        for ps in spec.placements:
            obj=backend._env.scene.object_registry('name',ps.entity)
            pos,quat=obj.get_position_orientation()
            import numpy as np
            xyz=np.array([float(v) for v in pos]);q=np.array([float(v) for v in quat])
            old=previous.get(ps.entity)
            rows[ps.entity]=dict(linear_speed=root_speed(obj, "linear"),
                angular_speed=root_speed(obj, "angular"),
                displacement_m=float(np.linalg.norm(xyz-old[0])) if old else None,
                rotation_rad=float(2*np.arccos(np.clip(abs(np.dot(q,old[1])),0,1))) if old else None)
            rows[ps.entity]['link_sleep_states'] = {}
            for link_name, link in obj.links.items():
                try:
                    rows[ps.entity]['link_sleep_states'][link_name] = bool(link.is_asleep)
                except Exception as exc:
                    rows[ps.entity]['link_sleep_states'][link_name] = type(exc).__name__
            previous[ps.entity]=(xyz,q)
        stable=all(v['linear_speed']<=.01 and v['angular_speed']<=.1
            and v['displacement_m'] is not None and v['displacement_m']<=.001
            and v['rotation_rad']<=.01 for v in rows.values())
        quiet=quiet+1 if stable else 0
        stability.append(dict(extra_steps=window*30,stable=stable,objects=rows))
        (destination/'settling.private.json').write_text(json.dumps(stability,indent=2))
        print('SETTLING',window,'quiet_windows',quiet,flush=True)
        if quiet>=3:break
        if window==3 and os.environ.get('YGBENCH_NATIVE_MOTION_CONVERGENCE')=='1':
            early_attempt=observe_native_motion(backend,spec,destination)
            if early_attempt['converged']:
                native_motion=early_attempt;break
        if window<60:backend.settle(30)
    result=dict(converged=quiet>=3,extra_steps=window*30,
        linear_speed_max=.01,angular_speed_max=.1,displacement_max_m=.001,
        rotation_max_rad=.01,required_consecutive_windows=3)
    if early_attempt is not None:result['early_native_motion_attempt']=early_attempt
    result['raw_velocity_windows']=window+1
    if quiet<3 and native_motion is None and os.environ.get('YGBENCH_NATIVE_MOTION_CONVERGENCE')=='1':
        native_motion=observe_native_motion(backend,spec,destination)
    if native_motion is not None:
        result['direct_native_motion']=native_motion
        result['converged']=native_motion['converged']
        result['measurement']='native_transforms_at_every_physics_substep'
    if quiet<3 and not (native_motion and native_motion['converged']):
        raise ValueError('LAYOUT_NOT_CONVERGED: see settling.private.json')
    return result,native_motion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene');group.add_argument('--task')
    p.add_argument('--watch',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--restore',action='store_true',help='Verify the exported frozen scene in a fresh simulator.')
    p.add_argument('--catalog-root',type=Path,default=Path('scenarios/dense_search_pilot'))
    p.add_argument('--output-root',type=Path,default=Path('runs/dense_search_pilot'))
    runtime_sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        ['src/rummagebench/sim/omnigibson/backend.py','src/rummagebench/sim/omnigibson/inside_volume.py','src/rummagebench/evaluation/goal_checker.py','scripts/v2/validate_dense_pilot.py',
         'src/rummagebench/core/scenario.py','src/rummagebench/authoring/tasks/household_state.py','src/rummagebench/sim/omnigibson/state_io.py']}
    a=p.parse_args();root=a.catalog_root;out=a.output_root;out.mkdir(parents=True,exist_ok=True)
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.state_io import dump_state,load_state,seed_everything,save_snapshot
    from rummagebench.sim.omnigibson import inside_volume
    paths=[path for path in sorted(root.glob('*/scenario.yaml')) if
           (load_scenario(path).id==a.task if a.task else load_scenario(path).scene.model==a.scene)]
    if not paths:raise ValueError('No matching recipe')
    original_paths=list(paths)
    if a.restore:paths=[out/load_scenario(path).id/'play.yaml' for path in paths]
    topology=lambda path:[(o.name,o.category,o.model) for o in load_scenario(path).objects]
    if any(topology(path)!=topology(paths[0]) for path in paths):
        raise ValueError('Scene recipes have different object topologies; validate one --task per native process.')
    a.scene=load_scenario(paths[0]).scene.model
    spec=load_scenario(paths[0])
    from rummagebench.adapters.python_api import prepare_renderer_grounding,install_renderer_grounding
    prepare_renderer_grounding(spec)
    b=OmniGibsonBackend(seed=a.seed);saved=[]
    def capture(backend):
        saved.append(dump_state(backend._sim))
        rows={}
        for obj in backend._env.scene.objects:
            cv=lambda t:[float(x) for x in t]
            rows[obj.name]=dict(aabb=[cv(x) for x in obj.aabb],volumes={n:[cv(x) for x in l.visual_aabb]
                for n,l in obj.links.items() if l.is_meta_link and 'fillable' in n})
        geometry_dir=out/spec.id;geometry_dir.mkdir(exist_ok=True)
        (geometry_dir/'geometry.private.json').write_text(json.dumps(rows,indent=2))
        return {'private_geometry':str(geometry_dir/'geometry.private.json')}
    fresh_build=None
    try:fresh_build=b.setup(spec,**({} if a.restore else {'pre_placement_capture':capture}))
    except Exception as e:
        traceback.print_exc()
        if not a.watch:
            dest=out/spec.id;dest.mkdir(exist_ok=True)
            (dest/('restore_validation.json' if a.restore else 'validation.json')).write_text(json.dumps(dict(id=spec.id,scenario_sha256=hashlib.sha256(original_paths[0].read_bytes()).hexdigest(),
                status='failed',stage='fresh_native_setup',error=str(e),full_skill_completion='unverified',runtime_sources=runtime_sources),indent=2))
            return
        if not saved:raise
    install_renderer_grounding(b)
    seen={}
    while True:
        for path in paths:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if seen.get(str(path))==digest:continue
            seen[str(path)]=digest;spec=load_scenario(path)
            # Export the effective runtime camera configuration. Reloading raw
            # YAML here used to undo prepare_renderer_grounding(include=[]),
            # causing the restored adapter to change the scenario identity.
            prepare_renderer_grounding(spec)
            report=dict(id=spec.id,scenario_sha256=hashlib.sha256(original_paths[paths.index(path)].read_bytes()).hexdigest(),seed=a.seed,
                status='failed',full_skill_completion='unverified',runtime_sources=runtime_sources)
            if a.restore:
                report['frozen_manifest_sha256']=hashlib.sha256(Path(spec.scene.frozen_member_manifest).read_bytes()).hexdigest()
            dest=out/spec.id;dest.mkdir(exist_ok=True)
            print('VALIDATE',spec.id,flush=True)
            try:
                importlib.reload(inside_volume)
                if fresh_build is not None:
                    report['build']=fresh_build;fresh_build=None;report['fresh_native_setup']=True
                else:
                    seed_everything(0);load_state(b._sim,saved[0]);b._scenario=spec;b._collision_body_cache=None
                    report['build']=b._apply_placements(spec,attempts=1);report['fresh_native_setup']=False
                b.teleport_robot(spec.anchors[spec.robot.init_anchor]);b.settle(60)
                # Observe physical convergence over time, never freeze contents or
                # zero velocities to make an unstable layout pass.
                report['settling'],native_motion=observe_convergence(b,spec,dest)
                b.validate_physics_state();b._validate_spawn_clearance()
                from types import SimpleNamespace
                from rummagebench.evaluation.goal_checker import check_goal
                initial_goal=check_goal(SimpleNamespace(held_object=None),spec,b)
                report['initial_goal']=dict(satisfied=initial_goal.satisfied,fault=initial_goal.fault,predicates=initial_goal.predicates)
                if initial_goal.satisfied or initial_goal.fault:raise ValueError('INVALID_INITIAL_GOAL')
                from omnigibson.object_states import Inside,OnTop,Open
                baseline={}
                for ps in spec.placements:
                    obj=b._env.scene.object_registry('name',ps.entity);rec=b._env.scene.object_registry('name',ps.receptacle)
                    if not b._verify_relation(obj,rec,ps.relation,Inside,OnTop) or not b._in_link(obj,rec,ps.link):
                        raise ValueError('POST_SETTLE_RELATION_FAILED: '+ps.entity)
                    pos,quat=obj.get_position_orientation()
                    baseline[ps.entity]=dict(position=[float(v) for v in pos],orientation=[float(v) for v in quat],
                        relation=ps.relation,receptacle=ps.receptacle,link=ps.link,
                        linear_speed=root_speed(obj, "linear"))
                    if native_motion is not None:
                        baseline[ps.entity]['native_reported_linear_speed']=baseline[ps.entity]['linear_speed']
                        baseline[ps.entity]['linear_speed']=native_motion['objects'][ps.entity]['max_linear_speed']
                        baseline[ps.entity]['linear_speed_source']='native_pose_finite_difference_every_physics_substep'
                    if baseline[ps.entity]['linear_speed']>.01:raise ValueError('UNSTABLE: '+ps.entity)
                for name,state in spec.initial_states.items():
                    obj=b._env.scene.object_registry('name',name)
                    if bool(obj.states[Open].get_value())!=state.open:raise ValueError('OPEN_STATE_CHANGED: '+name)
                    baseline[name]=dict(open=state.open,joint_positions=[float(v) for v in obj.get_joint_positions()])
                (dest/'restoration_baseline.private.json').write_text(json.dumps(baseline,indent=2))
                if not a.restore:
                    from rummagebench.authoring.tasks.household_state import export_native_member
                    export_native_member(b,spec,dest,report['build'],runtime_sources)
                from PIL import Image
                Image.fromarray(b.capture_panorama_frame().rgb).save(dest/('restored_panorama.png' if a.restore else 'initial_panorama.png'))
                report['status']='restore_validated' if a.restore else 'initialization_validated' if report['fresh_native_setup'] else 'authoring_replay_validated'
                report['human_review']='pending'
            except Exception as e:
                report['error']=str(e);traceback.print_exc()
            (dest/('restore_validation.json' if a.restore else 'validation.json')).write_text(json.dumps(report,indent=2))
            print('RESULT',spec.id,report['status'],report.get('error',''),flush=True)
        if not a.watch or (out/(a.scene+'.stop')).exists():break
        time.sleep(2)

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();os._exit(1)
    os._exit(0)
