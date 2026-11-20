"""Authoring-only causal controls. Produces diagnostics, never acceptance."""
from pathlib import Path
import argparse,json,hashlib,traceback
import numpy as np

def motion_delta(before,after):
    a,b=np.asarray(before,dtype=float),np.asarray(after,dtype=float)
    qa,qb=a[3:]/np.linalg.norm(a[3:]),b[3:]/np.linalg.norm(b[3:])
    return dict(translation_m=float(np.linalg.norm(b[:3]-a[:3])),
        rotation_rad=float(2*np.arccos(np.clip(abs(np.dot(qa,qb)),0,1))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['closure','trajectory','relations'],required=True)
    p.add_argument('--scenario',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--relocation',type=Path);p.add_argument('--steps',type=int,default=180)
    p.add_argument('--settle-mode',choices=['default','physics_only','renderer'],default='default')
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from rummagebench.core.scenario import load_scenario
    from rummagebench.adapters.python_api import prepare_renderer_grounding
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.state_io import dump_state,load_state,seed_everything
    spec=load_scenario(args.scenario);prepare_renderer_grounding(spec);seed_everything(0)
    b=OmniGibsonBackend(seed=0)
    doc=dict(mode=args.mode,scenario=str(args.scenario),scenario_sha256=hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='diagnostic_only',rows=[])
    def write():args.output.write_text(json.dumps(doc,indent=2))
    try:
        b.setup(spec,**({'native_scene_only':True,'native_scene_capture':lambda backend:dict(diagnostic=True)} if args.mode=='closure' else {}))
        from omnigibson.utils.usd_utils import RigidContactAPI
        def contacts(obj):
            return [list(pair) for pair in sorted(RigidContactAPI.get_contact_pairs(
                scene_idx=obj.scene.idx,query_set=[obj],with_set=None,current_only=True))]
        if args.mode=='closure':
            from omnigibson.object_states import Open
            config=json.loads(args.relocation.read_text());cab=b._env.scene.object_registry('name',config['entity'])
            saved=dump_state(b._sim);robot_pose=b._robot.get_position_orientation();cab_pose=cab.get_position_orientation()
            for label,away,moved in [('robot_present',False,False),('robot_away',True,False),
                                      ('cabinet_relocated_robot_away',True,True),('cabinet_relocated_robot_present',False,True),
                                      ('robot_present_repeat',False,False)]:
                load_state(b._sim,saved,settle_steps=0)
                rpos=[float(v) for v in robot_pose[0]]
                if away:rpos[:2]=[1000.,1000.]
                b._robot.set_position_orientation(position=rpos,orientation=robot_pose[1])
                cpos=[float(v) for v in cab_pose[0]]
                if moved:cpos[:2]=config['position_xy']
                cab.set_position_orientation(position=cpos,orientation=cab_pose[1])
                opened=b._set_open_carrying(cab,True);b.settle(30)
                row=dict(condition=label,additional_task_contents=0,opened_return=opened,
                         open_joint_positions=[float(v) for v in cab.get_joint_positions()],before_contacts=contacts(cab))
                row['closed_setter_return']=b._set_open_carrying(cab,False)
                row['immediate_open']=bool(cab.states[Open].get_value());row['immediate_joints']=[float(v) for v in cab.get_joint_positions()]
                b.settle(60)
                row.update(final_open=bool(cab.states[Open].get_value()),final_joints=[float(v) for v in cab.get_joint_positions()],
                           final_contacts=contacts(cab),robot_position=[float(v) for v in b._robot.get_position_orientation()[0]])
                doc['rows'].append(row);write();print('[closure-control] '+json.dumps(row),flush=True)
            load_state(b._sim,saved,settle_steps=0)
        elif args.mode=='relations':
            from omnigibson.object_states import Inside,OnTop,Touching,Adjacency
            doc['settle_mode']=args.settle_mode
            if args.settle_mode=='physics_only':
                from rummagebench.sim.omnigibson.raw_instance import install_physics_only_settle
                install_physics_only_settle(b)
            elif args.settle_mode=='renderer':
                from rummagebench.adapters.python_api import install_renderer_grounding
                install_renderer_grounding(b)
            # Match validator's teleport, 60-step warmup and three 30-step
            # convergence windows. Do not prime relative-state caches first.
            b.teleport_robot(spec.anchors[spec.robot.init_anchor]);b.settle(150)
            baseline_path=Path(spec.scene.frozen_member_manifest).parent/'restoration_baseline.private.json'
            baseline=json.loads(baseline_path.read_text())
            doc['initialization_baseline']=baseline
            def cv(value):return [float(v) for v in value]
            for step in range(0,args.steps+1,30):
                if step:b.settle(30)
                sample=dict(step=step,placements=[])
                for placement in spec.placements:
                    obj=b._env.scene.object_registry('name',placement.entity)
                    rec=b._env.scene.object_registry('name',placement.receptacle)
                    row=dict(entity=obj.name,receptacle=rec.name,relation=placement.relation,
                        pose=[cv(v) for v in obj.get_position_orientation()],
                        aabb=[cv(v) for v in obj.aabb],receptacle_aabb=[cv(v) for v in rec.aabb],
                        verified=b._verify_relation(obj,rec,placement.relation,Inside,OnTop),contacts=contacts(obj))
                    for state in (OnTop,Touching,Adjacency):
                        try:
                            value=obj.states[state].get_value(rec)
                            row[state.__name__]=value.tolist() if hasattr(value,'tolist') else value
                        except Exception as exc:row[state.__name__+'_error']=str(exc)
                    sample['placements'].append(row)
                doc['rows'].append(sample);write();print('[relation-control] '+json.dumps(sample),flush=True)
        else:
            doc['physics_dt']=b._sim.get_physics_dt();previous={}
            for step in range(args.steps+1):
                if step:b._sim.step()
                sample=dict(step=step,objects={})
                for o in spec.objects:
                    if o.fixed_base:continue
                    obj=b._env.scene.object_registry('name',o.name);link=obj.root_link
                    pos,quat=link.get_position_orientation();public=[float(v) for v in pos]+[float(v) for v in quat]
                    row=dict(public_link_pose=public,object_pose=[float(v) for v in obj.get_position_orientation()[0]],
                        linear_velocity=[float(v) for v in obj.get_linear_velocity()],angular_velocity=[float(v) for v in obj.get_angular_velocity()])
                    try:
                        view=link._rigid_prim_view._physics_view
                        row['direct_physics_pose']=[float(v) for v in view.get_transforms()[0]]
                        row['direct_physics_velocity']=[float(v) for v in view.get_velocities()[0]]
                        row['pose_readout_difference']=motion_delta(row['direct_physics_pose'],public)
                    except Exception as exc:row['direct_read_error']=type(exc).__name__+': '+str(exc)
                    if o.name in previous:row['public_step_motion']=motion_delta(previous[o.name],public)
                    previous[o.name]=public
                    if step%10==0:row['contacts']=contacts(obj)
                    sample['objects'][o.name]=row
                doc['rows'].append(sample)
                if step%30==0:write();print('[trajectory-control] step='+str(step),flush=True)
        doc['completed']=True;write()
    except Exception as exc:
        doc.update(completed=False,error=type(exc).__name__+': '+str(exc));write();traceback.print_exc();raise
    finally:
        try:
            import omnigibson as og
            og.shutdown()
        except (Exception,SystemExit):pass

if __name__=='__main__':main()
