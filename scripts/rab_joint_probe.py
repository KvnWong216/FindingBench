#!/usr/bin/env python3
"""Native RAB drawer opening/closing admission, separate from task success."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layout-layer',type=Path,help='Relocated verified RAB layout layer')
    parser.add_argument('--clutter-run',type=Path,help='Add a FindingBench drawer pilot using calibrated assets')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.repo.resolve()/'src'))
    report=dict(status='running',started_at=time.time(),task_success='not_evaluated',
        grasp_place='not_run',native_steps=0,source_assets_modified=False,
        actuation='Temporary drive in the diagnostic session layer; original asset files unchanged')
    app=None;simulation=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,multi_gpu=False,width=64,height=64,limit_cpu_threads=8,
            extra_args=['--/log/file='+str(out/'kit.log')]))
        import numpy as np
        import carb
        import omni.physx
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Gf
        from rab_assets import AssetNamespace,LayerBundle,joints
        namespace=AssetNamespace.from_runtime(args.runtime)
        logical='/home/lingjun/robot_agent_arena/api-protocol/assets/house_scene_l1/demo_sim/'
        base=namespace.resolve(logical+'scene_room_baseline.usda')
        slide=namespace.resolve(logical+'livingroom_drawer_slide.usda')
        bundle=LayerBundle(namespace,out/'relocated_layers')
        relocated_base=(args.layout_layer.resolve(strict=True) if args.layout_layer
                        else bundle.clone(base))
        relocated_slide=bundle.clone(slide)
        scene=out/'drawer_probe.usda';stage=Usd.Stage.CreateNew(str(scene))
        stage.GetRootLayer().subLayerPaths=[str(relocated_slide),str(relocated_base)]
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        stage.GetRootLayer().Save();report['resources']=bundle.receipt()
        stage=Usd.Stage.Open(str(scene))
        if not stage:raise RuntimeError('RAB_NATIVE_STAGE_OPEN_FAILED')
        stage.SetEditTarget(stage.GetSessionLayer())
        report['joints']=joints(stage)
        drawer='/World/Assets/CoffeeTable/black_box_1';joint_path=drawer+'_slide'
        prim=stage.GetPrimAtPath(drawer);joint=UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_path))
        if not joint or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('RAB_NATIVE_DRAWER_MECHANISM_MISSING')
        if args.layout_layer:
            path='/World/Assets/CoffeeTable'
            reference=Usd.Stage.OpenMasked(str(base),Usd.StagePopulationMask([path]))
            ref_cache=UsdGeom.XformCache()
            new_cache=UsdGeom.XformCache()
            original=ref_cache.GetLocalToWorldTransform(reference.GetPrimAtPath(path))
            current=new_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            delta=original.GetInverse()*current
            rotation=delta.RemoveScaleShear().ExtractRotationQuat()
            p=joint.GetLocalPos0Attr().Get()
            q=joint.GetLocalRot0Attr().Get()
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*delta.Transform(Gf.Vec3d(*p))))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(rotation*Gf.Quatd(q.GetReal(),Gf.Vec3d(*q.GetImaginary()))))
            report['calibration']=dict(reference_table=path,method='RAB table-relative joint anchor calibration')
        if prim.GetAttribute('physics:kinematicEnabled').Get() is True:raise ValueError('RAB_DRAWER_IS_KINEMATIC')
        low=float(joint.GetLowerLimitAttr().Get());high=float(joint.GetUpperLimitAttr().Get())
        if not all(math.isfinite(v) for v in (low,high)) or high-low<.10:raise ValueError('RAB_DRAWER_TRAVEL_NOT_ESTABLISHED')
        closed=min(high,max(low,0.));limit=high if abs(high-closed)>=abs(low-closed) else low
        opened=closed+math.copysign(min(.15,abs(limit-closed)),limit-closed)
        report['drawer']=dict(path=drawer,joint=joint_path,axis=joint.GetAxisAttr().Get(),
            lower_m=low,upper_m=high,closed_target_m=closed,open_target_m=opened)
        items=[]
        if args.clutter_run:
            from rab_available_resources import provision
            from rab_build_search import bounds,load_assets,placement
            run=args.clutter_run.resolve()
            mapping,_=provision(args.repo.resolve(),run,run/'available_resources')
            poses=json.loads((run/'pose_calibration/poses.json').read_text())['assets']
            assets,_=load_assets(mapping,args.repo.resolve(),bundle,poses)
            cache=UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
            floor=[p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)
                   and '/Bottom_0/' in str(p.GetPath())]
            if len(floor)!=1:raise ValueError('RAB_DRAWER_FLOOR_NOT_UNIQUE')
            floor_path=str(floor[0].GetPath())
            surface=bounds(cache,floor[0])
            # The released drawer bottom mesh lies inside the drawer walls.
            # Keep the object footprints away from the lips and from one another.
            fixtures={'drawer':[(surface,None,floor_path)]}
            target='glasses_case'
            inventory=(target,'cloth_clear_tape_measure','cloth_clear_memo_pad','cloth_clear_eraser')
            placements=[]
            rng=random.Random('drawer_search')
            for attempt in range(25):
                occupied=[]
                placements=[]
                for name in inventory:
                    asset=assets.get(name)
                    if asset is None or asset['box'][1,2]-asset['box'][0,2]>.12:
                        continue
                    packed=placement(fixtures,[],'drawer',asset['box'],rng,occupied,
                                     asset.get('clearance',0.))
                    if packed is not None:
                        box,_=packed
                        occupied.append(box)
                        placements.append((name,box))
                if len(placements)>=3 and placements[0][0]==target:
                    break
            for name,box in placements:
                asset=assets[name]
                parent=UsdGeom.Xform.Define(stage,'/World/FindingBench/'+name)
                model=stage.DefinePrim(str(parent.GetPath())+'/Model')
                model.GetReferences().AddReference(asset['layer'])
                translation=Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(box[0]-asset['box'][0])))
                parent.MakeMatrixXform().Set(asset['rotation']*translation)
                items.append(dict(asset=name,body=str(model.GetPath())+asset['body'],
                                  support=floor_path,target=name==target))
            if len(items)<3 or not any(item['target'] for item in items):
                raise ValueError('RAB_DRAWER_CLUTTER_DOES_NOT_FIT')
            from pxr import Sdf
            initial_layer=Sdf.Layer.CreateAnonymous('drawer_search.usda')
            initial_layer.TransferContent(stage.GetSessionLayer())
            initial_layer.subLayerPaths=[str(scene)]
            episode=out/'drawer_search.usda'
            initial_layer.Export(str(episode))
            report['drawer_search']=dict(scene=str(episode),placed=items,
                instruction='Could you find my glasses case in the living room and set it on the small side table?',
                delivery_fixture='/World/Assets/SmallCoffeeTable')
        # Only a probe actuator is added. Limits, bodies, collision meshes,
        # materials, mass and the source layer remain those of RAB.
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
        drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(4000.);drive.CreateDampingAttr(400.)
        drive.CreateMaxForceAttr(300.);drive.CreateTargetPositionAttr(closed)
        physics_scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(physics_scenes)!=1:raise ValueError('RAB_PHYSICS_SCENE_NOT_UNIQUE')
        physics=physics_scenes[0];api=PhysxSchema.PhysxSceneAPI.Apply(physics)
        report['source_physics']=dict(gpu=api.GetEnableGPUDynamicsAttr().Get(),hz=api.GetTimeStepsPerSecondAttr().Get())
        api.CreateEnableGPUDynamicsAttr(False);api.CreateTimeStepsPerSecondAttr(60)
        settings=carb.settings.get_settings()
        for key,value in (('/physics/suppressReadback',False),('/physics/updateToUsd',True),('/physics/updateVelocitiesToUsd',True)):
            settings.set_bool(key,value)
        stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        simulation=omni.physx.get_physx_simulation_interface()
        if not simulation.attach_stage(stage_id):raise RuntimeError('RAB_PHYSICS_ATTACH_FAILED')
        interface=omni.physx.get_physx_interface();clock=0.;samples=[]
        def item_positions():
            found=[]
            for item in items:
                pose=interface.get_rigidbody_transformation(item['body'])
                if not pose.get('ret_val'):raise ValueError('Drawer clutter native pose unavailable')
                found.append([float(v) for v in pose['position']])
            return np.asarray(found)
        def advance(phase,steps,transition=None):
            nonlocal clock
            last=None
            for i in range(steps):
                if transition:
                    fraction=(i+1)/steps
                    fraction=fraction*fraction*(3-2*fraction)
                    start,end=transition
                    drive.GetTargetPositionAttr().Set(start+(end-start)*fraction)
                simulation.simulate(1/60,clock);simulation.fetch_results();clock+=1/60;report['native_steps']+=1
                last=interface.get_rigidbody_transformation(drawer)
                if not last.get('ret_val'):raise ValueError('RAB_DRAWER_NATIVE_POSE_UNAVAILABLE')
                position=[float(v) for v in last['position']]
                if not all(math.isfinite(v) for v in position):raise ValueError('RAB_DRAWER_NATIVE_POSE_NONFINITE')
                if i%10==0:samples.append(dict(phase=phase,time=clock,position=position))
            return np.asarray(last['position'],dtype=float)
        initial=advance('closed_initial',120 if items else 60)
        items_initial=item_positions()
        if items:
            advance('opening',180,(closed,opened))
        drive.GetTargetPositionAttr().Set(opened);open_position=advance('open',180)
        items_open=item_positions()
        if items:
            advance('closing',180,(opened,closed))
        drive.GetTargetPositionAttr().Set(closed);closed_position=advance('close',180)
        items_closed=item_positions()
        travel=float(np.linalg.norm(open_position-initial));residual=float(np.linalg.norm(closed_position-initial))
        report.update(open_displacement_m=travel,close_residual_m=residual,
            open_passed=travel>=.10,close_passed=residual<=.03,samples=samples)
        if not report['open_passed'] or not report['close_passed']:raise ValueError('RAB_DRAWER_OPEN_CLOSE_ADMISSION_FAILED')
        if items:
            open_drift=np.linalg.norm((items_open-open_position)-(items_initial-initial),axis=1)
            close_drift=np.linalg.norm((items_closed-closed_position)-(items_initial-initial),axis=1)
            report['drawer_search'].update(open_relative_drift_m=open_drift.tolist(),
                close_relative_drift_m=close_drift.tolist(),
                containment_motion_passed=bool(np.all(open_drift<.04)&np.all(close_drift<.04)))
            if not report['drawer_search']['containment_motion_passed']:
                raise ValueError('RAB_DRAWER_CLUTTER_MOVED_OR_FELL')
        report['status']='joint_probe_passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        report['finished_at']=time.time();(out/'result.json').write_text(json.dumps(report,indent=2))
        if simulation is not None:
            try:simulation.detach_stage()
            except Exception:pass
    print('RAB_JOINT_PROBE '+json.dumps({k:report.get(k) for k in ('status','error','open_displacement_m','close_residual_m')}),flush=True)
    os._exit(0 if report['status']=='joint_probe_passed' else 1)


if __name__=='__main__':main()
