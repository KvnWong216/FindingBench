"""Measure stable orientations of the released RAB graspable assets."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / 'pose_calibration'
    output.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=64, height=64, multi_gpu=False,
        limit_cpu_threads=8, extra_args=['--/log/file='+str(output/'kit.log')]))
    try:
        import carb
        import numpy as np
        import omni.physx
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdUtils
        from rab_assets import LayerBundle
        from rab_available_resources import digest, namespace
        from rab_build_search import CATALOG, bounds, read_asset, save_json
        resources = run / 'available_resources'
        bundle = LayerBundle(namespace(resources), output/'layers')
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage,1)
        scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
        scene.CreateGravityDirectionAttr((0,0,-1))
        scene.CreateGravityMagnitudeAttr(9.81)
        api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        api.CreateEnableGPUDynamicsAttr(False)
        api.CreateTimeStepsPerSecondAttr(60)
        floor=UsdGeom.Cube.Define(stage,'/World/Floor')
        floor.AddTranslateOp().Set((4,3,-.05))
        floor.AddScaleOp().Set((10,8,.05))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        items={}
        for name in sorted({name for group in CATALOG.values() for name in group}):
            path=(resources/'libraries/current_assets/all_assets'/name/'asset.usda')
            dynamic=(args.repo.resolve()/'src/robotagentbench/backends/profiles/room'
                /'openend/tasks/cloth_clear/dynamic.f_hyv0g1'/(name+'.usda'))
            if dynamic.is_file():
                path=dynamic
            if not path.is_file():
                continue
            try:
                asset=read_asset(path,bundle)
            except (FileNotFoundError,ValueError):
                continue
            if not asset:
                continue
            index=len(items)
            parent=UsdGeom.Xform.Define(stage,'/World/Calibration/'+name)
            model=stage.DefinePrim(str(parent.GetPath())+'/Model')
            model.GetReferences().AddReference(asset['layer'])
            position=np.array([index%8,index//8,.012])-asset['box'][0]
            parent.AddTranslateOp().Set(Gf.Vec3d(*position))
            items[name]=(str(model.GetPath())+asset['body'],asset['sha256'])
        settings=carb.settings.get_settings()
        settings.set_bool('/physics/suppressReadback',False)
        settings.set_bool('/physics/updateToUsd',True)
        cache=UsdUtils.StageCache.Get()
        sid=cache.Insert(stage)
        sim=omni.physx.get_physx_simulation_interface()
        if not sim.attach_stage(sid.ToLongInt()):
            raise RuntimeError('Unable to attach calibration floor')
        interface=omni.physx.get_physx_interface()
        poses={}
        for step in range(190):
            sim.simulate(1/60,step/60)
            sim.fetch_results()
            if step in (0,159,189):
                frame={}
                for name,(path,_) in items.items():
                    pose=interface.get_rigidbody_transformation(path)
                    if pose.get('ret_val'):
                        frame[name]=np.array([*map(float,pose['position']),*map(float,pose['rotation'])])
                poses[step]=frame
        from scipy.spatial.transform import Rotation
        reports={}
        for name, (path,sha) in items.items():
            if not all(name in frame for frame in poses.values()):
                continue
            first,mid,last=(poses[i][name] for i in (0,159,189))
            motion=float(np.linalg.norm(last[:3]-mid[:3]))
            if motion>.008 or not np.isfinite(last).all() or last[2]<-.025:
                continue
            delta=Rotation.from_quat(last[3:])*Rotation.from_quat(first[3:]).inv()
            world=UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
            visual_box=bounds(world,stage.GetPrimAtPath('/World/Calibration/'+name))
            if visual_box is None:
                continue
            reports[name]=dict(source_sha256=sha,rotation_xyzw=delta.as_quat().tolist(),
                floor_to_bounds_min_m=float(visual_box[0,2]),settled_body_pose=last.tolist(),
                residual_motion_m=motion)
        sim.detach_stage()
        save_json(output/'poses.json',dict(assets=reports,scope='native rigid-body calibration on a flat plane'))
        print('Calibrated',len(reports),'of',len(items),flush=True)
        os._exit(0)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)


if __name__ == '__main__':
    main()
