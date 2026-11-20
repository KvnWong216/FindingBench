"""Check RAB episode initialization and rigid-body stability in native PhysX."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def validate(episode, sim, interface, *, before_attach=None, after_settle=None):
    import numpy as np
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdUtils
    stage = Usd.Stage.Open(episode['scene'])
    if not stage:
        raise ValueError('Episode USD cannot be loaded')
    stage.SetEditTarget(stage.GetSessionLayer())
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError(f'Expected one physics scene, found {len(scenes)}')
    api = PhysxSchema.PhysxSceneAPI.Apply(scenes[0])
    api.CreateEnableGPUDynamicsAttr(False)
    api.CreateTimeStepsPerSecondAttr(60)
    if before_attach is not None:
        before_attach(stage)
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage)
    attached = False
    try:
        attached = sim.attach_stage(stage_id.ToLongInt())
        if not attached:
            raise RuntimeError('Could not attach native PhysX stage')
        paths = [row['body'] for row in episode['placed']]
        def poses():
            result = []
            for path in paths:
                pose = interface.get_rigidbody_transformation(path)
                if not pose.get('ret_val'):
                    raise ValueError('No native rigid-body pose for ' + path)
                xyz = [float(v) for v in pose['position']]
                quaternion = [float(v) for v in pose['rotation']]
                result.append(xyz + quaternion)
            result = np.array(result)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite native rigid-body pose')
            return result
        measurements = {}
        for step in range(75):
            sim.simulate(1/60,step/60)
            sim.fetch_results()
            if step in (0,44,74):
                measurements[step] = poses()
        initial,mid,final = (measurements[i] for i in (0,44,74))
        drift = np.linalg.norm(final[:,:3]-initial[:,:3],axis=1)
        residual = np.linalg.norm(final[:,:3]-mid[:,:3],axis=1)
        angles = 2*np.arccos(np.clip(abs(np.sum(final[:,3:]*initial[:,3:],axis=1)),0,1))
        rows = []
        for index,path in enumerate(paths):
            stable = drift[index] <= .045 and residual[index] <= .008 and angles[index] <= .35
            rows.append(dict(body=path,stable=bool(stable),drift_m=float(drift[index]),
                             final_motion_m=float(residual[index]),rotation_rad=float(angles[index]),
                             initial_pose=initial[index].tolist(),settled_pose=final[index].tolist()))
        stable = all(r['stable'] for r in rows)
        result = dict(status='physx_initialization_pass' if stable else 'physx_initialization_failed',
                      steps=75,objects=rows,
                      scope='Native initial rigid-body stability; skill execution and task success not evaluated')
        if stable and after_settle is not None:
            result['dense_checks'] = after_settle(stage,poses,final)
            if not result['dense_checks']['passed']:
                result['status'] = 'dense_native_gate_failed'
        return result
    finally:
        if attached:
            sim.detach_stage()
        cache.Erase(stage_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    output = build/'native_validation'
    output.mkdir(exist_ok=True)
    started = time.time()
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True,width=64,height=64,multi_gpu=False,
        limit_cpu_threads=8,extra_args=['--/log/file='+str(output/'kit.log')]))
    import carb
    import omni.physx
    settings = carb.settings.get_settings()
    settings.set_bool('/physics/suppressReadback',False)
    settings.set_bool('/physics/updateToUsd',True)
    sim = omni.physx.get_physx_simulation_interface()
    interface = omni.physx.get_physx_interface()
    results = {}
    while time.time()-started < 23*3600:
        cases = sorted(build.glob('rab_*/episode.json'))
        for path in cases:
            ident = path.parent.name
            if ident in results:
                continue
            episode = json.loads(path.read_text())
            receipt = output/(ident+'.json')
            if receipt.is_file():
                prior = json.loads(receipt.read_text())
                if prior.get('scene_sha256') == episode['scene_sha256']:
                    results[ident] = prior
                    continue
            tick = time.time()
            try:
                report = validate(episode,sim,interface)
            except Exception as error:
                report = dict(status='native_error',error=str(error),traceback=traceback.format_exc())
            report.update(id=ident,scene_sha256=episode['scene_sha256'],elapsed_s=time.time()-tick)
            save(receipt,report)
            results[ident] = report
            save(output/'status.json',dict(processed=len(results),results=Counter(r['status'] for r in results.values()),
                 started_at=started,updated_at=time.time(),last_id=ident))
        if (build/'manifest.json').is_file():
            expected = json.loads((build/'manifest.json').read_text())['generated']
            if len(results) >= expected:
                break
        time.sleep(12)
    save(output/'summary.json',dict(processed=len(results),results=Counter(r['status'] for r in results.values()),
         started_at=started,finished_at=time.time()))
    os._exit(0)


if __name__ == '__main__':
    main()
