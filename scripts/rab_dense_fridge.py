"""Repair refrigerated search recipes with the released RAB fridge and bins."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

from rab_dense_search import author, digest, load, recipe, save

OLD = '/World/Background/Shell/KitchenRefrigerator'
FRIDGE = '/World/Background/Shell/RefrigeratorRecovered'
DOOR = FRIDGE+'/right_door'
BINS = ('col_shelf1_floor','col_shelf2_floor','col_bin34_floor','col_shelf4_floor')
# Keep refrigerated storage and delivery; preserve the physical comparison
# sample where the archived instruction contains one.
RECIPES = {
    'rab_F014': (('fridge_salad_box',), False, 'ice tray → salad box'),
    'rab_F054': (('fridge_takeout_box',), False, None),
    'rab_F199': (('good2grow',), True, 'yogurt cup → juice bottle'),
    'rab_F205': (('good2grow','good2grow'), True, 'yogurt cups → juice bottles'),
    'rab_F262': (('good2grow','meal_fork'), False, 'yogurt and teaspoon → juice and fork'),
}
FOOD_SOURCES = {'fridge_salad_box':'saladbox.usda','fridge_takeout_box':'foodbox.usda'}
# Per-room corrections measured from the released collider and kitchen joinery.
FRIDGE_OFFSETS = {'rab_F054':(-.075,.005), 'rab_F262':(0.,.78)}  # local width, depth in meters


def compose(layer, asset, output, offset=(0.,0.)):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage=Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths=[str(layer)]
    UsdGeom.SetStageUpAxis(stage,'Z')
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    old=stage.GetPrimAtPath(OLD)
    xf=UsdGeom.XformCache()
    world_pose=xf.GetLocalToWorldTransform(old)
    shift=world_pose.TransformDir(Gf.Vec3d(offset[0],0.,offset[1]))
    world_pose.SetTranslateOnly(world_pose.ExtractTranslation()+shift)
    pose=world_pose*xf.GetLocalToWorldTransform(old.GetParent()).GetInverse()
    stage.OverridePrim(OLD).SetActive(False)
    root=stage.DefinePrim(FRIDGE,'Xform')
    root.GetReferences().AddReference(str(asset))
    UsdGeom.Xformable(root).MakeMatrixXform().Set(pose)
    for side in ('left','right'):
        joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f'{FRIDGE}/{side}_door_hinge'))
        if not joint or not all(stage.GetPrimAtPath(x) for x in joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets()):
            raise ValueError('RAB refrigerator hinge has invalid body bindings')
    stage.GetRootLayer().Save()
    return stage


def pack_fridge(row, contract, stage, layout, assets, attempt):
    from pxr import Gf,UsdGeom
    import numpy as np
    from itertools import product
    from rab_build_search import placement, supports, triangles
    from rab_dense_search import quarter_turn
    names, match, _ = RECIPES[row['id']]
    if len(names)!=contract['target_count'] or not set(names)<=assets.keys():
        raise ValueError('Refrigerator target count or assets do not match archived contract')
    occupied, bins_occupied, records, delivery = [], [], [], []
    fixtures, obstacles = supports(stage)
    xf=UsdGeom.XformCache()
    x=xf.GetLocalToWorldTransform(stage.GetPrimAtPath(FRIDGE)).GetRow3(0)
    yaw=math.degrees(math.atan2(x[1],x[0]))
    to_world=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw))
    to_bins=to_world.GetInverse()
    def outline(points,matrix):
        return np.array([matrix.Transform(Gf.Vec3d(*v)) for v in points])
    def box_from(points):
        return np.array([points.min(axis=0),points.max(axis=0)])
    def move_box(box,matrix):
        return box_from(outline(product(*box.T),matrix))
    def cube_corners(prim):
        half=UsdGeom.Cube(prim).GetSizeAttr().Get()/2
        return outline(product((-half,half),repeat=3),xf.GetLocalToWorldTransform(prim)*to_bins)
    bin_obstacles=[]
    for path,box in obstacles:
        prim=stage.GetPrimAtPath(path)
        if (path.endswith('/ContinuousPantryCornerWall') and prim.IsA(UsdGeom.Mesh)
                and prim.GetAttribute('physics:approximation').Get() == 'none'):
            # Its L-shaped triangle collider leaves the fridge recess clear.
            faces=triangles(UsdGeom.Mesh(prim),xf)
            local_faces=outline(faces.reshape(-1,3),to_bins).reshape(-1,3,3)
            bin_obstacles.extend((path,box_from(face)) for face in local_faces)
            continue
        if path.startswith(FRIDGE+'/') and prim.IsA(UsdGeom.Cube):
            box=box_from(cube_corners(prim))
        else:
            box=move_box(box,to_bins)
        bin_obstacles.append((path,box))
    zones=[]
    headroom={}
    for index,name in enumerate(BINS):
        zone=f'drawer_fridge_bin_{index+1}'
        path=DOOR+'/'+name
        prim=stage.GetPrimAtPath(path)
        if not prim.IsA(UsdGeom.Cube) or not prim.GetAttribute('physics:collisionEnabled').Get():
            raise ValueError(f'Released bin floor unavailable: {path}')
        box=box_from(cube_corners(prim))
        fixtures[zone]=[(box,None,path)]
        zones.append(zone)
        headroom[zone]=(.22,.34,.32,.37)[index]
    rng=random.Random(f'{row["id"]}:fridge:{attempt}')
    sink=row['destination_fixture']
    def place(name,zone,target=False):
        asset=assets[name]
        variants=[asset,quarter_turn(asset)]
        if attempt%2:
            variants.reverse()
        for candidate in variants:
            size=candidate['box'][1]-candidate['box'][0]
            if zone in headroom and size[2]+.01>headroom[zone]:
                continue
            in_bin=zone in headroom
            used=bins_occupied if in_bin else occupied
            # Rigid food boxes fit these narrow bins with a measured positive gap.
            margin=.0015 if name in FOOD_SOURCES else .004
            placed=placement(fixtures,bin_obstacles if in_bin else obstacles,zone,
                             candidate['box'],rng,used,candidate.get('clearance',0.),
                             **(dict(inset=margin,separation=margin) if in_bin else {}))
            if placed is not None:
                box,path=placed
                used.append(box)
                if in_bin:
                    candidate=dict(candidate,rotation=candidate['rotation']*to_world,
                                   box=move_box(candidate['box'],to_world))
                    box=move_box(box,to_world)
                records.append(dict(asset=name,zone=zone,box=box,support=path,source=candidate,target=target))
                return True
        return False
    for name in names:
        if not place(name,sink,True):
            return None
        delivery.append(records.pop())
    if match:
        if not place(names[0],sink):
            return None
        records[-1]['comparison_sample']=True
    rng.shuffle(zones)
    target_zones=[]
    for name in names:
        storage=target_zones[:1] if row['id']=='rab_F262' and target_zones else zones
        zone=next((z for z in storage if place(name,z,True)),None)
        if not zone:
            return None
        target_zones.append(zone)
    primary=target_zones[0]
    others=[x for x in zones if x!=primary]
    # Distinct refrigerated packages of similar size and appearance.
    pool=[x for x in ('sprite_c1','grapefruit_can','starbucks','gatorade_m','gatorade_t',
                      'good2grow','coke_c3','fridge_salad_box','fridge_takeout_box')
          if x not in names and x in assets]
    rng.shuffle(pool)
    needed=max(6,contract['original_search_distractors'])
    for zone in [primary]+[z for _ in range(needed) for z in others+[primary]]:
        if sum(not r['target'] and r['zone'] in headroom for r in records)>=needed:
            break
        for name in sorted(pool,key=lambda k:sum(r['asset']==k for r in records)):
            if sum(r['asset']==name for r in records)<2 and place(name,zone):
                break
    alternative=next((z for z in others if sum(r['zone']==z and not r['target'] for r in records)>=2),None)
    if (sum(not r['target'] and r['zone'] in headroom for r in records)<needed or not alternative
            or not any(not r['target'] and r['zone']==primary for r in records)):
        return None
    rng.shuffle(records)
    for index,item in enumerate(records):
        item['id']=f'Item_{index:02d}'
    return dict(layer=str(layout),records=records,delivery=delivery,shield=None,group='pantry',
                search=primary,decoy=alternative,targets=names,minimum_distractors=needed,
                measured_drawer_headroom_m={},snack_drawers=False)


def native_check(episode, simulator, interface):
    import numpy as np
    from pxr import UsdPhysics
    from rab_validate_search import validate
    drives={}
    def pose(path):
        out=interface.get_rigidbody_transformation(path)
        if not out.get('ret_val'):
            raise ValueError(f'Missing native refrigerator rigid body: {path}')
        return np.array([*out['position'],*out['rotation']],np.float64)
    def before(stage):
        for side,target in (('left',-60.),('right',60.)):
            path=f'{FRIDGE}/{side}_door_hinge'
            prim=stage.GetPrimAtPath(path)
            joint=UsdPhysics.RevoluteJoint(prim)
            if not joint or not joint.GetLowerLimitAttr().Get()<=target<=joint.GetUpperLimitAttr().Get():
                raise ValueError('Refrigerator lacks the requested native hinge travel')
            body=stage.GetPrimAtPath(f'{FRIDGE}/{side}_door')
            if not body.HasAPI(UsdPhysics.RigidBodyAPI) or body.GetAttribute('physics:kinematicEnabled').Get():
                raise ValueError('Refrigerator door is not dynamic')
            drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(1100.)
            drive.CreateDampingAttr(130.)
            drive.CreateMaxForceAttr(180.)
            drive.CreateTargetPositionAttr(0.)
            drives[side]=(drive,target)
    def dir_error(a,b):
        a,b=a/np.linalg.norm(a),b/np.linalg.norm(b)
        return math.degrees(2*math.acos(np.clip(abs(a@b),0,1)))
    def local_points(items,door):
        from scipy.spatial.transform import Rotation
        return Rotation.from_quat(door[3:]).inv().apply(items[:,:3]-door[:3])
    def after(stage, poses, initial):
        attached=[i for i,r in enumerate(episode['placed']) if r['zone'].startswith('drawer_fridge_bin_')]
        loose=[i for i in range(len(initial)) if i not in attached]
        base=pose(DOOR)
        reference=local_points(initial[attached],base)
        clock=75/60
        def sweep(drive,a,b):
            nonlocal clock
            for step in range(240):
                f=min(1.,(step+1)/180)
                drive.GetTargetPositionAttr().Set(a+(b-a)*f*f*(3-2*f))
                simulator.simulate(1/60,clock)
                simulator.fetch_results()
                clock+=1/60
        checks=[]
        for side,(drive,target) in drives.items():
            path=f'{FRIDGE}/{side}_door'
            start=pose(path)
            sweep(drive,0.,target)
            opened=pose(path)
            open_poses=poses()
            sweep(drive,target,0.)
            closed=pose(path)
            close_poses=poses()
            open_angle,closed_angle=dir_error(start[3:],opened[3:]),dir_error(start[3:],closed[3:])
            open_drift=np.linalg.norm(local_points(open_poses[attached],pose(DOOR) if side=='left' else opened)-reference,axis=1)
            close_drift=np.linalg.norm(local_points(close_poses[attached],pose(DOOR))-reference,axis=1)
            exterior=np.linalg.norm(close_poses[loose,:3]-initial[loose,:3],axis=1)
            passed=(open_angle>=45. and closed_angle<=3. and np.max(open_drift)<.04
                    and np.max(close_drift)<.04 and (not len(exterior) or np.max(exterior)<.04))
            checks.append(dict(side=side,passed=bool(passed),open_angle_deg=open_angle,
                               close_error_deg=closed_angle,item_open_drift_m=open_drift.tolist(),
                               item_close_drift_m=close_drift.tolist(),exterior_drift_m=exterior.tolist()))
        reference_state=dict(frame='world',bodies={r['body']:dict(pose=initial[i].tolist(),
            support=r['support'],target=r['target'],zone=r['zone']) for i,r in enumerate(episode['placed'])},
            articulations_deg=dict(left=0.,right=0.))
        return dict(passed=all(c['passed'] for c in checks),hinges=checks,native_steps=1035,
                    scope='Native RAB refrigerator hinge motion with loaded bins',restoration_reference=reference_state)
    return validate(episode,simulator,interface,before_attach=before,after_settle=after)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--attempts',type=int,default=16)
    parser.add_argument('--task-id',choices=RECIPES,action='append')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run,repo,lane=args.run.resolve(),args.repo.resolve(),args.output.resolve()/'drawer'
    lane.mkdir(parents=True,exist_ok=True)
    lock=(lane/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=args.task_id or list(RECIPES)
    counts=Counter()
    def status(state, current=None):
        save(lane/'status.json',dict(pid=os.getpid(),status=state,total=len(selected),processed=sum(counts.values()),
                                     counts=dict(counts),current=current,updated_at=time.time()))
    status('starting')
    if not args.prepare_only:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=64,height=64,multi_gpu=False,limit_cpu_threads=8,
                               extra_args=['--/log/file='+str(lane/'kit.log')]))
        import carb
        import omni.physx
        settings=carb.settings.get_settings()
        settings.set_bool('/physics/suppressReadback',False)
        settings.set_bool('/physics/updateToUsd',True)
        sim=omni.physx.get_physx_simulation_interface()
        interface=omni.physx.get_physx_interface()
    from pxr import Gf,Sdf
    from rab_assets import LayerBundle
    from rab_available_resources import namespace,provision
    from rab_build_search import CATALOG,load_assets,read_asset
    from rab_dense_search import FIXTURE_NAMES
    resources=run/'available_resources'
    mapping,_=provision(repo,run,resources)
    bundle=LayerBundle(namespace(resources),lane/'asset_layers')
    assets,errors=load_assets(mapping,repo,bundle,load(run/'pose_calibration/poses.json')['assets'])
    CATALOG['pantry'].update(fridge_salad_box='salad box',fridge_takeout_box='takeout box',meal_fork='fork')
    for key,file in FOOD_SOURCES.items():
        item=read_asset(resources/'room/resources/shared_assets/fridge_food_boxes_v1'/file,bundle)
        if item is None:
            raise ValueError(f'No native food model: {key}')
        assets[key]=dict(item,rotation=Gf.Matrix4d(1.),clearance=0.,pose_variant='authored_flat_base')
    fridge=bundle.clone(resources/'room/all_assets/refrigerator/asset.usda')
    baseline={r['id']:r for r in load(run/'search_build_v3/release/physics_manifest.json')['tasks']}
    sources={r['id']:r for r in map(json.loads,(run/'migration_queue.jsonl').read_text().splitlines())}
    for ident in selected:
        status('building',ident)
        row=baseline[ident]
        contract=recipe(sources[ident])
        folder=lane/'cases'/ident
        folder.mkdir(parents=True,exist_ok=True)
        result=dict(id=ident,layout_id=row['layout_id'],status='pending_repair',attempts=[])
        try:
            source_layer=Sdf.Layer.FindOrOpen(row['scene'])
            original=list(source_layer.subLayerPaths)
            if len(original)!=1:
                raise ValueError('Expected a shared room layer')
            layout=folder/'fridge_layout.usda'
            offset=FRIDGE_OFFSETS.get(ident,(0.,0.))
            stage=compose(original[0],fridge,layout,offset)
            for attempt in range(args.attempts):
                plan=pack_fridge(row,contract,stage,layout,assets,attempt)
                if plan is None:
                    result['attempts'].append(dict(attempt=attempt,status='no_fridge_fit'))
                    continue
                dest=folder/f'attempt_{attempt+1:02d}'
                episode=author(row,plan,contract,dest)
                names,match,rewrite=RECIPES[ident]
                from rab_instruction_diversity import compose_diverse_instruction
                if match:
                    sample=next(r for r in plan['records'] if r.get('comparison_sample'))
                    episode['comparison_sample_body']=sample['body']
                episode['instruction']=compose_diverse_instruction(
                    seed_key=ident,
                    target_assets=list(names),
                    delivery_fixture=row['destination_fixture'],
                    search_scope='kitchen_fridge',
                    layout_id=int(row['layout_id']),
                    comparison_match=bool(match),
                )
                episode['arrangement']=dict(kind='articulated_refrigerator',articulations={side:
                    f'{FRIDGE}/{side}_door_hinge' for side in ('left','right')},shield=None)
                episode['search_storage_migration']=dict(kind='rab_refrigerator',rewrite=rewrite,
                    archived_sources=sorted({r['receptacle'] for r in contract['initial_relations']}),
                    native_refrigerator_asset_sha256=digest(resources/'room/all_assets/refrigerator/asset.usda'),
                    baseline_target_assets=[p['asset'] for p in row['placed'] if p['target']],
                    replacement_target_assets=list(names),comparison_sample_preserved=match,
                    fridge_local_width_depth_offset_m=list(offset),
                    storage='Released right-door bins, four distinct search levels')
                save(dest/'episode.json',episode)
                if args.prepare_only:
                    result.update(status='prepared_for_native',episode=str(dest/'episode.json'))
                    break
                status('validating_physics',ident)
                report=native_check(episode,sim,interface)
                report['scene_sha256']=episode['scene_sha256']
                save(dest/'native_validation.json',report)
                result['attempts'].append(dict(attempt=attempt,status=report['status']))
                if report['status']=='physx_initialization_pass':
                    result.update(status='accepted_dense_initialization',episode=str(dest/'episode.json'),
                                  physics_receipt=str(dest/'native_validation.json'))
                    break
        except Exception as exc:
            result.update(error=str(exc),traceback=traceback.format_exc())
        save(folder/'result.json',result)
        counts[result['status']]+=1
        status('building',ident)
    status('finished')
    os._exit(0)


if __name__=='__main__':
    main()
