"""Native RAB scene authoring, with separate immutable revisions and physical readback."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

from rab_spectator_data import ROOT, RAB_MIGRATION, digest, load, save
from spectator_editor import screen_ray

ROBOT = '/World/Robot'
BASE = ROBOT+'/base_link'


def prefix(path, root):
    return path == root or path.startswith(root+'/')


def matrix(pose):
    from pxr import Gf
    mat=Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]),Gf.Vec3d(*map(float,pose[3:6]))))
    return mat.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))


def rigid_part(transform):
    from pxr import Gf
    # Native transforms are rigid and do not contain the asset's authored
    # scale (e.g. the widened drawers). Keep scale in the USD layer.
    split=Gf.Transform(transform)
    rigid=Gf.Matrix4d().SetRotate(split.GetRotation())
    return rigid.SetTranslateOnly(transform.ExtractTranslation())


def set_world(stage, path, transform):
    from pxr import UsdGeom
    prim=stage.GetPrimAtPath(path)
    geom=UsdGeom.Xformable(prim)
    local=transform if geom.GetResetXformStack() else transform*UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()).GetInverse()
    reset=geom.GetResetXformStack()
    geom.MakeMatrixXform().Set(local)
    geom.SetResetXformStack(reset)


class NativeWorld:
    def __init__(self,stage,episode, mobile=False):
        import omni.physx
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils
        self.stage,self.episode=stage,episode
        self.rows=deepcopy(episode['placed'])
        self.drives={}
        self.opened=set()
        self.steps=0
        self.sim=omni.physx.get_physx_simulation_interface()
        self.api=omni.physx.get_physx_interface()
        self.query=omni.physx.get_physx_scene_query_interface()
        self.cache=UsdUtils.StageCache.Get()
        phys=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
        settings=PhysxSchema.PhysxSceneAPI.Apply(phys)
        settings.CreateEnableGPUDynamicsAttr(False)
        settings.CreateTimeStepsPerSecondAttr(60)
        for key,path in episode['arrangement'].get('articulations',{}).items():
            prim=stage.GetPrimAtPath(path)
            joint=UsdPhysics.Joint(prim)
            angular=prim.IsA(UsdPhysics.RevoluteJoint)
            target=(-88 if key=='left' else 88) if angular else .15
            drive=UsdPhysics.DriveAPI.Apply(prim,'angular' if angular else 'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(1100. if angular else 4000.)
            drive.CreateDampingAttr(130. if angular else 400.)
            drive.CreateMaxForceAttr(180. if angular else 300.)
            body=str(joint.GetBody1Rel().GetTargets()[0])
            initial=episode.get('editor_joint_targets',{}).get(key,0.)
            drive.CreateTargetPositionAttr(initial)
            if initial:
                self.opened.add(key)
            self.drives[key]=(drive,target,body,path,angular)
        from rab_embodiment import (
            UnitreeMobileController,
            apply_unitree_ready_pose,
            get_embodiment,
            swap_stage_to_unitree_g1,
        )
        self.embodiment = get_embodiment() if mobile else get_embodiment('agibot_g1')
        self.robot_prim = self.embodiment.robot_prim
        self.mobile=None
        if mobile and self.embodiment.name.startswith('unitree_g1'):
            swap_stage_to_unitree_g1(stage, self.embodiment)
            self.mobile = UnitreeMobileController(stage, self.embodiment)
        else:
            anchor=UsdPhysics.FixedJoint(stage.GetPrimAtPath(BASE+'/FixedJoint'))
            world=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
            if anchor:
                anchor.CreateLocalPos0Attr(Gf.Vec3f(world.ExtractTranslation()))
                anchor.CreateLocalRot0Attr(Gf.Quatf(world.ExtractRotationQuat()))
            if mobile:
                from sys import path as module_path
                from rab_skills import NATIVE_SOURCE
                module_path.insert(0,str(NATIVE_SOURCE))
                from arena_api.mobile_base_native import Controller
                from short_tasks.agent_probe_v1 import robot_parameter_alignment
                robot_parameter_alignment.apply(stage)
                self.mobile=Controller(stage)
        from pxr import UsdGeom
        from rab_build_search import bounds
        bbox=UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
        self.spawn_box=bounds(bbox,stage.GetPrimAtPath(self.robot_prim))
        self.attach()
        self.step(1 if self.mobile else 75)
        if self.mobile:
            self.mobile.bind()
            if self.embodiment.name.startswith('unitree_g1'):
                apply_unitree_ready_pose(self)
            else:
                from rab_skills import NATIVE_SOURCE
                from arena_api.robot_profiles import FOLLOWERS
                robot=self.mobile.view
                names=list(robot.shared_metatype.dof_names)
                ready=load(NATIVE_SOURCE/'layout_task_activation_v1/task_scene_registration_v1/recorded_ready_pose_042_v1.json')['targets_rad']
                target=robot.get_dof_positions().copy()
                limits=robot.get_dof_limits()[0]
                for name,value in {**ready,**{k:-ready[v] for k,v in FOLLOWERS.items()}}.items():
                    j=names.index(name)
                    if not limits[j,0]-1e-6<=value<=limits[j,1]+1e-6:
                        raise ValueError('RAB ready pose outside native robot limits: '+name)
                    target[0,j]=value
                idx=np.array([0],np.uint32)
                robot.set_dof_positions(target,idx)
                robot.set_dof_velocities(np.zeros_like(target),idx)
                robot.set_dof_position_targets(target,idx)
                self.step(45)

    def attach(self):
        self.sid=self.cache.Insert(self.stage)
        if not self.sim.attach_stage(self.sid.ToLongInt()):
            self.cache.Erase(self.sid)
            self.sid=None
            raise ValueError('RAB 无法启动原生物理场景')

    def close(self):
        if getattr(self,'sid',None) is not None:
            self.sim.detach_stage()
            self.cache.Erase(self.sid)
            self.sid=None

    def step(self,n,*,sync=True):
        for _ in range(n):
            self.sim.simulate(1/60,self.steps/60)
            self.sim.fetch_results()
            self.steps+=1
        if sync:
            self.api.update_transformations(False,True,True)

    def pose(self,path):
        p=self.api.get_rigidbody_transformation(path)
        if not p.get('ret_val'):
            raise ValueError('无法读取物体物理位姿')
        return np.array([*p['position'],*p['rotation']],float)

    def rays(self,origin,direction,limit=25.,ignore=()):
        from pxr import Gf
        hits=[]
        def record(hit):
            b,c=str(hit.rigid_body),str(hit.collision)
            if not any(prefix(b,x) or prefix(c,x) for x in ignore):
                hits.append(dict(body=b,path=c,position=list(hit.position),normal=list(hit.normal),distance=float(hit.distance)))
            return True
        self.query.raycast_all(Gf.Vec3f(*map(float,origin)),Gf.Vec3f(*map(float,direction)),float(limit),record)
        return sorted(hits,key=lambda h:h['distance'])

    def hit(self,camera,xy,fov,aspect,*,ignore=(),actor=False):
        o,d=screen_ray(camera,xy,fov,aspect)
        hits=self.rays(o,d,ignore=(*ignore,self.robot_prim) if actor else ignore)
        if not hits:
            raise ValueError('没有点到场景表面；请换个角度再试')
        return hits[0]

    def item(self,hit):
        return next((r for r in self.rows if prefix(hit['body'],r['body']) or prefix(hit['path'],r['body'].split('/Model',1)[0])),None)

    def joint(self,hit):
        return next((k for k,(_,_,b,_,_) in self.drives.items() if any(prefix(hit[s],b) for s in ('body','path'))),None)

    def swing(self,key,opening):
        drive,target,body,_,angular=self.drives[key]
        was_open=key in self.opened
        if was_open==opening:
            raise ValueError('该门或抽屉已经处于所选状态')
        old,target=(0.,target) if opening else (target,0.)
        start=self.pose(body)
        before={r['body']:self.pose(r['body']) for r in self.rows}
        def sweep(a,b):
            for i in range(240):
                f=min(1.,(i+1)/180)
                drive.GetTargetPositionAttr().Set(a+(b-a)*f*f*(3-2*f))
                self.step(1,sync=False)
            self.step(1)
        sweep(old,target)
        end=self.pose(body)
        from rab_restore import rotation_difference
        travel=math.degrees(rotation_difference(start[3:],end[3:])) if angular else float(np.linalg.norm(end[:3]-start[:3]))
        maximum=0.
        def carried(row):
            # Multiple books may be supported by one another inside a drawer.
            support=row['support']
            seen={row['body']}
            while support not in seen:
                if prefix(support,body):
                    return True
                seen.add(support)
                base=next((r for r in self.rows if r['body']==support),None)
                if base is None:
                    return False
                support=base['support']
            return False
        for row in self.rows:
            p=before[row['body']]
            q=self.pose(row['body'])
            if carried(row):
                p=(matrix(p)*matrix(start).GetInverse()).ExtractTranslation()
                q=(matrix(q)*matrix(end).GetInverse()).ExtractTranslation()
            maximum=max(maximum,math.dist(p[:3],q[:3]))
        if travel<(45. if angular else .10) or maximum>.05:
            sweep(target,old)
            raise ValueError(f'门或抽屉没有走完行程，或里面的物品移动太多；请先清理挡住的物体'
                             f'（实测行程 {travel:.3f} {"°" if angular else "m"}，物品最大偏移 {maximum*100:.1f} cm）')
        if opening:
            self.opened.add(key)
        else:
            self.opened.remove(key)
        return dict(travel=travel,unit='deg' if angular else 'm',max_item_drift_m=maximum)

    def floor(self,xy,max_z):
        for h in self.rays([*xy,max_z],[0,0,-1],limit=3.,ignore=(self.robot_prim,)):
            if h['normal'][2]>.8:
                return h
        raise ValueError('找不到水平承托面；请点在台面、抽屉底或层板附近')


class Catalog:
    ALIASES={'可乐':'coke', '果汁':'good2grow', '汽水':'soda', '纸杯':'paper cup','叉子':'fork',
             '遥控器':'remote','玩具':'toys','杯子':'cup','笔记本':'notebook','书':'book',
             '盒子':'box','瓶':'bottle','沙拉':'salad','餐具':'tableware','零食':'pantry'}
    def __init__(self, directory):
        from pxr import Gf
        from rab_assets import LayerBundle
        from rab_available_resources import namespace,provision
        from rab_build_search import CATALOG,load_assets,read_asset
        from rab_dense_fridge import FOOD_SOURCES
        repo=ROOT/'references/RobotAgentBench'
        resources=RAB_MIGRATION/'available_resources'
        mapping,_=provision(repo,RAB_MIGRATION,resources)
        bundle=LayerBundle(namespace(resources),directory)
        self.assets,_=load_assets(mapping,repo,bundle,load(RAB_MIGRATION/'pose_calibration/poses.json')['assets'])
        self.entries={}
        for group, names in CATALOG.items():
            for key,label in names.items():
                if key in self.assets:
                    self.entries[key]=(group,label)
        for key,name in FOOD_SOURCES.items():
            item=read_asset(resources/'room/resources/shared_assets/fridge_food_boxes_v1'/name,bundle)
            if item:
                self.assets[key]=dict(item,rotation=Gf.Matrix4d(1.),clearance=0.)
                self.entries[key]=('pantry',key.removeprefix('fridge_').replace('_',' '))
    def search(self,q,offset=0,limit=36):
        q=self.ALIASES.get(q.strip(),q.strip()).casefold()
        rows=[]
        for key,(group,label) in sorted(self.entries.items()):
            extra=' soda' if key in ('coke_c3','sprite_c1','grapefruit_can') else ''
            if all(t in f'{group} {key} {label}'.casefold()+extra for t in q.split()):
                size=np.ptp(self.assets[key]['box'],axis=0).tolist()
                rows.append(dict(model=key,category=group,label=label,size=size))
        return dict(total=len(rows),offset=offset,items=rows[offset:offset+limit])


class Editor:
    def __init__(self,args,task,state,world,member,revision):
        self.args,self.task,self.state,self.world,self.member,self.revision=args,task,state,world,member,revision
        self.stage=world.stage
        self.ed=state.status['editor']
        self.selected=None
        self.deleted=set()
        self.changed=set()
        self.ed.update(placement_mode='rab_gravity_v1',fov=args.fov,aspect=args.width/args.height,
                       width=args.width,height=args.height,message='RAB 原生编辑就绪')
        state.catalog=Catalog(Path(args.rab_edits).resolve()/'asset_layers')

    def pick(self,camera,xy):
        return self.world.hit(camera,xy,self.args.fov,self.args.width/self.args.height)

    def description(self):
        from pxr import UsdGeom
        from rab_build_search import bounds
        if not self.selected:
            return None
        row,key=self.world.item(self.selected),self.world.joint(self.selected)
        if row:
            root=row['body'].split('/Model',1)[0]
        elif key:
            root=self.world.drives[key][2]
        else:
            root=self.selected['path']
        prim=self.stage.GetPrimAtPath(root)
        box=bounds(UsdGeom.BBoxCache(0,['default','render']),prim) if prim else None
        return dict(name=row['asset'] if row else key or root.rsplit('/',1)[-1],
                    category='可移动物体' if row else '门或抽屉' if key else '场景表面',
                    model=row['asset'] if row else key or '',can_open=bool(key),
                    is_open=key in self.world.opened,can_delete=bool(row and self.can_delete(row)),
                    aabb=box.tolist() if box is not None else None)

    def can_delete(self,row):
        shield=self.world.episode['arrangement'].get('shield') or {}
        sample=self.world.episode.get('comparison_sample_body')
        return (not row['target'] and row['body'] not in (shield.get('blocker'),sample)
                and not any(prefix(r['support'],row['body']) for r in self.world.rows if r is not row)
                and (row.get('editor_added') or
                     sum(not r['target'] for r in self.world.rows)>
                     self.world.episode['minimum_distractors']))

    def place(self,hit,model,yaw=0,row=None):
        from pxr import Gf, UsdGeom
        from rab_build_search import bounds
        if yaw is not None and (isinstance(yaw,bool) or not isinstance(yaw,(int,float)) or not math.isfinite(yaw)):
            raise ValueError('水平角度必须为有限数值')
        path=row['body'].split('/Model',1)[0] if row else '/World/FindingBench/Edit_'+uuid4().hex[:12]
        if row and (row['target'] or any(prefix(r['support'],row['body']) for r in self.world.rows if r is not row)):
            raise ValueError('不能移动任务目标或仍有物品依赖的支撑物')
        added=not row
        asset=self.state.catalog.assets.get(model)
        if not asset:
            raise ValueError('当前版本没有该物体的可放置原生模型')
        cache=UsdGeom.XformCache()
        geometry=UsdGeom.Xformable(self.stage.GetPrimAtPath(path)) if row else UsdGeom.Xform.Define(self.stage,path)
        prim=geometry.GetPrim()
        if not row:
            self.stage.DefinePrim(path+'/Model').GetReferences().AddReference(asset['layer'])
        xf=geometry.MakeMatrixXform()
        old=cache.GetLocalToWorldTransform(prim) if row else None
        if yaw is None:
            def azimuth(mat):
                xx=mat.TransformDir(Gf.Vec3d(1,0,0))
                return math.degrees(math.atan2(xx[1],xx[0]))
            yaw=azimuth(old)-azimuth(asset['rotation']) if old is not None else 0.
        base=asset['rotation']*Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),float(yaw)))
        pos=np.asarray(hit['position'],float)
        before={x['body']:self.world.pose(x['body']) for x in self.world.rows if x is not row}
        def local(worldmat):
            return worldmat*UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        try:
            xf.Set(local(base))
            new=bounds(UsdGeom.BBoxCache(0,['default','render','proxy','guide']),prim)
            if new is None:
                raise ValueError('物体缺少可用原生网格')
            span=new[1]-new[0]
            normal=np.asarray(hit['normal'])
            if abs(normal[2])<.6:
                pos[:2]+=normal[:2]*(max(span[:2])/2+.018)
            if normal[2]<-.4:
                pos[2]-=.013
            floor_hits=self.world.rays([pos[0],pos[1],pos[2]+.035],[0,0,-1],limit=.75,ignore=(ROBOT,path))
            support=next((h for h in floor_hits if h['normal'][2]>.8),None)
            if support is None:
                raise ValueError('落点正下方没有合适的水平承托面')
            if self.world.item(support) is not None:
                raise ValueError('此处已有物品；请放到旁边空的台面或抽屉底')
            delta=np.r_[pos[:2]-(new[0,:2]+new[1,:2])/2,
                        support['position'][2]+.006+max(0.,asset.get('clearance',0.))-new[0,2]]
            if np.min(np.minimum(new[1]+delta,self.world.spawn_box[1])-
                      np.maximum(new[0]+delta,self.world.spawn_box[0]))>-.035:
                raise ValueError('放置会挡住机器人起点')
            body=row['body'] if row else path+'/Model'+asset['body']
            intended=base*Gf.Matrix4d().SetTranslate(Gf.Vec3d(*delta))
            xf.Set(local(intended))
            self.world.sim.flush_changes()
            self.world.step(1)
            start=self.world.pose(body)
            self.world.step(70)
            end=self.world.pose(body)
            self.world.step(16)
            final=self.world.pose(body)
            if math.dist(start[:3],final[:3])>.045 or math.dist(end[:3],final[:3])>.005:
                raise ValueError('物品没有放稳或滑落；请换到空位、离边缘远一点')
            if any(math.dist(p[:3],self.world.pose(b)[:3])>.025 for b,p in before.items()):
                raise ValueError('有其他物体被挤动；请把落点移开一些')
            zone=next((x['zone'] for x in self.world.rows if x['support']==support['path']), 'editor')
            if row is None:
                row=dict(body=body,asset=model,target=False,source_sha256=asset['sha256'],editor_added=True)
                self.world.rows.append(row)
            row.update(support=support['path'],zone=zone,
                expected_bounds_m=bounds(UsdGeom.BBoxCache(0,['default','render']),prim).tolist())
            self.changed.add(path)
            self.selected=dict(body=body,path=body)
        except Exception:
            if added:
                prim.SetActive(False)
            else:
                xf.Set(local(old))
            self.world.sim.flush_changes()
            self.world.step(1)
            # Physics can have displaced neighboring objects; restore measured poses.
            self.restore_poses(before)
            raise

    def restore_poses(self,poses):
        import omni.physics.tensors
        if not poses:
            return
        view=omni.physics.tensors.create_simulation_view('numpy',self.world.sid.ToLongInt())
        for path,p in poses.items():
            actor=view.create_rigid_body_view(path)
            if actor.count==1:
                idx=np.array([0],np.uint32)
                actor.set_transforms(np.asarray([p],np.float32),idx)
                actor.set_velocities(np.zeros((1,6),np.float32),idx)
        self.world.step(1)

    def publish(self):
        import os
        import subprocess
        import sys
        from pxr import Usd,UsdGeom,UsdPhysics
        from rab_spectator import build_cache
        original=load(self.task['episode_path'])
        self.member.mkdir(parents=True,exist_ok=True)
        identifier=uuid4().hex
        revision=self.member/'revisions'/identifier
        revision.mkdir(parents=True)
        with (self.member/'publish.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ptr=self.member/'current.json'
            if (load(ptr)['revision'] if ptr.exists() else None)!=self.revision:
                raise ValueError('该条目已在另一个窗口保存，请重新加载后再编辑')
            scene=revision/'scene.usda'
            stage=Usd.Stage.CreateNew(str(scene))
            stage.GetRootLayer().subLayerPaths=[self.world.episode['scene']]
            UsdGeom.SetStageUpAxis(stage,'Z')
            UsdGeom.SetStageMetersPerUnit(stage,1.)
            for path in self.deleted:
                stage.OverridePrim(path).SetActive(False)
            for row in self.world.rows:
                root=row['body'].split('/Model',1)[0]
                if not stage.GetPrimAtPath(root):
                    UsdGeom.Xform.Define(stage,root)
                    stage.DefinePrim(root+'/Model').GetReferences().AddReference(self.state.catalog.assets[row['asset']]['layer'])
                if root not in self.changed and not self.world.drives:
                    continue
                d=UsdGeom.XformCache()
                previous=d.GetLocalToWorldTransform(stage.GetPrimAtPath(row['body']))
                root_world=d.GetLocalToWorldTransform(stage.GetPrimAtPath(root))
                set_world(stage,root,root_world*rigid_part(previous).GetInverse()*matrix(self.world.pose(row['body'])))
            values={}
            for k,(drive,_,body,path,angular) in self.world.drives.items():
                values[k]=drive.GetTargetPositionAttr().Get()
                current=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(body))
                set_world(stage,body,current*rigid_part(current).GetInverse()*matrix(self.world.pose(body)))
                new=UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(path),'angular' if angular else 'linear')
                for name in ('type','stiffness','damping','maxForce','targetPosition'):
                    attr=stage.GetPrimAtPath(path).GetAttribute(f'drive:{"angular" if angular else "linear"}:physics:{name}')
                    oldattr=self.stage.GetPrimAtPath(path).GetAttribute(attr.GetName())
                    attr.Set(oldattr.Get())
            stage.GetRootLayer().Save()
            joint_pose={k:self.world.pose(d[2]).tolist() for k,d in self.world.drives.items()}
            episode=dict(self.world.episode,scene=str(scene),scene_sha256=digest(scene),
                         base_scene_sha256=original['scene_sha256'],placed=self.world.rows,
                         editor_joint_targets=values,editor_joint_pose=joint_pose,editor_revision=identifier)
            expected={r['body']:self.world.pose(r['body']) for r in self.world.rows}
            # A separate Kit process keeps the live USD context and its
            # render product attached while validating the authored revision.
            candidate=revision/'candidate.json'
            diagnostic=revision/'reload_diagnostic.json'
            save(candidate,episode)
            tmp=revision/'kit_tmp'
            tmp.mkdir()
            with (revision/'reload.log').open('wb') as log:
                native=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('rab_edit_verify.py')),
                        '--input',str(candidate),'--output',str(diagnostic)],
                        env=dict(os.environ,TMPDIR=str(tmp)),stdout=log,stderr=subprocess.STDOUT,timeout=110)
            if native.returncode or not diagnostic.exists():
                raise ValueError('独立重载进程异常，保存未发布；请查看 reload.log')
            report=load(diagnostic)
            if report['status']!='physx_initialization_pass':
                raise ValueError('重新加载后有物体或柜门没停在当前状态，保存未发布')
            from rab_restore import rotation_difference
            max_drift=max(math.dist(expected[r['body']][:3],r['settled_pose'][:3]) for r in report['objects'])
            max_deg=max(math.degrees(rotation_difference(expected[r['body']][3:],r['settled_pose'][3:])) for r in report['objects'])
            if max_drift>.025 or max_deg>12:
                raise ValueError('重新加载后的位姿与当前设计不同，保存未发布')
            subjects={r['body']:r for r in self.world.rows}
            originals={x['body']:x['settled_pose'] for x in report['objects']}
            report['restoration_reference']={'bodies':{b:dict(pose=p,support=subjects[b]['support'],
                  target=subjects[b]['target']) for b,p in originals.items()},'articulations_m':{}}
            report.update(scene_sha256=episode['scene_sha256'],max_reload_drift_m=max_drift,max_reload_angle_deg=max_deg,
                          scope='Edited variant; native independent reload and initial stability only')
            build_cache(scene,self.member/'render_cache'/episode['scene_sha256'])
            save(revision/'episode.json',episode)
            save(revision/'physics.json',report)
            save(ptr,dict(revision=identifier,episode_sha256=digest(revision/'episode.json'),
                          physics_sha256=digest(revision/'physics.json')))
            self.revision=identifier

    def tick(self,camera):
        with self.state.condition:
            cmd=self.state.commands.popleft() if self.state.commands else None
        if cmd is None:
            return
        op=cmd['op']
        err=None
        try:
            if op=='select':
                self.selected=self.pick(camera,cmd['xy'])
            elif op in ('add','move'):
                row=self.world.item(self.pick(camera,cmd['from_xy'])) if op=='move' else None
                if op=='move' and row is None:
                    raise ValueError('请从一个 RAB 可移动物体开始拖动')
                self.place(self.pick(camera,cmd['xy']),row['asset'] if row else cmd['model'],
                           cmd.get('yaw'),row)
            elif op in ('open','close'):
                key=self.world.joint(self.selected) if self.selected else None
                if key is None:
                    raise ValueError('请先点击选中一个可交互的抽屉或冰箱门')
                self.world.swing(key,op=='open')
            elif op=='delete':
                r=self.world.item(self.selected) if self.selected else None
                if not r or not self.can_delete(r):
                    raise ValueError('此物品属于任务、遮挡、对照样品或承托关系，不能删除；已加物品可删除')
                path=r['body'].split('/Model',1)[0]
                self.stage.GetPrimAtPath(path).SetActive(False)
                self.world.sim.flush_changes()
                self.world.step(1)
                self.world.rows.remove(r)
                self.deleted.add(path)
                self.selected=None
            elif op=='save':
                self.ed['saving']=True
                self.publish()
            else:
                raise ValueError('请通过拖动和朝向选择框修改物体角度')
            if op!='select':
                self.ed['dirty']=op!='save'
            message='已保存；此任务的 RAB 缓存已更新' if op=='save' else '已执行'
        except Exception as exc:
            err=exc
            message=str(exc)
            import traceback
            log=self.member/'errors'/(cmd['command_id']+'.json')
            save(log,dict(message=message,detail=traceback.format_exc(),op=op))
            self.ed['errors']=(self.ed.get('errors',[])+[dict(id=cmd['command_id'],message=message,
                    detail=repr(exc),log=str(log))])[-10:]
        self.ed.update(saving=False,selected=self.description(),message=message,last_error=str(err) if err else None,
                       last_command=dict(id=cmd['command_id'],ok=err is None))
