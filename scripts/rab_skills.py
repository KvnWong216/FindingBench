"""Pixel-based FindingBench actions backed by RAB's native scene and robot."""
from __future__ import annotations

import base64
from collections import deque
import io
import json
import math
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

import numpy as np

from rab_edit import ROBOT, prefix
from rab_spectator_data import RAB_MIGRATION, save
from spectator import FlightCamera, Handler, SharedState

NATIVE_SOURCE = (RAB_MIGRATION / 'Runtime-room-0.1.7/short_tasks/agent_probe_v1'
                 / 'deployment_release/source')


class Refused(ValueError):
    def __init__(self,code,reason=None):
        super().__init__(reason or code)
        self.code,self.reason=code,reason


class SkillState(SharedState):
    def __init__(self,task):
        super().__init__(task)
        # Do not expose inspector metadata or path bindings through skill mode.
        self.status=dict(ready=False,error=None)
        self.commands=deque(maxlen=1)
        self.observation=None
        self.reply_id=None
        self.result=None


class SkillHandler(Handler):
    def do_GET(self):
        route=urlsplit(self.path).path
        if route=='/':
            return self.reply(Path(__file__).with_suffix('.html').read_bytes(),'text/html; charset=utf-8')
        if route in ('/observation','/state'):
            with self.state.condition:
                obs=self.state.observation
                summary=dict(ready=obs is not None,observation=obs,result=self.state.result,
                             error='RAB runtime not available' if self.state.status.get('error') else None)
                if route=='/state' and obs:
                    summary['observation']={k:v for k,v in obs.items() if k!='image_png_b64'}
            return self.reply(summary)
        if route not in ('/frame.jpg','/stream'):
            return self.reply({'error':'not found'},code=404)
        return super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path!='/action':
            return self.reply({'error':'not found'},code=404)
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=4096:
                raise ValueError('Invalid action size')
            action=json.loads(self.rfile.read(size))
            with self.state.condition:
                if not self.state.observation or self.state.commands or self.state.reply_id is not None:
                    return self.reply({'error':'not ready or action outstanding'},code=409)
                ident=uuid4().hex
                self.state.commands.append((ident,action))
                self.state.reply_id=ident
                ok=self.state.condition.wait_for(lambda:self.state.result and self.state.result['id']==ident or self.state.stopped,timeout=120)
                out=self.state.result
            if not ok or self.state.stopped:
                return self.reply({'error':'action deadline; check /state'},code=504)
            return self.reply(out)
        except (ValueError,TypeError) as e:
            return self.reply({'error':str(e)},code=400)


def encoded(image):
    from PIL import Image
    buffer=io.BytesIO()
    Image.fromarray(np.asarray(image,np.uint8)).save(buffer,format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


class Skills:
    def __init__(self,args,task,state,world,renderer,camera,receipt):
        from pydantic import TypeAdapter
        from rummagebench.core.public_types import PUBLIC_SKILLS, PublicAction
        from arena_api import bimanual as joints
        self.adapter=TypeAdapter(PublicAction)
        self.names=list(world.mobile.view.shared_metatype.dof_names)
        self.joints=joints
        self.library=PUBLIC_SKILLS
        self.args,self.task,self.state,self.world,self.renderer=args,task,state,world,renderer
        self.yaw=0.
        self.pitch=-12.
        self.zoom=1.
        self.frame=uuid4().hex
        self.counter=0
        self.horizon=120
        self.done=False
        self.episode_status='RUNNING'
        self.held=None
        self.output=args.run_output
        if self.output:
            self.output.mkdir(parents=True)
            save(self.output/'manifest.json',dict(id=task['id'],scene_sha256=world.episode['scene_sha256'],
                 protocol='findingbench_rab_pinhole_v1', executor='rab_physx',horizon=self.horizon))
        self.receipt=receipt
        self.observation_dirty=True
        self.public_feedback=None
        self.inspection=None
        self.details={}
        # Evaluator-private interaction trace used ONLY for deterministic
        # terminal-failure attribution (never returned across the MCP boundary).
        self.trace=dict(grasped=set(),checked=set(),opened=set(),placed=[])

    def target_rows(self):
        return [r for r in self.world.rows if r.get('target')]

    def terminal_attribution(self,reported,success):
        """Deterministic Layer-B attribution of the episode outcome.

        Every label is computed from simulator ground truth plus the private
        interaction trace; no visibility / "seen" judgement is involved.
        """
        if success:
            return 'SUCCESS'
        targets=self.target_rows()
        target_bodies={r['body'] for r in targets}
        destination={p['support'] for p in self.world.episode['delivery']['placements']}
        on_destination=[r for r in targets if self.support(r) in destination]
        if reported:
            if self.held and self.held[0]['body'] in target_bodies:
                return 'TARGET_NOT_RELEASED'
            if targets and len(on_destination)==len(targets):
                return 'TARGET_UNSTABLE'
            if on_destination:
                return 'PARTIAL_DELIVERY'
            if any(not p['target'] and p['support'] in destination for p in self.trace['placed']):
                return 'WRONG_TARGET_DELIVERED'
            if target_bodies & self.trace['grasped']:
                return 'WRONG_DESTINATION'
            if self.trace['grasped']-target_bodies:
                return 'WRONG_TARGET_MANIPULATED'
            return 'PREMATURE_DONE'
        if target_bodies & (self.trace['grasped']|self.trace['checked']):
            return 'TARGET_ENGAGED_NOT_DELIVERED'
        target_zones={r.get('zone') for r in targets if r.get('zone') in self.world.drives}
        if target_zones & self.trace['opened']:
            return 'CONTAINER_OPENED_TARGET_NOT_ENGAGED'
        return 'TARGET_NEVER_ENGAGED'

    def base(self):
        from scipy.spatial.transform import Rotation
        value=self.world.mobile.pose()
        direction=Rotation.from_quat(value[3:]).apply([1,0,0])
        return value[:3],math.atan2(direction[1],direction[0])

    def view(self):
        from rab_spectator import home
        cam=home(self.world.stage)
        p,base_yaw=self.base()
        # Apply the base's live PhysX XY offset relative to its initial spawn
        # position so the head camera retains its exact USD local offset on the robot.
        if not hasattr(self,'_spawn_base_xy'):
            self._spawn_base_xy=(float(p[0]),float(p[1]))
            self._spawn_cam_pos=list(map(float,cam.position))
        dx=float(p[0])-self._spawn_base_xy[0]
        dy=float(p[1])-self._spawn_base_xy[1]
        cam.position=[self._spawn_cam_pos[0]+dx,self._spawn_cam_pos[1]+dy,self._spawn_cam_pos[2]]
        head_yaw=float(self.world.mobile.view.get_dof_positions()[0,self.names.index('idx11_head_joint1')]) if 'idx11_head_joint1' in self.names else 0.
        cam.yaw=(math.degrees(base_yaw+head_yaw)+self.yaw+180)%360-180
        cam.pitch=float(np.clip(self.pitch,-89.,89.))
        cam.zoom=self.zoom
        return cam

    def observe(self,rgb):
        if not self.observation_dirty:
            return
        payload=dict(instruction=self.task['instruction'],frame_id=self.frame,
                     image_png_b64=encoded(rgb),planning_step=self.counter,max_planning_steps=self.horizon,
                     skill_library=self.library,feedback=self.public_feedback,inspection_view=self.inspection)
        with self.state.condition:
            self.state.observation=payload
        self.observation_dirty=False
        if self.output:
            save(self.output/'observations'/f'{self.counter:04}.json',payload)

    def clear_base(self,start,target):
        # Check measured height bands, not semantic furniture classes.
        v=target[:2]-start[:2]
        d=np.linalg.norm(v)
        if not .06<d<=5.:
            return False
        v=v/d
        side=np.array([-v[1],v[0]])
        for u in (-.40,-.22,0,.22,.40):
            for z in (.13,.3,.53,.78,1.05,1.35):
                for hit in self.world.rays([*(start[:2]+side*u+v*.05),start[2]+z],[*v,0],d+.30,ignore=(ROBOT,)):
                    return False
        for f in np.linspace(0,1, max(3,int(d/.20)+1)):
            for u in (-.24,.24):
                point=start[:2]+d*f*v+side*u
                ground=self.world.rays([*point,start[2]+.12],[0,0,-1],.32,ignore=(ROBOT,))
                if not ground or ground[0]['normal'][2]<.8 or abs(ground[0]['position'][2]-start[2])>.075:
                    return False
        return True

    def reanchor_mobile(self):
        mobile=self.world.mobile
        if mobile.mode!='anchor':
            from arena_api.mobile_base import ZERO
            if any(abs(v)>1e-8 for v in mobile.previous.values()):
                mobile.apply(dict(ZERO))
                self.world.step(1)
            pose=mobile.pose()
            mobile.anchor.GetLocalPos0Attr().Set(mobile.Gf.Vec3f(*map(float,pose[:3])))
            mobile.anchor.GetLocalRot0Attr().Set(mobile.Gf.Quatf(float(pose[6]),mobile.Gf.Vec3f(*map(float,pose[3:6]))))
            for jnt in mobile.joints.values():
                jnt.GetJointEnabledAttr().Set(False)
            for drv in mobile.drives.values():
                drv.GetTargetVelocityAttr().Set(0.)
            mobile.anchor.GetJointEnabledAttr().Set(True)
            mobile.flush()
            mobile.mode='anchor'
            mobile.previous=dict(ZERO)
            mobile.verify()

    def motion(self,kind,amount):
        if abs(amount)<.015:
            return
        if kind=='turn' and abs(amount)>math.radians(145):
            self.motion(kind,amount/2)
            self.motion(kind,amount/2)
            return
        from arena_api.mobile_base import NAMES
        start,yaw=self.base()
        index=0 if kind=='drive' else 1
        speed=.18 if index==0 else .25
        accel=.12 if index==0 else .30
        sign=math.copysign(1,amount)
        dt=.05
        prev=progress=0.
        def send(v):
            pair=[0.,0.]
            pair[index]=sign*v
            self.world.mobile.apply(dict(zip(NAMES,pair)))
            self.world.step(3,sync=False)
        try:
            for _ in range(min(1100,int(abs(amount)/(.07*dt))+120)):
                pos,y=self.base()
                progress=sign*(float(np.dot(pos[:2]-start[:2],[math.cos(yaw),math.sin(yaw)])) if index==0 else angle(y-yaw))
                left=abs(amount)-progress
                want=min(speed,math.sqrt(2*accel*max(0.,left-.004)))
                new=float(np.clip(want,max(0.,prev-accel*dt),prev+accel*dt))
                send(new)
                prev=new
                if left<(.022 if index==0 else .012) and prev<=accel*dt+1e-7:
                    break
            while prev>1e-8:
                prev=max(0.,prev-accel*dt)
                send(prev)
            self.world.step(6)
        finally:
            # Reanchor at measured rest after each command so repeated turns do not
            # accumulate against the controller's relative ±170° mechanical stop,
            # and a failed drive never leaves the prismatic constraint locked.
            self.reanchor_mobile()
        if abs(abs(amount)-progress)>(.075 if index==0 else .10):
            raise Refused('UNSAFE',f'BASE_MOTION_SHORTFALL:{kind}:wanted={amount:.4f}:actual={progress:.4f}')

    def teleport_base(self,xyz,yaw):
        from scipy.spatial.transform import Rotation
        mobile=self.world.mobile
        from arena_api.mobile_base import ZERO
        if any(abs(v)>1e-8 for v in mobile.previous.values()):
            mobile.apply(dict(ZERO))
            self.world.step(1)
        # Note: mobile.pose() returns [x, y, z, qw, qx, qy, qz] (w-first), whereas
        # Rotation.as_quat() returns [qx, qy, qz, qw] (w-last).
        q_xyzw=Rotation.from_euler('z',float(yaw)).as_quat()
        qx,qy,qz,qw=float(q_xyzw[0]),float(q_xyzw[1]),float(q_xyzw[2]),float(q_xyzw[3])
        root=np.array([[float(xyz[0]),float(xyz[1]),float(xyz[2]),qw,qx,qy,qz]],np.float32)
        idx=np.array([0],np.uint32)
        for jnt in mobile.joints.values():
            jnt.GetJointEnabledAttr().Set(False)
        for drv in mobile.drives.values():
            drv.GetTargetVelocityAttr().Set(0.)
        mobile.anchor.GetJointEnabledAttr().Set(False)
        mobile.flush()
        mobile.view.set_root_transforms(root,idx)
        mobile.view.set_root_velocities(np.zeros((1,6),np.float32),idx)
        mobile.anchor.GetLocalPos0Attr().Set(mobile.Gf.Vec3f(float(xyz[0]),float(xyz[1]),float(xyz[2])))
        mobile.anchor.GetLocalRot0Attr().Set(mobile.Gf.Quatf(qw,mobile.Gf.Vec3f(qx,qy,qz)))
        mobile.anchor.GetJointEnabledAttr().Set(True)
        mobile.flush()
        mobile.mode='anchor'
        mobile.previous=dict(ZERO)
        mobile.verify()
        self.world.step(2,sync=True)
        # Verify whether set_root_transforms expects [qw,qx,qy,qz] or [qx,qy,qz,qw]
        _,measured_yaw=self.base()
        if abs(angle(measured_yaw-float(yaw)))>.05:
            root_alt=np.array([[float(xyz[0]),float(xyz[1]),float(xyz[2]),qx,qy,qz,qw]],np.float32)
            mobile.anchor.GetJointEnabledAttr().Set(False)
            mobile.flush()
            mobile.view.set_root_transforms(root_alt,idx)
            mobile.view.set_root_velocities(np.zeros((1,6),np.float32),idx)
            mobile.anchor.GetLocalPos0Attr().Set(mobile.Gf.Vec3f(float(xyz[0]),float(xyz[1]),float(xyz[2])))
            mobile.anchor.GetLocalRot0Attr().Set(mobile.Gf.Quatf(qw,mobile.Gf.Vec3f(qx,qy,qz)))
            mobile.anchor.GetJointEnabledAttr().Set(True)
            mobile.flush()
            self.world.step(2,sync=True)

    def turn(self,amount):
        if abs(amount)<.015:
            return
        p,yaw=self.base()
        for z in (.17,.4,.7,1.1):
            for theta in np.arange(0,2*math.pi,math.pi/8):
                hits=self.world.rays([p[0],p[1],p[2]+z],[math.cos(theta),math.sin(theta),0],.38,ignore=(ROBOT,))
                if hits:
                    raise Refused('UNSAFE',f'TURN_RADIAL_CLEARANCE_BLOCKED:z={z:.2f}:dist={hits[0]["distance"]:.3f}:hit={hits[0]["path"]}')
        self.teleport_base(p,angle(yaw+amount))

    def free_stand(self,xy):
        for z in (.17,.4,.7,1.1):
            for theta in np.arange(0,2*math.pi,math.pi/8):
                if self.world.rays([float(xy[0]),float(xy[1]),float(xy[2])+z],[math.cos(theta),math.sin(theta),0],.39,ignore=(ROBOT,)):
                    return False
        return True

    def route(self,start,goals):
        from heapq import heappop,heappush
        goals=[g for g in goals if self.free_stand(g)]
        if not goals:
            return None
        for end in goals:
            if self.clear_base(start,end):
                return [end]
        # Lazy, geometric A*: queries go to live native collision geometry.
        # Limit the planning graph, and verify every shortcut again.
        stride=.36
        home=np.asarray(start,float)
        origin=(0,0)
        directions=((-1,0),(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1))
        def position(k):
            return home+np.r_[np.array(k)*stride,0.]
        def heuristic(k):
            at=position(k)
            return min(float(np.linalg.norm(at[:2]-g[:2])) for g in goals)
        heap=[(heuristic(origin),0.,origin)]
        scores={origin:0.}
        parents={}
        edges={}
        expanded=0
        while heap and expanded<850:
            _,length,node=heappop(heap)
            if length!=scores.get(node):
                continue
            expanded+=1
            point=position(node)
            for goal in goals:
                if np.linalg.norm(goal[:2]-point[:2])<.32 and self.clear_base(point,goal):
                    path=[goal]
                    while node!=origin:
                        path.append(position(node))
                        node=parents[node]
                    path=path[::-1]
                    result=[]
                    current=start
                    while path:
                        far=max((i for i,v in enumerate(path) if self.clear_base(current,v)),default=0)
                        current=path[far]
                        result.append(current)
                        path=path[far+1:]
                    return result
            for dx,dy in directions:
                nxt=node[0]+dx,node[1]+dy
                if max(abs(nxt[0]),abs(nxt[1]))>34:
                    continue
                nl=length+stride*math.hypot(dx,dy)
                if nl>=scores.get(nxt,float('inf')):
                    continue
                edge=tuple(sorted((node,nxt)))
                if edge not in edges:
                    edges[edge]=self.clear_base(point,position(nxt))
                if edges[edge]:
                    scores[nxt]=nl
                    parents[nxt]=node
                    heappush(heap,(nl+heuristic(nxt),nl,nxt))
        return None

    def move_to(self,hit):
        p,initial_yaw=self.base()
        route_origin=p.copy()
        target=np.asarray(hit['position'],float)
        floor=abs(target[2]-p[2])<.10 and hit['normal'][2]>.75
        if floor:
            points=[target]
        else:
            theta=math.atan2(p[1]-target[1],p[0]-target[0])
            points=[np.r_[target[:2]+r*np.array([math.cos(theta+a),math.sin(theta+a)]),p[2]]
                    for r in (.48,.53,.58,.64,.70,.76,.82) for a in (0,-.35,.35,-.7,.7,-1.05,1.05,-1.4,1.4,-1.75,1.75,-2.1,2.1)]
        route=self.route(p,points)
        if not route:
            raise Refused('UNSAFE','NAV_NO_CLEAR_ROUTE')
        for point in route:
            p,yaw=self.base()
            if not self.clear_base(p,point):
                raise Refused('UNSAFE','NAV_SEGMENT_CLEARANCE_BLOCKED')
            direction=math.atan2(point[1]-p[1],point[0]-p[0])
            self.teleport_base(point,direction)
        now,yaw=self.base()
        desired=math.atan2(target[1]-now[1],target[0]-now[0]) if not floor else initial_yaw
        self.teleport_base(now,desired)
        final,_=self.base()
        self.details.update(start_base_xyz_m=list(map(float,route_origin)),end_base_xyz_m=final.tolist(),
                            planned_waypoints=len(route),terminal_error_m=float(np.linalg.norm(final[:2]-route[-1][:2])),
                            execution='collision_validated_endpoint')

    def bounds(self,row):
        from pxr import UsdGeom
        from rab_build_search import bounds
        from rab_edit import matrix, rigid_part
        root=row['body'].split('/Model',1)[0]
        box=bounds(UsdGeom.BBoxCache(0,['render','default','proxy','guide']),self.world.stage.GetPrimAtPath(root))
        if box is None:
            return None
        # Native PhysX rigid-body transforms do not automatically rewrite ancestor
        # Xforms above row['body'] when updateToUsd is disabled during fast steps;
        # apply the live PhysX translation offset of row['body'] relative to its USD transform.
        usd_t=UsdGeom.XformCache().GetLocalToWorldTransform(self.world.stage.GetPrimAtPath(row['body'])).ExtractTranslation()
        phys_p=self.world.pose(row['body'])[:3]
        delta=np.array([float(phys_p[0]-usd_t[0]),float(phys_p[1]-usd_t[1]),float(phys_p[2]-usd_t[2])])
        return box+delta

    def check(self,row):
        from PIL import Image
        from pxr import Usd,UsdGeom
        stage=self.world.stage
        # Isolate the selected visible object's exterior in the live renderer;
        # all scene facts, transforms and physical state stay unchanged.
        item=row['body'].split('/Model',1)[0]
        touched=[]
        try:
            for p in stage.Traverse():
                path=str(p.GetPath())
                if not path.startswith('/World/') or prefix(item,path) or prefix(path,item):
                    continue
                if any(prefix(path,x[0]) for x in touched):
                    continue
                if (p.GetTypeName() in ('DomeLight','DistantLight','RectLight','SphereLight','DiskLight')
                        or not p.IsA(UsdGeom.Imageable)):
                    continue
                # Hide top-level groups only if they do not contain a light.
                if any(x.GetTypeName().endswith('Light') for x in Usd.PrimRange(p)):
                    continue
                a=UsdGeom.Imageable(p).GetVisibilityAttr()
                touched.append((path,a,a.Get()))
                a.Set('invisible')
            box=self.bounds(row)
            center=box.mean(axis=0)
            span=float(np.ptp(box,axis=0).max())
            imgs=[]
            for yaw in (30,120,210,300):
                th=math.radians(yaw)
                rr=max(.13,span*1.65)
                eye=center+rr*np.array([-math.cos(th),-math.sin(th),.34])
                cam=FlightCamera(eye,yaw,-math.degrees(math.atan(.34)))
                imgs.append(Image.fromarray(self.renderer.capture(cam)).resize((360,225)))
            sheet=Image.new('RGB',(720,450))
            for i,img in enumerate(imgs):
                sheet.paste(img,(i%2*360,i//2*225))
            return dict(frame_id=uuid4().hex,image_png_b64=encoded(sheet),width=720,height=450,
                source='isolated_object_multiview')
        finally:
            for _,attr,value in reversed(touched):
                attr.Set(value)

    def arm(self,position, side='r', *, tolerance=.022):
        # Native articulation Jacobians and position drives; no pose setter.
        robot=self.world.mobile.view
        jmeta=robot.shared_metatype
        arm=list(self.joints.RIGHT_ARM if side=='r' else self.joints.LEFT_ARM)
        for body_j in ('idx01_body_joint1','idx02_body_joint2'):
            if body_j in self.names and body_j not in arm:
                arm.append(body_j)
        ix=[self.names.index(x) for x in arm]
        link=f'gripper_{side}_center_link'
        links=list(jmeta.link_names)
        lim=robot.get_dof_limits()[0,ix]
        idx=np.array([0],np.uint32)
        for _ in range(135):
            pose=self.world.pose(ROBOT+'/'+link)
            err=np.asarray(position)-pose[:3]
            if np.linalg.norm(err)<=tolerance:
                return
            jac=robot.get_jacobians()[0]
            i=links.index(link)-(1 if jac.shape[0]==len(links)-1 else 0)
            offset=jac.shape[-1]-len(self.names)
            j=jac[i,:3,:][:,np.array(ix)+offset]
            d=j.T@np.linalg.solve(j@j.T+np.eye(3)*.018**2,err)
            q=robot.get_dof_position_targets()
            measured=robot.get_dof_positions()[0,ix]
            q[0,ix]=np.clip(measured+np.clip(d,-.08,.08),lim[:,0],lim[:,1])
            robot.set_dof_position_targets(q,idx)
            self.world.step(5,sync=False)
        raise Refused('OUT_OF_CAPABILITY','ARM_REACH_FAILED')

    def grip(self,opened,side):
        robot=self.world.mobile.view
        grip=1 if side=='r' else 0
        outer=self.joints.GRIPPERS[grip]
        from arena_api.robot_profiles import INNER
        q=robot.get_dof_position_targets()
        value=.64 if opened else .01
        q[0,self.names.index(outer)]=value
        q[0,self.names.index(INNER[grip])]=-value
        robot.set_dof_position_targets(q,np.array([0],np.uint32))
        self.world.step(35)
    
    def arm_path_clear(self,start,end,allowed):
        v=np.asarray(end)-start
        d=float(np.linalg.norm(v))
        if d<.005:
            return
        for extra in (np.zeros(3),np.array([0,0,.03]),np.array([0,0,-.025])):
            hits=self.world.rays(np.asarray(start)+extra,v/d,d,ignore=(ROBOT,*allowed))
            if hits:
                raise Refused('UNSAFE','HAND_PATH_BLOCKED:hit='+hits[0]['path'])

    def set_body_pose(self,body,pose,*,disable_gravity=None):
        import omni.physics.tensors
        from pxr import PhysxSchema
        if disable_gravity is not None:
            prim=self.world.stage.GetPrimAtPath(body)
            if prim:
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(bool(disable_gravity))
                self.world.sim.flush_changes()
        view=omni.physics.tensors.create_simulation_view('numpy',self.world.sid.ToLongInt())
        actor=view.create_rigid_body_view(body)
        if actor.count!=1:
            raise ValueError(f'Rigid body view not found for {body}')
        idx=np.array([0],np.uint32)
        actor.set_transforms(np.asarray([pose],np.float32),idx)
        actor.set_velocities(np.zeros((1,6),np.float32),idx)
        self.world.step(1)

    def sync_held(self):
        if not self.held:
            return
        row,_=self.held
        p,yaw=self.base()
        # Standardized carried pose above the robot base (gravity disabled while held
        # so it stays stationary in air during multi-step drawer opening/closing).
        pose=self.world.pose(row['body']).copy()
        pose[:3]=[float(p[0]),float(p[1]),float(p[2]+2.40)]
        row['support']=ROBOT
        self.set_body_pose(row['body'],pose,disable_gravity=True)

    def ik_reachable(self,position,side='r',*,tolerance=.040):
        # Verify kinematic reachability using native articulation Jacobians,
        # then restore the robot's ready arm configuration (FindingBench contract:
        # IK + collision check -> direct primitive realization).
        robot=self.world.mobile.view
        jmeta=robot.shared_metatype
        arm=list(self.joints.RIGHT_ARM if side=='r' else self.joints.LEFT_ARM)
        for body_j in ('idx01_body_joint1','idx02_body_joint2'):
            if body_j in self.names and body_j not in arm:
                arm.append(body_j)
        ix=[self.names.index(x) for x in arm]
        link=f'gripper_{side}_center_link'
        links=list(jmeta.link_names)
        lim=robot.get_dof_limits()[0,ix]
        idx=np.array([0],np.uint32)
        saved_q=robot.get_dof_positions().copy()
        saved_t=robot.get_dof_position_targets().copy()
        try:
            if float(position[2])<.45:
                seed_q=saved_q.copy()
                seed_t=saved_t.copy()
                if 'idx01_body_joint1' in self.names:
                    j1=self.names.index('idx01_body_joint1')
                    seed_q[0,j1]=.02
                    seed_t[0,j1]=.02
                if 'idx02_body_joint2' in self.names:
                    j2=self.names.index('idx02_body_joint2')
                    seed_q[0,j2]=.65
                    seed_t[0,j2]=.65
                robot.set_dof_positions(seed_q,idx)
                robot.set_dof_position_targets(seed_t,idx)
                self.world.step(2,sync=True)
            for _ in range(135):
                pose=self.world.pose(ROBOT+'/'+link)
                err=np.asarray(position)-pose[:3]
                if np.linalg.norm(err)<=tolerance:
                    return float(np.linalg.norm(err))
                jac=robot.get_jacobians()[0]
                i=links.index(link)-(1 if jac.shape[0]==len(links)-1 else 0)
                offset=jac.shape[-1]-len(self.names)
                j=jac[i,:3,:][:,np.array(ix)+offset]
                d=j.T@np.linalg.solve(j@j.T+np.eye(3)*.018**2,err)
                q=robot.get_dof_positions().copy()
                q[0,ix]=np.clip(q[0,ix]+np.clip(d,-.12,.12),lim[:,0],lim[:,1])
                robot.set_dof_positions(q,idx)
                robot.set_dof_position_targets(q,idx)
                self.world.step(1,sync=True)
            pose=self.world.pose(ROBOT+'/'+link)
            err=float(np.linalg.norm(np.asarray(position)-pose[:3]))
            raise Refused('OUT_OF_CAPABILITY',f'ARM_IK_TARGET_FAILED:err={err:.4f}:pos={list(map(float,position))}:hand={list(map(float,pose[:3]))}')
        finally:
            robot.set_dof_positions(saved_q,idx)
            robot.set_dof_velocities(np.zeros_like(saved_q),idx)
            robot.set_dof_position_targets(saved_t,idx)
            self.world.step(2,sync=True)

    def grasp(self,row):
        if self.held:
            raise Refused('INVALID_ACTION','PRECONDITION_HAND_OCCUPIED')
        p,yaw=self.base()
        box=self.bounds(row)
        origin=self.world.pose(row['body']).copy()
        target=box.mean(axis=0)
        reach=float(np.linalg.norm(target[:2]-p[:2]))
        if reach>.92:
            raise Refused('OUT_OF_CAPABILITY',f'GRASP_TOO_FAR:dist={reach:.3f}')
        if np.min(np.ptp(box,axis=0))>.16:
            raise Refused('UNSAFE',f'UNSAFE_OVERSIZED_GRASP:min_extent={float(np.min(np.ptp(box,axis=0))):.3f}')
        side='l' if np.dot(target[:2]-p[:2],[-math.sin(yaw),math.cos(yaw)])>.06 else 'r'
        path=f'{ROBOT}/gripper_{side}_center_link'
        toward=np.r_[math.cos(yaw),math.sin(yaw),0.]
        z_key=row.get('zone')
        in_open_drawer=z_key in self.world.opened and z_key in self.world.drives
        pre=target-.10*toward+(.14 if in_open_drawer else .065)*np.array([0.,0.,1.])
        allowed=[row['body']]
        if in_open_drawer:
            allowed.append(self.world.drives[z_key][2])
        hand=self.world.pose(path)[:3]
        if in_open_drawer:
            # Verify vertical reach from above the open drawer cavity down to target
            self.arm_path_clear(pre,target,tuple(allowed))
        else:
            self.arm_path_clear(hand,pre,tuple(allowed))
            self.arm_path_clear(pre,target,tuple(allowed))
        try:
            ik_err=self.ik_reachable(target,side)
        except Refused as e:
            raise Refused(e.code,e.reason or 'ARM_IK_TARGET_FAILED')
        # IK + collision check passed -> standardized realization
        self.held=(row,side)
        self.trace['grasped'].add(row['body'])
        self.sync_held()
        final=self.world.pose(row['body'])[:3]
        self.details.update(grasp_body=row['body'],lift_m=float(final[2]-origin[2]),
                            grasp_is_target=bool(row.get('target')),
                            native_gripper=side,ik_error_m=ik_err,
                            realization='ik_collision_validated_direct')

    def support(self,row):
        box=self.bounds(row)
        p=self.world.pose(row['body'])[:3]
        center_xy=box.mean(axis=0)[:2] if box is not None else p[:2]
        bottom_z=float(box[0,2]) if box is not None else float(p[2])
        for h in self.world.rays([*center_xy,bottom_z+.026],[0,0,-1],.14,ignore=(ROBOT,row['body'],row['body'].split('/Model',1)[0])):
            if h['normal'][2]>.75:
                beneath=self.world.item(h)
                if beneath:
                    return beneath['body']
                if '/Box_1/Object_4/' in h['path']:
                    continue
                return h['path']
        # For items resting inside a drawer where BBox bottom_z is within a few
        # millimeters of the drawer floor collider, cast from slightly higher and
        # resolve stacked items inside the drawer first.
        z_key=row.get('zone')
        if z_key in self.world.drives:
            drawer_body=self.world.drives[z_key][2]
            for h in self.world.rays([*center_xy,float(p[2])+.045],[0,0,-1],.18,ignore=(ROBOT,row['body'],row['body'].split('/Model',1)[0])):
                if h['normal'][2]>.75:
                    beneath=self.world.item(h)
                    if beneath:
                        return beneath['body']
                    if '/Box_1/Object_4/' in h['path']:
                        continue
                    if prefix(h['path'],drawer_body) or prefix(h['body'],drawer_body):
                        return h['path']
        return None

    def place(self,hit):
        if not self.held:
            raise Refused('INVALID_ACTION','PRECONDITION_HAND_EMPTY')
        if hit['normal'][2]<.8:
            raise Refused('INVALID_ACTION',f'CLICK_NON_HORIZONTAL_SURFACE:nz={float(hit["normal"][2]):.3f}')
        row,side=self.held
        selected_support=self.world.item(hit)
        if selected_support is row:
            raise Refused('INVALID_ACTION','PLACE_ON_SELF')
        requested=selected_support['body'] if selected_support else hit['path']
        point=np.asarray(hit['position'],float)
        p,yaw=self.base()
        reach=float(np.linalg.norm(point[:2]-p[:2]))
        if reach>.92:
            raise Refused('OUT_OF_CAPABILITY',f'PLACE_TOO_FAR:dist={reach:.3f}')
        place_side='l' if np.dot(point[:2]-p[:2],[-math.sin(yaw),math.cos(yaw)])>.06 else 'r'
        path=f'{ROBOT}/gripper_{place_side}_center_link'
        hand=self.world.pose(path)[:3]
        high=point+np.array([0.,0.,.08])
        self.arm_path_clear(hand,high,(row['body'],))
        try:
            ik_err=self.ik_reachable(high,place_side)
        except Refused as e:
            raise Refused(e.code,e.reason or 'ARM_IK_PLACE_FAILED')
        # Compute object placement transform above clicked support surface
        pose=self.world.pose(row['body']).copy()
        box=self.bounds(row)
        center_xy=box.mean(axis=0)[:2]
        bottom_z=float(box[0,2])
        new_pose=pose.copy()
        new_pose[0]+=float(point[0]-center_xy[0])
        new_pose[1]+=float(point[1]-center_xy[1])
        new_pose[2]+=float(point[2]+.008-bottom_z)
        self.set_body_pose(row['body'],new_pose,disable_gravity=False)
        self.world.step(65)
        support=self.support(row)
        error=float(np.linalg.norm(self.bounds(row).mean(axis=0)[:2]-point[:2]))
        self.details.update(place_body=row['body'],observed_support=support,intended_support=requested,
                            placement_error_m=error,ik_error_m=ik_err,
                            realization='ik_collision_validated_direct')
        if support!=requested or error>.12:
            # Roll back to held state if target support failed settle check
            self.sync_held()
            raise Refused('UNSAFE',f'UNSAFE_PLACEMENT_UNSTABLE:support_match={support==requested}:err={error:.3f}')
        row['support']=support
        self.trace['placed'].append(dict(body=row['body'],support=support,target=bool(row.get('target'))))
        self.held=None

    def restoration(self):
        from rab_restore import score_restoration,rotation_difference
        ref=self.receipt.get('dense_checks',{}).get('restoration_reference') or self.receipt.get('restoration_reference')
        if not ref:
            return None
        ref=dict(ref,articulations_m=ref.get('articulations_m',{}))
        measured={}
        for row in self.world.rows:
            name=row['body']
            pose=self.world.pose(name)
            z=row['zone']
            record=dict(pose=pose.tolist(),support=self.support(row))
            if z in self.world.drives and 'drawer_relative_xyz_m' in ref['bodies'].get(name,{}):
                record['drawer_relative_xyz_m']=(pose[:3]-self.world.pose(self.world.drives[z][2])[:3]).tolist()
            measured[name]=record
        actual={k:(float(np.linalg.norm(self.world.pose(self.world.drives[k][2])[:3]-self.hinge_start[k][:3]))
                   if k in self.hinge_start else None) for k in ref['articulations_m']}
        result=score_restoration(ref,dict(bodies=measured,articulations_m=actual))['score']
        if ref.get('articulations_deg'):
            # Measure native geometry relative to the starting hinge pose.
            values=[math.degrees(rotation_difference(self.hinge_start[k][3:],self.world.pose(self.world.drives[k][2])[3:]))
                    for k in ref['articulations_deg'] if k in self.hinge_start]
            items=sum(not v['target'] for v in ref['bodies'].values())
            if values:
                result=((result or 0.)*items+sum(math.exp(-.5*(v/12.)**2) for v in values))/(items+len(values))
        return result

    def act(self,raw):
        if self.done:
            raise Refused('INVALID_ACTION','EPISODE_ALREADY_TERMINATED')
        action=self.adapter.validate_python(raw)
        skill=action.skill
        self.inspection=None
        if skill=='VIEW':
            self.yaw,self.pitch,self.zoom=action.yaw,action.pitch,action.zoom
        elif skill=='TURN':
            self.turn(math.radians(dict(left=90,right=-90,back=180)[action.direction]))
            self.sync_held()
            self.yaw=0.
            self.pitch=-12.
            self.zoom=1.
        elif skill=='REPORT_DONE':
            self.done=True
            targets=self.target_rows()
            destination={p['support'] for p in self.world.episode['delivery']['placements']}
            before=[self.world.pose(r['body']) for r in targets]
            self.world.step(30)
            success=bool(targets and not self.held and all(self.support(r) in destination and
                math.dist(p[:3],self.world.pose(r['body'])[:3])<.012 for p,r in zip(before,targets)))
            self.details.update(terminal_attribution=self.terminal_attribution(True,success))
            return dict(success=success,restoration=self.restoration())
        else:
            if action.point.frame_id!=self.frame:
                raise Refused('INVALID_ACTION','STALE_FRAME_ID')
            cam=self.view()
            angle_fov=math.degrees(2*math.atan(math.tan(math.radians(self.args.fov/2))/self.zoom))
            hit=self.world.hit(cam,[action.point.x,action.point.y],angle_fov,self.args.width/self.args.height,actor=True)
            item=self.world.item(hit)
            if skill=='MOVE_TO':
                self.move_to(hit)
                self.sync_held()
                self.yaw=0.
                self.pitch=-12.
                self.zoom=1.
            elif skill=='CHECK' and item:
                self.inspection=self.check(item)
                self.trace['checked'].add(item['body'])
                self.details.update(check_body=item['body'],check_is_target=bool(item.get('target')))
            elif skill in ('OPEN','CLOSE'):
                key=self.world.joint(hit)
                if key is None:
                    raise Refused('INVALID_ACTION','CLICK_NON_ARTICULATED')
                reach=float(np.linalg.norm(np.asarray(hit['position'])[:2]-self.base()[0][:2]))
                if reach>.94:
                    raise Refused('OUT_OF_CAPABILITY',f'ARTICULATION_TOO_FAR:dist={reach:.3f}')
                if (key in self.world.opened)==(skill=='OPEN'):
                    raise Refused('INVALID_ACTION','PRECONDITION_ALREADY_OPEN' if skill=='OPEN'
                                  else 'PRECONDITION_ALREADY_CLOSED')
                self.details.update(native_joint=key,**self.world.swing(key,skill=='OPEN'))
                if skill=='OPEN':
                    self.trace['opened'].add(key)
            elif skill=='GRASP' and self.held:
                raise Refused('INVALID_ACTION','PRECONDITION_HAND_OCCUPIED')
            elif skill=='GRASP' and item:
                self.grasp(item)
            elif skill=='PLACE':
                self.place(hit)
            elif skill in ('CHECK','GRASP'):
                raise Refused('INVALID_ACTION','CLICK_NON_INTERACTABLE')
            else:
                raise Refused('INVALID_ACTION','UNSUPPORTED_SKILL')
        return {}

    def tick(self,camera):
        from rummagebench.core.public_types import PublicActionFeedback, FEEDBACK_MESSAGES
        if not hasattr(self,'hinge_start'):
            self.hinge_start={k:self.world.pose(r[2]) for k,r in self.world.drives.items()}
        with self.state.condition:
            pair=self.state.commands.popleft() if self.state.commands else None
        result=None
        if pair:
            ident,raw=pair
            detail=None
            self.details={}
            extras={}
            if self.done:
                code='INVALID_ACTION'
                detail='EPISODE_ALREADY_TERMINATED'
                feedback=dict(code=code,message=FEEDBACK_MESSAGES.get(PublicActionFeedback(code),'Invalid action'))
                result=dict(id=ident,feedback=feedback,terminal=True,planning_step=self.counter,
                            episode_status=getattr(self,'episode_status','FAIL_MAX_STEPS'))
            else:
                try:
                    extras=self.act(raw)
                    code='EXECUTED'
                except Refused as e:
                    code=e.code
                    detail=e.reason
                except ValueError as e:
                    code='INVALID_ACTION'
                    detail='SCHEMA_INVALID:'+str(e).replace('\n',' ')[:400]
                except Exception:
                    import traceback
                    code='UNSAFE'
                    # Harness/runtime exception: attributed to infrastructure,
                    # never to the agent, by the offline metrics.
                    detail='INFRA_EXCEPTION:'+traceback.format_exc()
                    print(detail,flush=True)
                self.counter+=1
                if self.done:
                    # REPORT_DONE set self.done = True inside self.act(raw)
                    self.episode_status='SUCCESS' if extras.get('success') else 'FAIL_FALSE_COMPLETION'
                elif self.counter>=self.horizon:
                    self.done=True
                    self.episode_status='FAIL_MAX_STEPS'
                    extras=dict(success=False,restoration=self.restoration())
                    self.details['terminal_attribution']=self.terminal_attribution(False,False)
                else:
                    self.episode_status='RUNNING'
                self.frame=uuid4().hex
                self.public_feedback=dict(code=code,message=FEEDBACK_MESSAGES.get(PublicActionFeedback(code),'Executed'))
                result=dict(id=ident,feedback=self.public_feedback,terminal=self.done,
                            planning_step=self.counter,episode_status=self.episode_status,**extras)
                if self.output:
                    private_record=dict(raw=raw,result=result,private_error=detail,
                          physx_steps=self.world.steps,measurements=self.details)
                    if 'restoration' not in result:
                        private_record['restoration']=self.restoration()
                    save(self.output/'actions'/f'{self.counter:04}.json',private_record)
                self.observation_dirty=True
        actual=self.view()
        camera.position,camera.yaw,camera.pitch,camera.zoom=actual.position,actual.yaw,actual.pitch,actual.zoom
        if self.observation_dirty:
            rgb=self.renderer.capture(camera)
            self.observe(rgb)
        if result:
            with self.state.condition:
                self.state.result=result
                self.state.reply_id=None
                self.state.condition.notify_all()


def angle(value):
    return math.atan2(math.sin(value),math.cos(value))
