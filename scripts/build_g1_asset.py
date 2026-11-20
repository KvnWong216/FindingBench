"""Build the official Unitree G1/Dex1-1 Oracle-navigation variant.

Preserves official meshes and arm/waist/gripper kinematics. Legs are welded
at their neutral pose; a planar virtual base implements the benchmark's
existing controller-free navigation. This is not a biped walking policy.
"""
from pathlib import Path
import argparse, json, math, subprocess, xml.etree.ElementTree as ET
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/robots/unitree_source'
OUT=ROOT/'build/robots/g1_bench'
MODEL='g1_bench'

def prepare():
    import pinocchio as pin
    import trimesh
    src=SOURCE/'robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf'
    OUT.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(src);r=tree.getroot();r.set('name',MODEL)
    for m in r.findall('.//mesh'):
        path=(src.parent/m.get('filename')).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        m.set('filename',str(path))
    frozen=[]
    for j in r.findall('joint'):
        if any(part in j.get('name') for part in ('hip_', 'knee_', 'ankle_')):
            frozen.append(j.get('name'));j.set('type','fixed')
            for tag in ('limit','axis','dynamics'):
                e=j.find(tag)
                if e is not None:j.remove(e)
    tmp=OUT/'neutral.urdf';tree.write(tmp)
    m=pin.buildModelFromUrdf(str(tmp));d=m.createData();pin.framesForwardKinematics(m,d,pin.neutral(m))
    bounds=[]
    for link in r.findall('link'):
        transform=d.oMf[m.getFrameId(link.get('name'))].homogeneous
        for c in link.findall('collision'):
            mesh=c.find('geometry/mesh')
            if mesh is None:continue
            t=np.eye(4);o=c.find('origin')
            if o is not None:
                t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                t[:3,:3]=pin.rpy.rpyToMatrix(np.fromstring(o.get('rpy','0 0 0'),sep=' '))
            shape=trimesh.load(mesh.get('filename'),force='mesh')
            vertices=np.asarray(shape.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            T=transform@t;bounds.append(vertices@T[:3,:3].T+T[:3,3])
    vertices=np.concatenate(bounds);height=-float(vertices[:,2].min())
    def link(name):
        e=ET.SubElement(r,'link',name=name);i=ET.SubElement(e,'inertial');ET.SubElement(i,'mass',value='.01');ET.SubElement(i,'inertia',ixx='.00001',iyy='.00001',izz='.00001',ixy='0',ixz='0',iyz='0')
    def joint(name,parent,child,kind='fixed',xyz='0 0 0',axis='0 0 1',rpy='0 0 0'):
        e=ET.SubElement(r,'joint',name=name,type=kind);ET.SubElement(e,'parent',link=parent);ET.SubElement(e,'child',link=child);ET.SubElement(e,'origin',xyz=xyz,rpy=rpy)
        if kind!='fixed':
            ET.SubElement(e,'axis',xyz=axis);ET.SubElement(e,'limit',lower='-100',upper='100',effort='10000',velocity='10')
    for name in ('base_footprint','base_x_link','base_y_link','base_z_link','base_rx_link','base_ry_link','base_link'):link(name)
    joint('base_footprint_x_joint','base_footprint','base_x_link','prismatic',axis='1 0 0')
    joint('base_footprint_y_joint','base_x_link','base_y_link','prismatic',axis='0 1 0')
    joint('base_footprint_z_joint','base_y_link','base_z_link','prismatic',axis='0 0 1')
    joint('base_footprint_rx_joint','base_z_link','base_rx_link','revolute',axis='1 0 0')
    joint('base_footprint_ry_joint','base_rx_link','base_ry_link','revolute',axis='0 1 0')
    joint('base_footprint_rz_joint','base_ry_link','base_link','revolute')
    joint('standing_pelvis_joint','base_link','pelvis',xyz=f'0 0 {height}')
    for side in ('left','right'):
        name=f'{side}_eef_link';link(name)
        joint(f'{side}_eef_joint',f'{side}_dex1_base_link',name,xyz='.085 0 0',rpy=f'0 {math.pi/2} 0')
    # Pinocchio physical chain excludes virtual navigation DOFs.
    tree.write(OUT/'g1_bench_import.urdf',encoding='unicode')
    kin=ET.fromstring(ET.tostring(r))
    for j in kin.findall('joint'):
        if j.get('name').startswith('base_footprint_'):j.set('type','fixed')
    ET.ElementTree(kin).write(OUT/'g1_bench.urdf',encoding='unicode')
    arms={s:[f'{s}_{n}_joint' for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')] for s in ('left','right')}
    links={s:[x.replace('_joint','_link') for x in a] for s,a in arms.items()}
    controllers={'base':'HolonomicBaseJointController','trunk':'JointController',**{f'{part}_{s}':'JointController' for s in arms for part in ('arm','gripper')}}
    definition=dict(raw_controller_order=list(controllers),default_controllers=controllers,
        usd_path=str(OUT/'g1_bench.usda'),urdf_path=str(OUT/'g1_bench_import.urdf'),base_footprint_link_name='base_link',
        default_joint_pos=[0.0]*27,self_collisions=True,holonomic_base={'force_sphere_wheel_approximation':False},
        locomotion={'base_joint_names':['base_footprint_x_joint','base_footprint_y_joint','base_footprint_rz_joint'],'floor_touching_base_link_names':['left_ankle_roll_link','right_ankle_roll_link']},
        articulated_trunk={'trunk_joint_names':['waist_yaw_joint','waist_roll_joint','waist_pitch_joint'],'trunk_link_names':['waist_yaw_link','waist_roll_link','torso_link']},
        manipulation=dict(n_arms=2,arm_names=['left','right'],arm_joint_names=arms,arm_link_names=links,
          eef_link_names={s:f'{s}_eef_link' for s in arms},finger_link_names={s:[f'{s}_dex1_finger_link_{i}' for i in (1,2)] for s in arms},
          finger_joint_names={s:[f'{s}_dex1_finger_joint_{i}' for i in (1,2)] for s in arms},
          gripper_link_names={s:[f'{s}_dex1_base_link'] for s in arms},arm_workspace_range={s:[-90,90] for s in arms}))
    (OUT/'g1_bench.yaml').write_text(yaml.safe_dump(definition,sort_keys=False))
    cfg=dict(model=MODEL,obs_modalities=['rgb'],scale=1.,self_collisions=True,action_normalize=False,action_type='continuous',grasping_mode='sticky',controller_config={})
    for key,ctype in controllers.items():
        cfg['controller_config'][key]=dict(name=ctype,motor_type='position',command_input_limits=None,use_impedances=False)
        if key!='base':cfg['controller_config'][key]['use_delta_commands']=False
    (ROOT/'configs/robots/g1_bench_symbolic.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    install=Path('/home/ubuntu/ygwang/behavior-datasets/omnigibson-robot-assets/models/g1_bench')
    if not install.exists():install.symlink_to(OUT,target_is_directory=True)
    provenance=dict(source_url='https://github.com/unitreerobotics/unitree_ros',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),source_model=str(src),frozen_leg_joints=frozen,pelvis_height_m=height,neutral_collision_bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],navigation='virtual planar base; no biped balance/locomotion evaluation',end_effector='official Dex1-1')
    import shutil
    shutil.copy2(SOURCE/'LICENSE',OUT/'LICENSE.unitree')
    lower=vertices[vertices[:,2]+height<=.5].copy();lower[:,2]+=height
    provenance['navigation_lower_body_bounds_m']=[lower.min(0).tolist(),lower.max(0).tolist()]
    provenance['navigation_half_extent_m']=.25
    assert np.max(np.abs(lower[:,:2])) <= .25
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))

def normalize_collision_apis(stage):
    from pxr import Usd,UsdGeom,UsdPhysics
    # Put collider APIs on actual geometry, as required by the runtime exporter.
    for parent in list(stage.Traverse()):
        if not parent.HasAPI(UsdPhysics.CollisionAPI) or parent.IsA(UsdGeom.Gprim):continue
        approximation=parent.GetAttribute('physics:approximation').Get() or 'convexHull'
        geoms=[p for p in Usd.PrimRange(parent) if p.IsA(UsdGeom.Gprim)]
        if not geoms:raise ValueError(f'Collider contains no geometry: {parent.GetPath()}')
        for geom in geoms:
            UsdPhysics.CollisionAPI.Apply(geom)
            if geom.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr(approximation)
        parent.RemoveAPI(UsdPhysics.CollisionAPI);parent.RemoveAPI(UsdPhysics.MeshCollisionAPI)


def set_materials(stage):
    from pxr import UsdGeom,UsdShade,Sdf,Gf
    root=str(stage.GetDefaultPrim().GetPath())
    urdf=ET.parse(OUT/'g1_bench_import.urdf').getroot()
    colors={m.get('name'):m.find('color').get('rgba') for m in urdf.findall('material') if m.find('color') is not None}
    for link in urdf.findall('link'):
        material=link.find('visual/material')
        if material is None:continue
        c=material.find('color');rgba=c.get('rgba') if c is not None else colors.get(material.get('name'))
        if rgba is None:continue
        rgb=[float(x) for x in rgba.split()[:3]]
        path=root+'/Looks/color_'+''.join(f'{round(x*255):02x}' for x in rgb)
        mat=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader')
        shader.CreateImplementationSourceAttr('sourceAsset')
        shader.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'),'mdl');shader.SetSourceAssetSubIdentifier('OmniPBR','mdl')
        shader.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput('reflection_roughness_constant',Sdf.ValueTypeNames.Float).Set(.5)
        shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
        prefix=root+'/'+link.get('name')+'/visuals/'
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(prefix) and prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def convert():
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.asset.importer.urdf')
    _,cfg=omni.kit.commands.execute('URDFCreateImportConfig')
    cfg.merge_fixed_joints=False;cfg.fix_base=True;cfg.import_inertia_tensor=True
    cfg.self_collision=True;cfg.distance_scale=1.;cfg.make_default_prim=True
    ok,result=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=str(OUT/'g1_bench_import.urdf'),import_config=cfg,dest_path=str(OUT/'g1_bench.usda'))
    if not ok:raise RuntimeError(result)
    from pxr import Usd,UsdGeom,Gf,PhysxSchema
    stage=Usd.Stage.Open(str(OUT/'g1_bench.usda'));root=stage.GetDefaultPrim()
    for prim in stage.Traverse():
        from pxr import UsdPhysics
        if prim.GetName() in ('base_footprint_z_joint','base_footprint_rx_joint','base_footprint_ry_joint'):
            for kind in ('linear','angular'):prim.RemoveAPI(UsdPhysics.DriveAPI,kind)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    # OmniGibson traverses concrete mesh children, not instance proxies.
    while True:
        instances=[p for p in stage.Traverse() if p.IsInstance()]
        if not instances:break
        for prim in instances:prim.SetInstanceable(False)
    for prim in stage.Traverse():
        xf=UsdGeom.Xformable(prim)
        if not xf:continue
        transform=Gf.Transform(xf.GetLocalTransformation())
        xf.ClearXformOpOrder()
        def precision(name):
            t=str(prim.GetAttribute('xformOp:'+name).GetTypeName())
            return UsdGeom.XformOp.PrecisionFloat if t in ('float3','quatf') else UsdGeom.XformOp.PrecisionDouble
        xf.AddTranslateOp(precision=precision('translate')).Set(transform.GetTranslation())
        q=transform.GetRotation().GetQuat()
        prec=precision('orient')
        xf.AddOrientOp(precision=prec).Set(Gf.Quatf(q.GetReal(),*q.GetImaginary()) if prec==UsdGeom.XformOp.PrecisionFloat else q)
        xf.AddScaleOp(precision=precision('scale')).Set(transform.GetScale())
    camera=UsdGeom.Camera.Define(stage,str(root.GetPath())+'/head_link/head_camera')
    camera.AddTranslateOp().Set(Gf.Vec3d(.06,0,.57))
    camera.AddOrientOp().Set(Gf.Quatf(.5,.5,-.5,-.5))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100));camera.CreateFocalLengthAttr(17.)
    normalize_collision_apis(stage)
    set_materials(stage)
    stage.GetRootLayer().Save()
    print('G1_USD_READY',OUT/'g1_bench.usda',flush=True)
    app.close()

def replace_scenarios():
    if not (OUT/'smoke.json').is_file():
        raise RuntimeError('Validate G1 before replacing active scenarios')
    report=json.loads((OUT/'smoke.json').read_text())
    validation=json.loads((OUT/'g1_bench_validated_validation.json').read_text())
    if not validation.get('passed') or validation.get('num_samples',0)<50:
        raise RuntimeError('G1 native FK validation has not passed')
    required=['head_link','torso_link']+[f'{side}_{part}' for side in ('left','right')
        for part in ('shoulder_pitch_link','elbow_link','wrist_yaw_link','dex1_base_link','dex1_finger_link_1','dex1_finger_link_2')]
    if any(not report['links'].get(link,{}).get('collision_boxes') for link in required):
        raise RuntimeError('G1 exported body/arm/gripper colliders are incomplete')
    backup=OUT/'previous_scenarios.json'
    saved=json.loads(backup.read_text()) if backup.exists() else {}
    paths=sorted((ROOT/'scenarios').rglob('scenario.yaml'))+[ROOT/'assets/tasks/vla_difficulty_v2/common.yaml']
    changed=[]
    for path in paths:
        doc=yaml.safe_load(path.read_text());robot=doc.get('robot',{})
        if robot.get('model') not in ('r1pro','g1_bench'):continue
        if doc.get('scene',{}).get('frozen_member_manifest'):
            raise RuntimeError(f'Frozen topology requires separate rebuild: {path}')
        key=str(path.relative_to(ROOT))
        if key not in saved:saved[key]=path.read_text()
        robot.update(model='g1_bench',base_half_extent=.25)
        if robot.get('panorama'):
            robot['panorama'].update(head_link='head_link',mount_frame_link='torso_link',offset=[0,0,.57])
        robot['include_sensor_names']=None
        if robot.get('kinematics'):
            robot['kinematics'].update(urdf_path='build/robots/g1_bench/g1_bench_validated.urdf',base_link='base_link',end_effector_link='right_eef_link')
        path.write_text(yaml.safe_dump(doc,sort_keys=False));changed.append(key)
    backup.write_text(json.dumps(saved,indent=2))
    import hashlib
    manifest_path=ROOT/'scenarios/dense_search_pilot/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for row in manifest['tasks']:
        path=ROOT/row['scenario'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
        row['scenario_sha256']=sha;row['physics_status']='pending'
        audit=path.parent/'authoring.private.json'
        if audit.exists():
            data=json.loads(audit.read_text());data['scenario_sha256']=sha;data['robot_model']='g1_bench'
            audit.write_text(json.dumps(data,indent=2)+'\n')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('REPLACED',len(changed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--convert',action='store_true');p.add_argument('--replace',action='store_true');a=p.parse_args()
    if a.replace:replace_scenarios()
    elif a.convert:convert()
    else:prepare()
