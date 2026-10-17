"""Synthetic geometry fixtures only; never production assets or certificates."""
import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rummagebench.authoring.layout.indoor_geometry import (
    checked_mesh, digest, hull_mesh, layout_pose, native_layout_pose, rigid_rotation_update,
    load_geometry, native_scene_snapshot, world_to_local)
from rummagebench.authoring.layout.spec import Placement
from rummagebench.authoring.tasks.plan import sha256_path
from rummagebench.sim.omnigibson.behavior_adapter import object_config


@pytest.mark.parametrize('yaw', [-170., 0., 34., 180.])
def test_factory_geometry_matches_native_offset_and_front(yaw):
    # Asymmetric points and nonzero origin offset catch the previous swapped
    # bbox convention's easy-to-miss extra quarter turn.
    local = np.array([[.12, -.03, -.04], [-.09, .13, .05], [.05, .02, .08]])
    offset = np.array([.02, -.04, .01])
    proto_q = Rotation.from_euler('xyz',[.1,.2,.3]).as_quat()
    proto_root = [100., 102., 9.]
    world = Rotation.from_quat(proto_q).apply(local + offset) + proto_root
    center = Rotation.from_quat(proto_q).apply(offset) + proto_root
    factory = world_to_local(world,center,proto_q,factory=True)
    placement = Placement(name='unit',category='bowl',model_id='unit',role='clutter',
        aabb_size=[.2,.3,.1],position=[1.,2.,3.],orientation_deg=yaw+180)
    cfg = object_config(placement,dict(category='bowl',model='unit',base_link_offset=offset))
    native = Rotation.from_quat(cfg['orientation']).apply(local+offset) + cfg['position']
    gen = Rotation.from_euler('z',yaw,degrees=True).apply(factory) + placement.position
    assert np.allclose(gen,native,atol=1e-12)


def test_convex_hull_is_outward_and_not_a_concave_visual_surface():
    points = np.array([[x,y,z] for x in (-1,1) for y in (-2,2) for z in (-3,3)] + [[0,0,0]])
    vertices,faces = hull_mesh(points)
    normal = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    assert np.all(np.sum(normal*vertices[faces].mean(axis=1),axis=1)>0)
    assert 8 not in faces


def test_yaw_pose_roundtrip_rejects_unrepresentable_solver_changes():
    matrix=np.eye(4);matrix[:3,3]=[1,2,3]
    matrix[:3,:3]=Rotation.from_euler('z',37,degrees=True).as_matrix()
    pos,yaw=layout_pose(matrix)
    assert pos==[1,2,3] and yaw==pytest.approx(217)
    for bad in (np.diag([2,1,1,1]),np.diag([-1,1,1,1])):
        with pytest.raises(ValueError,match='TILT_OR_SCALE'):layout_pose(bad)
    matrix[:3,:3]=Rotation.from_euler('x',.01).as_matrix()
    with pytest.raises(ValueError,match='TILT_OR_SCALE'):layout_pose(matrix)


@pytest.mark.parametrize('angles', [[0,0,37], [5,-8,37], [90,0,-121]])
def test_full_native_pose_reproduces_every_solver_mesh_vertex(angles):
    matrix=np.eye(4)
    matrix[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
    matrix[:3,3]=[1,2,3]
    pose=native_layout_pose(matrix)
    offset=np.array([.03,-.07,.04])
    local=np.array([[.15,-.02,.01],[-.04,.06,.09],[0,.12,.03]])
    factory=world_to_local(local,[0,0,0],[0,0,0,1],factory=True)
    p=Placement(name='unit',category='bowl',model_id='unit',role='clutter',
        aabb_size=[.2,.3,.1],position=pose['position'],orientation_deg=0.)
    cfg=object_config(p,dict(category='bowl',model='unit',base_link_offset=offset),pose)
    actual=Rotation.from_quat(cfg['orientation']).apply(local+offset)+cfg['position']
    expected=factory@matrix[:3,:3].T+matrix[:3,3]
    assert np.allclose(actual,expected,atol=1e-12)


def test_full_pose_rejects_scale_reflection_and_shear():
    shear=np.eye(4);shear[0,1]=.01
    for bad in (np.diag([2,1,1,1]),np.diag([-1,1,1,1]),shear):
        with pytest.raises(ValueError,match='SCALE_OR_SHEAR'):native_layout_pose(bad)


def test_repeated_world_rotations_preserve_rigidity_at_float32_boundary():
    matrix=np.eye(4);matrix[:3,3]=[1,2,3]
    axis=np.array([1.,2.,3.]);angle=.031
    for _ in range(1000):
        q=rigid_rotation_update(matrix,axis,angle)
        matrix[:3,:3]=Rotation.from_quat(q).as_matrix().astype(np.float32)
        native_layout_pose(matrix)
    expected=Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle*1000).as_matrix()
    assert np.allclose(matrix[:3,:3],expected,atol=2e-6,rtol=0)
    assert matrix[:3,3].tolist()==[1,2,3]
    matrix[0,0]=2
    with pytest.raises(ValueError):rigid_rotation_update(matrix,axis,angle)


def test_geometry_result_cannot_fall_back_to_yaw_or_replace_position():
    from rummagebench.authoring.layout.indoor_background import verify_result
    p=dict(name='daily',category='bowl',model_id='unit',role='clutter',support_name='table',
           position=[1.,2.,3.],orientation_deg=180.,aabb_size=[.1,.2,.1])
    req=dict(geometry='unit_only',per_surface=1,native_context=dict(
        objects=[dict(name='table',support=True)],retained_native_objects=['table']),
        assets={'bowl/unit':dict(category='bowl',model='unit',base_link_offset=[.01,0,0])})
    res=dict(status='constraints_satisfied',placements=[p],violations={'daily_items_table':0})
    with pytest.raises(ValueError,match='POSES_INCOMPLETE'):verify_result(req,res)
    res['native_poses']={'daily':dict(position=p['position'],orientation=[0,0,0,1])}
    verify_result(req,res)
    res['native_poses']['daily']['position']=[1,2,4]
    with pytest.raises(ValueError,match='POSE_MISMATCH'):verify_result(req,res)
    res['native_poses']['daily']['position']=p['position']
    res['native_poses']['daily']['orientation']=[0,0,0,2]
    with pytest.raises(ValueError,match='QUATERNION_INVALID'):verify_result(req,res)


@pytest.mark.parametrize('v,f', [([[0,0,float('nan')]]*3,[[0,1,2]]),
    ([[0,0,0]]*3,[[0,1,3]]), ([[0,0,0]]*3,[[0.,1.,2.]])])
def test_invalid_mesh_cannot_reach_gen(v,f):
    with pytest.raises(ValueError,match='COLLISION_MESH_INVALID'):
        checked_mesh(v,f)


def fixture_export(tmp_path):
    snapshot=tmp_path/'native_scene.json'
    snapshot.write_text(json.dumps(dict(objects_info={'init_info':{'native':{}}},
        state={'registry':{'object_registry':{'native':{}}}})))
    mesh=dict(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],
              colliders=[dict(prim='/unit',kind='Mesh',approximation='none')])
    return dict(version=1,status='geometry_exported_pending_physics',scene='unit',
        scene_snapshot=str(snapshot),inputs={str(snapshot):sha256_path(snapshot)},
        native={'native':copy.deepcopy(mesh)},assets={'bowl/unit':copy.deepcopy(mesh)},
        asset_records={'bowl/unit':{'unit_only':True}})


def write(tmp_path,doc):
    doc['sha256']=digest(doc)
    path=tmp_path/'geometry.json';path.write_text(json.dumps(doc));return path


def test_geometry_binds_scene_state_and_approved_asset_set(tmp_path):
    doc=fixture_export(tmp_path);path=write(tmp_path,doc)
    assert load_geometry(path,scene='unit',assets=doc['asset_records'])==doc
    with pytest.raises(ValueError,match='SCENE_MISMATCH'):load_geometry(path,scene='elsewhere')
    with pytest.raises(ValueError,match='POOL_MISMATCH'):load_geometry(path,assets={})
    with open(doc['scene_snapshot'],'a') as f:f.write(' ')
    with pytest.raises(ValueError,match='INPUT_CHANGED'):load_geometry(path)


def test_measured_library_allows_exact_nonempty_sampling_subset_only(tmp_path):
    doc=fixture_export(tmp_path)
    doc['assets']['plate/unit']=copy.deepcopy(doc['assets']['bowl/unit'])
    doc['asset_records']['plate/unit']={'unit_only':True,'category':'plate'}
    path=write(tmp_path,doc)
    assert load_geometry(path,assets={'bowl/unit':doc['asset_records']['bowl/unit']})==doc
    for invalid in ({'bowl/unit':{'unit_only':False}}, {'unknown/unit':{'unit_only':True}}):
        with pytest.raises(ValueError,match='POOL_MISMATCH'):load_geometry(path,assets=invalid)


def test_curved_bottom_uses_existing_supporting_triangle_without_mesh_changes():
    from rummagebench.authoring.layout.indoor_geometry import supporting_bottom_faces
    v,f=hull_mesh([[0,0,-1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]])
    before=v.copy();faces_before=f.copy()
    mask=supporting_bottom_faces(v,f)
    assert mask.sum()==1 and np.array_equal(v,before) and np.array_equal(f,faces_before)
    face=v[f[mask][0]];normal=np.cross(face[1]-face[0],face[2]-face[0])
    assert normal[2]<0 and np.max((v-face[0])@normal)<=1e-9


def test_internal_downward_face_cannot_masquerade_as_support():
    from rummagebench.authoring.layout.indoor_geometry import supporting_bottom_faces
    v=[[0,0,0],[0,1,0],[1,0,0],[.2,.2,-.1]]
    with pytest.raises(ValueError,match='NO_REAL_SUPPORTING_BOTTOM'):
        supporting_bottom_faces(v,[[0,1,2]])


@pytest.mark.parametrize('kind',['native_missing','asset_missing','snapshot_unbound','failed'])
def test_resigned_incomplete_exports_are_rejected(tmp_path,kind):
    doc=fixture_export(tmp_path)
    if kind=='native_missing':doc['native']={}
    if kind=='asset_missing':doc['assets']={}
    if kind=='snapshot_unbound':doc['inputs']={}
    if kind=='failed':doc['status']='rejected'
    with pytest.raises(ValueError):load_geometry(write(tmp_path,doc))


def test_cooked_unknown_shape_is_not_silently_treated_as_source_mesh(tmp_path):
    doc=fixture_export(tmp_path)
    doc['native']['native']['colliders'][0]['approximation']='convexDecomposition'
    with pytest.raises(ValueError,match='REPRESENTATION_UNSUPPORTED'):load_geometry(write(tmp_path,doc))


def test_visual_only_exception_requires_original_scene_evidence(tmp_path):
    doc=fixture_export(tmp_path)
    doc['native']['native']=dict(vertices=[],faces=[],colliders=[],noncolliding_visual_only=True)
    with pytest.raises(ValueError,match='VISUAL_ONLY_EVIDENCE'):load_geometry(write(tmp_path,doc))
    snapshot=json.loads(open(doc['scene_snapshot']).read())
    snapshot['objects_info']['init_info']['native']={'args':{'visual_only':True}}
    with open(doc['scene_snapshot'],'w') as f:json.dump(snapshot,f)
    doc['inputs'][doc['scene_snapshot']]=sha256_path(doc['scene_snapshot'])
    load_geometry(write(tmp_path,doc))


def test_active_systems_are_not_silently_dropped(tmp_path):
    doc=fixture_export(tmp_path)
    snapshot=json.loads(open(doc['scene_snapshot']).read())
    snapshot['state']['registry']['system_registry']={'water':{}}
    with open(doc['scene_snapshot'],'w') as f:json.dump(snapshot,f)
    doc['inputs'][doc['scene_snapshot']]=sha256_path(doc['scene_snapshot'])
    with pytest.raises(ValueError,match='SYSTEM_STATE_RESTORE_UNSUPPORTED'):load_geometry(write(tmp_path,doc))


def test_native_snapshot_keeps_inventory_live_state_and_runtime_version():
    source=dict(versions={'sdk':'original'},objects_info={'init_info':{'table':{},'bowl':{}}},
                state={'old':True},metadata={'native':True})
    saved=dict(versions={'sdk':'runtime'},state={'registry':{'object_registry':{
        'table':{'pose':[1,2,3]},'bowl':{'pose':[4,5,6]},'robot':{}},'system_registry':{}}})
    result=native_scene_snapshot(source,saved)
    assert set(result['state']['registry']['object_registry'])=={'table','bowl'}
    assert result['state']['registry']['object_registry']['bowl']=={'pose':[4,5,6]}
    assert result['versions']=={'sdk':'runtime'} and result['metadata']==source['metadata']
    assert source['state']=={'old':True} and 'robot' in saved['state']['registry']['object_registry']
    del saved['state']['registry']['object_registry']['bowl']
    with pytest.raises(ValueError,match='STATE_INCOMPLETE'):native_scene_snapshot(source,saved)
