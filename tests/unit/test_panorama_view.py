import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError
from rummagebench.core.public_types import PublicAction
from rummagebench.core.scenario import PanoramaSpec
from rummagebench.perception.frame_store import VisualFramePrivate
from rummagebench.perception.panorama import spherical_rays, perspective_view
from rummagebench.perception.visual_bridge import VisualBridge


def panorama():
    h,w=90,180
    rgb=np.zeros((h,w,3),np.uint8)
    rgb[:,:,0]=np.arange(w)
    return VisualFramePrivate('',rgb,np.ones((h,w))*2,np.ones((h,w),int),
        np.eye(3),np.eye(4),w,h,'euclidean_range',
        dict(projection='equirectangular',pixel_rays=spherical_rays(w,h),
             visual_grounding_supported=True,instance_labels={'1':'target_knife'},bridge='segmentation'))


def test_view_schema_and_removed_observe():
    a=TypeAdapter(PublicAction)
    assert a.validate_python(dict(skill='VIEW',yaw=90)).pitch == 0
    assert a.validate_python(dict(skill='VIEW',yaw=90)).zoom == 1
    assert PanoramaSpec().detail_width == 1280
    assert PanoramaSpec().detail_height == 960
    for zoom in [0, 4.1, float('nan'), float('inf')]:
        with pytest.raises(ValidationError):
            a.validate_python(dict(skill='VIEW', yaw=0, zoom=zoom))
    for raw in [dict(skill='VIEW',yaw=float('nan')),dict(skill='VIEW',yaw=181),
                dict(skill='VIEW',yaw=0,pitch=91),dict(skill='VIEW',yaw=0,distance_cm=4),
                dict(skill='OBSERVE',point=dict(frame_id='x',x=.5,y=.5))]:
        with pytest.raises(ValidationError): a.validate_python(raw)


def test_zoom_changes_focal_length_and_keeps_grounding(visual_session, visual_backend):
    vs = visual_session
    vs._scenario.robot.panorama = PanoramaSpec(detail_width=65, detail_height=49)
    visual_backend.capture_panorama_frame = panorama
    vs._capture_current()
    primary_id = vs.observe().frame_id
    before = visual_backend.robot_pose()
    for zoom in (1, 4, 2, 1):
        result = vs.step(dict(skill='VIEW', yaw=0, pitch=0, zoom=zoom))
        detail = result.observation.detail_view
        assert detail.zoom == zoom
        assert detail.hfov_deg == pytest.approx(np.degrees(2*np.arctan(1/zoom)))
        frame = vs._store.get(detail.frame_id)
        assert frame.camera_intrinsics[0, 0] == pytest.approx(65*zoom/2)
        point = VisualBridge(patch_size=1).surface_point(frame, 1, (32, 24))
        np.testing.assert_allclose(point/2, [0,0,1], atol=.025)
        assert result.observation.frame_id == primary_id
        assert visual_backend.robot_pose() == before
        assert len(vs._store._frames) <= 2


def test_check_is_non_actionable_appearance_and_expires(visual_session, visual_backend, monkeypatch):
    from types import SimpleNamespace
    from rummagebench.agents.local_vlm import image_payload
    vs = visual_session
    vs._scenario.robot.panorama = PanoramaSpec(detail_width=64, detail_height=48)
    visual_backend.capture_panorama_frame = panorama
    vs._capture_current()
    old = vs.observe()
    monkeypatch.setattr(vs, '_resolve_point', lambda *args: ('target_knife', np.zeros(3)))
    monkeypatch.setattr(visual_backend, 'describe_entity', lambda _: SimpleNamespace(graspable=True, openable=False))
    calls = []
    def capture(entity):
        calls.append(entity)
        return np.full((64,64,3), 150, np.uint8)
    visual_backend.capture_check_image = capture
    r = vs.step(dict(skill='CHECK', point=dict(frame_id=old.frame_id, x=.5, y=.5)))
    assert r.feedback.code == 'EXECUTED'
    assert calls == ['target_knife']
    obs = r.observation
    assert obs.image_png_b64 == old.image_png_b64
    assert obs.frame_id == old.frame_id
    assert obs.detail_view is None
    assert 'frame_id' not in obs.inspection_view.model_dump()
    assert len(image_payload(obs.model_dump())) == 2
    assert vs.observe().inspection_view is not None
    r = vs.step(dict(skill='CHECK', point=dict(frame_id='inspection', x=.5, y=.5)))
    assert r.feedback.code == 'INVALID_ACTION'
    assert r.observation.inspection_view is None
    assert len(calls) == 1
    monkeypatch.setattr(visual_backend, 'describe_entity', lambda _: SimpleNamespace(graspable=False, openable=True))
    r = vs.step(dict(skill='CHECK', point=dict(frame_id=old.frame_id, x=.5, y=.5)))
    assert r.feedback.code == 'OUT_OF_CAPABILITY'
    assert r.observation.inspection_view is None
    assert len(calls) == 1


@pytest.mark.parametrize('yaw,pitch,expected',[(0,0,[0,0,1]),(90,0,[-1,0,0]),
    (-90,0,[1,0,0]),(180,0,[0,0,-1]),(90,90,[0,-1,0]),(90,45,[-.707,-.707,0])])
def test_angles_and_grounding(yaw,pitch,expected):
    frame=perspective_view(panorama(),yaw,pitch,width=33,height=33)
    p=VisualBridge(patch_size=1).surface_point(frame,1,(16,16))
    np.testing.assert_allclose(p/2,expected,atol=.025)
    assert frame.depth[16,16]==2


def test_view_keeps_panorama_and_replaces_only_detail(visual_session,visual_backend):
    vs=visual_session
    vs._scenario.robot.panorama=PanoramaSpec(detail_width=64, detail_height=48)
    visual_backend.capture_panorama_frame=panorama
    vs._capture_current()
    before=visual_backend.robot_pose()
    old=vs.observe().frame_id
    def forbidden(*a,**k): raise AssertionError('VIEW moved the robot or physics')
    visual_backend.teleport_robot=forbidden
    visual_backend.settle=forbidden
    r=vs.step(dict(skill='VIEW',yaw=90))
    assert r.feedback.code=='EXECUTED'
    assert r.planning_step==1
    assert r.observation.frame_id == old
    detail_id=r.observation.detail_view.frame_id
    assert detail_id != old
    assert r.observation.detail_view.yaw_deg==90
    assert visual_backend.robot_pose()==before
    vs.reset = forbidden
    r=vs.step(dict(skill='VIEW',yaw=0,pitch=30))
    assert r.feedback.code=='EXECUTED'
    assert r.observation.frame_id==old
    assert r.observation.detail_view.frame_id!=detail_id
    r=vs.step(dict(skill='GRASP',point=dict(frame_id=detail_id,x=.5,y=.5)))
    assert r.feedback.code=='INVALID_ACTION'
    assert vs._trace[-1]['private_reason']=='STALE_FRAME'


@pytest.mark.parametrize('renderer_intrinsics', [False, True])
def test_cube_faces_keep_rgb_ids_depth_and_geometry_together(renderer_intrinsics):
    from scipy.spatial.transform import Rotation
    from rummagebench.perception.camera_geometry import default_intrinsics
    from rummagebench.perception.panorama import compose_cube
    rotations=[np.eye(3)]+[Rotation.from_euler('y',a,degrees=True).as_matrix() for a in (90,180,-90)]+[Rotation.from_euler('x',a,degrees=True).as_matrix() for a in (90,-90)]
    faces=[]
    K = default_intrinsics(32,32)
    if renderer_intrinsics:
        K = np.array([[16/1.05,0,16],[0,16/1.05,16],[0,0,1.]])
    for i,R in enumerate(rotations):
        T=np.eye(4);T[:3,:3]=R
        faces.append(VisualFramePrivate('',np.full((32,32,3),i+1,np.uint8),
            np.ones((32,32))*2,np.ones((32,32),int),K,T,32,32,
            'euclidean_range',dict(instance_labels={'1':str(i+1)},visual_grounding_supported=True)))
    full=compose_cube(faces,128,64)
    assert set(np.unique(full.rgb))==set(range(1,7))
    for label,name in full.meta['instance_labels'].items():
        assert np.all(full.rgb[full.instance_segmentation==int(label)]==int(name))
    np.testing.assert_allclose(np.linalg.norm(full.meta['pixel_rays'],axis=-1),1)
    assert np.all(full.depth==2)
    faces[1].camera_extrinsics[:3,3]=1
    with pytest.raises(ValueError,match='optical centre'): compose_cube(faces,128,64)


def test_mount_above_tilted_zed_uses_neck_up_and_forward():
    from scipy.spatial.transform import Rotation
    from rummagebench.sim.omnigibson.panorama import mount_in_head
    neck = Rotation.from_euler('z', 40, degrees=True).as_matrix()
    # Actual R1Pro ZED joint basis is not aligned with the neck/body basis.
    head = neck @ Rotation.from_euler('xyz', [-1.9198999, 0, -1.5707999]).as_matrix()
    rotation, offset = mount_in_head(Rotation.from_matrix(head).as_quat(),
                                    Rotation.from_matrix(neck).as_quat(), [0, 0, .15])
    np.testing.assert_allclose(head @ offset, neck @ [0, 0, .15], atol=1e-12)
    np.testing.assert_allclose(head @ rotation, neck, atol=1e-12)


def test_dual_views_point_binding_refresh_and_many_views(visual_session,visual_backend,monkeypatch):
    vs=visual_session
    vs.set_planning_budget(100)
    vs._scenario.robot.panorama=PanoramaSpec(detail_width=64, detail_height=48)
    visual_backend.capture_panorama_frame=panorama
    vs._capture_current()
    first=vs.observe()
    seen=[]
    monkeypatch.setattr(vs,'_resolve_point',lambda frame,x,y,record:seen.append(frame.frame_id))
    for i in range(70):
        result=vs.step(dict(skill='VIEW',yaw=(i%3)*30))
        assert result.observation.image_png_b64==first.image_png_b64
        assert result.observation.frame_id==first.frame_id
    current=vs.observe()
    assert current.detail_view.frame_id!=current.frame_id
    for fid in [current.frame_id,current.detail_view.frame_id]:
        action=TypeAdapter(PublicAction).validate_python(dict(skill='GRASP',point=dict(frame_id=fid,x=.2,y=.3)))
        vs._step_point(action,None,{})
    assert seen==[current.frame_id,current.detail_view.frame_id]
    vs._capture_current()  # models a synchronized capture after movement
    new=vs.observe()
    assert new.frame_id!=current.frame_id
    assert new.detail_view is None
    for fid in [current.frame_id,current.detail_view.frame_id]:
        result=vs.step(dict(skill='GRASP',point=dict(frame_id=fid,x=.2,y=.3)))
        assert result.feedback.code=='INVALID_ACTION'
        assert vs._trace[-1]['private_reason']=='STALE_FRAME'


@pytest.mark.parametrize('action', [{}, {'skill':'MOVE','distance_cm':0},
    {'skill':'TURN','angle_deg':10}, {'skill':'VIEW','yaw':999}])
def test_supplement_only_in_successful_view_response(visual_session,visual_backend,action):
    vs=visual_session
    vs._scenario.robot.panorama=PanoramaSpec(detail_width=64, detail_height=48)
    visual_backend.capture_panorama_frame=panorama
    vs._capture_current()
    result=vs.step({'skill':'VIEW','yaw':90})
    fid=result.observation.detail_view.frame_id
    assert vs.observe().detail_view.frame_id==fid  # reads don't consume it
    result=vs.step(action)
    assert result.observation.detail_view is None
    assert vs.observe().detail_view is None
    result=vs.step({'skill':'GRASP','point':{'frame_id':fid,'x':.5,'y':.5}})
    assert vs._trace[-1]['private_reason']=='STALE_FRAME'
    assert result.observation.detail_view is None
    assert vs.step({'skill':'VIEW','yaw':0}).observation.detail_view is not None


def test_detail_click_consumed_after_binding(visual_session,visual_backend,monkeypatch):
    vs=visual_session
    vs._scenario.robot.panorama=PanoramaSpec(detail_width=64, detail_height=48)
    visual_backend.capture_panorama_frame=panorama
    vs._capture_current()
    detail=vs.step({'skill':'VIEW','yaw':90}).observation.detail_view
    seen=[]
    def resolve(frame,x,y,record):
        seen.append(frame.frame_id)
        vs._last_feedback=vs._feedback('INVALID_ACTION')
        record['feedback']='INVALID_ACTION'
        return None
    monkeypatch.setattr(vs,'_resolve_point',resolve)
    result=vs.step({'skill':'GRASP','point':{'frame_id':detail.frame_id,'x':.5,'y':.5}})
    assert seen==[detail.frame_id]
    assert result.observation.detail_view is None


def test_native_pinhole_view_path(visual_session,visual_backend):
    vs=visual_session
    vs._scenario.robot.panorama=PanoramaSpec(detail_width=64, detail_height=48)
    visual_backend.capture_panorama_frame=panorama
    vs._capture_current();calls=[]
    def direct(yaw,pitch,width,height,zoom=1):
        calls.append((yaw,pitch,width,height))
        result=perspective_view(panorama(),yaw,pitch,width,height)
        result.meta['view_source']='direct_pinhole_render'
        return result
    visual_backend.capture_view_frame=direct
    r=vs.step(dict(skill='VIEW',yaw=0,pitch=-70))
    assert calls==[(0,-70,64,48)]
    assert r.observation.detail_view.source=='direct_pinhole_render'
    assert r.observation.detail_view.width==64
    assert r.observation.detail_view.height==48
