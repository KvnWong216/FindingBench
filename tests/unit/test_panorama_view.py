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
    for raw in [dict(skill='VIEW',yaw=float('nan')),dict(skill='VIEW',yaw=181),
                dict(skill='VIEW',yaw=0,pitch=91),dict(skill='VIEW',yaw=0,distance_cm=4),
                dict(skill='OBSERVE',point=dict(frame_id='x',x=.5,y=.5))]:
        with pytest.raises(ValidationError): a.validate_python(raw)


@pytest.mark.parametrize('yaw,pitch,expected',[(0,0,[0,0,1]),(90,0,[-1,0,0]),
    (-90,0,[1,0,0]),(180,0,[0,0,-1]),(90,90,[0,-1,0]),(90,45,[-.707,-.707,0])])
def test_angles_and_grounding(yaw,pitch,expected):
    frame=perspective_view(panorama(),yaw,pitch,width=33,height=33)
    p=VisualBridge(patch_size=1).surface_point(frame,1,(16,16))
    np.testing.assert_allclose(p/2,expected,atol=.025)
    assert frame.depth[16,16]==2


def test_view_never_moves_robot_and_invalidates_old_frame(visual_session,visual_backend):
    vs=visual_session
    vs._scenario.robot.panorama=PanoramaSpec()
    visual_backend.capture_panorama_frame=panorama
    vs._panorama_frame=panorama()
    before=visual_backend.robot_pose()
    old=vs.observe().frame_id
    def forbidden(*a,**k): raise AssertionError('VIEW moved the robot or physics')
    visual_backend.teleport_robot=forbidden
    visual_backend.settle=forbidden
    r=vs.step(dict(skill='VIEW',yaw=90))
    assert r.feedback.code=='EXECUTED'
    assert r.planning_step==1
    assert r.observation.frame_id != old
    assert visual_backend.robot_pose()==before
    r=vs.step(dict(skill='GRASP',point=dict(frame_id=old,x=.5,y=.5)))
    assert r.feedback.code=='INVALID_ACTION'
    vs.reset = forbidden
    r=vs.step(dict(skill='VIEW',yaw=0,pitch=30))
    assert r.feedback.code=='EXECUTED'


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
