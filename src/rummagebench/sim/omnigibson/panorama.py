"""One logical panoramic camera implemented with six rigid cube render faces.

All faces share an optical centre and are children of the actual head link.
They never move relative to that link, including during VIEW. Rendering does
not step physics. RGB/range/raw renderer IDs share each face's render product.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from rummagebench.perception.frame_store import VisualFramePrivate
from rummagebench.perception.panorama import compose_cube
from rummagebench.perception.camera_geometry import world_from_usd_camera
from rummagebench.sim.omnigibson.raw_instance import (
    _array, _camera_in_sync, wait_for_intrinsics, canonical_labels,
)


def mount_in_head(head_quat, reference_quat, offset):
    """Convert a fixed forward/up mount basis into a tilted head link frame."""
    rotation = (Rotation.from_quat(_array(head_quat)).as_matrix().T
                @ Rotation.from_quat(_array(reference_quat)).as_matrix())
    return rotation, rotation @ np.asarray(offset, float)


class PanoramaCamera:
    def __init__(self, backend, config):
        import omni.replicator.core as rep
        from omnigibson.sensors import VisionSensor
        self.backend, self.config = backend, config
        link = backend._robot.links.get(config.head_link)
        if link is None:
            raise RuntimeError(f'panorama head link missing: {config.head_link}')
        reference = backend._robot.links.get(config.mount_frame_link)
        if reference is None:
            raise RuntimeError(f'panorama mount frame missing: {config.mount_frame_link}')
        mount_rotation, mount_offset = mount_in_head(
            link.get_position_orientation()[1], reference.get_position_orientation()[1], config.offset)
        self.faces = []
        optical = np.array([[0,0,1],[-1,0,0],[0,-1,0]], float)
        scene = backend._env.scene
        path = str(link.prim_path)
        # OG prim constructors take paths relative to the scene root.
        scene_path = str(scene.prim_path)
        if not path.startswith(scene_path + '/'):
            raise RuntimeError('camera mount is outside robot scene')
        relative = path[len(scene_path):]
        rotations = [np.eye(3)] + [Rotation.from_euler('y', a, degrees=True).as_matrix()
                                  for a in (90,180,-90)] + [
            Rotation.from_euler('x', a, degrees=True).as_matrix() for a in (90,-90)]
        for i, R in enumerate(rotations):
            sensor = VisionSensor(relative_prim_path=f'{relative}/findingbench_panorama_{i}',
                name=f'findingbench_panorama_{i}', modalities=['rgb','depth'],
                image_width=config.face_size, image_height=config.face_size,
                # Slight overlap avoids half-pixel gaps at cube seams with
                # the renderer's principal-point convention. Reprojection
                # still uses each sensor's actual calibrated intrinsics.
                focal_length=10, horizontal_aperture=21, clipping_range=(.01,1000))
            sensor.load(scene)
            q = Rotation.from_matrix(mount_rotation @ optical @ R @ np.diag([1,-1,-1])).as_quat()
            # Parent coordinates make the mount rigid; OG's setter also
            # invalidates transform caches. Never called by VIEW.
            sensor.set_position_orientation(position=mount_offset, orientation=q, frame='parent')
            sensor.initialize()
            annotator = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast')
            with backend._sim.editing_usd():
                annotator.attach([sensor.render_product])
            self.faces.append((sensor,annotator))

    def capture(self):
        backend = self.backend
        backend.validate_physics_state()
        intrinsics = [np.asarray(_array(wait_for_intrinsics(s,backend._sim))) for s,_ in self.faces]
        for i in range(30):
            backend._sim.render()
            if i>=5 and all(_camera_in_sync(s) for s,_ in self.faces): break
        else: raise RuntimeError('panorama renderer pose synchronization failed')
        roots = {str(o.prim_path):o.name for o in backend._env.scene.objects}
        frames = []
        for (sensor,annotator),K in zip(self.faces,intrinsics):
            data,_ = sensor.get_obs()
            raw = annotator.get_data()
            rgb = np.asarray(_array(data['rgb']))[...,:3].astype(np.uint8).copy()
            depth = np.asarray(_array(data['depth']),float).squeeze().copy()
            ids = np.asarray(raw['data']).copy()
            if depth.shape != rgb.shape[:2] or ids.shape != depth.shape:
                raise RuntimeError('panorama modality dimensions differ')
            pos,q = sensor.get_position_orientation()
            T = world_from_usd_camera(_array(pos),_array(q))
            frames.append(VisualFramePrivate('',rgb,depth,ids,K,T,rgb.shape[1],rgb.shape[0],
                'euclidean_range', dict(instance_labels=canonical_labels(raw['info']['idToLabels'],roots),
                                        visual_grounding_supported=True)))
        if not all(_camera_in_sync(s) for s,_ in self.faces):
            raise RuntimeError('panorama pose changed during capture')
        backend.validate_physics_state()
        return compose_cube(frames,self.config.width,self.config.height)

    def capture_view(self, yaw, pitch, width, height, zoom=1):
        """Render a virtual pinhole at the rigid panorama centre; no physics step."""
        import omni.replicator.core as rep
        from omnigibson.sensors import VisionSensor
        backend=self.backend
        before_pose=backend.robot_pose()
        before_tick=backend._sim.current_time_step_index
        link=backend._robot.links[self.config.head_link]
        if not hasattr(self,'detail_sensor'):
            scene=backend._env.scene
            relative=str(link.prim_path)[len(str(scene.prim_path)):]
            sensor=VisionSensor(relative_prim_path=f'{relative}/findingbench_detail',
                name='findingbench_detail',modalities=['rgb','depth'],image_width=width,
                image_height=height,focal_length=10,horizontal_aperture=20,
                clipping_range=(.01,1000))
            sensor.load(scene);sensor.initialize()
            annotator=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast')
            with backend._sim.editing_usd():annotator.attach([sensor.render_product])
            self.detail_sensor,self.detail_annotator=sensor,annotator
        sensor=self.detail_sensor
        sensor.focal_length = 10 * zoom
        # Read the fixed front face pose, then rotate gaze at the same optical centre.
        pos,q=self.faces[0][0].get_position_orientation()
        base=world_from_usd_camera(_array(pos),_array(q))
        gaze=Rotation.from_euler('y',-yaw,degrees=True).as_matrix() @ Rotation.from_euler('x',pitch,degrees=True).as_matrix()
        world_R=base[:3,:3] @ gaze
        world_q=Rotation.from_matrix(world_R @ np.diag([1,-1,-1])).as_quat()
        sensor.set_position_orientation(position=pos,orientation=world_q)
        K=np.asarray(_array(wait_for_intrinsics(sensor,backend._sim)))
        for i in range(30):
            backend._sim.render()
            K=np.asarray(_array(wait_for_intrinsics(sensor,backend._sim)))
            if (i>=5 and _camera_in_sync(sensor)
                    and np.isclose(K[0,0], width * zoom / 2, rtol=.01)):
                break
        else:raise RuntimeError('VIEW renderer pose or zoom synchronization failed')
        data,_=sensor.get_obs();raw=self.detail_annotator.get_data()
        rgb=np.asarray(_array(data['rgb']))[...,:3].astype(np.uint8).copy()
        depth=np.asarray(_array(data['depth']),float).squeeze().copy()
        ids=np.asarray(raw['data']).copy()
        roots={str(o.prim_path):o.name for o in backend._env.scene.objects}
        T=base.copy();T[:3,:3]=world_R
        u,v=np.meshgrid(np.arange(width),np.arange(height))
        rays=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)),-1)
        rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
        after_pose=backend.robot_pose()
        if before_tick != backend._sim.current_time_step_index or not all(
            np.allclose(a,b,atol=1e-8,rtol=0) for a,b in zip(before_pose,after_pose)):
            raise RuntimeError('VIEW unexpectedly changed physics time or robot pose')
        if rgb.shape != (height,width,3) or ids.shape != (height,width):
            raise RuntimeError('VIEW renderer modality dimensions differ')
        return VisualFramePrivate('',rgb,depth,ids,K,T,width,height,'euclidean_range',
            dict(projection='perspective',view_yaw=yaw,view_pitch=pitch,
                 view_zoom=zoom,hfov_deg=float(np.degrees(2*np.arctan(1/zoom))),
                 native_view_extrinsics=True,pixel_rays=rays,
                 instance_labels=canonical_labels(raw['info']['idToLabels'],roots),
                 visual_grounding_supported=True,view_source='direct_pinhole_render'))

    def capture_check(self, entity):
        """Four isolated external views; appearance assistance, not scene grounding.

        Only render visibility changes temporarily. No object transforms,
        articulation, material textures, or physics time may change.
        """
        from pxr import UsdGeom
        from omnigibson.sensors import VisionSensor
        from rummagebench.sim.omnigibson.entity_resolver import resolve_object
        backend = self.backend
        obj = resolve_object(backend._env.scene, entity)
        if obj is None:
            raise RuntimeError('CHECK selected object no longer exists')
        scene = backend._env.scene
        size = 768
        if not hasattr(self, 'check_sensor'):
            sensor = VisionSensor(relative_prim_path='/findingbench_check_camera',
                name='findingbench_check_camera', modalities=['rgb','depth'],
                image_width=size, image_height=size, focal_length=21.445069,
                horizontal_aperture=20, clipping_range=(.001,1000))
            sensor.load(scene); sensor.initialize()
            self.check_sensor = sensor
        sensor = self.check_sensor
        before_tick = backend._sim.current_time_step_index
        before_pose = backend.robot_pose()
        object_pose = tuple(_array(v).copy() for v in obj.get_position_orientation())
        lo, hi = (_array(v) for v in obj.aabb)
        center = (lo + hi) / 2
        radius = float(np.linalg.norm(hi-lo) / 2)
        distance = max(radius / np.sin(np.radians(25)) * 1.12, .05)
        basis = Rotation.from_quat(object_pose[1]).as_matrix()
        saved = []
        pictures = []
        try:
            # Preserve authored visibility opinions exactly, including absent ones.
            with backend._sim.editing_usd():
                for other in scene.objects:
                    if other.name == obj.name:
                        continue
                    attr = UsdGeom.Imageable(other.prim).GetVisibilityAttr()
                    if attr:
                        saved.append((attr, attr.HasAuthoredValueOpinion(), attr.Get()))
                        attr.Set(UsdGeom.Tokens.invisible)
            for azimuth in (0, 90, 180, 270):
                a, elevation = np.radians(azimuth), np.radians(12)
                direction = basis @ np.array([np.cos(a)*np.cos(elevation),
                                              np.sin(a)*np.cos(elevation), np.sin(elevation)])
                eye = center + direction * distance
                forward = (center-eye) / np.linalg.norm(center-eye)
                right = np.cross(forward, basis[:,2]); right /= np.linalg.norm(right)
                down = np.cross(forward, right)
                world_R = np.column_stack((right, down, forward))
                quat = Rotation.from_matrix(world_R @ np.diag([1,-1,-1])).as_quat()
                sensor.set_position_orientation(position=eye, orientation=quat)
                wait_for_intrinsics(sensor, backend._sim)
                for i in range(30):
                    backend._sim.render()
                    if i >= 7 and _camera_in_sync(sensor):
                        break
                else:
                    raise RuntimeError('CHECK renderer pose synchronization failed')
                data, _ = sensor.get_obs()
                rgb = np.asarray(_array(data['rgb']))[...,:3].astype(np.uint8).copy()
                # The only visible scene object is the selected one. Depth
                # removes the distant sky without requiring instance labels,
                # which the fast annotator drops while visibility is isolated.
                depth = np.asarray(_array(data['depth'])).squeeze()
                mask = np.isfinite(depth) & (depth > 0) & (depth < distance + radius * 1.1)
                if rgb.shape != (size,size,3) or mask.shape != (size,size) or not mask.any():
                    raise RuntimeError('CHECK selected object missing from inspection render')
                rgb[~mask] = [235,235,235]
                pictures.append(rgb)
        finally:
            with backend._sim.editing_usd():
                for attr, authored, value in reversed(saved):
                    if authored:
                        attr.Set(value)
                    else:
                        attr.Clear()
            for _ in range(8):
                backend._sim.render()
        if (backend._sim.current_time_step_index != before_tick
                or not all(np.allclose(a,b,atol=1e-8,rtol=0) for a,b in zip(before_pose,backend.robot_pose()))
                or not all(np.allclose(a,_array(b),atol=1e-8,rtol=0) for a,b in zip(object_pose,obj.get_position_orientation()))):
            raise RuntimeError('CHECK unexpectedly changed physical state')
        return np.concatenate((np.concatenate(pictures[:2],axis=1),
                               np.concatenate(pictures[2:],axis=1)),axis=0)
