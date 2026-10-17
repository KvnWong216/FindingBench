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
