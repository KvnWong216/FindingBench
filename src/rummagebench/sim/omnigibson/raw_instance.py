"""Real renderer instance-ID frames, independent of semantic-class annotators.

IDs stay renderer-produced and frame-local. Only labels are canonicalized to
the exact owning scene object; unknown/invalid paths never resolve to a target.
"""
from __future__ import annotations
import types
import numpy as np
from scipy.spatial.transform import Rotation

def canonical_labels(labels, object_roots):
    if not isinstance(labels, dict) or not labels:
        raise RuntimeError("Renderer instance-ID labels unavailable")
    roots=sorted(object_roots.items(),key=lambda p:len(p[0]),reverse=True)
    result={}
    for key,path in labels.items():
        if not isinstance(path,str):
            raise RuntimeError("Unexpected renderer instance-ID label type")
        owner=next((name for root,name in roots if path==root or path.startswith(root+"/")),None)
        result[str(key)]=owner if owner is not None else ("unlabelled" if path=="INVALID" else path)
    return result

def complete_instance_frame(frame, raw, object_roots, sensor_name):
    if frame.meta.get("sensor_name") != sensor_name:
        raise RuntimeError("Renderer instance-ID and RGB cameras differ")
    if not isinstance(raw,dict) or "data" not in raw:
        raise RuntimeError("Renderer instance-ID data unavailable")
    ids=np.asarray(raw["data"])
    if ids.shape != (frame.image_height,frame.image_width) or ids.dtype.kind not in "ui":
        raise RuntimeError("Invalid renderer instance-ID buffer")
    labels=canonical_labels(raw.get("info",{}).get("idToLabels"),object_roots)
    frame.instance_segmentation=ids.copy()
    remaining=[reason for reason in frame.meta.get("unsupported_reason","").split(", ")
               if reason and reason not in {"same-camera instance segmentation","same-camera instance labels"}]
    frame.meta.update(instance_labels=labels,instance_source="renderer:instance_id_segmentation_fast",
                      visual_grounding_supported=not remaining,
                      unsupported_reason=", ".join(remaining),
                      bridge="segmentation" if not remaining else "unsupported")
    return frame

def _array(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)

def _camera_in_sync(sensor):
    pos,quat=sensor.get_position_orientation()
    world=np.eye(4)
    world[:3,:3]=Rotation.from_quat(_array(quat)).as_matrix()
    world[:3,3]=_array(pos)
    rendered=_array(sensor.camera_parameters["cameraViewTransform"]).reshape(4,4).T
    return bool(np.isfinite(rendered).all() and
                np.allclose(rendered,np.linalg.inv(world),atol=1e-4,rtol=0))

def install_renderer_instance_capture(backend, annotator, sensor_name):
    """Attach an already-created real annotator to this backend's capture path."""
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    sensor=backend._robot.sensors[sensor_name]
    def capture(self):
        self.validate_physics_state()
        # Finish lazy camera-parameter attachment before any frame buffers.
        _=sensor.intrinsic_matrix
        for i in range(8):
            self._sim.render()
            if i>=5 and _camera_in_sync(sensor):break
        else:
            raise RuntimeError("Rendered camera parameters do not match current sensor pose")
        frame=OmniGibsonBackend.capture_visual_frame(self)
        if not _camera_in_sync(sensor):
            raise RuntimeError("Camera pose changed while collecting modalities")
        raw=annotator.get_data()
        roots={str(obj.prim_path):obj.name for obj in self._env.scene.objects}
        frame=complete_instance_frame(frame,raw,roots,sensor_name)
        self._instance_labels=dict(frame.meta["instance_labels"])
        self.validate_physics_state()
        return frame
    backend.capture_visual_frame=types.MethodType(capture,backend)
