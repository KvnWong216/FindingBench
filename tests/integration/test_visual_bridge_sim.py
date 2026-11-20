"""Evaluator-private real renderer checks; independent metric calibration is
scripts/calibrate_camera.py. These are not a privileged task-solving policy."""
import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from rummagebench.perception.visual_bridge import VisualBridge
pytestmark=pytest.mark.sim

def test_real_modalities_required(bridge_env):
    frame=bridge_env.capture_visual_frame()
    assert frame.meta["visual_grounding_supported"] is True
    assert frame.depth is not None and frame.instance_segmentation is not None
    assert frame.depth.shape==frame.instance_segmentation.shape==frame.rgb.shape[:2]
    assert np.isfinite(frame.depth).all()
    assert (frame.depth>0).any()
    assert len(frame.meta['instance_labels'])>0
    assert frame.meta["bridge"]=="segmentation"

def test_instance_resolution_to_owning_entity(bridge_env):
    frame=bridge_env.capture_visual_frame()
    bridge=VisualBridge()
    successes=0
    for key in frame.meta['instance_labels']:
        instance=int(key)
        entity=bridge_env.instance_to_entity(instance,frame)
        if entity is None: continue
        rows,cols=np.where(binary_erosion(frame.instance_segmentation==instance,iterations=6))
        if not len(rows): continue
        j=len(rows)//2; u,v=int(cols[j]),int(rows[j])
        resolved,point,_=bridge.resolve(frame,u/(frame.image_width-1),v/(frame.image_height-1),lambda instance: bridge_env.instance_to_entity(instance,frame))
        assert resolved==entity
        assert resolved in bridge_env.entity_names()
        assert np.isfinite(point).all()
        successes+=1
    assert successes>0,"No usable mapped renderer instance; this is a failed prerequisite, not a skip"

@pytest.fixture
def bridge_env(sim_session):
    sim_session.reset()
    return sim_session._backend
