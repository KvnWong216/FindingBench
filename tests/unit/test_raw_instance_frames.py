from types import SimpleNamespace
import numpy as np
import pytest
from rummagebench.sim.omnigibson.raw_instance import canonical_labels,complete_instance_frame
from rummagebench.perception.visual_bridge import VisualBridge,VisualGroundingError

def test_exact_ancestor_ownership_preserves_unknowns():
    labels={"7":"/World/a/mesh","8":"/World/ab/mesh","9":"INVALID"}
    assert canonical_labels(labels,{"/World/a":"target"})=={"7":"target","8":"/World/ab/mesh","9":"unlabelled"}

def frame(reason="same-camera instance segmentation, same-camera instance labels"):
    return SimpleNamespace(image_height=2,image_width=2,depth=np.ones((2,2)),
        meta={"sensor_name":"head","unsupported_reason":reason,"visual_grounding_supported":False})

def raw():
    return {"data":np.array([[7,0],[7,9]],dtype=np.uint32),
            "info":{"idToLabels":{"7":"/World/a/mesh","9":"INVALID"}}}

def test_real_ids_are_copied_and_missing_depth_stays_unsupported():
    f=frame("same-camera depth, same-camera instance segmentation, same-camera instance labels")
    r=raw();depth=f.depth.copy()
    complete_instance_frame(f,r,{"/World/a":"a"},"head")
    assert not f.meta["visual_grounding_supported"]
    assert f.meta["unsupported_reason"]=="same-camera depth"
    np.testing.assert_array_equal(f.depth,depth)
    np.testing.assert_array_equal(f.instance_segmentation,r["data"])
    r["data"][:]=0
    assert f.instance_segmentation[0,0]==7

def test_complete_real_instance_can_fill_only_instance_requirements():
    f=complete_instance_frame(frame(),raw(),{"/World/a":"a"},"head")
    assert f.meta["visual_grounding_supported"]
    assert f.meta["instance_labels"]["7"]=="a"

@pytest.mark.parametrize("change",["camera","shape","labels"])
def test_bad_renderer_contract_fails_closed(change):
    f=frame();r=raw()
    if change=="camera":f.meta["sensor_name"]="wrist"
    if change=="shape":r["data"]=np.zeros((3,3),dtype=np.uint32)
    if change=="labels":r["info"]={}
    with pytest.raises(RuntimeError):complete_instance_frame(f,r,{"/World/a":"a"},"head")
