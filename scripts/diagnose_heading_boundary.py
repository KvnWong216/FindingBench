"""Single-process full-modality heading / native-kinematics boundary probe."""
import sys,json,time,traceback,types
from pathlib import Path
import numpy as np
from PIL import Image
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
from rummagebench.skills.move import execute_move,execute_turn
out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
raw_instance="--raw-instance" in sys.argv
rgb_only="--rgb-only" in sys.argv or raw_instance
report={"diagnostic_only":True,"stages":[]};b=None;handed=False
def record(phase,**kw):
    report["stages"].append({"phase":phase,"time":time.time(),**kw})
    (out/"heading_report.json").write_text(json.dumps(report,indent=2))
    print("[heading]",phase,flush=True)
try:
    scenario=load_scenario("scenarios/knife_search_001/scenario.yaml")
    scenario.robot.include_sensor_names=["zed_link"]
    if rgb_only:scenario.robot.obs_modalities=["rgb"]
    b=OmniGibsonBackend();b.setup(scenario)
    original_settle=b.settle
    def physics_settle(self,steps=None):
        scope=self._sim.render_on_step(False);scope.__enter__()
        try:
            for i in range(steps if steps is not None else self._settle_steps):
                self._sim.step();self.validate_physics_state()
        finally:scope.__exit__(None,None,None)
    b.settle=types.MethodType(physics_settle,b)
    original_capture=b.capture_visual_frame
    def capture(self):
        self.validate_physics_state()
        for _ in range(4):self._sim.render()
        frame=original_capture()
        self.validate_physics_state()
        return frame
    b.capture_visual_frame=types.MethodType(capture,b)
    def frame(label):
        record(label+".before",pose=b.robot_pose(),pin_loaded="pinocchio" in sys.modules,coal_loaded="coal" in sys.modules)
        if rgb_only:
            sensor=next(s for n,s in b._robot.sensors.items() if "zed_link" in n)
            K=sensor.intrinsic_matrix
            for _ in range(6):b._sim.render()
            obs,_=sensor.get_obs()
            rgb=obs["rgb"].cpu().numpy()[...,:3]
            Image.fromarray(rgb.astype(np.uint8)).save(out/(label+".png"))
            record(label+".after",modalities=sorted(obs),grounding_supported=False)
        else:
            f=b.capture_visual_frame()
            Image.fromarray(f.rgb).save(out/(label+".png"))
            record(label+".after",sensor=f.meta.get("sensor_name"),modalities_shape=[list(f.rgb.shape),list(f.depth.shape),list(f.instance_segmentation.shape)])
    assert "pinocchio" not in sys.modules and "coal" not in sys.modules
    b.reset();frame("backend_initial")
    for i,(kind,value) in enumerate([("turn",90),("move",.1),("turn",90),("turn",90)]):
        pose=b.robot_pose()
        safe,_=(execute_move(b,pose,value,.4,.05,.03) if kind=="move" else execute_turn(b,pose,value,.4,5.,.03))
        record("action."+str(i),kind=kind,value=value,safe=safe)
        if safe:b.settle(5)
        frame("backend_action_"+str(i))
    if raw_instance:
        import omni.replicator.core as rep
        sensor_name,sensor=next((n,s) for n,s in b._robot.sensors.items() if "zed_link" in n)
        sensor.add_modality("depth_linear")
        frame("raw_before_attach")
        record("raw_instance_attach.before")
        annotator=rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
        with b._sim.editing_usd():
            annotator.attach([sensor.render_product])
        def raw_frame(label):
            b.validate_physics_state()
            for _ in range(6):b._sim.render()
            data=annotator.get_data()
            arr=np.asarray(data["data"])
            info=data.get("info",{})
            np.save(out/(label+".npy"),arr)
            (out/(label+"_info.json")).write_text(json.dumps(info,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x),indent=2))
            obs,_=sensor.get_obs()
            rgb=obs["rgb"].cpu().numpy()[...,:3]
            Image.fromarray(rgb.astype(np.uint8)).save(out/(label+".png"))
            record(label+".after",shape=list(arr.shape),dtype=str(arr.dtype),unique_ids=int(len(np.unique(arr))),info_keys=list(info),modalities=sorted(obs))
        raw_frame("raw_heading180")
        for i in range(2):
            safe,_=execute_turn(b,b.robot_pose(),90,.4,5.,.03)
            record("raw_turn."+str(i),safe=safe)
            if not safe:raise RuntimeError("diagnostic turn rejected")
            b.settle(5);raw_frame("raw_turn_"+str(i))
        record("raw_instance_ready_for_adapter")
        deadline=time.monotonic()+1800
        while not (out/"continue_to_agent.json").exists():
            if time.monotonic()>deadline:raise TimeoutError("waiting for verified raw instance adapter")
            time.sleep(.2)
        from rummagebench.sim.omnigibson.raw_instance import install_renderer_instance_capture
        install_renderer_instance_capture(b,annotator,sensor_name)
        scenario.robot.obs_modalities=["rgb","depth_linear","seg_instance_id"]
    elif rgb_only:
        sensor=next(s for n,s in b._robot.sensors.items() if "zed_link" in n)
        for modality in ("depth_linear","seg_semantic","seg_instance"):
            record("attach."+modality+".before",pose=b.robot_pose())
            sensor.add_modality(modality)
            frame("attach_"+modality)
            b.validate_physics_state()
        record("rgb_then_all_modalities_at_heading180_complete")
        raise SystemExit(0)
    record("before_agent_constructor",pose=b.robot_pose())
    from rummagebench.core.visual_session import VisualProtocolSession
    session=VisualProtocolSession(b,scenario)
    frame("after_agent_constructor")
    record("handoff_to_public_driver")
    import agent_acceptance_driver as driver
    driver.create_session=lambda *a,**kw:session
    sys.argv=[sys.argv[0],"--out",str(out/"public")]
    handed=True;driver.main()
except Exception as e:
    record("exception",error=repr(e));traceback.print_exc();raise SystemExit(2)
finally:
    if b is not None and not handed:b.close()
