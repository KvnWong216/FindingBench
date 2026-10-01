"""Bounded strict AGENT session using the verified real renderer ID path."""
import sys, json, types, time
from pathlib import Path
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
from rummagebench.sim.omnigibson.raw_instance import install_renderer_instance_capture
from rummagebench.adapters.python_api import _ensure_kinematics_urdf
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
scenario_path="scenarios/knife_search_001/scenario.yaml"
scenario=load_scenario(scenario_path)
scenario.robot.include_sensor_names=["zed_link"]
scenario.robot.obs_modalities=["rgb"]
b=OmniGibsonBackend()
print("[raw-agent] setup.begin",flush=True)
b.setup(scenario)
print("[raw-agent] setup.end",flush=True)
def physics_settle(self,steps=None):
    scope=self._sim.render_on_step(False);scope.__enter__()
    try:
        for _ in range(steps if steps is not None else self._settle_steps):
            self._sim.step();self.validate_physics_state()
    finally:scope.__exit__(None,None,None)
b.settle=types.MethodType(physics_settle,b)
import omni.replicator.core as rep
sensor_name,sensor=next((n,s) for n,s in b._robot.sensors.items() if "zed_link" in n)
sensor.add_modality("depth_linear")
annotator=rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
with b._sim.editing_usd():
    annotator.attach([sensor.render_product])
install_renderer_instance_capture(b,annotator,sensor_name)
scenario.robot.obs_modalities=["rgb","depth_linear","seg_instance_id"]
_ensure_kinematics_urdf(b,scenario,scenario_path)
from rummagebench.core.visual_session import VisualProtocolSession
session=VisualProtocolSession(b,scenario)
session.set_trace(out/"private"/"events.jsonl")
(out/"runtime.json").write_text(json.dumps({"time":time.time(),"modalities":scenario.robot.obs_modalities,"sensor":sensor_name,"budget":session._protocol.max_planning_steps if hasattr(session._protocol,"max_planning_steps") else 16},indent=2))
import agent_acceptance_driver as driver
driver.create_session=lambda *a,**kw:session
sys.argv=[sys.argv[0],"--out",str(out/"public")]
driver.main()
