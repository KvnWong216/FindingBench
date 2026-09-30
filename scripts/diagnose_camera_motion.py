"""Bounded private audit of camera identity, restore and motion rejection."""
import json,traceback,time,hashlib
from pathlib import Path
import numpy as np
from agent_acceptance_driver import array,snapshot,compare
from rummagebench.adapters.python_api import create_session
from rummagebench.skills.move import footprint_corners,_aabb_overlaps_footprint,yaw_from_quat
out=Path("runs/server_acceptance/debug_20260930_takeover/camera_motion_2")
out.mkdir(exist_ok=True)
report={"ok":False,"stages":[]}
session=None
def record(stage, **values):
 report["stages"].append({"stage":stage,**values})
 (out/"report.json").write_text(json.dumps(report,indent=2))
 print("[camera_motion]",stage,flush=True)
def poses(b):
 result={}
 for obj in b._env.scene.objects:
  for name,link in obj.links.items():
   p,q=link.get_position_orientation()
   result[obj.name+"/"+name]=(array(p),array(q),np.array([]))
 for name,s in b._robot.sensors.items():
  if hasattr(s,"get_position_orientation"):
   p,q=s.get_position_orientation()
   result["sensor/"+name]=(array(p),array(q),np.array([]))
 return result
try:
 session=create_session("scenarios/knife_search_001/scenario.yaml",run_dir=out/"private",mode="agent")
 obs=session.reset();b=session._backend
 record("code",backend_file=__import__("inspect").getfile(type(b)))
 frame=b.capture_visual_frame()
 record("ready",sensor_name=frame.meta.get("sensor_name"),
        sensors=[{"name":n,"prim_path":s.prim_path} for n,s in b._robot.sensors.items()],
        sim_device=str(b._sim.device))
 before=snapshot(b);before_poses=poses(b)
 p,q=b.robot_pose();yaw=yaw_from_quat(q);hits=[]
 # This audits the current footprint, not a private action selection.
 for name in b.entity_names():
  info=b.describe_entity(name)
  if info is None or not getattr(info,"fixed_base",False):continue
  aabb=b.entity_aabb(name)
  if aabb is not None and _aabb_overlaps_footprint(aabb,footprint_corners(p[0],p[1],yaw,.3),0):
   hits.append({"entity":name,"category":getattr(info,"category",None)})
 record("initial_footprint",fixed_entity_overlaps=hits,diagnostic_half_extent_m=.3)
 result=session.step({"skill":"OBSERVE","point":{"frame_id":obs.frame_id,"x":.90,"y":.57}})
 record("observe",feedback=result.observation.feedback.model_dump(),object_errors=compare(before,snapshot(b)),
        link_camera_errors=compare(before_poses,poses(b)))
 # Pure rendering/sync probes: no physics integration.
 for name,fn in [("render",b._sim.render),("sync",b._sim.sync_physx_to_fabric),("render_after_sync",b._sim.render)]:
  fn()
  frame=b.capture_visual_frame()
  from PIL import Image
  Image.fromarray(frame.rgb).save(out/(name+".png"))
  record(name,link_camera_errors=compare(before_poses,poses(b)),rgb_sha256=hashlib.sha256(frame.rgb.tobytes()).hexdigest())
 b.validate_physics_state();report["ok"]=True
except Exception as e:report["error"]=repr(e);traceback.print_exc()
finally:
 (out/"report.json").write_text(json.dumps(report,indent=2))
 if session is not None:session._backend.close()
