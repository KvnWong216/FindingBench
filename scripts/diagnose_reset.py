"""Identify state corruption stage without exposing task geometry."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
def arr(x): return x.detach().cpu().numpy() if hasattr(x,"detach") else np.asarray(x)
def main():
 p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--warm-camera",action="store_true");p.add_argument("--trace-setup",action="store_true");p.add_argument("--init-anchor-only",action="store_true");a=p.parse_args()
 b=OmniGibsonBackend(seed=0);report={"ok":False,"stages":[]};out=Path(a.out)
 def check(stage, env=None):
  issues={"bad_object_poses":0,"bad_link_poses":0,"bad_joint_positions":0,"bad_pose_queries":0}
  bad=[]
  for o in (env or b._env).scene.objects:
   try:
    pos,q=o.get_position_orientation()
    if not np.isfinite(arr(pos)).all() or not np.isfinite(arr(q)).all():issues["bad_object_poses"]+=1
   except Exception as e:
    issues["bad_pose_queries"]+=1;bad.append({"object":o.name,"kind":type(e).__name__})
   try:
    if o.n_joints and not np.isfinite(arr(o.get_joint_positions())).all():
     issues["bad_joint_positions"]+=1;bad.append({"object":o.name,"kind":"nonfinite_joint"})
   except Exception as e:
    issues["bad_pose_queries"]+=1;bad.append({"object":o.name,"kind":type(e).__name__})
   for l in o.links.values():
    try:
     pos,q=l.get_position_orientation()
     if not np.isfinite(arr(pos)).all() or not np.isfinite(arr(q)).all():issues["bad_link_poses"]+=1
    except Exception as e:
     issues["bad_pose_queries"]+=1;bad.append({"object":o.name,"link":l.name,"kind":type(e).__name__})
  entry={"stage":stage,**issues,"invalid":bad[:10]};report["stages"].append(entry);out.write_text(json.dumps(report,indent=2));print(entry,flush=True)
  if any(issues.values()):raise RuntimeError("Nonfinite world state at "+stage)
 if a.trace_setup:
  from rummagebench.sim.omnigibson import backend as module
  original_build=module.build_env
  def traced_build(scenario):
   result=original_build(scenario);check("env_constructed",result[0]);return result
  module.build_env=traced_build
  original_settle=b.settle
  count=[0]
  def traced_settle(steps=None):
   import inspect
   print("[settle]",count[0]+1,"steps",steps,"caller",inspect.stack()[1].lineno,flush=True)
   count[0]+=1;check("before_settle_"+str(count[0]))
   original_settle(steps);check("after_settle_"+str(count[0]))
  b.settle=traced_settle
 try:
  scenario=load_scenario("scenarios/knife_search_001/scenario.yaml")
  if a.init_anchor_only:
   scenario.anchors={scenario.robot.init_anchor:scenario.anchors[scenario.robot.init_anchor]}
   report["diagnostic_only"]=True
  b.setup(scenario);check("setup")
  if a.warm_camera:b.capture_visual_frame();check("first_camera")
  b.reset();check("backend_reset")
  b.capture_visual_frame();check("post_reset_capture")
  report["ok"]=True
 except Exception as e:report["error"]=repr(e);traceback.print_exc()
 finally:
  out.write_text(json.dumps(report,indent=2));b.close()
if __name__=="__main__":main()
