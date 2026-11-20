"""Bounded evaluator-only broadphase rejection and background-depth audit."""
import json,traceback,math
from pathlib import Path
import numpy as np
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
from rummagebench.skills.move import footprint_corners,_aabb_overlaps_footprint,yaw_from_quat
out=Path("runs/server_acceptance/debug_20260930_takeover/motion_bounds_1")
out.mkdir(exist_ok=True); report={"ok":False,"checks":[]};b=OmniGibsonBackend()
def save(): (out/"report.json").write_text(json.dumps(report,indent=2))
try:
 b.setup(load_scenario("scenarios/knife_search_001/scenario.yaml"));b.reset()
 p,q=b.robot_pose();yaw=yaw_from_quat(q)
 report["controlled"]=list(b.robot_entity_names())
 frame=b.capture_visual_frame();depth=np.asarray(frame.depth);seg=np.asarray(frame.instance_segmentation)
 report["depth"]={"positive_inf":int(np.isposinf(depth).sum()),"negative_inf":int(np.isneginf(depth).sum()),"nan":int(np.isnan(depth).sum()),"nonfinite_foreground":int(((~np.isfinite(depth)) & (seg!=0)).sum()),"nonpositive_finite":int(((depth<=0)&np.isfinite(depth)).sum()),"sensor":frame.meta["sensor_name"]}
 save()
 candidates={}
 for label,dx,dy,angle in [("initial",0,0,0),("turn5",0,0,5),("turn10",0,0,10),("back5",-.05*math.cos(yaw),-.05*math.sin(yaw),0),("forward5",.05*math.cos(yaw),.05*math.sin(yaw),0)]:
  corners=footprint_corners(p[0]+dx,p[1]+dy,yaw+math.radians(angle),.3);hits=[]
  for name in b.entity_names():
   if name in b.robot_entity_names():continue
   info=b.describe_entity(name)
   if info is None or not info.fixed_base:continue
   aabb=b.entity_aabb(name)
   if aabb is not None and _aabb_overlaps_footprint(aabb,corners,.03):
    hits.append({"entity":name,"category":info.category,"aabb":aabb})
    candidates[name]=aabb
  report["checks"].append({"label":label,"hits":hits});save();print("[bounds]",label,[(h["entity"],h["category"]) for h in hits],flush=True)
 # Audit individual existing collision bounds only for rejecting entities; no full scene mesh export.
 from pxr import Usd,UsdGeom,UsdPhysics
 from rummagebench.sim.omnigibson.usd_collision import iter_prims
 report["collider_bounds"]={}
 for name in candidates:
  obj=b._env.scene.object_registry("name",name)
  cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),includedPurposes=[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
  rows=[]
  for prim in iter_prims(obj.prim):
   if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
   bound=cache.ComputeWorldBound(prim).ComputeAlignedRange()
   lo=[float(v) for v in bound.min];hi=[float(v) for v in bound.max]
   rows.append({"prim":str(prim.GetPath()),"aabb":[lo,hi],"initial_overlap":_aabb_overlaps_footprint((lo,hi),footprint_corners(p[0],p[1],yaw,.3),.03)})
  report["collider_bounds"][name]=rows;save()
 b.validate_physics_state();report["ok"]=True
except Exception as e:report["error"]=repr(e);traceback.print_exc()
finally:save();b.close()
