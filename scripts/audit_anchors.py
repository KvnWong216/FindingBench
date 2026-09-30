"""Evaluator-only, non-mutating anchor collision audit using real collider geometry."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
from rummagebench.sim.omnigibson.usd_collision import collect_entity_collision
from rummagebench.feasibility.fcl_compat import collision_backend,transform3

def matrix(position,orientation):
 t=np.eye(4);t[:3,:3]=Rotation.from_quat(orientation).as_matrix();t[:3,3]=position;return t

def main():
 p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
 out=Path(a.out);report={"ok":False,"anchors":[]}
 b=OmniGibsonBackend(seed=0)
 try:
  scenario=load_scenario("scenarios/knife_search_001/scenario.yaml");b.setup(scenario)
  b.validate_physics_state()
  before=b.dump_state().clone()
  coal=collision_backend();cache={}
  robots=collect_entity_collision(b._robot,cache)
  worlds=[]
  for obj in b._env.scene.objects:
   if obj.name!=b._robot.name:worlds.extend(collect_entity_collision(obj,cache))
  assert robots and worlds
  origin=matrix(*b.robot_pose())
  world_objects=[(w,coal.CollisionObject(w.geometry,transform3(coal,w.pose.position,Rotation.from_quat(w.pose.orientation).as_matrix()))) for w in worlds if w.category not in ("floors","floor")]
  def hits(position,orientation):
   delta=matrix(position,orientation)@np.linalg.inv(origin)
   collisions=set()
   for r in robots:
    rt=delta@matrix(r.pose.position,r.pose.orientation)
    rob=coal.CollisionObject(r.geometry,transform3(coal,rt[:3,3],rt[:3,:3]))
    bound=None
    if r.aabb is not None:
     lo,hi=map(np.asarray,r.aabb)
     corners=np.array([[x,y,z,1.] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
     corners=(delta@corners.T).T[:,:3];bound=(corners.min(0),corners.max(0))
    for w,world in world_objects:
     if bound is not None and w.aabb is not None:
      if np.any(bound[1]<np.asarray(w.aabb[0])) or np.any(np.asarray(w.aabb[1])<bound[0]):continue
     result=coal.CollisionResult()
     if coal.collide(rob,world,coal.CollisionRequest(),result):
      collisions.add((w.entity,w.category,r.link,w.link,w.approximation))
   return [list(c) for c in sorted(collisions,key=str)]
  for index,(name,anchor) in enumerate(scenario.anchors.items()):
   collision=hits(anchor.position,anchor.orientation)
   record={"index":index,"name":name,"collision_free":not collision,"collisions":collision}
   if collision:
    offsets=sorted(((x/10,y/10) for x in range(-8,9) for y in range(-8,9) if x or y),key=lambda xy:xy[0]**2+xy[1]**2)
    for dx,dy in offsets:
     pos=np.asarray(anchor.position)+[dx,dy,0]
     if not hits(pos,anchor.orientation):
      record["private_candidate_position"]=pos.tolist();record["offset_distance_m"]=float(np.hypot(dx,dy));break
   report["anchors"].append(record);out.write_text(json.dumps(report,indent=2))
   print("[anchor_audit]",index,"collision_pairs",len(collision),"safe_candidate", "private_candidate_position" in record,flush=True)
  after=b.dump_state()
  report["snapshot_equal"]=bool(np.array_equal(before.cpu().numpy(),after.cpu().numpy(),equal_nan=True))
  report["ok"]=report["snapshot_equal"] and all(r["collision_free"] for r in report["anchors"])
 except Exception as e:report["error"]=repr(e);traceback.print_exc()
 finally:out.write_text(json.dumps(report,indent=2));b.close()
if __name__=="__main__":main()
