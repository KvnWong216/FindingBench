"""Offline authoring inventory of native wall/door primitives (no model)."""
import argparse,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.env_factory import apply_runtime_env,build_env
    apply_runtime_env();spec=load_scenario(a.scenario);env,_=build_env(spec)
    import numpy as np
    from pxr import UsdGeom,Usd,UsdPhysics
    from rummagebench.sim.omnigibson.usd_collision import iter_prims,collider_navigation_aabbs,physics_link_matrix
    rows=[];objects=[]
    for obj in env.scene.objects:
        pos,q=obj.get_position_orientation()
        objects.append(dict(name=obj.name,category=obj.category,position=pos.tolist(),orientation=q.tolist(),boxes=collider_navigation_aabbs(obj)))
        if obj.category not in ('walls','door'):continue
        for prim in iter_prims(obj.prim):
            if not prim.IsA(UsdGeom.Gprim):continue
            if not prim.IsA(UsdGeom.Mesh):raise ValueError('Inventory expects native wall/door meshes')
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
            body=prim
            while body!=obj.prim and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            base=np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0.),float)
            matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.),float)@np.linalg.inv(base)@physics_link_matrix(obj,body.GetName(),base)
            world=points@matrix[:3,:3]+matrix[3,:3]
            rows.append(dict(object=obj.name,category=obj.category,path=str(prim.GetPath()),type=prim.GetTypeName(),collision=prim.HasAPI(UsdPhysics.CollisionAPI),lo=world.min(0).tolist(),hi=world.max(0).tolist()))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(scene_model=spec.scene.model,primitives=rows,objects=objects),indent=2));print('DONE',flush=True)

if __name__=='__main__':
    import traceback
    try:main()
    except SystemExit as exc:
        import sys
        sys.stdout.flush();os._exit(exc.code or 0)
    except BaseException:traceback.print_exc();os._exit(1)
    os._exit(0)
