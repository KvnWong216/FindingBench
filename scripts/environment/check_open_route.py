"""Private authoring witness: replay legal MOVE/TURN actions into the kitchen."""
import os,json,math,traceback
from pathlib import Path
import numpy as np

OUT=Path(os.environ.get('CLEARANCE_OUTPUT','runs/store_honey_open_route_validation'));OUT.mkdir(parents=True,exist_ok=True)

def main():
    from rummagebench.environment.env import InteractiveSearchEnv
    from rummagebench.authoring.tasks.navigation import FootprintChecker,plan_route,ProtocolLimits
    from rummagebench.skills.move import yaw_from_quat
    from rummagebench.authoring.tasks.room_map import RoomMap
    from rummagebench.skills.move import navigation_aabbs
    import base64
    env=InteractiveSearchEnv(os.environ.get('CLEARANCE_SCENARIO','scenarios/behavior_store_honey_open_route/scenario.yaml'),mode='agent',trace_path=OUT/'private_trace.jsonl')
    obs=env.reset()['observation'];b=env._backend
    (OUT/'initial_panorama.png').write_bytes(base64.b64decode(obs['image_png_b64']))
    pos,q=b.robot_pose();start=(*pos[:2],yaw_from_quat(q))
    assert abs(pos[0])<10 and abs(pos[1])<10, 'Robot spawn lost during scene edit'
    entities=set(b.entity_names());manifest=json.loads(Path(env._scenario.scene.task_scene_manifest).read_text())
    assert not entities.intersection(manifest['removed_objects'])
    assert not b.behavior_goal_satisfied()
    boxes={n:navigation_aabbs(b,n) for n in entities if n not in b.robot_entity_names() and b.describe_entity(n) is not None}
    (OUT/'obstacles.json').write_text(json.dumps(dict(start=start,boxes=boxes),indent=2))
    chk=FootprintChecker([box for arr in boxes.values() for box in arr],.4,.03)
    grid=[(float(x),float(y)) for x in np.arange(-1.6,1.65,.2) for y in np.arange(-.2,3.05,.2)]
    pivots=[p for p in grid if chk.disc_free(*p)]
    scene_dir=Path('/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets/scenes/Rs_int')
    rooms=RoomMap.load(scene_dir,scene_dir.parents[1]/'metadata/room_categories.txt')
    candidates=[p for p in grid if rooms.room_at(*p)=='kitchen_0' and p[1]>2]
    candidates.sort(key=lambda p:math.hypot(p[0]+.25,p[1]-2.45))
    print('start',start,'pivots',len(pivots),'kitchen candidates',len(candidates),flush=True)
    route=None
    for goal in candidates[:30]:
        for yaw in [0,math.pi/2,math.pi,-math.pi/2]:
            if not chk.free(*goal,yaw):continue
            route=plan_route(chk,start,(*goal,yaw),pivots,max_points=60)
            if route:break
        if route:break
    if route is None:raise RuntimeError('No route found by conservative pivot planner')
    print('route',route.actions,flush=True)
    results=[]
    for action in route.actions:
        r=env.step(action);results.append(dict(action=action,feedback=r['feedback'],pose=b.robot_pose()))
        if r['feedback']['code']!='EXECUTED':raise RuntimeError(f'Public execution rejected {action}: {r["feedback"]}')
    pos,q=b.robot_pose();assert rooms.room_at(*pos[:2])=='kitchen_0'
    r=env.step(dict(skill='VIEW',yaw=0,pitch=-70))
    (OUT/'kitchen_panorama.png').write_bytes(base64.b64decode(r['observation']['image_png_b64']))
    # Validate deterministic restoration and persistence of object removals.
    env.reset();reset_pos,reset_q=b.robot_pose()
    assert np.allclose(reset_pos[:2],start[:2],atol=1e-5)
    assert not set(b.entity_names()).intersection(manifest['removed_objects'])
    report=dict(ok=True,scope='Public MOVE/TURN route to kitchen, not complete honey manipulation',
        scenario=os.environ.get('CLEARANCE_SCENARIO','scenarios/behavior_store_honey_open_route/scenario.yaml'),removed_objects=manifest['removed_objects'],
        source_sha256=manifest['source_sha256'],derived_sha256=manifest['derived_sha256'],
        footprint_half_extent_m=.4,obstacle_margin_m=.03,start=start,kitchen_pose=pos,
        actions=results,reset_verified=True,model_used=False,supplied_to_player=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=list));print('PASSED',flush=True)
    env.close()

if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        (OUT/'report.json').write_text(json.dumps(dict(ok=False,error=str(e),traceback=traceback.format_exc()),indent=2));print(traceback.format_exc(),flush=True);code=1
    os._exit(code)
