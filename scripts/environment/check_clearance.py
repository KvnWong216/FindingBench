"""Private local doorway witnesses, using unchanged public MOVE/TURN skills."""
import argparse,base64,json,math,os,traceback
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    from rummagebench.environment.env import InteractiveSearchEnv
    from rummagebench.core.scenario import AnchorSpec
    from rummagebench.authoring.tasks.navigation import FootprintChecker
    e=InteractiveSearchEnv(a.scenario,mode='agent',trace_path=out/'private_trace.jsonl');e.reset();b=e._backend
    start=b.robot_pose();doc=json.loads(Path(e._scenario.scene.clearance_manifest).read_text())
    report=dict(scope='Local public-skill doorway crossing/turning; not complete task solvability',model_used=False,supplied_to_player=False,portals=[])
    try:
        for index,w in enumerate(doc['wall_warps']):
            if w.get('portal') is False:continue
            axis=w['axis'];normal=1-axis;c=w.get('wall_plane');bounds=w['target'][1:3]
            if c is None:continue
            center=[0.,0.,start[0][2]];center[axis]=sum(bounds)/2;center[normal]=c
            checker=FootprintChecker.from_backend(b,.4,.03)
            item=dict(entity=w['entity'],center=center,nominal_width_m=bounds[1]-bounds[0],attempts=[])
            report['portals'].append(item)
            # Choose a clear approach on either side; explicit private setup,
            # never exposed as an agent teleport or navigation skill.
            found=False
            for sign in [-1,1]:
                approach=center.copy();approach[normal]+=sign*.65
                yaw=math.atan2(-sign if normal==1 else 0,-sign if normal==0 else 0)
                if not checker.free(approach[0],approach[1],yaw):continue
                if not checker.move_free(approach[0],approach[1],yaw,1.3):continue
                b.teleport_robot(AnchorSpec(position=approach,orientation=[0,0,math.sin(yaw/2),math.cos(yaw/2)]))
                actions=[dict(skill='MOVE',distance_cm=65),dict(skill='TURN',angle_deg=90),dict(skill='TURN',angle_deg=-90),dict(skill='MOVE',distance_cm=65)]
                for action in actions:
                    r=e.step(action);item['attempts'].append(dict(action=action,feedback=r['feedback']))
                    if r['feedback']['code']!='EXECUTED':raise RuntimeError(f'Public skill rejected {action}: {r["feedback"]}')
                (out/f'portal_{index:02d}.png').write_bytes(base64.b64decode(r['observation']['image_png_b64']))
                item['ok']=True;found=True;break
            if not found:item['ok']=False;item['reason']='No clear straight approach across the opening with the current furniture'
        e.reset();assert np.allclose(b.robot_pose()[0],start[0],atol=1e-4)
        report['reset_verified']=True;report['ok']=bool(report['portals']) and all(x['ok'] for x in report['portals'])
        if e._scenario.behavior_task:report['goal_initially_unsatisfied']=not b.behavior_goal_satisfied()
    except Exception as error:report['ok']=False;report['error']=str(error);report['traceback']=traceback.format_exc()
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    return 0 if report['ok'] else 1

if __name__=='__main__':
    try:code=main()
    except BaseException:traceback.print_exc();code=1
    os._exit(code)
