"""Real GPU smoke for the official pilot and motion-decoupled VIEW, no model."""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',default='scenarios/behavior_groceries_pilot/scenario.yaml')
    p.add_argument('--output',type=Path,default=Path('runs/ygbench_setup/pilot_smoke'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report={'ok':False,'checks':{},'model_used':False}
    try:
        from rummagebench.environment.env import InteractiveSearchEnv
        env=InteractiveSearchEnv(a.scenario,mode='agent',trace_path=a.output/'private_events.jsonl')
        obs=env.reset()['observation']
        report['checks']['reset']=True
        report['checks']['skills']=obs['skill_library']
        (a.output/'initial_panorama.png').write_bytes(base64.b64decode(obs['image_png_b64']))
        backend=env._backend
        pano=env._session._panorama_frame
        assert pano.rgb.shape == (512,1024,3),pano.rgb.shape
        assert float(np.std(pano.rgb)) > 2., 'panorama is blank'
        assert pano.meta['visual_grounding_supported']
        labels=pano.meta['instance_labels']
        entities=set(backend.entity_names())
        visible={labels.get(str(i)) for i in np.unique(pano.instance_segmentation)} & entities
        assert visible, 'no renderer IDs resolve to scene objects'
        from scipy.spatial.transform import Rotation
        from rummagebench.sim.omnigibson.raw_instance import _array
        from rummagebench.perception.camera_geometry import world_from_usd_camera
        camera=backend._panorama_camera
        sensor=camera.faces[0][0]
        pos,quat=sensor.get_position_orientation()
        head_pos,_=backend._robot.links[camera.config.head_link].get_position_orientation()
        _,reference_quat=backend._robot.links[camera.config.mount_frame_link].get_position_orientation()
        reference=Rotation.from_quat(_array(reference_quat)).as_matrix()
        np.testing.assert_allclose(_array(pos),_array(head_pos)+reference@camera.config.offset,atol=1e-4)
        transform=world_from_usd_camera(_array(pos),_array(quat))
        np.testing.assert_allclose(transform[:3,2],reference[:,0],atol=1e-4)
        report['checks']['head_mount_and_renderer']={'head_position':_array(head_pos).tolist(),
            'camera_position':_array(pos).tolist(),'visible_entity_count':len(visible),
            'rgb_std':float(np.std(pano.rgb))}
        before=np.array(backend.robot_pose()[0]+backend.robot_pose()[1])
        joints_before=backend.robot_joint_positions()
        time_before=backend._sim.current_time_step_index
        old_frame=obs['frame_id']
        records=[]
        for yaw,pitch in [(0,0),(90,0),(-90,0),(180,0),(0,60),(0,-60)]:
            result=env.step({'skill':'VIEW','yaw':yaw,'pitch':pitch})
            assert result['feedback']['code']=='EXECUTED',result['feedback']
            obs=result['observation']
            assert obs['frame_id']!=old_frame
            after=np.array(backend.robot_pose()[0]+backend.robot_pose()[1])
            np.testing.assert_allclose(after,before,atol=1e-9,rtol=0)
            assert backend.robot_joint_positions()==joints_before
            assert backend._sim.current_time_step_index==time_before
            (a.output/f'view_{yaw}_{pitch}.png').write_bytes(base64.b64decode(obs['image_png_b64']))
            records.append({'yaw':yaw,'pitch':pitch,'frame_id':obs['frame_id']})
        report['checks']['view_no_physics_or_motion']=records
        stale=env.step({'skill':'GRASP','point':{'frame_id':old_frame,'x':.5,'y':.5}})
        assert stale['feedback']['code']=='INVALID_ACTION'
        report['checks']['stale_frame_rejected']=True
        removed=env.step({'skill':'OBSERVE','point':{'frame_id':obs['frame_id'],'x':.5,'y':.5}})
        assert removed['feedback']['code']=='INVALID_ACTION'
        report['checks']['observe_removed']=True
        ground = [obj.name for obj in backend._env.scene.objects
                  if obj.fixed_base and obj.category in {'floors', 'paver'}]
        assert ground
        assert all(backend.navigation_obstacle_aabbs(name) == [] for name in ground)
        report['checks']['walkable_ground']=ground
        for action in ({'skill':'MOVE','distance_cm':5}, {'skill':'TURN','angle_deg':5}):
            moved=env.step(action)
            assert moved['feedback']['code']=='EXECUTED', moved['feedback']
        report['checks']['move_and_turn']=True
        reset=env.reset()['observation']
        assert reset['planning_step']==0
        report['checks']['second_reset']=True
        report['checks']['official_goal_initially_unsatisfied']=not backend.behavior_goal_satisfied()
        done=env.step({'skill':'REPORT_DONE'})
        assert done['episode_status']=='FAIL_FALSE_COMPLETION'
        report['checks']['official_report_done']=True
        report['ok']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        report['traceback']=traceback.format_exc()
    (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,indent=2,default=str),flush=True)
    return 0 if report['ok'] else 1

if __name__=='__main__':
    code=main()
    sys.stdout.flush();sys.stderr.flush()
    # Kit teardown can hang; reports are already flushed. Process exit frees GPU.
    os._exit(code)
