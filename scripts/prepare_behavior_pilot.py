"""Bind the selected official task to an installed official cached instance.

No sampling, placement generation, or invented scene/robot start pose.
"""
import argparse
import json
from pathlib import Path
import yaml
from rummagebench.datasets.behavior import catalog
from rummagebench.core.scenario import ScenarioSpec

SELECTION = Path(__file__).resolve().parents[1] / 'configs/benchmarks/behavior_pilot.yaml'


def prepare(bddl_root, assets_root, output):
    selection = yaml.safe_load(SELECTION.read_text())
    task_name = selection["activity_name"]
    records = catalog(bddl_root, assets_root)
    task = next(t for t in records['tasks'] if t['activity_name']==task_name and t['activity_definition_id']==selection['activity_definition_id'])
    if not task['rigid_only']:
        raise ValueError('pilot failed rigid audit')
    if not task['instances']:
        raise FileNotFoundError('official pre-sampled pilot instance not installed; no synthetic fallback')
    matches = [i for i in task['instances'] if i['scene_model'] == selection['scene_model']]
    if not matches:
        raise FileNotFoundError('selected house_double_floor_lower official task instance is not installed')
    instance = matches[0]  # deterministic; full official scene, all rooms retained
    spec = ScenarioSpec.model_validate(dict(
        id='behavior_groceries_pilot',
        instruction='Bring the groceries from the car into the refrigerator and close the car and refrigerator.',
        scene=dict(model=instance['scene_model']),
        behavior_task=dict(activity_name=task_name,activity_definition_id=selection['activity_definition_id'],bddl_root=str(Path(bddl_root).resolve()),
                           activity_instance_id=instance['activity_instance_id']),
        robot=dict(model=selection['robot'],panorama={},
                   image_width=640,image_height=480,obs_modalities=['rgb'],
                   kinematics=dict(urdf_path='build/robots/r1pro.urdf',base_link='base_link',
                                   end_effector_link='right_eef_link',controlled_joints='auto')),
        skills=['OPEN','CLOSE','GRASP','PLACE'],expose_available_skills=False,
        termination=dict(max_planning_steps=selection['max_planning_steps'],fail_on_wrong_grasp=False,
                         fail_on_unsafe_action=False,succeed_when_holding_target=False)))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(yaml.safe_dump(spec.model_dump(mode='json'),sort_keys=False))
    return dict(status='configured_not_sim_certified',task=task_name,instance=instance,
                definition_sha256=task['sha256'],scenario=str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bddl-root',type=Path,default=Path('assets/behavior_official/bddl'))
    p.add_argument('--assets-root',type=Path,required=True,help='behavior-1k-assets directory')
    p.add_argument('--output',type=Path,default=Path('scenarios/behavior_groceries_pilot/scenario.yaml'))
    a=p.parse_args()
    print(json.dumps(prepare(a.bddl_root,a.assets_root,a.output),indent=2))
