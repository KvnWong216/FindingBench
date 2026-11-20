from pathlib import Path
import pytest
from rummagebench.datasets.behavior import audit_definition, parse_bddl
from rummagebench.core.scenario import BehaviorTaskSpec
from rummagebench.datasets.behavior import assert_rigid_task
from rummagebench.skills.report_done import goal_satisfied

ROOT=Path(__file__).resolve().parents[2]/'assets/behavior_official/bddl'


def test_official_pilot_retains_garage_and_kitchen_and_native_goal():
    task=BehaviorTaskSpec(activity_name='carrying_in_groceries',bddl_root=str(ROOT))
    audit=assert_rigid_task(task)
    assert audit['rigid_only']
    text=Path(audit['definition']).read_text()
    assert all(room in text for room in ['garage','kitchen'])
    assert set(audit['predicates'])=={'ontop','inroom','inside','open'}


def test_deformable_and_nonmechanical_tasks_excluded(tmp_path):
    p=tmp_path/'washing';p.mkdir();p=p/'problem0.bddl'
    p.write_text('(define (problem washing) (:objects cloth_1 - cloth) (:init (ontop cloth_1 floor)) (:goal (clean cloth_1)))')
    a=audit_definition(p,{'cloth':{'cloth':{},'deformable':{}}})
    assert not a['rigid_only']
    assert 'unsupported_predicate:clean' in a['exclusions']


def test_native_report_uses_official_goal_not_holding():
    from types import SimpleNamespace
    scenario=SimpleNamespace(behavior_task=object())
    backend=SimpleNamespace(behavior_goal_satisfied=lambda:True)
    assert goal_satisfied(None,scenario,backend)
    backend.behavior_goal_satisfied=lambda:False
    assert not goal_satisfied(None,scenario,backend)


def test_prepare_requires_actual_official_template_and_no_authored_layout(tmp_path):
    import importlib.util
    import yaml
    script=Path(__file__).resolve().parents[2]/'scripts/prepare_behavior_pilot.py'
    spec=importlib.util.spec_from_file_location('prepare_pilot',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'scenario.yaml'
    with pytest.raises(FileNotFoundError): module.prepare(ROOT,tmp_path,out)
    assert not out.exists()
    folder=tmp_path/'scenes/house_double_floor_lower/json';folder.mkdir(parents=True)
    # Unit fixture represents only file discovery, never simulation evidence.
    (folder/'house_double_floor_lower_task_carrying_in_groceries_0_0_template.json').write_text('{}')
    result=module.prepare(ROOT,tmp_path,out)
    doc=yaml.safe_load(out.read_text())
    assert result['status']=='configured_not_sim_certified'
    assert not doc['objects'] and not doc['placements'] and not doc['anchors']
    assert doc['target'] is None
    assert doc['robot']['panorama']['head_link']=='zed_link'
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.sim.omnigibson.env_factory import build_env_config
    config=build_env_config(ScenarioSpec.model_validate(doc))
    assert config['task']['type']=='BehaviorTask'
    assert config['task']['online_object_sampling'] is False
    assert config['scene']['load_task_relevant_only'] is False
    assert 'position' not in config['robots'][0]


def test_catalog_finds_separate_challenge_package_without_demo_or_heldout(tmp_path):
    from rummagebench.datasets.behavior import catalog
    base = tmp_path / '2026-challenge-task-instances'
    filename = 'house_double_floor_lower_task_carrying_in_groceries_0_0_template.json'
    for directory in ['scenes/house_double_floor_lower/json',
                      'scenes/house_double_floor_lower/json/demo',
                      'heldout_scenes/house_double_floor_lower/json']:
        folder = base / directory
        folder.mkdir(parents=True)
        (folder / filename).write_text('{}')
    for root in [tmp_path, tmp_path / 'behavior-1k-assets']:
        records = catalog(ROOT, root)
        task = next(t for t in records['tasks'] if t['activity_name']=='carrying_in_groceries')
        assert len(task['instances']) == 1
        assert task['instances'][0]['template'] == str(base / 'scenes/house_double_floor_lower/json' / filename)
