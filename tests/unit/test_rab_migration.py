import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts'))


def test_retired_data_cannot_silently_launch_the_old_native_backend(tmp_path):
    from spectator import select_task
    (tmp_path/'retired.json').write_text(json.dumps(dict(status='retired')))
    with pytest.raises(ValueError,match='BEHAVIOR.*退役'):select_task(tmp_path,'F001')


def test_rab_only_contract_does_not_count_layout_ids_as_house_types():
    policy=json.loads((Path(__file__).parents[2]/'configs/rab_1k_migration.json').read_text())
    assert policy['expected_tasks']==1000 and policy['behavior_fallback'] is False
    assert policy['diversity']['published_layouts_expected']*policy['diversity']['target_tasks_per_layout']==1000
    assert policy['diversity']['layout_id_is_not_a_house_type']
    assert policy['publication']['drafts_are_not_generated_data']


def test_queue_preserves_source_semantics_and_does_not_promote_slots_to_data(tmp_path):
    import hashlib
    import yaml
    from rab_migration_queue import make_queue
    spec=tmp_path/'source.yaml'
    spec.write_text(yaml.safe_dump(dict(instruction='Find the spoon and bring it to the table.',
        objects=[dict(name='spoon',category='soupspoon')],
        goal=dict(kind='on_top',entity='spoon',receptacle='table'))))
    sha=hashlib.sha256(spec.read_bytes()).hexdigest()
    rows=[dict(parent_task_id=f'F{i:03d}',instruction='Find the spoon and bring it to the table.',
        source_spec=str(spec),source_spec_sha256=sha,original_scene='old_house') for i in range(4)]
    source=tmp_path/'sources.jsonl';source.write_text('\n'.join(json.dumps(r) for r in rows))
    policy=dict(expected_tasks=4,diversity=dict(published_layouts_expected=2,target_tasks_per_layout=2))
    queue=make_queue(source,policy)
    from collections import Counter
    assert Counter(r['requested_layout_id'] for r in queue)=={1:2,2:2}
    assert all(r['instruction'] is None and r['asset_bindings'] is None for r in queue)
    assert all(not r['native_physics_validated'] and r['status']=='pending_rab_binding' for r in queue)
    assert queue[0]['original_goals'][0]['category']=='soupspoon'
    spec.write_text(spec.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='BACKUP_RECIPE_CHANGED'):make_queue(source,policy)


def test_binding_candidates_do_not_relabel_unrelated_household_items():
    from rab_migration_after_assets import coverage
    audit=dict(assets=[dict(id='water_bottle',dynamic_single_body_candidate=True),
                       dict(id='glasses_case',dynamic_single_body_candidate=True)])
    rows=[dict(id='rab_P001',parent_task_id='P001',requested_layout_id=1,
        original_goals=[dict(category='bottle_of_shampoo'),dict(category='eyeglasses')])]
    report=coverage(rows,audit)
    assert report['status_counts']=={'requires_goal_rewrite':1}
    assert report['candidate_asset_coverage']=={}
    assert not report['tasks'][0]['playable'] and report['playable_generated']==0
