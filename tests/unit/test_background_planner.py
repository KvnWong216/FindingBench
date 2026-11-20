import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import background_planner as planner
from enrich_background import plan_props


def policy():
    return dict(version=planner.VERSION, supports=sorted(planner.SUPPORTS),
        room_categories=planner.ROOM_CATEGORIES,
        assets={category: [dict(category=category, model='native_'+category, size=[.08, .08, .08],
                                metadata_sha256='a'*64)] for category in
                {c for choices in planner.ROOM_CATEGORIES.values() for c in choices}})


def example():
    task = dict(id='unit_task', parent_task_id='T001', instruction='Find the notebook on the study shelf.',
                objects=[dict(name='target', category='notebook')])
    records = dict(target=dict(category='notebook', n_joints=0, rooms=['office_0'],
                               aabb=[[5., 5., .7], [5.2, 5.2, .72]]),
        table=dict(category='breakfast_table', n_joints=0, rooms=['dining_room_0'],
                   aabb=[[0., 0., 0.], [2., 1., .8]]),
        plant=dict(category='pot_plant', n_joints=0, rooms=['dining_room_0'],
                   aabb=[[.9, .4, .8], [1.1, .6, 1.1]]))
    return task, records


def test_semantic_family_and_comparison_inventory_exclusions():
    excluded, families = planner.excluded_categories(dict(
        instruction='Bring all the fruit from the basket to me.',
        objects=[dict(category='pear'), dict(category='paperback')]))
    assert {'apple', 'banana', 'hardback', 'notebook'} <= excluded
    assert {'fruit', 'paper'} <= set(families)
    excluded, _ = planner.excluded_categories(dict(instruction='Find the shampoo matching the sample.',
        objects=[dict(category='bottle_of_shampoo'), dict(category='bottle_of_lotion')]))
    assert 'soap_dispenser' in excluded


def test_finite_planner_avoids_real_plant_and_preserves_task_gap():
    task, records = example()
    plans = planner.propose(task, records, {'target'}, policy())
    assert plans and 1 <= len(plans[0]['props']) <= 2
    assert all(p['category'] not in ('notebook', 'hardback') for p in plans[0]['props'])
    for plan in plans:
        plan_props(records['table']['aabb'], plan['props'], [records['target']['aabb']], [records['plant']['aabb']])


def test_same_input_is_reproducible_but_variants_change_assets_or_spots():
    task, records = example()
    assert planner.propose(task, records, {'target'}, policy()) == planner.propose(task, records, {'target'}, policy())
    assert planner.propose(task, records, {'target'}, policy(), variant=1) != planner.propose(task, records, {'target'}, policy())


def test_light_default_adds_one_prop_and_reviewed_policy_can_allow_two():
    task, records = example()
    assert len(planner.propose(task, records, {'target'}, policy())[0]['props']) == 1
    assert len(planner.propose(task, records, {'target'}, dict(policy(), max_props=2))[0]['props']) == 2


def test_task_surface_or_unknown_room_or_jointed_support_never_accepted():
    task, records = example()
    for mode in ('protected', 'unknown_room', 'articulated'):
        candidate = json.loads(json.dumps(records))
        names = {'target', 'table'} if mode == 'protected' else {'target'}
        if mode == 'unknown_room': candidate['table']['rooms'] = ['unreviewed_space_0']
        if mode == 'articulated': candidate['table']['n_joints'] = 1
        with pytest.raises(ValueError, match='NO_SAFE'):
            planner.propose(task, candidate, names, policy())


def test_occupied_surface_is_not_called_an_enhancement():
    task, records = example()
    records['plant']['aabb'] = [[0, 0, .8], [2, 1, 1.3]]
    with pytest.raises(ValueError, match='NO_SAFE'):
        planner.propose(task, records, {'target'}, policy())


def test_unsafe_low_table_cannot_create_a_new_base_obstacle():
    task, records = example()
    records['table']['aabb'][1][2] = .4
    with pytest.raises(ValueError, match='NO_SAFE'):
        planner.propose(task, records, {'target'}, policy())


def test_queue_rejects_a_failed_reload_even_if_human_ready_file_exists(tmp_path):
    from enrich_background_queue import accepted_case
    parent = dict(id='base', parent_task_id='T001')
    (tmp_path/'validation.json').write_text(json.dumps(dict(status='initialization_validated', parent_task=parent)))
    (tmp_path/'restore_validation.json').write_text(json.dumps(dict(status='failed', parent_task=parent)))
    (tmp_path/'human_ready.json').write_text(json.dumps(dict(tasks=[dict(id='bad')])))
    assert accepted_case(tmp_path, parent, tmp_path) is None


def test_programming_errors_are_not_treated_as_bad_layouts(tmp_path):
    from enrich_background_queue import failure_kind
    (tmp_path/'validation.json').write_text(json.dumps(dict(error="'Robot' object has no attribute 'in_rooms'")))
    assert failure_kind(tmp_path, 'initialization', 1)[0] == 'programming_error'
    (tmp_path/'validation.json').write_text(json.dumps(dict(error='UNDEFINED_NAME', exception_type='NameError')))
    assert failure_kind(tmp_path, 'initialization', 1)[0] == 'programming_error'


def test_bad_native_surface_is_eligible_for_another_support(tmp_path):
    from enrich_background_queue import failure_kind
    (tmp_path/'validation.json').write_text(json.dumps(dict(error='BACKGROUND_HEIGHT_LIMIT', exception_type='ValueError')))
    assert failure_kind(tmp_path, 'initialization', 1)[0] == 'native_support'


def test_resume_needs_verified_frozen_creation_and_no_physical_reload_failure(tmp_path, monkeypatch):
    from enrich_background_queue import reusable_initial_case
    import spectator_cache
    parent = dict(id='base', parent_task_id='T001')
    report = dict(status='initialization_validated', parent_task=parent, task_content_preserved=True)
    (tmp_path/'validation.json').write_text(json.dumps(report))
    assert not reusable_initial_case(tmp_path, parent)
    (tmp_path/'exported_world_poses.private.json').write_text('{}')
    calls = []
    monkeypatch.setattr(spectator_cache, 'identity', lambda task: calls.append(task) or ('sha', {}))
    assert reusable_initial_case(tmp_path, parent) and len(calls) == 1
    (tmp_path/'restore_validation.json').write_text(json.dumps(dict(error='NATIVE_MOTION_NOT_CONVERGED', exception_type='ValueError')))
    assert not reusable_initial_case(tmp_path, parent)
    (tmp_path/'restore_validation.json').unlink()
    def broken(task):
        raise ValueError('DEPENDENCY_MISMATCH')
    monkeypatch.setattr(spectator_cache, 'identity', broken)
    with pytest.raises(ValueError, match='DEPENDENCY'):
        reusable_initial_case(tmp_path, parent)
