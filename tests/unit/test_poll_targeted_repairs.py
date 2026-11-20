"""Completion is tied to original-cohort native pairs, not queue exhaustion."""
import json
from pathlib import Path

import pytest

import poll_targeted_repairs as polling
import track_full_asset_queue as tracking


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_stale_case_binding_cannot_reserve_a_native_lane(tmp_path,monkeypatch):
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    save(tmp_path/'cases.json',{'U146':dict(record=dict(scenario='unrelated.yaml'))})
    action=dict(tag='immutable-job',cases='cases.json',tasks=['U086'])
    with pytest.raises(ValueError,match='ACTION_CASE_BINDING_MISMATCH.*U086'):
        polling.launch(action)
    save(tmp_path/'cases.json',{'U086':dict(record=dict(scenario='matched.yaml'))})
    polling.validate_action_cases(action)


def test_side_pose_recipe_requires_a_hash_pinned_supporting_schema(tmp_path):
    import hashlib
    recipe=tmp_path/'recipe.yaml';recipe.write_text('placements:\n- entity: tool\n  orientation_hint_xyzw: [0, 0, 0, 1]\n')
    runtime=tmp_path/'runtime'
    assert not polling.runtime_accepts_authoring_hints(recipe,runtime)
    schema=runtime/'src/rummagebench/core/scenario.py';schema.parent.mkdir(parents=True)
    schema.write_text('orientation_hint_xyzw: list\n')
    save(runtime/'release.json',dict(source_sha256={'src/rummagebench/core/scenario.py':hashlib.sha256(schema.read_bytes()).hexdigest()}))
    assert polling.runtime_accepts_authoring_hints(recipe,runtime)
    recipe.write_text('placements:\n- entity: tool\n  relation: on_top\n  position_xy: [1, 2]\n  orientation_hint_xyzw: [0, 0, 0, 1]\n')
    assert not polling.runtime_accepts_authoring_hints(recipe,runtime)
    schema.write_text('orientation_hint_xyzw: list\n# on_top orientation_hint_xyzw requires explicit XY\n')
    save(runtime/'release.json',dict(source_sha256={'src/rummagebench/core/scenario.py':hashlib.sha256(schema.read_bytes()).hexdigest()}))
    assert polling.runtime_accepts_authoring_hints(recipe,runtime)
    schema.write_text('changed code')
    assert not polling.runtime_accepts_authoring_hints(recipe,runtime)
    recipe.write_text('placements:\n- entity: tool\n')
    assert polling.runtime_accepts_authoring_hints(recipe,runtime)


def test_live_launch_snapshot_retains_checks_and_updates_verified_progress(tmp_path):
    save(tmp_path/'targeted_repair_launch.json',dict(verification={'selected_tests':388},all_task_progress={'passed':1}))
    save(tmp_path/'runtime_override.json',dict(runtime='frozen95',verified_by='native-proof.json',controls={'F035':True,'F090':True}))
    plan=dict(actions=[dict(tag='case',lane='lane1',runtime='frozen95')])
    states={'case':dict(stage='running',pid=123,tasks_run=['P007'])}
    cohort=dict(passed=100,total=100,remaining=[]);whole=dict(passed=999,total=1000,remaining=['P007'])
    polling.update_launch_snapshot(tmp_path,plan,states,cohort,whole)
    result=json.loads((tmp_path/'targeted_repair_launch.json').read_text())
    assert result['verification']=={'selected_tests':388} and result['all_tasks_progress']==whole
    assert 'all_task_progress' not in result and result['production_runtime']=='frozen95'
    assert result['lanes'][0]['active'][0]['pid']==123


def test_completion_keeps_legacy_workers_paused(monkeypatch,tmp_path):
    monkeypatch.setattr(polling,'ROOT',tmp_path);held=[]
    monkeypatch.setattr(polling,'hold_lane',lambda lane:held.append(lane))
    monkeypatch.setattr(polling,'release_lane',lambda lane:(_ for _ in ()).throw(AssertionError('Must not resume obsolete queues')))
    polling.complete_repair_campaign([dict(lane='lane1'),dict(lane='lane2'),dict(lane='lane1')])
    assert set(held)=={tmp_path/'lane1',tmp_path/'lane2'}


def test_catalogue_lag_does_not_duplicate_certified_cases_or_hide_stale_proofs(monkeypatch):
    monkeypatch.setattr(tracking,'reviewable_native_pair',lambda recipe,evidence:recipe.name=='valid.yaml')
    external={'done':dict(accepted=True,record=dict(scenario='valid.yaml'),evidence='proof'),
        'stale':dict(accepted=True,record=dict(scenario='changed.yaml'),evidence='proof'),
        'unchecked':dict(accepted=False,record=dict(scenario='valid.yaml'),evidence='proof')}
    assert polling.exclude_certified_external(['done','stale','unchecked','new'],external)==['stale','unchecked','new']


def test_cohort_requires_verified_pair_even_when_job_claims_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    lane=tmp_path/'runs/expert100_20261110'
    save(lane/'source_index.json',[{'id':f'F{i:03}'} for i in range(100)])
    save(lane/'jobs.json',{str(i):dict(task=f'F{i:03}',accepted=True,lane=f'job{i}',
        proposal=dict(record=dict(id=f'asset{i}',scenario=str(tmp_path/f'{i}.yaml')))) for i in [0,1,999]})
    calls=[]
    def check(recipe,evidence):
        calls.append((recipe,evidence));return recipe.name=='0.yaml'
    monkeypatch.setattr(tracking,'accepted_native_pair',check)
    monkeypatch.setattr(tracking,'known_authoring_issue',lambda _:None)
    result=polling.cohort_coverage()
    assert result['passed']==1 and result['total']==100
    assert 'F001' in result['remaining'] and 'F000' not in result['remaining']
    assert len(calls)==2  # Unrelated accepted jobs cannot inflate the cohort.
    monkeypatch.setattr(tracking,'known_authoring_issue',lambda _:'wrong support body')
    result=polling.cohort_coverage()
    assert result['native_passed']==1 and result['passed']==0
    assert result['known_authoring_issues']=={'F000':'wrong support body'}


def test_cohort_identity_cannot_be_shrunk_to_make_completion_easier(tmp_path,monkeypatch):
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    save(tmp_path/'runs/expert100_20261110/source_index.json',[{'id':'F001'}])
    with pytest.raises(ValueError,match='exactly 100'):polling.cohort_coverage()


def test_full_campaign_counts_parent_tasks_and_rejects_unreviewable_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    out=tmp_path/'runs/asset_full_queue_20261105'
    save(out/'queue.json',dict(tasks=[dict(id=f'F{i:04}') for i in range(1000)]))
    save(out/'human_ready.json',dict(tasks=[dict(parent_task_id=tid,scenario=recipe,play_scenario=str(tmp_path/'run/play.yaml'))
        for tid,recipe in [('F0000','good.yaml'),('F0000','duplicate.yaml'),('F0001','bad.yaml'),('other','good.yaml')]]))
    monkeypatch.setattr(tracking,'reviewable_native_pair',lambda recipe,evidence:recipe.name!='bad.yaml')
    result=polling.global_coverage()
    assert result['passed']==1 and result['total']==1000
    assert 'F0000' not in result['remaining'] and 'F0001' in result['remaining']


def test_global_tasks_are_scheduled_even_outside_fixed_cohort():
    cohort=dict(remaining=['F016']);whole=dict(remaining=['F016','U050'])
    assert polling.needed_tasks(dict(tasks=['U050'],scope='global_remaining'),cohort,whole)==['U050']
    assert polling.needed_tasks(dict(tasks=['U050']),cohort,whole)==[]
    assert polling.needed_tasks(dict(tasks=['F035'],scope='regression'),cohort,whole)==['F035']


def test_two_lanes_do_not_run_two_repairs_for_the_same_task_simultaneously():
    states={'a':dict(stage='running',tasks_run=['F169']),
            'old':dict(stage='finished',tasks_run=['U001'])}
    assert polling.task_running_elsewhere('b',['F169'],states)
    assert not polling.task_running_elsewhere('a',['F169'],states)
    assert not polling.task_running_elsewhere('b',['U001'],states)


def test_offline_preparation_deduplicates_completed_and_deferred_work_until_rules_change(tmp_path):
    save(tmp_path/'batch_preparation.json',{'F001':[dict(preparation_revision='r1',proposal=dict(status='deferred'))],
                                          'F002':[dict(preparation_revision='r1',proposal=dict(status='proposed'))]})
    assert polling.preparation_is_current(tmp_path,'batch',['F001','F002'],'r1')
    assert not polling.preparation_is_current(tmp_path,'batch',['F001','F002'],'r2')
    assert not polling.preparation_is_current(tmp_path,'batch',['F001','F003'],'r1')


def test_prepared_deferrals_skip_only_current_all_failed_offline_proposals(tmp_path,monkeypatch):
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    save(tmp_path/'lane/batch_preparation.json',{'F001':[dict(preparation_revision='r1',proposal=dict(status='deferred',reason='No safe changed layout'))],
        'F002':[dict(preparation_revision='r1',proposal=dict(status='proposed'))]})
    action=dict(lane='lane',tag='batch',prepare_repairs=True)
    assert polling.prepared_deferrals(action,['F001'],'r1')=={'F001':'No safe changed layout'}
    assert polling.prepared_deferrals(action,['F001'],'r2') is None
    assert polling.prepared_deferrals(action,['F001','F002'],'r1') is None
    assert polling.prepared_deferrals(action,['F003'],'r1') is None
    assert polling.prepared_deferrals(dict(action,prepare_repairs=False),['F001'],'r1') is None


def test_runtime_adoption_requires_reviewed_verified_candidate_and_keeps_running_jobs(tmp_path,monkeypatch):
    import expert_batch_repair
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    out=tmp_path/'campaign';source=out/'canary_results.json'
    save(source,dict(results={'F035':True,'F090':True,'probe':True}))
    save(out/'runtime_override.json',dict(runtime=str(tmp_path/'new'),source=str(source)))
    plan=dict(approved_runtime_candidates=['new'],reviewed_followups=dict(runtime='old'),actions=[
        dict(tag='pending',scope='global_remaining',prepare_repairs=True,runtime='old'),
        dict(tag='running',scope='global_remaining',prepare_repairs=True,runtime='old')])
    states={'running':dict(stage='running')}
    monkeypatch.setattr(expert_batch_repair,'synchronize_verified_runtime',lambda *args:False)
    assert not polling.adopt_reviewed_runtime(out/'plan.json',plan,states)
    assert plan['reviewed_followups']['runtime']=='old'
    monkeypatch.setattr(expert_batch_repair,'synchronize_verified_runtime',lambda *args:True)
    assert polling.adopt_reviewed_runtime(out/'plan.json',plan,states)
    assert plan['actions'][0]['runtime']=='new' and plan['actions'][1]['runtime']=='old'
    assert plan['actions'][0]['requires_success']=={'canary':['F035','F090','probe']}
    assert not polling.adopt_reviewed_runtime(out/'plan.json',plan,states)


def test_verified_secondary_lane_runtime_is_adopted_and_never_downgrades_to_ancestor(tmp_path,monkeypatch):
    import expert_batch_repair
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    primary=tmp_path/'campaign';secondary=tmp_path/'lane2';source=secondary/'probe_results.json'
    save(source,dict(results={'F035':True,'F090':True,'probe':True}))
    save(tmp_path/'new/release.json',dict(base='old'))
    save(secondary/'runtime_override.json',dict(runtime=str(tmp_path/'new'),source=str(source)))
    plan=dict(approved_runtime_candidates=['old','new'],reviewed_followups=dict(runtime='old',lanes=['campaign','lane2']),
        actions=[dict(tag='pending',scope='global_remaining',prepare_repairs=True,runtime='old')])
    def synchronize(lane,destination):
        assert lane==secondary
        save(destination/'runtime_override.json',json.loads((lane/'runtime_override.json').read_text()))
        return True
    monkeypatch.setattr(expert_batch_repair,'synchronize_verified_runtime',synchronize)
    assert polling.adopt_reviewed_runtime(primary/'plan.json',plan,{})
    assert plan['reviewed_followups']['runtime']=='new'
    save(secondary/'runtime_override.json',dict(runtime=str(tmp_path/'old'),source=str(source)))
    assert not polling.adopt_reviewed_runtime(primary/'plan.json',plan,{})
    assert plan['reviewed_followups']['runtime']=='new'


def test_finished_lanes_are_reconciled_before_priority_dispatch(tmp_path,monkeypatch):
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    actions=[dict(tag='priority',lane='lane'),dict(tag='earlier_batch',lane='lane')]
    states={'earlier_batch':dict(stage='running',pid=99999)}
    save(tmp_path/'lane/earlier_batch_status.json',dict(results={'F001':True}))
    held=[]
    monkeypatch.setattr(polling,'alive',lambda *args:False)
    monkeypatch.setattr(polling,'hold_lane',held.append)
    def no_child(*args):raise ChildProcessError()
    monkeypatch.setattr(polling.os,'waitpid',no_child)
    polling.refresh_running_actions(actions,states)
    assert states['earlier_batch']['stage']=='finished'
    assert states['earlier_batch']['native_result']['results']=={'F001':True}
    assert held==[tmp_path/'lane']


def test_reload_failure_is_not_replaced_by_an_older_initialization_failure(tmp_path,monkeypatch):
    import hashlib
    import os
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    monkeypatch.setattr(tracking,'accepted_native_pair',lambda *args:False)
    recipe=tmp_path/'recipe.yaml';recipe.write_text('same immutable recipe')
    digest=hashlib.sha256(recipe.read_bytes()).hexdigest()
    actions=[]
    for tag,stamp,status in [('reload',10,'initialization_validated'),('older',20,'failed')]:
        action=dict(tag=tag,lane='lane',tasks=['P001']);actions.append(action)
        save(tmp_path/'lane'/f'{tag}_results.json',dict(rows={'P001':dict(id='asset',scenario=str(recipe))}))
        report=tmp_path/'lane'/f'{tag}_P001/asset/validation.json'
        save(report,dict(scenario_sha256=digest,status=status,error=tag))
        os.utime(report,(stamp,stamp))
    restore=tmp_path/'lane/reload_P001/asset/restore_validation.json'
    save(restore,dict(status='failed',error='latest reload rejection'))
    os.utime(restore,(30,30))
    polling.reconcile(actions)
    case=json.loads((tmp_path/'runs/expert100_20261110/external_cases.json').read_text())['P001']
    assert case['error']=='latest reload rejection'
    assert case['evidence_updated']==30


def test_reviewed_followup_requires_control_proofs_and_never_requeues_same_input(tmp_path,monkeypatch):
    import hashlib
    import expert_batch_repair
    monkeypatch.setattr(polling,'ROOT',tmp_path)
    monkeypatch.setattr(expert_batch_repair,'preparation_revision',lambda:'reviewed-rules')
    monkeypatch.setattr(tracking,'accepted_native_pair',lambda *args:True)
    runtime=tmp_path/'runtime';runtime.mkdir();(runtime/'code.py').write_text('verified code')
    save(runtime/'release.json',dict(source_sha256={'code.py':hashlib.sha256((runtime/'code.py').read_bytes()).hexdigest()}))
    recipe=tmp_path/'recipe.yaml';recipe.write_text('immutable source')
    case=dict(id='F001',record=dict(scenario=str(recipe)),evidence=str(tmp_path/'failed'),error='native relation failure')
    save(tmp_path/'failed/validation.json',dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest()))
    out=tmp_path/'campaign';save(out/'external_cases.json',{})
    save(out/'original_failures.json',{'F001':case})
    plan=dict(actions=[dict(tag='controls',lane='lane1',tasks=['F035']),
        dict(tag='reviewed_followup_900',lane='lane1',tasks=['F002'])],reviewed_followups=dict(
        runtime='runtime',lanes=['lane1','lane2'],requires_success={'controls':['F035']},
        fallback_cases='campaign/original_failures.json'))
    whole=dict(remaining=['F001'])
    assert not polling.extend_reviewed_plan(out/'plan.json',plan,{},whole)
    save(tmp_path/'lane1/controls_results.json',dict(results={'F035':True},rows={'F035':dict(id='control',scenario=str(recipe))}))
    assert polling.extend_reviewed_plan(out/'plan.json',plan,{},whole)
    followup=plan['actions'][-1];assert followup['tasks']==['F001'] and followup['prepare_repairs']
    assert followup['tag']=='reviewed_followup_901'  # Sparse plans must not reuse immutable job IDs.
    assert followup['lane']=='lane2'  # Existing work must not strand the idle lane.
    # A completed attempt cannot trigger an identical retry on every poll.
    states={followup['tag']:dict(stage='finished')}
    assert not polling.extend_reviewed_plan(out/'plan.json',plan,states,whole)
    # A changing log filename alone also cannot create new native work.
    case['evidence']=str(tmp_path/'same_failure')
    save(tmp_path/'same_failure/validation.json',dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest()))
    save(out/'external_cases.json',{'F001':case})
    assert not polling.extend_reviewed_plan(out/'plan.json',plan,states,whole)
