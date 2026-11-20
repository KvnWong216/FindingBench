from pathlib import Path
import sys,json,hashlib
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/v2'))
import expert_batch_repair as b


def test_preparation_cache_follows_rules_and_sampling_policy_not_evidence_timestamps(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'ROOT',tmp_path)
    source=tmp_path/'scripts/v2/expert_storage_repair.py';source.parent.mkdir(parents=True);source.write_text('first rule')
    profile=tmp_path/'build/verified_surface_sites/index.json';profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(dict(assets={'shelf/native':dict(evidence='old')})))
    first=b.preparation_revision()
    profile.write_text(json.dumps(dict(assets={'shelf/native':dict(evidence='new',updated=123)})))
    assert b.preparation_revision()==first
    profile.write_text(json.dumps(dict(assets={'shelf/native':dict(native_sampling_evidence={'recipe':'proof'})})))
    learned=b.preparation_revision();assert learned!=first
    source.write_text('corrected rule');assert b.preparation_revision()!=learned


def test_staged_expert_candidate_keeps_hash_and_task_identity(tmp_path):
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path/'run';q.cat=tmp_path/'catalog';q.runtime=tmp_path/'runtime'
    source=tmp_path/'authored.yaml';source.write_text('an immutable expert candidate')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    row=dict(id='instance',parent_task_id='task',scenario=str(source),scenario_sha256=sha)
    proposal=dict(status='proposed',record=row)
    result=q.staged_expert_candidate(proposal,'candidate_'+sha[:12],dict(id='task'))
    assert result['status']=='proposed' and result['record']['scenario_sha256']==sha
    assert Path(result['record']['scenario']).read_bytes()==source.read_bytes()
    with pytest.raises(ValueError,match='TASK_MISMATCH'):
        q.staged_expert_candidate(proposal,'wrong_task',dict(id='other'))
    source.write_text('changed after expert screening')
    with pytest.raises(ValueError,match='RECIPE_CHANGED'):
        q.staged_expert_candidate(proposal,'mutated',dict(id='task'))


def test_case_diagnosis_keeps_native_blocker_and_conservative_policy(tmp_path):
    (tmp_path/'native_validation.log').write_text(
        '[placement] attempt=1 start salt inside cabinet link=drawer0\n'
        '[expert-layout-plan] '+json.dumps(dict(status='infeasible_model',reason='No allowed bin/orientation fits an item'))+'\n')
    d=b.diagnose_case(dict(error='EXPERT_LAYOUT_UNRESOLVED',evidence=str(tmp_path)),2)
    assert d['category']=='restricted_packing' and d['first_blocker']['receptacle']=='cabinet'
    assert d['native_rejections_in_current_queue']==2
    assert not b.intervention_for(d,'Find salt in the upper cupboard.')['clear_unstated_links']
    assert b.intervention_for(d,'Find salt in the cupboards.')['clear_unstated_links']
    assert b.intervention_for(d,'There is a sample on the cabinet top. Find the cookie box matching it among the kitchen groceries and bring it to the cabinet top.')['clear_unstated_links']


def test_cohort_excludes_accepted_and_hash_mismatched_recipes(tmp_path,monkeypatch):
    primary=tmp_path/'runs/overnight_expert_20261109';primary.mkdir(parents=True)
    ready=tmp_path/'runs/asset_full_queue_20261105/human_ready.json';ready.parent.mkdir(parents=True)
    ready.write_text(json.dumps(dict(tasks=[dict(parent_task_id='accepted')])))
    cases=[]
    for tid in ['accepted','stubborn','mismatch']:
        scenario=tmp_path/f'{tid}.yaml';scenario.write_text(tid)
        evidence=tmp_path/tid;evidence.mkdir()
        digest=hashlib.sha256(scenario.read_bytes()).hexdigest()
        (evidence/'validation.json').write_text(json.dumps(dict(scenario_sha256='wrong' if tid=='mismatch' else digest)))
        cases.append(dict(id=tid,error='EXPERT_SUPPORT_UNRESOLVED',evidence=str(evidence),
            record=dict(scenario=str(scenario),scene='house',instruction='Find salt.')))
    (primary/'source_index.json').write_text(json.dumps(cases))
    monkeypatch.setattr(b,'ROOT',tmp_path);monkeypatch.setattr(b,'PRIMARY',primary)
    selected,excluded=b.select_cohort(1)
    assert [c['id'] for c in selected]==['stubborn']
    assert [c['id'] for c in excluded]==['mismatch']


def test_replay_is_skipped_only_when_native_sources_and_settings_match(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir();source=runtime/'backend.py';source.write_text('correct geometry')
    (runtime/'release.json').write_text(json.dumps(dict(enable={'GPU':'1'})))
    evidence=tmp_path/'evidence';evidence.mkdir();report=dict(runtime_sources={'backend.py':hashlib.sha256(source.read_bytes()).hexdigest()},runtime_environment={'GPU':'1'})
    (evidence/'validation.json').write_text(json.dumps(report));case={'evidence':str(evidence)}
    assert b.native_runtime_unchanged(case,runtime)
    source.write_text('new actual geometry')
    assert not b.native_runtime_unchanged(case,runtime)


def test_authoring_revision_ignores_evidence_paths_and_detects_new_geometry(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'ROOT',tmp_path);out=tmp_path/'run';out.mkdir()
    path=tmp_path/'build/native_support_geometry/index.json';path.parent.mkdir(parents=True)
    profile=dict(assets={'cabinet/native':dict(components={'floor':dict(status='measured_plane')},evidence='old')})
    path.write_text(json.dumps(profile));initial=b.authoring_revision(out)
    profile['assets']['cabinet/native']['evidence']='new path, same measured geometry';path.write_text(json.dumps(profile))
    assert b.authoring_revision(out)==initial
    profile['assets']['cabinet/native']['components']['floor']['status']='unknown';path.write_text(json.dumps(profile))
    assert b.authoring_revision(out)!=initial


def test_other_failure_queue_covers_pending_ids_and_uses_hash_matched_fallback(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'ROOT',tmp_path);primary=tmp_path/'primary';primary.mkdir()
    monkeypatch.setattr(b,'PRIMARY',primary);out=tmp_path/'expert';out.mkdir()
    queue=tmp_path/'runs/asset_full_queue_20261105/queue.json';queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps(dict(tasks=[dict(id=t,status='ready_for_human_review' if t=='ready' else 'awaiting_physical_repair')
        for t in ['ready','cohort','other','unbound']])))
    recipe=tmp_path/'recipe.yaml';recipe.write_text('native recipe');evidence=tmp_path/'native';evidence.mkdir()
    (evidence/'validation.json').write_text(json.dumps(dict(scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())))
    initial=dict(id='other',record=dict(scenario=str(recipe),instruction='Find salt.',scene='house'),evidence=str(evidence),error='EXPERT_SUPPORT_UNRESOLVED')
    (primary/'source_index.json').write_text(json.dumps([initial]))
    (out/'latest_cases.json').write_text(json.dumps(dict(other=dict(initial,record=dict(initial['record'],scenario='missing.yaml')))))
    cases,plan=b.remaining_failure_cases(out,[{'id':'cohort'}])
    assert [c['id'] for c in cases]==['other']
    assert cases[0]['record']['scenario']==str(recipe)
    assert plan['total_remaining']==3 and plan['cohort_remaining']==1 and plan['other_remaining']==2
    assert plan['unbound']==[dict(id='unbound',reason='No source recipe matching an observed native report; requires expert rebinding')]


def test_runtime_synchronization_requires_current_probe_and_both_native_controls(monkeypatch,tmp_path):
    out=tmp_path/'expert';out.mkdir();primary=tmp_path/'primary';runtime=tmp_path/'runtime';runtime.mkdir()
    code=runtime/'backend.py';code.write_text('verified source');digest=hashlib.sha256(code.read_bytes()).hexdigest()
    rows={}
    for tid in ['F035','F090','probe']:
        recipe=tmp_path/'scenarios'/('lane_'+tid)/tid/'scenario.yaml';recipe.parent.mkdir(parents=True);recipe.write_text(tid)
        directory=out/('lane_'+tid)/tid;directory.mkdir(parents=True)
        (directory/'validation.json').write_text(json.dumps(dict(runtime_sources={'backend.py':digest})))
        rows[tid]=dict(id=tid,scenario=str(recipe))
    report=dict(rows=rows,results={t:True for t in rows})
    (out/'results.json').write_text(json.dumps(report))
    (out/'runtime_override.json').write_text(json.dumps(dict(runtime=str(runtime),source='results.json')))
    checked=[]
    monkeypatch.setattr(b,'accepted_native_pair',lambda recipe,directory:checked.append(Path(directory).name) or True)
    assert b.synchronize_verified_runtime(out,primary)
    promoted=json.loads((primary/'runtime_override.json').read_text())
    assert Path(promoted['source'])==(out/'results.json').resolve()
    assert checked==['F035','F090','probe']
    (runtime/'release.json').write_text(json.dumps(dict(enable={'PROPOSAL_MODE':'1'})))
    assert not b.synchronize_verified_runtime(out,primary)
    for tid in rows:
        validation=out/('lane_'+tid)/tid/'validation.json'
        validation.write_text(json.dumps(dict(runtime_sources={'backend.py':digest},runtime_environment={'PROPOSAL_MODE':'1'})))
    assert b.synchronize_verified_runtime(out,primary)
    before=(primary/'runtime_override.json').read_bytes()
    code.write_text('unverified edit')
    assert not b.synchronize_verified_runtime(out,primary)
    assert (primary/'runtime_override.json').read_bytes()==before
    code.write_text('verified source');report['results']['probe']=False;(out/'results.json').write_text(json.dumps(report))
    assert not b.synchronize_verified_runtime(out,primary)


def test_other_failure_repair_is_once_per_revision_and_uses_verified_input(tmp_path):
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path;q.ledger={};q.latest={'other':{'unbound':'stale'}}
    q.runtime=tmp_path/'runtime';q.accepted=lambda tid:False;q.status=lambda **kw:None
    calls=[]
    def prepare(case,lane,action,variant=0):
        if action=='direct':return dict(status='deferred',reason='Fixture needs a layout change')
        calls.append((case,lane,action));return dict(status='proposed',record=case['record'])
    q.prepare=prepare;q.validate=lambda *args:dict(accepted=False,exit_code=1)
    case=dict(id='other',record={'scenario':'verified.yaml'},expert_policy={},error='native failure',evidence=str(tmp_path/'observed'))
    q.run_other_failure(case,'rev1');q.run_other_failure(case,'rev1')
    assert len(calls)==1 and calls[0][0]['record']==case['record'] and calls[0][2]=='expert'
    assert q.ledger['full_queue_rev1:other']['scope']=='other_failed_task'
    q.run_other_failure(case,'rev2');assert len(calls)==2


def test_expert_priority_counts_native_attempts_and_can_change_without_restart(tmp_path):
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path;q.stop_requested=False;calls=[]
    q.run_other_failure=lambda case,revision:calls.append(case['id'])
    config=tmp_path/'scheduling.json';config.write_text(json.dumps(dict(cohort_native_attempts_per_other=4)))
    background=iter([{'id':'other1'},{'id':'other2'}])
    q.advance_other(background,'rev',initial=True)
    for _ in range(10):q.advance_other(background,'rev',native_used=False)
    for _ in range(3):q.advance_other(background,'rev',native_used=True)
    assert not calls
    q.advance_other(background,'rev',native_used=True);assert calls==['other1']
    config.write_text(json.dumps(dict(cohort_native_attempts_per_other=1)))
    q.advance_other(background,'rev',native_used=True);assert calls==['other1','other2']


def test_external_probe_observation_must_match_recipe_and_be_newer(tmp_path):
    import os
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path;q.latest={}
    source=tmp_path/'scenario.yaml';source.write_text('revised target and reference')
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    for directory in [old,new]:(directory/'validation.json').write_text(json.dumps(dict(scenario_sha256=digest)))
    os.utime(old/'validation.json',ns=(1000000,1000000));os.utime(new/'validation.json',ns=(2000000,2000000))
    before=dict(id='task',record={'scenario':str(source)},evidence=str(old));q.latest={'task':before}
    after=dict(before,evidence=str(new),error='new physical blocker',source='native_probe')
    (tmp_path/'external_cases.json').write_text(json.dumps({'task':after}))
    q.refresh_external_cases();assert q.latest['task']==after
    (tmp_path/'external_cases.json').write_text(json.dumps({'task':before}));q.refresh_external_cases()
    assert q.latest['task']==after
    source.write_text('unbound mutation');q.latest={'task':before}
    (tmp_path/'external_cases.json').write_text(json.dumps({'task':after}));q.refresh_external_cases()
    assert q.latest['task']==before


def test_two_native_shards_partition_all_tasks_without_duplicates():
    ids=[f'F{i:03d}' for i in range(350)]+[f'P{i:03d}' for i in range(350)]+[f'U{i:03d}' for i in range(300)]
    left={tid for tid in ids if b.shard_owns_task(tid,0,2)}
    right={tid for tid in ids if b.shard_owns_task(tid,1,2)}
    assert not left&right and left|right==set(ids)
    assert 400<len(left)<600 and 400<len(right)<600
    assert all(b.shard_owns_task(tid,0,1) for tid in ids)
    with pytest.raises(ValueError,match='INVALID_TASK_SHARD'):b.shard_owns_task('F001',2,2)


def test_unowned_task_cannot_launch_other_failure_native_job(tmp_path):
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path;q.ledger={}
    q.owns_task=lambda tid:False
    q.accepted=lambda tid:pytest.fail('Unowned task must be skipped before acceptance checks or native launch')
    q.run_other_failure(dict(id='F001'),'revision')
    assert q.ledger=={}


def test_cohort_only_never_consumes_or_launches_background_jobs(tmp_path):
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path;q.stop_requested=False;q.ledger={}
    (tmp_path/'scheduling.json').write_text(json.dumps(dict(cohort_only=True)))
    def forbidden():
        pytest.fail('Cohort-only mode consumed a background task')
        yield {}
    q.advance_other(forbidden(), 'revision',native_used=True,initial=True)
    q.owns_task=lambda tid:pytest.fail('Cohort-only mode reached background ownership checks')
    q.run_other_failure(dict(id='U001'),'revision')
    assert q.ledger=={}


def test_two_identical_physical_failures_require_changed_intervention(tmp_path,monkeypatch):
    q=b.ExpertBatch.__new__(b.ExpertBatch);q.out=tmp_path;q.runtime=tmp_path/'runtime';q.ledger={}
    monkeypatch.setattr(b,'ROOT',tmp_path)
    monkeypatch.setattr(b,'native_runtime_unchanged',lambda case,runtime:True)
    row=dict(id='instance',scenario_sha256='same')
    proposal=dict(status='proposed',record=row)
    for index in range(2):
        lane=f'phase_{index}';directory=tmp_path/lane/'instance';directory.mkdir(parents=True)
        (directory/'validation.json').write_text(json.dumps(dict(status='failed',scenario_sha256='same',error='POST_SETTLE_RELATION_FAILED: bottle')))
        q.ledger[lane]=dict(lane=lane,accepted=False,proposal=proposal)
    blocked=q.bound_identical_failures(proposal)
    assert blocked['status']=='deferred' and len(blocked['evidence'])==2
    changed=dict(status='proposed',record=dict(row,scenario_sha256='repaired'))
    assert q.bound_identical_failures(changed)==changed
    monkeypatch.setattr(b,'native_runtime_unchanged',lambda case,runtime:False)
    assert q.bound_identical_failures(proposal)==proposal
