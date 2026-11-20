"""Synthetic unit inputs only; no fake assets enter batch generation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import yaml
import pytest

from rummagebench.authoring.layout.indoor_batch import background_identity, summarize, make_plan
from rummagebench.authoring.layout.indoor_batch_probe import choose_probes


def test_background_dedup_ignores_seed_names_target_and_small_jitter():
    doc=dict(layout_seed=1,living_context=dict(native=dict(scene_json='/unit',inputs={'/unit':'unit-state'})),
        objects=[dict(placement=dict(name='one',category='bowl',model_id='unit',support_name='table',
            position=[1.,2.,.8],orientation_deg=0.))])
    other=deepcopy(doc);other['layout_seed']=900;other['target']='different';p=other['objects'][0]['placement']
    p['name']='renamed';p['position'][0]+=.001
    assert background_identity(doc)==background_identity(other)
    p['position'][0]+=.04
    assert background_identity(doc)!=background_identity(other)


def test_counts_do_not_promote_candidates_or_count_failures():
    records=[dict(id=f'x{i}',status='candidate_pending_simulation',scene='house',background_sha256=str(i),
                  target_category='spoon',per_surface=2) for i in range(3)]
    records.append(dict(id='bad',status='generation_failed',reason='constraint_failure'))
    s=summarize({'requested':2},records)
    assert s['generated']==s['unique_backgrounds']==2
    assert s['original_house_templates']==1 and s['reserve_candidates']==1
    assert s['evaluation_ready_episodes']==s['generated_structural_layouts']==0
    assert s['failures']=={'constraint_failure':1}


def test_probe_selection_is_house_stratified_not_success_filtered():
    rows=[dict(id='a0',scene='a'),dict(id='a1',scene='a'),dict(id='b0',scene='b')]
    assert [r['id'] for r in choose_probes(rows,['a0','a1','b0'],2)]==['a0','b0']


def test_unresolved_native_cabinets_stay_in_plan_and_target_does_not_drive_background(tmp_path,monkeypatch):
    import rummagebench.authoring.layout.indoor_batch as batch
    repo=Path(__file__).resolve().parents[2]
    native=dict(retained_native_objects=['left','right','table'],inputs={},objects=[
        dict(name=n,rooms=['kitchen_0'],category='cabinet',openable=True,support=False)
        for n in ['left','right']])
    native['objects'].append(dict(name='table',category='table',rooms=['kitchen_0'],openable=False,support=True))
    monkeypatch.setattr(batch,'native_context',lambda *a:deepcopy(native))
    monkeypatch.setattr(batch,'native_start',lambda *a:([0.,0.],{}))
    monkeypatch.setattr(batch,'asset_record',lambda *a,**k:dict(sources={}))
    slot=SimpleNamespace(parent_entity='left',is_container=True,
        to_dict=lambda:dict(slot_id='left/drawer',parent_entity='left',link='link_1'))
    monkeypatch.setattr(batch,'build_scene_slots',lambda *a:SimpleNamespace(slots=[slot]))
    config=dict(scenes=['unit'],room='kitchen_0',per_surface=[1,2,3],solver_steps=100,
        daily_pools=[[dict(category='bowl',model='unit')]],
        targets=[dict(category='spoon',model='one'),dict(category='fork',model='two')])
    path=tmp_path/'batch.yaml';path.write_text(yaml.safe_dump(config))
    first=make_plan(path,tmp_path,repo,8,31,8)
    assert first['homes'][0]['unresolved_native_bodies']==['right']
    assert {p['config']['smoke_container'] for p in first['proposals']}=={'left','right'}
    config['targets'].reverse();path.write_text(yaml.safe_dump(config))
    second=make_plan(path,tmp_path,repo,8,31,8)
    for a,b in zip(first['proposals'],second['proposals']):
        assert a['seed']==b['seed']
        assert a['config']['daily_assets']==b['config']['daily_assets']
        assert a['config']['per_surface']==b['config']['per_surface']
    with pytest.raises(ValueError,match='BOUNDED_BATCH_SIZE'):make_plan(path,tmp_path,repo,10,31,8)


@pytest.mark.parametrize('code,result',[
    (0,{}), (0,{'status':'running'}),
    (139,{'status':'execution_smoke_passed','scenario_sha256':'expected'}),
    (0,{'status':'execution_smoke_passed','scenario_sha256':'other'}),
])
def test_child_exit_or_claim_alone_cannot_pass(code,result):
    from rummagebench.authoring.layout.indoor_batch_probe import classify_process
    assert classify_process(result,code,None,{'scenario_sha256':'expected'})['status']=='execution_smoke_incomplete'


def test_archived_code_is_read_only_completion_fallback(tmp_path):
    from rummagebench.authoring.layout.indoor_batch import _check_inputs,archive_sources
    from rummagebench.authoring.tasks.plan import sha256_path
    source=tmp_path/'generator.py';source.write_text('# original unit source')
    asset=tmp_path/'unit_metadata.json';asset.write_text('{}')
    inputs={str(source):sha256_path(source),str(asset):sha256_path(asset)}
    out=tmp_path/'batch';out.mkdir()
    archive_sources({'inputs':inputs},out,tmp_path)
    source.write_text('# updated unit source')
    with pytest.raises(ValueError,match='BATCH_INPUT_CHANGED'):_check_inputs(inputs)
    _check_inputs(inputs,completed_root=out)
    asset.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='BATCH_INPUT_CHANGED'):_check_inputs(inputs,completed_root=out)


def test_completed_resume_validates_without_starting_any_worker(tmp_path, monkeypatch):
    import json
    import rummagebench.authoring.layout.indoor_batch as batch
    args = SimpleNamespace(out=tmp_path, count=1, seed=31, config=tmp_path/'config.yaml',
                           upstream=tmp_path/'upstream', python=str(tmp_path/'python'), max_proposals=2)
    proposal = dict(id='one')
    plan = dict(inputs={}, requested=1, seed=31, config=str(args.config), assets=str(tmp_path),
                runtime=dict(upstream=str(args.upstream), python=args.python), proposals=[proposal, dict(id='reserve')])
    row = dict(id='one', status='candidate_pending_simulation', scene='unit',
               background_sha256='unit', target_category='spoon', per_surface=1,
               proposal_sha256=batch.sha256_bytes(batch.canonical_json(proposal)),
               assembly='unit-assembly', assembly_sha256='unit-assembly-hash',
               scenario='unit-scenario', scenario_sha256='unit-scenario-hash')
    for name, value in [('plan', plan), ('records', [row]), ('summary', batch.summarize(plan, [row]))]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    monkeypatch.setattr(batch, 'load_manifest', lambda p: dict(manifest_sha256='unit-assembly-hash'))
    monkeypatch.setattr(batch, 'background_identity', lambda m: 'unit')
    monkeypatch.setattr(batch, 'sha256_path', lambda p: 'unit-scenario-hash')
    def forbidden(*a, **k):
        pytest.fail('completed resume must not start generation workers')
    monkeypatch.setattr(batch, 'ThreadPoolExecutor', forbidden)
    assert batch.generate_batch(args, tmp_path, tmp_path) == 0
    row['scenario_sha256'] = 'changed'
    (tmp_path/'records.json').write_text(json.dumps([row]))
    with pytest.raises(ValueError, match='BATCH_COMPLETED_OUTPUT_CHANGED'):
        batch.generate_batch(args, tmp_path, tmp_path)


def test_batch_probe_defaults_to_native_realization_without_overwriting_baseline(tmp_path,monkeypatch):
    import json
    import rummagebench.authoring.layout.indoor_batch_probe as probe
    repo=Path(__file__).resolve().parents[2]
    scenario=tmp_path/'unit.yaml';scenario.write_text('unit-only')
    row=dict(id='one',scene='unit',assembly='unit-only',assembly_sha256='unit',scenario=str(scenario),
             scenario_sha256=probe.sha256_path(scenario))
    (tmp_path/'one').mkdir();(tmp_path/'one/gpu').mkdir()
    (tmp_path/'one/gpu/original').write_text('baseline remains')
    (tmp_path/'summary.json').write_text(json.dumps(dict(selected_ids=['one'],generated=1)))
    (tmp_path/'records.json').write_text(json.dumps([row]))
    monkeypatch.setattr(probe,'load_manifest',lambda *a:dict(manifest_sha256='unit'))
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        folder=Path(command[command.index('--out')+1])
        (folder/'result.json').write_text(json.dumps(dict(status='execution_smoke_failed',error='unit failure')))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(probe.subprocess,'run',run)
    probe.probe_batch(SimpleNamespace(batch=tmp_path,limit=1,gpus=[4],sim_python='unit-only'),repo)
    assert '--realize-background' in commands[0]
    assert (tmp_path/'one/gpu/original').read_text()=='baseline remains'
    summaries=list(tmp_path.glob('gpu_summary_*.json'))
    assert len(summaries)==1
    summary=json.loads(summaries[0].read_text())
    assert summary['validation']['profile']=='observed-motion-v3'
    assert '--environment-only' in commands[0]
    assert summary['validation']['environment_only'] is True
    assert summary['records'][0]['validation']==summary['validation']
    assert not (tmp_path/'gpu_summary.json').exists()
