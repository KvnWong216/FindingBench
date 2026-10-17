"""Synthetic receipts only; these fixtures are never generated data."""
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def production(tmp_path, monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    spec=importlib.util.spec_from_file_location('unit_production_report',repo/'scripts/tasks/report_behavior_production.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    batch=tmp_path/'batch';folder=batch/'home_0000'/'gpu_realized_unit';folder.mkdir(parents=True)
    def write(path, value):path.write_text(json.dumps(value))
    write(batch/'summary.json',dict(generated=1,attempted=1,selected_ids=['home_0000']))
    write(batch/'records.json',[dict(id='home_0000',scenario_sha256='scenario',assembly_sha256='assembly')])
    write(folder/'result.json',dict(status='execution_smoke_passed',verify_background_restore=True))
    (folder/'restorable_background.pt').write_bytes(b'explicit synthetic checkpoint fixture')
    for name in ('background_realization','background_after_target','background_restore'):
        write(folder/f'{name}.json',dict(status='restore_verified_pending_family' if name=='background_restore' else 'passed',
            checkpoint_sha256=cli.sha256_path(folder/'restorable_background.pt')))
    receipt=dict(status='execution_smoke_passed',returncode=0,scenario_sha256='scenario',assembly_sha256='assembly',
        validation=dict(restore=True,profile='observed-motion-v3',mode='native-realization'),
        result_sha256=cli.sha256_path(folder/'result.json'),stage_sha256={
            name:cli.sha256_path(folder/f'{name}.json') for name in
            ('background_realization','background_after_target','background_restore')})
    write(folder/'process.json',receipt)
    return cli,tmp_path,folder,receipt,write


def test_bound_execution_is_counted_without_search_promotion(production):
    cli,root,folder,receipt,write=production
    result=cli.report(root)
    assert result['execution_passed']==1 and result['pending_gpu']==0
    assert result['evaluation_ready_episodes']==0 and result['generated_new_floorplans']==0
    receipt.pop('validation');write(folder/'process.json',receipt)
    result=cli.report(root)
    assert result['execution_passed']==0 and result['pending_gpu']==1


def test_skill_rejection_keeps_successful_background_evidence(production):
    cli,root,folder,receipt,write=production
    write(folder/'result.json',dict(status='execution_smoke_failed',verify_background_restore=True))
    receipt.update(status='execution_smoke_failed',returncode=1,
        error='EXIT_1_OR_MISSING_FINAL_RECEIPT',failures={'BASE_COLLISION':13,'OPEN_OUT_OF_CAPABILITY':12},
        result_sha256=cli.sha256_path(folder/'result.json'))
    write(folder/'process.json',receipt)
    result=cli.report(root)
    assert result['execution_failed']==1 and result['execution_passed']==0
    assert result['validation_failures']=={'PUBLIC_SKILL_CHAIN_REJECTED':1}
    assert result['background_stage_passes']['background_restore']==1
    assert result['skill_attempt_failures']['OPEN_OUT_OF_CAPABILITY']==12


def test_new_acceptance_does_not_relabel_old_success(production):
    cli,root,folder,receipt,write=production
    write(root/'production_status.json',dict(active_acceptance_policy='execution-v1'))
    result=cli.report(root)
    assert result['execution_passed']==0 and result['pending_gpu']==1


@pytest.fixture
def environment(production):
    cli,root,folder,receipt,write=production
    write(root/'production_status.json',dict(environment_only=True))
    for name in ('environment_start.pt','environment_start.png'):
        (folder/name).write_bytes(b'unit-only artifact, not a real environment')
    write(folder/'environment_ready.json',dict(status='environment_ready',assembly_sha256='assembly',
        checkpoint_sha256=cli.sha256_path(folder/'environment_start.pt'),
        image_sha256=cli.sha256_path(folder/'environment_start.png'),
        public_reset=True,spawn_clear=True,target_relation=True,policy_executed=False))
    for stage,status in [('background_realization','support_and_stability_passed_pending_family'),
                         ('background_after_target','daily_props_physically_ready'),
                         ('background_restore','restore_verified_pending_family')]:
        write(folder/f'{stage}.json',dict(status=status,assembly_sha256='assembly',
            checkpoint_sha256=cli.sha256_path(folder/'restorable_background.pt')))
    result=dict(status='environment_ready',scenario_sha256='scenario',assembly_sha256='assembly',
        runner_sha256='unit-runner',verify_background_restore=True,environment_only=True,attempts=[])
    write(folder/'result.json',result)
    receipt.update(status='environment_ready',result_sha256=cli.sha256_path(folder/'result.json'))
    receipt['validation'].update(environment_only=True,sources={'scripts/tasks/behavior_pipeline.py':'unit-runner'})
    receipt['stage_sha256']={name:cli.sha256_path(folder/f'{name}.json') for name in
        ('background_realization','background_after_target','background_restore','environment_ready')}
    write(folder/'process.json',receipt)
    return cli,root,folder,receipt,write,result


def test_environment_ready_without_any_policy_actions(environment):
    cli,root,folder,receipt,write,result=environment
    summary=cli.report(root)
    assert summary['environment_ready']==1
    assert summary['execution_passed']==summary['execution_failed']==0
    assert summary['evaluation_ready_episodes']==0
    # A historical policy run sharing the candidates must not absorb new
    # environment receipts (and vice versa).
    write(root/'production_status.json',dict(environment_only=False))
    summary=cli.report(root)
    assert summary['environment_ready']==0 and summary['pending_gpu']==1


@pytest.mark.parametrize('defect',['crash','policy_ran','spawn_unchecked','missing_stage','changed_checkpoint','changed_image'])
def test_environment_readiness_requires_bound_physical_evidence(environment,defect):
    cli,root,folder,receipt,write,result=environment
    if defect=='crash':receipt['returncode']=139
    if defect=='policy_ran':
        result['attempts']=[{'trace':['unit action']}]
        write(folder/'result.json',result);receipt['result_sha256']=cli.sha256_path(folder/'result.json')
    if defect=='spawn_unchecked':
        evidence=json.loads((folder/'environment_ready.json').read_text());evidence['spawn_clear']=False
        write(folder/'environment_ready.json',evidence)
        receipt['stage_sha256']['environment_ready']=cli.sha256_path(folder/'environment_ready.json')
    if defect=='missing_stage':receipt['stage_sha256'].pop('background_after_target')
    if defect=='changed_checkpoint':(folder/'environment_start.pt').write_bytes(b'tampered')
    if defect=='changed_image':(folder/'environment_start.png').write_bytes(b'tampered')
    write(folder/'process.json',receipt)
    if defect.startswith('changed_'):
        with pytest.raises(ValueError,match='PRODUCTION_ENVIRONMENT_ARTIFACT_CHANGED'):cli.report(root)
    else:
        summary=cli.report(root)
        assert summary['environment_ready']==0 and summary['environment_needs_review']==1


@pytest.mark.parametrize('failure',['changed_stage','missing_restore_binding','wrong_profile','result_failed','changed_checkpoint'])
def test_unverified_success_rejects(production,failure):
    cli,root,folder,receipt,write=production
    if failure=='changed_stage':write(folder/'background_restore.json',dict(status='failed'))
    if failure=='missing_restore_binding':receipt['stage_sha256'].pop('background_restore')
    if failure=='wrong_profile':receipt['validation']['profile']='legacy'
    if failure=='changed_checkpoint':(folder/'restorable_background.pt').write_bytes(b'changed fixture')
    if failure=='result_failed':
        write(folder/'result.json',dict(status='execution_smoke_failed',verify_background_restore=True))
        receipt['result_sha256']=cli.sha256_path(folder/'result.json')
    write(folder/'process.json',receipt)
    with pytest.raises(ValueError,match='PRODUCTION_'):cli.report(root)
