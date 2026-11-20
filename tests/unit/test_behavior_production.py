"""Process orchestration fixtures only. No generated-data or GPU claims."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import yaml


@pytest.mark.parametrize('passes',[True,False])
def test_production_routes_all_candidates_through_geometry_restore_without_policy(tmp_path,monkeypatch,passes):
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    spec=importlib.util.spec_from_file_location('unit_production',repo/'scripts/tasks/produce_behavior.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(dict(scenes=['unit'],room='kitchen_0',
        daily_pools=[[dict(category='bowl',model='unit')]],targets=[],per_surface=[1],solver_steps=1)))
    monkeypatch.setattr(cli,'host_config',lambda:dict(behavior_assets_root=tmp_path/'unit_assets'))
    monkeypatch.setattr(cli,'make_plan',lambda *a:dict(homes=[dict(scene='unit')],rejected_homes=[]))
    monkeypatch.setattr(cli,'load_geometry',lambda *a,**k:{})
    out=tmp_path/'out';commands=[]
    def run(command,**kw):
        commands.append(command)
        if 'export-geometry' in command:
            output=Path(command[command.index('--out')+1]);output.mkdir(parents=True)
            config=Path(command[command.index('--config')+1])
            (output/'process.json').write_text(json.dumps(dict(status='geometry_exported_pending_physics',
                returncode=0,config_sha256=cli.sha256_path(config))))
        else:
            assert '--verify-background-restore' in command and 'observed-motion-v3' in command
            batch=Path(command[command.index('--batch')+1])
            status='environment_ready' if passes else 'environment_failed'
            (batch/'gpu_summary_unit.json').write_text(json.dumps(dict(probed=1,status_counts={status:1},
                validation={'acceptance_policy':'strict-v3','environment_only':True},
                records=[dict(status=status,error=None if passes else 'RESTORE_FAILED')],evaluation_ready_episodes=0)))
        return NS(returncode=0)
    monkeypatch.setattr(cli.subprocess,'run',run)
    def popen(command,**kw):
        commands.append(command);assert kw['start_new_session']
        batch=Path(command[command.index('--out')+1]);batch.mkdir()
        (batch/'summary.json').write_text(json.dumps(dict(generated=1,attempted=1,failures={})))
        return NS(poll=lambda:0,returncode=0)
    monkeypatch.setattr(cli.subprocess,'Popen',popen)
    args=NS(out=out,gpus=[4,5],config=cfg,count=1,max_proposals=1,seed=31,workers=1,
            sim_python='unit-sim',cpu_python='unit-cpu')
    assert cli.run(args)==(0 if passes else 2)
    receipt=json.loads((out/'production_status.json').read_text())
    assert receipt['status']==('completed' if passes else 'completed_with_rejections')
    assert receipt['evaluation_ready_episodes']==0 and receipt['environment_ready']==int(passes)
    assert yaml.safe_load((out/'bound_config.yaml').read_text())['require_geometry']
    args.seed=32
    with pytest.raises(ValueError,match='RESUME_INPUT_CONFLICT'):cli.run(args)
    assert not args._may_publish_failure


def test_rejected_second_coordinator_cannot_overwrite_running_status(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    spec=importlib.util.spec_from_file_location('unit_locked_production',repo/'scripts/tasks/produce_behavior.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    original='{"status":"generating_and_validating"}'
    (tmp_path/'production_status.json').write_text(original)
    def rejected(args):
        args._may_publish_failure=False
        raise BlockingIOError('already owned')
    monkeypatch.setattr(cli,'run',rejected)
    with pytest.raises(BlockingIOError):cli.main(['--out',str(tmp_path)])
    assert (tmp_path/'production_status.json').read_text()==original


@pytest.mark.parametrize('passes',[True,False])
def test_existing_candidates_revalidate_without_generation_or_relabeling(tmp_path,monkeypatch,passes):
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    spec=importlib.util.spec_from_file_location('unit_revalidation',repo/'scripts/tasks/produce_behavior.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    cfg=tmp_path/'config.yaml';cfg.write_text('{}')
    old=tmp_path/'old';old.mkdir()
    for name,value in [('plan',{'inputs':{}}),('records',[]),('summary',{'generated':1})]:
        (old/f'{name}.json').write_text(json.dumps(value))
    historical=old/'gpu_summary_old.json';historical.write_text(json.dumps(dict(validation={
        'acceptance_policy':'strict-v3','environment_only':True,'sources':{}})))
    before={p.name:p.read_bytes() for p in old.iterdir()}
    checked=[]
    monkeypatch.setattr(cli,'host_config',lambda:dict(behavior_assets_root=tmp_path))
    monkeypatch.setattr(cli,'_check_inputs',lambda inputs,completed_root:checked.append(completed_root))
    monkeypatch.setattr(cli,'validate_completed',lambda *a:checked.append(a[1]))
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k:pytest.fail('must not regenerate candidates'))
    def execute(command,**kwargs):
        assert 'batch-probe' in command and '--verify-background-restore' in command
        assert command[command.index('--acceptance-policy')+1]=='execution-v1'
        batch=Path(command[command.index('--batch')+1])
        (batch/'gpu_summary_new.json').write_text(json.dumps(dict(validation_id='new',probed=1,
            validation=dict(acceptance_policy='execution-v1',environment_only=True,sources={
                'scripts/tasks/behavior_pipeline.py':cli.sha256_path(repo/'scripts/tasks/behavior_pipeline.py')}),
            status_counts={'environment_ready' if passes else 'environment_failed':1})))
        return NS(returncode=0)
    monkeypatch.setattr(cli.subprocess,'run',execute)
    out=tmp_path/'new'
    args=NS(out=out,config=cfg,gpus=[4,5],count=1,max_proposals=1,seed=1,workers=1,
        sim_python='unit',cpu_python='unit',existing_batch=old,acceptance_policy='execution-v1')
    assert cli.run(args)==(0 if passes else 2)
    assert checked==[old,old]
    assert all((old/name).read_bytes()==value for name,value in before.items())
    status=json.loads((out/'production_status.json').read_text())
    assert status['environment_ready']==int(passes) and status['active_validation_id']=='new'
