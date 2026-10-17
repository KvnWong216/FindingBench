"""Synthetic launcher fixtures, never physical environment evidence."""
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def sample(tmp_path, monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts'))
    spec=importlib.util.spec_from_file_location('unit_play_data',repo/'scripts/play_data.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    def write(path, value):path.write_text(json.dumps(value))
    batch=tmp_path/'batch';folder=batch/'home_0001'/'gpu_realized_unit';folder.mkdir(parents=True)
    scenario=tmp_path/'scenario.yaml';scenario.write_text('synthetic unit fixture')
    assembly=tmp_path/'assembly.json';assembly.write_text('{"manifest_sha256":"'+cli.hashlib.sha256(b'{}').hexdigest()+'"}')
    checksum=cli.read(assembly)['manifest_sha256']
    row=dict(id='home_0001',scenario=str(scenario),scenario_sha256=cli.digest(scenario),
             assembly=str(assembly),assembly_sha256=checksum)
    write(batch/'records.json',[row]);write(batch/'summary.json',dict(selected_ids=['home_0001']))
    write(tmp_path/'production_status.json',dict(active_acceptance_policy='execution-v1'))
    for name in ('environment_start.pt','environment_start.png'):
        (folder/name).write_bytes(b'unit fixture only')
    write(folder/'result.json',dict(status='environment_ready',environment_only=True))
    write(folder/'environment_ready.json',dict(status='environment_ready',policy_executed=False,
        public_reset=True,spawn_clear=True,target_relation=True,
        checkpoint_sha256=cli.digest(folder/'environment_start.pt'),image_sha256=cli.digest(folder/'environment_start.png')))
    for name in ('background_realization','background_restore','background_after_target'):
        write(folder/f'{name}.json',dict(status='unit fixture'))
    receipt=dict(status='environment_ready',returncode=0,scenario_sha256=row['scenario_sha256'],
        assembly_sha256=checksum,validation=dict(environment_only=True,restore=True,acceptance_policy='execution-v1'),
        result_sha256=cli.digest(folder/'result.json'),stage_sha256={name:cli.digest(folder/f'{name}.json')
            for name in ('background_realization','background_restore','background_after_target','environment_ready')})
    write(folder/'process.json',receipt)
    return cli,tmp_path,folder,receipt,write


def test_one_argument_resolves_same_saved_environment(sample):
    cli,root,folder,receipt,write=sample
    assert cli.select_sample(root,'1')==cli.select_sample(root,'home_0001')
    assert cli.select_sample(root,'1')['folder']==str(folder)


@pytest.mark.parametrize('changed',['environment_start.pt','environment_start.png','background_restore.json','result.json'])
def test_changed_saved_artifact_refuses_play(sample,changed):
    cli,root,folder,receipt,write=sample
    (folder/changed).write_bytes(b'changed')
    with pytest.raises(ValueError,match='改变'):cli.select_sample(root,'1')


@pytest.mark.parametrize('state',['environment_failed','running','execution_smoke_passed'])
def test_unverified_or_old_policy_data_not_silently_regenerated(sample,state):
    cli,root,folder,receipt,write=sample
    receipt['status']=state;write(folder/'process.json',receipt)
    with pytest.raises(ValueError,match='唯一'):cli.select_sample(root,'1')


def test_ambiguous_receipts_and_path_traversal_reject(sample):
    import shutil
    cli,root,folder,receipt,write=sample
    shutil.copytree(folder,folder.parent/'gpu_realized_other')
    with pytest.raises(ValueError,match='唯一'):cli.select_sample(root,'1')
    with pytest.raises(ValueError,match='格式'):cli.select_sample(root,'../home_0001')


def test_existing_ui_uses_injected_environment_without_importing_simulator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    import serve_play
    calls=[]
    class Env:
        def reset(self):calls.append('reset');return {'observation':{'instruction':'unit task'}}
    state=serve_play.PlayState('unit',0,env_factory=lambda *a,**k:Env())
    state.boot()
    assert state.ready and calls==['reset']
    assert state.state_payload()['instruction']=='unit task'
