"""Process-boundary unit test only: no simulator, GPU or production assets."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('exit_code', [0, 139])
def test_direct_smoke_writes_process_receipt_and_rejects_crashed_success(tmp_path, monkeypatch, exit_code):
    from rummagebench.sim import gpu_isolation
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    spec=importlib.util.spec_from_file_location('unit_behavior_cli',repo/'scripts/tasks/behavior_pipeline.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    scenario=tmp_path/'scenario.yaml'
    scenario.write_text('scene: {model: unit-house, assembly_manifest: unit-only}\n')
    output=tmp_path/'sample'/'gpu_realized'
    monkeypatch.setattr(cli,'load_manifest',lambda p:dict(manifest_sha256='unit-manifest'))
    monkeypatch.setattr(gpu_isolation,'resolve_gpu',lambda index:index)
    monkeypatch.setattr(gpu_isolation,'isolated_gpu_command',lambda *a:(['unit-process'],{}))
    def fake_run(*a,**k):
        output.mkdir(parents=True)
        result=dict(status='environment_ready',environment_only=True,scenario_sha256=cli.sha256_path(scenario),
                    assembly_sha256='unit-manifest',runner_sha256=cli.sha256_path(cli.__file__),gpu=4,
                    background_stability_profile='observed-motion-v3',verify_background_restore=True,acceptance_policy='strict-v3')
        (output/'result.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(cli.subprocess,'run',fake_run)
    monkeypatch.setattr(cli.sys,'argv',['behavior_pipeline.py','gpu-smoke','--scenario',str(scenario),
                                     '--out',str(output),'--gpu','4','--realize-background','--verify-background-restore',
                                     '--background-stability-profile','observed-motion-v3'])
    assert cli.main()==(0 if exit_code==0 else 1)
    receipt=json.loads((output/'process.json').read_text())
    assert receipt['returncode']==exit_code
    assert receipt['evaluation_ready_episodes']==0
    assert receipt['result_sha256']==cli.sha256_path(output/'result.json')
    assert receipt['status']==('environment_ready' if exit_code==0 else 'environment_incomplete')
