"""Synthetic device topology tests; these never import CUDA or launch GPU work."""
from types import SimpleNamespace
from pathlib import Path
import pytest
from rummagebench.sim import gpu_isolation as gpu


@pytest.mark.parametrize('index,minor',[(4,6),(5,7)])
def test_nvml_index_resolves_pci_minor_not_index(monkeypatch,index,minor):
    monkeypatch.setattr(gpu.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=
        f'{index}, 00000000:a1:00.0, GPU-unit-fixture\n'))
    monkeypatch.setattr(Path,'read_text',lambda self:f'GPU UUID: GPU-unit-fixture\nDevice Minor: {minor}\n')
    monkeypatch.setattr(Path,'is_char_device',lambda self:True)
    monkeypatch.setattr(Path,'glob',lambda self,pattern:[Path('/dev/dri/renderD134')])
    actual=gpu.resolve_gpu(index)
    assert actual['device']==f'/dev/nvidia{minor}' and actual['index']==index
    monkeypatch.setattr(gpu.shutil,'which',lambda name:'/usr/bin/bwrap')
    command,env=gpu.isolated_gpu_command(['python','unit-fixture-only'],actual)
    assert f'/dev/nvidia{minor}' in command and f'/dev/nvidia{index}' not in command
    assert '--dev' in command and '--unshare-net' in command
    assert env['CUDA_VISIBLE_DEVICES']=='GPU-unit-fixture'
    assert env['OMNIGIBSON_GPU_ID']=='0'
    assert env['FINDINGBENCH_GPU_INDEX']==str(index)


@pytest.mark.parametrize('index',[0,1,2,3,6,7])
def test_other_physical_gpu_is_rejected_before_any_launch(index):
    with pytest.raises(ValueError,match='REQUIRE_PHYSICAL_GPU_4_OR_5'):
        gpu.isolated_gpu_command(['unit-fixture-only'],{'index':index})


def test_pipeline_unauthorized_gpu_is_rejected_by_cli_before_simulator_import():
    import subprocess,sys
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'scripts/tasks/household_pipeline.py'),
        'gpu-preflight-infinigen','--gpu','3','--spec','missing','--assembly','missing',
        '--start','missing','--out','missing'],capture_output=True,text=True)
    assert result.returncode==2 and 'invalid choice: 3' in result.stderr
    assert 'isaacsim' not in result.stderr and 'GPU isolation:' not in result.stderr


def test_wrong_pci_uuid_is_rejected(monkeypatch):
    monkeypatch.setattr(gpu.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=
        '4, 00000000:81:00.0, GPU-unit-fixture\n'))
    monkeypatch.setattr(Path,'read_text',lambda self:'GPU UUID: GPU-another-fixture\nDevice Minor: 6\n')
    with pytest.raises(ValueError,match='GPU_PCI_UUID'):gpu.resolve_gpu(4)


@pytest.mark.parametrize('extra_device,device_count,uuid',[('/dev/nvidia0',1,'unit-fixture'),
    (None,2,'unit-fixture'),(None,1,'wrong-fixture')])
def test_runtime_must_have_one_device_and_exact_uuid(monkeypatch,extra_device,device_count,uuid):
    monkeypatch.setenv('FINDINGBENCH_GPU_UUID','GPU-unit-fixture')
    monkeypatch.setenv('FINDINGBENCH_GPU_DEVICE','/dev/nvidia6')
    exposed=[Path('/dev/nvidia6')]+([Path(extra_device)] if extra_device else [])
    monkeypatch.setattr(Path,'glob',lambda self,pattern:exposed)
    fake=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:device_count,
        get_device_properties=lambda i:SimpleNamespace(uuid=uuid)))
    with pytest.raises(RuntimeError,match='ISOLATION_NOT_VERIFIED'):gpu.verify_isolated_cuda(fake)


def test_kit_disables_global_p2p_and_rejects_multiple_devices():
    calls=[]
    sdk=SimpleNamespace(SimulationApp=lambda config,**kwargs:calls.append(config))
    gpu.configure_single_gpu_kit(sdk)
    sdk.SimulationApp({'active_gpu':0,'physics_gpu':0,'multi_gpu':False})
    assert calls[0]['extra_args']==['--/validate/p2p/enabled=false']
    with pytest.raises(RuntimeError,match='KIT_SINGLE'):sdk.SimulationApp({'active_gpu':4,'physics_gpu':4,'multi_gpu':True})
