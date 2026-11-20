"""Physical single-GPU device isolation, without importing CUDA or Kit.

NVML index, NVIDIA character-device minor and Vulkan index are distinct.
Resolve via PCI/UUID; never assume /dev/nvidia<N> is nvidia-smi GPU N.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess


def resolve_gpu(index: int) -> dict:
    result=subprocess.run(['nvidia-smi','--query-gpu=index,pci.bus_id,uuid','--format=csv,noheader'],
                          capture_output=True,text=True,check=True,timeout=10)
    records=[]
    for line in result.stdout.splitlines():
        i,bus,uuid=[part.strip() for part in line.split(',')]
        if int(i)==index:
            pci=bus.lower()[-12:]
            info=(Path('/proc/driver/nvidia/gpus')/pci/'information').read_text()
            minor=re.search(r'^Device Minor:\s*(\d+)\s*$',info,re.M)
            actual_uuid=re.search(r'^GPU UUID:\s*(\S+)\s*$',info,re.M)
            if not minor or not actual_uuid or actual_uuid[1]!=uuid:
                raise ValueError('GPU_PCI_UUID_MAPPING_UNVERIFIED')
            device=Path(f'/dev/nvidia{minor[1]}')
            if not device.is_char_device():raise ValueError('GPU_DEVICE_MISSING')
            dri=[str(p.resolve()) for p in sorted(Path('/dev/dri/by-path').glob(f'pci-{pci}-*'))]
            if not dri or any(not Path(p).is_char_device() for p in dri):
                raise ValueError('GPU_DRI_MAPPING_UNVERIFIED')
            records.append(dict(index=index,pci=pci,uuid=uuid,device=str(device),dri=dri))
    if len(records)!=1:raise ValueError('GPU_INDEX_UNRESOLVED')
    return records[0]


def isolated_gpu_command(argv: list[str], gpu: dict) -> tuple[list[str],dict]:
    if gpu.get('index') not in (4, 5):
        raise ValueError('FINDINGBENCH_EXPERIMENTS_REQUIRE_PHYSICAL_GPU_4_OR_5')
    bwrap=shutil.which('bwrap')
    if not bwrap:raise RuntimeError('GPU_DEVICE_SANDBOX_UNAVAILABLE')
    devices=[gpu['device'],*gpu['dri'],'/dev/nvidiactl','/dev/nvidia-uvm',
             '/dev/nvidia-uvm-tools','/dev/nvidia-modeset']
    cmd=[bwrap,'--bind','/','/','--dev','/dev']
    for device in devices:
        if not Path(device).is_char_device():raise ValueError('GPU_CONTROL_DEVICE_MISSING')
        cmd.extend(['--dev-bind',device,device])
    cmd.extend(['--proc','/proc','--tmpfs','/tmp','--unshare-user','--unshare-pid',
                '--unshare-net','--die-with-parent','--',*argv])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMNIGIBSON_GPU_ID='0',
             OMNIGIBSON_HEADLESS='1',PYTHONUNBUFFERED='1',
             FINDINGBENCH_GPU_UUID=gpu['uuid'],FINDINGBENCH_GPU_DEVICE=gpu['device'],
             FINDINGBENCH_GPU_INDEX=str(gpu['index']))
    return cmd,env


def verify_isolated_cuda(torch) -> None:
    expected=os.environ.get('FINDINGBENCH_GPU_UUID')
    device=os.environ.get('FINDINGBENCH_GPU_DEVICE')
    exposed=[str(p) for p in Path('/dev').glob('nvidia[0-9]*')]
    if (not expected or exposed!=[device] or torch.cuda.device_count()!=1
            or 'GPU-'+str(torch.cuda.get_device_properties(0).uuid)!=expected):
        raise RuntimeError('GPU_DEVICE_ISOLATION_NOT_VERIFIED')
    torch.cuda.set_device(0)


def configure_single_gpu_kit(isaacsim) -> None:
    """Use the SDK's extra_args hook before launch; no all-card P2P test."""
    original=isaacsim.SimulationApp
    def launch(config=None,**kwargs):
        config=dict(config or {})
        if config.get('active_gpu')!=0 or config.get('physics_gpu')!=0 or config.get('multi_gpu') is not False:
            raise RuntimeError('KIT_SINGLE_ISOLATED_GPU_CONFIGURATION_REQUIRED')
        config['extra_args']=[*config.get('extra_args',[]),'--/validate/p2p/enabled=false']
        return original(config,**kwargs)
    isaacsim.SimulationApp=launch
