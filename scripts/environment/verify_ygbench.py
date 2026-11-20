"""CPU/import checks for the dedicated ygbench environment; no simulator launch."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    report = {'python':sys.version, 'executable':sys.executable,
              'conda_prefix':os.getenv('CONDA_PREFIX'), 'checks':{}}
    for name,module in [('rummagebench','rummagebench'),('torch','torch'),('bddl','bddl'),
                        ('mcp','mcp'),('pin','pinocchio'),('usd-core','pxr.Usd'),
                        ('pytest','pytest'),('warp-lang','warp'),('pyarrow','pyarrow'),
                        ('torchcodec','torchcodec'),('omnigibson','omnigibson')]:
        try:
            importlib.import_module(module)
            report['checks'][name]={'ok':True,'version':importlib.metadata.version(name)}
        except Exception as error:
            report['checks'][name]={'ok':False,'error':str(error)}
    try:
        import torch
        value = torch.ones(8,device='cuda').sum().item()
        report['checks']['cuda']={'ok':value==8,'device':torch.cuda.get_device_name(0),
                                  'cuda_version':torch.version.cuda}
    except Exception as error:
        report['checks']['cuda']={'ok':False,'error':str(error)}
    report['checks']['git']={'ok':shutil.which('git') is not None,'path':shutil.which('git')}
    report['checks']['compiler']={'ok':shutil.which(os.getenv('CXX','g++')) is not None, 'path':shutil.which(os.getenv('CXX','g++'))}
    report['checks']['ffmpeg']={'ok':shutil.which('ffmpeg') is not None, 'path':shutil.which('ffmpeg')}
    try:
        from rummagebench.adapters.mcp_server import create_server
        server=create_server()
        report['checks']['mcp_server']={'ok':True}
    except Exception as error:
        report['checks']['mcp_server']={'ok':False,'error':str(error)}
    report['ok']=all(c['ok'] for c in report['checks'].values())
    output=Path('runs/ygbench_setup/imports.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['ok'] else 1

if __name__=='__main__':
    raise SystemExit(main())
