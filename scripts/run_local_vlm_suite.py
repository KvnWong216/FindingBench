"""Serially own one inference server and one simulator worker, across models.

Run in ygbench. Each model gets its own server process/environment. This
script never adopts or terminates a pre-existing server on the chosen port.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from rummagebench.agents.local_vlm import request_json, prompt_fingerprint, PROMPT_VERSION, source_fingerprint

ROOT=Path(__file__).resolve().parents[1]


def stop(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['qwen3-vl-4b','internvl3.5-4b','molmo2-4b'])
    p.add_argument('--variants',nargs='+')
    p.add_argument('--all',action='store_true')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--agent-policy',choices=['strict','humanclaw'],default='strict')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--load-timeout',type=int,default=1800)
    p.add_argument('--episode-timeout',type=int,default=7200)
    a=p.parse_args();a.out_root=a.out_root.resolve()
    if a.agent_policy=='humanclaw':
        from rummagebench.agents.humanclaw_vlm import VERSION as prompt_version, fingerprint as prompt_hash
    else:
        prompt_version, prompt_hash = PROMPT_VERSION, prompt_fingerprint
    if a.max_new_tokens<1 or (a.max_steps is not None and a.max_steps<1):p.error('budgets must be positive')
    profiles=json.loads((ROOT/'configs/models/local_vlm.json').read_text())['models']
    if a.all and a.variants:p.error('choose --all or --variants')
    if len(set(a.models))!=len(a.models) or not set(a.models)<=set(profiles):p.error('duplicate/unknown model')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:sock.bind(('127.0.0.1',a.port))
        except OSError:p.error(f'port {a.port} is occupied; stop the existing service or choose another --port')
    if (a.out_root/'suite.json').exists():p.error('suite already exists; choose a new output directory')
    a.out_root.mkdir(parents=True,exist_ok=True)
    state={'status':'running','pid':os.getpid(),'prompt_version':prompt_version,
           'prompt_sha256':prompt_hash(),'source_sha256':source_fingerprint(),'seed':a.seed,
           'requested_models':a.models,'max_new_tokens':a.max_new_tokens,'max_steps':a.max_steps,'agent_policy':a.agent_policy,'models':{name:{'status':'pending'} for name in a.models},
           'variants':'all 36 compiled variants' if a.all else (a.variants or ['T01_knife_fetch__L2_S1'])}
    def save():
        tmp=a.out_root/'suite.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.out_root/'suite.json')
    save()
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    server=worker=None
    try:
        for name in a.models:
            if source_fingerprint()!=state['source_sha256']:raise RuntimeError('benchmark sources changed during suite')
            row=state['models'][name]={'status':'loading'};save()
            server_log=a.out_root/(name+'.server.log')
            with server_log.open('w') as log:
                server_script='serve_extended_vlm.sh' if profiles[name].get('backend') else 'serve_vlm.sh'
                server_env=dict(os.environ)
                if profiles[name].get('environment'):server_env['YGBENCH_VLM_ENV']=profiles[name]['environment']
                server=subprocess.Popen([str(ROOT/'scripts'/server_script),'--model',name,'--port',str(a.port)],
                    env=server_env,
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                try:
                    endpoint=f'http://127.0.0.1:{a.port}'
                    deadline=time.monotonic()+a.load_timeout
                    while time.monotonic()<deadline:
                        if server.poll() is not None:raise RuntimeError(f'server exited {server.returncode}; see {server_log.name}')
                        try:health=request_json(endpoint+'/health',timeout=2)
                        except Exception:time.sleep(2);continue
                        if health.get('model_id')!=profiles[name]['model_id'] or health.get('revision')!=profiles[name]['revision']:
                            raise RuntimeError('server model mismatch')
                        break
                    else:raise TimeoutError('model loading timed out')
                    row.update(status='evaluating',server=health);save()
                    cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_local_vlm.py'),
                         '--model',name,'--endpoint',endpoint,'--seed',str(a.seed),
                         '--out-root',str(a.out_root),'--timeout',str(a.episode_timeout),
                         '--agent-policy',a.agent_policy,'--max-new-tokens',str(a.max_new_tokens)]
                    if a.max_steps is not None:cmd+=['--max-steps',str(a.max_steps)]
                    cmd+=['--all'] if a.all else ['--variants',*(a.variants or ['T01_knife_fetch__L2_S1'])]
                    with (a.out_root/(name+'.evaluation.log')).open('w') as evaluation_log:
                        worker=subprocess.Popen(cmd,cwd=ROOT,stdout=evaluation_log,stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL,start_new_session=True)
                        code=worker.wait()
                    row.update(status='complete' if code==0 else 'completed_with_errors',exit_code=code)
                    summary=a.out_root/name/f'seed_{a.seed}'/'summary.json'
                    if summary.exists():row['summary']=json.loads(summary.read_text())['overall']
                except Exception as e:
                    row.update(status='infrastructure_error',error=f'{type(e).__name__}: {e}')
                finally:
                    stop(worker);worker=None;stop(server);server=None;save()
            print(name,json.dumps(row),flush=True)
        state['status']='complete' if all(r['status']=='complete' for r in state['models'].values()) else 'completed_with_errors'
    except KeyboardInterrupt:
        state['status']='interrupted'
    except Exception as error:
        state['status']='infrastructure_error'
        state['error']=f'{type(error).__name__}: {error}'
    finally:
        stop(worker);stop(server);save()
    return 0 if state['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
