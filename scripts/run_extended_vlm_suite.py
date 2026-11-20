"""Serially own one inference server and one simulator worker, across models.

Run in ygbench. Each model gets its own server process/environment. This
script never adopts or terminates a pre-existing server on the chosen port.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from rummagebench.agents.local_vlm import request_json, prompt_fingerprint, PROMPT_VERSION, source_fingerprint as base_source_fingerprint

ROOT=Path(__file__).resolve().parents[1]


def source_fingerprint():
    digest=hashlib.sha256(base_source_fingerprint().encode())
    for name in ['scripts/run_extended_vlm_suite.py','scripts/serve_extended_vlm.py',
                 'scripts/extended_vlm_adapters.py','configs/models/local_vlm.json']:
        digest.update((ROOT/name).read_bytes())
    return digest.hexdigest()


def await_dependencies(previous, downloads, save_state):
    while True:
        old=json.loads(previous.read_text())
        if old['status'] not in ('running','waiting'):
            break
        try:os.kill(old['pid'],0)
        except ProcessLookupError:
            raise RuntimeError('previous suite has stale running state; inspect it before starting another GPU workload')
        save_state('waiting_previous_suite')
        time.sleep(30)
    while True:
        if downloads.exists():
            data=json.loads(downloads.read_text())
            if data['status']=='complete':return data['models']
        save_state('waiting_downloads');time.sleep(30)



def stop(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['smolvlm2-2.2b','phi4-multimodal','llava-onevision-7b','minicpm-v4.5-8b','gemma3-4b'])
    p.add_argument('--variants',nargs='+')
    p.add_argument('--all',action='store_true')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--load-timeout',type=int,default=1800)
    p.add_argument('--episode-timeout',type=int,default=7200)
    p.add_argument('--after-suite',type=Path,default=ROOT/'runs/vlm_grid_v3/suite.json')
    p.add_argument('--downloads',type=Path,default=ROOT/'runs/vlm_expansion_setup/downloads.json')
    a=p.parse_args();a.out_root=a.out_root.resolve()
    profiles=json.loads((ROOT/'configs/models/local_vlm.json').read_text())['models']
    if a.all and a.variants:p.error('choose --all or --variants')
    if len(set(a.models))!=len(a.models) or not set(a.models)<=set(profiles):p.error('duplicate/unknown model')
    if (a.out_root/'suite.json').exists():p.error('suite already exists; choose a new output directory')
    a.out_root.mkdir(parents=True,exist_ok=True)
    state={'status':'running','pid':os.getpid(),'prompt_version':PROMPT_VERSION,
           'prompt_sha256':prompt_fingerprint(),'source_sha256':source_fingerprint(),'seed':a.seed,
           'requested_models':a.models,'max_new_tokens':512,'models':{name:{'status':'pending'} for name in a.models},
           'variants':'all 36 compiled variants' if a.all else (a.variants or ['T01_knife_fetch__L2_S1'])}
    def save():
        tmp=a.out_root/'suite.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.out_root/'suite.json')
    save()
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    server=worker=None
    try:
        def waiting(status):
            state['status']=status;save()
        ready=await_dependencies(a.after_suite,a.downloads,waiting)
        state['status']='running';save()
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:sock.bind(('127.0.0.1',a.port))
            except OSError:p.error(f'port {a.port} is occupied; stop the existing service or choose another --port')
        for name in a.models:
            if source_fingerprint()!=state['source_sha256']:raise RuntimeError('benchmark sources changed during suite')
            if ready.get(name,{}).get('status')!='ready':
                state['models'][name]={'status':'blocked', 'download':ready.get(name)};save();continue
            row=state['models'][name]={'status':'loading'};save()
            server_log=a.out_root/(name+'.server.log')
            with server_log.open('w') as log:
                server=subprocess.Popen([str(ROOT/'scripts/serve_extended_vlm.sh'),'--model',name,'--port',str(a.port)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                    env={**os.environ,'YGBENCH_VLM_ENV':profiles[name]['environment']})
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
                    row.update(status='preflight',server=health);save()
                    preflight=a.out_root/'preflight'/name
                    with (a.out_root/(name+'.preflight.log')).open('w') as preflight_log:
                        worker=subprocess.Popen([sys.executable,str(ROOT/'scripts/check_local_vlm.py'),
                            '--model',name,'--endpoint',endpoint,'--out',str(preflight)],cwd=ROOT,
                            stdout=preflight_log,stderr=subprocess.STDOUT,start_new_session=True)
                        worker.wait(timeout=600)
                    # A malformed model action is not an infrastructure error.
                    if not (preflight/'calls.jsonl').exists():
                        raise RuntimeError('no successful generation in preflight; see preflight report')
                    row.update(status='pilot');save()
                    pilot_root=a.out_root/'pilots'
                    with (a.out_root/(name+'.pilot.log')).open('w') as pilot_log:
                        worker=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate_local_vlm.py'),
                            '--model',name,'--endpoint',endpoint,'--out-root',str(pilot_root),
                            '--seed',str(a.seed),'--timeout',str(a.episode_timeout)],cwd=ROOT,
                            stdout=pilot_log,stderr=subprocess.STDOUT,start_new_session=True)
                        worker.wait(timeout=a.episode_timeout+120)
                    report=pilot_root/name/f'seed_{a.seed}'/'T01_knife_fetch__L2_S1/report.json'
                    if not report.exists() or not json.loads(report.read_text()).get('summary',{}).get('run_valid'):
                        raise RuntimeError('pilot infrastructure failed; full grid withheld')
                    row.update(status='evaluating');save()
                    cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_local_vlm.py'),
                         '--model',name,'--endpoint',endpoint,'--seed',str(a.seed),
                         '--out-root',str(a.out_root),'--timeout',str(a.episode_timeout)]
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
