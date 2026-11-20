"""Resume a dense RAB cohort and run one native Isaac worker at a time."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from rab_dense_search import load, save


def alive(pid):
    try:
        return not Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].startswith('Z ')
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--prior-run',type=Path)
    args=parser.parse_args()
    out=args.output.resolve()
    script=Path(__file__).with_name('rab_dense_search.py')
    children={}
    deadline=time.time()+12*3600
    def status(stage, **details):
        save(out/'overnight_status.json',dict(status=stage,pid=os.getpid(),
             workers=children,updated_at=time.time(),**details))
    def start(lane, mode, number, task_ids=()):
        directory=out/lane
        directory.mkdir(exist_ok=True)
        command=[sys.executable,'-u',str(script),'--run',str(args.run.resolve()),
                 '--repo',str(args.repo.resolve()),'--output',str(out),'--lane',lane,
                 '--attempts','14' if lane=='drawer' else '16']
        if mode=='prepare':
            command+=['--prepare-only']
        if args.prior_run:
            command+=['--prior-run',str(args.prior_run.resolve())]
        for ident in task_ids:
            command+=['--task-id',ident]
        env=dict(os.environ,PXR_WORK_THREAD_LIMIT='4',OMNI_KIT_ACCEPT_EULA='YES')
        env['TMPDIR']=tempfile.mkdtemp(prefix='native_',dir=directory)
        with (directory/f'{mode}_{number:02d}.log').open('ab') as stream:
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children[lane]=dict(pid=process.pid,mode=mode,attempt=number,command=command)
        status('running')
        return process
    def wait(process,lane):
        while process.poll() is None and time.time()<deadline:
            status('running',waiting_for=lane)
            time.sleep(12)
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            return 124
        return process.returncode

    if args.prior_run:
        # Prove the fix across the cases that reached the old open/close gate
        # before spending time on fresh layouts and asset-incompatible rows.
        prior=args.prior_run.resolve()
        drawer_ids = []
        for report in (prior/'drawer/cases').glob('*/result.json'):
            row=load(report)
            if any(item['status']=='dense_native_gate_failed' for item in row['attempts']):
                drawer_ids.append(row['id'])
        phases=[('drawer','replay',sorted(drawer_ids)),('open','repair',()),('drawer','repair',())]
        outcomes={}
        for lane,phase,selected in phases:
            if phase=='replay' and not selected:
                continue
            for retry in range(1,4):
                if time.time()>=deadline:
                    break
                process=start(lane,phase,retry,selected)
                outcomes[phase+'_'+lane]=wait(process,phase+'_'+lane)
                if process.returncode==0:
                    break
        lanes={name:load(out/name/'status.json') for name in ('open','drawer')
               if (out/name/'status.json').exists()}
        complete=(len(outcomes)==len(phases) and all(code==0 for code in outcomes.values()))
        status('all_cases_attempted' if complete else 'native_validation_incomplete',
               lanes=lanes,outcomes=outcomes)
        return 0 if complete else 2

    existing=load(out/'open/launch.json')
    children['open']=dict(pid=existing['pid'],mode='native',command=existing['command'])
    cpu=start('drawer','prepare',1)
    while alive(existing['pid']) and time.time()<deadline:
        status('running',waiting_for='open_native_and_drawer_cpu')
        time.sleep(12)
    if time.time()>=deadline:
        status('time_limit',remaining='drawer_native')
        return 2

    # Recover an interrupted open worker using its per-episode receipts.
    if load(out/'open/status.json')['status']!='finished':
        for retry in range(1,4):
            process=start('open','native',retry)
            if wait(process,'open_native')==0:
                break
    cpu_exit=wait(cpu,'drawer_cpu')
    status('drawer_preparation_finished',cpu_exit=cpu_exit)
    for retry in range(1,4):
        if time.time()>=deadline:
            break
        process=start('drawer','native',retry)
        if wait(process,'drawer_native')==0:
            break
    lanes={name:load(out/name/'status.json') for name in ('open','drawer')}
    complete=all(row['status']=='finished' for row in lanes.values())
    status('all_cases_attempted' if complete else 'native_validation_incomplete',lanes=lanes)
    return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
