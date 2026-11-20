#!/usr/bin/env python3
"""Select a prepared task and record human play through the benchmark skills."""
import argparse
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
PILOTS={
 'store_honey_open_route':'scenarios/behavior_store_honey_open_route/scenario.yaml',
 'store_honey':'scenarios/behavior_store_honey/scenario.yaml',
 'boxing_books_up_for_storage':'scenarios/behavior_boxing_books_up_for_storage/scenario.yaml',
 'carrying_in_groceries':'scenarios/behavior_groceries_pilot/scenario.yaml',
}

# L2 goal-only instructions: never display private source locations, route
# traces, spatial-condition labels, or oracle step counts in the play menu.
SMOKE_TASKS = [
 'X09002__L2_S1',  # rice
 'X12007__L2_S1',  # mug
 'X07014__L2_S1',  # shampoo
 'X06010__L2_S1',  # picture frame
 'X14013__L2_S1',  # drill
 'X02005__L2_S2',  # tea leaves
 'X03007__L2_S2',  # drip pot
 'X01009__L2_S3',  # teacup
 'X06005__L2_S3',  # banana
 'X12008__L2_S1',  # milk
]

def smoke_tasks():
    import yaml
    manifest=json.loads((ROOT/'build/vla_difficulty_v2/manifest.json').read_text())
    configs=json.loads((ROOT/'runs/vla_difficulty_v2_expansion/validation/summary.json').read_text())['configs']
    variants={v['variant_id']:v for v in manifest['variants']}
    metrics_path=ROOT/'build/vla_difficulty_v2/difficulty/metrics.json'
    metrics={r['config']:r for r in json.loads(metrics_path.read_text())['rows']} if metrics_path.exists() else {}
    result=[]
    for name in SMOKE_TASKS:
        v=variants[name]
        key=v['task_id']+'|'+v['spatial_level']
        if configs.get(key,{}).get('status')!='accepted':
            raise ValueError(f'{name}: current execution validation is not accepted')
        path=ROOT/v['scenario']
        instruction=yaml.safe_load(path.read_text())['instruction']
        band=metrics.get(key,{}).get('difficulty_band','unclassified')
        result.append(dict(scenario=path,scene=v['scene'],instruction=instruction,difficulty_band=band))
    return result


def dense_tasks():
    import yaml
    manifest=json.loads((ROOT/'scenarios/dense_search_pilot/manifest.json').read_text())
    result=[]
    for task in manifest['tasks']:
        path=ROOT/task['scenario']
        report_path=ROOT/'runs/dense_search_pilot'/task['id']/'validation.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        current=report.get('scenario_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()
        current=current and all((ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
            for name,sha in report.get('runtime_sources',{}).items())
        current=current and bool(report.get('fresh_native_setup'))
        status=report.get('status','unvalidated') if current else 'unvalidated'
        result.append(dict(scenario=path,scene=task['scene'],instruction=yaml.safe_load(path.read_text())['instruction'],
                           difficulty_band=status,ready=status=='initialization_validated'))
    return result


def asset_tasks(root):
    from rummagebench.core.scenario import load_scenario
    from rummagebench.authoring.tasks.household_state import load_frozen_member
    manifest=json.loads((root/'human_ready.json').read_text())
    result=[]
    for task in manifest['tasks']:
        path=Path(task['play_scenario'])
        scenario=load_scenario(path)
        # The menu must not admit a stale or missing exported checkpoint.
        load_frozen_member(Path(scenario.scene.frozen_member_manifest),scenario)
        result.append(dict(scenario=path,scene=task['scene'],instruction=task['instruction'],
            runtime_root=task.get('runtime_root',str(ROOT)),
            runtime_environment=task.get('runtime_environment',{}),
            difficulty_band='initialization and restore verified; human completion pending',ready=True))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=list(PILOTS))
    p.add_argument('--scenario',type=Path,help='Use another prepared scenario')
    p.add_argument('--suite',choices=['smoke','dense','asset'],help='Human trial menu; asset tasks restore verified frozen initial states')
    p.add_argument('--asset-run',type=Path,default=ROOT/'runs/asset_full_queue_20261105')
    p.add_argument('--pick',type=int,help='Select a numbered task from --suite smoke')
    p.add_argument('--list',action='store_true')
    p.add_argument('--port',type=int,default=8090)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    runtime_root=ROOT;runtime_environment={}
    if a.pick is not None and not a.suite:p.error('--pick requires --suite smoke')
    if a.suite:
        if a.task or a.scenario:p.error('--suite cannot be combined with --task or --scenario')
        try:tasks=asset_tasks(a.asset_run) if a.suite=='asset' else dense_tasks() if a.suite=='dense' else smoke_tasks()
        except (KeyError,ValueError,OSError) as e:p.error(str(e))
        if a.list or a.pick is None:
            print('Asset search: frozen initial states; human completion pending.' if a.suite=='asset' else
                  'Dense search pilot: physical initialization status; completion and human difficulty pending.' if a.suite=='dense' else
                  'Human trial set: execution-verified tasks; goal-only instructions.')
            for i,t in enumerate(tasks,1):print(f"{i:2}. [{t['difficulty_band'].title()}] [{t['scene']}] {t['instruction']}")
        if a.list:return 0
        if not tasks:p.error('No tasks are ready yet. Wait for initialization and restore verification.')
        try:
            pick=a.pick if a.pick is not None else int(input(f'Select task [1-{len(tasks)}]: '))
            if not 1 <= pick <= len(tasks):raise ValueError()
        except (ValueError,EOFError):p.error(f'Select a number from 1 to {len(tasks)}')
        if a.suite=='dense' and not tasks[pick-1]['ready']:
            p.error('This recipe has not passed physical initialization validation. It is not ready for human testing.')
        a.scenario=tasks[pick-1]['scenario']
        runtime_root=Path(tasks[pick-1].get('runtime_root',ROOT)).resolve()
        runtime_environment=tasks[pick-1].get('runtime_environment',{})
    if a.task and a.scenario:p.error('Choose --task or --scenario, not both')
    if a.list:
        for i,name in enumerate(PILOTS,1): print(f'{i}. {name} [initialization checked; completion unverified]')
        return 0
    if not a.task and not a.scenario:
        for i,name in enumerate(PILOTS,1):print(f'{i}. {name}')
        try:
            selected=int(input(f'Select task [1-{len(PILOTS)}]: '))
            if selected not in range(1,len(PILOTS)+1):raise ValueError()
            a.task=list(PILOTS)[selected-1]
        except (ValueError,IndexError):p.error(f'Select a number from 1 to {len(PILOTS)}')
    scenario=(ROOT/(a.scenario or PILOTS[a.task])).resolve()
    if not scenario.is_file():p.error(f'Scenario not found: {scenario}')
    output=(a.output or ROOT/'runs/human_oracle'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}').resolve()
    print(f'Recording: {output}',flush=True)
    print('Unassisted human play. MOVE_TO selects a floor destination; TURN rotates left/right/back; VIEW changes gaze only.',flush=True)
    os.chdir(runtime_root)
    os.environ.update({key:str(value) for key,value in runtime_environment.items()})
    # Older pinned runtimes read the native dataset path at SDK import time.
    # Preserve an explicit user override; discover only complete installations.
    if 'OMNIGIBSON_DATA_PATH' not in os.environ:
        for parent in (ROOT,*ROOT.parents):
            data=parent/'behavior-datasets'
            if all((data/name).is_dir() for name in ('behavior-1k-assets','omnigibson-robot-assets')):
                os.environ['OMNIGIBSON_DATA_PATH']=str(data);break
    os.environ['PYTHONPATH']=str(runtime_root/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    os.execv(sys.executable,[sys.executable,'-u',str(runtime_root/'scripts/serve_play.py'),
        '--scenario',str(scenario),'--port',str(a.port),'--host',a.host,'--seed',str(a.seed),'--output',str(output)])

if __name__=='__main__':raise SystemExit(main())
