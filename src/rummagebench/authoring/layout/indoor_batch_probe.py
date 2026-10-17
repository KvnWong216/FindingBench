"""Bounded, resumable real-GPU diagnostics of CPU-generated home candidates."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
import threading

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import sha256_path, sha256_bytes, canonical_json
from rummagebench.sim.omnigibson.behavior_adapter import load_manifest


def choose_probes(records, selected_ids, limit):
    selected=set(selected_ids)
    groups={}
    for r in records:
        if r['id'] in selected: groups.setdefault(r['scene'],[]).append(r)
    # Round-robin homes, independent of previous GPU outcomes. Do not pick
    # only target locations already known to succeed.
    ordered=[]
    while any(groups.values()):
        for scene in sorted(groups):
            if groups[scene]:ordered.append(groups[scene].pop(0))
    return ordered[:limit]


def classify_process(result, code, error, expected):
    result=dict(result)
    incomplete='environment_incomplete' if expected.get('environment_only') else 'execution_smoke_incomplete'
    if code!=0 or result.get('status') in (None,'running'):
        result.update(status=incomplete if result.get('status') in (None,'running','execution_smoke_passed','environment_ready')
                      else result['status'],process_error=error or f'EXIT_{code}_OR_MISSING_FINAL_RECEIPT')
    if result.get('status') in ('execution_smoke_passed','environment_ready') and any(result.get(k)!=v for k,v in expected.items()):
        result.update(status=incomplete,process_error='GPU_RESULT_INPUT_BINDING_MISMATCH')
    return result


def probe_batch(args, repo: Path):
    root=args.batch.resolve()
    mode=getattr(args,'background_mode','native-realization')
    if mode not in ('native-realization','direct-proposal'):raise ValueError('INVALID_BACKGROUND_MODE')
    realized=mode=='native-realization'
    profile=getattr(args,'background_stability_profile','observed-motion-v3')
    restore=getattr(args,'verify_background_restore',True)
    acceptance=getattr(args,'acceptance_policy','strict-v3')
    if acceptance not in ('strict-v3','execution-v1') or (acceptance!='strict-v3' and (not realized or profile!='observed-motion-v3')):
        raise ValueError('INVALID_EXECUTION_ACCEPTANCE_PROFILE')
    if profile not in ('legacy','observed-motion-v3') or (not realized and (profile!='legacy' or restore)):
        raise ValueError('INVALID_BATCH_VALIDATION_PROFILE')
    if not realized or not restore or profile!='observed-motion-v3':
        raise ValueError('ENVIRONMENT_PRODUCTION_REQUIRES_NATIVE_RESTORE_AND_OBSERVATION')
    sources={str(p.relative_to(repo)):sha256_path(p) for p in [
        repo/'scripts/tasks/behavior_pipeline.py',Path(__file__),
        *[repo/'src/rummagebench/authoring/layout'/n for n in ('indoor_realization.py','indoor_stability.py')],
        *[repo/'src/rummagebench/sim/omnigibson'/n for n in ('backend.py','spatial_snapshot.py','state_io.py')]]}
    validation=dict(profile=profile,restore=restore,mode=mode,sources=sources,acceptance_policy=acceptance,
                    environment_only=True)
    validation_id=sha256_bytes(canonical_json(validation))[:16]
    folder_name=('gpu_realized_' if realized else 'gpu_')+validation_id
    summary=json.loads((root/'summary.json').read_text())
    records=json.loads((root/'records.json').read_text())
    if not 1<=args.limit<=len(summary['selected_ids']):raise ValueError('INVALID_PROBE_COUNT')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<={4,5}:
        raise ValueError('ONLY_UNIQUE_GPUS_4_5_ALLOWED')
    selected=choose_probes(records,summary['selected_ids'],args.limit)
    import fcntl
    with (root/'.gpu-probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress={};progress_lock=threading.Lock()
        def publish(record):
            with progress_lock:
                progress[record['id']]=record
                current=json.loads((root/'summary.json').read_text())
                rows=list(progress.values())
                report=dict(generated=current['generated'],probed=len(rows),
                    validation=validation,validation_id=validation_id,background_mode=mode,
                    status_counts=dict(Counter(r['status'] for r in rows)),
                    pending_gpu=max(0,current['generated']-len(rows)),evaluation_ready_episodes=0,
                    scope='environment readiness only; no policy execution or search-cost certification',
                    records=sorted(rows,key=lambda r:r['id']))
                atomic_write_json(root/f'gpu_summary_{validation_id}.json',report)
        def worker(gpu,items):
            reports=[]
            for item in items:
                manifest=load_manifest(Path(item['assembly']))
                if (sha256_path(item['scenario'])!=item['scenario_sha256']
                        or manifest['manifest_sha256']!=item['assembly_sha256']):
                    raise ValueError('GPU_BATCH_SCENARIO_CHANGED')
                folder=root/item['id']/folder_name;receipt=folder/'process.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text())
                    if (r['scenario_sha256']!=item['scenario_sha256']
                            or r['assembly_sha256']!=manifest['manifest_sha256']
                            or r.get('validation')!=validation
                            or r['result_sha256']!=sha256_path(folder/'result.json')):
                        raise ValueError('GPU_BATCH_RECEIPT_CONFLICT')
                    for name,digest in r.get('stage_sha256',{}).items():
                        if sha256_path(folder/f'{name}.json')!=digest:raise ValueError('GPU_BATCH_STAGE_CHANGED')
                    reports.append(r);publish(r);continue
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError(f'GPU_BATCH_INTERRUPTED_KEEP_AND_REVIEW:{folder}')
                folder.mkdir(exist_ok=True)
                command=[args.sim_python,str(repo/'scripts/tasks/behavior_pipeline.py'),'gpu-smoke',
                         '--scenario',item['scenario'],'--out',str(folder),'--gpu',str(gpu),'--environment-only']
                if realized:command.append('--realize-background')
                command.extend(['--background-stability-profile',profile,'--acceptance-policy',acceptance])
                if restore:command.append('--verify-background-restore')
                runner_digest=sha256_path(repo/'scripts/tasks/behavior_pipeline.py')
                started=time.monotonic();code=None;error=None
                # The child runner bounds its isolated simulator to 1200 s;
                # this outer deadline allows cleanup. All-card device access
                # remains impossible in the existing isolated child.
                with (folder/'run.log').open('w') as log:
                    try:
                        code=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1'),
                            timeout=1260,check=False).returncode
                    except subprocess.TimeoutExpired:error='GPU_PROCESS_TIMEOUT'
                result_path=folder/'result.json'
                result=json.loads(result_path.read_text()) if result_path.exists() else {}
                result=classify_process(result,code,error,dict(scenario_sha256=item['scenario_sha256'],
                    assembly_sha256=manifest['manifest_sha256'],gpu=gpu,
                    runner_sha256=runner_digest,background_stability_profile=profile,
                    verify_background_restore=restore,acceptance_policy=acceptance,environment_only=True))
                atomic_write_json(result_path,result)
                passed=code==0 and result.get('status')=='environment_ready'
                r=dict(id=item['id'],scene=item['scene'],gpu=gpu,
                    status='environment_ready' if passed else result.get('status','environment_incomplete'),
                    scenario_sha256=item['scenario_sha256'],assembly_sha256=manifest['manifest_sha256'],
                    result_sha256=sha256_path(result_path),returncode=code,
                    elapsed_s=round(time.monotonic()-started,2),evaluation_ready_episodes=0,
                    validation=validation,
                    background_mode=mode,stage_sha256={name:sha256_path(folder/f'{name}.json')
                        for name in ['background_realization','background_after_target','background_restore','environment_ready']
                        if (folder/f'{name}.json').exists()},
                    error=result.get('error') or result.get('process_error'),
                    failures=dict(Counter(a['failure'] for a in result.get('attempts',[]) if 'failure' in a)))
                atomic_write_json(receipt,r);reports.append(r)
                publish(r)
                print(json.dumps(r,ensure_ascii=False),flush=True)
            return reports
        reports=[]
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(worker,gpu,selected[i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
            for f in as_completed(futures):reports.extend(f.result())
        summary=json.loads((root/'summary.json').read_text())
        report=dict(generated=summary['generated'],probed=len(reports),
            validation=validation,validation_id=validation_id,
            background_mode=mode,
            status_counts=dict(Counter(r['status'] for r in reports)),
            pending_gpu=summary['generated']-len(reports),evaluation_ready_episodes=0,
            scope='environment readiness only; no policy execution or search-cost certification',
            records=sorted(reports,key=lambda r:r['id']))
        atomic_write_json(root/f'gpu_summary_{validation_id}.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return 0
