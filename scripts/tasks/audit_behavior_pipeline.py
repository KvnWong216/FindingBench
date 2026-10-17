"""Offline, stage-level failure attribution; never starts a simulator."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import statistics

from _common import REPO
from rummagebench.authoring.layout.indoor_background import exclude_covered_support_envelopes, qualify_supports_for_pool, verify_result
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import sha256_path


def audit(batch, replays=None):
    records=json.loads((batch/'records.json').read_text())
    by_id={r['id']:r for r in records}
    houses=defaultdict(Counter);constraints=Counter();covered_failures=0;failed_seconds=0.
    catalogs={};pool_catalogs={};failures_with_evidence=0;bindings={};covered_or_occupied_failures=0
    for row in records:
        houses[row['scene']][row['status']]+=1
        attempts=sorted((batch/row['id']).glob('attempt_*/solver_request.json'))
        if not attempts:continue
        request_path=attempts[-1];request=json.loads(request_path.read_text())
        if row['scene'] not in catalogs:
            copied=deepcopy(request['native_context']['objects'])
            catalogs[row['scene']]=exclude_covered_support_envelopes(copied)
        if row['status']!='generation_failed':continue
        failed_seconds+=row['elapsed_s']
        log=request_path.parent/'solver.log'
        matches=re.findall(r'violations=(\{[^\n]+\})',log.read_text(errors='replace')) if log.exists() else []
        if not matches:continue
        violations=ast.literal_eval(matches[-1]);failures_with_evidence+=1
        constraints.update(violations.keys())
        pool_key=row['scene']+'|'+','.join(sorted(request['assets']))
        if pool_key not in pool_catalogs:
            copied=deepcopy(request['native_context'])
            copied['skipped_supports'].extend(exclude_covered_support_envelopes(copied['objects']))
            qualify_supports_for_pool(copied,request['assets'])
            pool_catalogs[pool_key]=[s for s in copied['skipped_supports'] if s['reason'] in
                ('covered_by_native_support_envelope','no_pool_fit_in_native_occupied_envelope')]
        covered={f"daily_items_{v['entity']}" for v in catalogs[row['scene']]}
        covered_failures+=bool(covered & set(violations))
        covered_or_occupied_failures+=bool({f"daily_items_{v['entity']}" for v in pool_catalogs[pool_key]} & set(violations))
        bindings[str(log)]=sha256_path(log)
    gpu=Counter();no_realizer=0;timings=[];actions=Counter()
    for row in records:
        folder=batch/row['id']/'gpu'
        if not (folder/'process.json').exists():continue
        p=json.loads((folder/'process.json').read_text());r=json.loads((folder/'result.json').read_text())
        if (p['result_sha256']!=sha256_path(folder/'result.json')
                or p['assembly_sha256']!=row['assembly_sha256'] or p['scenario_sha256']!=row['scenario_sha256']):
            raise ValueError('PIPELINE_AUDIT_GPU_BINDING_CHANGED')
        no_realizer+='background_realizer_sha256' not in r
        timings.append(p['elapsed_s'])
        if 'BACKGROUND_SUPPORT_FAILED' in (r.get('error') or ''):gpu['background_support_rejected']+=1
        elif r.get('status')=='execution_smoke_passed':gpu['known_target_skill_success']+=1
        elif len(r.get('attempts',[]))==25:gpu['bounded_pose_diagnostic_exhausted']+=1
        else:gpu['other_failure_or_incomplete']+=1
        actions.update(a['failure'] for a in r.get('attempts',[]) if 'failure' in a)
    pairs=[]
    if replays:
        for folder in sorted(replays.glob('home_*')):
            path=folder/'solver_request.json'
            if not path.exists():continue
            old_path=sorted((batch/folder.name).glob('attempt_*/solver_request.json'))[-1]
            old=json.loads(old_path.read_text())
            new=json.loads(path.read_text())
            expected=deepcopy(old['native_context'])
            expected['skipped_supports'].extend(exclude_covered_support_envelopes(expected['objects']))
            qualify_supports_for_pool(expected,new['assets'])
            if any(old[k]!=new[k] for k in ['seed','assets','per_surface','steps','upstream']) or expected!=new['native_context']:
                raise ValueError('PAIRED_REPLAY_HAS_UNCONTROLLED_INPUT_CHANGE:'+folder.name)
            result_path=folder/'solver_result.json'
            status='no_success_receipt'
            if result_path.exists():
                result=json.loads(result_path.read_text());verify_result(new,result)
                if result['request_sha256']!=sha256_path(path):raise ValueError('REPLAY_REQUEST_BINDING_CHANGED')
                status=result['status']
            pairs.append(dict(id=folder.name,seed=new['seed'],per_surface=new['per_surface'],
                old_status=by_id[folder.name]['status'],new_status=status,
                retained_native_objects=len(new['native_context']['retained_native_objects']),
                old_supports=sum(o['support'] for o in old['native_context']['objects']),
                new_supports=sum(o['support'] for o in new['native_context']['objects']),
                support_exclusions=new['native_context']['skipped_supports'],
                request_sha256=sha256_path(path)))
    result=dict(scope='pipeline failure attribution, not per-episode certification',
        by_house={k:dict(v) for k,v in houses.items()},constraint_failure_counts=dict(constraints),
        failed_proposals_with_constraint_evidence=failures_with_evidence,
        failures_including_covered_support_constraint=covered_failures,
        failures_including_covered_or_occupied_support_constraints=covered_or_occupied_failures,
        pool_conditioned_proposal_exclusions=pool_catalogs,
        failed_proposal_worker_seconds=round(failed_seconds,2),covered_support_envelopes=catalogs,
        baseline_gpu_stages=dict(gpu),baseline_runs_missing_realization_stage=no_realizer,
        baseline_gpu_median_process_seconds=statistics.median(timings) if timings else None,
        bounded_diagnostic_failure_counts=dict(actions),controlled_cpu_replays=pairs,
        evaluation_ready_episodes=0,source_logs=bindings)
    atomic_write_json(batch/'pipeline_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_logs'},indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--replays',type=Path)
    a=p.parse_args();audit(a.batch.resolve(),a.replays.resolve() if a.replays else None)
