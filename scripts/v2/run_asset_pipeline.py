"""Resume the complete authored-task production workflow; never weaken semantics.

All catalogue entries get a stage. Reviewed bindings compile; unresolved goals
remain in a concrete authoring queue. Native jobs, seeded retries, fresh restore
checks and human-ready exports run automatically with bounded concurrency.
"""
from pathlib import Path
from collections import Counter
import argparse
import fcntl
import hashlib
import html
import json
import os
import signal
import subprocess
import sys
import time

import validate_dense_batch as batch

ROOT=Path(__file__).resolve().parents[2]
CHILD=None
STOP=False
OUTPUT=None

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {} if default is None else default

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def worker_limit(out,maximum):
    path=out/'worker_budget.json'
    if not path.exists():return maximum
    value=json.loads(path.read_text()).get('workers')
    if type(value) is not int or not 1<=value<=maximum:raise ValueError('Invalid live worker budget')
    return value


def task_runtime_sources(task):
    return batch.runtime_sources(task['runtime_root']) if task.get('runtime_root') else batch.runtime_sources()


def fingerprint(task):
    identity=dict(recipe=hashlib.sha256((ROOT/task['scenario']).read_bytes()).hexdigest(),
                  runtime=task_runtime_sources(task))
    if task.get('runtime_environment'):identity['environment']=task['runtime_environment']
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def current(task,report,accepted):
    return (report.get('status')==accepted and bool(report.get('fresh_native_setup'))
        and report.get('runtime_sources')==task_runtime_sources(task)
        and (not task.get('runtime_environment') or report.get('runtime_environment')==task['runtime_environment'])
        and report.get('scenario_sha256')==hashlib.sha256((ROOT/task['scenario']).read_bytes()).hexdigest())


def assign_new_runtime(tasks,out,runtime,ledger):
    """Pin only never-started recipes; preserve every earlier task's identity."""
    path=out/'runtime_assignments.json'
    assignments=read(path)
    release=read(runtime/'release.json')
    flags=release.get('enable')
    if not isinstance(flags,dict) or not flags:raise ValueError('Runtime has no explicit acceleration flags')
    allowed={'YGBENCH_GPU_PROPOSALS','YGBENCH_MESH_ACCELERATION','YGBENCH_READONLY_MATERIAL_PROBE',
             'YGBENCH_GPU_OCCUPANCY_RANKING','YGBENCH_BATCH_POSE_WRITES','YGBENCH_ISAAC_THREADS','YGBENCH_CANONICAL_XFORM','YGBENCH_PHYSICS_VELOCITY_WRITES','YGBENCH_PHYSICS_POSITION_WRITES','YGBENCH_AUTHORING_CONTACT_AUDIT','YGBENCH_GPU_LOCAL_PROPOSALS','YGBENCH_EXPERT_LAYOUT','YGBENCH_DEFER_AUTHORING_RENDER','YGBENCH_BATCH_SURFACE_RAYS','YGBENCH_EARLY_STABILITY_REJECTION','YGBENCH_AXIS_ORIENTATION_PROPOSALS','YGBENCH_GPU_VOLUME_COMPONENTS','YGBENCH_REUSE_PREPARED_USD'}
    allowed.add('YGBENCH_NATIVE_MOTION_CONVERGENCE')
    allowed.add('YGBENCH_MEASURED_SURFACE_LAYOUT')
    if set(flags)-allowed:raise ValueError('Unsupported runtime feature flag: '+', '.join(sorted(set(flags)-allowed)))
    # Explicitly clear optional features so inherited environment cannot enable
    # code paths absent from the validated release configuration.
    environment={key:('32' if key=='YGBENCH_ISAAC_THREADS' else '0') for key in allowed}
    environment.update({key:str(value) for key,value in flags.items()})
    sources=batch.runtime_sources(runtime)
    attempted={key.split(':',1)[0] for key in ledger}
    for task in tasks:
        tid=task['id']
        if tid in assignments or tid in attempted or (out/tid).exists():continue
        assignments[tid]=dict(runtime_root=str(runtime.resolve()),runtime_environment=environment,
            runtime_sources=sources,scenario_sha256=hashlib.sha256((ROOT/task['scenario']).read_bytes()).hexdigest())
    write(path,assignments)
    return assignments


def apply_runtime_assignments(tasks,out):
    assignments=read(out/'runtime_assignments.json')
    for task in tasks:
        record=assignments.get(task['id'])
        if record is None:continue
        if record['scenario_sha256']!=hashlib.sha256((ROOT/task['scenario']).read_bytes()).hexdigest():
            raise ValueError('Pinned recipe changed: '+task['id'])
        if record['runtime_sources']!=batch.runtime_sources(record['runtime_root']):
            raise ValueError('Pinned native runtime changed: '+task['id'])
        task.update(runtime_root=record['runtime_root'],runtime_environment=record['runtime_environment'])
    return tasks


def failure_kind(report):
    error=str(report.get('error') or '')
    if report.get('status')=='initialization_timeout':return 'sampling_timeout'
    if 'LAYOUT_NOT_CONVERGED' in error:return 'unstable_layout'
    if 'placement failed' in error or 'RELATION_FAILED' in error:return 'containment_or_support_failed'
    if 'nan' in error.lower() or 'quaternion' in error:return 'physics_numerical_failure'
    if report.get('status')=='authoring_blocked':return 'semantic_binding_error'
    return report.get('status','pending')

def restored(task,out):
    report=read(out/task['id']/'restore_validation.json')
    frozen=out/task['id']/'frozen_member.private.json'
    return (current(task,report,'restore_validated') and frozen.exists() and
        report.get('frozen_manifest_sha256')==hashlib.sha256(frozen.read_bytes()).hexdigest())

def stop(signum,frame):
    global STOP
    STOP=True
    batch.stop(signum,frame)
    if CHILD and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass

def export(catalogue,out,semantic,status):
    tasks=apply_runtime_assignments(read(catalogue/'manifest.json').get('tasks',[]),out)
    batch.CATALOG=catalogue
    parents=batch.prepare_semantic_queue(semantic,tasks)
    physical=[];ready=[];repairs=[]
    active={r['id'] for r in read(out/'batch_status.json').get('active',[])}
    for task in tasks:
        report=read(out/task['id']/'validation.json')
        restore=read(out/task['id']/'restore_validation.json')
        valid=current(task,report,'initialization_validated')
        frozen=out/task['id']/'frozen_member.private.json'
        replay=(valid and current(task,restore,'restore_validated') and frozen.exists()
            and restore.get('frozen_manifest_sha256')==hashlib.sha256(frozen.read_bytes()).hexdigest())
        state='ready_for_human_review' if replay else 'restore_pending' if valid else failure_kind(report)
        if valid and restore and not replay:state='restore_'+failure_kind(restore)
        if task.get('authoring_issues'):state='semantic_binding_error'
        elif task['id'] in active:state='restore_running' if valid else 'build_running'
        if report and (report.get('runtime_sources')!=task_runtime_sources(task) or
                       task.get('runtime_environment') and report.get('runtime_environment')!=task['runtime_environment']) and not task.get('authoring_issues'):
            state='stale_validation'
        row=dict(id=task['id'],parent_task_id=task['parent_task_id'],scene=task['scene'],instruction=task['instruction'],
            stage=state,scenario=task['scenario'],play_scenario=str(out/task['id']/'play.yaml'),
            runtime_root=str(task.get('runtime_root',batch.ROOT)),runtime_environment=task.get('runtime_environment',{}),
            full_skill_completion='unverified',error=task.get('authoring_issues') or
                (restore.get('error') if valid and not replay else report.get('error')),seed=report.get('seed'))
        physical.append(row)
        if replay:ready.append(row)
        elif state not in ('pending','restore_pending'):repairs.append(row)
    members_by_parent={};ready_ids={r['id'] for r in ready}
    for member in physical:members_by_parent.setdefault(member['parent_task_id'],[]).append(member)
    for p in parents:
        members=members_by_parent.get(p['id'],[])
        p['instance_states']={r['id']:r['stage'] for r in members}
        if members:
            p['stage']='human_review_pending' if any(r['id'] in ready_ids for r in members) else 'physical_pipeline'
    summary=dict(status=status,total_semantic_tasks=len(parents),physical_instances=len(tasks),
        semantic_stage_counts=dict(Counter(r['stage'] for r in parents)),
        physical_counts=dict(Counter(r['stage'] for r in physical)),human_ready_instances=len(ready),
        updated_unix=time.time(),full_skill_completion='unverified')
    write(out/'pipeline_summary.json',summary)
    write(out/'semantic_queue.json',dict(summary=summary,tasks=parents))
    write(out/'repair_queue.json',dict(semantic=[p for p in parents if not p['instances']],physical=repairs))
    write(out/'human_ready.json',dict(status='initialization_and_fresh_restore_only',tasks=ready))
    fields=['id','parent_task_id','scene','stage','instruction']
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in fields)+'</tr>' for r in physical)
    page=('<!doctype html><meta charset="utf-8"><title>Asset search production</title>'
        '<style>body{font:15px sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>'
        '<h1>Asset search production</h1><p>Initialization and fresh-process restore are verified separately. '
        'Skill completion, visual identity and human difficulty remain unverified.</p><pre>'
        +html.escape(json.dumps(summary,indent=2))+'</pre><table><tr>'
        +''.join('<th>'+k+'</th>' for k in fields)+'</tr>'+table+'</table>')
    (out/'review.html').write_text(page)
    (out/'prompts.txt').write_text('\n\n'.join(r['id']+'\n'+r['instruction'] for r in physical)+'\n')
    return tasks

def main():
    global CHILD,OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-root',type=Path,default=ROOT/'scenarios/asset_search_v1')
    parser.add_argument('--semantic-catalogue',type=Path,default=ROOT/'build/dense_search_semantic_1k')
    parser.add_argument('--output-root',type=Path,default=ROOT/'runs/asset_search_pipeline')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--rolling',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--reserve-worker-for-run',type=Path,
                        help='Reserve one slot while this earlier one-worker pipeline is alive.')
    parser.add_argument('--attempts',type=int,default=2)
    parser.add_argument('--restore-after-build',action='store_true',help='Repair lane: verify each new layout immediately.')
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--bindings',type=Path,nargs='*',default=[])
    parser.add_argument('--bindings-only',action='store_true')
    parser.add_argument('--runtime-root',type=Path,default=ROOT,help='Immutable native runtime; compiler remains in this workspace.')
    parser.add_argument('--new-runtime-root',type=Path,help='Pin unstarted tasks to this immutable runtime; keep previous tasks on their original release.')
    parser.add_argument('--skip-compile',action='store_true',help='Validate an existing, explicitly selected repair catalogue.')
    args=parser.parse_args()
    if args.new_runtime_root and not args.rolling:parser.error('Per-task runtimes require rolling scheduler')
    if not 1<=args.attempts<=3:parser.error('Use 1 to 3 bounded placement seeds.')
    os.chdir(ROOT);out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    OUTPUT=out
    batch.ROOT=args.runtime_root.resolve()
    catalogue=args.catalog_root.resolve();semantic=args.semantic_catalogue.resolve()
    lock=(out/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This pipeline is already running.')
    if args.report_only:export(catalogue,out,semantic,'report_only');return
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    ledger=read(out/'attempts.json')
    def worker_count():
        limit=worker_limit(out,args.workers)
        if args.reserve_worker_for_run:
            previous=read(args.reserve_worker_for_run/'pipeline_status.json')
            if previous.get('status')=='running' and previous.get('pid'):
                try:os.kill(int(previous['pid']),0)
                except ProcessLookupError:pass
                else:return max(1,limit-1)
        return limit
    def command(argv,stage):
        global CHILD
        argv=list(argv)
        if '--workers' in argv:argv[argv.index('--workers')+1]=worker_count()
        write(out/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),workers=worker_count(),worker_budget=args.workers))
        with (out/'pipeline.log').open('a') as log:
            native=stage.startswith(('build_seed_','verify_frozen_restore'))
            cwd=batch.ROOT if native else ROOT
            env=dict(os.environ)
            if native:env['PYTHONPATH']=str(cwd/'src')+os.pathsep+env.get('PYTHONPATH','')
            CHILD=subprocess.Popen([sys.executable,'-u',*map(str,argv)],cwd=cwd,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            updated=time.monotonic()
            while CHILD.poll() is None:
                if STOP:stop(None,None)
                time.sleep(2)
                if stage!='compile_reviewed_bindings' and time.monotonic()-updated>=30:
                    export(catalogue,out,semantic,'running');updated=time.monotonic()
            code=CHILD.returncode;CHILD=None
        if code and not STOP:raise RuntimeError(f'{stage} failed with exit code {code}; see pipeline.log')
    compile_args=['scripts/v2/compile_asset_search.py','--append','--out',catalogue]
    if args.bindings:compile_args+=['--bindings',*args.bindings]
    if args.bindings_only:compile_args+=['--bindings-only']
    if not args.skip_compile:command(compile_args,'compile_reviewed_bindings')
    if read(catalogue/'manifest.json').get('binding_reviews'):
        command(['scripts/v2/sync_bound_recipes.py','--catalog-root',catalogue],'reconcile_reviewed_recipes')
    if args.new_runtime_root:
        assign_new_runtime(read(catalogue/'manifest.json').get('tasks',[]),out,args.new_runtime_root.resolve(),ledger)
    tasks=export(catalogue,out,semantic,'running')
    # Snapshot every input before native workers start. Logs are tied to exact recipes.
    import shutil
    snapshot=out/'source_snapshot'/str(time.time_ns());snapshot.mkdir(parents=True)
    for name in (*batch.RUNTIME_FILES,'scripts/v2/compile_asset_search.py','scripts/v2/generate_dense_pilot.py',
                 'scripts/v2/run_asset_pipeline.py','scripts/v2/constructed_storage.py','scripts/v2/sync_bound_recipes.py',
                 'scripts/v2/rolling_asset_jobs.py','scripts/v2/validate_dense_batch.py',
                 'scripts/v2/audit_dense_layouts.py'):
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2((batch.ROOT if name in batch.RUNTIME_FILES else ROOT)/name,dest)
    for runtime in {task['runtime_root'] for task in tasks if task.get('runtime_root')}:
        label=hashlib.sha256(runtime.encode()).hexdigest()[:16]
        for name in batch.RUNTIME_FILES:
            target=snapshot/'assigned_runtimes'/label/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(runtime)/name,target)
    if (out/'runtime_assignments.json').exists():shutil.copy2(out/'runtime_assignments.json',snapshot/'runtime_assignments.json')
    shutil.copytree(catalogue,snapshot/'catalogue')
    geometry_index=ROOT/'build/native_fillable_geometry/index.json'
    if geometry_index.exists():shutil.copy2(geometry_index,snapshot/'native_fillable_geometry_index.json')
    shutil.copy2(semantic/'catalogue.json',snapshot/'semantic_catalogue.json')
    shutil.copy2(semantic/'asset_model_plan.json',snapshot/'asset_model_plan.json')
    # First wave checks two known successes and a new multi-object AND goal.
    priority={'asset_P002_01':0,'asset_U017_01':1,'asset_F067_01':2}
    tasks.sort(key=lambda t:(priority.get(t['id'],t.get('repair_priority',3)),t['id']))
    if args.rolling:
        import rolling_asset_jobs
        rolling_asset_jobs.run(sys.modules[__name__],tasks,catalogue,out,semantic,ledger,
                               args.attempts,args.timeout,worker_count,restore_after_build=args.restore_after_build)
        status='interrupted' if STOP else 'drained' if batch.STATUS.get('status')=='drained' else 'finished_with_pending_review'
        export(catalogue,out,semantic,status)
        write(out/'pipeline_status.json',dict(status=status,pid=os.getpid(),workers=worker_count()))
        return
    base=['scripts/v2/validate_dense_batch.py','--workers',args.workers,'--timeout',args.timeout,
          '--catalog-root',catalogue,'--output-root',out]
    pending_restore=[t for t in tasks if current(t,read(out/t['id']/'validation.json'),'initialization_validated')
                     and not restored(t,out)]
    if pending_restore and not STOP:
        command([*base,'--restore','--only',*[t['id'] for t in pending_restore]],'verify_frozen_restore')
        export(catalogue,out,semantic,'running')
    for seed in range(args.attempts):
        if STOP:break
        pending=[t for t in tasks if not t.get('authoring_issues')
            and not current(t,read(out/t['id']/'validation.json'),'initialization_validated')
            and f"{t['id']}:{fingerprint(t)}:{seed}" not in ledger]
        start=0
        while start<len(pending):
            if STOP:break
            wave=pending[start:start+worker_count()]
            start+=len(wave)
            command([*base,'--seed',seed,'--only',*[t['id'] for t in wave]],f'build_seed_{seed}')
            if not STOP:
                for t in wave:ledger[f"{t['id']}:{fingerprint(t)}:{seed}"]=dict(time=time.time(),report=read(out/t['id']/'validation.json'))
                write(out/'attempts.json',ledger)
            # Verify reload immediately, rather than waiting for the whole dataset.
            restore=[t for t in wave if current(t,read(out/t['id']/'validation.json'),'initialization_validated')
                and not restored(t,out)]
            if restore and not STOP:
                command([*base,'--restore','--only',*[t['id'] for t in restore]],'verify_frozen_restore')
            export(catalogue,out,semantic,'running')
    # Resume a run stopped after generation but before restoration.
    remaining=[t for t in tasks if current(t,read(out/t['id']/'validation.json'),'initialization_validated')
        and not restored(t,out)]
    if remaining and not STOP:command([*base,'--restore','--only',*[t['id'] for t in remaining]],'verify_frozen_restore')
    status='interrupted' if STOP else 'finished_with_pending_review'
    export(catalogue,out,semantic,status)
    write(out/'pipeline_status.json',dict(status=status,pid=os.getpid(),workers=args.workers))

if __name__=='__main__':
    try:main()
    except Exception as error:
        stop(None,None)
        if OUTPUT:write(OUTPUT/'pipeline_status.json',dict(status='pipeline_error',error=str(error),pid=os.getpid()))
        raise
