"""Bounded native initialization checks; no VLM, solver, or policy runs.

Each recipe gets its own simulator process. Failed generation is not an agent
failure. Results never certify successful skill execution or human solvability.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import threading
import tempfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/dense_search_pilot'
CATALOG=ROOT/'scenarios/dense_search_pilot'
ACTIVE={}
LOCK=threading.RLock()
STOP=threading.Event()
STATUS=None
STATUS_PATH=None
SEMANTIC_ROWS=[]
_FILE_CACHE={}
_PROFILE_CLAIMED=False

def cached_file(path, *, digest=False):
    """Cache immutable reads, invalidate replacements and same-size edits."""
    path=Path(path);stat=path.stat()
    key=(stat.st_mtime_ns,stat.st_ctime_ns,stat.st_size,stat.st_ino)
    old=_FILE_CACHE.get(str(path))
    if old is None or old[0]!=key:
        raw=path.read_bytes()
        old=(key,raw,hashlib.sha256(raw).hexdigest())
        _FILE_CACHE[str(path)]=old
    return old[2] if digest else json.loads(old[1])

def missing_report_status(timed_out, stopped, exit_code):
    # A late supervisor shutdown must not relabel an already-exited/crashed
    # native job as interrupted; that would erase retry/provenance evidence.
    if timed_out:return 'initialization_timeout','native_timeout'
    if stopped and exit_code in (-signal.SIGTERM,-signal.SIGKILL):
        return 'authoring_interrupted','supervisor_stop_requested'
    return 'infrastructure_failure','native_process_exited_without_current_report'


def measured_command(command, usage_path):
    timer=Path('/usr/bin/time')
    if not timer.exists():return command
    record='{"wall_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"max_rss_kib":%M,"exit_code":%x}'
    # append preserves prior attempts; failures may also emit a text status line.
    return [str(timer),'-a','-o',str(usage_path),'-f',record,'--',*command]


def native_environment(base, root=None):
    root=ROOT if root is None else Path(root)
    env=dict(base)
    if 'OMNIGIBSON_DATA_PATH' not in env:
        for parent in (root,*root.parents):
            data=parent/'behavior-datasets'
            if all((data/name).is_dir() for name in ('behavior-1k-assets','omnigibson-robot-assets')):
                env['OMNIGIBSON_DATA_PATH']=str(data);break
    # Small geometry kernels do not warrant one host-sized pool per simulator.
    threads=env.get('YGBENCH_CPU_THREADS','2')
    if not threads.isdigit() or int(threads)<1:raise ValueError('Invalid YGBENCH_CPU_THREADS')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]=threads
    env['TORCHINDUCTOR_COMPILE_THREADS']=env.get('YGBENCH_COMPILE_THREADS','1')
    # Keep compiler artifacts beyond the per-job USD extraction directory.
    # Torch/Triton own content keys and concurrent cache locking; namespace by
    # interpreter environment so unrelated installations do not share files.
    namespace=hashlib.sha256(str(Path(sys.executable).resolve()).encode()).hexdigest()[:16]
    cache=Path(env.get('YGBENCH_COMPILE_CACHE_ROOT',str(ROOT/'build/native_compiler_cache'))).resolve()/namespace
    env.setdefault('TORCHINDUCTOR_CACHE_DIR',str(cache/'inductor'))
    env.setdefault('TRITON_CACHE_DIR',str(cache/'triton'))
    env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    return env

def prepare_semantic_queue(directory, tasks):
    """Inventory every semantic task without inventing missing goal bindings."""
    plans={r['id']:r for r in cached_file(directory/'asset_model_plan.json')['tasks']}
    manifest=cached_file(CATALOG/'manifest.json')
    deferred={r['id']:r for r in manifest.get('deferred',[])}
    # Match the compiler's configured native asset root.
    assets=Path('/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets')
    rows=[];instances_by_parent={}
    for member in tasks:instances_by_parent.setdefault(member.get('parent_task_id'),[]).append(member['id'])
    for task in cached_file(directory/'catalogue.json')['tasks']:
        plan=plans.get(task['id'],{}); selections=plan.get('selections',[])
        review=manifest.get('binding_reviews',{}).get(task['id'],{})
        overrides=review.get('target_model_overrides',[]) if review.get('status')=='ready' else []
        issues=[]
        if not overrides and not any(r['role']=='target' for r in selections):issues.append('No target model allocated.')
        if plan.get('instruction_sha256')!=hashlib.sha256(task['instruction'].encode()).hexdigest():
            issues.append('Asset allocation is stale relative to the instruction.')
        for selection in selections:
            if overrides and selection['role']=='target':continue
            category,model=selection['asset_id'].split('/')
            meta=assets/'objects'/category/model/'misc/metadata.json'
            if not meta.exists():issues.append('Missing native metadata: '+selection['asset_id'])
            elif cached_file(meta,digest=True)!=selection.get('metadata_sha256'):
                issues.append('Changed native metadata: '+selection['asset_id'])
        for key in overrides:
            category,model=key.split('/')
            if not (assets/'objects'/category/model/'misc/metadata.json').exists():
                issues.append('Missing reviewed native metadata: '+key)
        instances=instances_by_parent.get(task['id'],[])
        review=manifest.get('binding_reviews',{}).get(task['id'],{})
        rows.append(dict(id=task['id'],instruction=task['instruction'],tier=task['tier'],
            binding_review=review,
            source_room=review.get('source_room_type',task['source_room_type']),destination_room=review.get('destination_room_type',task['destination_room_type']),
            target_models=overrides or [r['asset_id'] for r in selections if r['role']=='target'],
            instances=instances,asset_preflight_issues=issues,
            stage='physics_validation' if instances else 'asset_binding_blocked' if issues else
                  'semantic_binding_blocked' if review and review.get('status')!='ready' else
                  'layout_binding_pending' if task['id'] in deferred else 'goal_binding_pending',
            pending_requirements=plan.get('unresolved_requirements',[]),
            layout_issue=deferred.get(task['id']),
            full_skill_completion='unverified'))
    return rows

def publish_semantic_queue():
    if not SEMANTIC_ROWS:return
    from collections import Counter
    active=set(ACTIVE)
    for row in SEMANTIC_ROWS:
        states={}
        for iid in row['instances']:
            report=OUT/iid/'validation.json'
            if iid in active:states[iid]='running'
            elif report.exists():
                data=json.loads(report.read_text())
                states[iid]=data.get('status','pending')
                if states[iid]=='initialization_validated' and data.get('runtime_sources')!=runtime_sources():
                    states[iid]='stale_validation'
            else:states[iid]='pending'
        row['instance_states']=states
    data=dict(total=len(SEMANTIC_ROWS),workers=STATUS['workers'],updated_unix=time.time(),
        stage_counts=dict(Counter(r['stage'] for r in SEMANTIC_ROWS)),
        note='Pending goal/layout bindings require authoring; the simulator does not automatically solve semantic grounding.',
        tasks=SEMANTIC_ROWS)
    target=OUT/'semantic_queue.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(target)

def publish():
    if STATUS is None:return
    with LOCK:
        STATUS['active']=[dict(id=k,pid=v.pid) for k,v in ACTIVE.items()]
        STATUS['updated_unix']=time.time()
        tmp=STATUS_PATH.with_suffix('.tmp');tmp.write_text(json.dumps(STATUS,indent=2));tmp.replace(STATUS_PATH)
        publish_semantic_queue()

def stop(signum,frame):
    STOP.set()
    with LOCK:
        for proc in list(ACTIVE.values()):
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass

RUNTIME_FILES=('src/rummagebench/sim/omnigibson/backend.py',
               'src/rummagebench/sim/omnigibson/inside_volume.py',
               'src/rummagebench/evaluation/goal_checker.py','scripts/v2/validate_dense_pilot.py',
               'src/rummagebench/core/scenario.py','src/rummagebench/authoring/tasks/household_state.py',
               'src/rummagebench/sim/omnigibson/state_io.py')
SEED=0
RESTORE=False

_RUNTIME_HASH_CACHE = {}

def runtime_sources(root=None):
    root=ROOT if root is None else Path(root)
    # Native releases are immutable; stat-keyed caching also invalidates edits in
    # the development workspace without rereading megabytes for every recipe.
    result={}
    for name in RUNTIME_FILES:
        path=root/name;stat=path.stat()
        key=(str(path),stat.st_mtime_ns,stat.st_ctime_ns,stat.st_size,stat.st_ino)
        if key not in _RUNTIME_HASH_CACHE:
            _RUNTIME_HASH_CACHE[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        result[name]=_RUNTIME_HASH_CACHE[key]
    return result


def single_task_catalog(task, root=None):
    """Give the unchanged validator one recipe, avoiding an O(dataset) YAML scan."""
    root=ROOT if root is None else Path(root)
    directory=OUT/'.job_catalogs'/task['id']
    directory.mkdir(parents=True,exist_ok=True)
    link=directory/task['id'];source=(root/task['scenario']).parent.resolve()
    if link.is_symlink() and link.resolve()!=source:link.unlink()
    if not link.exists():link.symlink_to(source,target_is_directory=True)
    return directory


def read_native_report(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        # A killed process may leave an empty/partial report. It is not success.
        return {}


# A rejection is pinned to the exact native asset that produced the observed
# failure. Unknown assets / edited assets are left to native validation.
CAPABILITY_REJECTIONS={('nightstand','uudzva','open'):
    '7169894fdb47dbd3e6ef226ae6755a79356f45a8b125327a40b35bfe019590e5'}
ASSET_OBJECTS=Path('/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets/objects')

def capability_issues(spec):
    objects={o['name']:o for o in spec.get('objects',[])}
    issues=[]
    for name,state in spec.get('initial_states',{}).items():
        obj=objects.get(name,{})
        if state.get('open') is None or obj.get('abilities') is not None:continue
        key=(obj.get('category'),obj.get('model'),'open')
        expected=CAPABILITY_REJECTIONS.get(key)
        if expected is None:continue
        asset=ASSET_OBJECTS/key[0]/key[1]/'usd'/(key[1]+'.encrypted.usd')
        if not asset.exists() or cached_file(asset,digest=True)!=expected:continue
        issues.append(dict(entity=name,asset=key[0]+'/'+key[1],capability='open',
            asset_sha256=expected,reason='This native asset has no Open state.',
            evidence='runs/asset_search_gpu_new_20261106/asset_F249_01/validation.json'))
    return issues


def run(task,timeout,*,restore=None,seed=None):
    global _PROFILE_CLAIMED
    root=Path(task.get('runtime_root',ROOT))
    runtime=runtime_sources(root) if 'runtime_root' in task else runtime_sources()
    flags=task.get('runtime_environment',{})
    restore=RESTORE if restore is None else restore
    seed=SEED if seed is None else seed
    if STOP.is_set():return dict(id=task['id'],status='cancelled')
    if task.get('authoring_issues'):
        dest=OUT/task['id'];dest.mkdir(parents=True,exist_ok=True)
        result=dict(id=task['id'],status='authoring_blocked',issues=task['authoring_issues'])
        (dest/'validation.json').write_text(json.dumps(result,indent=2))
        return result
    tid=task['id'];path=root/task['scenario'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
    dest=OUT/tid;dest.mkdir(parents=True,exist_ok=True)
    report_name='restore_validation.json' if restore else 'validation.json'
    log_name='restore_validation.log' if restore else 'native_validation.log'
    report=dest/report_name
    previous=read_native_report(report)
    frozen=dest/'frozen_member.private.json'
    bundle_matches=(not restore or frozen.exists() and
        previous.get('frozen_manifest_sha256')==hashlib.sha256(frozen.read_bytes()).hexdigest())
    if bundle_matches and previous.get('scenario_sha256')==sha and previous.get('status')==('restore_validated' if restore else 'initialization_validated') and previous.get('fresh_native_setup') and previous.get('runtime_sources')==runtime and (not flags or previous.get('runtime_environment')==flags):
        return dict(id=tid,status='skipped_current_fresh_validation')
    if report.exists() or (dest/log_name).exists():
        import shutil
        archive=dest/'attempt_history'/str(time.time_ns());archive.mkdir(parents=True)
        for name in (report_name,log_name):
            if (dest/name).exists():shutil.copy2(dest/name,archive/name)
    if not restore:
        import yaml
        issues=capability_issues(yaml.safe_load(path.read_text()))
        if issues:
            result=dict(id=tid,scenario_sha256=sha,seed=seed,status='authoring_incompatible_asset',
                fresh_native_setup=False,runtime_sources=runtime,runtime_environment=flags,issues=issues,
                error='Native asset capability mismatch; reauthoring required. No simulator launched.')
            report.write_text(json.dumps(result,indent=2))
            (dest/log_name).write_text('[authoring-preflight] '+json.dumps(issues)+'\n')
            return dict(id=tid,status=result['status'],elapsed_s=0,exit_code=None)
    started=time.time();timed_out=False
    # OmniGibson's native shutdown may bypass Python TemporaryDirectory cleanup.
    # Own the extraction root in the supervisor, which survives native exits.
    with tempfile.TemporaryDirectory(prefix='findingbench-native-') as native_tmp, (dest/log_name).open('w') as log:
        env=native_environment(dict(os.environ,**flags),root)
        env.update(TMPDIR=native_tmp,TEMP=native_tmp,TMP=native_tmp)
        (dest/('restore_resources.json' if restore else 'build_resources.json')).write_text(json.dumps(
            {k:env.get(k,'0') for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','YGBENCH_GPU_PROPOSALS','YGBENCH_MESH_ACCELERATION','YGBENCH_READONLY_MATERIAL_PROBE','YGBENCH_GPU_OCCUPANCY_RANKING','YGBENCH_BATCH_POSE_WRITES','YGBENCH_ISAAC_THREADS','YGBENCH_CANONICAL_XFORM','YGBENCH_PHYSICS_VELOCITY_WRITES','YGBENCH_PHYSICS_POSITION_WRITES','YGBENCH_AUTHORING_CONTACT_AUDIT','YGBENCH_GPU_LOCAL_PROPOSALS','YGBENCH_EXPERT_LAYOUT','YGBENCH_DEFER_AUTHORING_RENDER','YGBENCH_BATCH_SURFACE_RAYS','YGBENCH_EARLY_STABILITY_REJECTION','YGBENCH_AXIS_ORIENTATION_PROPOSALS','YGBENCH_GPU_VOLUME_COMPONENTS','YGBENCH_REUSE_PREPARED_USD','YGBENCH_NATIVE_MOTION_CONVERGENCE')},indent=2))
        command=[sys.executable,'-u',str(root/'scripts/v2/validate_dense_pilot.py'),'--task',tid,'--catalog-root',str(single_task_catalog(task,root)),'--output-root',str(OUT),'--seed',str(seed)]+(['--restore'] if restore else [])
        with LOCK:
            profile=env.get('YGBENCH_PROFILE_FIRST')=='1' and not _PROFILE_CLAIMED
            if profile:_PROFILE_CLAIMED=True
        if profile:
            spy=Path(sys.executable).with_name('py-spy')
            if spy.exists():command=[str(spy),'record','--rate','25','--format','raw',
                '--output',str(dest/'cpu_profile.raw'),'--',*command]
        # GNU time measures this job and its descendants, unlike process-wide
        # RUSAGE_CHILDREN deltas which mix concurrent worker jobs together.
        usage=dest/('restore_usage.jsonl' if restore else 'build_usage.jsonl')
        (dest/('restore_usage_scope.json' if restore else 'build_usage_scope.json')).write_text(json.dumps(dict(
            cpu_rss_scope='profiler_wrapper_only_not_native' if profile and spy.exists() else 'native_process_tree',
            usable_for_native_cpu_comparison=not (profile and spy.exists()),
            note='Wall duration includes profiler overhead when enabled.'),indent=2))
        command=measured_command(command,usage)
        proc=subprocess.Popen(command,
                              cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK:ACTIVE[tid]=proc;publish()
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=proc.returncode
        finally:
            with LOCK:ACTIVE.pop(tid,None);publish()
    result=read_native_report(report)
    if timed_out or not result or not report.exists() or report.stat().st_mtime<started:
        lines=(dest/log_name).read_text(errors='replace').splitlines()
        progress=[line for line in lines if any(tag in line for tag in ('[setup]', '[placement]', '[placement-volume]', 'VALIDATE'))]
        status,error=missing_report_status(timed_out,STOP.is_set(),code)
        result=dict(id=tid,scenario_sha256=sha,seed=seed,status=status,
                    runtime_sources=runtime,last_progress=progress[-1] if progress else None,
                    error=error,exit_code=code)
        report.write_text(json.dumps(result,indent=2))
    if flags:
        result['runtime_environment']=dict(flags)
        report.write_text(json.dumps(result,indent=2))
    return dict(id=tid,status=result.get('status','infrastructure_failure'),elapsed_s=round(time.time()-started),exit_code=code)

def main():
    global OUT,CATALOG,STATUS,STATUS_PATH,SEMANTIC_ROWS,SEED,RESTORE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--only',nargs='*')
    p.add_argument('--catalog-root',type=Path,default=CATALOG)
    p.add_argument('--output-root',type=Path,default=OUT)
    p.add_argument('--semantic-catalogue',type=Path,help='Track all semantic tasks and audit their asset bindings alongside native jobs.')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--restore',action='store_true')
    a=p.parse_args();CATALOG=a.catalog_root.resolve();OUT=a.output_root.resolve()
    SEED=a.seed;RESTORE=a.restore
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if not 1<=a.workers<=3:p.error('Use one to three workers on this host.')
    os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    tasks=json.loads((CATALOG/'manifest.json').read_text())['tasks']
    if a.only:tasks=[t for t in tasks if t['id'] in a.only]
    if a.semantic_catalogue:SEMANTIC_ROWS=prepare_semantic_queue(a.semantic_catalogue.resolve(),tasks)
    OUT.mkdir(parents=True,exist_ok=True);status=dict(status='running',pid=os.getpid(),total=len(tasks),workers=a.workers,results=[])
    target=OUT/'batch_status.json'
    if target.exists():
        archive=OUT/'batch_history';archive.mkdir(exist_ok=True)
        (archive/f'{time.time_ns()}.json').write_bytes(target.read_bytes())
    STATUS=status;STATUS_PATH=target;publish()
    def heartbeat():
        while status['status']=='running' and not STOP.wait(10):publish()
    threading.Thread(target=heartbeat,daemon=True).start()
    condition=threading.Condition();capacity=[a.workers]
    def scheduled(task):
        survey=ROOT/'runs/vla_difficulty_v2/_survey'/f"{task['scene']}.json"
        count=len(json.loads(survey.read_text()).get('entities',{}))
        weight=a.workers if count>250 else 1
        with condition:
            condition.wait_for(lambda:capacity[0]>=weight);capacity[0]-=weight
        try:return run(task,a.timeout)
        finally:
            with condition:capacity[0]+=weight;condition.notify_all()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(scheduled,t):t['id'] for t in tasks}
        for future in as_completed(futures):
            try:result=future.result()
            except Exception as e:result=dict(id=futures[future],status='orchestrator_error',error=str(e))
            with LOCK:status['results'].append(result);publish()
            print(json.dumps(result),flush=True)
            if CATALOG==(ROOT/'scenarios/dense_search_pilot'):
                subprocess.run([sys.executable,str(ROOT/'scripts/v2/generate_dense_pilot.py'),'--refresh-review'],cwd=ROOT,check=True)
    status['status']='interrupted' if STOP.is_set() else 'completed';publish()

if __name__=='__main__':main()
