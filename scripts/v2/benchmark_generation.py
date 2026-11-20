"""Controlled, fixed-recipe full initialization + fresh reload throughput test.

Refuses to share the simulator with production. Output must be new: resuming a
partly cached run would silently remove already-spent time from the denominator.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,os,time,sys,signal
import validate_dense_batch as batch

ROOT=Path(__file__).resolve().parents[2]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def accepted_pair(build,restore,recipe_hash,runtime,frozen_hash,seed,environment=None):
    return bool(frozen_hash and
        build.get('status')=='initialization_validated' and restore.get('status')=='restore_validated' and
        all(r.get('fresh_native_setup') and r.get('scenario_sha256')==recipe_hash and
            r.get('runtime_sources')==runtime and r.get('seed')==seed and
            (environment is None or r.get('runtime_environment')==environment) for r in (build,restore)) and
        restore.get('frozen_manifest_sha256')==frozen_hash)

def summarize(trials,expected):
    totals={}
    for trial in trials:
        row=totals.setdefault(trial['variant'],dict(accepted=0,attempted=0,wall_seconds=0.))
        row['accepted']+=sum(r['accepted'] for r in trial['jobs'])
        row['attempted']+=len(trial['jobs']);row['wall_seconds']+=trial['wall_seconds']
    for row in totals.values():
        row['accepted_per_hour']=row['accepted']*3600/row['wall_seconds'] if row['wall_seconds'] else 0.
    baselines=[totals[k] for k in ('cpu','gpu') if k in totals]
    candidate=totals.get('candidate',totals.get('mesh_cache',{}))
    complete=len(totals)==3 and all(v['attempted']==expected for v in totals.values())
    best=max((b['accepted_per_hour'] for b in baselines),default=0.)
    ratio=candidate.get('accepted_per_hour',0)/best if complete and best else None
    return dict(totals=totals,complete=complete,speedup_vs_best_baseline=ratio,
        achieved_10x=bool(complete and ratio is not None and ratio>=10 and
            candidate['accepted']>=max(b['accepted'] for b in baselines)))

def active_native_processes():
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(arg.endswith(b'/validate_dense_pilot.py') or arg==b'validate_dense_pilot.py' for arg in argv):result.append(int(proc.name))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    variants=dict(cpu=ROOT/'build'/protocol['baselines'][0],gpu=ROOT/'build'/protocol['baselines'][1],candidate=ROOT/'build'/protocol['candidate'])
    for t in protocol['tasks']:
        if digest(t['scenario'])!=t['scenario_sha256']:raise ValueError('Recipe changed: '+t['id'])
    runtime_hashes={}
    for name,path in variants.items():
        batch.ROOT=path
        runtime_hashes[name]=batch.runtime_sources()
    if args.check_only:
        print(json.dumps(dict(tasks=len(protocol['tasks']),seeds=protocol['seeds'],workers=protocol['native_workers'],variants=list(variants),inputs_verified=True)));return
    others=active_native_processes()
    if others:raise RuntimeError('Production/native jobs still active: '+str(others))
    signal.signal(signal.SIGTERM,batch.stop)
    signal.signal(signal.SIGINT,batch.stop)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'inputs.json').write_text(json.dumps(dict(protocol=protocol,protocol_hash=digest(args.protocol),runtime_hashes=runtime_hashes,driver_sha256=digest(__file__)),indent=2))
    trials=[];expected=len(protocol['tasks'])*len(protocol['seeds'])
    for ordinal,seed in enumerate(protocol['seeds']):
        if batch.STOP.is_set():raise RuntimeError('Benchmark interrupted; results are incomplete')
        order=list(variants);order=order[ordinal%3:]+order[:ordinal%3]
        for variant in order:
            if batch.STOP.is_set():raise RuntimeError('Benchmark interrupted; results are incomplete')
            # Never start a new arm while a native process from elsewhere is live.
            others=active_native_processes()
            if others:raise RuntimeError('Concurrent native workload invalidates comparison: '+str(others))
            directory=args.output/variant/f'seed_{seed}';directory.mkdir(parents=True)
            batch.ROOT=variants[variant];batch.OUT=directory;batch.STOP.clear();batch.SEMANTIC_ROWS=[]
            batch.STATUS=dict(status='running',pid=os.getpid(),workers=protocol['native_workers'],results=[])
            batch.STATUS_PATH=directory/'batch_status.json'
            os.environ.update(YGBENCH_GPU_PROPOSALS='0' if variant=='cpu' else '1',
                YGBENCH_MESH_ACCELERATION='1' if variant=='candidate' else '0',
                YGBENCH_READONLY_MATERIAL_PROBE='0',YGBENCH_GPU_OCCUPANCY_RANKING='0',YGBENCH_BATCH_POSE_WRITES='0',YGBENCH_ISAAC_THREADS='32',YGBENCH_CANONICAL_XFORM='0',YGBENCH_PHYSICS_VELOCITY_WRITES='0',YGBENCH_PHYSICS_POSITION_WRITES='0',YGBENCH_AUTHORING_CONTACT_AUDIT='0',YGBENCH_GPU_LOCAL_PROPOSALS='0',YGBENCH_EXPERT_LAYOUT='0',YGBENCH_DEFER_AUTHORING_RENDER='0',YGBENCH_BATCH_SURFACE_RAYS='0',YGBENCH_EARLY_STABILITY_REJECTION='0',YGBENCH_AXIS_ORIENTATION_PROPOSALS='0',YGBENCH_GPU_VOLUME_COMPONENTS='0',YGBENCH_REUSE_PREPARED_USD='0',
                YGBENCH_CPU_THREADS='2',YGBENCH_COMPILE_THREADS='1',
                TORCHINDUCTOR_CACHE_DIR=str((args.output/'compiler_cache'/variant/'inductor').resolve()),
                TRITON_CACHE_DIR=str((args.output/'compiler_cache'/variant/'triton').resolve()),
                YGBENCH_GEOMETRY_CACHE=str((args.output/'geometry_cache').resolve()))
            if variant=='candidate':
                flags=protocol.get('candidate_environment',{})
                allowed={'YGBENCH_GPU_PROPOSALS','YGBENCH_MESH_ACCELERATION',
                         'YGBENCH_READONLY_MATERIAL_PROBE','YGBENCH_GPU_OCCUPANCY_RANKING',
                         'YGBENCH_BATCH_POSE_WRITES','YGBENCH_ISAAC_THREADS','YGBENCH_CANONICAL_XFORM','YGBENCH_PHYSICS_VELOCITY_WRITES','YGBENCH_PHYSICS_POSITION_WRITES','YGBENCH_AUTHORING_CONTACT_AUDIT','YGBENCH_GPU_LOCAL_PROPOSALS','YGBENCH_EXPERT_LAYOUT','YGBENCH_DEFER_AUTHORING_RENDER','YGBENCH_BATCH_SURFACE_RAYS','YGBENCH_EARLY_STABILITY_REJECTION','YGBENCH_AXIS_ORIENTATION_PROPOSALS','YGBENCH_GPU_VOLUME_COMPONENTS','YGBENCH_REUSE_PREPARED_USD'}
                if set(flags)-allowed:raise ValueError('Unknown candidate acceleration flag')
                os.environ.update({key:str(value) for key,value in flags.items()})
            os.environ.pop('YGBENCH_PROFILE_FIRST',None)
            # Pin flags into each job/report rather than relying only on the
            # parent environment. Otherwise a correct source hash can conceal
            # a different enabled optimization in the measured native job.
            measured_environment={key:os.environ[key] for key in (
                'YGBENCH_GPU_PROPOSALS','YGBENCH_MESH_ACCELERATION',
                'YGBENCH_READONLY_MATERIAL_PROBE','YGBENCH_GPU_OCCUPANCY_RANKING',
                'YGBENCH_BATCH_POSE_WRITES','YGBENCH_ISAAC_THREADS',
                'YGBENCH_CANONICAL_XFORM','YGBENCH_PHYSICS_VELOCITY_WRITES','YGBENCH_PHYSICS_POSITION_WRITES','YGBENCH_AUTHORING_CONTACT_AUDIT','YGBENCH_GPU_LOCAL_PROPOSALS','YGBENCH_EXPERT_LAYOUT',
                'YGBENCH_DEFER_AUTHORING_RENDER','YGBENCH_BATCH_SURFACE_RAYS','YGBENCH_EARLY_STABILITY_REJECTION','YGBENCH_AXIS_ORIENTATION_PROPOSALS','YGBENCH_GPU_VOLUME_COMPONENTS','YGBENCH_REUSE_PREPARED_USD','YGBENCH_CPU_THREADS',
                'YGBENCH_COMPILE_THREADS','YGBENCH_GEOMETRY_CACHE',
                'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
            def job(task):
                task=dict(task,runtime_root=str(variants[variant]),runtime_environment=measured_environment)
                began=time.monotonic()
                build_result=batch.run(task,args.timeout,restore=False,seed=seed)
                d=directory/task['id'];build=batch.read_native_report(d/'validation.json')
                restore_result=None
                if build.get('status')=='initialization_validated':restore_result=batch.run(task,args.timeout,restore=True,seed=seed)
                restore=batch.read_native_report(d/'restore_validation.json');frozen=d/'frozen_member.private.json'
                accepted=accepted_pair(build,restore,task['scenario_sha256'],runtime_hashes[variant],digest(frozen) if frozen.exists() else None,seed,measured_environment)
                return dict(id=task['id'],seed=seed,accepted=accepted,seconds=time.monotonic()-began,build=build_result,restore=restore_result)
            started=time.monotonic()
            with ThreadPoolExecutor(max_workers=protocol['native_workers']) as pool:jobs=list(pool.map(job,protocol['tasks']))
            trials.append(dict(variant=variant,seed=seed,order=order,wall_seconds=time.monotonic()-started,jobs=jobs))
            (args.output/'results.json').write_text(json.dumps(dict(trials=trials,summary=summarize(trials,expected)),indent=2))
    print(json.dumps(summarize(trials,expected),indent=2))

if __name__=='__main__':main()
