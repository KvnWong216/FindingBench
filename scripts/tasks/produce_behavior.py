"""Resume bounded native-home production using the existing stage commands.

GPU exports precede CPU generation. GPU validation consumes completed CPU
candidates while generation continues. Rejected locations remain in receipts;
No policy is executed. Environment readiness does not certify search difficulty.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import yaml

from _common import REPO,host_config
from rummagebench.authoring.layout.indoor_batch import make_plan, validate_completed, _check_inputs
from rummagebench.authoring.layout.indoor_geometry import load_geometry
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import sha256_path


def run(args):
    args._may_publish_failure=False
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<={4,5}:
        raise ValueError('ONLY_UNIQUE_GPUS_4_5_ALLOWED')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    runner=str(REPO/'scripts/tasks/behavior_pipeline.py')
    cfg=yaml.safe_load(args.config.read_text())
    assets=Path(host_config()['behavior_assets_root'])
    sources=[Path(__file__).resolve(),Path(runner),*list((REPO/'src/rummagebench/authoring/layout').glob('*.py')),
             *list((REPO/'src/rummagebench/sim/omnigibson').glob('*.py'))]
    inputs={str(p):sha256_path(p) for p in sources}
    request=dict(count=args.count,max_proposals=args.max_proposals,seed=args.seed,
        config=str(args.config.resolve()),config_sha256=sha256_path(args.config),sources=inputs,
        sim_python=str(Path(args.sim_python).resolve()),cpu_python=str(Path(args.cpu_python).resolve()))
    acceptance=getattr(args,'acceptance_policy','strict-v3')
    request['acceptance_policy']=acceptance
    request['environment_only']=True
    existing=getattr(args,'existing_batch',None)
    if existing is not None:
        existing=existing.resolve()
        request['existing_batch']=str(existing)
        request['candidate_inputs']={n:sha256_path(existing/n) for n in ('plan.json','records.json','summary.json')}
    with (root/'.production.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request_path=root/'production_request.json'
        if request_path.exists() and json.loads(request_path.read_text())!=request:
            raise ValueError('PRODUCTION_RESUME_INPUT_CONFLICT')
        atomic_write_json(request_path,request)
        args._may_publish_failure=True
        archive=root/'production_sources'
        for name,digest in inputs.items():
            src=Path(name);dst=archive/src.relative_to(REPO);dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() and sha256_path(dst)!=digest: raise ValueError('PRODUCTION_ARCHIVE_CONFLICT')
            if not dst.exists():dst.write_bytes(src.read_bytes())
        if existing is not None:
            plan=json.loads((existing/'plan.json').read_text())
            summary=json.loads((existing/'summary.json').read_text())
            _check_inputs(plan['inputs'],completed_root=existing)
            validate_completed(plan,existing,summary)
            if summary['generated']!=args.count:raise ValueError('EXISTING_CANDIDATE_COUNT_MISMATCH')
            batch=root/'batch'
            if batch.exists():
                if batch.resolve()!=existing:raise ValueError('EXISTING_BATCH_LINK_CONFLICT')
            else:batch.symlink_to(existing,target_is_directory=True)
            old_exports=existing.parent/'exports'
            if old_exports.exists() and not (root/'exports').exists():
                (root/'exports').symlink_to(old_exports.resolve(),target_is_directory=True)
            report=dict(status='validating_existing_candidates',requested_candidates=args.count,
                candidate_count=args.count,evaluation_ready_episodes=0,generated_new_floorplans=0,
                environment_only=True,active_acceptance_policy=acceptance,source_batch=str(existing))
            atomic_write_json(root/'production_status.json',report)
            with (root/'validation.log').open('a') as log:
                code=subprocess.run([args.cpu_python,runner,'batch-probe','--batch',str(batch),
                    '--limit',str(args.count),'--gpus',*[str(g) for g in args.gpus],
                    '--sim-python',args.sim_python,'--background-stability-profile','observed-motion-v3',
                    '--verify-background-restore','--acceptance-policy',acceptance],cwd=REPO,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            reports=[json.loads(p.read_text()) for p in batch.glob('gpu_summary_*.json')]
            reports=[r for r in reports if r['validation'].get('acceptance_policy','strict-v3')==acceptance and r['validation'].get('environment_only') is True
                     and r['validation']['sources'].get('scripts/tasks/behavior_pipeline.py')==sha256_path(runner)]
            if code or len(reports)!=1:raise ValueError('EXISTING_BATCH_VALIDATION_FAILED_OR_AMBIGUOUS')
            gpu=reports[0];passed=gpu['status_counts'].get('environment_ready',0)
            complete=gpu['probed']==args.count
            report.update(status=('completed' if passed==args.count else 'completed_with_rejections') if complete else 'incomplete',
                validated=gpu['probed'],environment_ready=passed,environment_needs_review=gpu['probed']-passed,
                active_validation_id=gpu['validation_id'],updated_unix=time.time())
            atomic_write_json(root/'production_status.json',report)
            return 0 if complete and passed==args.count else 2
        preflight=make_plan(args.config,assets,REPO,args.count,args.seed,args.max_proposals)
        atomic_write_json(root/'preflight.json',preflight)
        report=dict(status='exporting_geometry',requested_candidates=args.count,
            evaluation_ready_episodes=0,generated_new_floorplans=0,original_house_templates=len(preflight['homes']),
            rejected_homes=preflight['rejected_homes'],candidate_count=0,environment_ready=0,environment_needs_review=0,
            environment_only=True,active_acceptance_policy=acceptance)
        def publish(**updates):
            report.update(updates,updated_unix=time.time())
            atomic_write_json(root/'production_status.json',report)
            print(json.dumps(report,ensure_ascii=False),flush=True)
        publish()
        pool={a['category']+'/'+a['model']:a for p in cfg['daily_pools'] for a in p}
        def exports(gpu,homes):
            result={}
            for home in homes:
                folder=root/'exports'/home['scene'];folder.mkdir(parents=True,exist_ok=True)
                config=folder/'config.yaml';desired=dict(scene=home['scene'],room=cfg['room'],daily_assets=list(pool.values()))
                if config.exists() and yaml.safe_load(config.read_text())!=desired:raise ValueError('EXPORT_CONFIG_CONFLICT')
                config.write_text(yaml.safe_dump(desired))
                output=folder/'live';receipt=output/'process.json'
                if not receipt.exists():
                    if output.exists() and any(output.iterdir()):raise ValueError('INTERRUPTED_EXPORT_REQUIRES_REVIEW:'+str(output))
                    with (folder/'export.log').open('w') as log:
                        subprocess.run([args.sim_python,runner,'export-geometry','--config',str(config),
                            '--seed','127','--gpu',str(gpu),'--out',str(output)],cwd=REPO,env=env,
                            stdout=log,stderr=subprocess.STDOUT,timeout=960,check=False)
                process=json.loads(receipt.read_text())
                if (process['status']!='geometry_exported_pending_physics' or process['returncode']!=0
                        or process['config_sha256']!=sha256_path(config)):
                    raise ValueError('NATIVE_GEOMETRY_EXPORT_FAILED:'+home['scene'])
                path=output/'collision_geometry.json';load_geometry(path,scene=home['scene'])
                result[home['scene']]=str(path)
            return result
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool_executor:
            futures=[pool_executor.submit(exports,gpu,preflight['homes'][i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
            geometries={k:v for future in futures for k,v in future.result().items()}
        bound=dict(cfg,scenes=[h['scene'] for h in preflight['homes']],require_geometry=True,
                   geometry_by_scene=geometries,solver_timeout=900)
        bound_path=root/'bound_config.yaml'
        if bound_path.exists() and yaml.safe_load(bound_path.read_text())!=bound:raise ValueError('BOUND_CONFIG_CONFLICT')
        bound_path.write_text(yaml.safe_dump(bound,sort_keys=False))
        batch=root/'batch';publish(status='generating_and_validating',geometry_exports=len(geometries))
        command=[args.cpu_python,runner,'batch-home','--config',str(bound_path),'--count',str(args.count),
            '--max-proposals',str(args.max_proposals),'--seed',str(args.seed),'--workers',str(args.workers),'--out',str(batch)]
        with (root/'generation.log').open('a') as log:
            generator=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            last_count=-1;probe_code=None
            try:
                while True:
                    summary_path=batch/'summary.json'
                    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
                    count=summary.get('generated',0)
                    if count and (count!=last_count or generator.poll() is not None):
                        with (root/'validation.log').open('a') as validation_log:
                            probe_code=subprocess.run([args.cpu_python,runner,'batch-probe','--batch',str(batch),
                                '--limit',str(count),'--gpus',*[str(g) for g in args.gpus],
                                '--sim-python',args.sim_python,'--background-stability-profile','observed-motion-v3',
                                '--verify-background-restore','--acceptance-policy',acceptance],cwd=REPO,env=env,stdout=validation_log,
                                stderr=subprocess.STDOUT,check=False).returncode
                        if probe_code:raise ValueError('BATCH_VALIDATION_COORDINATOR_FAILED')
                        last_count=count
                    reports=[p for p in batch.glob('gpu_summary_*.json') if
                        json.loads(p.read_text())['validation'].get('acceptance_policy','strict-v3')==acceptance and
                        json.loads(p.read_text())['validation'].get('environment_only') is True]
                    if len(reports)>1:raise ValueError('MULTIPLE_VALIDATION_PROFILES_IN_PRODUCTION')
                    gpu=json.loads(reports[0].read_text()) if reports else {}
                    counts=gpu.get('status_counts',{})
                    passed=counts.get('environment_ready',0)
                    publish(candidate_count=count,cpu_attempted=summary.get('attempted',0),
                        cpu_failures=summary.get('failures',{}),validated=gpu.get('probed',0),
                        environment_ready=passed,environment_needs_review=gpu.get('probed',0)-passed,
                        validation_status_counts=counts,
                        failure_reasons=dict(Counter(r.get('error') or r['status'] for r in gpu.get('records',[])
                            if r['status']!='environment_ready')))
                    if generator.poll() is not None:
                        latest=json.loads(summary_path.read_text()) if summary_path.exists() else {}
                        if latest.get('generated',0)!=count:continue
                        complete=generator.returncode==0 and count==args.count and gpu.get('probed',0)==count
                        publish(status=('completed' if passed==args.count else 'completed_with_rejections') if complete else 'incomplete',
                                generation_returncode=generator.returncode)
                        return 0 if complete and passed==args.count else 2
                    time.sleep(20)
            finally:
                # This coordinator owns this child only; never terminate a
                # shared simulator or another user's GPU process.
                if generator.poll() is None:
                    os.killpg(generator.pid,signal.SIGTERM)
                    try:generator.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(generator.pid,signal.SIGKILL);generator.wait(timeout=10)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/layouts/behavior_batch_v1.yaml')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--max-proposals',type=int,default=400)
    parser.add_argument('--seed',type=int,default=20261016)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--existing-batch',type=Path,help='Revalidate a completed candidate batch without regeneration')
    parser.add_argument('--acceptance-policy',choices=['strict-v3','execution-v1'],default='execution-v1')
    parser.add_argument('--gpus',type=int,nargs='+',choices=[4,5],default=[4,5])
    parser.add_argument('--sim-python',default='/data1/ygwang/miniconda3/envs/behavior/bin/python')
    parser.add_argument('--cpu-python',default='/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python')
    arguments=parser.parse_args(argv)
    try:return run(arguments)
    except Exception as exc:
        # A rejected second coordinator or changed resume input must not
        # rewrite the authoritative status of the already-owned run.
        if getattr(arguments,'_may_publish_failure',False):
            path=arguments.out.resolve()/'production_status.json'
            prior=json.loads(path.read_text()) if path.exists() else {}
            atomic_write_json(path,dict(prior,status='failed',error=f'{type(exc).__name__}:{exc}',evaluation_ready_episodes=0))
        raise


if __name__=='__main__':
    raise SystemExit(main())
