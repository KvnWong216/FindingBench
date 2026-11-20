"""CPU contract checks; none of these fixtures certify a simulator episode."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_new_optional_authoring_hint_preserves_legacy_frozen_identity():
    from rummagebench.core.scenario import load_scenario
    from rummagebench.authoring.tasks.household_state import scenario_state_identity
    from rummagebench.authoring.tasks.plan import canonical_json,sha256_bytes
    spec=load_scenario(ROOT/'scenarios/asset_search_v1/asset_U017_01/scenario.yaml')
    legacy=spec.model_dump(mode='json');legacy['scene'].pop('frozen_member_manifest',None)
    for placement in legacy['placements']:placement.pop('orientation_hint_xyzw',None)
    assert scenario_state_identity(spec)==sha256_bytes(canonical_json(legacy))


@pytest.mark.parametrize('frozen',[False,True])
def test_clearance_keeps_frozen_recipe_identity_without_skipping_geometry(monkeypatch,frozen):
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson import clearance
    from rummagebench.authoring.tasks.household_state import scenario_state_identity
    spec=load_scenario(ROOT/'scenarios/asset_search_v1/asset_U017_01/scenario.yaml')
    spec.scene.clearance_manifest='unit-clearance.json'
    old=spec.anchors[spec.robot.init_anchor].position[0];identity=scenario_state_identity(spec)
    calls=[]
    def apply(backend,path):
        calls.append(path)
        backend._scenario.anchors[spec.robot.init_anchor].position[0]+=1
        backend.geometry_edited=True
        return {'physical_edit_executed':True}
    monkeypatch.setattr(clearance,'apply_clearance',apply)
    backend=SimpleNamespace(_scenario=spec,geometry_edited=False)
    result=OmniGibsonBackend._apply_declared_clearance(backend,spec,preserve_recipe_identity=frozen)
    assert len(calls)==1 and backend.geometry_edited and result['physical_edit_executed']
    assert backend._scenario is spec
    assert spec.anchors[spec.robot.init_anchor].position[0]==old+(0 if frozen else 1)
    assert (scenario_state_identity(spec)==identity)==frozen


def test_oriented_proposal_retains_every_native_rejection():
    from rummagebench.sim.omnigibson.inside_volume import centered_volume_source,oriented_volume_source
    candidates=list(ROOT.parents[1].glob('BEHAVIOR-1K*/OmniGibson/omnigibson/object_states/inside.py'))
    if not candidates:pytest.skip('local native SDK unavailable')
    source=candidates[0].read_text()
    method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    method_source=ast.get_source_segment(source,method)
    before=ast.parse(centered_volume_source(method_source));after=ast.parse(oriented_volume_source(method_source))
    # Added rotation-aware bounds cannot remove or alter native predicates.
    predicates=lambda tree:[ast.dump(n.test) for n in ast.walk(tree) if isinstance(n,ast.If)]
    assert predicates(before)==predicates(after)
    assert 'th.abs(T.quat2mat(orientation)) @ (self.obj.native_bbox * self.obj.scale)' in ast.unparse(after)


def test_native_bundle_rejects_changed_checkpoint_and_dependency(tmp_path,monkeypatch):
    from rummagebench.authoring.tasks.household_state import export_native_member,load_frozen_member
    from rummagebench.sim.omnigibson import state_io
    class Scenario:
        id='unit_native'
        scene=SimpleNamespace(source='behavior',clearance_manifest=None)
        def model_dump(self,**kw):return {'id':self.id,'scene':{'source':'behavior','model':'unit'}}
    spec=Scenario();directory=tmp_path/'unit_native';dependency=tmp_path/'runtime.py';dependency.write_text('original')
    backend=SimpleNamespace(_sim=None,_env=SimpleNamespace(scene=SimpleNamespace(objects=[])))
    monkeypatch.setattr(state_io,'dump_state',lambda sim:'unit-state')
    monkeypatch.setattr(state_io,'save_snapshot',lambda path,state,meta:Path(path).write_text(json.dumps([state,meta])))
    manifest=export_native_member(backend,spec,directory,{},
        {str(dependency):hashlib.sha256(dependency.read_bytes()).hexdigest()})
    assert load_frozen_member(manifest,spec)['version']==3
    dependency.write_text('changed')
    with pytest.raises(ValueError,match='DEPENDENCY_CHANGED'):load_frozen_member(manifest,spec)
    dependency.write_text('original')
    (directory/'initial_state.private.pt').write_text('corrupt')
    with pytest.raises(ValueError,match='FILE_MISSING_OR_CHANGED'):load_frozen_member(manifest,spec)


def test_old_success_does_not_publish_changed_recipe(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import run_asset_pipeline as pipeline
    monkeypatch.setattr(pipeline.batch,'runtime_sources',lambda:{'source':'hash'})
    path=tmp_path/'scenario.yaml';path.write_text('original')
    task={'scenario':str(path)}
    report={'status':'initialization_validated','fresh_native_setup':True,
        'runtime_sources':{'source':'hash'},'scenario_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert pipeline.current(task,report,'initialization_validated')
    assert not pipeline.current(task,report,'restore_validated')
    path.write_text('changed goal')
    assert not pipeline.current(task,report,'initialization_validated')


def test_panorama_export_survives_adapter_preparation(tmp_path,monkeypatch):
    from rummagebench.core.scenario import load_scenario
    from rummagebench.adapters.python_api import prepare_renderer_grounding
    from rummagebench.authoring.tasks.household_state import export_native_member,load_frozen_member
    from rummagebench.sim.omnigibson import state_io
    spec=load_scenario(ROOT/'scenarios/asset_search_v1/asset_U017_01/scenario.yaml')
    prepare_renderer_grounding(spec)
    backend=SimpleNamespace(_sim=None,_env=SimpleNamespace(scene=SimpleNamespace(objects=[])))
    monkeypatch.setattr(state_io,'dump_state',lambda sim:'unit-state')
    monkeypatch.setattr(state_io,'save_snapshot',lambda path,state,meta:Path(path).write_text(json.dumps([state,meta])))
    dependency=tmp_path/'code.py';dependency.write_text('unit')
    manifest=export_native_member(backend,spec,tmp_path/'episode',{},
        {str(dependency):hashlib.sha256(dependency.read_bytes()).hexdigest()})
    replay=load_scenario(tmp_path/'episode/play.yaml')
    prepare_renderer_grounding(replay)
    assert load_frozen_member(manifest,replay)['scenario_id']==spec.id


def test_single_task_catalog_preserves_recipe_and_excludes_other_tasks(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    monkeypatch.setattr(batch,'OUT',tmp_path/'results')
    for name in ('one','two'):
        folder=tmp_path/'recipes'/name;folder.mkdir(parents=True)
        (folder/'scenario.yaml').write_text('id: '+name+'\n')
    task={'id':'one','scenario':'recipes/one/scenario.yaml'}
    catalog=batch.single_task_catalog(task)
    paths=list(catalog.glob('*/scenario.yaml'))
    assert len(paths)==1
    assert paths[0].resolve()==tmp_path/'recipes/one/scenario.yaml'
    assert paths[0].read_bytes()==(tmp_path/task['scenario']).read_bytes()


def test_rolling_refills_slot_before_slow_peer_finishes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from rolling_asset_jobs import rolling_map
    import threading
    replacement=threading.Event();lock=threading.Lock();active=0;peak=0
    def work(item):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:
            if item==0:assert replacement.wait(2),'fast slot was not refilled'
            if item==2:replacement.set()
        finally:
            with lock:active-=1
    rolling_map([0,1,2],work,capacity=lambda:2,weight=lambda item:1,
                stopped=lambda:False,tick=lambda:None,max_workers=3)
    assert replacement.is_set() and peak<=2


@pytest.mark.parametrize('isolated_runtime',[False,True])
def test_parallel_native_modes_and_seeds_do_not_leak(tmp_path,monkeypatch,isolated_runtime):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    import threading
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(batch,'ROOT',tmp_path);monkeypatch.setattr(batch,'OUT',tmp_path/'out')
    monkeypatch.setattr(batch,'runtime_sources',lambda root=None:{})
    monkeypatch.setattr(batch,'publish',lambda:None)
    monkeypatch.setattr(batch,'STOP',threading.Event());monkeypatch.setattr(batch,'ACTIVE',{})
    barrier=threading.Barrier(2);commands=[]
    class Process:
        def __init__(self,argv,**kwargs):
            self.argv=argv;self.pid=100+len(commands);commands.append((argv,kwargs))
        def wait(self,timeout):
            barrier.wait(timeout=2)
            tid=self.argv[self.argv.index('--task')+1]
            restore='--restore' in self.argv
            name='restore_validation.json' if restore else 'validation.json'
            report=batch.OUT/tid/name
            report.write_text(json.dumps({'status':'restore_validated' if restore else 'initialization_validated'}))
            # The fake process finishes within one filesystem clock tick;
            # model a report written after launch, as the native job does.
            import os,time
            stamp=time.time()+1
            os.utime(report,(stamp,stamp))
            return 0
    monkeypatch.setattr(batch.subprocess,'Popen',Process)
    tasks=[]
    for name in ('build','reload'):
        p=tmp_path/name;p.mkdir();(p/'scenario.yaml').write_text('id: '+name)
        tasks.append(dict(id=name,scenario=str(p/'scenario.yaml')))
        if isolated_runtime:
            runtime=tmp_path/('runtime_'+name);runtime.mkdir()
            tasks[-1].update(runtime_root=str(runtime),runtime_environment={'YGBENCH_GPU_PROPOSALS':'1' if name=='build' else '0'})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(batch.run,tasks[0],10,restore=False,seed=5),
                 pool.submit(batch.run,tasks[1],10,restore=True,seed=7)]
        assert {f.result()['status'] for f in futures}=={'initialization_validated','restore_validated'}
    for argv,kw in commands:
        tid=argv[argv.index('--task')+1]
        assert ('--restore' in argv)==(tid=='reload')
        assert argv[argv.index('--seed')+1]==('7' if tid=='reload' else '5')
        runtime=tmp_path/('runtime_'+tid) if isolated_runtime else tmp_path
        assert kw['env']['PYTHONPATH'].startswith(str(runtime/'src'))
        assert kw['cwd']==runtime
        assert str(runtime/'scripts/v2/validate_dense_pilot.py') in argv
        if isolated_runtime:
            assert kw['env']['YGBENCH_GPU_PROPOSALS']==('1' if tid=='build' else '0')
            report=batch.OUT/tid/('validation.json' if tid=='build' else 'restore_validation.json')
            assert json.loads(report.read_text())['runtime_environment']=={'YGBENCH_GPU_PROPOSALS':kw['env']['YGBENCH_GPU_PROPOSALS']}
        assert kw['env']['TMPDIR']==kw['env']['TEMP']==kw['env']['TMP']
        assert not Path(kw['env']['TMPDIR']).exists()


def test_weighted_queue_backfills_small_task(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from rolling_asset_jobs import rolling_map
    import threading
    filled=threading.Event()
    def work(item):
        if item=='large1':assert filled.wait(2)
        if item=='small':filled.set()
    rolling_map(['large1','large2','small'],work,capacity=lambda:3,
                weight=lambda item:1 if item=='small' else 2,
                stopped=lambda:False,tick=lambda:None,max_workers=3)
    assert filled.is_set()


def test_runtime_hash_cache_invalidates_edits(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    monkeypatch.setattr(batch,'RUNTIME_FILES',('runtime.py',))
    p=tmp_path/'runtime.py';p.write_text('old')
    first=batch.runtime_sources()
    assert batch.runtime_sources()==first
    p.write_text('new implementation')
    assert batch.runtime_sources()!=first


@pytest.mark.parametrize('restore_after_build',[False,True])
@pytest.mark.parametrize('preexisting',[False,True])
def test_first_pass_builds_all_before_restore(tmp_path,monkeypatch,restore_after_build,preexisting):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import rolling_asset_jobs as rolling
    import threading
    events=[];guard=threading.Lock()
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    batch=SimpleNamespace(ROOT=tmp_path,STOP=threading.Event(),LOCK=threading.RLock(),publish=lambda:None)
    def native(task,timeout,restore=False,seed=0):
        dest=tmp_path/task['id'];dest.mkdir(exist_ok=True)
        with guard:
            if restore and not restore_after_build:assert ({'b'} if preexisting else {'a','b'})<={i for phase,i in events if phase=='build'}
            events.append(('restore' if restore else 'build',task['id']))
        (dest/'frozen_member.private.json').write_text('{}')
        status='restore_validated' if restore else 'initialization_validated'
        write(dest/('restore_validation.json' if restore else 'validation.json'),{'status':status})
        return {'id':task['id'],'status':status}
    batch.run=native
    pipeline=SimpleNamespace(batch=batch,STOP=False,read=read,write=write,
        current=lambda task,report,status:report.get('status')==status,
        fingerprint=lambda task:task['id'],restored=lambda task,out:False,export=lambda *a:None)
    tasks=[{'id':i,'scene':'scene'} for i in ('a','b')]
    if preexisting:
        write(tmp_path/'a'/'validation.json',{'status':'initialization_validated'})
        write(tmp_path/'a'/'frozen_member.private.json',{})
    rolling.run(pipeline,tasks,tmp_path,tmp_path,tmp_path,{},1,10,lambda:1,restore_after_build=restore_after_build)
    assert len(events)==(3 if preexisting else 4)
    if preexisting:
        assert events[0]==(('restore','a') if restore_after_build else ('build','b'))
    else:
        assert events[1][0]==('restore' if restore_after_build else 'build')


def test_interrupted_native_report_is_not_a_success(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    p=tmp_path/'validation.json';p.write_text('{"status":')
    assert batch.read_native_report(p)=={}
    p.write_text('{"status":"failed"}')
    assert batch.read_native_report(p)=={'status':'failed'}


def test_native_resource_budget(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    env=batch.native_environment({'OMP_NUM_THREADS':'20','OPENBLAS_NUM_THREADS':'20'})
    assert env['OMP_NUM_THREADS']==env['OPENBLAS_NUM_THREADS']=='2'
    assert env['TORCHINDUCTOR_COMPILE_THREADS']=='1'
    assert batch.native_environment({'YGBENCH_CPU_THREADS':'3'})['MKL_NUM_THREADS']=='3'
    with pytest.raises(ValueError):batch.native_environment({'YGBENCH_CPU_THREADS':'0'})


def test_compiler_cache_survives_per_job_temp_directories(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    base={'YGBENCH_COMPILE_CACHE_ROOT':str(tmp_path/'persistent')}
    first=batch.native_environment(dict(base,TMPDIR=str(tmp_path/'job1')))
    second=batch.native_environment(dict(base,TMPDIR=str(tmp_path/'job2')))
    for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):
        assert first[key]==second[key]
        assert Path(first[key]).is_relative_to(tmp_path/'persistent')
    explicit=batch.native_environment(dict(base,TORCHINDUCTOR_CACHE_DIR='/explicit/inductor'))
    assert explicit['TORCHINDUCTOR_CACHE_DIR']=='/explicit/inductor'


def test_native_startup_discovers_complete_dataset_and_keeps_user_override(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    runtime=tmp_path/'workspace/build/pinned'
    data=tmp_path/'behavior-datasets'
    (data/'behavior-1k-assets').mkdir(parents=True)
    assert 'OMNIGIBSON_DATA_PATH' not in batch.native_environment({},runtime)
    (data/'omnigibson-robot-assets').mkdir()
    assert batch.native_environment({},runtime)['OMNIGIBSON_DATA_PATH']==str(data)
    assert batch.native_environment({'OMNIGIBSON_DATA_PATH':'/explicit/data'},runtime)['OMNIGIBSON_DATA_PATH']=='/explicit/data'


def test_cached_metadata_invalidates_and_does_not_share_mutations(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    path=tmp_path/'metadata.json';path.write_text('{"a":1}')
    first=batch.cached_file(path,digest=True)
    value=batch.cached_file(path);value['a']=9
    assert batch.cached_file(path)=={'a':1}
    path.write_text('{"a":2}')
    import os,time
    stamp=time.time()+1;os.utime(path,(stamp,stamp))
    assert batch.cached_file(path,digest=True)!=first
    assert batch.cached_file(path)=={'a':2}


def test_large_scenes_use_idle_capacity_only_when_memory_allows(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from rolling_asset_jobs import scene_weight
    assert scene_weight(500,40*1024**3)==1
    assert scene_weight(500,12*1024**3)==2
    assert scene_weight(500,None)==2
    assert scene_weight(100,12*1024**3)==1


def test_three_large_scenes_can_start_together_with_memory_headroom(monkeypatch):
    import threading
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from rolling_asset_jobs import rolling_map,scene_weight
    barrier=threading.Barrier(3)
    rolling_map(['a','b','c'],lambda item:barrier.wait(timeout=2),capacity=lambda:3,
                weight=lambda item:scene_weight(500,40*1024**3),
                stopped=lambda:False,tick=lambda:None,max_workers=3)


def test_throughput_summary_charges_failed_jobs_and_best_baseline(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from benchmark_generation import summarize
    trials=[dict(variant='cpu',wall_seconds=100,jobs=[{'accepted':True},{'accepted':True}]),
            dict(variant='gpu',wall_seconds=200,jobs=[{'accepted':True},{'accepted':False}]),
            dict(variant='mesh_cache',wall_seconds=10,jobs=[{'accepted':True},{'accepted':False}])]
    result=summarize(trials,2)
    assert result['totals']['gpu']['accepted_per_hour']==18
    assert result['speedup_vs_best_baseline']==5
    assert not result['achieved_10x']
    trials[-1]['jobs'][1]['accepted']=True
    assert summarize(trials,2)['achieved_10x']
    assert not summarize(trials[:2],2)['complete']


def test_throughput_acceptance_requires_matching_fresh_reload(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from benchmark_generation import accepted_pair
    common=dict(fresh_native_setup=True,scenario_sha256='recipe',runtime_sources={'file':'hash'},seed=2)
    build=dict(common,status='initialization_validated')
    restore=dict(common,status='restore_validated',frozen_manifest_sha256='frozen')
    assert accepted_pair(build,restore,'recipe',{'file':'hash'},'frozen',2)
    for changed in [dict(fresh_native_setup=False),dict(seed=1),dict(frozen_manifest_sha256='stale'),dict(runtime_sources={})]:
        assert not accepted_pair(build,{**restore,**changed},'recipe',{'file':'hash'},'frozen',2)
    flags={'YGBENCH_BATCH_POSE_WRITES':'1'}
    assert not accepted_pair(build,restore,'recipe',{'file':'hash'},'frozen',2,flags)
    build['runtime_environment']=dict(flags);restore['runtime_environment']=dict(flags)
    assert accepted_pair(build,restore,'recipe',{'file':'hash'},'frozen',2,flags)
    restore['runtime_environment']['YGBENCH_BATCH_POSE_WRITES']='0'
    assert not accepted_pair(build,restore,'recipe',{'file':'hash'},'frozen',2,flags)


def test_capability_preflight_is_asset_hash_pinned(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    monkeypatch.setattr(batch,'ASSET_OBJECTS',tmp_path)
    p=tmp_path/'nightstand/uudzva/usd/uudzva.encrypted.usd';p.parent.mkdir(parents=True);p.write_bytes(b'observed asset')
    monkeypatch.setattr(batch,'CAPABILITY_REJECTIONS',{('nightstand','uudzva','open'):hashlib.sha256(p.read_bytes()).hexdigest()})
    spec=dict(objects=[dict(name='cabinet',category='nightstand',model='uudzva')],initial_states={'cabinet':{'open':False}})
    assert len(batch.capability_issues(spec))==1
    p.write_bytes(b'changed native asset')
    assert batch.capability_issues(spec)==[]
    spec['initial_states']={}
    assert batch.capability_issues(spec)==[]


def test_native_usage_records_per_child_including_failure(monkeypatch,tmp_path):
    import subprocess,sys
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import validate_dense_batch as batch
    if not Path('/usr/bin/time').exists():pytest.skip('GNU time unavailable')
    path=tmp_path/'usage.jsonl'
    for code in (0,3):
        result=subprocess.run(batch.measured_command([sys.executable,'-c',f'import sys; x=sum(range(100000)); sys.exit({code})'],path))
        assert result.returncode==code
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    assert [row['exit_code'] for row in rows]==[0,3]
    assert all(row['max_rss_kib']>0 and row['user_cpu_seconds']>=0 and row['system_cpu_seconds']>=0 for row in rows)


def test_late_stop_does_not_relabel_completed_native_failure(monkeypatch):
    import signal
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from validate_dense_batch import missing_report_status
    assert missing_report_status(False,True,0)[0]=='infrastructure_failure'
    assert missing_report_status(False,True,-signal.SIGSEGV)[0]=='infrastructure_failure'
    assert missing_report_status(False,True,-signal.SIGTERM)[0]=='authoring_interrupted'
    assert missing_report_status(True,True,-signal.SIGTERM)[0]=='initialization_timeout'


def test_new_runtime_assignment_preserves_started_tasks_and_pins_evidence(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    import run_asset_pipeline as pipeline
    import hashlib
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline.batch,'runtime_sources',lambda root=None:{'file':str(root or 'legacy')})
    out=tmp_path/'out';out.mkdir();runtime=tmp_path/'fast';runtime.mkdir()
    (runtime/'release.json').write_text(json.dumps({'enable':{'YGBENCH_MESH_ACCELERATION':'1',
        'YGBENCH_NATIVE_MOTION_CONVERGENCE':'1','YGBENCH_MEASURED_SURFACE_LAYOUT':'1'}}))
    tasks=[]
    for name in ['existing','attempted','new']:
        recipe=tmp_path/(name+'.yaml');recipe.write_text('id: '+name)
        tasks.append({'id':name,'scenario':str(recipe)})
    (out/'existing').mkdir()
    legacy=pipeline.fingerprint(tasks[0])
    expected=hashlib.sha256(json.dumps(dict(recipe=hashlib.sha256(Path(tasks[0]['scenario']).read_bytes()).hexdigest(),runtime={'file':'legacy'}),sort_keys=True).encode()).hexdigest()
    assert legacy==expected
    mapping=pipeline.assign_new_runtime(tasks,out,runtime,{'attempted:hash:0':{}})
    assert set(mapping)=={'new'}
    attached=pipeline.apply_runtime_assignments([dict(t) for t in tasks],out)
    assert 'runtime_root' not in attached[0] and 'runtime_root' not in attached[1]
    assert pipeline.fingerprint(attached[0])==legacy
    assert attached[2]['runtime_environment']['YGBENCH_GPU_OCCUPANCY_RANKING']=='0'
    assert attached[2]['runtime_environment']['YGBENCH_NATIVE_MOTION_CONVERGENCE']=='1'
    assert attached[2]['runtime_environment']['YGBENCH_MEASURED_SURFACE_LAYOUT']=='1'
    assert pipeline.fingerprint(attached[2])!=pipeline.fingerprint(tasks[2])
    task=attached[2];report=dict(status='initialization_validated',fresh_native_setup=True,
        scenario_sha256=mapping['new']['scenario_sha256'],runtime_sources=mapping['new']['runtime_sources'],
        runtime_environment=mapping['new']['runtime_environment'])
    assert pipeline.current(task,report,'initialization_validated')
    assert not pipeline.current(task,{**report,'runtime_environment':{}},'initialization_validated')
    Path(task['scenario']).write_text('changed')
    with pytest.raises(ValueError,match='Pinned recipe changed'):
        pipeline.apply_runtime_assignments([dict(t) for t in tasks],out)


def test_graceful_drain_finishes_active_work_without_starting_pending(monkeypatch):
    import threading
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from rolling_asset_jobs import rolling_map
    requested=threading.Event();started=[];finished=[]
    def work(item):
        started.append(item);requested.set();finished.append(item)
    drained=rolling_map([1,2,3],work,capacity=lambda:1,weight=lambda _:1,
        stopped=lambda:False,tick=lambda:None,max_workers=1,draining=requested.is_set)
    assert drained and started==finished==[1]
    started.clear()
    assert rolling_map([1],work,capacity=lambda:1,weight=lambda _:1,
        stopped=lambda:False,tick=lambda:None,draining=requested.is_set)
    assert not started


def test_live_worker_budget_fails_closed(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v2'))
    from run_asset_pipeline import worker_limit
    assert worker_limit(tmp_path,2)==2
    path=tmp_path/'worker_budget.json';path.write_text('{"workers":1}')
    assert worker_limit(tmp_path,2)==1
    for value in (0,3,True,None):
        path.write_text(json.dumps({'workers':value}))
        with pytest.raises(ValueError):worker_limit(tmp_path,2)
    path.write_text('{')
    with pytest.raises(ValueError):worker_limit(tmp_path,2)


def test_tracker_requires_matching_native_build_and_frozen_restore(tmp_path,monkeypatch):
    import json,hashlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v2'))
    from track_full_asset_queue import accepted_native_pair
    recipe=tmp_path/'scenario.yaml';recipe.write_text('id: example\n')
    state=tmp_path/'state.pt';state.write_bytes(b'original state')
    ref=tmp_path.name+'/state.pt'
    frozen=tmp_path/'frozen_member.private.json';frozen.write_text(json.dumps({'state':ref,'artifacts':{ref:hashlib.sha256(state.read_bytes()).hexdigest()}}))
    build=dict(status='initialization_validated',fresh_native_setup=True,seed=0,
        scenario_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),runtime_sources={'backend.py':'hash'},
        runtime_environment={'YGBENCH_GPU_PROPOSALS':'1'})
    restore=dict(build,status='restore_validated',frozen_manifest_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
    (tmp_path/'validation.json').write_text(json.dumps(build))
    def save(value): (tmp_path/'restore_validation.json').write_text(json.dumps(value))
    save(restore);assert accepted_native_pair(recipe,tmp_path)
    save(dict(restore,seed=1));assert accepted_native_pair(recipe,tmp_path)
    for key,value in [('fresh_native_setup',False),('runtime_sources',{'backend.py':'changed'}),
                      ('runtime_environment',{}),('frozen_manifest_sha256','old'),('scenario_sha256','old')]:
        save(dict(restore,**{key:value}));assert not accepted_native_pair(recipe,tmp_path)
    save(restore);state.write_bytes(b'tampered state')
    assert not accepted_native_pair(recipe,tmp_path)
    state.write_bytes(b'original state')
    assert accepted_native_pair(recipe,tmp_path)
    # Even an intact state plus matching initialization/reload reports is not
    # publishable if a native manifest's code dependency has since changed.
    dependency=tmp_path/'backend.py';dependency.write_text('frozen implementation')
    native=dict(version=3,state=ref,artifacts={ref:hashlib.sha256(state.read_bytes()).hexdigest()},
                dependencies={str(dependency):hashlib.sha256(dependency.read_bytes()).hexdigest()})
    payload=json.dumps(native,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    native['manifest_sha256']=hashlib.sha256(payload).hexdigest();frozen.write_text(json.dumps(native))
    restore['frozen_manifest_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest();save(restore)
    assert accepted_native_pair(recipe,tmp_path)
    dependency.write_text('new working-copy implementation')
    assert not accepted_native_pair(recipe,tmp_path)
    dependency.write_text('frozen implementation')
    assert accepted_native_pair(recipe,tmp_path)
    native['manifest_sha256']='tampered';frozen.write_text(json.dumps(native))
    restore['frozen_manifest_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest();save(restore)
    assert not accepted_native_pair(recipe,tmp_path)
    save(restore);frozen.write_text('{"state":"replaced"}')
    assert not accepted_native_pair(recipe,tmp_path)
    (tmp_path/'restore_validation.json').write_text('{')
    assert not accepted_native_pair(recipe,tmp_path)
