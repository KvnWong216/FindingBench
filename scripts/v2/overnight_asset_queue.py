"""Serial generation plus evidence-directed repairs of every unaccepted task.

All candidates retain independent native initialization/reload acceptance.
No VLM acts here. Expert curation is explicit, bounded rule-based authoring;
unresolved cases stay unresolved with their evidence, never counted as ready.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,runpy,select,signal,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/ubuntu/miniforge3/envs/ygbench/bin/python'


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def wait_external(pid):
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    try:select.select([fd],[],[])
    finally:os.close(fd)


def storage_scope_text(instruction):
    """Separate search scope from a reference sentence and delivery clause."""
    text=str(instruction)
    find=re.search(r'\b(?:find|locate|look\s+for|search\s+for)\b',text,re.I)
    if find and re.search(r'\b(?:sample|reference)\b',text[:find.start()],re.I):text=text[find.start():]
    return re.split(r'\b(?:and|then)\s+(?:bring|put|set|place|deliver|leave|return|carry|take)\b',text,maxsplit=1,flags=re.I)[0]


def has_qualified_storage_location(instruction):
    """A destination cabinet top does not constrain the searched drawer."""
    text=storage_scope_text(instruction)
    fixture=r'(?:drawer|drawers|cupboard|cupboards|cabinet|cabinets|shelf|shelves|basket|baskets)'
    qualifier=r'(?:upper|lower|left|right|middle|top|bottom|first|second|third|topmost|bottommost|leftmost|rightmost)'
    return bool(re.search(r'\b'+qualifier+r'\s+(?:[\w-]+\s+){0,2}'+fixture+r'\b',text,re.I)
                or re.search(r'\b'+fixture+r'\s+(?:on|to)\s+(?:the\s+)?(?:left|right)\b',text,re.I))


def native_policy_signature(sources):
    keys=['src/rummagebench/sim/omnigibson/backend.py','src/rummagebench/sim/omnigibson/inside_volume.py']
    if not all(sources.get(k) for k in keys):return None
    return hashlib.sha256(json.dumps({k:sources[k] for k in keys},sort_keys=True).encode()).hexdigest()


def should_curate_first(error):
    return any(s in str(error) for s in ('placement failed','LAYOUT_NOT_CONVERGED','no Open state','capability mismatch','native_timeout',
                                       'EXPERT_LAYOUT','EXPERT_SUPPORT','EXPERT_CONTAINER'))


def infrastructure_error(error):
    return any(s in str(error) for s in ('has no attribute','native_process_exited_without_current_report',
                                        'INVALID_EXPERT_LAYOUT_CANDIDATE',
                                        'FROZEN_MEMBER_SCENARIO_OR_HASH_MISMATCH',
                                        'CUDA out of memory','No space left on device',
                                        'method not intended for entity prim with no joints'))


class Queue:
    def __init__(self,args):
        self.args=args;self.out=args.output.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.cat=ROOT/'scenarios'/self.out.name;self.runtime=ROOT/'build/generator_runtime_expert_layout_v3_20261109'
        self.env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMNIGIBSON_DATA_PATH='/home/ubuntu/ygwang/behavior-datasets',
            CXX='/home/ubuntu/miniforge3/envs/ygbench/bin/x86_64-conda-linux-gnu-c++',YGBENCH_CPU_THREADS='2',YGBENCH_COMPILE_THREADS='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        self.check=runpy.run_path(str(ROOT/'scripts/v2/track_full_asset_queue.py'))['reviewable_native_pair']
        self.helper=None;self.serial=None;self.replaced=False;self.helper_log=None
        self.child=None;self.stop_requested=False
        self.ledger=read(self.out/'jobs.json');self.latest=read(self.out/'latest_cases.json')
        self.exclusions=read(self.out/'exclude_sources.json') or read(ROOT/'runs/serial_repairs_20261109/exclude_sources.json')

    def status(self,**values):
        write(self.out/'controller_status.json',dict(pid=os.getpid(),updated=time.time(),**values))

    def helper_revision(self):
        paths=['scripts/v2/expert_storage_repair.py','scripts/v2/overnight_asset_queue.py','scripts/v2/constructed_storage.py',
               'scripts/v2/audit_dense_layouts.py','scripts/v2/generate_dense_pilot.py',
               'src/rummagebench/sim/omnigibson/inside_volume.py','build/native_fillable_geometry/index.json',
               'build/expert_editorial_revisions/index.json','build/expert_support_overrides/index.json','build/expert_scene_contact_repairs/index.json']
        return tuple((p,(ROOT/p).stat().st_mtime_ns,(ROOT/p).stat().st_size) for p in paths if (ROOT/p).exists())

    def rpc(self,request):
        if self.helper_revision()!=getattr(self,'helper_source_revision',None):
            if self.helper and self.helper.poll() is None:self.helper.terminate();self.helper.wait()
            self.start_helper()
        for retry in range(2):
            if self.stop_requested:raise RuntimeError('Queue interrupted')
            try:
                self.helper.stdin.write(json.dumps(request)+'\n');self.helper.stdin.flush()
                if not select.select([self.helper.stdout],[],[],180)[0]:raise TimeoutError('Expert geometry worker did not reply')
                line=self.helper.stdout.readline()
                if not line:raise RuntimeError('Expert geometry worker exited')
                return json.loads(line)
            except (BrokenPipeError,RuntimeError,TimeoutError):
                if retry or self.stop_requested:raise
                if self.helper.poll() is None:self.helper.kill()
                self.helper.wait()
                self.start_helper()

    def start_helper(self):
        self.helper_source_revision=self.helper_revision()
        self.helper=subprocess.Popen([PYTHON,'-u',str(ROOT/'scripts/v2/expert_storage_repair.py'),'--serve'],cwd=ROOT,
            env=self.env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.helper_log,text=True,bufsize=1)

    def stop(self,*args):
        self.stop_requested=True
        if self.child and self.child.poll() is None:
            try:os.killpg(self.child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        if self.helper and self.helper.poll() is None:self.helper.terminate()

    def accepted(self,tid):
        # The tracker verifies original frozen hashes before publishing this file.
        if any(t.get('parent_task_id')==tid for t in read(ROOT/'runs/asset_full_queue_20261105/human_ready.json').get('tasks',[])):return True
        for manifest in self.cat.glob('*/manifest.json'):
            for row in read(manifest).get('tasks',[]):
                if row.get('parent_task_id')==tid and self.check(row['scenario'],self.out/manifest.parent.name/row['id']):return True
        return False

    def command(self,cmd,log,*,stage,tid,timeout=1500):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as output:
            child=subprocess.Popen(cmd,cwd=ROOT,env=self.env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            self.child=child
            self.status(stage=stage,task=tid,child_pid=child.pid,runtime=str(self.runtime))
            try:return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                return 124
            finally:self.child=None

    def diagnose(self,row,lane,case):
        directory=self.out/lane/row['id'];a=read(directory/'validation.json');b=read(directory/'restore_validation.json')
        report=b if a.get('status')=='initialization_validated' and b else a
        updated=dict(case,record=row,evidence=str(directory),error=report.get('error',report.get('status','missing_report')))
        self.latest[case['id']]=updated;write(self.out/'latest_cases.json',self.latest)
        # Match the compiler's source signature. This only avoids an observed
        # failed layout for this task; it never globally bans a native asset.
        signature=None if infrastructure_error(updated['error']) else self.rpc(
            dict(action='signature',case=updated,out=str(self.cat/lane))).get('signature')
        if signature and signature not in self.exclusions.setdefault(case['id'],[]):self.exclusions[case['id']].append(signature)
        write(self.out/'exclude_sources.json',self.exclusions)
        return updated

    def validate(self,row,lane,case):
        out=self.out/lane
        override=read(self.out/'runtime_override.json').get('runtime')
        if override:
            candidate=Path(override)
            if not (candidate/'release.json').is_file():raise ValueError('Runtime override has no immutable release')
            self.runtime=candidate
        contract_key=(str(self.runtime),hashlib.sha256((self.runtime/'release.json').read_bytes()).hexdigest())
        if getattr(self,'checked_runtime_contract',None)!=contract_key:
            subprocess.run([PYTHON,str(ROOT/'scripts/v2/check_frozen_layout_runtime.py'),str(self.runtime)],
                           cwd=ROOT,env=self.env,check=True,timeout=60,stdout=subprocess.DEVNULL)
            self.checked_runtime_contract=contract_key
        cmd=['bash','scripts/build_asset_search.sh','--skip-compile','--workers','1','--attempts','1',
             '--timeout',str(self.args.native_timeout),'--restore-after-build','--runtime-root',str(self.runtime),
             '--new-runtime-root',str(self.runtime),'--catalog-root',str(self.cat/lane),'--output-root',str(out)]
        code=self.command(cmd,out/'launcher.log',stage='native_build_and_reload',tid=case['id'])
        self.diagnose(row,lane,case)
        return dict(exit_code=code,accepted=self.check(row['scenario'],out/row['id']))

    def prepare(self,case,lane,action,variant=0):
        runtime=read(self.out/'runtime_override.json').get('runtime') or self.runtime
        signature=native_policy_signature(read(Path(runtime)/'release.json').get('source_sha256',{}))
        return self.rpc(dict(action=action,case=case,out=str(self.cat/lane),variant=variant,native_policy_signature=signature))

    def generate(self,case,lane):
        catalog=self.cat/lane;catalog.mkdir(parents=True,exist_ok=True)
        cmd=[PYTHON,'-u','scripts/v2/compile_asset_search.py','--append','--bindings',
             str(ROOT/'runs/serial_repairs_20261109/bindings.json'),'--exclude-source-layouts',str(self.out/'exclude_sources.json'),
             '--instances-per-task','1','--only',case['id'],'--out',str(catalog)]
        code=self.command(cmd,self.out/lane/(case['id']+'_compile.log'),stage='generate_alternative',tid=case['id'],timeout=240)
        row=next((r for r in read(catalog/'manifest.json').get('tasks',[]) if r.get('parent_task_id')==case['id']),None)
        return dict(status='proposed',record=row) if row else dict(status='deferred',reason='Compiler found no admissible alternative',exit_code=code)

    def run_case(self,initial,phase):
        tid=initial['id'];key=f'{phase}:{tid}'
        if key in self.ledger or self.accepted(tid):return
        case=self.latest.get(tid,initial);started=time.time()
        self.status(stage='expert_triage',task=tid,phase=phase,error=case.get('error'))
        if phase=='replay_corrected':
            lane='corrected_runtime';proposal=self.prepare(case,lane,'direct')
        elif phase.startswith('replay_'):
            lane=phase[len('replay_'):]
            if not re.fullmatch('[a-z0-9_]+',lane):raise ValueError('Invalid immutable replay lane')
            proposal=self.prepare(case,lane,'direct')
        elif phase=='guided':
            action='curate' if should_curate_first(case.get('error')) else 'direct'
            lane='curated' if action=='curate' else 'direct'
            proposal=self.prepare(case,lane,action)
            if proposal.get('status')!='proposed':
                action='direct';lane='direct';proposal=self.prepare(case,lane,action)
            if proposal.get('status')!='proposed':
                lane='initial_generate';proposal=self.generate(case,lane)
        elif phase in ('generate','generate_second'):
            lane=phase;proposal=self.generate(case,lane)
        else:
            variant=1 if phase=='repair_variant_1' else 2
            lane=phase;proposal=self.prepare(case,lane,'curate',variant)
        result=dict(phase=phase,task=tid,lane=lane,proposal=proposal,started=started)
        if proposal.get('status')=='proposed':result.update(self.validate(proposal['record'],lane,case))
        result['finished']=time.time();self.ledger[key]=result;write(self.out/'jobs.json',self.ledger)

    def run(self):
        if self.args.wait_pid:
            self.status(stage='waiting_expert_canaries',wait_pid=self.args.wait_pid);wait_external(self.args.wait_pid)
        serial=read(ROOT/'runs/serial_repairs_20261109/controller_status.json');pid=serial.get('pid')
        if pid and Path(f'/proc/{pid}').exists():
            if b'serial_repairs_20261109/controller.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('Serial coordinator identity changed')
            self.serial=pid;os.kill(pid,signal.SIGSTOP)
            serial=read(ROOT/'runs/serial_repairs_20261109/controller_status.json')
            self.status(stage='waiting_current_native_batch',serial_controller=pid,child_pid=serial.get('child_pid'))
            if serial.get('child_pid'):wait_external(serial['child_pid'])
        with (self.out/'geometry_worker.log').open('a') as log:
            self.helper_log=log;self.start_helper()
            index_path=self.out/'source_index.json'
            cases=read(index_path,[])
            if not cases:
                cases=self.rpc(dict(action='index'))['cases'];write(index_path,cases)
            if not cases:raise RuntimeError('No remaining-task index; preserve the existing coordinator')
            write(self.out/'exclude_sources.json',self.exclusions)
            if self.serial and Path(f'/proc/{self.serial}').exists():
                # Its current native batch has finished. Stop only the idle
                # coordinator; every unaccepted task is now in this index.
                os.kill(self.serial,signal.SIGTERM);os.kill(self.serial,signal.SIGCONT);wait_external(self.serial)
            self.replaced=True
            write(self.out/'plan.json',dict(tasks=[c['id'] for c in cases],native_workers=1,
                phases=['guided','generate','repair_variant_1','generate_second','repair_variant_2'],
                acceptance='fresh native initialization plus independent frozen-state reload',
                original_inventory_preserved=True,source='source_index.json'))
            for phase in ['guided','generate','repair_variant_1','generate_second','repair_variant_2']:
                for case in sorted(cases,key=lambda c:(0 if c['id'] in ('F008','F003','F019') else 1,c['id'])):
                    if self.stop_requested:
                        self.status(stage='interrupted',phase=phase);return
                    try:
                        # Explicit expert interventions can be inserted between
                        # native jobs without interrupting a running simulator.
                        for urgent in read(self.out/'urgent_jobs.json',[]):
                            selected=next((c for c in cases if c['id']==urgent['id']),None)
                            if selected:self.run_case(selected,urgent.get('phase','replay_corrected'))
                        self.run_case(case,phase)
                    except Exception as exc:
                        traceback.print_exc();self.ledger[f'{phase}:{case["id"]}']=dict(error=repr(exc),status='controller_error',finished=time.time())
                        write(self.out/'jobs.json',self.ledger)
            remaining=[c['id'] for c in cases if not self.accepted(c['id'])]
            write(self.out/'unresolved.json',[self.latest.get(tid,next(c for c in cases if c['id']==tid)) for tid in remaining])
            self.status(stage='complete' if not remaining else 'needs_further_expert_work',remaining=remaining,total=len(cases))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/overnight_expert_20261109')
    parser.add_argument('--wait-pid',type=int);parser.add_argument('--native-timeout',type=int,default=600);args=parser.parse_args()
    queue=Queue(args)
    signal.signal(signal.SIGTERM,queue.stop);signal.signal(signal.SIGINT,queue.stop)
    with (queue.out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:queue.run()
        finally:
            if queue.helper and queue.helper.poll() is None:queue.helper.terminate();queue.helper.wait()
            if not queue.replaced and queue.serial and Path(f'/proc/{queue.serial}').exists():os.kill(queue.serial,signal.SIGCONT)


if __name__=='__main__':main()
