"""Refresh whole-catalogue diagnostics and serially validate new repair candidates."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,hashlib,traceback
import audit_dense_layouts as audit
import run_asset_pipeline as pipeline
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/manual_dense_repairs_20261106/candidate'
CAT=ROOT/'scenarios/manual_dense_repairs_20261106'
STATE=ROOT/'runs/dense_layout_audit_20261106'
pipeline.batch.ROOT=ROOT/'build/generator_runtime_v3_20261105'

def alive(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (OSError,ValueError,TypeError):return False

def merge():
    manifest=pipeline.read(CAT/'manifest.json');known={t['id'] for t in manifest['tasks']}
    proposals=pipeline.read(ROOT/'scenarios/dense_compartment_repairs_20261106/manifest.json')
    added=[]
    for t in proposals.get('tasks',[]):
        if t['id'] in known:continue
        link=CAT/t['id']
        if not link.exists():link.symlink_to(Path(t['scenario']).parent,target_is_directory=True)
        manifest['tasks'].append(dict(t,repair_priority=5));added.append(t['id'])
    if added:pipeline.write(CAT/'manifest.json',manifest)
    return added

def pending():
    ledger=pipeline.read(OUT/'attempts.json');result=[]
    for t in pipeline.read(CAT/'manifest.json').get('tasks',[]):
        if t.get('authoring_issues'):continue
        a=pipeline.read(OUT/t['id']/'validation.json')
        if pipeline.current(t,a,'initialization_validated'):
            f=OUT/t['id']/'frozen_member.private.json'
            if f.exists() and not pipeline.restored(t,OUT):
                key=f"{t['id']}:{pipeline.fingerprint(t)}:restore:{hashlib.sha256(f.read_bytes()).hexdigest()}"
                if key not in ledger:result.append(t['id'])
        elif f"{t['id']}:{pipeline.fingerprint(t)}:0" not in ledger:result.append(t['id'])
    return result

def main():
    with (STATE/'repair_monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last=0
        while True:
            try:
                if time.monotonic()-last>300:
                    audit.audit();added=merge();last=time.monotonic()
                    pipeline.write(STATE/'monitor_status.json',dict(pid=os.getpid(),status='watching',added=added,updated=time.time()))
                running=pipeline.read(OUT/'pipeline_status.json')
                if running.get('status')=='running' and running.get('pid')!=os.getpid() and alive(running.get('pid')):
                    time.sleep(10);continue
                todo=pending()
                if not todo:
                    pipeline.write(OUT/'pipeline_status.json',dict(status='idle',stage='awaiting_new_repair_candidates',pid=os.getpid(),workers=0))
                    time.sleep(10);continue
                # Reserve one slot before starting a native child. Production's existing
                # --reserve-worker-for-run reads this exact status and stops refilling it.
                pipeline.write(OUT/'pipeline_status.json',dict(status='running',stage='waiting_for_repair_slot',pid=os.getpid(),workers=1))
                active=pipeline.read(ROOT/'runs/asset_search_full_20261105/batch_status.json').get('active',[])
                if sum(alive(x.get('pid')) for x in active)>2:
                    time.sleep(2);continue
                command=['bash','scripts/build_asset_search.sh','--workers','1','--attempts','1','--restore-after-build',
                    '--skip-compile','--runtime-root',str(pipeline.batch.ROOT),'--catalog-root',str(CAT),'--output-root',str(OUT)]
                # New repair candidates use the validated acceleration release;
                # attempted candidates retain their original runtime identity.
                accelerated=ROOT/'build/generator_runtime_prepared_velocity_probe_20261106'
                if (accelerated/'release.json').is_file():
                    command += ['--new-runtime-root',str(accelerated)]
                with (OUT/'launcher.log').open('a') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pipeline.write(STATE/'monitor_status.json',dict(pid=os.getpid(),status='repair_launched',child_pid=child.pid,pending=todo,updated=time.time()))
                # Let the child acquire its lock and publish its own PID.
                for _ in range(30):
                    state=pipeline.read(OUT/'pipeline_status.json')
                    if state.get('pid')==child.pid or child.poll() is not None:break
                    time.sleep(1)
                time.sleep(10)
            except Exception as exc:
                # A partial report or temporary filesystem error must not strand
                # the remaining queue. Keep evidence and retry with a delay.
                traceback.print_exc()
                try:
                    pipeline.write(STATE/'monitor_status.json',dict(pid=os.getpid(),
                        status='recovering',error=repr(exc),updated=time.time()))
                except Exception:
                    traceback.print_exc()
                time.sleep(60)

if __name__=='__main__':main()
