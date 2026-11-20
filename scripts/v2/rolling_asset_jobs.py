"""Keep native slots occupied; each build and restore remains a separate process."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import hashlib
import os
import threading
import time

def scene_weight(entity_count, available_bytes):
    """Keep a conservative fallback when memory pressure is high or unknown.

    Observed native RSS is ~6.5 GiB. With 24 GiB still available, admitting a
    third process leaves room for import-time growth and the desktop.
    """
    return 2 if entity_count>250 and (available_bytes is None or available_bytes<24*1024**3) else 1

def available_memory():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    except (OSError,ValueError):pass
    return None


def rolling_map(items, work, *, capacity, weight, stopped, tick, max_workers=3, draining=lambda:False):
    waiting=list(items);pending={};used=0;last=0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while waiting or pending:
            if stopped():waiting.clear()
            if draining() and not pending:return True
            while waiting and len(pending)<max_workers and not stopped() and not draining():
                candidate=next(((i,min(weight(item),capacity())) for i,item in enumerate(waiting)
                                if used+min(weight(item),capacity())<=capacity()),None)
                if candidate is None:break
                index,cost=candidate;item=waiting.pop(index)
                pending[pool.submit(work,item)]=cost;used+=cost
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                used-=pending.pop(future);future.result()
            if done or time.monotonic()-last>=30:
                tick();last=time.monotonic()
    return draining()


def run(pipeline, tasks, catalogue, out, semantic, ledger, attempts, timeout, capacity, restore_after_build=False):
    batch=pipeline.batch
    batch.OUT=out;batch.CATALOG=catalogue;batch.STOP.clear();batch.SEMANTIC_ROWS=[]
    batch.STATUS=dict(status='running',pid=os.getpid(),total=len(tasks),workers=capacity(),
                      scheduler='rolling',results=[])
    batch.STATUS_PATH=out/'batch_status.json'
    if batch.STATUS_PATH.exists():
        history=out/'batch_history';history.mkdir(exist_ok=True)
        (history/f'{time.time_ns()}.json').write_bytes(batch.STATUS_PATH.read_bytes())
    draining=lambda:(out/'drain.request').exists()
    ledger_lock=threading.Lock()
    scene_sizes={}
    memory_sample=[0.,None]
    def weight(task):
        scene=task['scene']
        if scene not in scene_sizes:
            survey=batch.ROOT/'runs/vla_difficulty_v2/_survey'/f'{scene}.json'
            scene_sizes[scene]=len(pipeline.read(survey).get('entities',{}))
        if time.monotonic()-memory_sample[0]>2:
            memory_sample[:]=[time.monotonic(),available_memory()]
        return min(scene_weight(scene_sizes[scene],memory_sample[1]),capacity())
    def restore_key(task):
        frozen=out/task['id']/'frozen_member.private.json'
        if not frozen.exists():return None
        return f"{task['id']}:{pipeline.fingerprint(task)}:restore:{hashlib.sha256(frozen.read_bytes()).hexdigest()}"
    def initial_ok(task):
        return pipeline.current(task,pipeline.read(out/task['id']/'validation.json'),'initialization_validated')
    def save_attempt(key,report):
        with ledger_lock:
            ledger[key]=dict(time=time.time(),report=report)
            pipeline.write(out/'attempts.json',ledger)
    def tick():
        batch.STATUS['workers']=capacity();batch.publish()
        try:
            pipeline.export(catalogue,out,semantic,'running')
        except Exception as exc:
            # A dashboard failure must be visible and must not strand worker futures.
            import traceback
            with (out/'report_errors.log').open('a') as log:traceback.print_exc(file=log)
            batch.STATUS['report_error']=str(exc)
        pipeline.write(out/'pipeline_status.json',dict(status='running',stage='rolling',
            pid=os.getpid(),workers=capacity(),worker_budget=3))
    def restore_work(task):
        if pipeline.STOP:return
        key=restore_key(task)
        if key is None:return
        result=batch.run(task,timeout,restore=True,seed=0)
        if pipeline.STOP:return
        save_attempt(key,pipeline.read(out/task['id']/'restore_validation.json'))
        with batch.LOCK:batch.STATUS['results'].append(dict(result,phase='restore',seed=0))
    # Delivery mode also drains successes from previous invocations before
    # starting more builds. Each reload still runs in a fresh native process.
    if restore_after_build:
        pending=[task for task in tasks if initial_ok(task) and not pipeline.restored(task,out)
                 and restore_key(task) not in ledger]
        rolling_map(pending,restore_work,capacity=capacity,weight=weight,stopped=lambda:pipeline.STOP,
                    tick=tick,max_workers=3,draining=draining)
    for seed in range(attempts):
        if pipeline.STOP or draining():break
        pending=[]
        for task in tasks:
            if task.get('authoring_issues'):continue
            if initial_ok(task):continue
            elif f"{task['id']}:{pipeline.fingerprint(task)}:{seed}" not in ledger:pending.append(task)
        def work(task):
            if pipeline.STOP:return
            if not initial_ok(task):
                result=batch.run(task,timeout,restore=False,seed=seed)
                if pipeline.STOP:return
                save_attempt(f"{task['id']}:{pipeline.fingerprint(task)}:{seed}",
                             pipeline.read(out/task['id']/'validation.json'))
                with batch.LOCK:batch.STATUS['results'].append(dict(result,phase='build',seed=seed))
            if not draining() and restore_after_build and initial_ok(task) and not pipeline.restored(task,out):
                if restore_key(task) not in ledger:restore_work(task)
        rolling_map(pending,work,capacity=capacity,weight=weight,stopped=lambda:pipeline.STOP,
                    tick=tick,max_workers=3,draining=draining)
    # Only after every requested build seed has been visited, reload successes.
    pending=[task for task in tasks if initial_ok(task) and not pipeline.restored(task,out)
             and restore_key(task) not in ledger]
    rolling_map(pending,restore_work,capacity=capacity,weight=weight,stopped=lambda:pipeline.STOP,
                tick=tick,max_workers=3,draining=draining)
    batch.STATUS['status']='interrupted' if pipeline.STOP else 'drained' if draining() else 'completed'
    batch.publish()
