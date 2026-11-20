#!/usr/bin/env python3
"""Resume a bounded, per-member render-cache queue without changing dataset files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from rab_spectator_data import DEFAULT_RELEASE, DEFAULT_OVERLAYS, is_release
    parser.add_argument('--asset-run', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--cache-dir', type=Path, help='Override the backend-specific cache root')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/spectator_cache_rab')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--attempts', type=int, default=2)
    args = parser.parse_args()
    args.asset_run = args.asset_run.resolve()
    if is_release(args.asset_run):
        from rab_cache_queue import main as rab_main
        return rab_main(['--asset-run',str(args.asset_run),'--rab-edits',str(args.cache_dir or DEFAULT_OVERLAYS),
                         '--output',str(args.output)])
    args.cache_dir = (args.cache_dir or ROOT/'build/spectator_cache').resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    lockfile = (args.output / 'queue.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lockfile.write(str(os.getpid()))
    lockfile.flush()
    # This import verifies frozen manifests without importing the simulator.
    from spectator_cache import lookup
    tasks = json.loads((args.asset_run / 'human_ready.json').read_text())['tasks']
    (args.output / 'queue_manifest.json').write_text(json.dumps(dict(
        asset_run=str(args.asset_run), total=len(tasks),
        tasks=[{key: task.get(key) for key in ('id', 'parent_task_id', 'scene', 'play_scenario', 'runtime_root')}
               for task in tasks]), indent=2))
    stop = threading.Event()
    mutex = threading.RLock()
    active = {}
    rows = {t['parent_task_id']: dict(id=t['id'], status='pending', attempts=0) for t in tasks}
    started = time.time()
    # Check only existing cache candidates, rather than starting 1000 engines
    # or delaying the first two jobs for a whole-dataset preflight.
    candidates = {}
    for path in args.cache_dir.glob('*/manifest.json'):
        try:
            meta = json.loads(path.read_text())
            candidates[meta.get('task_id')] = path.parent
        except (ValueError, OSError):
            pass
    for task in tasks:
        if task['id'] in candidates:
            try:
                hit = lookup(args.cache_dir, task)
                if hit:
                    rows[task['parent_task_id']].update(status='ready', cache=str(hit[0]), reused=True)
            except Exception:
                pass

    def save():
        with mutex:
            counts = {name: sum(r['status'] == name for r in rows.values())
                      for name in ('pending', 'running', 'ready', 'failed', 'interrupted')}
            status = 'complete' if counts['ready'] == len(tasks) else 'stopped' if stop.is_set() else 'running'
            if not active and not counts['pending'] and counts['failed']:
                status = 'finished_with_failures'
            doc = dict(status=status, total=len(tasks), counts=counts, workers=args.workers,
                       pid=os.getpid(), started_at=started, updated_at=time.time(),
                       cache_dir=str(args.cache_dir), asset_run=str(args.asset_run),
                       active=active.copy(), tasks=rows)
            tmp = args.output / 'status.json.tmp'
            tmp.write_text(json.dumps(doc, indent=2))
            os.replace(tmp, args.output / 'status.json')

    def halt(*_):
        stop.set()
        with mutex:
            children = list(active.values())
        for child in children:
            try:
                os.killpg(child['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
        def force_stop():
            with mutex:
                remaining = list(active.values())
            for child in remaining:
                try:
                    os.killpg(child['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        timer = threading.Timer(10, force_stop)
        timer.daemon = True
        timer.start()

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    save()

    def build(task, lane):
        name = task['parent_task_id']
        row = rows[name]
        if stop.is_set():
            return
        try:
            existing = lookup(args.cache_dir, task)
            if existing:
                with mutex:
                    row.update(status='ready', cache=str(existing[0]), reused=True)
                save()
                return
        except Exception as error:
            with mutex:
                row.update(status='failed', error=f'Source preflight: {error}')
            save()
            return
        for attempt in range(1, args.attempts + 1):
            if stop.is_set():
                break
            if shutil.disk_usage(args.cache_dir).free < 10 * 1024**3:
                row.update(status='failed', error='Less than 10 GiB cache disk space remains')
                save()
                return
            output = args.output / name / f'attempt_{attempt:02d}'
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(ROOT / 'scripts/spectator.py'),
                       '--task', task['id'], '--asset-run', str(args.asset_run),
                       '--cache-dir', str(args.cache_dir), '--port', str(18091 + lane),
                       '--width', '640', '--height', '360', '--fps', '5',
                       '--smoke-seconds', '.1', '--smoke-output', str(output)]
            env = os.environ.copy()
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env[key] = '2'
            begin = time.monotonic()
            with (output / 'renderer.log').open('w') as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                with mutex:
                    row.update(status='running', attempts=attempt, log=str(output / 'renderer.log'))
                    active[name] = dict(pid=process.pid, lane=lane, started_at=time.time())
                save()
                timed_out = False
                try:
                    process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                finally:
                    with mutex:
                        active.pop(name, None)
            try:
                cached = lookup(args.cache_dir, task)
                if cached:
                    with mutex:
                        row.update(status='ready', cache=str(cached[0]), reused=False,
                                   elapsed_seconds=time.monotonic() - begin,
                                   child_returncode=process.returncode)
                        row.pop('error', None)
                    save()
                    print(f'{name}: cache ready ({time.monotonic() - begin:.1f}s)', flush=True)
                    return
                detail = json.loads((output / 'report.json').read_text()).get('error', 'No valid cache was published')
            except Exception as error:
                detail = str(error)
            with mutex:
                row.update(status='interrupted' if stop.is_set() else 'failed',
                           error='Native cache build timed out' if timed_out else detail,
                           elapsed_seconds=time.monotonic() - begin)
            save()
            print(f'{name}: attempt {attempt} did not publish a valid cache: {row["error"]}', flush=True)
        if stop.is_set():
            row['status'] = 'interrupted'
            save()

    # Assign one long-lived worker to each lane; lane ports cannot overlap.
    index = 0

    def worker(lane):
        nonlocal index
        while not stop.is_set():
            with mutex:
                if index >= len(tasks):
                    return
                task = tasks[index]
                index += 1
            if rows[task['parent_task_id']]['status'] == 'ready':
                continue
            try:
                build(task, lane)
            except Exception as error:
                with mutex:
                    rows[task['parent_task_id']].update(status='failed', error=f'Queue worker: {error}')
                save()
                print(f'{task["parent_task_id"]}: queue error: {error}', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, lane) for lane in range(args.workers)]):
            future.result()
    save()
    return 0 if all(r['status'] == 'ready' for r in rows.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
