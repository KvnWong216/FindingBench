#!/usr/bin/env python3
"""Detached, resumable native enrichment → independent reload → cache queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import tempfile
import time

ROOT = Path(os.environ.get('FINDINGBENCH_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def atomic(path, doc):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(doc, indent=2)); os.replace(tmp, path)


def accepted_case(case, parent, cache_dir, expected_policy=None, allow_quick=False):
    """A status string alone cannot accept a partial, stale or wrong member."""
    from spectator_cache import lookup
    case = Path(case)
    initial, restored = read(case/'validation.json'), read(case/'restore_validation.json')
    quick=allow_quick and initial.get('validation_mode')=='sparse_authoring'
    if quick:
        if (initial.get('status')!='initialization_validated' or initial.get('error')
                or initial.get('task_content_preserved') is not True
                or initial.get('navigation_verified') is not True
                or not initial.get('cache')):return None
        # This route has no independent reload; never relabel it as one.
        restored=initial
    if expected_policy and any(r.get('policy_version') != expected_policy for r in (initial, restored)):
        return None
    if expected_policy and restored.get('navigation_verified') is not True:
        return None
    if (initial.get('status') != 'initialization_validated' or initial.get('error')
            or (not quick and restored.get('status') != 'restore_validated') or restored.get('error')):
        return None
    if any(r.get('parent_task', {}).get('id') != parent['id'] for r in (initial, restored)):
        return None
    if (not quick and initial.get('process_id') is not None and initial.get('process_id') == restored.get('process_id')):
        return None
    tasks = read(case/'human_ready.json').get('tasks', [])
    if len(tasks) != 1:
        return None
    task = tasks[0]
    if quick and task.get('enrichment_validation')!='sparse_authoring':return None
    if (task.get('parent_task_id') != parent['parent_task_id'] or task.get('instruction') != parent['instruction']
            or task.get('parent_variant') != parent['id'] or task.get('runtime_root') != parent['runtime_root']):
        return None
    cached = lookup(cache_dir, task)
    if cached is None:
        return None
    return task, cached[0]


def failure_kind(case, phase, returncode):
    report = read(Path(case)/('restore_validation.json' if phase == 'reload' else 'validation.json'))
    message = report.get('error', '')
    if (report.get('exception_type') in ('AttributeError', 'NameError', 'TypeError', 'KeyError', 'ImportError', 'ModuleNotFoundError')
            or "object has no attribute" in message):
        return 'programming_error', message
    if not message:
        return 'infrastructure', f'Native {phase} exited {returncode}; see process log'
    tests = [('NO_SAFE_', 'no_safe_surface'), ('NO_PREFLIGHT_', 'no_safe_surface'),
        ('UNCHANGED_PARENT_', 'parent_dynamics'), ('ORIGINAL_WORLD_MOVED', 'world_drift'),
        ('RELATION_CHANGED', 'relation'), ('INITIAL_OPEN_', 'articulation'),
        ('NATIVE_BACKGROUND_SUPPORT', 'native_support'), ('BACKGROUND_HEIGHT_', 'native_support'),
        ('SUPPORT_EDGE_', 'native_support'), ('NEW_NAVIGATION_', 'navigation'),
        ('NATIVE_MOTION_', 'motion'), ('SPARSE_', 'density'), ('FROZEN_', 'snapshot'),
        ('DEPENDENCY_', 'snapshot'), ('TASK_CONTENT_', 'task_contract')]
    return next((kind for token, kind in tests if token in message), 'native_validation'), message


def reusable_initial_case(case, parent):
    """Resume an exported valid creation without rebuilding its original scene."""
    case = Path(case)
    initial = read(case/'validation.json')
    if (initial.get('status') != 'initialization_validated' or initial.get('error')
            or not initial.get('task_content_preserved')
            or {k:v for k,v in initial.get('parent_task',{}).items() if k!='index'} != {k:v for k,v in parent.items() if k!='index'}
            or not (case/'exported_world_poses.private.json').is_file()):
        return False
    restored = read(case/'restore_validation.json')
    if restored.get('error') and restored.get('exception_type') not in ('AttributeError', 'NameError', 'ImportError', 'ModuleNotFoundError'):
        return False
    from spectator_cache import identity
    identity(dict(parent, play_scenario=str(case/'play.yaml')))
    return True


def unstable_furniture_locations(case):
    """Only a failed independent motion observation can blacklist a location."""
    case=Path(case)
    report=read(case/'restore_validation.json')
    prefix='RELOAD_FURNITURE_UNSTABLE:'
    error=report.get('error','')
    if not error.startswith(prefix):return set()
    entities=set(error[len(prefix):].split(','))
    return {loc['id'] for loc in read(case/'resolved_plan.json').get('locations',[])
            if loc.get('entity') in entities}


def freeze_tools(output, revision='light_v4', furniture=False):
    """Running jobs use immutable code even when experts edit workspace tools."""
    if not revision or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in revision):
        raise ValueError('INVALID_TOOL_REVISION')
    destination = output/'tool_snapshots'/revision
    manifest_path = destination/'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        for ref, expected in manifest['sha256'].items():
            if hashlib.sha256((destination/ref).read_bytes()).hexdigest() != expected:
                raise ValueError('RUNNING_TOOL_SNAPSHOT_CHANGED:'+ref)
        return destination
    sources = ['scripts/enrich_background.py', 'scripts/background_planner.py',
        'scripts/enrich_background_queue.py', 'scripts/spectator.py', 'scripts/spectator_cache.py',
        'scripts/v2/validate_dense_pilot.py']
    if furniture:
        sources.extend(['scripts/furniture_planner.py', 'scripts/furnish_native.py'])
    hashes = {}
    for ref in sources:
        source, dest = ROOT/ref, destination/ref
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
        hashes[ref] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if furniture:
        from furniture_planner import build_policy
    else:
        from background_planner import build_policy
    data = Path(os.environ.get('OMNIGIBSON_DATA_PATH', ROOT.parents[1]/'behavior-datasets'))
    policy = build_policy(data/'behavior-1k-assets')
    atomic(destination/'policy.json', policy)
    hashes['policy.json'] = hashlib.sha256((destination/'policy.json').read_bytes()).hexdigest()
    atomic(manifest_path, dict(version=1, created_at=time.time(), sha256=hashes))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-run', type=Path, default=ROOT/'build/search_1k_20261115')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/background_1k_20261116')
    parser.add_argument('--release', type=Path, default=ROOT/'build/search_1k_lived_in_20261116')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'build/spectator_cache')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--furniture', action='store_true', help='One sparse prop per empty tabletop/native cabinet compartment')
    parser.add_argument('--quick-furniture', action='store_true', help='Sparse authoring and cache, without whole-world counterfactual or separate reload')
    parser.add_argument('--tool-revision', default='light_v4', help='Immutable code release; use a new name after a generator fix')
    args = parser.parse_args()
    if args.quick_furniture and not args.furniture:parser.error('--quick-furniture requires --furniture')
    for name in ('asset_run', 'output', 'release', 'cache_dir'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = read(args.asset_run/'human_ready.json')['tasks']
    if len({t['parent_task_id'] for t in tasks}) != len(tasks):
        raise ValueError('DUPLICATE_SOURCE_PARENT')
    snapshot = freeze_tools(args.output, args.tool_revision, furniture=args.furniture)
    expected_policy = None
    if args.furniture:
        from furniture_planner import VERSION
        expected_policy = VERSION
        if read(snapshot/'policy.json').get('version') != VERSION:
            raise ValueError('FURNITURE_SNAPSHOT_POLICY_MISMATCH')
    stop, mutex = threading.Event(), threading.RLock()
    active, accepted, rows = {}, {}, {}
    prior = read(args.output/'status.json').get('tasks', {})
    started = time.time()
    learned = read(args.output/'asset_failure_evidence.json')
    for index, task in enumerate(tasks, 1):
        name = task['parent_task_id']
        rows[name] = dict(index=index, id=task['id'], scene=task['scene'], status='pending', attempts=0)
        candidates = [Path(prior[name]['case'])] if prior.get(name, {}).get('case') else []
        if name == 'F001' and not args.furniture:
            candidates.append(ROOT/'runs/background_pilot_20261116/F001')
        candidates.extend(sorted((args.output/'cases'/name).glob('attempt_*'), reverse=True))
        for case in candidates:
            try:
                hit = accepted_case(case, task, args.cache_dir, expected_policy,allow_quick=args.quick_furniture)
            except Exception:
                hit = None
            if hit:
                member, cache = hit
                accepted[name] = dict(member, index=index, enrichment_case=str(case))
                rows[name].update(status='ready', case=str(case), cache=str(cache), reused=True)
                break
    atomic(args.output/'queue_manifest.json', dict(total=len(tasks), source_release=str(args.asset_run),
        source_sha256=hashlib.sha256((args.asset_run/'human_ready.json').read_bytes()).hexdigest(),
        task_order=[t['parent_task_id'] for t in tasks], tools=str(snapshot),
        tasks=[dict(index=i, **t) if 'index' not in t else t for i, t in enumerate(tasks, 1)]))

    def save(publish=False):
        with mutex:
            counts = {key: sum(r['status'] == key for r in rows.values())
                      for key in ('pending', 'running', 'ready', 'needs_repair', 'interrupted')}
            coverage = [member['furniture_coverage'] for member in accepted.values() if 'furniture_coverage' in member]
            furniture = dict(locations=sum(c['total_locations'] for c in coverage),
                added=sum(c['added'] for c in coverage), already_populated=sum(c['already_populated'] for c in coverage),
                unresolved=sum(len(c['unresolved']) for c in coverage), complete_tasks=sum(c['complete'] for c in coverage))
            status = ('complete' if counts['ready'] == len(tasks) else 'stopped' if stop.is_set()
                      else 'finished_with_failures' if not active and not counts['pending'] and counts['needs_repair']
                      else 'prepared' if args.prepare_only else 'running')
            atomic(args.output/'status.json', dict(status=status, total=len(tasks), counts=counts,
                workers=args.workers, pid=os.getpid(), started_at=started, updated_at=time.time(),
                active=active.copy(), release=str(args.release), furniture_coverage=furniture, tasks=rows))
            if publish:
                selected = sorted(accepted.values(), key=lambda r: r['index'])
                atomic(args.release/'human_ready.json', dict(tasks=selected, stage='complete' if len(selected) == len(tasks) else 'partial_validated',
                    expected_total=len(tasks), validated_count=len(selected), select_by_parent_id_while_partial=True))
                atomic(args.release/'manifest.json', dict(version=1, source_release=str(args.asset_run), total_expected=len(tasks),
                    validated_count=len(selected), missing_parents=[n for n in rows if n not in accepted],
                    full_skill_completion='unverified', action_failure_rate='not measured', furniture_coverage=furniture, tasks=selected))
                out = io.StringIO(); writer = csv.writer(out)
                writer.writerow(['original_index', 'parent_task_id', 'scene', 'instruction'])
                writer.writerows([t['index'], t['parent_task_id'], t['scene'], t['instruction']] for t in selected)
                path = args.release/'prompts.csv'; tmp = path.with_suffix('.csv.tmp'); tmp.write_text(out.getvalue()); os.replace(tmp, path)
    save(publish=True)
    if args.prepare_only:
        print(f'Prepared {len(tasks)} tasks; reused {len(accepted)}; pending {len(tasks)-len(accepted)}', flush=True)
        return 0

    def terminate_group(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def halt(*_):
        stop.set()
        with mutex:
            children = [item['pid'] for item in active.values() if item.get('pid')]
        for pid in children:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)

    def phase(task, case, mode):
        name = task['parent_task_id']; report_path = case/('restore_validation.json' if mode == 'reload' else 'validation.json')
        command = [sys.executable, '-u', str(snapshot/'scripts/enrich_background.py'),
            '--task', task['id'], '--asset-run', str(args.asset_run), '--plan', str(case/'job.json'),
            '--output', str(case), '--observer', str(snapshot/'scripts/v2/validate_dense_pilot.py')]
        if mode == 'reload':
            command.append('--restore')
        env = os.environ.copy(); env['FINDINGBENCH_PROJECT_ROOT'] = str(ROOT)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '2'
        # Kit exits through os._exit(), bypassing its tempfile cleanup. Own
        # the extraction root in the supervisor, which survives each child.
        # Render caches are flattened; native snapshots use source asset IDs.
        with tempfile.TemporaryDirectory(prefix='.native_tmp_',dir=case) as native_tmp, (case/(mode+'.log')).open('w') as stream:
            env.update(TMPDIR=native_tmp,TEMP=native_tmp,TMP=native_tmp)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            with mutex:
                active[name].update(pid=child.pid, phase=mode, started_at=time.time())
                rows[name].update(phase=mode, case=str(case), log=str(case/(mode+'.log')))
            save()
            try:
                child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                terminate_group(child)
            finally:
                if stop.is_set() and child.poll() is None:
                    terminate_group(child)
        result = read(report_path)
        expected = 'restore_validated' if mode == 'reload' else 'initialization_validated'
        return result.get('status') == expected and not result.get('error'), child.returncode

    def build(task, lane):
        name = task['parent_task_id']; row = rows[name]
        case_root = args.output/'cases'/name; case_root.mkdir(parents=True, exist_ok=True)
        existing = [int(p.name.split('_')[-1]) for p in case_root.glob('attempt_*') if p.name.split('_')[-1].isdigit()]
        offset = max(existing, default=0)
        avoid_assets, avoid_supports, avoid_locations = set(), set(), set()
        # Failed native surfaces are facts, not a fresh random seed. Preserve
        # this evidence across a code update or a supervisor restart.
        previous_cases = sorted(case_root.glob('attempt_*'), reverse=True)
        for previous in previous_cases:
            if args.furniture:avoid_locations.update(unstable_furniture_locations(previous))
            for previous_phase in ('initialization', 'reload'):
                kind, _ = failure_kind(previous, previous_phase, 1)
                if kind == 'native_support':
                    support = read(previous/'resolved_plan.json').get('support')
                    if support:
                        avoid_supports.add(support)
        with mutex:
            active[name] = dict(lane=lane, phase='preflight', pid=None)
            row.update(status='running')
        save()
        try:
            # Fail closed before importing the engine if the parent changed.
            from spectator_cache import identity
            identity(task)
            for previous in previous_cases:
                if stop.is_set():
                    break
                if args.quick_furniture:break
                try:
                    resumable = reusable_initial_case(previous, task)
                except Exception:
                    resumable = False
                if not resumable:
                    continue
                row['attempts'] = int(previous.name.split('_')[-1])
                ok, code = phase(task, previous, 'reload')
                hit = accepted_case(previous, task, args.cache_dir, expected_policy) if ok else None
                if hit:
                    member, cache = hit
                    with mutex:
                        accepted[name] = dict(member, index=row['index'], enrichment_case=str(previous))
                        row.update(status='ready', case=str(previous), cache=str(cache),
                            resumed_initialization=True, finished_at=time.time())
                    print(f'{name}: READY (resumed independent reload)', flush=True)
                    return
                kind, error = failure_kind(previous, 'reload', code)
                print(f'{name}: resumed reload: {kind}: {error}', flush=True)
                if kind == 'programming_error':
                    halt()
                break
            for local_attempt in range(args.attempts):
                if stop.is_set():
                    break
                if shutil.disk_usage(args.output).free < 20*1024**3:
                    # Leave untouched entries pending rather than recording
                    # hundreds of fictitious per-task failures for one disk.
                    stop.set()
                    raise ValueError('INFRA_DISK_BELOW_20_GIB')
                number = offset+local_attempt+1
                case = case_root/f'attempt_{number:02d}'; case.mkdir()
                with mutex:
                    blocked = {asset for asset, parents in learned.items() if len(set(parents)) >= 2}
                    row['attempts'] = number
                atomic(case/'job.json', dict(parent_task_id=name, auto_policy=str(snapshot/'policy.json'),
                    furniture_policy=args.furniture,
                    quick_furniture=args.quick_furniture,
                    semantic_review='User-approved contextual sparse background policy; task inventory and semantic-family exclusions are checked against live native geometry.',
                    variant=number-1, avoid_assets=sorted(avoid_assets|blocked), avoid_supports=sorted(avoid_supports),
                    avoid_locations=sorted(avoid_locations)))
                ok, code = phase(task, case, 'initialization')
                failed_phase = 'initialization'
                if ok and not stop.is_set() and not args.quick_furniture:
                    ok, code = phase(task, case, 'reload'); failed_phase = 'reload'
                if ok:
                    hit = accepted_case(case, task, args.cache_dir, expected_policy,allow_quick=args.quick_furniture)
                    if hit:
                        member, cache = hit
                        with mutex:
                            accepted[name] = dict(member, index=row['index'], enrichment_case=str(case))
                            row.update(status='ready', case=str(case), cache=str(cache), finished_at=time.time())
                            row.pop('error', None); row.pop('failure_kind', None)
                        print(f'{name}: READY ({number} attempts)', flush=True)
                        return
                kind, error = failure_kind(case, failed_phase, code)
                if ok:
                    kind, error = 'cache_integrity', 'Native pair passed but no integrity-valid cache was published'
                report = read(case/('restore_validation.json' if failed_phase == 'reload' else 'validation.json'))
                resolved = read(case/'resolved_plan.json')
                if args.furniture:avoid_locations.update(unstable_furniture_locations(case))
                if kind == 'native_support':
                    avoid_supports.add(resolved.get('support', ''))
                motion = read(case/'native_motion_convergence.private.json')
                for entity, metrics in motion.get('objects', {}).items():
                    if entity.startswith('background_') and not metrics.get('converged'):
                        index = int(entity.split('_')[-1]); props = resolved.get('props', [])
                        if index < len(props):
                            asset = f'{props[index]["category"]}/{props[index]["model"]}'
                            avoid_assets.add(asset)
                            with mutex:
                                learned[asset] = sorted(set(learned.get(asset, []))|{name})
                                atomic(args.output/'asset_failure_evidence.json', learned)
                with mutex:
                    row.update(failure_kind=kind, error=error)
                atomic(case/'attempt_result.json', dict(status='failed', phase=failed_phase, kind=kind, error=error))
                print(f'{name}: attempt {number}: {kind}: {error}', flush=True)
                save()
                if kind == 'programming_error':
                    # A model/placement retry cannot fix code. Stop our own
                    # native children and retain the whole pending manifest.
                    atomic(args.output/'blocked_programming_error.json', dict(
                        task=name, case=str(case), error=error, tools=str(snapshot), at=time.time()))
                    halt()
                    break
                if kind in ('relation','task_contract','snapshot'):
                    # These attempts have no corrective change to the recipe.
                    # Reimporting the same world cannot repair its contract.
                    print(f'{name}: no unchanged retry for {kind}',flush=True)
                    break
            with mutex:
                row['status'] = 'interrupted' if stop.is_set() else 'needs_repair'
        except Exception as error:
            with mutex:
                row.update(status='interrupted' if stop.is_set() else 'needs_repair', error=str(error), failure_kind='queue_or_source_preflight')
            print(f'{name}: queue error: {error}', flush=True)
        finally:
            with mutex:
                active.pop(name, None)
            save(publish=True)

    # Cover distinct scenes early, then process every remaining member. All
    # 1000 entries are in the manifest before either native worker starts.
    pending = [t for t in tasks if rows[t['parent_task_id']]['status'] != 'ready']
    first, later, scenes = [], [], set()
    for task in pending:
        if task['scene'] in scenes:
            later.append(task)
        else:
            first.append(task); scenes.add(task['scene'])
    ordered, cursor = first+later, 0
    def worker(lane):
        nonlocal cursor
        while not stop.is_set():
            with mutex:
                if cursor >= len(ordered):
                    return
                task = ordered[cursor]; cursor += 1
            build(task, lane)
    def heartbeat():
        while not stop.wait(30):
            save()
    threading.Thread(target=heartbeat, daemon=True).start()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, lane) for lane in range(args.workers)]):
            future.result()
    save(publish=True)
    return 0 if len(accepted) == len(tasks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
