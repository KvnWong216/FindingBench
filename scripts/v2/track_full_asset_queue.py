"""Track every authored task, including exclusions, without claiming blocked work ran."""
import argparse
from collections import Counter
import fcntl
import json
import hashlib
from functools import lru_cache
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
SEMANTIC = ROOT / 'build/dense_search_semantic_1k'
OUT = ROOT / 'runs/asset_full_queue_20261105'
RUNS = ['asset_search_pipeline', 'asset_search_remaining_v2_20261104',
        'asset_search_repairs_v2_20261104', 'asset_remaining_repairs_20261105',
        'asset_search_full_20261105', 'asset_search_gpu_new_20261106', 'manual_dense_repairs_20261106/candidate',
        'throughput_10x_20261106/offline_repair_validation', 'expert_capability_repairs_20261107',
        'expert_layout_repairs_20261107', 'expert_repair_queue_20261107', 'expert_profile_repairs_20261108',
        'stability_settle_solver_20261108', 'stability_repairs_20261108', 'remaining_repairs_20261108',
        'serial_repairs_20261109', 'generator_regression_20261109', 'expert_directed_20261109',
        'expert_curated_20261109', 'expert_curated_v2_20261109', 'expert_curated_v3_20261109',
        *['overnight_expert_20261109/'+lane for lane in
          ('curated','direct','initial_generate','generate','repair_variant_1','generate_second','repair_variant_2','corrected_runtime','bbox_center_v6','reuse_native_v6','support_planes_v7','expert_models_v7','expert_single_drawer_v7','expert_supported_drawer_v7','proposal_fixed_v7')]]


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return default

@lru_cache(maxsize=16384)
def _file_hash(path, identity):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def file_hash(path):
    path=Path(path);s=path.stat()
    # This filesystem timestamps rapid in-place writes at millisecond
    # granularity. Fresh same-size writes can share ALL stat fields. Hash
    # recently touched files directly; cache only quiescent evidence.
    if time.time_ns()-max(s.st_mtime_ns,s.st_ctime_ns)<2_000_000_000:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return _file_hash(str(path),(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))


@lru_cache(maxsize=8192)
def _proof_json(path,identity):
    return json.loads(Path(path).read_text())


def proof_json(path):
    """Read-only evidence cache; every file replacement/edit invalidates it."""
    path=Path(path)
    try:
        s=path.stat()
        if time.time_ns()-max(s.st_mtime_ns,s.st_ctime_ns)<2_000_000_000:
            return json.loads(path.read_text())
        return _proof_json(str(path),(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    except (OSError,ValueError):return {}


def accepted_native_pair(recipe, directory):
    """Require the original initialization and independent frozen-state reload."""
    recipe,directory=Path(recipe),Path(directory)
    a=proof_json(directory/'validation.json');b=proof_json(directory/'restore_validation.json')
    frozen=directory/'frozen_member.private.json'
    if not recipe.is_file() or not frozen.is_file():return False
    if a.get('status')!='initialization_validated' or b.get('status')!='restore_validated':return False
    if not (a.get('fresh_native_setup') and b.get('fresh_native_setup')):return False
    # Reload restores the saved state instead of resampling. Its bootstrap seed
    # need not equal the successful generation seed; the frozen artifact is the
    # identity. Controlled throughput trials separately enforce their seed plan.
    if type(a.get('seed')) is not int or type(b.get('seed')) is not int:return False
    if not a.get('runtime_sources') or a.get('runtime_sources')!=b.get('runtime_sources'):return False
    if a.get('runtime_environment',{})!=b.get('runtime_environment',{}):return False
    doc=proof_json(frozen)
    if not doc.get('artifacts') or doc.get('state') not in doc['artifacts']:return False
    try:
        if doc.get('version')==3:
            # A paired report cannot make an old mutable-code dependency
            # loadable. Apply the same dependency gate as the native loader.
            payload={k:v for k,v in doc.items() if k!='manifest_sha256'}
            encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),
                               ensure_ascii=False,allow_nan=False).encode('utf-8')
            if (not doc.get('dependencies') or doc.get('manifest_sha256')!=hashlib.sha256(encoded).hexdigest()
                    or any(not Path(name).is_file() or file_hash(name)!=expected
                           for name,expected in doc['dependencies'].items())):return False
        root=directory.resolve().parent
        for ref,digest in doc['artifacts'].items():
            path=(root/ref).resolve()
            if not path.is_relative_to(root) or not path.is_file() or file_hash(path)!=digest:return False
    except (OSError,TypeError,ValueError):return False
    sha=file_hash(recipe)
    return (a.get('scenario_sha256')==b.get('scenario_sha256')==sha and
            b.get('frozen_manifest_sha256')==file_hash(frozen))


def known_authoring_issue(recipe):
    """Hash-bound authoring defects do not erase physical validation evidence."""
    registry=proof_json(ROOT/'build/expert_surface_binding_audit/index.json')
    issue=registry.get('recipes',{}).get(file_hash(recipe))
    return issue.get('reason') if issue else None


def reviewable_native_pair(recipe,directory):
    return accepted_native_pair(recipe,directory) and not known_authoring_issue(recipe)


def refresh():
    rows = {r['id']: dict(id=r['id'], instruction=r['instruction'],
                         status='awaiting_binding', instances={})
            for r in read(SEMANTIC / 'catalogue.json', {})['tasks']}
    for name in ('a', 'b', 'rewrites'):
        for binding in read(SEMANTIC / f'production_bindings_{name}.json', {}).get('tasks', []):
            row = rows[binding['id']]
            row['binding_status'] = binding['status']
            row['reason'] = binding.get('reason') or binding.get('construction_requirements')
            row['status'] = 'awaiting_layout' if binding['status'] == 'ready' else 'awaiting_binding'
    for excluded in read(SEMANTIC / 'excluded_tasks.json', []):
        task = excluded['task']
        if task['id'] in rows:
            raise ValueError('Excluded task duplicates active catalogue: ' + task['id'])
        rows[task['id']] = dict(id=task['id'], instruction=task['instruction'],
                               status='awaiting_editorial_rewrite', reason=excluded['reason'], instances={})
    ready = {};binding_blocked=set()
    # Expert interventions create immutable lanes while the tracker runs.
    # Discover their manifests at refresh time; strict native-pair checks below
    # still gate every published accepted artifact.
    dynamic=[]
    for prefix in ('overnight_expert_20261109','expert100_20261110','expert100_lane2_20261111'):
        expert_catalog=ROOT/'scenarios'/prefix
        dynamic.extend(prefix+'/'+p.parent.name for p in expert_catalog.glob('*/manifest.json'))
    for name in dict.fromkeys([*RUNS,*sorted(dynamic)]):
        run = ROOT / 'runs' / name
        for parent in read(run / 'semantic_queue.json', {}).get('tasks', []):
            row = rows[parent['id']]
            for instance, state in parent.get('instance_states', {}).items():
                row['instances'][instance] = dict(state=state, run=name)
        for task in read(run / 'human_ready.json', {}).get('tasks', []):
            recipe=ROOT/task['scenario']
            if accepted_native_pair(recipe,run/task['id']):
                issue=known_authoring_issue(recipe)
                if issue:
                    binding_blocked.add(task['parent_task_id'])
                    rows[task['parent_task_id']]['authoring_issue']=issue
                else:ready[task['id']] = task
    for row in rows.values():
        if row['instances']:
            states = {x['state'] for x in row['instances'].values()}
            row['status'] = ('ready_for_human_review' if any(i in ready for i in row['instances'])
                             else 'physical_running' if states & {'build_running', 'restore_running'}
                             else 'physical_pending' if states & {'pending', 'restore_pending'}
                             else 'awaiting_physical_repair')
    summary = dict(total_authored_tasks=len(rows),
                   native_validated_tasks=len({task['parent_task_id'] for task in ready.values()}),
                   native_validation_remaining=len(rows)-len({task['parent_task_id'] for task in ready.values()}),
                   task_status_counts=dict(Counter(r['status'] for r in rows.values())),
                   human_ready_instances=len(ready), updated_unix=time.time(),
                   known_surface_binding_repairs=len(binding_blocked-{t['parent_task_id'] for t in ready.values()}),
                   concurrency_budget=len(read(ROOT/'runs/expert100_20261110/parallel_workers.json',{}).get('lanes',[])) or 1,
                   full_skill_completion='unverified')
    for filename, payload in [('queue.json', dict(summary=summary, tasks=list(rows.values()))),
                              ('summary.json', summary),
                              ('human_ready.json', dict(tasks=list(ready.values())))]:
        tmp = OUT / (filename + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(OUT / filename)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'tracker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            summary = refresh()
            if not args.watch:
                print(json.dumps(summary, indent=2))
                break
            time.sleep(30)
