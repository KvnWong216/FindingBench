"""Read the verified dense RAB release and optional per-task editor revisions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('RUMMAGEBENCH_ROOT', Path(__file__).resolve().parents[1])).resolve()
DEFAULT_RELEASE = Path(os.environ.get('FINDINGBENCH_RAB_RELEASE', ROOT / 'runs/rab_dense_serial_20261119')).resolve()
DEFAULT_OVERLAYS = ROOT / 'build/rab_spectator'
RAB_MIGRATION = ROOT / 'runs/rab_migration_20261118'


def digest(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
        tmp = f.name
    try:
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def is_release(path):
    return (Path(path) / 'combined_manifest.json').is_file()


def release_key(directory):
    return digest(Path(directory) / 'combined_manifest.json')


def edits_root(base, directory):
    return Path(base) / release_key(directory)


def list_tasks(directory):
    m = load(Path(directory) / 'combined_manifest.json')
    rows = m['tasks']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('RAB release contains duplicate task IDs')
    # Preserve the original FindingBench numeric order if its queue is available.
    queue = RAB_MIGRATION / 'migration_queue.jsonl'
    if queue.is_file():
        order = {json.loads(line)['id']: i for i, line in enumerate(queue.read_text().splitlines()) if line}
        rows = sorted(rows, key=lambda r: (order.get(r['id'], len(order)), r['id']))
    tasks = []
    for i, item in enumerate(rows, 1):
        if item['status'] != 'accepted_dense_initialization':
            raise ValueError(f'Non-admitted RAB task in release: {item["id"]}')
        e = load(item['episode'])
        if item['id'] != e['id']:
            raise ValueError('RAB episode ID mismatch')
        tasks.append(dict(id=e['id'], parent_task_id=e['parent_task_id'], instruction=e['instruction'],
                          scene=f'RAB layout {e["layout_id"]:02}', index=i, backend='rab',
                          episode_path=item['episode'], physics_receipt=item['physics_receipt']))
    return tasks


def select(directory, query, base=DEFAULT_OVERLAYS):
    rows = list_tasks(directory)
    q = str(query).strip().casefold()
    if q.isdecimal():
        i = int(q)
        if not 1 <= i <= len(rows):
            raise ValueError(f'Task index must be 1..{len(rows)}')
        return rows[i - 1]
    matches = [r for r in rows if q in (r['id'].casefold(), r['parent_task_id'].casefold())]
    if len(matches) != 1:
        raise ValueError(f'Expected one RAB task for {query!r}; found {len(matches)}')
    return matches[0]


def current(directory, task, base=DEFAULT_OVERLAYS):
    original = load(task['episode_path'])
    if digest(original['scene']) != original['scene_sha256']:
        raise ValueError('RAB source USD hash mismatch')
    member = edits_root(base, directory) / task['id']
    pointer = member / 'current.json'
    if not pointer.exists():
        receipt=load(task['physics_receipt'])
        if 'scene_sha256' in receipt and receipt['scene_sha256']!=original['scene_sha256']:
            raise ValueError('RAB source does not match its physics receipt')
        return original, receipt, member, None
    revision = load(pointer)
    path = member / 'revisions' / revision['revision']
    if path.parent != member / 'revisions' or digest(path / 'episode.json') != revision['episode_sha256']:
        raise ValueError('RAB editor revision integrity check failed')
    e = load(path / 'episode.json')
    if (e['id'] != task['id'] or e['base_scene_sha256'] != original['scene_sha256'] or
            digest(e['scene']) != e['scene_sha256'] or digest(path / 'physics.json') != revision['physics_sha256']):
        raise ValueError('RAB editor revision does not match the selected task')
    return e, load(path / 'physics.json'), member, revision['revision']
