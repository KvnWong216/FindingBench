#!/usr/bin/env python3
"""Account for every retired episode; a migration request is not playable data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_goals(spec):
    objects={r['name']:r for r in spec['objects']}
    goals=spec['goal'] if isinstance(spec['goal'],list) else [spec['goal']]
    result=[]
    for goal in goals:
        name=goal.get('entity')
        if name not in objects:
            raise ValueError('SOURCE_GOAL_HAS_NO_OBJECT:'+str(name))
        result.append(dict(entity=name,category=objects[name]['category'],
                           relation=goal['kind'],destination=goal.get('receptacle')))
    return result


def make_queue(source,policy):
    rows=[json.loads(s) for s in Path(source).read_text().splitlines() if s.strip()]
    if len(rows)!=policy['expected_tasks']:raise ValueError('MIGRATION_SOURCE_COUNT')
    if len({r['parent_task_id'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_PARENT_TASK')
    # Shuffle layout slots, not instructions. Do not correlate domain/old
    # source ordering with one particular house layout or difficulty split.
    slots=[(i+1,j) for i in range(policy['diversity']['published_layouts_expected'])
           for j in range(policy['diversity']['target_tasks_per_layout'])]
    random.Random(20261118).shuffle(slots)
    result=[]
    for row,(layout,slot) in zip(rows,slots,strict=True):
        path=Path(row['source_spec']);actual=digest(path)
        if actual!=row['source_spec_sha256']:raise ValueError('BACKUP_RECIPE_CHANGED:'+row['parent_task_id'])
        spec=yaml.safe_load(path.read_text())
        if spec['instruction']!=row['instruction']:raise ValueError('BACKUP_INSTRUCTION_MISMATCH')
        result.append(dict(id='rab_'+row['parent_task_id'],parent_task_id=row['parent_task_id'],
            asset_provider='robotagentbench',behavior_fallback=False,
            original_instruction=row['instruction'],original_scene=row['original_scene'],
            source_backup=str(path.resolve()),source_sha256=actual,original_goals=source_goals(spec),
            requested_layout_id=layout,layout_slot=slot,
            requested_regime='short_range' if slot<8 else 'cross_zone_or_composite',
            status='pending_rab_binding',instruction=None,asset_bindings=None,
            native_physics_validated=False,render_cache=None,
            rewrite_receipt=None))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    policy=json.loads(args.policy.read_text());rows=make_queue(args.source,policy)
    target=args.output/'migration_queue.jsonl'
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if target.exists() and target.read_text()!=raw:raise ValueError('EXISTING_MIGRATION_QUEUE_CHANGED')
    target.write_text(raw)
    report=dict(status='migration_requests_created',requests=len(rows),generated_playable=0,
        queue_sha256=digest(target),source_sha256=digest(args.source),policy_sha256=digest(args.policy),
        per_layout=dict(Counter(r['requested_layout_id'] for r in rows)),
        requested_regimes=dict(Counter(r['requested_regime'] for r in rows)),
        original_goal_categories=dict(Counter(g['category'] for r in rows for g in r['original_goals'])),
        caveat='Layout slots are allocation requests, not 50 verified house types. No old instruction has been silently retargeted.')
    (args.output/'queue_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ('status','requests','generated_playable','requested_regimes')}))


if __name__=='__main__':main()
