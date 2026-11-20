#!/usr/bin/env python3
"""Continue real USD/coverage checks when the pinned resource worker is ready.

Does not manufacture 1K task success from queue slots or missing asset bindings.
All subprocesses are bounded; records survive disconnection from the editor.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Semantic candidates, not automatic identity/label/affordance equivalences.
# In particular a glasses CASE is not eyeglasses, a used toothbrush is not a
# new hygiene item, and a beverage bottle is not a shampoo/soy-sauce bottle.
CANDIDATES={
    'bottle_of_water':['water_bottle'], 'tablefork':['meal_fork'],
    'table_knife':['meal_knife'], 'paper_cup':['paper_cup'],
    'notebook':['notebook_green'], 'notepad':['cloth_clear_memo_pad'],
    'wallet':['wallet'], 'cap':['baseball_cap'],
    'baseball_cap':['baseball_cap'], 'remote_control':['remote_tv'],
    'tv_remote':['remote_tv'], 'paperback_book':['book_red'],
    'hardback':['book_red'], 'toy_car':['cloth_clear_car'],
    'toy_train':['cloth_clear_train'],
    'toy_figure':['cloth_clear_bear','cloth_clear_cat','cloth_clear_duck',
                  'cloth_clear_penguin','cloth_clear_rabbit','cloth_clear_robot'],
    'rubber_eraser':['cloth_clear_eraser'],
    'measuring_tape':['cloth_clear_tape_measure'],
    'can_of_soda':['coke_c3','sprite_c1'],
    'box_of_cookies':['oreo_gf'], 'box_of_crackers':['honeymaid'],
    'box_of_granola_bars':['zbar_t'],
    'mug':['coffee_cup_blue_compact','coffee_cup_cream_classic','coffee_cup_green_tall'],
    'teacup':['coffee_cup_blue_compact','coffee_cup_cream_classic','coffee_cup_green_tall'],
    'tray':['japanese_tray','meal_tray_large'],
}


def coverage(queue,audit):
    assets={r['id']:r for r in audit['assets']}
    rows=[];missing=Counter();candidates=Counter()
    for row in queue:
        bindings=[]
        for goal in row['original_goals']:
            choices=[k for k in CANDIDATES.get(goal['category'],[]) if k in assets]
            dynamic=[k for k in choices if assets[k]['dynamic_single_body_candidate']]
            bindings.append(dict(original_goal=goal,category_candidates=choices,
                dynamic_schema_candidates=dynamic,identity_verified=False,
                instruction_rewrite_required=True))
            if not choices:missing[goal['category']]+=1
            candidates.update(choices)
        all_categories=all(r['category_candidates'] for r in bindings)
        rows.append(dict(task_id=row['id'],parent_task_id=row['parent_task_id'],
            requested_layout_id=row['requested_layout_id'],goals=bindings,
            status='requires_instruction_and_layout_binding' if all_categories else 'requires_goal_rewrite',
            native_physics_validated=False,playable=False))
    return dict(schema='findingbench.rab_migration_coverage.v1',requests=len(queue),
        playable_generated=0,status_counts=dict(Counter(r['status'] for r in rows)),
        unsupported_original_goal_categories=dict(missing),candidate_asset_coverage=dict(candidates),
        eligible_assets_not_proposed=[r['id'] for r in audit['assets']
            if r['dynamic_single_body_candidate'] and r['id'] not in candidates],
        note='Candidates only. Old printed labels, samples, room names and user intent require explicit review/rewrite; no auto-equivalence.',
        tasks=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();out=args.run.resolve();state=dict(status='waiting_resources',pid=os.getpid(),started_at=time.time())
    def save(**kw):
        state.update(kw,updated_at=time.time());path=out/'continuation_status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,path)
    save()
    try:
        runtime=out/'Runtime-room-0.1.7'
        # This is a durable worker dependency wait, not a user-facing polling
        # loop. Installation promotion is atomic; never inspect a partial tree.
        deadline=time.time()+12*3600
        while not (runtime/'runtime.json').is_file():
            if time.time()>deadline:raise TimeoutError('RAB_RESOURCE_DEPENDENCY_TIMEOUT')
            if (out/'status.json').is_file():
                upstream=json.loads((out/'status.json').read_text())
                if upstream['status']=='failed':raise RuntimeError('RAB_RESOURCE_WORKER_FAILED:'+str(upstream.get('error')))
            save();time.sleep(30)
        audit=out/'diversity_audit_v1';audit.mkdir(exist_ok=True)
        script=Path(__file__).with_name('rab_diversity.py')
        cmd=[sys.executable,'-u',str(script),'--repo',str(args.repo.resolve()),'--runtime',str(runtime),'--output',str(audit)]
        save(status='auditing_composed_usd',command=cmd,audit_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
        with (audit/'audit.log').open('w') as stream:
            process=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=2*3600)
        report=json.loads((audit/'diversity.json').read_text())
        if process.returncode or report['status']!='cpu_usd_audit_passed':
            raise RuntimeError('RAB_DIVERSITY_AUDIT_INCOMPLETE:'+str(report.get('errors')))
        queue=[json.loads(s) for s in (out/'migration_queue.jsonl').read_text().splitlines()]
        result=coverage(queue,report);(out/'binding_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        save(status='coverage_audited_requires_semantic_binding',playable_generated=0,
            requests=len(queue),status_counts=result['status_counts'],
            architectural_geometry_groups=len(report['architectural_geometry_groups']),
            layouts=report['layouts'],asset_wrappers=report['asset_wrappers'],finished_at=time.time())
    except Exception as exc:
        save(status='failed',error=str(exc),finished_at=time.time());raise


if __name__=='__main__':main()
