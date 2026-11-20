"""Repack failed search episodes and collect confirmed native initializations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--extra-attempts',type=int,default=0)
    args = parser.parse_args()
    run,build,repo = (getattr(args,key).resolve() for key in ('run','build','repo'))
    output = build/'release'
    output.mkdir(exist_ok=True)
    begin = time.time()
    def load(path):
        return json.loads(path.read_text())
    def save(path,value):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        temp.replace(path)
    while not ((build/'manifest.json').exists()
               and (build/'native_validation/summary.json').exists()):
        for worker in ('build','native'):
            launch = load(build/(worker+'_launch.json'))
            proc = Path('/proc')/str(launch['pid'])/'stat'
            if not proc.exists() or proc.read_text().split(') ',1)[1].startswith('Z '):
                required = build/('manifest.json' if worker=='build' else 'native_validation/summary.json')
                if not required.exists():
                    save(output/'status.json',dict(status='waiting_worker_failed',worker=worker,
                                                   updated_at=time.time()))
                    return
        if time.time()-begin > 22*3600:
            save(output/'status.json',dict(status='dependency_deadline_exhausted'))
            return
        save(output/'status.json',dict(status='waiting_for_first_pass',updated_at=time.time()))
        time.sleep(20)
    candidates = load(build/'manifest.json')
    first_pass = {row['id']:row for row in candidates['tasks']}
    cohort = [json.loads(line) for line in (run/'migration_queue.jsonl').read_text().splitlines()]
    admitted = {}
    stable_single_goal_fallbacks = {}
    for row in candidates['tasks']:
        receipt = build/'native_validation'/(row['id']+'.json')
        if not receipt.is_file():
            continue
        validation = load(receipt)
        if (validation['scene_sha256']==row['scene_sha256']
                and validation['status']=='physx_initialization_pass'):
            validated = dict(row,physical_validation='physx_initialization_pass',physics_receipt=str(receipt))
            target_count = sum(obj['target'] for obj in row['placed'])
            if target_count<len(row['original_categories']):
                stable_single_goal_fallbacks[row['id']] = validated
            else:
                admitted[row['id']] = validated
    pending = [r for r in cohort if r['id'] not in admitted]
    for row in pending:
        revisions = sorted((output/'revisions'/row['id']).glob('attempt_*/episode.json'))
        for path in revisions:
            receipt = path.with_name('validation.json')
            if not receipt.exists():
                continue
            revision = load(path)
            validation = load(receipt)
            if (validation['scene_sha256']==revision['scene_sha256']
                    and validation['status']=='physx_initialization_pass'):
                admitted[row['id']] = dict(revision,physical_validation=validation['status'],
                                            physics_receipt=str(receipt))
                break
    pending = [r for r in pending if r['id'] not in admitted]
    save(output/'status.json',dict(status='repairing',initial_pass=len(admitted),remaining=len(pending)))
    if pending:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True,width=64,height=64,multi_gpu=False,
            limit_cpu_threads=8,extra_args=['--/log/file='+str(output/'repair_kit.log')]))
        import carb
        import omni.physx
        from pxr import Usd
        from rab_assets import LayerBundle
        from rab_available_resources import digest,namespace,provision
        from rab_build_search import build_episode,enable_shelf_support,load_assets,supports
        from rab_validate_search import validate
        resources = run/'available_resources'
        mapping,_ = provision(repo,run,resources)
        bundle = LayerBundle(namespace(resources),output/'repair_layers')
        poses = load(run/'pose_calibration/poses.json')['assets']
        assets,asset_errors = load_assets(mapping,repo,bundle,poses)
        settings = carb.settings.get_settings()
        settings.set_bool('/physics/suppressReadback',False)
        settings.set_bool('/physics/updateToUsd',True)
        sim = omni.physx.get_physx_simulation_interface()
        interface = omni.physx.get_physx_interface()
        errors = {}
        usage = Counter(r['target_asset'] for r in admitted.values())
        prefix = 'libraries/original_absolute/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/'
        for layout in sorted({r['requested_layout_id'] for r in pending}):
            key = prefix+f'ID_{layout:02d}_LIVING_STRAIGHT.usdc'
            try:
                source = Path(mapping[key])
                layer = bundle.clone(source)
                stage = Usd.Stage.Open(str(layer))
                overlay = enable_shelf_support(stage,output/'repair_layers'/f'layout_{layout:02d}_shelf.usda')
                fixtures,obstacles = supports(stage)
            except Exception as error:
                errors[layout] = str(error)
                continue
            used = {(r['target_asset'],r['search_fixture'],r['destination_fixture'])
                    for r in admitted.values() if r['layout_id']==layout}
            for row in [r for r in pending if r['requested_layout_id']==layout]:
                previous = first_pass.get(row['id'])
                unstable = set()
                original_receipt = build/'native_validation'/(row['id']+'.json')
                if previous and original_receipt.is_file():
                    bad_bodies = {obj['body'] for obj in load(original_receipt).get('objects', [])
                                  if not obj['stable']}
                    unstable.update(obj['asset'] for obj in previous['placed']
                                    if obj['body'] in bad_bodies)
                for number in range(1,7+args.extra_attempts):
                    directory = output/'revisions'/row['id']/f'attempt_{number}'
                    if (directory/'episode.json').is_file() and (directory/'validation.json').is_file():
                        prior_episode=load(directory/'episode.json')
                        prior_validation=load(directory/'validation.json')
                        bad_bodies = {obj['body'] for obj in prior_validation.get('objects', [])
                                      if not obj['stable']}
                        unstable.update(obj['asset'] for obj in prior_episode['placed']
                                        if obj['body'] in bad_bodies)
                        continue
                    attempted = set(used)
                    candidate = dict(row,placement_retry=number,excluded_distractors=sorted(unstable),
                                     preserve_goal_count=True)
                    retry_assets = dict(assets)
                    if previous and number<=2:
                        target = previous['target_asset']
                        candidate['required_target'] = target
                        if target in unstable and target in assets:
                            base = assets[target]
                            # Try space above the support for a clean settle
                            # before substituting a target that was unstable.
                            retry_assets[target] = dict(base,clearance=base.get('clearance',0.)+.015*number)
                    else:
                        candidate['excluded_targets'] = sorted(unstable)
                    episode = build_episode(candidate,overlay or layer,fixtures,
                                            obstacles,retry_assets,attempted,usage.copy(),directory)
                    if episode is None:
                        continue
                    episode.update(layout_id=layout,layout_sha256=digest(source),placement_retry=number)
                    try:
                        receipt = validate(episode,sim,interface)
                    except Exception as error:
                        receipt = dict(status='native_error',error=str(error),traceback=traceback.format_exc())
                    receipt.update(id=row['id'],scene_sha256=episode['scene_sha256'])
                    save(directory/'validation.json',receipt)
                    save(directory/'episode.json',episode)
                    if receipt['status']=='physx_initialization_pass':
                        episode.update(physical_validation=receipt['status'],
                                       physics_receipt=str(directory/'validation.json'))
                        admitted[row['id']] = episode
                        used = attempted
                        usage[episode['target_asset']] += 1
                        break
                    bad_bodies = {obj['body'] for obj in receipt.get('objects', []) if not obj['stable']}
                    unstable.update(obj['asset'] for obj in episode['placed'] if obj['body'] in bad_bodies)
                if row['id'] not in admitted and row['id'] in stable_single_goal_fallbacks:
                    fallback = stable_single_goal_fallbacks[row['id']]
                    key = (fallback['target_asset'],fallback['search_fixture'],fallback['destination_fixture'])
                    if key not in used:
                        admitted[row['id']] = dict(fallback,goal_count_preserved=False)
                        used.add(key)
                save(output/'status.json',dict(status='repairing',native_pass=len(admitted),
                     total=len(cohort),last_id=row['id'],errors=errors,updated_at=time.time()))
    missing = [r['id'] for r in cohort if r['id'] not in admitted]
    rows = [admitted[r['id']] for r in cohort if r['id'] in admitted]
    save(output/'physics_manifest.json',dict(tasks=rows,requested=len(cohort),physx_initialized=len(rows),
         pending=missing,layout_counts=Counter(r['layout_id'] for r in rows),
         target_counts=Counter(r['target_asset'] for r in rows),
         asset_counts=Counter(p['asset'] for r in rows for p in r['placed']),
         regime_counts=Counter(r['regime'] for r in rows),
         multi_goal_counts=Counter(sum(p['target'] for p in r['placed']) for r in rows),
         original_goal_count_changed=sum(sum(p['target'] for p in r['placed'])!=len(r['original_categories'])
                                         for r in rows),
         acceptance_scope='Native scene initialization and rigid-body stability',
         navigation_grasp_place_restore_evaluated=False))
    save(output/'status.json',dict(status='initialization_complete' if not missing else 'partial',
         native_pass=len(rows),remaining=len(missing),finished_at=time.time()))
    os._exit(0)


if __name__ == '__main__':
    main()
