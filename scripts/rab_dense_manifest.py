"""Merge independent RAB dense repair passes without counting sparse fallback."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rab_dense_search import digest, load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', nargs='+', type=Path, help='Repair passes in chronological order')
    args = parser.parse_args()
    baseline = {row['id']: row for row in load(args.baseline)['tasks']}
    accepted, pending = {}, {}
    checked = set()
    for run in args.runs:
        for lane in ('drawer', 'open'):
            for path in (run/lane/'cases').glob('*/result.json'):
                result = load(path)
                ident = result['id']
                if ident not in baseline or ident != path.parent.name:
                    raise ValueError(f'Result outside baseline: {path}')
                if result['status'] == 'pending_repair':
                    attempts = result['attempts']
                    pending[ident] = dict(id=ident, lane=lane, result=str(path.resolve()),
                        reason=result.get('error') or (attempts[-1]['status'] if attempts else 'no completed attempt'))
                elif result['status'] == 'accepted_dense_initialization':
                    if (result['episode'], result['physics_receipt']) in checked:
                        continue
                    episode = load(result['episode'])
                    receipt = load(result['physics_receipt'])
                    parent = baseline[ident]
                    contract = episode['original_recipe']
                    if not (episode['id'] == ident and episode['baseline_scene_sha256'] == parent['scene_sha256']
                            and episode['source_backup_sha256'] == parent['source_backup_sha256']
                            and digest(episode['scene']) == episode['scene_sha256'] == receipt['scene_sha256']
                            and receipt['status'] == 'physx_initialization_pass'
                            and receipt.get('dense_checks', {}).get('passed') is True
                            and sum(x['target'] for x in episode['placed']) == contract['target_count']
                            and len(episode['delivery']['placements']) == contract['target_count']
                            and sum(not x['target'] for x in episode['placed']) >= episode['minimum_distractors']
                            >= max(6,contract['original_search_distractors'])
                            and bool(episode['arrangement'].get('articulations') or
                                     episode['arrangement'].get('articulation')) == (lane == 'drawer')
                            and bool(contract['closed_container_required']) == (lane == 'drawer')):
                        raise ValueError(f'Dense receipt failed provenance or contract checks: {path}')
                    checked.add((result['episode'], result['physics_receipt']))
                    accepted[ident] = dict(result, lane=lane, chosen_from=str(run.resolve()),
                        storage_migration=episode.get('search_storage_migration'))
    remaining = [pending.get(ident, dict(id=ident, reason='no repair receipt'))
                 for ident in sorted(baseline.keys()-accepted.keys())]
    summary = dict(requested=len(baseline), accepted=len(accepted), pending=len(remaining),
        by_lane=dict(Counter(row['lane'] for row in accepted.values())),
        snack_drawer_rebindings=sum((row['storage_migration'] or {}).get('kind')=='living_room_snack_storage'
                                    for row in accepted.values()),
        refrigerator_rebindings=sum((row['storage_migration'] or {}).get('kind')=='rab_refrigerator'
                                    for row in accepted.values()),
        pending_reasons=dict(Counter(row['reason'] for row in remaining)))
    save(args.output, dict(summary=summary, baseline_sha256=digest(args.baseline),
        acceptance_scope='Native initial stability and movable occlusion or diagnostic drawer/fridge motion; no robot rollout',
        tasks=[accepted[key] for key in sorted(accepted)], pending=remaining))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
