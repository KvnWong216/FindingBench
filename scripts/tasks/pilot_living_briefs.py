"""Small paired CPU experiment; uses the existing prepare-home entry point.

All attempts and failures are retained. GPU realization uses gpu-smoke on the
resulting scenarios, separately; this report never certifies physical data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from _common import REPO
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import sha256_path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=127)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), REPO/'scripts/tasks/behavior_pipeline.py',
        REPO/'src/rummagebench/authoring/layout/indoor_background.py',
        REPO/'src/rummagebench/authoring/layout/living_brief.py']
    sources = {}
    for path in source_paths:
        dest = out/'sources'/path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        sources[str(path.relative_to(REPO))] = sha256_path(path)
    atomic_write_json(out/'sources.json', sources)

    def run(arm):
        config = REPO/'configs/layouts/living_brief_pilot'/f'{arm}.yaml'
        copy = out/f'{arm}.yaml'; copy.write_bytes(config.read_bytes())
        argv = [sys.executable, str(REPO/'scripts/tasks/behavior_pipeline.py'),
                'prepare-home', '--config', str(copy), '--seed', str(args.seed),
                '--out', str(out/arm)]
        started = time.monotonic()
        row = dict(arm=arm, argv=argv, config_sha256=sha256_path(copy),
                   candidate_count=0, physical_verified=0, evaluation_ready=0)
        with (out/f'{arm}.log').open('w') as log:
            try:
                proc = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=600, check=False)
                row['returncode'] = proc.returncode
            except subprocess.TimeoutExpired:
                row['failure'] = 'CPU_STAGE_TIMEOUT'
        row['elapsed_seconds'] = round(time.monotonic()-started, 3)
        request_path = out/arm/'solver_request.json'
        if request_path.exists(): row['request_sha256'] = sha256_path(request_path)
        result = out/arm/'solver_result.json'
        if row.get('returncode') == 0 and result.exists():
            receipt = json.loads(result.read_text()); request = json.loads(request_path.read_text())
            row.update(candidate_count=1, status='pending_simulation',
                       placements=len(receipt['placements']), result_sha256=sha256_path(result),
                       by_support=dict(Counter(p['support_name'] for p in receipt['placements'])))
            rooms = {o['name']: o['rooms'] for o in request['native_context']['objects']}
            row['by_rooms'] = dict(Counter('|'.join(rooms[p['support_name']]) for p in receipt['placements']))
        else:
            row['status'] = 'rejected'
            row.setdefault('failure', (out/f'{arm}.log').read_text()[-3000:])
        atomic_write_json(out/f'{arm}_attempt.json', row)
        return row

    # Same house, assets, total count, seed and iteration budget. A single seed
    # is an interface/mechanism test, not an acceptance-rate estimate.
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(run, ['uniform', 'cooking', 'meal']))
    report = dict(scope='paired_semantic_proposal_pilot_one_seed', seed=args.seed,
                  attempts=rows, candidates=sum(r['candidate_count'] for r in rows),
                  physical_verified=0, evaluation_ready=0)
    atomic_write_json(out/'cpu_summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
