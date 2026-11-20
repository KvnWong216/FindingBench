"""Prebuild the RAB spectator USD composition caches without starting Kit."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import time

from rab_spectator_data import (
    DEFAULT_OVERLAYS, DEFAULT_RELEASE, current, list_tasks, save,
)
from rab_spectator import build_cache, cache_lookup


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-run', type=Path, default=DEFAULT_RELEASE)
    p.add_argument('--rab-edits', type=Path, default=DEFAULT_OVERLAYS)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', action='append')
    a=p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=True)
    rows=list_tasks(a.asset_run)
    if a.task:
        provided={x.casefold() for x in a.task}
        rows=[r for r in rows if {r['id'].casefold(),r['parent_task_id'].casefold()} & provided]
        if not rows:
            p.error('No matching tasks')
    started=time.time()
    results={}
    def update(status):
        counts={k:sum(r['status']==k for r in results.values()) for k in ('ready','failed')}
        save(a.output/'status.json',dict(status=status,pid=os.getpid(),total=len(rows),
             counts=counts,remaining=len(rows)-len(results),started_at=started,tasks=results))
    with (a.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        update('running')
        from pxr import Usd,UsdUtils
        # Keep the original shared assets open across all 1K small scene layers.
        c=UsdUtils.StageCache.Get()
        with Usd.StageCacheContext(c):
            for i,t in enumerate(rows,1):
                ident=t['id']
                try:
                    e,_,member,_=current(a.asset_run,t,a.rab_edits.resolve())
                    target=member/'render_cache'/e['scene_sha256']
                    path=cache_lookup(e['scene'],target)
                    reused=path is not None
                    if not reused:
                        source=Usd.Stage.Open(e['scene'])
                        try:
                            path=build_cache(e['scene'],target,source)
                        finally:
                            if source:
                                c.Erase(source)
                    results[ident]=dict(status='ready',cache=str(path),reused=reused)
                except Exception as exc:
                    results[ident]=dict(status='failed',error=str(exc))
                if i%10==0:
                    update('running')
        failed=any(r['status']=='failed' for r in results.values())
        update('finished_with_failures' if failed else 'complete')
        return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
