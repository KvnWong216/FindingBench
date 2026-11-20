#!/usr/bin/env python3
"""Prepare pinned RAB resources in an isolated workspace, retaining provenance."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import tempfile
import time


def atomic(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    os.replace(temporary,path)


def verify(path, record):
    if not path.is_file() or path.stat().st_size!=record['bytes']:return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,9))
    parser.add_argument('--range-workers',type=int,default=2,choices=range(1,9))
    parser.add_argument('--joint-probe',type=Path)
    args=parser.parse_args();repo=args.repo.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'src'))
    from robotagentbench import resources
    from robotagentbench.installation import configure
    from rab_download import fetch_ranges
    manifest_path=repo/'resources/assets-0.1.7.json';manifest=resources.read_manifest(manifest_path)
    source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    expected='fa207f52d8da139f591f9dc590eb002c58f72383'
    if source_commit!=expected:raise ValueError('RAB_CODE_REVISION_CHANGED')
    downloads=out/'downloads';downloads.mkdir(exist_ok=True)
    rows={name:dict(status='pending',bytes=r['bytes']) for name,r in manifest['archives'].items()}
    lock=threading.Lock();state=dict(status='downloading',pid=os.getpid(),started_at=time.time(),
        code_commit=source_commit,asset_version=manifest['version'],asset_revision=manifest['hosting']['revision'],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),archives=rows,
        behavior_assets_deleted=(out/'behavior_retirement_cleanup.json').is_file(),
        source_furnishing_queue_resumed=False,archive_workers=args.workers,range_workers=args.range_workers)
    def save():
        with lock:
            state['updated_at']=time.time();state['verified_archives']=sum(r['status']=='verified' for r in rows.values())
            state['total_archives']=len(rows);state['verified_bytes']=sum(r['bytes'] for r in rows.values() if r['status']=='verified')
            state['free_bytes']=shutil.disk_usage(out).free;atomic(out/'status.json',state)
    save();finished=threading.Event()
    def heartbeat():
        while not finished.wait(20):save()
    threading.Thread(target=heartbeat,daemon=True).start()
    def fetch(name):
        target=downloads/name;record=manifest['archives'][name]
        with lock:rows[name]['status']='downloading'
        save()
        if not verify(target,record):
            if shutil.disk_usage(out).free<record['bytes']+12*1024**3:raise ValueError('RAB_DOWNLOAD_DISK_RESERVE')
            host=manifest['hosting']
            from urllib.parse import quote
            url=f"https://huggingface.co/datasets/{host['repository']}/resolve/{host['revision']}/{quote(name,safe='/')}"
            def progress(count):
                with lock:rows[name]['received_bytes']=rows[name].get('received_bytes',0)+count
                save()
            received=fetch_ranges(url,target,record,workers=args.range_workers,progress=progress)
            if received.resolve()!=target.resolve() or not verify(target,record):raise ValueError('RAB_ARCHIVE_INTEGRITY:'+name)
        with lock:rows[name].update(status='verified',finished_at=time.time())
        save();print('VERIFIED',name,flush=True)
    try:
        # Small dependencies first, then the two articulation-audit archives.
        order=sorted(rows,key=lambda n:(manifest['archives'][n]['bytes']>12*1024**2,
            n not in ('assets-0.1.0-005.tar.gz','assets-0.1.0-007.tar.gz'),n))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(fetch,n):n for n in order}
            for future in as_completed(futures):
                name=futures[future]
                try:future.result()
                except Exception as error:
                    with lock:rows[name].update(status='failed',error=str(error))
                    save()
        if any(r['status']!='verified' for r in rows.values()):raise ValueError('RAB_DOWNLOAD_INCOMPLETE')
        state['status']='installing';save();assets=out/'Assets-0.1.7'
        if assets.exists():receipt=resources.verify(manifest,assets)
        else:receipt=resources.install(manifest,downloads,assets)
        atomic(out/'asset_installation.json',receipt)
        state['status']='configuring';save();runtime=out/'Runtime-room-0.1.7'
        if runtime.exists():raise ValueError('EXISTING_RUNTIME_REQUIRES_EXPLICIT_VERIFICATION')
        receipt=configure(manifest,assets,runtime,'room');atomic(out/'runtime_configuration.json',receipt)
        index=json.loads((assets/manifest['index']).read_text());configuration=json.loads((runtime/'runtime.json').read_text())
        prefix='libraries/current_assets/all_assets/'
        catalog=[dict(id=Path(location[len(prefix):]).parts[0],location=location,
            path=configuration['resources'][location],sha256=digest)
            for location,digest in index['locations'].items()
            if location.startswith(prefix) and location.endswith('/asset.usda') and len(Path(location[len(prefix):]).parts)==2]
        atomic(out/'native_asset_catalog.json',dict(version=manifest['version'],assets=catalog,
            capability_scope='Asset identities only; articulated interaction must be admitted separately'))
        state.update(status='assets_ready',assets=str(assets),runtime=str(runtime),native_pilot='not_run',
            catalog_assets=len(catalog),rab_1000_generated=False);save()
        if args.joint_probe:
            probe=out/'native_joint_probe';probe.mkdir(exist_ok=True)
            state.update(status='native_joint_probe',native_pilot='running');save()
            command=[sys.executable,'-u',str(args.joint_probe.resolve()),'--repo',str(repo),
                     '--runtime',str(runtime),'--output',str(probe)]
            with tempfile.TemporaryDirectory(prefix='.native_tmp_',dir=probe) as temporary, (probe/'native.log').open('w') as stream:
                env=os.environ.copy();env.update(TMPDIR=temporary,TEMP=temporary,TMP=temporary)
                child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['native_pid']=child.pid;save()
                try:child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            result=json.loads((probe/'result.json').read_text()) if (probe/'result.json').exists() else {}
            if child.returncode or result.get('status')!='joint_probe_passed':raise ValueError('RAB_NATIVE_PROBE_FAILED:'+str(result.get('error','see native.log')))
            state.update(status='joint_probe_passed',native_pilot='joint_probe_passed',
                         full_search_loop='pending_grasp_place_and_restore_admission')
        state['finished_at']=time.time();save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished_at=time.time());save();raise
    finally:finished.set()


if __name__=='__main__':main()
