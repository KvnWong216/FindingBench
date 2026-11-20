"""Resumable public HTTP ranges with immutable whole-archive verification."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request,urlopen


def fetch_ranges(url,target,record,*,workers=2,piece_size=32*1024**2,attempts=8,progress=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    parts=target.parent/'.parts'/record['sha256'];parts.mkdir(parents=True,exist_ok=True)
    size=record['bytes'];chunks=[(i,lo,min(size,lo+piece_size)) for i,lo in enumerate(range(0,size,piece_size))]
    def download(chunk):
        i,lo,hi=chunk;part=parts/f'{i:05d}.part'
        if part.exists() and part.stat().st_size==hi-lo:
            if progress:progress(hi-lo)
            return part
        for attempt in range(attempts):
            try:
                partial=part.with_suffix('.partial')
                written=partial.stat().st_size if partial.exists() else 0
                if written>hi-lo:raise ValueError('HTTP_RANGE_PARTIAL_OVERSIZE')
                if written==hi-lo:
                    os.replace(partial,part)
                    if progress:progress(hi-lo)
                    return part
                start=lo+written
                req=Request(url,headers={'Range':f'bytes={start}-{hi-1}','User-Agent':'FindingBench-RAB-resource-fetch'})
                with urlopen(req,timeout=60) as response:
                    full_response=response.status==200 and start==0 and hi==size
                    if not full_response and (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{hi-1}/{size}'):
                        raise ValueError('SERVER_DID_NOT_HONOR_EXACT_RANGE')
                    # A dropped connection keeps its verified-position prefix.
                    # Retry only the remaining byte range instead of wasting
                    # another full 32 MiB transfer. Whole-archive SHA is still
                    # mandatory before promotion, including across restarts.
                    with partial.open('ab') as stream:
                        n=written
                        while data:=response.read(512*1024):
                            n+=len(data)
                            if n>hi-lo:raise ValueError('HTTP_RANGE_EXCEEDS_PINNED_LENGTH')
                            stream.write(data)
                    if n!=hi-lo:raise ValueError('HTTP_RANGE_TRUNCATED')
                os.replace(partial,part)
                if progress:progress(hi-lo)
                return part
            except Exception:
                if attempt==attempts-1:raise
                time.sleep(min(8,2**attempt))
    with ThreadPoolExecutor(max_workers=workers) as pool:paths=list(pool.map(download,chunks))
    assembled=target.with_name(target.name+'.assembling');digest=hashlib.sha256()
    with assembled.open('wb') as stream:
        for part in paths:
            with part.open('rb') as source:
                while data:=source.read(1024**2):digest.update(data);stream.write(data)
    if assembled.stat().st_size!=size or digest.hexdigest()!=record['sha256']:
        raise ValueError('RAB_ASSEMBLED_ARCHIVE_INTEGRITY_FAILED')
    os.replace(assembled,target);shutil.rmtree(parts)
    return target
