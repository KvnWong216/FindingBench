"""Consume atomically published reviewed bindings with two native workers."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/asset_search_remaining_20261104'
FILES=[ROOT/'build/dense_search_semantic_1k'/f'production_bindings_{x}.json' for x in ('a','b','rewrites')]
child=None;stopped=False

def stop(*args):
    global stopped
    stopped=True
    if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)

def main():
    global child,OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUT)
    parser.add_argument('--catalog-root',type=Path,default=ROOT/'scenarios/asset_search_remaining_20261104')
    parser.add_argument('--runtime-root',type=Path,default=ROOT)
    args=parser.parse_args();OUT=args.output_root.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    previous=None
    while not stopped:
        digest=hashlib.sha256(b''.join(p.read_bytes() if p.exists() else b'' for p in FILES)).hexdigest()
        docs=[json.loads(p.read_text()) if p.exists() else {} for p in FILES]
        if digest!=previous:
            with (OUT/'watch.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/v2/run_asset_pipeline.py'),
                    '--workers','2','--attempts','2','--timeout','900',
                    '--catalog-root',str(args.catalog_root.resolve()),'--runtime-root',str(args.runtime_root.resolve()),
                    '--output-root',str(OUT),'--bindings',*map(str,FILES),'--bindings-only'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:time.sleep(2)
                if child.returncode and not stopped:raise RuntimeError('Production child failed; inspect watch.log')
            previous=digest
        latest=hashlib.sha256(b''.join(p.read_bytes() if p.exists() else b'' for p in FILES)).hexdigest()
        if all(d.get('complete') for d in docs) and latest==previous:break
        for _ in range(15):
            if stopped:break
            time.sleep(2)

if __name__=='__main__':main()
