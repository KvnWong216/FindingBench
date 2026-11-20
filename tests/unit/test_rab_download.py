import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts'))
from rab_download import fetch_ranges


def test_ranges_are_assembled_and_pinned_hash_is_required(tmp_path):
    data=bytes(range(256))*400;requested=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            bounds=self.headers['Range'].removeprefix('bytes=');lo,hi=map(int,bounds.split('-'));requested.append((lo,hi))
            self.send_response(206);self.send_header('Content-Range',f'bytes {lo}-{hi}/{len(data)}');self.end_headers();self.wfile.write(data[lo:hi+1])
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        url=f'http://127.0.0.1:{server.server_port}/asset';target=tmp_path/'asset.tar.gz'
        record=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        fetch_ranges(url,target,record,workers=2,piece_size=8192)
        assert target.read_bytes()==data
        assert sorted(requested)==[(lo,min(len(data)-1,lo+8191)) for lo in range(0,len(data),8192)]
        broken=tmp_path/'bad.tar.gz'
        with pytest.raises(ValueError,match='INTEGRITY_FAILED'):
            fetch_ranges(url,broken,dict(bytes=len(data),sha256='0'*64),piece_size=8192)
        assert not broken.exists()
    finally:server.shutdown();server.server_close()


def test_truncated_range_resumes_the_written_prefix_instead_of_restarting(tmp_path,monkeypatch):
    import rab_download
    monkeypatch.setattr(rab_download.time,'sleep',lambda _:None)
    data=bytes(range(256))*4;requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            lo,hi=map(int,self.headers['Range'].removeprefix('bytes=').split('-'));requests.append((lo,hi))
            self.send_response(206);self.send_header('Content-Range',f'bytes {lo}-{hi}/{len(data)}');self.end_headers()
            self.wfile.write(data[lo:(512 if len(requests)==1 else hi+1)])
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        target=tmp_path/'asset.tar.gz'
        fetch_ranges(f'http://127.0.0.1:{server.server_port}/asset',target,
            dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),piece_size=1024,workers=1)
        assert target.read_bytes()==data
        assert requests==[(0,1023),(512,1023)]
    finally:server.shutdown();server.server_close()
