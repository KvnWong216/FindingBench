from types import SimpleNamespace
import pytest
from rummagebench.adapters import mcp_server

@pytest.mark.parametrize("command", ["exit", "eof"])
def test_worker_closes_backend_and_synchronizes_gpu(monkeypatch, command):
    import os
    monkeypatch.setattr(mcp_server,"_redirect_worker_output",lambda:None)
    from rummagebench.adapters import python_api
    closed=[]
    obs=SimpleNamespace(model_dump=lambda **kw: {"frame_id":"current"})
    fake=SimpleNamespace(reset=lambda:obs,_backend=SimpleNamespace(close=lambda:closed.append(True)))
    monkeypatch.setattr(python_api,"create_session",lambda *a,**kw:fake)
    class Connection:
        def send(self,value): pass
        def recv(self):
            if command=="eof": raise EOFError()
            return ("exit",None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    monkeypatch.setenv("RUMMAGEBENCH_MCP_GPU","1")
    mcp_server._worker_main(Connection(),"scenario.yaml",5)
    assert closed==[True]
    assert os.environ["CUDA_VISIBLE_DEVICES"]==os.environ["RUMMAGEBENCH_MCP_GPU"]=="5"

def test_native_worker_output_cannot_pollute_stdio():
    import subprocess
    import sys
    result=subprocess.run([sys.executable,"-c",
        "from rummagebench.adapters.mcp_server import _redirect_worker_output; "
        "import os; _redirect_worker_output(); print('python-log',flush=True); os.write(1,b'native-log')"],
        capture_output=True,text=True,check=True)
    assert result.stdout==""
    assert "python-log" in result.stderr and "native-log" in result.stderr
