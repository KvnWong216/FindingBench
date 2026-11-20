import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def queue_module():
    path=Path(__file__).resolve().parents[2]/'scripts/run_extended_vlm_suite.py'
    spec=importlib.util.spec_from_file_location('extended_suite',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_queue_waits_for_previous_suite_and_downloads(queue_module,tmp_path,monkeypatch):
    previous=tmp_path/'previous.json';downloads=tmp_path/'downloads.json'
    previous.write_text(json.dumps({'status':'running','pid':123}))
    downloads.write_text(json.dumps({'status':'downloading'}))
    checks=[]
    monkeypatch.setattr(queue_module.os,'kill',lambda pid,sig: checks.append(pid))
    def tick(_):
        if json.loads(previous.read_text())['status']=='running':
            previous.write_text(json.dumps({'status':'complete'}))
        else:downloads.write_text(json.dumps({'status':'complete','models':{'gemma':{'status':'blocked_access'}}}))
    monkeypatch.setattr(queue_module.time,'sleep',tick)
    statuses=[]
    result=queue_module.await_dependencies(previous,downloads,statuses.append)
    assert checks==[123]
    assert statuses==['waiting_previous_suite','waiting_downloads']
    assert result['gemma']['status']=='blocked_access'


def test_queue_rejects_stale_running_state(queue_module,tmp_path,monkeypatch):
    previous=tmp_path/'previous.json';previous.write_text(json.dumps({'status':'running','pid':123}))
    def missing(*_):raise ProcessLookupError()
    monkeypatch.setattr(queue_module.os,'kill',missing)
    with pytest.raises(RuntimeError,match='stale'):
        queue_module.await_dependencies(previous,tmp_path/'downloads.json',lambda _:None)
