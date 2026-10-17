"""CPU-only SSH command/remote lifecycle tests with an explicit fake web server."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def cli():
    path=Path(__file__).resolve().parents[2]/'scripts/play_remote.py'
    spec=importlib.util.spec_from_file_location('unit_play_remote',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def invoke(cli,repo,**payload):
    source=cli.REMOTE.replace('__PAYLOAD__',repr(json.dumps(dict(repo=str(repo),**payload))))
    result=subprocess.run([sys.executable,'-c',source],capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr
    return json.loads(next(line[len(cli.MARKER):] for line in result.stdout.splitlines() if line.startswith(cli.MARKER)))


def test_remote_starts_reuses_and_only_stops_its_own_service(cli,tmp_path):
    repo=tmp_path/"repo with ' quotes";(repo/'scripts').mkdir(parents=True)
    (repo/'scripts/play_data.py').write_text('''import os,json
from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":false,"error":null}')
 def log_message(self,*a):pass
HTTPServer(('127.0.0.1',int(os.environ['FINDINGBENCH_PLAY_PORT'])),Handler).serve_forever()
''')
    first=invoke(cli,repo,mode='start',data='home_0001')
    try:
        assert first['reused'] is False
        again=invoke(cli,repo,mode='start',data='home_0001')
        assert again['reused'] is True and again['pid']==first['pid'] and again['port']==first['port']
        rejected=invoke(cli,repo,mode='stop',data='home_0001',pid=first['pid'],token='wrong')
        assert rejected['stopped'] is False
        os.kill(first['pid'],0)
    finally:
        stopped=invoke(cli,repo,mode='stop',data='home_0001',pid=first['pid'],token=first['token'])
        assert stopped['stopped'] is True


def test_first_setup_saves_only_destination_and_reuses_it(cli,tmp_path,monkeypatch):
    monkeypatch.delenv('FINDINGBENCH_SSH',raising=False)
    monkeypatch.delenv('FINDINGBENCH_REMOTE_REPO',raising=False)
    monkeypatch.setattr('builtins.input',lambda prompt:'user@ssh-alias')
    path=tmp_path/'config.json'
    config=cli.config(path)
    assert set(json.loads(path.read_text()))=={'host','repo'}
    monkeypatch.setattr('builtins.input',lambda prompt:pytest.fail('must not ask again'))
    assert cli.config(path)==config


@pytest.mark.parametrize('value',['--proxy','../home_0001','1; touch /tmp/no'])
def test_sample_input_rejects_options_and_shell_syntax(cli,value):
    with pytest.raises(ValueError):cli.sample_id(value)


def test_command_quotes_payload_instead_of_interpolating_shell(cli):
    command=cli.remote_command(dict(repo="/a ' b; $(echo do-not-execute)",mode='start',data='home_0001'))
    assert '$(echo' not in command
    import shlex,base64
    parsed=shlex.split(command)
    assert parsed[:2]==['python3','-u']
    compile(parsed[3],'<remote-wrapper>','exec')


def test_tunnel_failure_is_reported_before_browser_open(cli):
    class Dead:
        def poll(self):return 255
    with pytest.raises(RuntimeError,match='隧道已退出'):
        cli.wait_http('http://127.0.0.1:1',Dead(),timeout=.1)


@pytest.mark.parametrize('reused',[True,False])
def test_one_argument_connects_forwards_opens_browser_and_preserves_reused_server(cli,monkeypatch,reused):
    events=[]
    monkeypatch.setattr(cli.shutil,'which',lambda _: '/unit/ssh')
    monkeypatch.setattr(cli,'config',lambda **k:dict(host='unit-host',repo='/unit'))
    def remote(cfg,**payload):
        events.append(payload['mode'])
        return dict(port=12345,pid=123,reused=reused,token='unit')
    monkeypatch.setattr(cli,'remote',remote)
    monkeypatch.setattr(cli,'local_port',lambda:54321)
    class Tunnel:
        returncode=0
        def poll(self):return 0
    def popen(command):
        assert '127.0.0.1:54321:127.0.0.1:12345' in command
        assert 'ExitOnForwardFailure=yes' in command and command[-1]=='unit-host'
        events.append('tunnel');return Tunnel()
    monkeypatch.setattr(cli.subprocess,'Popen',popen)
    monkeypatch.setattr(cli,'wait_http',lambda url,tunnel:events.append('http_ready'))
    monkeypatch.setattr(cli.webbrowser,'open',lambda url:events.append(url))
    assert cli.main(['1'])==0
    assert events==['start','tunnel','http_ready','http://127.0.0.1:54321']+([] if reused else ['stop'])
