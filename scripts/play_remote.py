#!/usr/bin/env python3
"""Run on your own computer: python play_remote.py 1.

Standard-library only. First launch saves an SSH destination, then each launch
starts/reuses the remote viewer, forwards a free local port and opens a browser.
Uses your existing OpenSSH keys/config/password prompt; never stores passwords.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

DEFAULT_REPO = '/data1/ygwang/codes/FindingBench'
CONFIG = Path.home() / '.config/findingbench/play.json'
MARKER = 'FINDINGBENCH_REMOTE='

# Executed remotely using Python's standard library, without installing this
# client or any dependencies on the user's computer.
REMOTE = r'''
import json,os,pathlib,signal,socket,subprocess,sys,time,urllib.request,uuid
cfg=json.loads(__PAYLOAD__)
repo=pathlib.Path(cfg['repo']).resolve()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def emit(data):print('FINDINGBENCH_REMOTE='+json.dumps(data),flush=True)
def arguments(pid):return (pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
def normalize(value):
    return 'home_'+str(int(value)).zfill(4) if value.isdigit() else value
def viewer_process(pid):
    p=pathlib.Path('/proc')/str(pid)
    if p.stat().st_uid!=os.getuid():return False
    args=[v.decode(errors='replace') for v in arguments(pid) if v]
    if '--worker' in args or not any(pathlib.Path(v).name=='play_data.py' for v in args):return False
    index=next(i for i,v in enumerate(args) if pathlib.Path(v).name=='play_data.py')
    script=pathlib.Path(args[index])
    if not script.is_absolute():script=(p/'cwd').resolve()/script
    return script.resolve()==repo/'scripts/play_data.py' and normalize(args[index+1])==cfg['data']
def environment(pid):
    return dict(x.split(b'=',1) for x in (pathlib.Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0') if b'=' in x)
def owns_port(pid,port):
    inodes=set()
    for line in pathlib.Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields=line.split()
        if fields[1]=='0100007F:'+format(port,'04X') and fields[3]=='0A':inodes.add('socket:['+fields[9]+']')
    for fd in (pathlib.Path('/proc')/str(pid)/'fd').iterdir():
        try:
            if os.readlink(fd) in inodes:return True
        except OSError:pass
    return False
def alive(port):
    try:
        with opener.open('http://127.0.0.1:'+str(port)+'/state',timeout=2) as r:
            state=json.load(r)
            return isinstance(state,dict) and 'ready' in state and not state.get('error')
    except (OSError,ValueError):return False
if cfg['mode']=='stop':
    pid=cfg['pid']
    try:
        if viewer_process(pid) and environment(pid).get(b'FINDINGBENCH_PLAY_LAUNCH_TOKEN',b'').decode()==cfg['token']:
            os.kill(pid,signal.SIGINT);emit({'stopped':True})
        else:emit({'stopped':False,'reason':'process identity changed'})
    except (OSError,IndexError):emit({'stopped':False,'reason':'already exited'})
    raise SystemExit
if not (repo/'scripts/play_data.py').is_file():raise SystemExit('Remote play_data.py not found: '+str(repo))
import fcntl
lockdir=repo/'build/play_remote';lockdir.mkdir(parents=True,exist_ok=True)
lock=(lockdir/(cfg['data']+'.lock')).open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        pid=int(p.name)
        if not viewer_process(pid):continue
        port=int(environment(pid).get(b'FINDINGBENCH_PLAY_PORT',b'8090'))
        if owns_port(pid,port) and alive(port):
            emit(dict(port=port,pid=pid,reused=True));raise SystemExit
    except (OSError,ValueError,IndexError):pass
with socket.socket() as s:
    s.bind(('127.0.0.1',0));port=s.getsockname()[1]
token=uuid.uuid4().hex
log=repo/'build/play_remote'/('launch_'+token+'.log');log.parent.mkdir(parents=True,exist_ok=True)
with log.open('w') as stream:
    child=subprocess.Popen([sys.executable,str(repo/'scripts/play_data.py'),cfg['data']],cwd=repo,
        env=dict(os.environ,FINDINGBENCH_PLAY_HOST='127.0.0.1',FINDINGBENCH_PLAY_PORT=str(port),FINDINGBENCH_PLAY_LAUNCH_TOKEN=token),
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(150):
    if child.poll() is not None:
        raise SystemExit('Viewer failed: '+log.read_text()[-3000:])
    if owns_port(child.pid,port) and alive(port):
        emit(dict(port=port,pid=child.pid,reused=False,token=token,log=str(log)));break
    time.sleep(.2)
else:
    child.send_signal(signal.SIGINT)
    raise SystemExit('Viewer did not start within 30 seconds; log: '+str(log))
'''


def sample_id(value):
    if value.isdigit():
        return f'home_{int(value):04d}'
    if re.fullmatch(r'home_\d{4,}', value):
        return value
    raise ValueError('数据编号应为 1 或 home_0001')


def config(path=CONFIG, configure=False):
    saved = json.loads(path.read_text()) if path.exists() else {}
    host = os.environ.get('FINDINGBENCH_SSH') or saved.get('host')
    if configure or not host:
        host = input('服务器 SSH 登录目标（例如 ygw@服务器地址，或 SSH 配置别名）：').strip()
    if not host or host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('SSH 登录目标无效；端口和跳板机可使用现有 ~/.ssh/config 别名配置')
    repo = os.environ.get('FINDINGBENCH_REMOTE_REPO') or saved.get('repo', DEFAULT_REPO)
    result = dict(host=host, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    if os.name != 'nt':
        path.chmod(0o600)
    return result


def ssh_base(cfg):
    return ['ssh', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']


def remote_command(payload):
    # Shell receives only a base64 token; paths and user input stay in JSON.
    import shlex
    source = REMOTE.replace('__PAYLOAD__', repr(json.dumps(payload)))
    encoded = base64.b64encode(source.encode()).decode()
    return 'python3 -u -c ' + shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')


def remote(cfg, **payload):
    result = subprocess.run([*ssh_base(cfg), cfg['host'], remote_command(dict(repo=cfg['repo'], **payload))],
                            stdout=subprocess.PIPE, text=True, check=False)
    if result.returncode:
        raise RuntimeError('SSH 或远端启动失败，请查看上方错误信息。')
    lines = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise RuntimeError('未收到远端服务地址。')
    return json.loads(lines[0])


def local_port():
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', 8090))
        except OSError:
            s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_http(url, tunnel, timeout=30):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if tunnel.poll() is not None:
            raise RuntimeError('SSH 隧道已退出，请检查登录和端口转发权限。')
        try:
            with opener.open(url+'/state', timeout=2) as response:
                state=json.load(response)
                if 'ready' in state:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(.2)
    raise RuntimeError('本地端口转发连接超时。')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='数据编号，例如 1 或 home_0001')
    parser.add_argument('--configure', action='store_true', help='重新设置服务器地址')
    args=parser.parse_args(argv)
    if not shutil.which('ssh'):
        parser.exit(1, '请先安装或启用 OpenSSH 客户端；此脚本不需要其他 Python 库。\n')
    tunnel=None;server=None;cfg=None
    try:
        data=sample_id(args.data)
        cfg=config(configure=args.configure)
        print('正在连接服务器并准备数据 '+data+'…', flush=True)
        server=remote(cfg, mode='start', data=data)
        port=local_port();url=f'http://127.0.0.1:{port}'
        tunnel=subprocess.Popen([*ssh_base(cfg), '-N', '-o', 'ExitOnForwardFailure=yes',
            '-L', f'127.0.0.1:{port}:127.0.0.1:{server["port"]}', cfg['host']])
        wait_http(url,tunnel)
        print(('复用已有试玩服务。' if server['reused'] else '试玩服务已启动。')+
              f'\n打开 {url}\n加载房屋期间请等待。保持本终端开启，Ctrl+C 退出。',flush=True)
        webbrowser.open(url)
        while tunnel.poll() is None:
            time.sleep(1)
        if tunnel.returncode:
            raise RuntimeError('SSH 连接中断。')
    except KeyboardInterrupt:
        print('\n正在关闭连接…',flush=True)
    except (ValueError,OSError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr)
        return 1
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill();tunnel.wait()
        if server and not server['reused']:
            try:
                remote(cfg,mode='stop',data=data,pid=server['pid'],token=server['token'])
            except (OSError,RuntimeError):
                print('远端清理连接失败；下次启动会尝试复用该服务。',file=sys.stderr)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
