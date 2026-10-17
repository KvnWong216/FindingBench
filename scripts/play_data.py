#!/usr/bin/env python3
"""Play a saved environment: python scripts/play_data.py 1 (or home_0001).

Browser UI stays on host loopback. The simulator runs on isolated physical GPU
5 with pipe RPC, so no network namespace or GPU isolation needs to be relaxed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
DEFAULT_RUN = REPO / 'build/behavior_production_environment_v1'
PREFIX = '@FINDINGBENCH_PLAY@'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def select_sample(run, sample):
    """Resolve one completed environment, refusing stale/ambiguous evidence."""
    if str(sample).isdigit():
        sample = f'home_{int(sample):04d}'
    if not re.fullmatch(r'home_\d{4,}', sample):
        raise ValueError('数据编号格式应为 1 或 home_0001')
    run = Path(run).resolve()
    status = read(run / 'production_status.json')
    batch = run / 'batch'
    rows = [r for r in read(batch / 'records.json') if r['id'] == sample]
    if len(rows) != 1 or sample not in read(batch / 'summary.json')['selected_ids']:
        raise ValueError(f'找不到数据 {sample}')
    row = rows[0]
    candidates = []
    reasons = []
    for p in (batch / sample).glob('gpu_realized_*/process.json'):
        proc = read(p)
        validation = proc.get('validation', {})
        if not validation.get('environment_only'):
            continue
        if validation.get('acceptance_policy') != status.get('active_acceptance_policy'):
            continue
        if status.get('active_validation_id') and p.parent.name != 'gpu_realized_' + status['active_validation_id']:
            continue
        if proc.get('status') != 'environment_ready':
            reasons.append(proc.get('error') or proc.get('status'))
            continue
        if proc.get('returncode') != 0 or validation.get('restore') is not True:
            raise ValueError('环境验收进程未正常结束')
        for key in ('scenario_sha256', 'assembly_sha256'):
            if proc.get(key) != row[key]:
                raise ValueError('数据与验收记录不匹配')
        folder = p.parent
        if digest(folder / 'result.json') != proc['result_sha256']:
            raise ValueError('验收结果文件已改变')
        required = {'background_realization', 'background_restore', 'background_after_target', 'environment_ready'}
        if not required <= set(proc.get('stage_sha256', {})):
            raise ValueError('环境验收证据不完整')
        for name in required:
            if digest(folder / f'{name}.json') != proc['stage_sha256'][name]:
                raise ValueError(f'验收证据已改变: {name}')
        result = read(folder / 'result.json')
        ready = read(folder / 'environment_ready.json')
        if (result.get('status') != 'environment_ready' or result.get('environment_only') is not True
                or ready.get('status') != 'environment_ready' or ready.get('policy_executed') is not False
                or any(ready.get(k) is not True for k in ('public_reset', 'spawn_clear', 'target_relation'))):
            raise ValueError('缺少环境就绪证据')
        for filename, key in [('environment_start.pt', 'checkpoint_sha256'), ('environment_start.png', 'image_sha256')]:
            if digest(folder / filename) != ready[key]:
                raise ValueError(f'保存的环境文件已改变: {filename}')
        candidates.append(folder.resolve())
    if len(candidates) != 1:
        raise ValueError(f'{sample} 需要唯一的已完成环境验收；找到 {len(candidates)} 条。' +
                         (' 失败原因: ' + '; '.join(reasons) if reasons else ' 当前可能仍待验收。'))
    if digest(row['scenario']) != row['scenario_sha256']:
        raise ValueError('场景文件已改变')
    assembly = read(row['assembly'])
    payload = {k: v for k, v in assembly.items() if k != 'manifest_sha256'}
    checksum = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if checksum != row['assembly_sha256'] or assembly['manifest_sha256'] != checksum:
        raise ValueError('装配文件已改变')
    return dict(row=row, folder=str(candidates[0]), run=str(run))


class SimulatorProxy:
    """Only public observation/action JSON crosses from simulator to browser."""
    def __init__(self, selected, output):
        from rummagebench.sim.gpu_isolation import isolated_gpu_command, resolve_gpu
        self.output = output
        self.responses = queue.Queue()
        sim_python = read(Path(selected['run']) / 'production_request.json')['sim_python']
        gpu = int(os.environ.get('FINDINGBENCH_PLAY_GPU', '5'))
        command, env = isolated_gpu_command([sim_python, str(Path(__file__).resolve()),
            selected['row']['id'], '--worker'], resolve_gpu(gpu))
        env.update(FINDINGBENCH_PLAY_RUN=selected['run'], FINDINGBENCH_PLAY_OUTPUT=str(output),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='1')
        limit_path = Path(selected['run']) / 'cpu_limit.json'
        cpus = read(limit_path)['cpus'] if limit_path.exists() else sorted(os.sched_getaffinity(0))
        env['FINDINGBENCH_PLAY_CPUS'] = ','.join(map(str, cpus[:2]))
        self.process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        with (self.output / 'simulator.log').open('w') as log:
            for line in self.process.stdout:
                if line.startswith(PREFIX):
                    try:
                        self.responses.put(json.loads(line[len(PREFIX):]))
                    except ValueError:
                        self.responses.put({'error': 'invalid simulator reply'})
                else:
                    log.write(line)
                    log.flush()
        self.responses.put({'error': '模拟器已退出，请查看 simulator.log'})

    def _call(self, method, action=None):
        self.process.stdin.write(json.dumps(dict(method=method, action=action)) + '\n')
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=1200)
        except queue.Empty:
            raise RuntimeError('模拟器等待超时，请查看 simulator.log') from None
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['result']

    def reset(self):
        return self._call('reset')

    def step(self, action):
        return self._call('step', action)

    def close(self):
        import signal
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.write('{"method":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            # This Popen created the process group; never target production.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            except ProcessLookupError:
                pass


def build_saved_session(selected, output):
    from copy import deepcopy
    import numpy as np
    from rummagebench.adapters.python_api import prepare_renderer_grounding, install_renderer_grounding, _ensure_kinematics_urdf
    from rummagebench.core.scenario import load_scenario
    from rummagebench.core.visual_session import VisualProtocolSession
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.behavior_adapter import load_manifest
    from rummagebench.sim.omnigibson.state_io import load_snapshot
    from rummagebench.sim.omnigibson.probe_support import relation_holds
    from rummagebench.authoring.layout.indoor_realization import parking_position

    folder = Path(selected['folder'])
    scenario = load_scenario(selected['row']['scenario'])
    scenario.expose_available_skills = False
    manifest = load_manifest(Path(scenario.scene.assembly_manifest))
    snapshot, meta = load_snapshot(folder / 'environment_start.pt')
    if (meta.get('assembly_sha256') != manifest['manifest_sha256']
            or meta.get('format') != 'observe-state-with-spatial-cache-v1'):
        raise ValueError('PLAY_CHECKPOINT_METADATA_MISMATCH')

    class SavedBackend(OmniGibsonBackend):
        def _restore_saved(self):
            # Check object identities before loading the SDK's serialized state.
            from rummagebench.sim.omnigibson import spatial_snapshot
            actual = spatial_snapshot._inventory(spatial_snapshot._objects(self._env.scene))
            if actual != snapshot['spatial_cache']['inventory']:
                raise ValueError('PLAY_CHECKPOINT_INVENTORY_MISMATCH')
            self.restore_observe_state(deepcopy(snapshot))
            self.validate_physics_state()
            self._validate_spawn_clearance()

        def _apply_placements(self, scenario, attempts=3):
            # Same builder / same object topology; restore instead of sampling.
            self._restore_saved()
            return {'placements': [], 'restored_environment': str(folder)}

        def reset(self):
            held = getattr(self._robot, '_ag_obj_in_hand', {})
            for arm in list(getattr(self._robot, 'arm_names', []) or []):
                if held.get(arm) is not None:
                    self._robot._release_grasp(arm=arm)
            self._restore_saved()

    prepare_renderer_grounding(scenario)
    backend = SavedBackend(seed=manifest['layout_seed'])
    try:
        backend.setup(scenario, task_seed_pose=lambda b,s: parking_position(manifest))
        install_renderer_grounding(backend)
        _ensure_kinematics_urdf(backend, scenario, selected['row']['scenario'])
        session = VisualProtocolSession(backend, scenario)
        session.set_trace(output / 'public_trace.jsonl')
    except BaseException:
        backend.close()
        raise

    class SavedEnv:
        def reset(self):
            observation = session.reset()
            if not relation_holds(backend, scenario.target.entity, scenario.placements[0].receptacle, 'inside'):
                raise ValueError('PLAY_TARGET_RELATION_LOST')
            expected = read(folder / 'background_after_target.json')['actual_poses']
            from rummagebench.sim.omnigibson.entity_resolver import resolve_object
            errors = {}
            for name, (pos, ori) in expected.items():
                obj = resolve_object(backend._env.scene, name)
                now, rot = obj.get_position_orientation()
                now = np.asarray(now.detach().cpu() if hasattr(now, 'detach') else now)
                rot = np.asarray(rot.detach().cpu() if hasattr(rot, 'detach') else rot)
                dist = float(np.linalg.norm(now-np.asarray(pos)))
                angle = float(np.degrees(2*np.arccos(np.clip(abs(np.dot(rot,ori)), 0, 1))))
                if not np.isfinite([dist,angle]).all() or dist>.005 or angle>2:
                    errors[name] = dict(position_m=dist, rotation_deg=angle)
            if errors:
                raise ValueError('PLAY_RESTORE_POSE_MISMATCH:'+json.dumps(errors))
            (output/'replay_check.json').write_text(json.dumps(dict(status='restored',
                sample=selected['row']['id'], checked_background_objects=len(expected),
                checkpoint_sha256=digest(folder/'environment_start.pt'), target_relation=True)))
            return {'observation': observation.model_dump(mode='json')}

        def step(self, action):
            return session.step(action).model_dump(mode='json')

        def close(self):
            backend.close()
    return SavedEnv()


def worker(selected):
    cpus = set(map(int, os.environ['FINDINGBENCH_PLAY_CPUS'].split(',')))
    os.sched_setaffinity(0, cpus)
    import torch
    from rummagebench.sim.gpu_isolation import verify_isolated_cuda, configure_single_gpu_kit
    verify_isolated_cuda(torch)
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    import isaacsim
    configure_single_gpu_kit(isaacsim)
    env = None
    def reply(payload):
        print('\n' + PREFIX + json.dumps(payload), flush=True)
    try:
        env = build_saved_session(selected, Path(os.environ['FINDINGBENCH_PLAY_OUTPUT']))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request['method'] == 'close':
                    break
                if request['method'] == 'reset':
                    result = env.reset()
                elif request['method'] == 'step':
                    result = env.step(request['action'])
                else:
                    raise ValueError('unknown request')
                reply({'result': result})
            except Exception as exc:
                reply({'error': f'{type(exc).__name__}: {exc}'})
    except Exception as exc:
        reply({'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='数据编号，例如 1 或 home_0001')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = Path(os.environ.get('FINDINGBENCH_PLAY_RUN', DEFAULT_RUN))
    try:
        selected = select_sample(run, args.data)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    os.chdir(REPO)
    if args.worker:
        worker(selected)
        return
    import serve_play as ui
    output = REPO/'build/play'/f'{selected["row"]["id"]}_{time.time_ns()}'
    output.mkdir(parents=True)
    (output/'selection.json').write_text(json.dumps(selected, indent=2))
    proxy = None
    def factory(*a, **k):
        nonlocal proxy
        proxy = SimulatorProxy(selected, output)
        return proxy
    state = ui.PlayState(selected['row']['scenario'], 0, env_factory=factory)
    ui.Handler.state = state
    ui.PAGE = ui.PAGE.replace('__ICONS__', json.dumps(ui.ICONS))
    port = int(os.environ.get('FINDINGBENCH_PLAY_PORT', '8090'))
    host = os.environ.get('FINDINGBENCH_PLAY_HOST', '127.0.0.1')
    server = ui.ThreadingHTTPServer((host, port), ui.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'数据: {selected["row"]["id"]}\n浏览器: http://{host}:{server.server_port}\n日志: {output}\n加载真实房屋需要几分钟，Ctrl+C 退出。', flush=True)
    try:
        state.boot()
        state.run_worker()
    except KeyboardInterrupt:
        pass
    finally:
        if proxy is not None:
            proxy.close()
        server.shutdown()


if __name__ == '__main__':
    main()
