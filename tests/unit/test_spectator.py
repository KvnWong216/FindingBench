"""Spectator controls must not accidentally become scene/game controls."""
import importlib.util
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('spectator', Path(__file__).parents[2] / 'scripts/spectator.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cache_spec = importlib.util.spec_from_file_location('spectator_cache', Path(__file__).parents[2] / 'scripts/spectator_cache.py')
cache = importlib.util.module_from_spec(cache_spec)
cache_spec.loader.exec_module(cache)


def test_flight_basis_and_usd_projection_agree():
    camera = m.FlightCamera([0, 0, 0], yaw=73, pitch=-41)
    rotation = Rotation.from_quat(camera.usd_quaternion()).as_matrix()
    forward, right, up = camera.basis()
    assert np.allclose(-rotation[:, 2], forward)
    assert np.allclose(rotation[:, 0], right)
    assert np.allclose(rotation[:, 1], up)
    camera.advance({'KeyW'}, 0, 0, .1, 1)
    assert np.allclose(camera.position, np.asarray(forward) * .1)


def test_native_float32_camera_pose_is_json_serializable():
    camera = m.FlightCamera(np.array([1, 2, 3], dtype=np.float32))
    assert json.loads(json.dumps(camera.position)) == [1., 2., 3.]


def test_normalized_motion_mouse_directions_and_reset():
    camera = m.FlightCamera([0, 0, 1])
    camera.advance({'KeyW', 'KeyD'}, 0, 0, .1, 1)
    assert np.allclose(camera.position, [.1 / math.sqrt(2), -.1 / math.sqrt(2), 1])
    camera.advance(set(), 750, 750, .1, 1)
    assert camera.yaw == -90 and camera.pitch == -89
    camera.advance({'KeyW'}, 100, 100, .1, 1, reset=True)
    assert camera.position == [0, 0, 1] and camera.yaw == camera.pitch == 0
    camera.advance({'Space', 'ControlLeft'}, 0, 0, 10, 1)
    assert np.allclose(camera.position, [0, 0, 1.3])  # lag cannot cause teleportation
    camera.advance({'ShiftLeft'}, 0, 0, .1, 1)
    assert np.allclose(camera.position, [0, 0, 1.2])


def test_input_coalesces_and_disconnect_releases_movement(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    state = m.SharedState({})
    state.receive(dict(keys=['KeyW'], dx=10))
    state.receive(dict(keys=['KeyW'], dx=20, dy=3))
    assert state.consume()[:3] == ({'KeyW'}, 30, 3)
    assert state.consume()[1:3] == (0, 0)
    clock[0] += .6
    assert state.consume()[0] == set()


@pytest.mark.parametrize('body', [None, [], {'keys': 'KeyW'}, {'keys': ['GRASP']},
                                    {'dx': float('nan')}, {'speed': 0}, {'dy': True},
                                    {'reset': 'yes'}, {'keys': [None]}])
def test_invalid_controls_cannot_poison_camera(body):
    state = m.SharedState({})
    with pytest.raises(ValueError):
        state.receive(body)
    assert state.consume()[:3] == (set(), 0, 0)


def test_select_stable_index_or_exact_case_insensitive_id(tmp_path):
    tasks = [dict(id='asset_F001_01', parent_task_id='F001'), dict(id='asset_P007_01', parent_task_id='P007')]
    (tmp_path / 'human_ready.json').write_text(json.dumps({'tasks': tasks}))
    for query in ('2', 'p007', 'asset_P007_01'):
        assert m.select_task(tmp_path, query) == dict(tasks[1], index=2)
    for query in ('0', '3', 'P', '../secret'):
        with pytest.raises(ValueError):
            m.select_task(tmp_path, query)


def test_runtime_uses_selected_frozen_sources_and_clears_foreign_flags(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    runtime = root / 'build/pinned'
    (runtime / 'src/rummagebench').mkdir(parents=True)
    data = tmp_path / 'behavior-datasets'
    for name in ('behavior-1k-assets', 'omnigibson-robot-assets'):
        (data / name).mkdir(parents=True)
    monkeypatch.setattr(m, 'ROOT', root)
    env = {'YGBENCH_UNRELATED_AUTHORING_FLAG': '1', 'PYTHONPATH': 'wrong/src'}
    monkeypatch.setattr(m.os, 'environ', env)
    calls = []
    monkeypatch.setattr(m.os, 'chdir', lambda p: calls.append(p))
    monkeypatch.setattr(m.os, 'execv', lambda *a: calls.append(a))
    m.enter_runtime(dict(runtime_root=str(runtime), runtime_environment={'YGBENCH_GPU_PROPOSALS': '0'}), ['--task', '1'])
    assert env['PYTHONPATH'] == str(runtime / 'src')
    assert env['RUMMAGEBENCH_ROOT'] == str(runtime)
    assert env['OMNIGIBSON_DATA_PATH'] == str(data)
    assert env['YGBENCH_GPU_PROPOSALS'] == '0' and 'YGBENCH_UNRELATED_AUTHORING_FLAG' not in env
    assert calls[0] == runtime
    assert calls[-1][1][-1] == '--_native'


def test_http_has_no_skill_or_physics_endpoint():
    state = m.SharedState(dict(parent_task_id='F001'))
    handler = type('TestHandler', (m.Handler,), {'state': state})
    server = m.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f'http://127.0.0.1:{server.server_port}'
    try:
        assert b'requestPointerLock' in urlopen(root).read()
        assert not json.load(urlopen(root + '/state'))['ready']
        request = Request(root + '/input', data=json.dumps(dict(keys=['KeyW'], dx=8)).encode(), method='POST')
        assert json.load(urlopen(request)) == {'ok': True}
        assert state.consume()[:2] == ({'KeyW'}, 8)
        for route in ('/action', '/reset', '/step'):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(root + route, data=b'{}', method='POST'))
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(root + '/input', data=b'{"dx":NaN}', method='POST'))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_physical_audit_handles_native_fixed_links_and_detects_changes():
    pose = [np.array([0., 1., 2.]), np.array([0., 0., 0., 1.])]
    fixed = SimpleNamespace(get_position_orientation=lambda: pose)
    obj = SimpleNamespace(name='fixed_wall', links={'base_link': fixed}, n_joints=0,
                          get_position_orientation=lambda: pose)
    backend = SimpleNamespace(_env=SimpleNamespace(scene=SimpleNamespace(objects=[obj])))
    before = m.physical_fingerprint(backend)
    assert before == m.physical_fingerprint(backend)
    pose[0][0] += .001
    assert before != m.physical_fingerprint(backend)


@pytest.mark.parametrize('changed', ['source', 'stage', 'texture', 'missing_texture', 'none'])
def test_render_cache_rejects_stale_world_or_materials(tmp_path, monkeypatch, changed):
    root = tmp_path / 'entry'
    root.mkdir()
    stage = root / 'scene.usdc'
    stage.write_bytes(b'local render stage test fixture')
    texture = tmp_path / 'texture.png'
    texture.write_bytes(b'texture fixture')
    source = {'frozen_manifest_sha256': 'original-world'}
    stat = texture.stat()
    meta = dict(version=cache.VERSION, source=source, scene_sha256=cache.digest(stage),
                external_files={str(texture): [stat.st_size, stat.st_mtime_ns]})
    (root / 'manifest.json').write_text(json.dumps(meta))
    monkeypatch.setattr(cache, 'identity', lambda task: ('entry', source))
    if changed == 'source':
        monkeypatch.setattr(cache, 'identity', lambda task: ('entry', {'frozen_manifest_sha256': 'different-world'}))
    elif changed == 'stage':
        stage.write_bytes(b'changed geometry')
    elif changed == 'texture':
        texture.write_bytes(b'changed material image')
    elif changed == 'missing_texture':
        texture.unlink()
    result = cache.lookup(tmp_path, {})
    assert (result is not None) == (changed == 'none')


@pytest.mark.parametrize('valid', [True, False])
def test_bulk_queue_reuses_caches_or_records_invalid_sources_without_launching_engines(tmp_path, monkeypatch, valid):
    import sys
    path = Path(__file__).parents[2] / 'scripts/build_spectator_cache.py'
    module_spec = importlib.util.spec_from_file_location('cache_queue_test', path)
    queue = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(queue)
    asset_run = tmp_path / 'dataset'
    asset_run.mkdir()
    rows = [dict(id=f'asset_F00{i}_01', parent_task_id=f'F00{i}') for i in range(1, 4)]
    (asset_run / 'human_ready.json').write_text(json.dumps({'tasks': rows}))
    def lookup(directory, task):
        if not valid:
            raise ValueError('Frozen source changed')
        return directory / task['id'], {}
    monkeypatch.setitem(sys.modules, 'spectator_cache', SimpleNamespace(lookup=lookup))
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No native process should start'))
    monkeypatch.setattr(sys, 'argv', ['build_spectator_cache.py', '--asset-run', str(asset_run),
                                    '--cache-dir', str(tmp_path / 'cache'), '--output', str(tmp_path / 'queue')])
    assert queue.main() == (0 if valid else 1)
    result = json.loads((tmp_path / 'queue/status.json').read_text())
    assert result['total'] == 3 and result['active'] == {}
    assert result['counts']['ready' if valid else 'failed'] == 3
