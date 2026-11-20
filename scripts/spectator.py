#!/usr/bin/env python3
"""Inspect a frozen FindingBench task with a collision-free browser camera.

    conda activate ygbench
    python scripts/spectator.py --task P007
    python scripts/spectator.py --task 1

This is privileged dataset inspection, not a benchmark observation or human
oracle run. Only the camera moves; no skills or physics steps are executed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
KEYS = frozenset(('KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'ControlLeft',
                  'ControlRight', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight'))


def select_task(directory, query):
    directory=Path(directory)
    if (directory/'retired.json').exists():
        raise ValueError('旧 BEHAVIOR 数据已按要求退役；当前正在迁移 RAB。请使用已发布的 RAB 数据目录，迁移进度见 runs/rab_migration_20261118/status.json。')
    tasks = json.loads((directory / 'human_ready.json').read_text())['tasks']
    query = str(query).strip()
    if query.isdigit():
        index = int(query)
        if not 1 <= index <= len(tasks):
            raise ValueError(f'Task index must be 1..{len(tasks)}')
        return dict(tasks[index - 1], index=index)
    matches = [(i, t) for i, t in enumerate(tasks, 1)
               if query.lower() in (t['id'].lower(), t['parent_task_id'].lower())]
    if len(matches) != 1:
        raise ValueError(f'Expected one task for {query!r}; found {len(matches)}')
    index, task = matches[0]
    return dict(task, index=index)


def enter_runtime(task, args):
    """Execute this viewer against the member's immutable native source tree."""
    runtime = Path(task['runtime_root']).resolve()
    if not (runtime / 'src/rummagebench').is_dir():
        raise ValueError(f'Pinned runtime is missing: {runtime}')
    # Do not let flags from a previous generation shell contaminate old members.
    for key in list(os.environ):
        if key.startswith('YGBENCH_'):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in task.get('runtime_environment', {}).items()})
    if 'OMNIGIBSON_DATA_PATH' not in os.environ:
        for parent in (ROOT, *ROOT.parents):
            data = parent / 'behavior-datasets'
            if all((data / name).is_dir() for name in
                   ('behavior-1k-assets', 'omnigibson-robot-assets')):
                os.environ['OMNIGIBSON_DATA_PATH'] = str(data)
                break
    compiler = Path(sys.prefix) / 'bin/x86_64-conda-linux-gnu-c++'
    if compiler.exists():
        os.environ.setdefault('CXX', str(compiler))
    os.environ['RUMMAGEBENCH_ROOT'] = str(runtime)
    os.environ['OMNIGIBSON_HEADLESS'] = 'True'
    os.environ['PYTHONPATH'] = str(runtime / 'src')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    os.chdir(runtime)
    os.execv(sys.executable, [sys.executable, '-u', str(Path(__file__).resolve()),
                            *args, '--_native'])


class FlightCamera:
    """World Z is up; yaw=0 looks along +X, positive yaw looks left."""
    def __init__(self, position, yaw=0., pitch=0.):
        self.home = ([float(v) for v in position], float(yaw), float(pitch))
        self.reset()

    def reset(self):
        self.position = list(self.home[0])
        self.yaw, self.pitch = self.home[1:]

    def basis(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        forward = (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))
        right = (math.sin(y), -math.cos(y), 0.)
        up = (-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p))
        return forward, right, up

    def advance(self, keys, dx, dy, dt, speed, reset=False):
        if reset:
            self.reset()
            return
        self.yaw = (self.yaw - dx * .12 + 180) % 360 - 180
        self.pitch = max(-89., min(89., self.pitch - dy * .12))
        forward, right, _ = self.basis()
        fb = int('KeyW' in keys) - int('KeyS' in keys)
        lr = int('KeyD' in keys) - int('KeyA' in keys)
        ud = int(bool(keys & {'Space', 'KeyE'})) - int(bool(keys & {'ShiftLeft', 'ShiftRight', 'KeyQ'}))
        v = [fb * forward[i] + lr * right[i] + (ud if i == 2 else 0) for i in range(3)]
        norm = math.sqrt(sum(x*x for x in v))
        boost = 3 if keys & {'ControlLeft', 'ControlRight'} else 1
        if norm:
            for i in range(3):
                self.position[i] += v[i] / norm * speed * boost * min(max(dt, 0.), .1)

    def usd_quaternion(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        forward, right, up = self.basis()
        # USD cameras look along local -Z, with local +Y up.
        return Rotation.from_matrix(np.column_stack((right, up, -np.asarray(forward)))).as_quat()


class SharedState:
    def __init__(self, task):
        self.condition = threading.Condition()
        self.task = task
        self.keys, self.dx, self.dy = set(), 0., 0.
        self.updated = 0.
        self.speed = 1.
        self.reset = False
        self.stopped = False
        self.jpeg = None
        self.seq = 0
        self.status = dict(ready=False, error=None, task=task,
                           mode='Privileged inspection; physics frozen')

    def receive(self, body):
        if not isinstance(body, dict) or not isinstance(body.get('keys', []), list):
            raise ValueError('Expected a control object with a keys array')
        keys = body.get('keys', [])
        if len(keys) > 20 or any(not isinstance(k, str) or k not in KEYS for k in keys):
            raise ValueError('Unknown movement key')
        values = [body.get('dx', 0), body.get('dy', 0), body.get('speed', 1)]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError('Non-finite or invalid camera input')
        dx, dy, speed = values
        if abs(dx) > 20000 or abs(dy) > 20000 or not .05 <= speed <= 8:
            raise ValueError('Camera input outside accepted range')
        if not isinstance(body.get('reset', False), bool):
            raise ValueError('reset must be boolean')
        with self.condition:
            self.keys = set(keys)
            # Coalesce relative mouse motion; never build an input queue.
            self.dx += dx
            self.dy += dy
            self.speed = speed
            self.reset |= body.get('reset', False)
            self.updated = time.monotonic()

    def consume(self):
        with self.condition:
            if time.monotonic() - self.updated > .5:
                self.keys.clear()  # lost focus/disconnected browser cannot fly forever
            result = (set(self.keys), self.dx, self.dy, self.speed, self.reset)
            self.dx = self.dy = 0.
            self.reset = False
            return result

    def publish(self, jpeg, **status):
        with self.condition:
            self.jpeg = jpeg
            self.seq += 1
            self.status.update(status, seq=self.seq)
            self.condition.notify_all()


class Handler(BaseHTTPRequestHandler):
    state: SharedState
    def log_message(self, *args):
        pass

    def reply(self, payload, content_type='application/json', code=200):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = urlsplit(self.path).path
        try:
            if route == '/':
                self.reply(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
            elif route == '/state':
                with self.state.condition:
                    status = dict(self.state.status)
                self.reply(status)
            elif route == '/frame.jpg':
                with self.state.condition:
                    jpeg = self.state.jpeg
                if jpeg is None:
                    self.reply({'error': 'Scene is loading'}, code=503)
                else:
                    self.reply(jpeg, 'image/jpeg')
            elif route == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                seq = 0
                while not self.state.stopped:
                    with self.state.condition:
                        self.state.condition.wait_for(lambda: self.state.seq != seq or self.state.stopped, timeout=1)
                        jpeg, current = self.state.jpeg, self.state.seq
                    if jpeg is None or current == seq:
                        continue
                    seq = current
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                     str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                    self.wfile.flush()
            else:
                self.reply({'error': 'Not found'}, code=404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if urlsplit(self.path).path != '/input':
            return self.reply({'error': 'Not found'}, code=404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('Input body must be 1..4096 bytes')
            self.state.receive(json.loads(self.rfile.read(size)))
            self.reply({'ok': True})
        except (ValueError, TypeError) as e:
            self.reply({'error': str(e)}, code=400)


def physical_fingerprint(backend):
    """Camera/sensor state excluded; object and articulated physical state included."""
    import numpy as np
    result = hashlib.sha256()
    for obj in sorted(backend._env.scene.objects, key=lambda o: o.name):
        result.update(obj.name.encode())
        values = [*obj.get_position_orientation()]
        # Native fixed/kinematic links have no velocity getter; BaseObject's
        # delegating getter exists but raises on those assets.
        for name, link in sorted(obj.links.items()):
            result.update(name.encode())
            values.extend(link.get_position_orientation())
            if hasattr(link, 'get_linear_velocity'):
                values.extend((link.get_linear_velocity(), link.get_angular_velocity()))
        if obj.n_joints:
            values.extend((obj.get_joint_positions(), obj.get_joint_velocities()))
        for value in values:
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            result.update(np.asarray(value).tobytes())
    return result.hexdigest()


def run_flight(args, state, report, camera, capture, audit):
    """The same input/render loop for native worlds and their baked USD views."""
    from PIL import Image
    stop_at = time.monotonic() + args.smoke_seconds if args.smoke_seconds else None
    last = last_audit = time.monotonic()
    while not state.stopped:
        now = time.monotonic()
        if stop_at and now >= stop_at:
            break
        keys, dx, dy, speed, reset = state.consume()
        camera.advance(keys, dx, dy, now - last, speed, reset)
        last = now
        if hasattr(state, 'editor'):
            state.editor.tick(camera)
        rgb = capture(camera)
        audit(now - last_audit > 5)
        if now - last_audit > 5:
            last_audit = now
        if args.smoke_output and state.seq == 0:
            args.smoke_output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.smoke_output / 'initial.png')
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format='JPEG', quality=90)
        if state.seq == 0:
            report['first_frame_seconds'] = time.monotonic() - report.pop('_started')
            print(f'Ready in {report["first_frame_seconds"]:.1f}s: http://{args.host}:{args.port} · click image to fly; Ctrl+C to quit', flush=True)
        state.publish(buffer.getvalue(), ready=True, position=camera.position.copy(),
                      yaw=camera.yaw, pitch=camera.pitch, physics_tick=report['physics_tick'],
                      mode=report['mode'], render_fps=1 / max(time.monotonic() - now, .001))
        time.sleep(max(0, 1 / args.fps - (time.monotonic() - now)))
    audit(True)
    report.update(status='passed', frames=state.seq, final_camera_position=camera.position,
                  final_camera_yaw=camera.yaw, final_camera_pitch=camera.pitch)
    if args.smoke_output and state.seq:
        Image.fromarray(rgb).save(args.smoke_output / 'final.png')


def usd_world_fingerprint(stage):
    import numpy as np
    from pxr import UsdGeom
    result = hashlib.sha256()
    transforms = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World') and UsdGeom.Xformable(prim):
            result.update(str(prim.GetPath()).encode())
            result.update(np.asarray(transforms.GetLocalToWorldTransform(prim)).tobytes())
    return result.hexdigest()


def cached_viewer(args, task, cached):
    """Render a baked scene directly in Kit; never construct a physics world."""
    import importlib.util
    import numpy as np
    root, meta = cached
    state = SharedState(task)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(state, 'stopped', True))
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    started = time.monotonic()
    report = dict(task=task, status='loading', mode='render_cache', cache=str(root),
                  physics_tick=meta['source_physics_tick'], _started=started)
    app = None
    print(f'Spectator: http://{args.host}:{args.port} · loading render cache (no physics)', flush=True)
    try:
        sdk = Path(importlib.util.find_spec('omnigibson').origin).parent
        os.environ['MDL_USER_PATH'] = str(sdk / 'materials')
        os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height,
                             'limit_cpu_threads': 8, 'extra_args': ['--/log/level=error']})
        print(f'[cache] Kit ready: {time.monotonic() - started:.1f}s', flush=True)
        import carb.settings
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        settings = carb.settings.get_settings()
        for key, value in {
            '/rtx/rtx/modes/rt/enabled': True, '/rtx/rtx/modes/rt2/enabled': True,
            '/rtx/rendermode': 'RealTimePathTracing',
            '/rtx/raytracing/fractionalCutoutOpacity': True,
            '/rtx/reflections/enabled': True, '/rtx/indirectDiffuse/enabled': True,
            '/rtx/post/dlss/execMode': 0, '/rtx/ambientOcclusion/enabled': True,
            '/rtx/directLighting/sampledLighting/enabled': True,
            '/rtx/raytracing/showLights': 1, '/rtx/sceneDb/ambientLightIntensity': 1.,
            '/app/renderer/skipMaterialLoading': False,
        }.items():
            settings.set(key, value)
        context = omni.usd.get_context()
        print('[cache] Opening baked USD scene', flush=True)
        if not context.open_stage(str(root / 'scene.usdc')):
            raise RuntimeError('Could not open spectator render cache')
        stage = context.get_stage()
        print(f'[cache] USD opened: {time.monotonic() - started:.1f}s', flush=True)
        # This timeline stays stopped; loading USD is not simulator replay.
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline_start = timeline.get_current_time()
        camera = FlightCamera(*meta['home'])
        prim = UsdGeom.Camera.Define(stage, '/findingbench_cached_camera')
        prim.CreateFocalLengthAttr(10 / math.tan(math.radians(args.fov / 2)))
        prim.CreateHorizontalApertureAttr(20.)
        prim.CreateVerticalApertureAttr(20. * args.height / args.width)
        prim.CreateClippingRangeAttr(Gf.Vec2f(.005, 1000))
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate = xform.AddTranslateOp()
        orient = xform.AddOrientOp()
        product = rep.create.render_product(str(prim.GetPath()), (args.width, args.height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb_annotator.attach([product])
        print(f'[cache] RGB product attached: {time.monotonic() - started:.1f}s', flush=True)
        before_world = usd_world_fingerprint(stage)

        def capture(camera):
            translate.Set(Gf.Vec3d(*camera.position))
            q = camera.usd_quaternion()
            orient.Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.)
            rgb = np.asarray(rgb_annotator.get_data())
            # Shader compilation / first texture uploads can take extra frames.
            for _ in range(120 if state.seq == 0 else 0):
                if rgb.shape[:2] == (args.height, args.width):
                    break
                rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.)
                rgb = np.asarray(rgb_annotator.get_data())
            if rgb.shape[:2] != (args.height, args.width):
                raise RuntimeError(f'Cached camera returned invalid RGB shape {rgb.shape}')
            return rgb[..., :3].astype(np.uint8)

        def audit(full):
            if timeline.is_playing():
                raise RuntimeError('Render-only inspection unexpectedly started the timeline')
            if timeline.get_current_time() != timeline_start:
                raise RuntimeError('Render-only inspection unexpectedly advanced time')
            if full and usd_world_fingerprint(stage) != before_world:
                raise RuntimeError('Render-only inspection unexpectedly changed scene transforms')

        # Allow textures / materials to populate before announcing readiness.
        for _ in range(24):
            app.update()
        print(f'[cache] Renderer warm-up finished: {time.monotonic() - started:.1f}s', flush=True)
        report.update(load_seconds=time.monotonic() - started,
                      native_source_physical_sha256=meta['native_physical_sha256'],
                      initial_render_world_sha256=before_world,
                      baked_nodes=meta['baked_nodes'], max_world_matrix_error=meta['max_world_matrix_error'])
        run_flight(args, state, report, camera, capture, audit)
        report['timeline_time_unchanged'] = True
        report['final_render_world_sha256'] = usd_world_fingerprint(stage)
    except KeyboardInterrupt:
        report['status'] = 'stopped'
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='failed', error=str(error))
        raise
    finally:
        state.stopped = True
        with state.condition:
            state.condition.notify_all()
        server.shutdown()
        server.server_close()
        report.pop('_started', None)
        if args.smoke_output:
            args.smoke_output.mkdir(parents=True, exist_ok=True)
            (args.smoke_output / 'report.json').write_text(json.dumps(report, indent=2))
        if app is not None:
            app.close()
    return 0


def native_viewer(args, task):
    # Imports MUST occur after re-exec into the selected member's pinned runtime.
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from rummagebench.core.scenario import load_scenario
    from rummagebench.adapters.python_api import prepare_renderer_grounding
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.raw_instance import _array, _camera_in_sync, wait_for_intrinsics

    if getattr(args, 'edit', False):
        from spectator_editor import EditorState, EditorHandler
        state = EditorState(task)
        handler = type('DatasetEditorHandler', (EditorHandler,), {'state': state})
    else:
        state = SharedState(task)
        handler = type('DatasetSpectatorHandler', (Handler,), {'state': state})
    # Kit graph callbacks can swallow KeyboardInterrupt. Set a Python stop
    # flag instead, and let the render thread leave through its own finally.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(state, 'stopped', True))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'Spectator: http://{args.host}:{args.port} (loading {task["parent_task_id"]})', flush=True)
    backend = OmniGibsonBackend(seed=task.get('seed', 0))
    started = time.monotonic()
    report = {'task': task, 'status': 'loading', 'mode': 'native', '_started': started}
    try:
        scenario = Path(task['play_scenario'])
        if not scenario.is_absolute():
            scenario = ROOT / scenario
        spec = load_scenario(scenario)
        if spec.scene.frozen_member_manifest is None:
            raise ValueError('Spectator requires a frozen scene; authoring/resampling is disabled')
        prepare_renderer_grounding(spec)
        backend.setup(spec)
        from omnigibson.sensors import VisionSensor
        sim = backend._sim
        before_tick = sim.current_time_step_index
        before_world = physical_fingerprint(backend)
        panorama = getattr(backend, '_panorama_camera', None)
        if panorama is not None:
            position, quaternion = panorama.faces[0][0].get_position_orientation()
        else:
            position, quaternion = backend.robot_pose()
            position = np.asarray(_array(position)) + [0, 0, 1.6]
        rotation = Rotation.from_quat(_array(quaternion)).as_matrix()
        forward = -rotation[:, 2] if panorama is not None else rotation[:, 0]
        camera = FlightCamera(_array(position), math.degrees(math.atan2(forward[1], forward[0])),
                              math.degrees(math.asin(float(np.clip(forward[2], -1, 1)))))
        sensor = VisionSensor(relative_prim_path='/findingbench_spectator_camera',
                              name='findingbench_spectator_camera', modalities=['rgb'],
                              image_width=args.width, image_height=args.height,
                              focal_length=10 / math.tan(math.radians(args.fov / 2)),
                              horizontal_aperture=20, clipping_range=(.005, 1000))
        sensor.load(backend._env.scene)
        sensor.initialize()
        wait_for_intrinsics(sensor, sim)
        # The restored benchmark has six panorama products plus robot cameras.
        # Inspection only displays this RGB camera; avoid rendering all the
        # unused RGB/depth/segmentation products on every mouse movement.
        paused_products = 0
        with sim.editing_usd():
            for other in VisionSensor.SENSORS.values():
                if other is not sensor:
                    other.render_product.hydra_texture.set_updates_enabled(False)
                    paused_products += 1
        report.update(load_seconds=time.monotonic() - started, physics_tick=before_tick,
                      initial_physical_sha256=before_world, paused_render_products=paused_products)
        if getattr(args, 'edit', False):
            from spectator_editor import NativeEditor
            state.editor = NativeEditor(args, task, spec, backend, state, camera, report, sensor)
            report['mode'] = 'native_editor'
        if not args.no_cache and not getattr(args, 'edit', False):
            from spectator_cache import export
            try:
                cache_root, cache_meta = export(args.cache_dir, task, backend, camera, report)
                report.update(cache=str(cache_root), cache_export_seconds=cache_meta['export_seconds'])
                print(f'Render cache saved ({cache_meta["export_seconds"]:.1f}s); subsequent loads skip physics setup', flush=True)
            except Exception as error:
                report['cache_export_error'] = str(error)
                print(f'Render cache unavailable; continuing native inspection: {error}', flush=True)

        def capture(camera):
            sensor.set_position_orientation(position=np.asarray(camera.position), orientation=camera.usd_quaternion())
            # Renderer lag is measured, not hidden behind physics steps.
            for i in range(12):
                sim.render()
                if i >= 1 and _camera_in_sync(sensor):
                    break
            else:
                raise RuntimeError('Spectator camera failed renderer pose synchronization')
            data, _ = sensor.get_obs()
            if getattr(args, 'edit', False):
                state.editor.frame(data)
            rgb = np.asarray(_array(data['rgb']))[..., :3].astype(np.uint8)
            if rgb.shape != (args.height, args.width, 3):
                raise RuntimeError(f'Unexpected spectator image shape: {rgb.shape}')
            return rgb

        def audit(full):
            if getattr(args, 'edit', False):
                return state.editor.audit(full)
            if sim.current_time_step_index != before_tick:
                raise RuntimeError('Inspection unexpectedly advanced physics')
            if full and physical_fingerprint(backend) != before_world:
                raise RuntimeError('Inspection unexpectedly changed object/robot state')

        run_flight(args, state, report, camera, capture, audit)
        after_world = physical_fingerprint(backend)
        if not getattr(args, 'edit', False) and after_world != before_world:
            raise RuntimeError('Inspection changed physical state')
        report.update(final_physical_sha256=after_world, final_physics_tick=sim.current_time_step_index)
    except KeyboardInterrupt:
        report['status'] = 'stopped'
    except Exception as error:
        import traceback
        traceback.print_exc()
        state.status.update(error=f'{type(error).__name__}: {error}')
        report.update(status='failed', error=state.status['error'])
        raise
    finally:
        state.stopped = True
        with state.condition:
            state.condition.notify_all()
        server.shutdown()
        server.server_close()
        report.pop('_started', None)
        if hasattr(state, 'editor'):
            state.editor.close()
        if args.smoke_output:
            args.smoke_output.mkdir(parents=True, exist_ok=True)
            (args.smoke_output / 'report.json').write_text(json.dumps(report, indent=2))
        backend.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', '--pick', help='Dataset index (1..1000), parent ID (P007), or instance ID')
    from rab_spectator_data import DEFAULT_RELEASE, DEFAULT_OVERLAYS, is_release
    parser.add_argument('--asset-run', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--rab-edits', type=Path, default=DEFAULT_OVERLAYS,
                        help='Separate RAB editor revisions and render caches')
    parser.add_argument('--skills', action='store_true', help='Use the RAB embodied skills interface')
    parser.add_argument('--run-output', type=Path, help='New directory for skill actions and rendered observations')
    parser.add_argument('--list', action='store_true', help='List entries without loading the simulator')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--fov', type=float, default=75)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'build/spectator_cache')
    parser.add_argument('--rebuild-cache', action='store_true', help='Rebuild the render snapshot using the native simulator')
    parser.add_argument('--no-cache', action='store_true', help='Use the full native load without reading/writing render caches')
    parser.add_argument('--edit', action='store_true', help='Edit native assets/layout and save this dataset entry with a rebuilt cache')
    parser.add_argument('--_native', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--smoke-seconds', type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--smoke-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.asset_run = args.asset_run.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if args.smoke_output:
        args.smoke_output = args.smoke_output.resolve()
    if not (160 <= args.width <= 3840 and 120 <= args.height <= 2160 and
            1 <= args.fps <= 60 and 20 <= args.fov <= 120 and 1 <= args.port <= 65535):
        parser.error('Invalid resolution, frame rate, FOV or port')
    rab = is_release(args.asset_run)
    if args.edit and args.skills:
        parser.error('--edit and --skills are separate modes')
    if args.list:
        if rab:
            from rab_spectator_data import list_tasks
            tasks = list_tasks(args.asset_run)
        else:
            tasks = json.loads((args.asset_run / 'human_ready.json').read_text())['tasks']
        for i, task in enumerate(tasks, 1):
            print(f'{i:4}  {task["parent_task_id"]}  [{task["scene"]}] {task["instruction"]}')
        return 0
    if args.task is None:
        args.task = input('Task index or ID (e.g. 1 / F001 / P007): ').strip()
    try:
        if rab:
            from rab_spectator_data import select
            task = select(args.asset_run, args.task, args.rab_edits)
        else:
            task = select_task(args.asset_run, args.task)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if rab:
        from rab_spectator import run
        return run(args, task)
    if args.skills:
        parser.error('--skills requires a verified RAB release')
    if not args._native:
        print(f'Inspecting #{task["index"]} {task["parent_task_id"]}: {task["instruction"]}', flush=True)
        forwarded = ['--task', str(args.task), '--asset-run', str(args.asset_run),
                     '--host', args.host, '--port', str(args.port), '--width', str(args.width),
                     '--height', str(args.height), '--fps', str(args.fps), '--fov', str(args.fov)]
        forwarded += ['--cache-dir', str(args.cache_dir)]
        if args.no_cache:
            forwarded += ['--no-cache']
        if args.edit:
            forwarded += ['--edit']
        if args.rebuild_cache:
            forwarded += ['--rebuild-cache']
        if args.smoke_seconds:
            forwarded += ['--smoke-seconds', str(args.smoke_seconds)]
        if args.smoke_output:
            forwarded += ['--smoke-output', str(args.smoke_output)]
        enter_runtime(task, forwarded)
        return 0
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if not args.edit and not args.no_cache and not args.rebuild_cache:
        from spectator_cache import lookup
        cached = lookup(args.cache_dir, task)
        if cached is not None:
            return cached_viewer(args, task, cached)
    return native_viewer(args, task)


if __name__ == '__main__':
    raise SystemExit(main())
