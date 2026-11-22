"""RAB USD viewer. Reuses FindingBench flight UI without constructing BEHAVIOR."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import threading
import time
from http.server import ThreadingHTTPServer

from rab_spectator_data import current, digest, load, save
from spectator import FlightCamera, Handler, SharedState, run_flight, usd_world_fingerprint

HEAD = '/World/Robot/head_link2/Head_Camera'
CAMERA = '/FindingBenchView'


def build_cache(scene, target, source=None):
    """Compose local layers only; the released meshes/textures stay shared."""
    from pxr import Sdf, Usd, UsdUtils
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    sha = digest(scene)
    meta = target / 'cache.json'
    out = target / (sha + '.thin.usdc')
    source = source if source is not None else Usd.Stage.Open(str(scene))
    if not source:
        raise ValueError('RAB USD could not be opened')
    # The room already has a shared, resolved USD composition. Compile the
    # task layer without copying 50–90 MB of room data into each cache entry.
    dependencies = {l.realPath: [Path(l.realPath).stat().st_size,
                    Path(l.realPath).stat().st_mtime_ns] for l in source.GetUsedLayers() if l.realPath}
    from uuid import uuid4
    temporary = target / (uuid4().hex + '.usdc')
    try:
        root=source.GetRootLayer()
        layer=Sdf.Layer.CreateAnonymous('rab_spectator.usda')
        layer.TransferContent(root)
        UsdUtils.ModifyAssetPaths(layer,lambda p:Sdf.ComputeAssetPathRelativeToLayer(root,p))
        if not layer.Export(str(temporary)):
            raise ValueError('Could not export RAB render cache')
        check = Usd.Stage.Open(str(temporary))
        for path in (HEAD, '/World/FindingBench'):
            if not check or not check.GetPrimAtPath(path):
                raise ValueError('RAB render cache lost a scene binding')
        os.replace(temporary, out)
        old=None
        if meta.exists():
            try:
                old=Path(load(meta)['scene'])
            except (ValueError,OSError,KeyError,TypeError):
                pass
        save(meta, dict(version=2,source_sha256=sha, scene=str(out), cache_sha256=digest(out), layers=dependencies))
        # Retire only the earlier flat cache generated for this exact task/hash.
        if old == target/(sha+'.usdc'):
            old.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def cache_lookup(scene, target):
    path = Path(target) / 'cache.json'
    try:
        meta = load(path)
        if meta.get('version')!=2 or meta['source_sha256'] != digest(scene):
            return None
        cache = Path(meta['scene'])
        if cache.parent != Path(target) or digest(cache) != meta['cache_sha256']:
            return None
        for item, (size, modified) in meta['layers'].items():
            stat = Path(item).stat()
            if stat.st_size != size or stat.st_mtime_ns != modified:
                return None
        return cache
    except (OSError, ValueError, KeyError, TypeError):
        return None


def home(stage):
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(HEAD)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    look = transform.TransformDir(Gf.Vec3d(0,0,-1)).GetNormalized()
    return FlightCamera(transform.ExtractTranslation(),
                        math.degrees(math.atan2(look[1],look[0])),
                        math.degrees(math.asin(max(-1.,min(1.,look[2])))))


class Renderer:
    def __init__(self, stage, width, height, fov):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        self.rep,self.Gf = rep,Gf
        self.shape = height,width
        camera = UsdGeom.Camera.Define(stage, CAMERA)
        camera.CreateHorizontalApertureAttr(20.)
        camera.CreateVerticalApertureAttr(20.*height/width)
        self.focal = camera.CreateFocalLengthAttr(10/math.tan(math.radians(fov/2)))
        self.default_focal = self.focal.Get()
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005,120.))
        self.matrix = camera.MakeMatrixXform()
        self.product = rep.create.render_product(CAMERA, (width,height))
        self.rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        self.rgb.attach([self.product])

    def close(self):
        self.rgb.detach()
        self.product.destroy()

    def capture(self,camera):
        import numpy as np
        Gf = self.Gf
        q = camera.usd_quaternion()
        x = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[3]),Gf.Vec3d(*q[:3])))
        x.SetTranslateOnly(Gf.Vec3d(*camera.position))
        self.matrix.Set(x)
        self.focal.Set(self.default_focal*getattr(camera,'zoom',1.))
        for _ in range(12):
            self.rep.orchestrator.step(rt_subframes=2,pause_timeline=True,delta_time=0.)
            frame=np.asarray(self.rgb.get_data())
            if frame.shape[:2]==self.shape and frame.ndim==3:
                return frame[:,:,:3].astype(np.uint8)
        raise RuntimeError('No valid RAB RGB frame')


def run(args, task):
    from spectator_editor import EditorHandler, EditorState
    episode, receipt, member, revision = current(args.asset_run, task, args.rab_edits.resolve())
    state = EditorState(task) if args.edit else SharedState(task)
    handler = EditorHandler if args.edit else Handler
    if args.skills:
        from rab_skills import SkillHandler, SkillState
        state, handler = SkillState(task), SkillHandler
    handler.state = state
    server = ThreadingHTTPServer((args.host,args.port),handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_: setattr(state,'stopped',True))
    start=time.monotonic()
    report=dict(task=task, scene_sha256=episode['scene_sha256'], revision=revision,
                status='loading', _started=start, mode='rab_usd', physics_tick=0)
    print(f'RAB {task["id"]} · http://{args.host}:{args.port} · loading',flush=True)
    app=world=renderer=None
    try:
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
        # Native plugins create thousands of watchers in a large release.
        os.environ.setdefault('PXR_WORK_THREAD_LIMIT','4')
        # Preload conda env's warp 1.12+ BEFORE SimulationApp prepends Isaac Sim's
        # older bundled omni.warp.core extension (required by NVLabs cuRoboV2-Fast).
        if args.skills:
            try:
                import warp as _wp  # noqa: F401
            except Exception:
                pass
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=args.width,height=args.height,multi_gpu=False,
               limit_cpu_threads=8,extra_args=['--/log/level=error']))
        import carb.settings
        import omni.timeline
        import omni.usd
        settings=carb.settings.get_settings()
        for name in ('updateToUsd','updateVelocitiesToUsd'):
            settings.set_bool('/physics/'+name,True)
        settings.set_bool('/physics/suppressReadback',False)
        ctx=omni.usd.get_context()
        scene=episode['scene']
        if not (args.edit or args.skills or args.no_cache):
            target=member/'render_cache'/episode['scene_sha256']
            cache=None if args.rebuild_cache else cache_lookup(scene,target)
            if cache is None:
                cache=build_cache(scene,target)
                report['cache']='built'
            else:
                report['cache']='hit'
            scene=str(cache)
        if not ctx.open_stage(scene):
            raise ValueError('Could not open RAB scene')
        stage=ctx.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        timeline=omni.timeline.get_timeline_interface()
        timeline.stop()
        start_time=timeline.get_current_time()
        renderer=Renderer(stage,args.width,args.height,args.fov)
        camera=home(stage)
        report['home']=camera.home
        if args.edit or args.skills:
            from rab_edit import NativeWorld
            world=NativeWorld(stage,episode,mobile=args.skills)
            if args.edit:
                from rab_edit import Editor
                state.editor=Editor(args,task,state,world,member,revision)
                report['mode']='rab_editor'
            else:
                from rab_skills import Skills
                from pxr import UsdGeom
                # The head camera is recessed in the head geometry during
                # wide virtual gaze. Suppress own-body rendering in this
                # sensor, matching point rays which ignore the robot.
                UsdGeom.Imageable(stage.GetPrimAtPath(world.robot_prim)).MakeInvisible()
                report['mode']='rab_skills'
                state.editor=Skills(args,task,state,world,renderer,camera,receipt)
        before=None if world else usd_world_fingerprint(stage)
        def audit(full):
            if timeline.is_playing() or timeline.get_current_time()!=start_time:
                raise RuntimeError('Unexpected Kit timeline advance')
            if full and before and usd_world_fingerprint(stage)!=before:
                raise RuntimeError('Spectator changed the RAB world')
        def capture(c):
            if world:
                report['physics_tick']=world.steps
            return renderer.capture(c)
        run_flight(args,state,report,camera,capture,audit)
        report['timeline_time_unchanged']=True
        if before:
            report['world_unchanged']=usd_world_fingerprint(stage)==before
    except Exception as e:
        report.update(status='failed',error=str(e))
        with state.condition:
            state.status['error']=str(e)
        raise
    finally:
        state.stopped=True
        with state.condition:
            state.condition.notify_all()
        server.shutdown()
        server.server_close()
        if world:
            world.close()
        if args.smoke_output:
            report.pop('_started',None)
            save(args.smoke_output/'report.json',report)
        if renderer:
            try:
                renderer.close()
            except Exception as e:
                print(f'Render product cleanup: {e}', flush=True)
        if app:
            # Kit 5.1 can crash in extension teardown even after the server,
            # render product and attached PhysX stage have been released.
            # Like the native RAB batch workers this is a dedicated process.
            if report['status']=='failed':
                import traceback
                traceback.print_exc()
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if report['status']=='passed' else 1)
    return 0
