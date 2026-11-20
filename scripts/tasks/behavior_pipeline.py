"""Native BEHAVIOR layouts -> existing OmniGibson builder and public skills."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml
from _common import REPO, host_config
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.layout.spec import LayoutConfig
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.sim.omnigibson.behavior_adapter import prepare_layout, load_manifest, asset_record


def prepare(args):
    doc = yaml.safe_load(args.config.read_text())
    cfg = LayoutConfig.model_validate(doc['layout'])
    assets = args.assets or host_config()['behavior_assets_root']
    manifest = prepare_layout(cfg, doc['pool'], assets, args.seed)
    target = asset_record(assets, **doc['target'])
    manifest['assets'][f"{target['category']}/{target['model']}"] = target
    manifest['manifest_sha256'] = sha256_bytes(canonical_json({
        k: v for k, v in manifest.items() if k != 'manifest_sha256'}))
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('OUTPUT_NOT_EMPTY: use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / 'assembly.json', manifest)
    storage = [i for i in manifest['objects'] if i['placement']['role'] == 'storage']
    if not storage:
        raise ValueError('SMOKE_REQUIRES_STORAGE')
    robot = yaml.safe_load((REPO / 'configs/tasks/robots/r1pro.yaml').read_text())
    scenario = dict(id=f'{cfg.id}_{args.seed}', instruction=f"Find and pick up the {target['category'].replace('_', ' ')}.",
        scene=dict(model=cfg.base_scene, source='behavior_layout', assembly_manifest=str(out / 'assembly.json')),
        robot=dict(robot['robot'], init_anchor='start'), feasibility=robot['feasibility'],
        anchors=dict(start=dict(position=[*cfg.robot_spawn, .0053], orientation=[0., 0., 0., 1.])),
        target=dict(entity='search_target', category=target['category']),
        objects=[dict(name='search_target', category=target['category'], model=target['model'])],
        placements=[dict(entity='search_target', relation='inside', receptacle=storage[0]['placement']['name'])],
        initial_states={i['placement']['name']: {'open': False} for i in storage},
        skills=['NAV', 'OPEN', 'CLOSE', 'GRASP', 'PLACE'], expose_available_skills=False,
        termination=dict(max_planning_steps=48, fail_on_wrong_grasp=False))
    from rummagebench.core.scenario import ScenarioSpec
    ScenarioSpec.model_validate(scenario)
    (out / 'scenario.yaml').write_text(yaml.safe_dump(scenario, sort_keys=False))
    report = dict(status='pending_simulation', assembly=str(out/'assembly.json'),
                  instantiated_scenes=0, candidate_layouts=1, evaluation_ready_episodes=0,
                  target_asset=target, scope='native_asset_interface_smoke',
                  objects=len(manifest['objects']))
    atomic_write_json(out / 'prepare.json', report)
    print(json.dumps(report, indent=2))
    return 0


def prepare_living_home(args):
    from rummagebench.authoring.layout.indoor_background import prepare_home, native_start
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.skills.move import quat_from_yaw
    config = yaml.safe_load(args.config.read_text())
    root = (args.assets or host_config()['behavior_assets_root']).resolve()
    manifest = prepare_home(root, config['scene'], config['daily_assets'], args.seed,
        args.out, args.upstream, args.python, per_surface=config['per_surface'], steps=config['solver_steps'],
        living_brief=config.get('living_brief'), geometry=getattr(args,'geometry',None),
        timeout=config.get('solver_timeout',900 if getattr(args,'geometry',None) else 300))
    context = manifest['living_context']['native']
    candidates = [o for o in context['objects'] if config['room'] in o['rooms']
                  and 'cabinet' in o['category'] and o['openable']]
    selected = next(o for o in candidates if o['name'] == config['smoke_container'])
    target = asset_record(root, **config['target'])
    xy, start_inputs = native_start(root, config['scene'], config['room'])
    manifest['living_context']['inputs'].update(start_inputs)
    manifest['assets'][f"{target['category']}/{target['model']}"] = target
    manifest['search_candidates'] = [o['name'] for o in candidates]
    manifest['manifest_sha256'] = sha256_bytes(canonical_json({k:v for k,v in manifest.items() if k != 'manifest_sha256'}))
    out = args.out.resolve()
    atomic_write_json(out/'assembly.json', manifest)
    # The authored initial view must not reveal which cabinet holds the target.
    # Diagnostic interaction poses below are explicitly evaluator-only.
    yaw = float(config.get('start_yaw', 0.))
    robot = yaml.safe_load((REPO/'configs/tasks/robots/r1pro.yaml').read_text())
    scenario = dict(id=f"native_home_{config['scene']}_{args.seed}",
        instruction=f"Find and pick up the {target['category'].replace('_', ' ')} in the kitchen cabinets.",
        scene=dict(model=config['scene'], source='behavior_layout', assembly_manifest=str(out/'assembly.json')),
        robot=dict(robot['robot'],init_anchor='start'),feasibility=robot['feasibility'],
        anchors=dict(start=dict(position=[*xy,.0053],orientation=quat_from_yaw(yaw))),
        target=dict(entity='search_target',category=target['category']),
        objects=[dict(name='search_target',category=target['category'],model=target['model'])],
        placements=[dict(entity='search_target',relation='inside',receptacle=selected['name'])],
        initial_states={o['name']:{'open':False} for o in candidates},
        skills=['NAV','OPEN','CLOSE','GRASP','PLACE'], expose_available_skills=False,
        termination=dict(max_planning_steps=48,fail_on_wrong_grasp=False))
    if config.get('smoke_link') is not None:
        scenario['placements'][0]['link'] = config['smoke_link']
    ScenarioSpec.model_validate(scenario)
    (out/'scenario.yaml').write_text(yaml.safe_dump(scenario,sort_keys=False))
    report = dict(status='pending_simulation', native_objects=len(context['retained_native_objects']),
                  new_daily_objects=len(manifest['objects']), cabinet_candidates=len(candidates),
                  structural_variants=0, evaluation_ready_episodes=0,
                  scope='InfiniGen daily-object solve within preserved native home; furniture layout fixed')
    atomic_write_json(out/'prepare.json',report)
    print(json.dumps(report,indent=2))
    return 0


def export_collision_geometry(args):
    from rummagebench.sim.gpu_isolation import (resolve_gpu, isolated_gpu_command,
        verify_isolated_cuda, configure_single_gpu_kit)
    if not args.gpu_isolated:
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError('OUTPUT_NOT_EMPTY')
        args.out.mkdir(parents=True,exist_ok=True)
        command, env = isolated_gpu_command([sys.executable,str(Path(__file__).resolve()),
            *sys.argv[1:],'--gpu-isolated'],resolve_gpu(args.gpu))
        code=None;error=None
        try:
            code=subprocess.run(command,env=env,timeout=900,check=False).returncode
        except subprocess.TimeoutExpired:
            error='GEOMETRY_EXPORT_TIMEOUT'
        from rummagebench.authoring.layout.indoor_geometry import load_geometry
        try:
            doc=load_geometry(args.out/'collision_geometry.json')
            if code!=0:raise ValueError('GEOMETRY_EXPORT_PROCESS_FAILED')
            status='geometry_exported_pending_physics'
        except Exception as exc:
            status='failed';error=error or str(exc)
        atomic_write_json(args.out/'process.json',dict(status=status,returncode=code,error=error,
            gpu=args.gpu,seed=args.seed,config_sha256=sha256_path(args.config),runner_sha256=sha256_path(__file__),
            evaluation_ready_episodes=0))
        return 0 if status=='geometry_exported_pending_physics' else 1
    import torch
    verify_isolated_cuda(torch)
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    import isaacsim
    configure_single_gpu_kit(isaacsim)
    from rummagebench.authoring.layout.indoor_background import native_context,native_start
    from rummagebench.authoring.layout.indoor_geometry import export_geometry
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    config=yaml.safe_load(args.config.read_text())
    root=(args.assets or host_config()['behavior_assets_root']).resolve()
    context=native_context(root,config['scene'])
    records={f"{a['category']}/{a['model']}":asset_record(root,**a) for a in config['daily_assets']}
    xy,inputs=native_start(root,config['scene'],config['room']);context['inputs'].update(inputs)
    robot=yaml.safe_load((REPO/'configs/tasks/robots/r1pro.yaml').read_text())
    # No hidden target or target placement is loaded for geometry authoring.
    scenario=ScenarioSpec.model_validate(dict(id='native_geometry_export',instruction='Authoring only.',
        scene=dict(model=config['scene']),robot=dict(robot['robot'],init_anchor='start'),
        feasibility=robot['feasibility'],anchors=dict(start=dict(position=[*xy,.0053],orientation=[0,0,0,1])),
        target=dict(entity='authoring_unused',category=config['daily_assets'][0]['category']),
        objects=[dict(name='authoring_unused',**config['daily_assets'][0])],placements=[],
        expose_available_skills=False))
    backend=OmniGibsonBackend(seed=args.seed)
    try:
        backend.setup(scenario,native_scene_capture=lambda b:{'scope':'geometry_export'},native_scene_only=True)
        export_geometry(backend,context,records,args.out.resolve())
    finally:
        backend.close()
    return 0


def check_environment(backend, scenario, manifest, output):
    """Reset/render/checkpoint only. No actions, pose search or task policy."""
    from rummagebench.core.visual_session import VisualProtocolSession
    from rummagebench.sim.omnigibson.state_io import save_snapshot, load_snapshot
    from rummagebench.sim.omnigibson.probe_support import relation_holds
    from rummagebench.authoring.layout.indoor_realization import verify_after_target
    from PIL import Image
    import numpy as np

    session=VisualProtocolSession(backend,scenario)
    session.reset()  # existing reset validates physics, spawn and visual grounding
    baseline=json.loads((output/'background_realization.json').read_text())
    verify_after_target(backend,manifest,baseline,output,require_matched_background=False)
    checkpoint=output/'environment_start.pt'
    save_snapshot(checkpoint,backend.capture_observe_state(),meta=dict(
        assembly_sha256=manifest['manifest_sha256'],format='observe-state-with-spatial-cache-v1',
        scope='environment_with_target_at_declared_spawn'))
    state,meta=load_snapshot(checkpoint)
    backend.restore_observe_state(state)
    backend.validate_physics_state()
    backend._validate_spawn_clearance()
    if not relation_holds(backend,scenario.target.entity,scenario.placements[0].receptacle,'inside'):
        raise ValueError('ENVIRONMENT_TARGET_RELATION_LOST_ON_RESET_OR_RESTORE')
    frame=session._capture_current()
    rgb=np.asarray(frame.rgb)
    if rgb.ndim!=3 or min(rgb.shape[:2])<2 or not np.isfinite(rgb).all():
        raise ValueError('ENVIRONMENT_INVALID_CAMERA_FRAME')
    Image.fromarray(rgb).save(output/'environment_start.png')
    return dict(status='environment_ready',assembly_sha256=manifest['manifest_sha256'],
        checkpoint_sha256=sha256_path(checkpoint),checkpoint_format=meta['format'],
        image_sha256=sha256_path(output/'environment_start.png'),
        public_reset=True,spawn_clear=True,target_relation=True,policy_executed=False,
        scope='physical setup/reset/camera/checkpoint; solvability and search difficulty unmeasured')


def smoke(args):
    """Physical environment validation only; no baseline policy or action rollout."""
    from rummagebench.adapters.python_api import prepare_renderer_grounding, install_renderer_grounding
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.probe_support import relation_holds
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    import numpy as np

    scenario = load_scenario(args.scenario)
    # Initialize the existing public pixel interface without enumerating hidden
    # oracle actions. Baselines choose their own actions after environment reset.
    scenario.expose_available_skills = False
    manifest = load_manifest(Path(scenario.scene.assembly_manifest))
    for obj in scenario.objects:
        if f'{obj.category}/{obj.model}' not in manifest['assets']:
            raise ValueError('TASK_ASSET_NOT_BOUND_TO_ASSEMBLY')
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', scope='physical_environment_only',
                  evaluation_ready_episodes=0, assembly_sha256=manifest['manifest_sha256'],
                  scenario_sha256=sha256_path(args.scenario),
                  runner_sha256=sha256_path(__file__),
                  gpu=int(os.environ['FINDINGBENCH_GPU_INDEX']),
                  background_velocity_iterations=getattr(args,'background_velocity_iterations',None),
                  background_stability_profile=args.background_stability_profile,
                  acceptance_policy=args.acceptance_policy,
                  verify_background_restore=args.verify_background_restore,
                  environment_only=args.environment_only,
                  legacy_skill_enumeration=False, attempts=[])
    def save():
        atomic_write_json(args.out / 'result.json', report)
    save()
    backend = None
    try:
        prepare_renderer_grounding(scenario)
        backend = OmniGibsonBackend(seed=manifest['layout_seed'])
        setup_kwargs={}
        if getattr(args,'realize_background',False):
            from rummagebench.authoring.layout.indoor_realization import realize_background, parking_position
            parking=parking_position(manifest)
            def before_place(b):
                if manifest['living_context'].get('geometry'):
                    report['geometry_restore']=b._env._findingbench_geometry_restore
                    save()
                if args.background_velocity_iterations is not None:
                    from rummagebench.authoring.layout.indoor_dynamics import apply_velocity_iteration_experiment
                    report['physics_experiment']=apply_velocity_iteration_experiment(
                        b,manifest,args.out,args.background_velocity_iterations)
                    report['scope']='environment_with_diagnostic_physics_override'
                    save()
                try:
                    result = realize_background(b,manifest,args.out,
                        diagnostic_steps=360 if getattr(args,'diagnose_background',False) else 0,
                        stability_profile=args.background_stability_profile, acceptance_policy=args.acceptance_policy)
                except Exception:
                    if args.diagnose_motion_controls:
                        from rummagebench.authoring.layout.indoor_dynamics import motion_controls
                        try:
                            motion_controls(b,manifest,args.out/'motion_controls',
                                physics_steps=args.background_stability_profile=='observed-motion-v3')
                        except Exception as exc:
                            report['motion_controls_error']=f'{type(exc).__name__}:{exc}'
                            save()
                    raise
                if args.diagnose_motion_controls:
                    from rummagebench.authoring.layout.indoor_dynamics import motion_controls
                    motion_controls(b,manifest,args.out/'motion_controls',
                        physics_steps=args.background_stability_profile=='observed-motion-v3')
                    raise ValueError('MOTION_CONTROLS_DIAGNOSTIC_ONLY')
                if args.verify_background_restore:
                    from rummagebench.authoring.layout.indoor_realization import verify_background_restore
                    report['background_restore']=verify_background_restore(b,manifest,args.out,acceptance_policy=args.acceptance_policy)
                    save()
                return result
            setup_kwargs=dict(task_seed_pose=lambda b,s:parking,
                pre_placement_capture=before_place)
            report['background_realizer_sha256']=sha256_path(REPO/'src/rummagebench/authoring/layout/indoor_realization.py')
            report['target_independent_parking']=parking
            save()
        report['builder'] = backend.setup(scenario,**setup_kwargs)
        if getattr(args,'realize_background',False):
            from rummagebench.authoring.layout.indoor_realization import verify_after_target
            report['background_after_target']=verify_after_target(backend,manifest,
                report['builder']['frozen_background'],args.out,require_matched_background=False)
            save()
        install_renderer_grounding(backend)
        backend.render_snapshot(str(args.out/'scene_start.png'))
        from omnigibson.object_states.contains import m as contains_macros
        inventory = []
        for item in manifest['objects']:
            obj = resolve_object(backend._env.scene, item['placement']['name'])
            lo, hi = (np.asarray(v.cpu() if hasattr(v, 'cpu') else v, float) for v in obj.aabb)
            inventory.append(dict(name=obj.name, category=obj.category, model=obj.model,
                                  aabb=[lo.tolist(), hi.tolist()],
                                  fillable_links=[name for name, link in obj.links.items()
                                      if link.is_meta_link and link.meta_link_type in contains_macros.CONTAINER_META_LINK_TYPES],
                                  joints=list(obj.joints), links=list(obj.links)))
            if item['placement']['role'] == 'clutter':
                if not relation_holds(backend, obj.name, item['placement']['support_name'], 'on_top'):
                    raise ValueError(f'BACKGROUND_SUPPORT_FAILED:{obj.name}')
        report['loaded_objects'] = inventory
        target = scenario.target.entity
        container = scenario.placements[0].receptacle
        if not relation_holds(backend, target, container, 'inside'):
            raise ValueError('TARGET_INSIDE_FAILED')
        expected=set(manifest['living_context']['native']['retained_native_objects'])
        if not expected <= set(backend.entity_names()):
            raise ValueError('NATIVE_HOUSE_OBJECTS_LOST')
        report['retained_native_objects']=len(expected)
        evidence=check_environment(backend,scenario,manifest,args.out)
        atomic_write_json(args.out/'environment_ready.json',evidence)
        report.update(status='environment_ready',scope=evidence['scope'],environment=evidence)
        save()
        return 0
    except Exception as exc:
        report.update(status='environment_failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    finally:
        # Kit must shut down before Python tears down renderer extensions.
        # SDK shutdown may exit zero; the parent still requires the receipt.
        if backend is not None:
            backend.close()


def main():
    ap = argparse.ArgumentParser(__doc__)
    subs = ap.add_subparsers(dest='stage', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--config', type=Path, default=REPO/'configs/layouts/behavior_native_smoke.yaml')
    p.add_argument('--assets', type=Path)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('prepare-home', help='CPU: InfiniGen daily-object constraints in a complete native house')
    p.add_argument('--config',type=Path,default=REPO/'configs/layouts/behavior_home_v1.yaml')
    p.add_argument('--assets',type=Path)
    p.add_argument('--upstream',type=Path,default=Path('/data1/ygwang/codes/_external/infinigen-indoors-b11700e'))
    p.add_argument('--python',default='/data1/ygwang/codes/_external/infinigen-cpu-b11700e/bin/python')
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--geometry',type=Path,help='Live collision export; binds Gen and native scene restoration')
    p = subs.add_parser('export-geometry',help='GPU: native collision meshes and restorable scene state for Gen')
    p.add_argument('--config',type=Path,default=REPO/'configs/layouts/behavior_home_v1.yaml')
    p.add_argument('--assets',type=Path)
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=[4,5],default=5)
    p.add_argument('--gpu-isolated',action='store_true',help=argparse.SUPPRESS)
    p = subs.add_parser('batch-home', help='Resumable CPU generation of distinct native-home backgrounds')
    p.add_argument('--config',type=Path,default=REPO/'configs/layouts/behavior_batch_v1.yaml')
    p.add_argument('--assets',type=Path)
    p.add_argument('--upstream',type=Path,default=Path('/data1/ygwang/codes/_external/infinigen-indoors-b11700e'))
    p.add_argument('--python',default='/data1/ygwang/codes/_external/infinigen-cpu-b11700e/bin/python')
    p.add_argument('--count',type=int,default=200)
    p.add_argument('--max-proposals',type=int,default=300)
    p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--out',type=Path,required=True)
    p = subs.add_parser('gpu-smoke')
    p.add_argument('--environment-only',action='store_true',default=True,help='Validate reset, camera and physical state without running a policy')
    p.add_argument('--scenario', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=[4, 5], default=5)
    p.add_argument('--realize-background',action='store_true',
                   help='Retry failed native supports before target placement; record actual poses and stability')
    p.add_argument('--diagnose-background',action='store_true',
                   help='With --realize-background, record 360 ordinary steps after rejection; never promotes a failure')
    p.add_argument('--diagnose-motion-controls',action='store_true',
                   help='Private stationary/slide/tipped-drop controls; always rejects the smoke episode')
    p.add_argument('--background-stability-profile',choices=['legacy','observed-motion-v3'],default='legacy',
                   help='Versioned motion observation; v3 records every physics step for three seconds')
    p.add_argument('--acceptance-policy',choices=['strict-v3','execution-v1'],default='strict-v3')
    p.add_argument('--verify-background-restore',action='store_true',help='Require disk checkpoint restoration before target insertion')
    p.add_argument('--background-velocity-iterations',type=int,choices=[1,4],
                   help='Diagnostic-only solver-iteration experiment on added rigid props; not a dataset profile')
    p.add_argument('--gpu-isolated', action='store_true', help=argparse.SUPPRESS)
    p = subs.add_parser('batch-probe', help='GPU 4/5 environment-only validation with per-case receipts')
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--gpus',type=int,nargs='+',choices=[4,5],default=[4,5])
    p.add_argument('--background-mode',choices=['native-realization','direct-proposal'],
                   default='native-realization',help='Native physical realization by default; direct proposals are diagnostic only')
    p.add_argument('--background-stability-profile',choices=['legacy','observed-motion-v3'],default='observed-motion-v3')
    p.add_argument('--acceptance-policy',choices=['strict-v3','execution-v1'],default='strict-v3')
    p.add_argument('--verify-background-restore',action='store_true',default=True)
    p.add_argument('--sim-python',default='/data1/ygwang/miniconda3/envs/behavior/bin/python')
    args = ap.parse_args()
    if args.stage=='gpu-smoke' and args.environment_only and (
            not args.realize_background or not args.verify_background_restore
            or args.background_stability_profile!='observed-motion-v3'):
        ap.error('--environment-only requires realized background, observed-motion-v3 and restore validation')
    if args.stage=='gpu-smoke' and args.acceptance_policy=='execution-v1' and args.background_stability_profile!='observed-motion-v3':
        ap.error('execution-v1 requires observed-motion-v3 measurements')
    if getattr(args,'diagnose_background',False) and not args.realize_background:
        ap.error('--diagnose-background requires --realize-background')
    if getattr(args,'diagnose_motion_controls',False) and not args.realize_background:
        ap.error('--diagnose-motion-controls requires --realize-background')
    if args.stage=='gpu-smoke' and (args.background_stability_profile!='legacy' or args.verify_background_restore or args.acceptance_policy!='strict-v3') and not args.realize_background:
        ap.error('--background-stability-profile requires --realize-background')
    if getattr(args,'background_velocity_iterations',None) is not None and not args.realize_background:
        ap.error('--background-velocity-iterations requires --realize-background')
    os.chdir(REPO)
    if args.stage == 'prepare':
        return prepare(args)
    if args.stage == 'prepare-home':
        return prepare_living_home(args)
    if args.stage == 'export-geometry':
        return export_collision_geometry(args)
    if args.stage == 'batch-home':
        from rummagebench.authoring.layout.indoor_batch import generate_batch
        return generate_batch(args,REPO,args.assets or host_config()['behavior_assets_root'])
    if args.stage == 'batch-probe':
        from rummagebench.authoring.layout.indoor_batch_probe import probe_batch
        return probe_batch(args,REPO)
    from rummagebench.sim.gpu_isolation import (resolve_gpu, isolated_gpu_command,
        verify_isolated_cuda, configure_single_gpu_kit)
    if not args.gpu_isolated:
        if (args.out/'result.json').exists():
            raise ValueError('OUTPUT_RESULT_EXISTS: use a new GPU output directory')
        from collections import Counter
        from rummagebench.authoring.layout.indoor_batch_probe import classify_process
        task_doc=yaml.safe_load(args.scenario.read_text())
        manifest=load_manifest(Path(task_doc['scene']['assembly_manifest']))
        expected=dict(scenario_sha256=sha256_path(args.scenario),
                      assembly_sha256=manifest['manifest_sha256'],
                      runner_sha256=sha256_path(__file__),gpu=args.gpu)
        expected['background_velocity_iterations']=args.background_velocity_iterations
        expected['background_stability_profile']=args.background_stability_profile
        expected['verify_background_restore']=args.verify_background_restore
        expected['acceptance_policy']=args.acceptance_policy
        if args.environment_only:expected['environment_only']=True
        command, env = isolated_gpu_command([sys.executable, str(Path(__file__).resolve()),
                                            *sys.argv[1:], '--gpu-isolated'], resolve_gpu(args.gpu))
        code=None;error=None
        try:
            code=subprocess.run(command, env=env, timeout=1200, check=False).returncode
        except subprocess.TimeoutExpired:
            error='GPU_PROCESS_TIMEOUT'
        # SDK cleanup can return zero after an unsuccessful job. Evidence is
        # mandatory, and this receipt is limited to execution, never data release.
        receipt = args.out/'result.json'
        result=classify_process(json.loads(receipt.read_text()) if receipt.exists() else {},code,error,expected)
        atomic_write_json(receipt,result)
        stages={name:sha256_path(args.out/f'{name}.json')
                for name in ['background_realization','background_after_target','background_restore','environment_ready','background_dynamics','background_physics_experiment']
                if (args.out/f'{name}.json').exists()}
        atomic_write_json(args.out/'process.json',dict(id=args.out.parent.name,
            scene=task_doc['scene']['model'],**expected,status=result['status'],
            returncode=code,result_sha256=sha256_path(receipt),stage_sha256=stages,
            error=result.get('error') or result.get('process_error'),evaluation_ready_episodes=0,
            failures=dict(Counter(a['failure'] for a in result.get('attempts',[]) if 'failure' in a))))
        return 0 if code==0 and result['status']=='environment_ready' else 1
    import torch
    verify_isolated_cuda(torch)
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    import isaacsim
    configure_single_gpu_kit(isaacsim)
    return smoke(args)


if __name__ == '__main__':
    raise SystemExit(main())
