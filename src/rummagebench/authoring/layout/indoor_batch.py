"""Resumable native-home candidates, with real InfiniGen solves per background.

All manifests and target slots are evaluator-private. CPU success never grants
physics, family, search-cost, or benchmark-release certification.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import yaml

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.layout.indoor_background import native_context, native_start
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.authoring.tasks.slots import build_scene_slots
from rummagebench.sim.omnigibson.behavior_adapter import asset_record, load_manifest


def background_identity(manifest: dict) -> str:
    """Ignore seeds, paths, instance numbering and sub-centimetre jitter."""
    rows = []
    for item in manifest['objects']:
        p = item['placement']
        rows.append([p['category'], p['model_id'], p['support_name'],
                     [round(v / .02) for v in p['position']],
                     round((p['orientation_deg'] % 360) / 15) % 24])
    native = manifest['living_context']['native']
    source_hash = native['inputs'][native['scene_json']]
    return sha256_bytes(canonical_json([source_hash, sorted(rows)]))


def archive_sources(plan, out, repo):
    index={}
    for source,digest in plan['inputs'].items():
        p=Path(source)
        if p.suffix=='.py' and p.is_relative_to(repo):
            data=p.read_bytes()
            if sha256_bytes(data)!=digest:raise ValueError('GENERATOR_SOURCE_CHANGED_DURING_PLAN')
            dst=out/'generation_sources'/p.relative_to(repo);dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(data)
            index[source]=dict(path=str(dst.relative_to(out)),sha256=digest)
    atomic_write_json(out/'generation_sources/index.json',index)


def _check_inputs(inputs, *, completed_root=None):
    archive={}
    if completed_root is not None:
        path=completed_root/'generation_sources/index.json'
        if path.exists():archive=json.loads(path.read_text())
    for path, digest in inputs.items():
        if sha256_path(path) != digest:
            # A completed batch can be audited using its exact historical
            # generator source. This fallback NEVER permits new generation,
            # changed assets, changed config, or partial-batch continuation.
            item=archive.get(path)
            if item and Path(path).suffix=='.py':
                saved=(completed_root/item['path']).resolve()
                if (saved.is_relative_to((completed_root/'generation_sources').resolve())
                        and item['sha256']==digest and sha256_path(saved)==digest):continue
            raise ValueError(f'BATCH_INPUT_CHANGED:{path}')


def validate_completed(plan, out, summary):
    records=json.loads((out/'records.json').read_text())
    expected=summarize(plan,records)
    if expected!=summary or summary['generated']!=plan['requested']:
        raise ValueError('BATCH_COMPLETION_COUNTS_MISMATCH')
    proposals={p['id']:p for p in plan['proposals']}
    selected=set(summary['selected_ids']);seen=set()
    for r in records:
        if r['id'] not in selected:continue
        if r['proposal_sha256']!=sha256_bytes(canonical_json(proposals[r['id']])):
            raise ValueError('BATCH_PROPOSAL_RECEIPT_CONFLICT')
        m=load_manifest(Path(r['assembly']));identity=background_identity(m)
        if (m['manifest_sha256']!=r['assembly_sha256'] or identity!=r['background_sha256']
                or sha256_path(r['scenario'])!=r['scenario_sha256'] or identity in seen):
            raise ValueError('BATCH_COMPLETED_OUTPUT_CHANGED')
        seen.add(identity)
    if len(seen)!=plan['requested']:raise ValueError('BATCH_COMPLETION_COVERAGE_MISMATCH')


def make_plan(config: Path, assets: Path, repo: Path, count: int, seed: int,
              max_proposals: int) -> dict:
    if not 1 <= count <= max_proposals <= 2000:
        raise ValueError('BOUNDED_BATCH_SIZE_REQUIRED')
    cfg = yaml.safe_load(config.read_text())
    sources = [config, Path(__file__), repo/'scripts/tasks/behavior_pipeline.py',
               repo/'src/rummagebench/authoring/layout/indoor_background.py',
               repo/'src/rummagebench/authoring/layout/indoor_geometry.py',
               repo/'src/rummagebench/sim/omnigibson/behavior_adapter.py',
               repo/'assets/tasks/slot_rules_v1.yaml']
    inputs = {str(p.resolve()): sha256_path(p) for p in sources}
    assets = assets.resolve()
    for item in [*cfg['targets'], *[a for pool in cfg['daily_pools'] for a in pool]]:
        inputs.update(asset_record(assets, **item)['sources'])
    rules = yaml.safe_load((repo/'assets/tasks/slot_rules_v1.yaml').read_text())
    homes, rejected = [], []
    for scene in cfg['scenes']:
        try:
            geometry=cfg.get('geometry_by_scene',{}).get(scene)
            if cfg.get('require_geometry') and geometry is None:
                raise ValueError('BATCH_NATIVE_GEOMETRY_REQUIRED:'+scene)
            if geometry:
                from rummagebench.authoring.layout.indoor_geometry import load_geometry
                approved={f"{a['category']}/{a['model']}":asset_record(assets,**a)
                          for pool in cfg['daily_pools'] for a in pool}
                measured=load_geometry(geometry,scene=scene,assets=approved)
                inputs[str(Path(geometry).resolve())]=sha256_path(geometry)
                inputs.update(measured['inputs'])
                context=native_context(assets,scene,scene_json=measured['scene_snapshot'])
            else:
                context = native_context(assets, scene)
            _, map_inputs = native_start(assets, scene, cfg['room'])
            slots = build_scene_slots(scene, assets, rules)
            cabinets = {o['name'] for o in context['objects'] if cfg['room'] in o['rooms']
                        and 'cabinet' in o['category'] and o['openable']}
            candidates = [s.to_dict() for s in slots.slots if s.parent_entity in cabinets and s.is_container]
            missing = cabinets - {s['parent_entity'] for s in candidates}
            # Legacy slot rules omit some native openable models. Preserve
            # those cabinet bodies as unresolved regions, never erase them
            # and renormalize the remaining locations. Live Inside placement
            # and cavity census still have to validate each body.
            candidates.extend(dict(slot_id=f'{name}/unresolved_native_interior',
                parent_entity=name,link=None,classification='native_openable_body_unresolved_region')
                for name in sorted(missing))
            if not candidates: raise ValueError('NO_KITCHEN_CONTAINER_CANDIDATES')
            native_categories = {o['category'] for o in context['objects']}
            targets = [a for a in cfg['targets'] if a['category'] not in native_categories]
            if not targets: raise ValueError('NO_UNAMBIGUOUS_TARGET_CATEGORY')
            inputs.update(context['inputs']); inputs.update(map_inputs)
            homes.append(dict(scene=scene, native_objects=len(context['retained_native_objects']),
                surfaces=sum(o['support'] for o in context['objects']), candidates=candidates,
                targets=targets, cabinet_bodies=sorted(cabinets),
                unresolved_native_bodies=sorted(missing),
                source_layout_group=scene.rsplit('_',2)[0],
                candidate_catalog_status='metadata_slots_pending_live_cavity_census'))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            if cfg.get('require_geometry'): raise
            rejected.append(dict(scene=scene, reason=f'{type(exc).__name__}:{exc}'))
    if not homes: raise ValueError('NO_NATIVE_HOMES_PASSED_CPU_PREFLIGHT')
    # Separate streams: background sampling never sees the selected target slot.
    bg_rng, task_rng = random.Random(seed), random.Random(seed ^ 0x5EED)
    slot_orders = {}
    for home in homes:
        order = sorted(home['candidates'], key=lambda s:s['slot_id'])
        task_rng.shuffle(order); slot_orders[home['scene']] = order
    proposals = []
    for i in range(max_proposals):
        home = homes[i % len(homes)]; rank = i // len(homes)
        slots = slot_orders[home['scene']]; slot = slots[rank % len(slots)]
        target = dict(home['targets'][task_rng.randrange(len(home['targets']))])
        pool = list(cfg['daily_pools'][bg_rng.randrange(len(cfg['daily_pools']))])
        if target['category'] in {a['category'] for a in pool}:
            raise ValueError('TARGET_CATEGORY_AMBIGUOUS_WITH_DAILY_POOL')
        sample = dict(scene=home['scene'], room=cfg['room'], daily_assets=pool,
            per_surface=bg_rng.choice(cfg['per_surface']), solver_steps=cfg['solver_steps'],
            target=target, smoke_container=slot['parent_entity'], start_yaw=0.)
        if cfg.get('geometry_by_scene',{}).get(home['scene']):
            sample['geometry']=str(Path(cfg['geometry_by_scene'][home['scene']]).resolve())
            sample['solver_timeout']=cfg.get('solver_timeout',900)
        if slot['link'] is not None: sample['smoke_link'] = slot['link']
        proposals.append(dict(id=f'home_{i:04d}', seed=bg_rng.randrange(2**31),
            config=sample, target_slot=slot['slot_id'], split='development',
            source_layout_group=home['source_layout_group']))
    return dict(version=1, requested=count, seed=seed, assets=str(assets),
        config=str(config.resolve()), inputs=inputs, homes=homes, rejected_homes=rejected,
        proposals=proposals, generated_structural_layouts=0,
        scope='native house templates with InfiniGen daily-object variation; pending simulation')


def summarize(plan, records):
    valid = [r for r in records if r['status']=='candidate_pending_simulation']
    selected = valid[:plan['requested']]
    return dict(requested=plan['requested'], attempted=len(records), generated=len(selected),
        reserve_candidates=max(0,len(valid)-len(selected)), evaluation_ready_episodes=0,
        unique_backgrounds=len({r['background_sha256'] for r in selected}),
        original_house_templates=len({r['scene'] for r in selected}), generated_structural_layouts=0,
        by_scene=dict(Counter(r['scene'] for r in selected)),
        by_target=dict(Counter(r['target_category'] for r in selected)),
        by_density=dict(Counter(str(r['per_surface']) for r in selected)),
        failures=dict(Counter(r.get('reason',r['status']) for r in records
                              if r['status']!='candidate_pending_simulation')),
        selected_ids=[r['id'] for r in selected], split='development',
        status='candidates_generated_pending_simulation' if len(selected)==plan['requested'] else 'incomplete')


def generate_batch(args, repo: Path, assets: Path):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'plan.json'
    # An exclusive non-blocking lock prevents concurrent coordinators from
    # choosing the same attempt paths; completed work is never overwritten.
    import fcntl
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
            summary_path=out/'summary.json'
            complete=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            complete=complete if complete.get('status')=='candidates_generated_pending_simulation' else None
            _check_inputs(plan['inputs'],completed_root=out if complete is not None else None)
            if (plan['requested']!=args.count or plan['seed']!=args.seed
                    or plan['config']!=str(args.config.resolve()) or plan['assets']!=str(assets.resolve())
                    or plan['runtime']!={'upstream':str(args.upstream.resolve()),'python':str(Path(args.python).resolve())}
                    or len(plan['proposals'])!=args.max_proposals):
                raise ValueError('BATCH_RESUME_REQUEST_CONFLICT')
            if complete is not None:
                validate_completed(plan,out,complete)
                print(json.dumps(dict(status='completed_batch_verified_no_generation',generated=complete['generated'])),flush=True)
                return 0
        else:
            if any(p.name!='.lock' for p in out.iterdir()): raise ValueError('BATCH_OUTPUT_NOT_EMPTY')
            plan=make_plan(args.config,assets,repo,args.count,args.seed,args.max_proposals)
            plan['runtime']={'upstream':str(args.upstream.resolve()),'python':str(Path(args.python).resolve())}
            atomic_write_json(plan_path,plan)
            archive_sources(plan,out,repo)
        cores=sorted(os.sched_getaffinity(0))
        workers=min(args.workers,max(1,len(cores)//2))
        if not 1<=workers<=8:raise ValueError('BATCH_WORKERS_MUST_BE_1_TO_8')
        def one(proposal, worker):
            folder=out/proposal['id'];folder.mkdir(exist_ok=True)
            receipt=folder/'receipt.json'
            if receipt.exists():
                record=json.loads(receipt.read_text())
                if record['proposal_sha256']!=sha256_bytes(canonical_json(proposal)):
                    raise ValueError('BATCH_PROPOSAL_RECEIPT_CONFLICT')
                if record['status']=='candidate_pending_simulation':
                    doc=load_manifest(Path(record['assembly']))
                    if (doc['manifest_sha256']!=record['assembly_sha256']
                            or background_identity(doc)!=record['background_sha256']
                            or sha256_path(record['scenario'])!=record['scenario_sha256']):
                        raise ValueError('BATCH_COMPLETED_OUTPUT_CHANGED')
                return record
            # Interrupted attempts are retained. Retry identical inputs in a
            # fresh directory; no target substitution or reduced constraints.
            attempt=0
            while (folder/f'attempt_{attempt:02d}').exists():attempt+=1
            if attempt>=3:raise ValueError('BATCH_INTERRUPTED_ATTEMPT_LIMIT')
            target=folder/f'attempt_{attempt:02d}'
            config=folder/f'config_{attempt:02d}.yaml'
            config.write_text(yaml.safe_dump(proposal['config'],sort_keys=False))
            cpu=cores[2*worker:2*worker+2] or cores[:1]
            argv=['taskset','--cpu-list',','.join(map(str,cpu)),sys.executable,
                str(repo/'scripts/tasks/behavior_pipeline.py'),'prepare-home',
                '--config',str(config),'--assets',str(assets),'--seed',str(proposal['seed']),
                '--out',str(target),'--upstream',str(args.upstream),'--python',args.python]
            if proposal['config'].get('geometry'):
                argv.extend(['--geometry',proposal['config']['geometry']])
            started=time.monotonic();returncode=None;reason=None
            with (folder/f'attempt_{attempt:02d}.log').open('w') as log:
                try:
                    returncode=subprocess.run(argv,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                        timeout=proposal['config'].get('solver_timeout',300)+120,check=False,
                        env=dict(os.environ,PYTHONUNBUFFERED='1')).returncode
                    if returncode!=0:raise ValueError(f'GENERATOR_EXIT_{returncode}')
                    doc=load_manifest(target/'assembly.json')
                except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    reason=f'{type(exc).__name__}:{exc}'
            record=dict(id=proposal['id'],scene=proposal['config']['scene'],
                proposal_sha256=sha256_bytes(canonical_json(proposal)),
                status='generation_failed' if reason else 'candidate_pending_simulation',
                reason=reason,returncode=returncode,elapsed_s=round(time.monotonic()-started,3),
                target_category=proposal['config']['target']['category'],target_slot=proposal['target_slot'],
                per_surface=proposal['config']['per_surface'],split='development',
                source_layout_group=proposal['source_layout_group'])
            if reason is not None:
                log_lines=(folder/f'attempt_{attempt:02d}.log').read_text(errors='replace').splitlines()
                record['failure_detail']='\n'.join(log_lines[-8:])
            if reason is None:
                record.update(assembly=str(target/'assembly.json'),scenario=str(target/'scenario.yaml'),
                    assembly_sha256=doc['manifest_sha256'],
                    scenario_sha256=sha256_path(target/'scenario.yaml'),background_sha256=background_identity(doc),
                    daily_objects=len(doc['objects']))
            atomic_write_json(receipt,record)
            return record
        completed={}; next_index=0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending={}
            for worker in range(min(workers,len(plan['proposals']))):
                pending[executor.submit(one,plan['proposals'][next_index],worker)]=worker
                next_index+=1
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                free=[]
                for future in done:
                    free.append(pending.pop(future))
                    record=future.result(); completed[record['id']]=record
                records=[];seen=set()
                for record in sorted(completed.values(),key=lambda r:r['id']):
                    if record['status']=='candidate_pending_simulation':
                        if record['background_sha256'] in seen:
                            record=dict(record,status='duplicate_background',reason='DUPLICATE_2CM_15DEG_BACKGROUND')
                        else:seen.add(record['background_sha256'])
                    records.append(record)
                summary=summarize(plan,records)
                atomic_write_json(out/'records.json',records);atomic_write_json(out/'summary.json',summary)
                print(json.dumps({k:v for k,v in summary.items() if k!='selected_ids'},ensure_ascii=False),flush=True)
                if summary['generated']<args.count:
                    for worker in free:
                        if next_index>=len(plan['proposals']):break
                        pending[executor.submit(one,plan['proposals'][next_index],worker)]=worker
                        next_index+=1
        return 0 if summary['generated']==args.count else 1
