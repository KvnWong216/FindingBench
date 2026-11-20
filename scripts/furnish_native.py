"""Native all-furniture enrichment; finite proposals, gravity and read-only gates."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import furniture_planner as planner
LAST_CLEARANCE_ERROR = None


def settle_physics(backend,steps):
    """Keep native physics/state updates; defer unseen renders until capture."""
    if not getattr(backend,'_furniture_physics_only',True):
        return backend.settle(steps)
    sim=backend._sim;previous=sim._render_on_step
    try:
        with sim.render_on_step(False):
            return backend.settle(steps)
    finally:
        # The pinned SDK context does not restore its flag on exceptions.
        sim._render_on_step=previous


def arr(v):
    import numpy as np
    return np.asarray(v.detach().cpu() if hasattr(v,'detach') else v,dtype=float)


def probe_record(hit):
    """PhysX ray results contain tensors; keep a small, serializable audit."""
    return dict(hit=bool(hit.get('hit')),body=str(hit.get('rigidBody','')),
        position=arr(hit.get('position',[])).tolist(),normal=arr(hit.get('normal',[])).tolist())


def quat_angle(a,b):
    import numpy as np
    a,b=arr(a),arr(b)
    return float(2*np.arccos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)),0,1)))


def records(backend):
    from omnigibson.object_states import Open
    return {o.name:dict(category=o.category,model=getattr(o,'model',None),
        n_joints=o.n_joints,rooms=list(getattr(o,'in_rooms',None) or []),
        fixed_base=bool(o.fixed_base),robot=o in backend._env.scene.robots,
        open_state=bool(o.states[Open].get_value()) if Open in o.states else None,
        aabb=backend.entity_aabb(o.name)) for o in backend._env.scene.objects}


def floor_at(backend,rec,bounds,inside):
    """Measure the real horizontal collider inside a native fillable volume."""
    from omnigibson.utils.sampling_utils import raytest
    lo,hi=bounds; hits=[]
    for fx,fy in itertools.product((.25,.5,.75),repeat=2):
        xy=[lo[0]+fx*(hi[0]-lo[0]),lo[1]+fy*(hi[1]-lo[1])]
        start=(lo[2]+hi[2])/2 if inside else lo[2]+.05
        end=lo[2]-.12
        all_hits=raytest([*xy,start],[*xy,end],only_closest=False)
        own=[h for h in all_hits if backend._owning_object(str(h.get('rigidBody',''))) is rec
             and h.get('normal',[0,0,0])[2]>.98 and lo[2]-.12<=h['position'][2]<start]
        if own:hits.append(max(float(h['position'][2]) for h in own))
    # Large cabinets may contain several shelves or a divider. Find a real
    # candidate plane here; all nine rays over the SMALL object's footprint
    # still have to verify that same plane in placement_clear().
    clusters=[]
    for height in sorted(hits):
        group=next((g for g in clusters if abs(height-g[0])<=.003),None)
        if group is None:clusters.append([height])
        else:group.append(height)
    clusters.sort(key=lambda g:(-len(g),g[0]))
    return max(clusters[0]) if clusters and len(clusters[0])>=3 else None


def locations(backend,task,source):
    from omnigibson.object_states.contains import m
    scene=backend._env.scene; rows=records(backend); result=[]
    for obj in scene.objects:
        if not planner.eligible(obj.category):continue
        room=next((planner.room_kind(r) for r in rows[obj.name]['rooms'] if planner.room_kind(r)),None)
        if room is None:
            # Native authored storage has no in_rooms: infer from nearest
            # room-labelled furniture, never from an arbitrary default room.
            center=arr(rows[obj.name]['aabb']).mean(0)
            nearby=sorted((float(((arr(r['aabb']).mean(0)[:2]-center[:2])**2).sum()),name)
                for name,r in rows.items() if planner.eligible(r['category']) and r['rooms'])
            if nearby and nearby[0][0]<9:
                room=next((planner.room_kind(r) for r in rows[nearby[0][1]]['rooms'] if planner.room_kind(r)),None)
        volumes=[]
        for key,link in obj.links.items():
            if link.is_meta_link and link.meta_link_type in m.CONTAINER_META_LINK_TYPES:
                box=[arr(v).tolist() for v in link.visual_aabb]
                if planner.valid_box(box):volumes.append((key,box))
        if obj.category in planner.TABLES:
            box=rows[obj.name]['aabb']; top=box[1][2]
            if .35<top<1.65:
                result.append(dict(id=obj.name+'::top',support=obj.name,link=None,mode='surface',
                    bounds=[[box[0][0],box[0][1],top],[box[1][0],box[1][1],top+.32]],room_kind=room))
        if obj.category in planner.CABINETS or 'cabinet' in obj.category:
            if not volumes:
                result.append(dict(id=obj.name+'::inside',support=obj.name,link=None,mode='inside',
                    bounds=None,room_kind=room,status='no_native_fillable_volume'))
            for key,box in volumes:
                result.append(dict(id=obj.name+'::'+key,support=obj.name,link=key,mode='inside',bounds=box,room_kind=room))
    for loc in result:
        if loc.get('status'):continue
        rec=scene.object_registry('name',loc['support'])
        floor=floor_at(backend,rec,loc['bounds'],loc['mode']=='inside')
        if floor is None:
            loc.update(status='no_verified_horizontal_support');continue
        loc['floor_z']=floor
        loc['existing']=planner.occupied(loc,rows)
        if loc['existing']:loc['status']='already_populated'
        elif not loc['room_kind']:loc['status']='unknown_room_semantics'
        else:loc['status']='empty'
    return result,rows


def placement_clear(backend,obj,rec,location,xy,size):
    """Actual collider rays across the candidate, before adding its solid body."""
    from omnigibson.utils.sampling_utils import raytest
    global LAST_CLEARANCE_ERROR
    LAST_CLEARANCE_ERROR = None
    floor=location['floor_z']; sx,sy,sz=size; ignore=[l.prim_path for l in obj.links.values()]
    heights=[]
    for dx,dy in itertools.product((-.35,0,.35),repeat=2):
        x,y=xy[0]+sx*dx,xy[1]+sy*dy
        hit=raytest([x,y,floor+sz+.008],[x,y,floor-.012],ignore_bodies=ignore)
        if (not hit.get('hit') or backend._owning_object(str(hit.get('rigidBody',''))) is not rec
                or hit.get('normal',[0,0,0])[2]<.98 or abs(hit['position'][2]-floor)>.03):
            LAST_CLEARANCE_ERROR=dict(reason='footprint_not_supported',hit=probe_record(hit),expected_floor=floor)
            return False
        heights.append(hit['position'][2])
    if max(heights)-min(heights)>.003:
        LAST_CLEARANCE_ERROR=dict(reason='uneven_real_support',heights=[float(h) for h in heights])
        return False
    floor=float(max(heights))
    if floor+sz+.006>location['bounds'][1][2]:return False
    location['floor_z']=floor
    # Cross rays detect internal walls, handles and closed panels in the empty
    # volume, including the cabinet's own collider (not its outer AABB).
    for z in (floor+min(.012,sz*.25),floor+sz*.5,floor+sz-.002):
        for axis in (0,1):
            for fraction in (-.35,0,.35):
                start=[xy[0],xy[1],z];end=start.copy();other=1-axis
                start[other]+=size[other]*fraction;end[other]=start[other]
                start[axis]-=size[axis]*.5+.003;end[axis]+=size[axis]*.5+.003
                hit=raytest(start,end,ignore_bodies=ignore)
                if hit.get('hit'):
                    LAST_CLEARANCE_ERROR=dict(reason='solid_collider_inside_candidate',hit=probe_record(hit),ray=[start,end])
                    return False
    return True


def inside_bounds(backend,obj,rec,loc):
    import numpy as np
    import torch
    box=arr(backend.entity_aabb(obj.name));lo,hi=arr(loc['bounds'])
    if np.any(box[0][:2]<lo[:2]+.008) or np.any(box[1][:2]>hi[:2]-.008):return False
    if loc['mode']=='inside':
        if box[1][2]>hi[2]-.006:return False
        # Native Inside is a centre-in-convex-volume predicate. A virtual
        # fillable hull is not collision geometry; its inset floor must not
        # reject an object resting on the measured real collision floor.
        # Body clearance is checked by footprint/cross rays, bounded extents,
        # physical contacts and the uninterrupted gravity observation.
        volume=rec.links[loc['link']]
        pts=torch.tensor([box.mean(0).tolist()],dtype=torch.float32,device=volume.get_position_orientation()[0].device)
        if not bool(volume.check_points_in_volume(pts).all().item()):return False
    return abs(box[0][2]-loc['floor_z'])<=.012


def certify(backend,obj,rec,loc,orientation,start_position=None,previous_pose=None,dt=None):
    import numpy as np
    from omnigibson.object_states import OnTop,Inside
    from omnigibson.utils.usd_utils import RigidContactAPI
    from rummagebench.skills.move import navigation_aabbs
    p,q=obj.get_position_orientation();box=backend.entity_aabb(obj.name)
    tilt=quat_angle(q,orientation);raw_speed=float(np.linalg.norm(arr(obj.get_linear_velocity())))
    raw_spin=float(np.linalg.norm(arr(obj.get_angular_velocity())))
    speed,spin=raw_speed,raw_spin
    if previous_pose is not None:
        speed=float(np.linalg.norm(arr(p)-previous_pose[0]))/dt
        spin=quat_angle(q,previous_pose[1])/dt
    relation=bool(obj.states[Inside if loc['mode']=='inside' else OnTop].get_value(rec))
    pairs=RigidContactAPI.get_contact_pairs(scene_idx=obj.scene.idx,query_set=[obj],with_set=None,current_only=True)
    foreign=[];support_contact=False
    for pair in pairs:
        owners={backend._owning_object(str(path)) for path in pair}
        if rec in owners:support_contact=True
        if any(owner not in (obj,rec,None) for owner in owners):foreign.append(list(pair))
    drift=float(np.linalg.norm(arr(p)-start_position)) if start_position is not None else 0.
    navigation_started=time.perf_counter()
    above_or_below=planner.outside_navigation_band(box,float(backend.robot_pose()[0][2]))
    navigation_preserved=(above_or_below or planner.preserves_navigation(
        navigation_aabbs(backend,obj.name),navigation_aabbs(backend,rec.name)))
    navigation_stats=getattr(backend,'_furniture_navigation_stats',dict(outside_band=0,full_native=0,seconds=0.))
    navigation_stats['outside_band' if above_or_below else 'full_native']+=1
    navigation_stats['seconds']+=time.perf_counter()-navigation_started
    backend._furniture_navigation_stats=navigation_stats
    passed=(relation and support_contact and not foreign and navigation_preserved and tilt<.08 and speed<.025 and spin<.15
            and drift<.006 and inside_bounds(backend,obj,rec,loc))
    return dict(passed=bool(passed),relation=relation,support_contact=support_contact,
        foreign_contacts=foreign,navigation_preserved=navigation_preserved,rotation_from_authored_rad=tilt,linear_speed_m_s=speed,
        angular_speed_rad_s=spin,reported_linear_speed_m_s=raw_speed,reported_angular_speed_rad_s=raw_spin,
        velocity_source='successive_native_world_poses' if previous_pose is not None else 'native_solver_report',
        drift_m=drift,aabb=box)


def observe(backend,locs,steps):
    """Use realized physical motion; retain solver velocity as a diagnostic."""
    objects={loc['entity']:backend._env.scene.object_registry('name',loc['entity']) for loc in locs if loc.get('entity')}
    previous={name:tuple(arr(v) for v in obj.get_position_orientation()) for name,obj in objects.items()}
    starts={name:pose[0] for name,pose in previous.items()};samples=[];failed=set()
    timings=dict(steps=steps,props=len(objects),physics_seconds=0.,certification_seconds=0.)
    for tick in range(steps):
        started=time.perf_counter();settle_physics(backend,1)
        timings['physics_seconds']+=time.perf_counter()-started
        started=time.perf_counter()
        for loc in locs:
            name=loc.get('entity')
            if not name:continue
            obj=objects[name];rec=backend._env.scene.object_registry('name',loc['support'])
            check=certify(backend,obj,rec,loc,loc['orientation'],starts[name],previous[name],backend._sim.get_sim_step_dt())
            samples.append(dict(tick=tick,entity=name,**check))
            if not check['passed']:failed.add(name)
            previous[name]=tuple(arr(v) for v in obj.get_position_orientation())
        timings['certification_seconds']+=time.perf_counter()-started
    measurements=getattr(backend,'_furniture_observation_timings',[])
    measurements.append(timings);backend._furniture_observation_timings=measurements
    return samples,failed


def retain_originals(backend,baseline):
    complete=deepcopy(backend._sim.dump_state(serialized=False))
    old=baseline[0]['registry']['object_registry'];actual=complete[0]['registry']['object_registry']
    if not set(old)<=set(actual):raise ValueError('ORIGINAL_OBJECT_REMOVED')
    for name in old:actual[name]=deepcopy(old[name])
    backend._sim.load_state(complete,serialized=False)


def remove_added(backend,locations,names,reason):
    """Discard only our failing additions, preserving every other native state."""
    sim=backend._sim;scene=backend._env.scene
    before=deepcopy(sim.dump_state(serialized=False));sim.stop()
    for loc in locations:
        if loc.get('entity') in names:
            scene.remove_object(scene.object_registry('name',loc['entity']))
            loc.update(status=reason,rejected_entity=loc.pop('entity'))
    sim.play();sim.step()
    complete=deepcopy(sim.dump_state(serialized=False))
    old=before[0]['registry']['object_registry'];actual=complete[0]['registry']['object_registry']
    for name in actual:actual[name]=deepcopy(old[name])
    sim.load_state(complete,serialized=False)


def run(args,task,job,root):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from enrich_background import world_poses,compare_poses
    from rummagebench.core.scenario import load_scenario,ScenarioSpec
    from rummagebench.adapters.python_api import prepare_renderer_grounding,install_renderer_grounding
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.authoring.tasks.household_state import export_native_member,load_frozen_member,restore_frozen_member
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    policy_path=Path(job['auto_policy']);policy=json.loads(policy_path.read_text())
    report=dict(status='failed',policy_version=planner.VERSION,parent_task=task,process_id=os.getpid(),
        full_skill_completion='unverified',started_at=time.time(),
        validation_mode='sparse_authoring' if job.get('quick_furniture') else 'paired_native',
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    backend=None
    try:
        if policy['version']!=planner.VERSION:raise ValueError('FURNITURE_POLICY_VERSION_MISMATCH')
        source=load_scenario(task['play_scenario']);prepare_renderer_grounding(source)
        frozen=load_frozen_member(Path(source.scene.frozen_member_manifest),source)
        spec=load_scenario(output/'play.yaml') if args.restore else source
        prepare_renderer_grounding(spec);backend=OmniGibsonBackend(seed=task.get('seed',0))
        backend._furniture_physics_only=bool(job.get('physics_only',True))
        report['physics_only_unobserved_steps']=backend._furniture_physics_only
        setup_started=time.perf_counter()
        backend.setup(spec);install_renderer_grounding(backend);scene=backend._env.scene;sim=backend._sim
        report['setup_seconds']=time.perf_counter()-setup_started
        observation_steps=math.ceil(policy['stable_seconds']/sim.get_sim_step_dt())
        report['observation_seconds']=observation_steps*sim.get_sim_step_dt()
        if args.restore:
            resolved=json.loads((output/'resolved_plan.json').read_text());locs=resolved['locations']
            expected=json.loads((output/'exported_world_poses.private.json').read_text())
            report['reload_world_preservation']=compare_poses(expected,world_poses(backend))
            props=[scene.object_registry('name',r['entity']) for r in locs if r.get('entity')]
            samples,failed=observe(backend,locs,observation_steps)
            (output/'reload_native_placement_samples.private.json').write_text(json.dumps(samples))
            checks=[sample for sample in samples if sample['tick']==observation_steps-1]
            if failed:raise ValueError('RELOAD_FURNITURE_UNSTABLE:'+','.join(sorted(failed)))
            report['reload_placement_checks']=checks
            restore_frozen_member(backend,Path(spec.scene.frozen_member_manifest),spec)
            report['cache_restore_pose_match']=compare_poses(expected,world_poses(backend))
        else:
            baseline=deepcopy(sim.dump_state(serialized=False));original=world_poses(backend)
            (output/'original_world_poses.private.json').write_text(json.dumps(original))
            Image.fromarray(backend.capture_panorama_frame().rgb).save(output/'before.png')
            sim.step() # synchronize native scene-query proxies after snapshot restore
            locs,rows=locations(backend,task,source)
            (output/'location_inventory.private.json').write_text(json.dumps(locs,indent=2))
            (output/'scene_geometry.private.json').write_text(json.dumps(rows))
            doc=source.model_dump(mode='json',exclude_none=True)
            plans=[]
            structural=('walls','floors','ceilings')
            for loc in locs:
                if loc['status']!='empty':continue
                if loc['id'] in job.get('avoid_locations',[]):
                    loc['status']='prior_independent_reload_instability';continue
                obstacles=[r['aabb'] for name,r in rows.items() if name!=loc['support'] and r['category'] not in structural]
                # Preserve a clear central landing patch on task destinations.
                if any(getattr(g,'receptacle',None)==loc['support'] for g in source.goal):
                    lo,hi=loc['bounds'];cx=(lo[0]+hi[0])/2;cy=(lo[1]+hi[1])/2
                    obstacles.append([[cx-.15,cy-.15,loc['floor_z']],[cx+.15,cy+.15,loc['floor_z']+.35]])
                candidates=planner.candidates(loc,policy,dict(doc,id=task['id']),obstacles)
                if not candidates:loc['status']='no_fitting_semantic_asset_or_free_space';continue
                # Bound native work: one real model, six exact free-space fallbacks.
                chosen=candidates[0];loc['candidate_model']=chosen['category']+'/'+chosen['model']
                plans.append((loc,[p for p in candidates if p['model']==chosen['model']][:6]))
            sim.load_state(deepcopy(baseline),serialized=False)
            # Passive original joints can relax even without any new props.
            # Pair the same stop/play import lifecycle and observation horizon
            # instead of misattributing baseline motion to furnishing.
            sim.stop();sim.play();sim.step();sim.load_state(deepcopy(baseline),serialized=False)
            settle_physics(backend,observation_steps)
            control=world_poses(backend)
            (output/'control_world_poses.private.json').write_text(json.dumps(control))
            sim.load_state(deepcopy(baseline),serialized=False)
            from omnigibson.objects import DatasetObject
            sim.stop();allocated=[]
            for i,(loc,candidates) in enumerate(plans):
                p=candidates[0];obj=DatasetObject(name=f'furnishing_{i:03d}',category=p['category'],model=p['model'],fixed_base=False)
                scene.add_object(obj);obj.set_position_orientation(position=[100+i*2,100,5],orientation=[0,0,0,1])
                allocated.append(obj)
            sim.play();sim.step();retain_originals(backend,baseline)
            accepted=[];rejected=[]
            # Author all independent surfaces before one shared settling pass.
            # No per-object physics rollout or blind resampling loop.
            for obj,(loc,candidates) in zip(allocated,plans):
                rec=scene.object_registry('name',loc['support']);errors=[]
                q=[0.,0.,0.,1.];obj.set_position_orientation(orientation=q)
                box=arr(obj.aabb);position=arr(obj.get_position_orientation()[0]);size=(box[1]-box[0]).tolist()
                for p in candidates:
                    if not placement_clear(backend,obj,rec,loc,p['xy'],size):
                        errors.append(LAST_CLEARANCE_ERROR);continue
                    destination=position+[*p['xy'],loc['floor_z']+size[2]/2+.004]-box.mean(0)
                    obj.set_position_orientation(position=destination.tolist(),orientation=q);obj.sleep_threshold=0.
                    obj.set_linear_velocity(torch.zeros(3));obj.set_angular_velocity(torch.zeros(3))
                    loc.update(status='provisional',entity=obj.name,category=obj.category,model=obj.model,
                        orientation=q,position_xy=p['xy'])
                    accepted.append(obj);break
                else:
                    loc.update(status='native_candidates_rejected',rejections=errors);rejected.append(obj)
                print('FURNITURE_LOCATION '+json.dumps(dict(id=loc['id'],status=loc['status'],entity=loc.get('entity'))),flush=True)
            retain_originals(backend,baseline)
            settle_physics(backend,44)
            warm_previous={o.name:tuple(arr(v) for v in o.get_position_orientation()) for o in accepted}
            settle_physics(backend,1)
            for loc in locs:
                if loc.get('status')!='provisional':continue
                obj=scene.object_registry('name',loc['entity']);rec=scene.object_registry('name',loc['support'])
                check=certify(backend,obj,rec,loc,loc['orientation'],previous_pose=warm_previous[obj.name],dt=sim.get_sim_step_dt())
                loc['placement_check']=check
                if check['passed']:loc['status']='added'
                else:
                    loc.update(status='native_gravity_rejected',rejected_entity=loc.pop('entity'))
                    rejected.append(obj);accepted.remove(obj)
            (output/'placement_progress.private.json').write_text(json.dumps(locs,indent=2))
            # Remove only the rejected new props, retain exact named originals.
            if rejected:
                before=deepcopy(sim.dump_state(serialized=False));sim.stop()
                for obj in rejected:scene.remove_object(obj)
                sim.play();sim.step()
                complete=deepcopy(sim.dump_state(serialized=False))
                old=before[0]['registry']['object_registry'];actual=complete[0]['registry']['object_registry']
                for name in actual:actual[name]=deepcopy(old[name])
                sim.load_state(complete,serialized=False)
            retain_originals(backend,baseline)
            # A real unmodified 3-second gravity observation, without freezes,
            # velocity resets or pose writes during the observation interval.
            samples=[]
            for repair_round in range(3):
                measured,failed=observe(backend,locs,observation_steps)
                samples.extend(dict(repair_round=repair_round,**sample) for sample in measured)
                (output/'native_placement_samples.private.json').write_text(json.dumps(samples))
                if not failed:break
                if repair_round==2:raise ValueError('FURNITURE_UNSTABLE_AFTER_LOCAL_REPAIR:'+','.join(sorted(failed)))
                remove_added(backend,locs,failed,'removed_after_measured_instability')
                retain_originals(backend,baseline)
            if job.get('quick_furniture'):
                # Sparse authoring: certify the added props and task relations.
                # Unrelated original hinges are not counterfactual experiments.
                current=world_poses(backend)
                report['whole_world_counterfactual']='not_run_sparse_authoring'
                report['independent_reload']='not_run_sparse_authoring'
                (output/'observed_world_poses.private.json').write_text(json.dumps(current))
            else:
                # Compare against an unchanged-parent control with the SAME native
                # actor topology. Parking only our props avoids changing constraint
                # indexing merely by adding bodies, which moves unrelated doors.
                enriched=deepcopy(backend.capture_observe_state())
                enriched_world=world_poses(backend)
                for index,loc in enumerate(locs):
                    if loc.get('entity'):
                        obj=scene.object_registry('name',loc['entity'])
                        obj.set_position_orientation(position=[100.+index*2,100.,5.],orientation=loc['orientation'])
                        obj.set_linear_velocity(torch.zeros(3));obj.set_angular_velocity(torch.zeros(3))
                retain_originals(backend,baseline)
                settle_physics(backend,observation_steps)
                control=world_poses(backend)
                (output/'same_topology_control_world_poses.private.json').write_text(json.dumps(control))
                control=planner.original_control_poses(control,original)
                backend.restore_observe_state(enriched)
                current=world_poses(backend)
                report['control_roundtrip_world_match']=compare_poses(enriched_world,current)
                (output/'observed_world_poses.private.json').write_text(json.dumps(current))
                protected={o.name for o in source.objects}|{p.receptacle for p in source.placements}
                protected.update(g.receptacle for g in source.goal if getattr(g,'receptacle',None))
                protected.update(source.initial_states);protected.update(backend.robot_entity_names())
                decorated_supports={loc['support'] for loc in locs if loc.get('entity')}
                passive_doors={name for name,row in rows.items()
                    if row['n_joints'] and name not in protected and name not in decorated_supports}
                from omnigibson.object_states import Open
                allowed_doors={name for name in passive_doors
                    if planner.passive_articulation_within_baseline(original[name],control[name],current[name])
                    and (rows[name]['open_state'] is None
                         or bool(scene.object_registry('name',name).states[Open].get_value())==rows[name]['open_state'])}
                report['passive_unreferenced_articulations_within_unchanged_baseline_range']=sorted(allowed_doors)
                strict_control={name:value for name,value in control.items() if name not in allowed_doors}
                drifting=[]
                for name in strict_control:
                    try:compare_poses({name:strict_control[name]},current)
                    except ValueError as e:drifting.append(dict(entity=name,error=str(e)))
                report['original_world_drift_details']=drifting
                # If decoration nudged an existing door, remove our own additions
                # in that cabinet and observe again. Never edit a parent joint to
                # manufacture a pass.
                removed=[loc for loc in locs if loc.get('entity') and loc['support'] in {d['entity'] for d in drifting}]
                if removed:
                    before=deepcopy(sim.dump_state(serialized=False));sim.stop()
                    for loc in removed:
                        scene.remove_object(scene.object_registry('name',loc['entity']))
                        loc.update(status='removed_to_preserve_parent_articulation',rejected_entity=loc.pop('entity'))
                    sim.play();sim.step()
                    complete=deepcopy(sim.dump_state(serialized=False))
                    old=before[0]['registry']['object_registry'];actual=complete[0]['registry']['object_registry']
                    for name in actual:actual[name]=deepcopy(old[name])
                    sim.load_state(complete,serialized=False);retain_originals(backend,baseline)
                    repair_samples=[]
                    for repair_round in range(3):
                        measured,failed=observe(backend,locs,observation_steps)
                        repair_samples.extend(dict(repair_round=repair_round,**sample) for sample in measured)
                        (output/'post_parent_repair_samples.private.json').write_text(json.dumps(repair_samples))
                        if not failed:break
                        if repair_round==2:raise ValueError('POST_REPAIR_FURNITURE_UNSTABLE:'+','.join(sorted(failed)))
                        remove_added(backend,locs,failed,'removed_after_parent_repair_instability')
                        retain_originals(backend,baseline)
                    current=world_poses(backend)
                    (output/'observed_world_poses.private.json').write_text(json.dumps(current))
                    report['removed_for_original_world_preservation']=[loc['id'] for loc in removed]
                report['paired_original_world_preservation']=compare_poses(strict_control,current)
            parent=source.model_dump(mode='json',exclude_none=True);doc=deepcopy(parent)
            doc['id']=source.id+'_furnished_v1';doc['scene'].pop('frozen_member_manifest')
            for loc in locs:
                if not loc.get('entity'):continue
                doc['objects'].append(dict(name=loc['entity'],category=loc['category'],model=loc['model'],fixed_base=False))
                doc['placements'].append(planner.frozen_placement(loc))
            spec=ScenarioSpec.model_validate(doc);backend._scenario=spec
            checkdoc=spec.model_dump(mode='json',exclude_none=True);checkdoc['id']=parent['id']
            checkdoc['scene']['frozen_member_manifest']=parent['scene']['frozen_member_manifest']
            checkdoc['objects']=checkdoc['objects'][:len(parent['objects'])];checkdoc['placements']=checkdoc['placements'][:len(parent['placements'])]
            if checkdoc!=parent:raise ValueError('TASK_CONTENT_CHANGED')
            report['task_content_preserved']=True
            validate_task(backend,spec)
            resolved=dict(policy_version=planner.VERSION,locations=locs)
            (output/'resolved_plan.json').write_text(json.dumps(resolved,indent=2))
            report['placement_observation']=dict(velocity_source='successive_native_world_poses',seconds=report['observation_seconds'],surviving_props=sum(bool(l.get('entity')) for l in locs))
            (output/'exported_world_poses.private.json').write_text(json.dumps(current))
            deps=dict(frozen['dependencies'])
            for path in (Path(__file__),Path(planner.__file__),Path(__file__).with_name('enrich_background.py'),
                         Path(__file__).with_name('background_planner.py'),args.plan,policy_path,output/'resolved_plan.json'):
                archived=output/'authoring_sources'/path.name;archived.parent.mkdir(exist_ok=True)
                archived.write_bytes(path.read_bytes());deps[str(archived)]=hashlib.sha256(archived.read_bytes()).hexdigest()
            export_native_member(backend,spec,output,dict(source=planner.VERSION,parent_frozen_sha256=frozen['manifest_sha256']),deps)
        backend.validate_physics_state();backend._validate_spawn_clearance()
        validate_task(backend,spec)
        report['coverage']=dict(total_locations=len(locs),added=sum(l['status']=='added' for l in locs),
            already_populated=sum(l['status']=='already_populated' for l in locs),
            unresolved=[dict(id=l['id'],reason=l['status']) for l in locs if l['status'] not in ('added','already_populated')])
        report['coverage']['complete']=not report['coverage']['unresolved']
        report['navigation_verified']=True
        Image.fromarray(backend.capture_panorama_frame().rgb).save(output/('restored.png' if args.restore else 'after.png'))
        if args.restore or job.get('quick_furniture'):
            variant=dict(task,id=spec.id,play_scenario=str(output/'play.yaml'),scenario=str(output/'play.yaml'),
                stage='ready_for_human_review',parent_variant=task['id'],furniture_coverage=report['coverage'],
                enrichment_validation=report['validation_mode'])
            from spectator_cache import export,lookup
            from spectator import physical_fingerprint
            parent_cache=lookup(root/'build/spectator_cache',task)
            if parent_cache is None:raise ValueError('PARENT_CAMERA_CACHE_UNAVAILABLE')
            fingerprint=physical_fingerprint(backend);tick=sim.current_time_step_index
            cache,_=export(root/'build/spectator_cache',variant,backend,SimpleNamespace(home=parent_cache[1]['home']),
                dict(physics_tick=tick,initial_physical_sha256=fingerprint))
            if physical_fingerprint(backend)!=fingerprint or sim.current_time_step_index!=tick:raise ValueError('CACHE_EXPORT_CHANGED_NATIVE_WORLD')
            report['cache']=str(cache)
            (output/'human_ready.json').write_text(json.dumps(dict(tasks=[variant]),indent=2))
        report['status']='restore_validated' if args.restore else 'initialization_validated'
    except Exception as e:
        import traceback
        traceback.print_exc();report.update(error=str(e),exception_type=type(e).__name__)
    finally:
        if backend is not None:
            report['observation_timings']=getattr(backend,'_furniture_observation_timings',[])
            report['navigation_timings']=getattr(backend,'_furniture_navigation_stats',{})
        report['finished_at']=time.time()
        (output/('restore_validation.json' if args.restore else 'validation.json')).write_text(json.dumps(report,indent=2))
    print('FURNITURE_RESULT '+json.dumps(dict(status=report['status'],error=report.get('error'),coverage=report.get('coverage'))),flush=True)
    os._exit(0 if report['status']!='failed' else 1)


def validate_task(backend,spec):
    """Retain actual parent relations, articulation and initial goal status."""
    from types import SimpleNamespace
    from omnigibson.object_states import Inside,OnTop,Open
    from rummagebench.evaluation.goal_checker import check_goal
    for placement in spec.placements:
        obj=backend._env.scene.object_registry('name',placement.entity)
        rec=backend._env.scene.object_registry('name',placement.receptacle)
        if (not backend._verify_relation(obj,rec,placement.relation,Inside,OnTop)
                or not backend._in_link(obj,rec,placement.link)):
            raise ValueError('RELATION_CHANGED:'+placement.entity)
    for name,state in spec.initial_states.items():
        if state.open is not None and bool(backend._env.scene.object_registry('name',name).states[Open].get_value())!=state.open:
            raise ValueError('INITIAL_OPEN_STATE_CHANGED:'+name)
    goal=check_goal(SimpleNamespace(held_object=None),spec,backend)
    if goal.satisfied or goal.fault:raise ValueError('INVALID_INITIAL_GOAL')
