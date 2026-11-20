"""Select one real native fillable region using the existing Inside sampler.

The SDK sampler otherwise chooses the first fillable link. Its collision,
support, settling and container-drift checks are retained unchanged. No scene
links are reordered, no proxy objects are introduced and no asset is modified.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import inspect
import textwrap
import copy
import math
import time


def isolate_authoring_robot(method):
    """Park only the robot while authoring props, then restore its full state.

    No collision or gravity flags change. Episode spawn, joint/controller
    state and all post-setup physical checks are retained, even on failure.
    """
    from functools import wraps
    @wraps(method)
    def run(backend,*args,**kwargs):
        import os,numpy as np
        if os.environ.get('YGBENCH_EXPERT_LAYOUT')!='1':return method(backend,*args,**kwargs)
        robot=backend._robot
        if any(obj is not None for obj in (getattr(robot,'_ag_obj_in_hand',{}) or {}).values()):
            raise ValueError('AUTHORING_ROBOT_MUST_START_EMPTY_HANDED')
        saved=copy.deepcopy(robot.dump_state(serialized=False))
        position,orientation=robot.get_position_orientation()
        position=np.asarray([float(v) for v in position]);orientation=np.asarray([float(v) for v in orientation])
        if not np.isfinite([*position,*orientation]).all():raise ValueError('INVALID_AUTHORING_ROBOT_STATE')
        try:
            robot.set_position_orientation(position=[1000.,1000.,float(position[2])+2.],orientation=orientation.tolist())
            result=method(backend,*args,**kwargs)
        finally:
            robot.load_state(copy.deepcopy(saved),serialized=False)
            restored_position,restored_orientation=robot.get_position_orientation()
            p=np.asarray([float(v) for v in restored_position]);q=np.asarray([float(v) for v in restored_orientation])
            if not np.isfinite([*p,*q]).all() or not np.allclose(p,position,atol=1e-4,rtol=0) or not (
                    abs(float(np.dot(q/np.linalg.norm(q),orientation/np.linalg.norm(orientation))))>1-1e-6):
                raise ValueError('AUTHORING_ROBOT_RESTORE_MISMATCH')
        if isinstance(result,dict):result['robot_authoring_isolation']=dict(
            method='temporary_translation_then_native_full_state_restore',spawn_changed=False,
            gravity_and_collision_flags_changed=False,post_setup_native_checks_required=True)
        return result
    return run


def reuse_prepared_usd(method, eligible, stats):
    """Reuse one preparation on the same object only, with unchanged inputs.

    Capture post-preparation state because native preparation updates scale,
    kinematic_only and the articulation path. Unsupported configurations always
    take the native path. No cache survives an object or a native process.
    """
    import enum
    from pathlib import Path
    from functools import wraps
    import numpy as np
    import torch

    def freeze(value):
        if value is None or type(value) in (bool, int, float, str):
            return (type(value).__name__, repr(value))
        if isinstance(value, enum.Enum):
            return (type(value).__module__, type(value).__qualname__, freeze(value.value))
        if isinstance(value, torch.Tensor):
            return ('tensor', str(value.dtype), str(value.device), tuple(value.shape),
                    value.detach().cpu().contiguous().numpy().tobytes())
        if isinstance(value, np.ndarray) and not value.dtype.hasobject:
            return ('array', str(value.dtype), value.shape, value.tobytes())
        if type(value) in (list, tuple):
            return (type(value).__name__, tuple(freeze(v) for v in value))
        if type(value) is dict and all(type(k) is str for k in value):
            return ('dict', tuple((k, freeze(v)) for k, v in sorted(value.items())))
        raise ValueError('Unsupported preparation input')

    def source(obj):
        return obj._usd_path.replace('.usd', '.encrypted.usd') if obj._encrypted else obj._usd_path

    def file_key(path):
        path = Path(path)
        stat = path.stat()
        return (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns,
                stat.st_ctime_ns, hashlib.sha256(path.read_bytes()).digest())

    def key(obj):
        return (freeze([obj._load_config, obj._relative_prim_path, obj.fixed_base,
                       obj.name, obj.category, obj.model, obj._usd_path,
                       obj._encrypted, obj._expected_file_hash]), file_key(source(obj)))

    @wraps(method)
    def prepared(obj):
        cached = obj.__dict__.pop('_findingbench_prepared_reuse', None)
        if not eligible(obj):
            stats['prepare_fallbacks'] += 1
            return method(obj)
        hit = False
        try:
            hit = cached and key(obj) == cached[0] and file_key(cached[1]) == cached[2]
        except (OSError, ValueError, TypeError, RuntimeError):
            stats['prepare_fallbacks'] += 1
        if hit:
            # Preserve native file-hash validation, including its exceptions.
            obj.check_hash(source(obj))
            stats['prepare_hits'] += 1
            return cached[1]
        result = method(obj)
        stats['prepare_misses'] += 1
        try:
            obj.__dict__['_findingbench_prepared_reuse'] = (key(obj), result, file_key(result))
        except (OSError, ValueError, TypeError, RuntimeError):
            stats['prepare_fallbacks'] += 1
        return result
    return prepared


def defer_authoring_rendering(method):
    """Opt-in native no-render stepping during placement only.

    Simulator.step still performs the same physics substeps and non-physics
    state updates. Initialization's explicit renders remain native. Restore
    rendering even if sampling raises; never apply this to agent execution or
    independent reload validation. This experiment requires native acceptance.
    """
    from functools import wraps
    import os
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if os.environ.get('YGBENCH_DEFER_AUTHORING_RENDER') != '1':
            return method(self, *args, **kwargs)
        sim = self._sim
        previous = sim._render_on_step
        try:
            # The SDK's context manager does not restore on exceptions.
            sim._render_on_step = False
            result = method(self, *args, **kwargs)
        finally:
            sim._render_on_step = previous
        if previous:
            # Native comments specify two renders for changed transforms to
            # propagate; these do not advance physics or replace any checks.
            sim.render()
            sim.render()
        return result
    return wrapped


def placement_order(placements, volumes):
    """Dependency layers, then largest rigid object first; indices retain identity."""
    names=[p.entity for p in placements]
    if len(set(names))!=len(names):
        raise ValueError('DUPLICATE_PLACEMENT_ENTITY')
    pending=set(range(len(placements)));done=set();order=[]
    while pending:
        ready=[i for i in pending if placements[i].receptacle not in names
               or placements[i].receptacle in done]
        if not ready:raise ValueError('CYCLIC_PLACEMENT_DEPENDENCY')
        ready.sort(key=lambda i:(-volumes[placements[i].entity],i))
        order.extend(ready);done.update(placements[i].entity for i in ready)
        pending.difference_update(ready)
    return order


def fitting_volume_names(obj, receptacle, margin=.005):
    """Authoring candidates for an unspecified compartment, conservative axis-permutation fit.

    Prefer larger fitting compartments to avoid exhausting shallow drawers.
    This is only a size prefilter. The native sampler must still pass every
    collision, support, drift and containment check. Explicit drawer choices
    do not use this selection and are never redirected to another drawer.
    """
    import numpy as np
    from omnigibson.object_states.contains import m as contains_macros
    def array(value):
        if hasattr(value, 'detach'): value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float)
    size=array(obj.native_bbox)*array(obj.scale)
    if size.shape!=(3,) or not np.isfinite(size).all() or (size<=0).any():
        raise ValueError('INVALID_NATIVE_OBJECT_DIMENSIONS')
    candidates=[];audit=[]
    for name,volume in receptacle.links.items():
        if not volume.is_meta_link or volume.meta_link_type not in contains_macros.CONTAINER_META_LINK_TYPES:
            continue
        lo,hi=volume.visual_aabb;extent=array(hi)-array(lo)
        valid=extent.shape==(3,) and np.isfinite(extent).all() and (extent>0).all()
        fits=bool(valid and np.all(np.sort(extent)>=np.sort(size)+margin))
        audit.append(dict(link=name,extent_m=extent.tolist(),size_fit=fits))
        if fits:candidates.append((-float(np.prod(extent)),name))
    return [name for _,name in sorted(candidates)],dict(object_extent_m=size.tolist(),volumes=audit,
        live_extent_m=array(obj.aabb_extent).tolist() if hasattr(obj,'aabb_extent') else None,
        joint_positions=array(obj.get_joint_positions()).tolist() if getattr(obj,'n_joints',0) else [])


def selected_volume_source(source: str) -> str:
    tree=ast.parse(textwrap.dedent(source))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('NATIVE_INSIDE_SAMPLER_SOURCE_UNSUPPORTED')
    function=tree.body[0]
    if function.name!='_set_value' or function.decorator_list:
        raise ValueError('NATIVE_INSIDE_SAMPLER_SIGNATURE_CHANGED')
    loops=[n for n in ast.walk(function) if isinstance(n,ast.For)
           and ast.unparse(n.target)=='link' and ast.unparse(n.iter)=='other.links.values()']
    if len(loops)!=1 or len(loops[0].body)!=1 or not isinstance(loops[0].body[0],ast.If):
        raise ValueError('NATIVE_INSIDE_VOLUME_SELECTION_CHANGED')
    selection=loops[0].body[0]
    if ('link.is_meta_link' not in ast.unparse(selection.test)
            or 'CONTAINER_META_LINK_TYPES' not in ast.unparse(selection.test)
            or ast.unparse(selection.body[-1])!='break'):
        raise ValueError('NATIVE_INSIDE_VOLUME_SELECTION_CHANGED')
    function.args.kwonlyargs.append(ast.arg(arg='_findingbench_volume_link'))
    function.args.kw_defaults.append(None)  # Required keyword, no silent default.
    selection.test=ast.BoolOp(op=ast.And(),values=[selection.test,
        ast.Compare(left=ast.Attribute(value=ast.Name(id='link',ctx=ast.Load()),attr='name',ctx=ast.Load()),
                    ops=[ast.Eq()],comparators=[ast.Name(id='_findingbench_volume_link',ctx=ast.Load())])])
    return ast.unparse(ast.fix_missing_locations(tree))


def diagnostic_volume_source(source: str) -> str:
    """Add counters only; retain all native proposals and rejection predicates."""
    tree=ast.parse(selected_volume_source(source));function=tree.body[0]
    function.args.kwonlyargs.append(ast.arg(arg='_findingbench_audit'))
    function.args.kw_defaults.append(None)
    def count(label):
        return ast.parse(f"_findingbench_audit[{label!r}] = _findingbench_audit.get({label!r}, 0) + 1").body[0]
    # Count rejected branches at the point that the unmodified sampler retries.
    for node in list(ast.walk(function)):
        if isinstance(node,ast.If) and any(isinstance(n,ast.Continue) for n in node.body):
            label=ast.unparse(node.test)
            node.body.insert(0,count(label))
        if isinstance(node,ast.For) and ast.unparse(node.target)=='attempt_idx':
            node.body.insert(0,count('attempts'))
            node.body.append(count('final_membership_or_reachability_failed'))
        if isinstance(node,ast.For) and any(isinstance(n,ast.Continue) for n in node.orelse):
            node.orelse.insert(0,count('no_support_contact_within_native_horizon'))
    return ast.unparse(ast.fix_missing_locations(tree))


def centered_volume_source(source: str) -> str:
    """Sample bbox centers, not asset origins; preserve native rejection checks."""
    tree=ast.parse(diagnostic_volume_source(source))
    setters=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and ast.unparse(n.func)=='self.obj.set_position_orientation']
    if len(setters)!=1:
        raise ValueError('NATIVE_INSIDE_POSE_PROPOSAL_CHANGED')
    setters[0].func.attr='set_bbox_center_position_orientation'
    return ast.unparse(ast.fix_missing_locations(tree))


def oriented_volume_source(source: str) -> str:
    """Inset for the proposed rotation, rather than the previous parked pose.

    Only proposal geometry changes. Native collision, drift, support and final
    membership rejection predicates remain mandatory.
    """
    tree=ast.parse(centered_volume_source(source))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
              and ast.unparse(n.target)=='attempt_idx')
    matches=[i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
             and any(ast.unparse(t)=='orientation' for t in n.targets)
             and 'quat_multiply' in ast.unparse(n.value)]
    if len(matches)!=1:raise ValueError('NATIVE_INSIDE_ORIENTATION_PROPOSAL_CHANGED')
    loop.body[matches[0]+1:matches[0]+1]=ast.parse('''
obj_extent = th.abs(T.quat2mat(orientation)) @ (self.obj.native_bbox * self.obj.scale)
inset_aabb_low = aabb_low + obj_extent / 2.0
inset_aabb_high = aabb_high - obj_extent / 2.0
''').body
    return ast.unparse(ast.fix_missing_locations(tree))


def settled_candidate_source(source):
    """Reject residual velocity after the native settling horizon, not at exit.

    This is an optional proposal-quality filter. Every original native check
    and the independent final convergence audit remain mandatory.
    """
    tree=ast.parse(source)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='attempt_idx')
    indices=[i for i,n in enumerate(loop.body) if isinstance(n,ast.While)
             and 'self.obj.get_linear_velocity()' in ast.unparse(n.test)
             and 'settle_step_idx < n_steps_max' in ast.unparse(n.test)]
    if len(indices)!=1:raise ValueError('NATIVE_INSIDE_SETTLING_HORIZON_CHANGED')
    loop.body[indices[0]+1:indices[0]+1]=ast.parse('''
_findingbench_linear = self.obj.get_linear_velocity()
_findingbench_angular = self.obj.get_angular_velocity()
if (not bool(th.isfinite(_findingbench_linear).all()) or
    not bool(th.isfinite(_findingbench_angular).all()) or
    th.norm(_findingbench_linear) > 0.01 or th.norm(_findingbench_angular) > 0.1):
    _findingbench_audit['candidate_velocity_above_final_threshold'] = _findingbench_audit.get('candidate_velocity_above_final_threshold', 0) + 1
    og.sim.load_state(state, serialized=False)
    continue
''').body
    return ast.unparse(ast.fix_missing_locations(tree))


@lru_cache(maxsize=8)
def _sampler(method, gpu=False, stable=False):
    source=inspect.getsource(method)
    adapted=gpu_volume_source(source) if gpu else oriented_volume_source(source)
    if stable:adapted=settled_candidate_source(adapted)
    namespace=dict(vars(inspect.getmodule(method)))
    namespace["_findingbench_gpu_proposals"]=native_gpu_proposals
    exec(compile(adapted,'findingbench_native_inside_selected_volume','exec'),namespace)
    return namespace['_set_value'],hashlib.sha256(source.encode()).hexdigest()


def selected_volume_contains(obj,receptacle,link_name):
    """Use native convex-volume membership, matching Inside's center semantics."""
    from omnigibson.object_states import Inside
    from omnigibson.object_states.contains import m as contains_macros
    volume=receptacle.links.get(link_name)
    if volume is None or not volume.is_meta_link or volume.meta_link_type not in contains_macros.CONTAINER_META_LINK_TYPES:
        raise ValueError('ACTUAL_NATIVE_FILLABLE_LINK_REQUIRED')
    if Inside not in obj.states or not bool(obj.states[Inside].get_value(receptacle)):
        return False
    center=obj.aabb_center
    import torch
    if not bool(torch.isfinite(center).all()):raise ValueError('NATIVE_VOLUME_QUERY_POINT_NONFINITE')
    return bool(volume.check_points_in_volume(center.unsqueeze(0)).item())


class SamplingAudit(dict):
    """Evaluator-only retry counters that survive a native timeout in logs.

    Log at attempt entry: rejection counters describe preceding attempts.
    No new native rejection predicates, retries or simulator operations are
    added. Optional diagnostic probes read poses and contact state only.
    """
    def __init__(self, obj_name, receptacle_name, volume_name, contact_probe=None):
        import time
        super().__init__()
        self.context=dict(object=obj_name,receptacle=receptacle_name,volume=volume_name)
        self.started=time.monotonic()
        self.contact_probe=contact_probe
        self.contact_events=[]

    def __setitem__(self, key, value):
        super().__setitem__(key,value)
        if self.contact_probe is not None and ('RigidContactAPI.is_in_contact' in key or 'container_joint_positions_delta' in key or key=='final_membership_or_reachability_failed') and (value==1 or value%20==0):
            import json
            try:
                event=dict(self.context,attempt=self.get('attempts'),rejection=key,**self.contact_probe())
            except Exception as exc:
                event=dict(self.context,attempt=self.get('attempts'),diagnostic_error=type(exc).__name__+': '+str(exc))
            self.contact_events.append(event)
            print('[inside-contact-audit] '+json.dumps(event),flush=True)
        if key=='attempts' and (value==1 or value%20==0):
            import json,time
            print('[inside-sampling-progress] '+json.dumps(dict(self.context,
                attempt_started=value,elapsed_s=round(time.monotonic()-self.started,3),
                counts=dict(self))),flush=True)


def sample_selected_volume(obj,receptacle,link_name):
    import omnigibson as og
    from omnigibson.object_states import Inside
    state=obj.states.get(Inside)
    volume=receptacle.links.get(link_name)
    if state is None or volume is None or not volume.is_meta_link:
        raise ValueError('ACTUAL_NATIVE_INSIDE_STATE_AND_FILLABLE_LINK_REQUIRED')
    import os
    gpu=os.environ.get("YGBENCH_GPU_PROPOSALS")=="1"
    sampler,digest=_sampler(type(state)._set_value,gpu=gpu,
        stable=os.environ.get('YGBENCH_EARLY_STABILITY_REJECTION')=='1')
    if not state._initialized:raise ValueError('NATIVE_INSIDE_STATE_NOT_INITIALIZED')
    before=og.sim.dump_state(serialized=False)
    try:
        # Same wrapper lifecycle as BaseObjectState.set_value().
        state.clear_cache()
        # Native links are keyed by short names, but RigidPrim.name is
        # qualified as "<object>:<link>" in the installed SDK.
        probe=None
        if os.environ.get('YGBENCH_AUTHORING_CONTACT_AUDIT')=='1':
            has_joints=bool(getattr(receptacle,'n_joints',0))
            initial_joints=[float(v) for v in receptacle.get_joint_positions()] if has_joints else []
            def probe():
                from omnigibson.utils.usd_utils import RigidContactAPI
                pairs=RigidContactAPI.get_contact_pairs(scene_idx=obj.scene.idx,
                    query_set=[obj],with_set=None,current_only=True)
                pos,quat=obj.get_position_orientation()
                return dict(contact_pairs=[list(p) for p in sorted(pairs)],
                    position=[float(v) for v in pos],orientation=[float(v) for v in quat],
                    aabb=[[float(v) for v in bound] for bound in obj.aabb],
                    initial_container_joints=initial_joints,
                    current_container_joints=[float(v) for v in receptacle.get_joint_positions()] if has_joints else [],
                    container_joint_velocities=[float(v) for v in receptacle.get_joint_velocities()] if has_joints else [])
        audit=SamplingAudit(getattr(obj,'name','unknown'),getattr(receptacle,'name','unknown'),link_name,contact_probe=probe)
        success=bool(sampler(state,receptacle,True,reset_before_sampling=True,use_trav_map=False,
            _findingbench_volume_link=volume.name,_findingbench_audit=audit))
        import json
        print('[inside-sampling] '+json.dumps(dict(object=getattr(obj,"name","unknown"),receptacle=getattr(receptacle,"name","unknown"),
            volume=link_name,success=success,counts=audit)),flush=True)
        obj.state_updated()
        if success and selected_volume_contains(obj,receptacle,link_name):
            return True,{'sampler_source_sha256':digest,'fillable_link':link_name,
                         'proposal':'native_bbox_center','counts':audit,'checks':'all_original_Inside_checks_and_selected_native_volume_membership'}
    except Exception:
        og.sim.load_state(before,serialized=False)
        raise
    og.sim.load_state(before,serialized=False)
    return False,{'sampler_source_sha256':digest,'fillable_link':link_name,'proposal':'native_bbox_center','counts':audit,'checks':'placement_rejected'}


def gpu_proposal_bank(size, low, high, rotations, probabilities, count, *, planes=(), obstacles=(), components=(), device='cuda', generator=None):
    """Batch proposal math on GPU; planes only ORDER proposals, never certify them.

    Preserve native orientation distribution (world yaw, asset rotation, local
    yaw), inset/full-bound halves and attempt budget. Random streams differ from
    the scalar sampler. Every returned proposal still needs native validation.
    """
    import torch as th
    def tensor(x):return th.as_tensor(x,dtype=th.float32,device=device)
    size,low,high=tensor(size),tensor(low),tensor(high)
    rotations=tensor(rotations);probabilities=tensor(probabilities)
    if count<1 or not bool(th.isfinite(probabilities).all()) or bool((probabilities<0).any()) or float(probabilities.sum())<=0:
        raise ValueError('INVALID_GPU_PROPOSAL_DISTRIBUTION')
    base=rotations[th.multinomial(probabilities,count,replacement=True,generator=generator)]
    def multiply(a,b):
        av,aw=a[:,:3],a[:,3:];bv,bw=b[:,:3],b[:,3:]
        return th.cat((aw*bv+bw*av+th.linalg.cross(av,bv),aw*bw-(av*bv).sum(-1,keepdim=True)),dim=-1)
    angles=th.rand((count,2),device=device,generator=generator)*2*th.pi
    def yaw(a):
        zero=th.zeros_like(a)
        return th.stack((zero,zero,th.sin(a/2),th.cos(a/2)),dim=-1)
    q=multiply(multiply(yaw(angles[:,0]),base),yaw(angles[:,1]))
    q=q/th.linalg.vector_norm(q,dim=-1,keepdim=True)
    x,y,z,w=q.unbind(-1)
    matrix=th.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
    extent=th.abs(matrix)@size
    inset_low=low+extent/2;inset_high=high-extent/2
    inset=(th.arange(count,device=device)<count//2)&(inset_low<inset_high).all(-1)
    lo=th.where(inset[:,None],inset_low,low);hi=th.where(inset[:,None],inset_high,high)
    if len(components):
        boxes=tensor(components)
        if boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or not bool(th.isfinite(boxes).all()) or bool((boxes[:,1]<=boxes[:,0]).any()):
            raise ValueError('INVALID_GPU_VOLUME_COMPONENTS')
        # A meta-link may contain disjoint native volume meshes separated by
        # solid partitions. Inset each component separately, never their union
        # AABB. The final native placement adds +1 cm before its checks.
        lower=th.maximum(boxes[None,:,0],low)+extent[:,None]/2
        upper=th.minimum(boxes[None,:,1],high)-extent[:,None]/2-tensor([0,0,.01])
        fits=(lower<upper).all(-1)
        scores=th.rand(fits.shape,device=device,generator=generator).masked_fill(~fits,-1)
        selected=scores.argmax(-1)
        use=fits.any(-1)&(th.arange(count,device=device)<count//2)
        rows=th.arange(count,device=device)
        lo=th.where(use[:,None],lower[rows,selected],lo)
        hi=th.where(use[:,None],upper[rows,selected],hi)
        # No fitting component is NOT a rejection: retain aggregate proposals.
        # Keep the full-bound second half and every original physical check.
    pos=lo+th.rand((count,3),device=device,generator=generator)*(hi-lo)
    # Union of native visual-mesh convex hulls. Invalid/unsupported geometry is
    # represented by no planes and simply disables ordering, never acceptance.
    valid=th.zeros(count,dtype=th.bool,device=device)
    for plane in planes:
        plane=tensor(plane)
        valid|=((pos@plane[:,:3].T+plane[:,3])<=1e-6).all(-1)
    score=th.zeros(count,device=device)
    if obstacles:
        boxes=tensor(obstacles)
        if boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or not bool(th.isfinite(boxes).all()) or bool((boxes[:,1]<boxes[:,0]).any()):
            raise ValueError('INVALID_GPU_OCCUPANCY_BOXES')
        # Match the native proposal's later +1 cm vertical offset. This is a
        # conservative AABB ordering heuristic, never a collision certificate.
        centers=pos+tensor([0,0,.01])
        lower=centers-extent/2;upper=centers+extent/2
        overlap=th.clamp(th.minimum(upper[:,None],boxes[None,:,1])-th.maximum(lower[:,None],boxes[None,:,0]),min=0)
        score=overlap.prod(-1).sum(-1)/th.clamp(extent.prod(-1),min=1e-12)
    if planes:score+=(~valid)*(len(obstacles)+1)
    if planes or obstacles:
        indices=th.arange(count,device=device)
        order=th.cat([part[th.argsort(score[part],stable=True)]
                      for part in (indices[:count//2],indices[count//2:])])
        pos,q=pos[order],q[order]
    return th.cat((pos,q),dim=-1)


def proposal_orientation_distribution(options, *, augment_empty=False):
    """Optional authoring coverage, never a physical acceptance criterion.

    Preserve asset-provided stable poses. For missing metadata only, the
    experiment retains upright candidates and adds each other local axis as
    the vertical axis. World/local yaw sampling and the attempt budget remain
    in gpu_proposal_bank. Native collision and settling decide actual validity.
    """
    if options:
        return [o['rotation'] for o in options.values()], [o['prob'] for o in options.values()]
    if not augment_empty:
        return [[0,0,0,1]], [1.]
    from math import sqrt
    s=sqrt(.5)
    return [[0,0,0,1], [s,0,0,s], [0,s,0,s]], [.5,.25,.25]


def refine_proposal_positions(bank, size, low, high, *, obstacles=(), planes=(),
                              iterations=24, generator=None):
    """Experimental batched local search on the inset half of proposal poses.

    Inspired by Infinigen's constrained local moves, not a port of its solver.
    Rotations, candidate count and the fallback half remain intact. Geometric
    scores only propose poses; native collision/support/drift checks certify.
    """
    import torch as th
    device=bank.device
    tensor=lambda x:th.as_tensor(x,dtype=bank.dtype,device=device)
    result=bank.clone();n=len(bank)//2
    if n==0 or iterations<=0 or (not obstacles and not planes):return result
    size,low,high=tensor(size),tensor(low),tensor(high)
    q=bank[:n,3:];x,y,z,w=q.unbind(-1)
    rotation=th.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
    extent=rotation.abs()@size
    lift=tensor([0,0,.01]);lower=low+extent/2;upper=high-extent/2-lift
    eligible=(lower<upper).all(-1)
    boxes=tensor(obstacles) if obstacles else None
    if boxes is not None and (boxes.ndim!=3 or boxes.shape[1:]!=(2,3)
            or not bool(th.isfinite(boxes).all()) or bool((boxes[:,1]<boxes[:,0]).any())):
        raise ValueError('INVALID_GPU_OCCUPANCY_BOXES')
    normalized=[]
    for p in planes:
        p=tensor(p);norm=th.linalg.vector_norm(p[:,:3],dim=-1)
        if not bool(th.isfinite(p).all()) or bool((norm<=0).any()):raise ValueError('INVALID_VOLUME_PLANES')
        normalized.append(p/norm[:,None])
    def score(pos):
        centers=pos+lift
        value=th.relu(lower-pos).sum(-1)+th.relu(pos-upper).sum(-1)
        if boxes is not None:
            overlap=th.relu(th.minimum(centers[:,None]+extent[:,None]/2,boxes[None,:,1])
                           -th.maximum(centers[:,None]-extent[:,None]/2,boxes[None,:,0]))
            value=value+overlap.prod(-1).sum(-1)/extent.prod(-1).clamp_min(1e-12)
        if normalized:
            violations=[]
            for p in normalized:
                # Oriented-box support against each actual convex component;
                # the minimum selects a component, never the components' hull.
                support=(th.einsum('fj,njk->nfk',p[:,:3],rotation).abs()*(size/2)).sum(-1)
                violations.append(th.relu(centers@p[:,:3].T+p[:,3]+support).amax(-1))
            value=value+10*th.stack(violations).amin(0)
        return value
    current=bank[:n,:3].clone();current_score=score(current)
    best=current.clone();best_score=current_score.clone()
    for step in range(iterations):
        temperature=.1*(.01/.1)**(step/max(1,iterations-1))
        noise=th.randn(current.shape,device=device,dtype=bank.dtype,generator=generator)
        candidate=th.maximum(lower,th.minimum(upper,current+noise*(high-low)*temperature))
        proposed_score=score(candidate)
        accept=eligible&((proposed_score<=current_score)|
            (th.log(th.rand((n,),device=device,generator=generator).clamp_min(1e-12))
             <(current_score-proposed_score)/temperature))
        current=th.where(accept[:,None],candidate,current);current_score=th.where(accept,proposed_score,current_score)
        better=eligible&(current_score<best_score)
        best=th.where(better[:,None],current,best);best_score=th.minimum(best_score,current_score)
    result[:n,:3]=th.where(eligible[:,None],best,bank[:n,:3])
    return result


def native_gpu_proposals(obj,volume,low,high,count):
    """One geometry upload and one proposal download per native sampler call."""
    import numpy as np
    import torch as th
    planned=getattr(obj,'_findingbench_layout_candidate',None)
    if planned is not None:
        pose=th.as_tensor(planned,dtype=th.float32)
        if pose.shape==(7,):pose=pose.reshape(1,7)
        if pose.ndim!=2 or pose.shape[1]!=7 or not 1<=len(pose)<=3 or not bool(th.isfinite(pose).all()):
            raise ValueError('INVALID_EXPERT_LAYOUT_CANDIDATE')
        return pose
    from scipy.spatial import ConvexHull,QhullError
    if not th.cuda.is_available():raise RuntimeError('GPU_PROPOSALS_REQUIRE_CUDA')
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    import os
    options=getattr(obj,'orientations',None) or {}
    rotations,probabilities=proposal_orientation_distribution(options,
        augment_empty=os.environ.get('YGBENCH_AXIS_ORIENTATION_PROPOSALS')=='1')
    planes=[];components=[]
    for mesh in volume.visual_meshes.values():
        try:
            points=array(mesh.points);transform=array(mesh.scaled_transform)
            hull=ConvexHull(points)
            planes.append(hull.equations@np.linalg.inv(transform))
            if os.environ.get('YGBENCH_GPU_VOLUME_COMPONENTS')=='1':
                world=points@transform[:3,:3].T+transform[:3,3]
                components.append([world.min(axis=0).tolist(),world.max(axis=0).tolist()])
        except (QhullError,ValueError,np.linalg.LinAlgError):
            # Unknown member of a union must disable the ordering approximation.
            planes=[];components=[];break
    bank=gpu_proposal_bank(array(obj.native_bbox*obj.scale),array(low),array(high),
                             rotations,probabilities,count,planes=planes,components=components,
                             obstacles=getattr(obj,'_findingbench_proposal_obstacles',()))
    if os.environ.get('YGBENCH_GPU_LOCAL_PROPOSALS')=='1':
        bank=refine_proposal_positions(bank,array(obj.native_bbox*obj.scale),array(low),array(high),
            planes=planes,obstacles=getattr(obj,'_findingbench_proposal_obstacles',()))
    return bank.cpu()


def gpu_volume_source(source):
    """Replace proposal construction only; retain every native rejection gate."""
    tree=ast.parse(oriented_volume_source(source));function=tree.body[0]
    loop=next(n for n in function.body if isinstance(n,ast.For) and ast.unparse(n.target)=='attempt_idx')
    budget=next(n for n in function.body if isinstance(n,ast.Assign)
                and any(ast.unparse(t)=='total_attempts' for t in n.targets))
    function.body[function.body.index(budget)+1:function.body.index(budget)+1]=ast.parse('''
if getattr(self.obj, '_findingbench_layout_candidate', None) is not None:
    total_attempts = 1
''').body
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(ast.unparse(t)=='orientation' for t in n.targets))
    end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and 'container_link.check_points_in_volume' in ast.unparse(n.test))
    loop.body[start:end]=ast.parse('pos = _gpu_bank[attempt_idx, :3].clone()\norientation = _gpu_bank[attempt_idx, 3:]').body
    index=function.body.index(loop)
    function.body[index:index]=ast.parse('''
_gpu_bank = _findingbench_gpu_proposals(self.obj, container_link, aabb_low, aabb_high, total_attempts)
total_attempts = len(_gpu_bank)
_findingbench_audit['gpu_candidates_generated'] = total_attempts
''').body
    return ast.unparse(ast.fix_missing_locations(tree))


def triangulate_faces(counts, indices):
    """Exact native fan triangulation without per-vertex Python/Torch loops."""
    import numpy as np
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=np.int64)
    counts,indices=array(counts),array(indices)
    if counts.ndim!=1 or indices.ndim!=1 or (counts<0).any() or counts.sum()!=len(indices):
        raise ValueError('INVALID_MESH_FACE_TOPOLOGY')
    if (counts==3).all():return indices.reshape(-1,3)
    triangles=np.maximum(counts-2,0)
    starts=np.repeat(np.cumsum(counts)-counts,triangles)
    offsets=np.arange(triangles.sum())-np.repeat(np.cumsum(triangles)-triangles,triangles)+1
    return np.stack((indices[starts],indices[starts+offsets],indices[starts+offsets+1]),axis=-1)


def vectorized_mesh_source(source):
    """Replace only the SDK fan loop; fail closed on an unfamiliar SDK."""
    tree=ast.parse(textwrap.dedent(source));function=tree.body[0]
    function.decorator_list=[]
    found=0
    for owner in ast.walk(function):
        for field,body in ast.iter_fields(owner):
            if not isinstance(body,list):continue
            for i,node in enumerate(list(body)):
                if not isinstance(node,ast.For) or ast.unparse(node.target)!='count' or ast.unparse(node.iter)!='face_vertex_counts':continue
                if i<2 or ast.unparse(body[i-2])!='faces = []' or ast.unparse(body[i-1])!='i = 0':
                    raise ValueError('SDK_TRIANGULATION_CHANGED')
                if 'face_indices[i + j + 2]' not in ast.unparse(node):raise ValueError('SDK_TRIANGULATION_CHANGED')
                body[i-2:i+1]=ast.parse('faces = _findingbench_triangulate(face_vertex_counts, face_indices)').body
                found+=1
    if found!=1:raise ValueError('SDK_TRIANGULATION_CHANGED')
    return ast.unparse(ast.fix_missing_locations(tree))


def vectorized_parallel_rays_source(source):
    """Batch offsets only; retain the native random frame and every ray."""
    tree=ast.parse(textwrap.dedent(source));function=tree.body[0]
    expected={
        'sources':'[source + offsets @ orthogonal_vectors for offsets in ray_grid_flattened]',
        'destinations':'[destination + offsets @ orthogonal_vectors for offsets in ray_grid_flattened]',
    }
    found=set();body=[]
    for node in function.body:
        name=ast.unparse(node.targets[0]) if isinstance(node,ast.Assign) and len(node.targets)==1 else None
        if name in expected:
            if ast.unparse(node.value)!=expected[name]:raise ValueError('NATIVE_PARALLEL_RAYS_CHANGED')
            if not found:body.extend(ast.parse('_findingbench_ray_offsets = ray_grid_flattened @ orthogonal_vectors').body)
            origin='source' if name=='sources' else 'destination'
            node.value=ast.parse(f'list({origin} + _findingbench_ray_offsets)',mode='eval').body
            found.add(name)
        body.append(node)
    if found!=set(expected):raise ValueError('NATIVE_PARALLEL_RAYS_CHANGED')
    function.body=body
    return ast.unparse(ast.fix_missing_locations(tree))


def fresh_relative_state_getter(original_getter):
    """Read recomputed tensorized relations instead of a second pose-only cache.

    The native getter still refreshes GPU/CPU state and checks initialization.
    Only its per-instance result cache is invalidated. Contact can change with
    unchanged poses, so a KinematicsMixin cache is insufficient here.
    """
    from functools import wraps
    @wraps(original_getter)
    def get_value(self, *args, **kwargs):
        self.clear_cache()
        return original_getter(self, *args, **kwargs)
    return get_value


def native_inside_reference_getter(original_getter):
    """Evaluate Inside using the SDK's actual convex-volume point query.

    Same predicate: object AABB centre in the container AABB and in a native
    fillable visual volume. No tolerance, pose correction or assumed support.
    This is a correctness reference for disagreement with tensorized Inside.
    """
    from functools import wraps
    @wraps(original_getter)
    def get_value(self, other):
        import torch as th
        from omnigibson.utils.constants import PrimType
        from omnigibson.object_states.contains import m as contains_macros
        if other.prim_type==PrimType.CLOTH:return original_getter(self,other)
        if self.obj is other:return False
        center=self.obj.aabb_center
        lower,upper=other.aabb
        if not bool(th.isfinite(center).all() and th.isfinite(lower).all() and th.isfinite(upper).all()):
            raise ValueError('NONFINITE_NATIVE_INSIDE_GEOMETRY')
        if not bool(((center>=lower)&(center<=upper)).all()):return False
        for link in other.links.values():
            if (link.is_meta_link and link.meta_link_type in contains_macros.CONTAINER_META_LINK_TYPES
                    and bool(link.check_points_in_volume(center.unsqueeze(0)).item())):
                return True
        return False
    return get_value


def install_mesh_acceleration():
    """Exact mesh conversion plus content-keyed local mass-property reuse.

    No cached world transforms, collision outcomes, or simulated object states.
    """
    import os,sys,sqlite3,json,threading
    import numpy as np
    import torch as th
    import trimesh
    from pathlib import Path
    from functools import cached_property
    from omnigibson.utils import usd_utils
    from omnigibson.prims.geom_prim import GeomPrim
    if getattr(usd_utils,'_findingbench_mesh_acceleration',False):return
    from omnigibson.object_states import Inside
    Inside._get_value=native_inside_reference_getter(Inside._get_value)
    Inside.get_value=fresh_relative_state_getter(Inside.get_value)
    print('[native-inside-reference] exact AABB-centre and SDK fillable-volume query enabled',flush=True)
    originals=[usd_utils.mesh_prim_mesh_to_trimesh_mesh,GeomPrim.points_and_faces.func]
    converted=[]
    for function in originals:
        namespace=dict(function.__globals__);namespace['_findingbench_triangulate']=triangulate_faces
        exec(compile(vectorized_mesh_source(inspect.getsource(function)),'findingbench_vector_mesh','exec'),namespace)
        converted.append(namespace[function.__name__])
    original_mass=usd_utils.get_mesh_volume_and_com
    signature=hashlib.sha256((inspect.getsource(original_mass)+inspect.getsource(originals[0])+str(trimesh.__version__)+str(np.__version__)).encode()).hexdigest()
    directory=Path(os.environ.get('YGBENCH_GEOMETRY_CACHE',str(Path.home()/'.cache/findingbench_geometry')))
    directory.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(directory/'local_mass.sqlite3',timeout=30,check_same_thread=False)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE IF NOT EXISTS mass (key TEXT PRIMARY KEY, payload TEXT NOT NULL)');connection.commit()
    guard=threading.Lock();stats={'hits':0,'misses':0,'fallbacks':0}
    if os.environ.get('YGBENCH_REUSE_PREPARED_USD') == '1':
        from omnigibson.objects import usd_object, dataset_object
        expected = {
            usd_object: '1203004ab636ce19a32e2d933d0559faf441d08874400d59627358486f794b40',
            dataset_object: 'cb0d371ad233d2eb5921e02b82e312bf8e12030533ef103e2e6129e68be4b349',
        }
        if any(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest
               for module, digest in expected.items()):
            raise RuntimeError('PREPARED_USD_REUSE_SDK_CHANGED')
        stats.update(prepare_hits=0, prepare_misses=0, prepare_fallbacks=0)
        usd_object.USDObject._prepare_to_load = reuse_prepared_usd(
            usd_object.USDObject._prepare_to_load,
            lambda obj: type(obj) is dataset_object.DatasetObject
                and obj.model not in dataset_object.MODEL_USD_HOTFIXES,
            stats,
        )
        print('[mesh-acceleration] guarded same-object prepared USD reuse enabled', flush=True)
    def mass(mesh_prim,world_frame=False):
        if world_frame or mesh_prim.GetTypeName()!='Mesh':return original_mass(mesh_prim,world_frame)
        try:
            key=hashlib.sha256(signature.encode())
            for name,dtype in [('points',np.float32),('faceVertexCounts',np.int64),('faceVertexIndices',np.int64)]:
                values=np.asarray(mesh_prim.GetAttribute(name).Get(),dtype=dtype)
                key.update(str(values.shape).encode());key.update(values.tobytes())
            key=key.hexdigest()
            with guard:row=connection.execute('SELECT payload FROM mass WHERE key=?',(key,)).fetchone()
            if row:
                payload=json.loads(row[0]);stats['hits']+=1
                return float(payload[0]),th.tensor(payload[1],dtype=th.float32)
        except (OSError,ValueError,sqlite3.Error):
            stats['fallbacks']+=1;return original_mass(mesh_prim,world_frame)
        result=original_mass(mesh_prim,world_frame);stats['misses']+=1
        volume,com=result
        if np.isfinite(float(volume)) and bool(th.isfinite(com).all()):
            try:
                with guard:
                    connection.execute('INSERT OR IGNORE INTO mass VALUES (?,?)',(key,json.dumps([float(volume),com.tolist()])));connection.commit()
            except sqlite3.Error:stats['fallbacks']+=1
        return result
    # Some native modules import these functions directly. Patch references to
    # these exact objects only, before any scene objects are loaded.
    for module in list(sys.modules.values()):
        if module is None or not getattr(module,'__name__','').startswith('omnigibson'):continue
        for name,value in list(vars(module).items()):
            if value is originals[0]:setattr(module,name,converted[0])
            elif value is original_mass:setattr(module,name,mass)
    descriptor=cached_property(converted[1]);descriptor.__set_name__(GeomPrim,'points_and_faces')
    GeomPrim.points_and_faces=descriptor
    if os.environ.get('YGBENCH_READONLY_MATERIAL_PROBE')=='1':
        from omnigibson.prims.xform_prim import XFormPrim
        original_probe=XFormPrim.has_material
        namespace=dict(original_probe.__globals__)
        exec(compile(readonly_material_source(inspect.getsource(original_probe)),
                     'findingbench_readonly_material','exec'),namespace)
        XFormPrim.has_material=namespace['has_material']
        print('[mesh-acceleration] read-only direct material lookup enabled',flush=True)
    if os.environ.get('YGBENCH_BATCH_POSE_WRITES')=='1':
        from omnigibson.prims.xform_prim import XFormPrim
        original_pose=XFormPrim.set_position_orientation
        namespace=dict(original_pose.__globals__)
        exec(compile(batched_pose_source(inspect.getsource(original_pose)),
                     'findingbench_batched_pose','exec'),namespace)
        XFormPrim.set_position_orientation=namespace['set_position_orientation']
        print('[mesh-acceleration] atomic translation/rotation USD write enabled',flush=True)
    if os.environ.get('YGBENCH_CANONICAL_XFORM')=='1':
        from omnigibson.prims.xform_prim import XFormPrim
        original=XFormPrim._set_xform_properties
        stats.update(canonical_skipped=0,canonical_fallback=0)
        def canonical_guard(prim):
            result=canonical_xform(prim)
            stats['canonical_skipped' if result else 'canonical_fallback']+=1
            return result
        namespace=dict(original.__globals__);namespace['_findingbench_canonical_xform']=canonical_guard
        exec(compile(canonical_xform_source(inspect.getsource(original)),
                     'findingbench_canonical_xform','exec'),namespace)
        XFormPrim._set_xform_properties=namespace['_set_xform_properties']
        print('[mesh-acceleration] canonical transform no-op guard enabled',flush=True)
    if os.environ.get('YGBENCH_PHYSICS_VELOCITY_WRITES')=='1':
        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
        from omnigibson.prims.entity_prim import EntityPrim
        stats.update(velocity_tensor_writes=0,velocity_fallback=0)
        def velocity_scope(sim,view):return physics_velocity_scope(sim,view,stats)
        for cls,name in [(RigidDynamicPrim,'set_linear_velocity'),(RigidDynamicPrim,'set_angular_velocity'),(EntityPrim,'set_joint_velocities')]:
            original=getattr(cls,name);namespace=dict(original.__globals__)
            namespace['_findingbench_velocity_scope']=velocity_scope
            exec(compile(physics_velocity_source(inspect.getsource(original)),
                         'findingbench_physics_velocity','exec'),namespace)
            setattr(cls,name,namespace[name])
        print('[mesh-acceleration] guarded physics-buffer velocity writes enabled',flush=True)
    if os.environ.get('YGBENCH_PHYSICS_POSITION_WRITES')=='1':
        from omnigibson.prims.entity_prim import EntityPrim
        stats.update(position_tensor_writes=0,position_fallback=0)
        def position_scope(sim,view):return physics_velocity_scope(sim,view,stats,prefix='position')
        original=EntityPrim.set_joint_positions;namespace=dict(original.__globals__)
        namespace['_findingbench_position_scope']=position_scope
        exec(compile(physics_joint_position_source(inspect.getsource(original)),
                     'findingbench_physics_joint_position','exec'),namespace)
        EntityPrim.set_joint_positions=namespace['set_joint_positions']
        print('[mesh-acceleration] guarded joint position writes; native PhysX/Fabric sync retained',flush=True)
    if os.environ.get('YGBENCH_BATCH_SURFACE_RAYS')=='1':
        from omnigibson.utils import sampling_utils
        original=sampling_utils.get_parallel_rays
        namespace=dict(original.__globals__)
        exec(compile(vectorized_parallel_rays_source(inspect.getsource(original)),
                     'findingbench_parallel_rays','exec'),namespace)
        sampling_utils.get_parallel_rays=namespace['get_parallel_rays']
        print('[mesh-acceleration] batched parallel ray offsets enabled',flush=True)
    usd_utils._findingbench_mesh_acceleration=True
    usd_utils._findingbench_mesh_cache_stats=stats
    print('[mesh-acceleration] exact vectorized triangulation and content-keyed local mass cache enabled',flush=True)


def readonly_material_source(source):
    """Avoid authoring a MaterialBindingAPI just to read a direct binding."""
    tree=ast.parse(textwrap.dedent(source));function=tree.body[0]
    if function.name!='has_material':raise ValueError('SDK_MATERIAL_PROBE_CHANGED')
    class Replace(ast.NodeTransformer):
        count=0
        def visit_Attribute(self,node):
            if ast.unparse(node)=='self._binding_api':
                self.count+=1
                return ast.copy_location(ast.parse('lazy.pxr.UsdShade.MaterialBindingAPI(self.prim)',mode='eval').body,node)
            return self.generic_visit(node)
    transform=Replace();tree=transform.visit(tree)
    if transform.count!=1:raise ValueError('SDK_MATERIAL_PROBE_CHANGED')
    return ast.unparse(ast.fix_missing_locations(tree))


def batched_pose_source(source):
    """Co-author translation and rotation in one native USD edit transaction.

    No synchronization is removed after the complete pose update. Transformation
    math, validation, Fabric update and tensor-cache invalidation are unchanged.
    """
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    if fn.name!='set_position_orientation':raise ValueError('SDK_POSE_SETTER_CHANGED')
    writes=[n for n in fn.body if isinstance(n,ast.Expr) and
            ast.unparse(n.value)=="self.set_attribute('xformOp:translate', position)"]
    edits=[n for n in fn.body if isinstance(n,ast.With) and
           len(n.items)==1 and ast.unparse(n.items[0].context_expr)=='og.sim.editing_usd()']
    if len(writes)!=1 or len(edits)!=1 or ast.unparse(edits[0].body[0])!='xform_op.Set(rotq)':
        raise ValueError('SDK_POSE_SETTER_CHANGED')
    start=fn.body.index(writes[0]);end=fn.body.index(edits[0])
    expected="""orientation = orientation[[3, 0, 1, 2]].tolist()
if 'xformOp:orient' not in properties:
    logger.error('Orient property needs to be set for {} before setting its orientation'.format(self.name))
xform_op = self._prim.GetAttribute('xformOp:orient')
if xform_op.GetTypeName() == 'quatf':
    rotq = lazy.pxr.Gf.Quatf(*orientation)
else:
    rotq = lazy.pxr.Gf.Quatd(*orientation)
"""
    if [ast.dump(n) for n in fn.body[start+1:end]]!=[ast.dump(n) for n in ast.parse(expected).body]:
        raise ValueError('SDK_POSE_SETTER_CHANGED')
    fn.body.remove(writes[0])
    edits[0].body.insert(0,ast.parse("self._prim.GetAttribute('xformOp:translate').Set(position)").body[0])
    return ast.unparse(ast.fix_missing_locations(tree))


def configure_isaac_thread_limit():
    """Apply Isaac's own startup limit, before it overwrites BLAS/TBB limits."""
    import os
    raw=os.environ.get('YGBENCH_ISAAC_THREADS')
    if raw is None:return
    if not raw.isdigit() or int(raw)<1:raise ValueError('Invalid YGBENCH_ISAAC_THREADS')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    from isaacsim import SimulationApp
    if 'limit_cpu_threads' not in SimulationApp.DEFAULT_LAUNCHER_CONFIG:
        raise RuntimeError('ISAAC_THREAD_CONFIGURATION_UNSUPPORTED')
    SimulationApp.DEFAULT_LAUNCHER_CONFIG['limit_cpu_threads']=int(raw)
    print(f'[native-resources] Isaac startup thread limit={raw}',flush=True)


def canonical_xform(prim):
    """Conservative no-op guard for already normalized, static USD transforms."""
    import numpy as np
    from pxr import UsdGeom,Usd
    names=['xformOp:translate','xformOp:orient','xformOp:scale']
    if prim.IsInstanceable() or prim.IsInstanceProxy():return False
    xform=UsdGeom.Xformable(prim)
    if not xform or xform.GetResetXformStack():return False
    order=prim.GetAttribute('xformOpOrder').Get()
    if order is None or list(order)!=names:return False
    if {n for n in prim.GetPropertyNames() if n.startswith('xformOp:')}!=set(names):return False
    attributes=[prim.GetAttribute(n) for n in names]
    if any(not a or a.GetNumTimeSamples() or a.Get() is None for a in attributes):return False
    if str(attributes[0].GetTypeName()) not in ('double3','float3') or str(attributes[2].GetTypeName()) not in ('double3','float3') or str(attributes[1].GetTypeName()) not in ('quatf','quatd'):return False
    pos,quat,scale=[a.Get() for a in attributes]
    if not hasattr(quat,'GetReal'):return False
    q=np.array([quat.GetReal(),*quat.GetImaginary()],dtype=float)
    if not all(np.isfinite(v).all() for v in (np.asarray(pos),q,np.asarray(scale))):return False
    if abs(float(q@q)-1)>1e-12 or np.any(np.asarray(scale)<=0):return False
    parent=prim.GetParent()
    if parent and not parent.IsPseudoRoot():
        parent_xform=UsdGeom.Xformable(parent)
        if not parent_xform:return False
        matrix=np.asarray(parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
        linear=matrix[:3,:3];gram=linear@linear.T
        # Avoid parent shear, reflection and nonuniform scaling: the native
        # world-to-local normalization may intentionally alter these cases.
        if not np.isfinite(matrix).all() or np.linalg.det(linear)<=0:return False
        if not np.allclose(matrix[:,3],[0,0,0,1],rtol=0,atol=1e-12):return False
        if not np.allclose(gram,np.eye(3)*gram[0,0],rtol=1e-12,atol=1e-12):return False
    return True


def canonical_xform_source(source):
    """Skip redundant transform edits, retaining native final pose assertions."""
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    if fn.name!='_set_xform_properties':raise ValueError('SDK_XFORM_INITIALIZER_CHANGED')
    edits=[(i,n) for i,n in enumerate(fn.body) if isinstance(n,ast.With)]
    if len(edits)!=1:raise ValueError('SDK_XFORM_INITIALIZER_CHANGED')
    i,edit=edits[0]
    if ast.unparse(edit.items[0].context_expr)!='og.sim.editing_usd()':raise ValueError('SDK_XFORM_INITIALIZER_CHANGED')
    expected='XFormPrim.set_position_orientation(self, position=current_position, orientation=current_orientation)'
    if ast.unparse(fn.body[i+1])!=expected or not any(isinstance(n,ast.Assert) for n in fn.body[i+2:]):
        raise ValueError('SDK_XFORM_INITIALIZER_CHANGED')
    fn.body[i:i+2]=[ast.If(test=ast.parse('not _findingbench_canonical_xform(self.prim)',mode='eval').body,
                          body=[edit,fn.body[i+1]],orelse=[])]
    return ast.unparse(ast.fix_missing_locations(tree))


def physics_velocity_scope(sim, view, stats=None, prefix='velocity'):
    """Avoid USD synchronization only for guarded live physics-buffer writes."""
    from contextlib import contextmanager
    @contextmanager
    def scope():
        fast=(sim.is_playing() and getattr(sim,'_usd_guard_enabled',False)
              and getattr(sim,'_in_sim_lifecycle',1)==0
              and getattr(view,'_physics_view',None) is not None
              and view.is_physics_handle_valid())
        if not fast:
            if stats is not None:stats[prefix+'_fallback']+=1
            with sim.editing_usd():yield
            return
        # Retain the native context's entry guards. Keep the USD listener active:
        # unexpected writes remain errors, rather than silently losing updates.
        assert not sim._editing_usd, 'Cannot nest editing_usd() contexts'
        sim._check_usd_guard()
        assert not sim.currently_in_isaac_step, 'Cannot edit USD while simulation is stepping!'
        if stats is not None:stats[prefix+'_tensor_writes']+=1
        try:yield
        finally:sim._check_usd_guard()
    return scope()


def physics_joint_position_source(source):
    """Retain native pose refresh and cache invalidation after tensor writes."""
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    if fn.name!='set_joint_positions':raise ValueError('SDK_JOINT_POSITION_SETTER_CHANGED')
    expected={'self._articulation_view.set_joint_positions',
              'self._articulation_view.set_joint_position_targets'}
    found=set()
    for node in ast.walk(fn):
        if not isinstance(node,ast.With) or len(node.items)!=1 or len(node.body)!=1:continue
        statement=node.body[0]
        if not isinstance(statement,ast.Expr) or not isinstance(statement.value,ast.Call):continue
        method=ast.unparse(statement.value.func)
        if method not in expected:continue
        if method in found or ast.unparse(node.items[0].context_expr)!='og.sim.editing_usd()':
            raise ValueError('SDK_JOINT_POSITION_SETTER_CHANGED')
        found.add(method)
        node.items[0].context_expr=ast.parse('_findingbench_position_scope(og.sim,self._articulation_view)',mode='eval').body
    if found!=expected or 'og.sim.sync_physx_to_fabric()' not in ast.unparse(fn) or 'TensorizedState.caches_dirty = True' not in ast.unparse(fn):
        raise ValueError('SDK_JOINT_POSITION_SETTER_CHANGED')
    return ast.unparse(ast.fix_missing_locations(tree))


def physics_velocity_source(source):
    """Change only the context around the existing native velocity setter."""
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    methods={'set_linear_velocity':('self._rigid_prim_view','set_linear_velocities'),
             'set_angular_velocity':('self._rigid_prim_view','set_angular_velocities'),
             'set_joint_velocities':('self._articulation_view','set_joint_velocities')}
    if fn.name not in methods:raise ValueError('SDK_VELOCITY_SETTER_CHANGED')
    view,method=methods[fn.name];changed=0
    for node in ast.walk(fn):
        if not isinstance(node,ast.With) or len(node.items)!=1 or len(node.body)!=1:continue
        call=node.body[0]
        if not isinstance(call,ast.Expr) or not isinstance(call.value,ast.Call):continue
        if ast.unparse(call.value.func)!=view+'.'+method:continue
        if ast.unparse(node.items[0].context_expr)!='og.sim.editing_usd()':raise ValueError('SDK_VELOCITY_SETTER_CHANGED')
        node.items[0].context_expr=ast.parse('_findingbench_velocity_scope(og.sim, '+view+')',mode='eval').body
        changed+=1
    if changed!=1:raise ValueError('SDK_VELOCITY_SETTER_CHANGED')
    return ast.unparse(ast.fix_missing_locations(tree))


@lru_cache(maxsize=1024)
def _solve_layout_gpu_cached(item_rows, volume_rows, fixed_rows, grid_m, margin_m,
                              time_limit_s, device, max_nodes, max_candidates, halfspace_rows=(),obstacle_rows=(),lateral_inset=False,rotation_rows=(),overlap_rows=()):
    """CUDA finite rectangle-packing CSP with vectorized candidate constraints.

    Coordinates use normal patterns (subset sums of the possible oriented
    widths/heights). Every compacted axis-aligned grid packing has such a
    representation. Exhausting the tree proves this box model infeasible;
    hitting a time/node/memory budget is UNKNOWN. No physics is certified here.
    """
    import torch as th
    started=time.perf_counter();nodes=0;tested=0;orders_tried=0
    names=[n for n,_ in item_rows];bins=[n for n,_ in volume_rows]
    shared=dict(overlap_rows);group_names=[shared.get(name,name) for name in bins]
    fixed=dict(fixed_rows);halfspaces=dict(halfspace_rows);obstacles=dict(obstacle_rows)
    def result(status,**extra):
        return dict(status=status,device=device,backend='torch_cuda_finite_csp',
            solve_seconds=time.perf_counter()-started,nodes=nodes,candidate_tests=tested,
            search_orders_tried=orders_tried,
            grid_m=grid_m,margin_m=margin_m,
            model='upright boxes; yaw 0/90; single-layer '+('shared physical cavity groups; ' if overlap_rows else 'independent bins; ')+(
                'integer grid with convex centre halfspaces / fixed obstacles' if halfspaces or obstacles else 'normal-pattern integer coordinates'),
            physical_validation_required=True,**extra)
    if not str(device).startswith('cuda') or not th.cuda.is_available():
        return result('unknown',reason='CUDA required; no automatic CPU solver fallback')
    group_ids=th.tensor([group_names.index(name) for name in group_names],device=device,dtype=th.long)
    if not names:return result('feasible_model',assignment={},poses={})
    if not bins:return result('infeasible_model',reason='No bins')
    dimensions=th.tensor([v for _,v in item_rows],dtype=th.float64,device=device)
    limits=th.tensor([v for _,v in volume_rows],dtype=th.float64,device=device)
    clearance=math.ceil(margin_m/grid_m-1e-9)
    sizes=th.ceil(dimensions[:,:2]/grid_m-1e-9).to(th.int64)
    # Inflate on the positive side to represent inter-item clearance. The
    # final item's unused positive clearance is added back to the bin bounds.
    capacity=th.floor(limits[:,:2]/grid_m+1e-9).to(th.int64)-clearance
    oriented=th.stack([sizes,sizes.flip(-1)],dim=1)+clearance
    allowed=th.tensor([[n not in fixed or b in fixed[n] for b in bins] for n in names],dtype=th.bool,device=device)
    fit=(oriented[:,None,:,:]<=capacity[None,:,None,:]).all(-1)
    fit &= (dimensions[:,None,None,2]+margin_m<=limits[None,:,None,2]+1e-9)
    fit &= allowed[:,:,None]
    rotations=dict(rotation_rows)
    if rotations:
        fit &= th.tensor([[n not in rotations or yaw in rotations[n] for yaw in (0,90)]
                          for n in names],dtype=th.bool,device=device)[:,None,:]
    square=sizes[:,0]==sizes[:,1];fit[:, :,1] &= ~(square[:,None]&fit[:,:,0])
    choice_counts=fit.sum((1,2))
    bad=th.nonzero(choice_counts==0).flatten()
    if bad.numel():
        return result('infeasible_model',reason='No allowed bin/orientation fits an item',blocking_entities=[names[i] for i in bad.cpu().tolist()])
    # Thin utensils often have a simple strip witness. Construct it before
    # allocating millions of subset-coordinate candidates. This only proves
    # the same box model feasible; all native validation remains mandatory.
    if not halfspaces and not obstacles and max_nodes>0:
        for bin_index in range(len(bins)):
            for long_x in (True,False):
                rotate=(sizes[:,0]<sizes[:,1]) if long_x else (sizes[:,0]>=sizes[:,1])
                rotations=rotate.to(th.int64);indices=th.arange(len(names),device=device)
                if not bool(fit[indices,bin_index,rotations].all()):continue
                dims=oriented[indices,rotations]
                for axis in (1,0):
                    if not bool((dims[:,axis].sum()<=capacity[bin_index,axis])&
                                (dims[:,1-axis].amax()<=capacity[bin_index,1-axis])):continue
                    starts=th.cumsum(dims[:,axis],0)-dims[:,axis]
                    lower=th.zeros_like(dims);lower[:,axis]=starts
                    footprint=(dims-clearance).to(th.float64)*grid_m
                    low=(lower+clearance).to(th.float64)*grid_m;centers=low+footprint/2
                    records=th.cat((low,footprint,centers,rotations[:,None]),-1).cpu().tolist()
                    poses={}
                    for name,record in zip(names,records):
                        x,y,w,h,cx,cy,rot=record
                        poses[name]=dict(bin=bins[bin_index],yaw_degrees=int(rot)*90,
                            lower_xy_m=[x,y],footprint_m=[w,h],center_xy_m=[cx,cy])
                    return result('feasible_model',assignment={n:bins[bin_index] for n in names},poses=poses,
                                  constructive_witness='single strip; exact quantized box and clearance checks')
    area=oriented[:,0].prod(-1)
    if bool((area.sum()>capacity.prod(-1).clamp_min(0).sum()).item()):
        return result('infeasible_model',reason='Total inflated footprint exceeds total bin area')
    possible_bins=fit.any(-1)
    if len(bins)<=10:
        subsets=th.arange(1,1<<len(bins),device=device)[:,None]
        membership=(subsets&(1<<th.arange(len(bins),device=device)[None,:]))!=0
        confined=(~possible_bins[None,:,:]|membership[:,None,:]).all(-1)
        required=(confined*area[None,:]).sum(-1)
        available=(membership*capacity.prod(-1)[None,:]).sum(-1)
        if bool((required>available).any().item()):
            return result('infeasible_model',reason='A bin subset lacks area for all items confined to it')
    for b in range(len(bins)):
        forced=(possible_bins.sum(-1)==1)&possible_bins[:,b]
        minimum=oriented.masked_fill(~fit[:,b,:,None],max(1,int(capacity.max().item()))*3).amin(1)
        for axis in range(2):
            strip=forced&(2*minimum[:,axis]>capacity[b,axis])
            if bool((minimum[strip,1-axis].sum()>capacity[b,1-axis]).item()):
                return result('infeasible_model',reason='Forced wide rectangles exceed the orthogonal strip capacity')
        compatible=((oriented[:,None,:,None,0]+oriented[None,:,None,:,0]<=capacity[b,0]) |
                    (oriented[:,None,:,None,1]+oriented[None,:,None,:,1]<=capacity[b,1]))
        compatible &= fit[:,b,None,:,None]&fit[None,:,b,None,:]
        pair_possible=compatible.any((-1,-2))
        conflict=forced[:,None]&forced[None,:]&~pair_possible&~th.eye(len(names),dtype=th.bool,device=device)
        if bool(conflict.any().item()):
            return result('infeasible_model',reason='Two items forced into one bin have no non-overlapping orientation pair')
    # Most constrained / largest first. This changes branch order, not domains.
    by_area=th.argsort(area,descending=True,stable=True)
    ordered=by_area[th.argsort(choice_counts[by_area],stable=True)].cpu().tolist()
    maxdim=int(capacity.max().item())
    if maxdim>20000:return result('unknown',reason='Coordinate domain exceeds configured memory guard')
    indices=th.arange(maxdim+1,device=device)
    domain=th.zeros(maxdim+1,dtype=th.bool,device=device);domain[0]=True
    for i in range(len(names)):
        old=domain;expanded=old.clone()
        for axis in range(2):
            shift=sizes[i,axis]+clearance
            source=indices-shift
            expanded |= (source>=0)&old[source.clamp_min(0)]
        domain=expanded
    coordinates=th.nonzero(domain).flatten()
    plane_tensors={bins.index(name):th.tensor(planes,dtype=th.float64,device=device)
                   for name,planes in halfspaces.items()}
    obstacle_tensors={bins.index(name):th.tensor(boxes,dtype=th.float64,device=device)
                      for name,boxes in obstacles.items()}
    # Clip each axis to its actual bin/orientation before forming products.
    # A long shallow drawer must not allocate a square of the longest axis.
    search_started=time.perf_counter()
    static_domains={};static_counts={};cached_rows=0
    # Geometry and settled obstacles do not change during this CSP call.
    # Evaluate them once per item/bin/orientation instead of rebuilding the
    # same million-point grids on every backtrack. The remaining filter below
    # only checks overlap against the current partial packing.
    for i in ordered:
        count=0;available=0
        for choice in th.nonzero(fit[i]):
            bin_index,rotation=choice;index=int(bin_index);rot_index=int(rotation)
            w,h=oriented[i,rotation]
            if index in plane_tensors or index in obstacle_tensors:
                xs=th.arange(max(0,int(capacity[index,0]-w)+1),device=device)
                ys=th.arange(max(0,int(capacity[index,1]-h)+1),device=device)
            else:
                xs=coordinates[coordinates+w<=capacity[bin_index,0]]
                ys=coordinates[coordinates+h<=capacity[bin_index,1]]
            if index in plane_tensors:
                planes=plane_tensors[index]
                inset=th.zeros(len(planes),dtype=th.float64,device=device)
                if lateral_inset:
                    radius=th.stack((w-clearance,h-clearance)).to(th.float64)*grid_m/2
                    inset=(radius@planes[:,:2].abs().T)*(th.linalg.vector_norm(planes[:,:2],dim=1)>.5)
                # Sound interval projection before forming the product. Even
                # slightly tilted planes can reject X values for which no Y
                # in the current interval satisfies that plane (and vice versa).
                # The full coupled constraints are still checked below.
                for axis in (0,1):
                    coordinates_axis=xs if axis==0 else ys
                    other=ys if axis==0 else xs
                    if not len(other) or not len(coordinates_axis):continue
                    width=w if axis==0 else h
                    other_width=h if axis==0 else w
                    center=(coordinates_axis+clearance+(width-clearance)/2)*grid_m
                    other_lo=(other[0]+clearance+(other_width-clearance)/2)*grid_m
                    other_hi=(other[-1]+clearance+(other_width-clearance)/2)*grid_m
                    best_other=th.minimum(planes[:,1-axis]*other_lo,planes[:,1-axis]*other_hi)
                    values=center[:,None]*planes[None,:,axis]+best_other[None]+dimensions[i,2]/2*planes[None,:,2]+planes[None,:,3]+inset[None]
                    coordinates_axis=coordinates_axis[(values < -.00001+1e-10).all(-1)]
                    if axis==0:xs=coordinates_axis
                    else:ys=coordinates_axis
            size=len(xs)*len(ys);count+=size
            if count>max_candidates:return result('unknown',reason='Candidate tensor memory budget exhausted')
            yy,xx=th.meshgrid(ys,xs,indexing='ij');x,y=xx.flatten(),yy.flatten()
            b=bin_index.expand(size);rot=rotation.expand(size)
            widths,heights=w.expand(size),h.expand(size)
            valid=th.ones(size,dtype=th.bool,device=device)
            if index in plane_tensors:
                planes=plane_tensors[index]
                centers=th.stack(((x+clearance+(w-clearance)/2)*grid_m,
                    (y+clearance+(h-clearance)/2)*grid_m,
                    (dimensions[i,2]/2).expand(size)),-1)
                values=centers@planes[:,:3].T+planes[:,3]
                if lateral_inset:
                    radius=th.stack((w-clearance,h-clearance)).to(th.float64)*grid_m/2
                    values+=(radius@planes[:,:2].abs().T)*(th.linalg.vector_norm(planes[:,:2],dim=1)>.5)
                valid &= (values < -.00001).all(-1)
            if index in obstacle_tensors:
                boxes=obstacle_tensors[index]
                active=(boxes[:,5]>0)&(boxes[:,2]<dimensions[i,2]+margin_m)
                separate=(x[:,None]+w)*grid_m<=boxes[None,:,0]-margin_m
                separate |= (x[:,None]+clearance)*grid_m>=boxes[None,:,3]+margin_m
                separate |= (y[:,None]+h)*grid_m<=boxes[None,:,1]-margin_m
                separate |= (y[:,None]+clearance)*grid_m>=boxes[None,:,4]+margin_m
                valid &= (separate|~active[None,:]).all(-1)
            candidates=th.stack([b,x,y,widths,heights,rot],-1)[valid]
            static_domains[i,index,rot_index]=candidates
            available+=len(candidates);cached_rows+=len(candidates)
            if cached_rows>max_candidates*8:return result('unknown',reason='Static domain cache memory budget exhausted')
        tested+=count;static_counts[i]=available
        if not available:return result('infeasible_model',reason='No static convex/obstacle domain fits an item',blocking_entities=[names[i]])
        if max_nodes<=0 or time.perf_counter()-search_started>time_limit_s:
            return result('unknown',reason='Search budget exhausted; no infeasibility claim')
    legacy_order=list(ordered);constrained_order=sorted(ordered,key=lambda i:static_counts[i])
    # Rotating a non-square box supplies another pose, not another physical
    # compartment. Count actual cavities before applying the largest-first
    # packing heuristic; keep the previous orders as bounded alternatives.
    group_membership=group_ids[:,None]==th.arange(len(bins),device=device)[None,:]
    physical_counts=(fit.any(-1)[:,:,None]&group_membership[None,:,:]).any(1).sum(-1)
    physical_order=by_area[th.argsort(physical_counts[by_area],stable=True)].cpu().tolist()
    search_orders=[]
    for candidate_order in [physical_order,constrained_order,legacy_order]:
        if candidate_order not in search_orders:search_orders.append(candidate_order)
    ordered=search_orders[0];order_index=0;orders_tried=1
    remaining_budget=max(0.,time_limit_s-(time.perf_counter()-search_started))
    order_deadline=time.perf_counter()+remaining_budget/len(search_orders)
    order_node_deadline=max(1,max_nodes//len(search_orders))
    frames=[];placed=[];depth=0
    while True:
        if depth==len(names):
            assignments={};poses={}
            host=th.stack(placed).cpu().tolist()
            for item_index,record in zip(ordered,host):
                b,x,y,w,h,rot=record;name=names[item_index];w-=clearance;h-=clearance
                assignments[name]=bins[b]
                poses[name]=dict(bin=bins[b],yaw_degrees=90*rot,
                    lower_xy_m=[(x+clearance)*grid_m,(y+clearance)*grid_m],
                    footprint_m=[w*grid_m,h*grid_m],
                    center_xy_m=[(x+clearance+w/2)*grid_m,(y+clearance+h/2)*grid_m])
            return result('feasible_model',assignment=assignments,poses=poses)
        if nodes>=max_nodes:
            return result('unknown',reason='Search budget exhausted; no infeasibility claim')
        if time.perf_counter()>order_deadline or nodes>=order_node_deadline:
            if order_index+1<len(search_orders) and time.perf_counter()-search_started<time_limit_s:
                order_index+=1;ordered=search_orders[order_index];orders_tried+=1
                frames=[];placed=[];depth=0
                remaining_orders=len(search_orders)-order_index
                order_deadline=time.perf_counter()+max(0.,time_limit_s-(time.perf_counter()-search_started))/remaining_orders
                order_node_deadline=nodes+max(1,(max_nodes-nodes)//remaining_orders)
                continue
            return result('unknown',reason='Search budget exhausted; no infeasibility claim')
        if len(frames)<=depth:
            i=ordered[depth];choices=th.nonzero(fit[i])
            bin_indices=choices[:,0];rot_indices=choices[:,1]
            dims=oriented[i,rot_indices]
            load=th.zeros(len(bins),device=device,dtype=th.float64)
            if placed:
                previous=th.stack(placed)
                load.scatter_add_(0,group_ids[previous[:,0]],(previous[:,3]*previous[:,4]).to(th.float64))
            priority=(load[group_ids[bin_indices]]+dims.prod(-1))/capacity[bin_indices].prod(-1).clamp_min(1)
            order=th.argsort(priority,stable=True);choices=choices[order];dims=dims[order]
            chunks=[]
            for choice,dimension in zip(choices,dims):
                bin_index,rotation=choice;w,h=dimension
                index=int(bin_index)
                candidates=static_domains[i,index,int(rotation)]
                b,x,y,widths,heights,rot=candidates.unbind(-1)
                valid=th.ones(len(candidates),dtype=th.bool,device=device)
                if placed:
                    previous=th.stack(placed);same=group_ids[b][:,None]==group_ids[previous[:,0]][None,:]
                    separate=(x[:,None]+w<=previous[None,:,1]) | (previous[None,:,1]+previous[None,:,3]<=x[:,None])
                    separate |= (y[:,None]+h<=previous[None,:,2]) | (previous[None,:,2]+previous[None,:,4]<=y[:,None])
                    valid &= (~same|separate).all(-1)
                chunks.append(candidates[valid])
            candidates=th.cat(chunks,0)
            frames.append([candidates,0])
        candidates,cursor=frames[depth]
        if cursor>=len(candidates):
            frames.pop()
            if depth==0:return result('infeasible_model',reason='Exhausted finite normal-pattern search tree')
            depth-=1;placed.pop();continue
        frames[depth][1]+=1;placed.append(candidates[cursor]);depth+=1;nodes+=1


def compact_candidate_frontier(candidates, stride):
    """GPU proposal subset: lattice corners blocked on both coordinate axes.

    Keep original candidate coordinates and constraints. Exhausting this
    smaller subset is UNKNOWN, never proof that the complete grid is empty.
    """
    import torch as th
    if len(candidates)==0:return candidates
    keys=candidates[:,2]*stride+candidates[:,1]
    order=th.argsort(keys,stable=True);keys=keys[order];rows=candidates[order]
    left=keys-1;below=keys-stride;right=keys+1;above=keys+stride
    il=th.searchsorted(keys,left).clamp_max(len(keys)-1)
    ib=th.searchsorted(keys,below).clamp_max(len(keys)-1)
    ir=th.searchsorted(keys,right).clamp_max(len(keys)-1)
    ia=th.searchsorted(keys,above).clamp_max(len(keys)-1)
    has_left=(rows[:,1]>0)&(keys[il]==left)
    has_below=(rows[:,2]>0)&(keys[ib]==below)
    has_right=(rows[:,1]<stride-1)&(keys[ir]==right)
    has_above=(rows[:,2]<stride-1)&(keys[ia]==above)
    return rows[(~has_left|~has_right)&(~has_below|~has_above)]


def solve_layout(items,volumes,fixed=None,*,grid_m=.001,margin_m=.005,time_limit_s=.5,
                 device="cuda",max_nodes=4096,max_candidates=1000000,center_halfspaces=None,obstacles=None,lateral_inset=False,allowed_yaws=None,overlap_groups=None):
    """Return feasible, infeasible or unknown in the explicitly stated model."""
    if grid_m<=0 or margin_m<0 or time_limit_s<=0:raise ValueError('INVALID_LAYOUT_SOLVER_LIMITS')
    for size in [*items.values(),*volumes.values()]:
        if len(size)!=3 or not all(math.isfinite(float(v)) and v>0 for v in size):raise ValueError('INVALID_LAYOUT_DIMENSIONS')
    fixed=fixed or {}
    halfspaces=[]
    for name,planes in sorted((center_halfspaces or {}).items()):
        if name not in volumes:raise ValueError('UNKNOWN_HALFSPACE_BIN')
        if any(len(p)!=4 or not all(math.isfinite(float(v)) for v in p) for p in planes):raise ValueError('INVALID_LAYOUT_HALFSPACES')
        if planes:halfspaces.append((name,tuple(tuple(float(v) for v in p) for p in planes)))
    obstacle_rows=[]
    for name,boxes in sorted((obstacles or {}).items()):
        if name not in volumes:raise ValueError('UNKNOWN_OBSTACLE_BIN')
        if any(len(b)!=6 or not all(math.isfinite(float(v)) for v in b) or any(b[k]>=b[k+3] for k in range(3)) for b in boxes):raise ValueError('INVALID_LAYOUT_OBSTACLES')
        if boxes:obstacle_rows.append((name,tuple(tuple(float(v) for v in b) for b in boxes)))
    normalized=tuple((n,tuple(sorted(v if isinstance(v,(set,list,tuple)) else [v]))) for n,v in sorted(fixed.items()))
    rotations=tuple((name,tuple(sorted(set(yaws)))) for name,yaws in sorted((allowed_yaws or {}).items()))
    if any(name not in items or not yaws or any(yaw not in (0,90) for yaw in yaws) for name,yaws in rotations):
        raise ValueError('INVALID_LAYOUT_YAWS')
    overlaps=tuple(sorted((overlap_groups or {}).items()))
    if any(name not in volumes or not isinstance(group,str) for name,group in overlaps):raise ValueError('INVALID_LAYOUT_OVERLAP_GROUPS')
    return copy.deepcopy(_solve_layout_gpu_cached(tuple((n,tuple(v)) for n,v in sorted(items.items())),
        tuple((n,tuple(v)) for n,v in sorted(volumes.items())),normalized,grid_m,margin_m,time_limit_s,device,max_nodes,max_candidates,tuple(halfspaces),tuple(obstacle_rows),bool(lateral_inset),rotations,overlaps))




def solve_layout_with_coarse_fallback(items,volumes,fixed=None,**kwargs):
    """Try a conservative 5-mm proposal grid after a bounded fine-grid miss.

    All dimensions round outward and all geometric constraints still apply.
    A coarse-grid failure cannot prove the original 1-mm model infeasible.
    Neither proposal certifies native physics or changes acceptance gates.
    """
    primary=solve_layout(items,volumes,fixed,**kwargs)
    retryable=('memory budget','Compact corner proposal subset exhausted')
    if primary['status']!='unknown' or not any(text in primary.get('reason','') for text in retryable):return primary
    fine=float(kwargs.get('grid_m',.001))
    if fine>=.005 or kwargs.get('max_nodes',4096)<=0:return primary
    coarse=solve_layout(items,volumes,fixed,**dict(kwargs,grid_m=.005))
    if coarse['status']=='feasible_model':
        return dict(coarse,proposal_fallback=dict(original_grid_m=fine,original_status=primary['status'],
                    original_reason=primary.get('reason'),policy='Conservative quantized witness; native validation required'))
    return dict(primary,coarse_proposal_attempt=dict(grid_m=.005,status=coarse['status'],reason=coarse.get('reason')))


def solve_authoring_layout(items,volumes,fixed=None,**kwargs):
    """Prefer spacious proposals, then try a fixed 5-mm authoring spacing.

    This changes empty space in generated layouts, never a native collision,
    support, grasp-safety or acceptance threshold. Static preflight already
    uses 5 mm. Every proposed pose still runs all original physical checks.
    """
    preferred=solve_layout_with_coarse_fallback(items,volumes,fixed,**kwargs)
    spacing=float(kwargs.get('margin_m',.005))
    if preferred['status']=='feasible_model' or spacing<=.005:return preferred
    dense=solve_layout_with_coarse_fallback(items,volumes,fixed,**dict(kwargs,margin_m=.005))
    return dict(dense,authoring_spacing_fallback=dict(preferred_margin_m=spacing,minimum_margin_m=.005,
        preferred_status=preferred['status'],preferred_reason=preferred.get('reason'),
        native_physical_checks='unchanged; not a physical certificate'))


def native_drop_candidates(candidate,origin,halfspaces,lifts=(.015,.03)):
    """At most three deterministic drop starts within the same real cavity.

    The native sampler adds its own 1-cm lift and retains every rejection
    check. Never introduce random retries or move a candidate in XY.
    """
    import numpy as np
    pose=np.asarray(candidate,float);origin=np.asarray(origin,float);planes=np.asarray(halfspaces,float)
    if pose.shape!=(7,) or origin.shape!=(3,) or planes.ndim!=2 or planes.shape[1]!=4 or not np.isfinite(pose).all():
        raise ValueError('INVALID_NATIVE_DROP_CANDIDATE')
    result=[pose.tolist()]
    for lift in lifts[:2]:
        if not 0.<lift<=.04:raise ValueError('INVALID_NATIVE_DROP_LIFT')
        proposed=pose.copy();proposed[2]+=lift
        local=proposed[:3]-origin;local[2]+=.01
        if bool((planes[:,:3]@local+planes[:,3]<-.00001).all()):result.append(proposed.tolist())
    return result


def placement_proposal_dimensions(obj,placement):
    """Use the real rotated native box, without scaling or changing a mesh."""
    import numpy as np
    size=obj.native_bbox*obj.scale
    if hasattr(size,'detach'):size=size.detach().cpu().numpy()
    size=np.asarray(size,float)
    hint=getattr(placement,'orientation_hint_xyzw',None)
    if hint is not None:
        from scipy.spatial.transform import Rotation
        size=np.abs(Rotation.from_quat(hint).as_matrix())@size
    return size


def bbox_center_root_pose(center, yaw_degrees, scaled_root_offset,orientation_hint_xyzw=None):
    """Match DatasetObject.set_bbox_center_position_orientation's transform.

    The SDK offset is the root position relative to the desired bbox centre;
    This helper predicts the final root pose for diagnostics. The adapted
    Inside sampler consumes bbox centres and performs this conversion itself.
    """
    import numpy as np
    center=np.asarray(center,dtype=float);offset=np.asarray(scaled_root_offset,dtype=float)
    if center.shape!=(3,) or offset.shape!=(3,) or not np.isfinite([*center,*offset,yaw_degrees]).all():
        raise ValueError('INVALID_BBOX_ROOT_TRANSFORM')
    yaw=math.radians(yaw_degrees);c,s=math.cos(yaw),math.sin(yaw)
    rotation=np.asarray([[c,-s,0],[s,c,0],[0,0,1]])
    if orientation_hint_xyzw is not None:
        from scipy.spatial.transform import Rotation
        rotation=rotation@Rotation.from_quat(orientation_hint_xyzw).as_matrix()
        return [*(center+rotation@offset).tolist(),*Rotation.from_matrix(rotation).as_quat().tolist()]
    return [*(center+rotation@offset).tolist(),0.,0.,math.sin(yaw/2),math.cos(yaw/2)]


def regularize_native_box_floor(points,faces,*,thickness_m=.002):
    """Thicken a verified degenerate eight-corner floor in its parent frame.

    Keep XY and the original mid-plane. Face topology, not tiny Z noise,
    identifies the two sides. Caller limits this to the audited native asset.
    """
    import numpy as np
    points=np.asarray(points,float);faces=np.asarray(faces,int)
    if not np.isfinite(thickness_m) or not .0001<=thickness_m<=.005:raise ValueError('INVALID_REPAIR_THICKNESS')
    if points.shape!=(8,3) or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('UNKNOWN_FLOOR_GEOMETRY')
    extent=np.ptp(points,axis=0)
    if min(extent[:2])<.05 or not 0<extent[2]<.00001:raise ValueError('FLOOR_IS_NOT_VERIFIED_DEGENERATE_BOX')
    tris=points[faces];normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);length=np.linalg.norm(normals,axis=1)
    normals/=np.maximum(length[:,None],1e-30)
    inward=np.sum(normals*(tris.mean(1)-points.mean(0)),axis=1)<0;normals[inward]*=-1
    top=np.unique(faces[(normals[:,2]>.98)&(length>1e-12)])
    bottom=np.unique(faces[(normals[:,2]<-.98)&(length>1e-12)])
    if len(top)!=4 or len(bottom)!=4 or len(np.intersect1d(top,bottom)):raise ValueError('UNKNOWN_FLOOR_FACE_TOPOLOGY')
    mid=(points[:,2].min()+points[:,2].max())/2;fixed=points.copy()
    fixed[top,2]=mid+thickness_m/2;fixed[bottom,2]=mid-thickness_m/2
    return fixed


def fit_inclined_support(hits, *, max_residual_m=.0005, max_height_spread_m=.03):
    """Recognize one measured inclined collider, only for drop proposals.

    A plane fit alone can bridge separate steps. Require all nine owned hits,
    full-rank XY coverage and agreement of every measured collision normal.
    Actual settling, contact and native Inside remain authoritative.
    """
    import numpy as np
    if len(hits)!=9 or not all(h.get('valid') for h in hits):return None
    points=np.asarray([h['position'] for h in hits],float)
    normals=np.asarray([h['normal'] for h in hits],float)
    if points.shape!=(9,3) or normals.shape!=(9,3) or not np.isfinite([points,normals]).all():return None
    if len({h['body'] for h in hits})!=1:return None
    design=np.column_stack((points[:,:2]-points[:,:2].mean(0),np.ones(9)))
    coefficients,_,rank,_=np.linalg.lstsq(design,points[:,2],rcond=None)
    if rank!=3:return None
    residual=float(np.max(np.abs(design@coefficients-points[:,2])))
    spread=float(np.ptp(points[:,2]));normal=np.array([-coefficients[0],-coefficients[1],1.])
    normal/=np.linalg.norm(normal)
    lengths=np.linalg.norm(normals,axis=1)
    if (lengths<=0).any():return None
    agreement=(normals/lengths[:,None])@normal
    if residual>max_residual_m or spread>max_height_spread_m or normal[2]<.98 or (agreement<.9995).any():return None
    return dict(floor_z_m=float(points[:,2].max()),height_spread_m=spread,
        measured_height_range_m=[float(points[:,2].min()),float(points[:,2].max())],
        inclined_support=True,normal=normal.tolist(),plane_fit_max_residual_m=residual,
        scope='Measured collider drop proposal; native settling and membership required')


def measure_support_plane(lo, hi, owner_paths, raycast=None):
    """Probe actual empty-cavity collision floors, without changing physics.

    Sparse rays guide proposals only; native collision/Inside validation is
    still required. Unknown or uneven surfaces are not treated as flat floors.
    """
    import numpy as np
    if raycast is None:
        from omnigibson.utils.sampling_utils import raytest
        raycast=raytest
    lo,hi=np.asarray(lo,dtype=float),np.asarray(hi,dtype=float)
    heights=[];hits=[]
    for x in (.2,.5,.8):
        for y in (.2,.5,.8):
            xy=lo[:2]+(hi[:2]-lo[:2])*[x,y]
            hit=raycast([*xy,float((lo[2]+hi[2])/2)],[*xy,float(lo[2]-.12)])
            body=str(hit.get('rigidBody',''));position=np.asarray(hit.get('position',[]),dtype=float)
            normal=np.asarray(hit.get('normal',[]),dtype=float)
            owned=any(body==p or body.startswith(p+'/') for p in owner_paths)
            valid=(hit.get('hit') and owned and position.shape==(3,) and normal.shape==(3,)
                   and np.isfinite(position).all() and np.isfinite(normal).all() and normal[2]>=.98)
            hits.append(dict(hit=bool(hit.get('hit')),body=body,position=position.tolist(),normal=normal.tolist(),valid=bool(valid)))
            if valid:heights.append(float(position[2]))
    report=dict(status='unknown',hits=hits,probe_count=9,physical_validation_required=True)
    if len(heights)!=9:
        return dict(report,reason='Missing, foreign or non-horizontal collision support')
    spread=max(heights)-min(heights)
    if spread>.003:return dict(report,reason='Collision support is not a single plane',height_spread_m=spread)
    floor=max(heights)
    if not lo[2]-.12<=floor<hi[2]-.015:return dict(report,reason='Support plane outside usable vertical range')
    return dict(report,status='measured_plane',floor_z_m=floor,height_spread_m=spread,
                fillable_bottom_z_m=float(lo[2]),floor_minus_fillable_bottom_m=float(floor-lo[2]))


def measure_shelf_drop_height(xy,footprint,receptacle_aabb,owner_paths,raycast):
    """Find a real shelf below its outer frame, solely as a drop proposal.

    All nine footprint rays must hit the named shelf at one height. A foreign
    object, hole or vertical panel cannot supply a missing support sample.
    Native OnTop, collision checks and settling still decide acceptance.
    """
    import numpy as np
    xy=np.asarray(xy,float);size=np.asarray(footprint,float)
    lo,hi=np.asarray(receptacle_aabb,float)
    if xy.shape!=(2,) or size.shape!=(2,) or np.any(size<=0) or not np.isfinite([*xy,*size,*lo,*hi]).all():
        raise ValueError('INVALID_SHELF_DROP_GEOMETRY')
    for fraction in (.95,.7,.45,.2):
        start=float(lo[2]+fraction*(hi[2]-lo[2]));heights=[]
        for dx in (-.3,0.,.3):
            for dy in (-.3,0.,.3):
                point=xy+size*[dx,dy]
                hit=raycast([*point,start],[*point,float(lo[2]-.02)])
                body=str(hit.get('rigidBody',''));position=np.asarray(hit.get('position',[]),float)
                normal=np.asarray(hit.get('normal',[]),float)
                if (hit.get('hit') and any(body==p or body.startswith(p+'/') for p in owner_paths)
                        and position.shape==(3,) and normal.shape==(3,) and np.isfinite([position,normal]).all()
                        and normal[2]>=.98 and lo[2]-.001<=position[2]<start):
                    heights.append(float(position[2]))
        if len(heights)==9 and max(heights)-min(heights)<=.003:
            return dict(height_m=max(heights),probe_start_m=start,probe_count=9,
                        physical_validation_required=True)
    return None


def measure_support_patches(lo, hi, owner_paths, raycast=None, *, grid_size=7, max_patches=4):
    """Find disjoint flat support patches when a complete cavity is not flat.

    Rectangles are bounded by measured grid points, not extrapolated over a
    missing/foreign/sloped hit. They are proposal domains, never certificates.
    """
    import numpy as np
    if raycast is None:
        from omnigibson.utils.sampling_utils import raytest
        raycast=raytest
    if grid_size<3 or max_patches<1:raise ValueError('INVALID_SUPPORT_PATCH_LIMITS')
    lo,hi=np.asarray(lo,dtype=float),np.asarray(hi,dtype=float)
    xs=np.linspace(lo[0]+.08*(hi[0]-lo[0]),hi[0]-.08*(hi[0]-lo[0]),grid_size)
    ys=np.linspace(lo[1]+.08*(hi[1]-lo[1]),hi[1]-.08*(hi[1]-lo[1]),grid_size)
    heights=np.full((grid_size,grid_size),np.nan)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            hit=raycast([float(x),float(y),float((lo[2]+hi[2])/2)],
                        [float(x),float(y),float(lo[2]-.12)])
            body=str(hit.get('rigidBody',''));p=np.asarray(hit.get('position',[]),dtype=float)
            normal=np.asarray(hit.get('normal',[]),dtype=float)
            if (hit.get('hit') and any(body==owner or body.startswith(owner+'/') for owner in owner_paths)
                and p.shape==(3,) and normal.shape==(3,) and np.isfinite(p).all()
                and np.isfinite(normal).all() and normal[2]>=.98 and lo[2]-.12<=p[2]<hi[2]-.015):
                heights[i,j]=p[2]
    rectangles=np.asarray([(a,b,c,d) for a in range(grid_size-1) for b in range(a+1,grid_size)
                           for c in range(grid_size-1) for d in range(c+1,grid_size)])
    gx,gy=np.mgrid[:grid_size,:grid_size]
    masks=((gx[None]>=rectangles[:,0,None,None])&(gx[None]<=rectangles[:,1,None,None])&
           (gy[None]>=rectangles[:,2,None,None])&(gy[None]<=rectangles[:,3,None,None]))
    valid=(~masks|np.isfinite(heights)[None]).all((1,2))
    low=np.where(masks,np.nan_to_num(heights,nan=np.inf)[None],np.inf).min((1,2))
    high=np.where(masks,np.nan_to_num(heights,nan=-np.inf)[None],-np.inf).max((1,2))
    valid &= (high-low<=.003)
    areas=(xs[rectangles[:,1]]-xs[rectangles[:,0]])*(ys[rectangles[:,3]]-ys[rectangles[:,2]])
    valid &= areas>=.0025
    patches=[]
    for _ in range(max_patches):
        if not valid.any():break
        index=int(np.argmax(np.where(valid,areas,-1.)))
        a,b,c,d=rectangles[index];floor=float(high[index])
        patches.append(dict(lo=[float(xs[a]),float(ys[c]),floor],hi=[float(xs[b]),float(ys[d]),float(hi[2])],
            status='measured_patch',floor_z_m=floor,height_spread_m=float(high[index]-low[index]),
            fillable_bottom_z_m=float(lo[2]),floor_minus_fillable_bottom_m=float(floor-lo[2]),
            probe_count=grid_size**2,grid_indices=[int(a),int(b),int(c),int(d)],
            physical_validation_required=True))
        # Domains may touch at a boundary, but their interiors never overlap.
        overlap=(rectangles[:,0]<b)&(rectangles[:,1]>a)&(rectangles[:,2]<d)&(rectangles[:,3]>c)
        valid &= ~overlap
    return patches


def native_surface_group(scenario,receptacle_name):
    """Dynamic authored props sharing a top; never floor-mounted furniture."""
    if receptacle_name in {getattr(g,'receptacle',None) for g in getattr(scenario,'goal',[])}:
        return []  # Preserve the qualified reference/delivery layout.
    fixed={o.name for o in scenario.objects if getattr(o,'fixed_base',False)}
    parents={p.receptacle for p in scenario.placements if p.relation=='inside'}
    return [p for p in scenario.placements if p.relation=='on_top'
            and p.receptacle==receptacle_name and p.entity not in fixed|parents]


def surface_patch_layout(items,patches,*,time_limit_s=2.):
    """Pack complete footprints in measured rectangles, not a body's AABB."""
    import numpy as np
    valid={f'patch:{i}':p for i,p in enumerate(patches) if p.get('status')=='measured_patch'}
    volumes={k:[*list((np.asarray(p['hi'])-p['lo'])[:2]),4.] for k,p in valid.items()}
    solved=None
    for margin in (.025,.015,.005):
        solved=solve_layout(items,volumes,margin_m=margin,time_limit_s=time_limit_s,max_candidates=4000000)
        if solved['status']=='feasible_model':break
    if solved['status']!='feasible_model':
        for i,patch in enumerate(patches):
            if patch.get('status')!='support_hull_candidate':continue
            key=f'hull:{i}';low=np.asarray(patch['lo']);extent=np.asarray(patch['hi'])-low
            planes=[[a,b,0.,d+a*low[0]+b*low[1]] for a,b,d in patch['center_halfspaces_world_xy']]
            for margin in (.025,.015,.005):
                candidate=solve_layout(items,{key:[*extent[:2],4.]},margin_m=margin,time_limit_s=time_limit_s,
                    max_candidates=4000000,center_halfspaces={key:planes},lateral_inset=True)
                if candidate['status']=='feasible_model':solved=candidate;valid[key]=patch;break
            if solved['status']=='feasible_model':break
    if solved['status']!='feasible_model':return solved,{}
    plan={}
    for name,pose in solved['poses'].items():
        patch=valid[pose['bin']]
        xy=np.asarray(patch['lo'][:2])+pose['center_xy_m']
        plan[name]=dict(position_xy=xy.tolist(),support_height_m=float(patch['floor_z_m']),
            yaw_degrees=pose['yaw_degrees'],support_patch=patch,
            footprint_m=pose['footprint_m'],physical_validation_required=True)
    return solved,plan


def plan_native_surface_group(backend,scenario,receptacle_name):
    """Initialization only: measure real owned support and jointly place props.

    Sparse collision rays propose domains. Native OnTop, settling and the
    independent frozen reload remain authoritative. No surface is fabricated.
    """
    import numpy as np,json
    def array(x):return np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x,float)
    rec=backend._env.scene.object_registry('name',receptacle_name)
    lo,hi=(array(v) for v in rec.aabb)
    probe_lo=lo.copy();probe_lo[2]=hi[2]-.02
    probe_hi=hi.copy();probe_hi[2]+=.3
    owners=[link.prim_path for link in rec.links.values()]
    patches=measure_support_patches(probe_lo,probe_hi,owners,grid_size=17,max_patches=8)
    envelope=support_envelope_candidate(probe_lo,probe_hi,patches)
    if envelope is not None:patches.append(envelope)
    items={}
    for p in native_surface_group(scenario,receptacle_name):
        obj=backend._env.scene.object_registry('name',p.entity)
        items[p.entity]=array(obj.native_bbox*obj.scale).tolist()
    # Reserve the whole initial pile's footprint, not only its bottom book.
    children={p.entity:[] for p in scenario.placements}
    for p in scenario.placements:
        if p.relation=='on_top':children.setdefault(p.receptacle,[]).append(p.entity)
    for name in items:
        pending=list(children.get(name,[]));seen={name}
        while pending:
            child=pending.pop()
            if child in seen:continue
            seen.add(child);pending.extend(children.get(child,[]))
            body=backend._env.scene.object_registry('name',child)
            size=array(body.native_bbox*body.scale)
            items[name]=[max(items[name][0],float(size[0])),max(items[name][1],float(size[1])),items[name][2]+float(size[2])]
    solved,plan=surface_patch_layout(items,patches)
    print('[expert-surface-layout] '+json.dumps(dict(receptacle=receptacle_name,
        items=items,patches=patches,solver=solved,physical_validation_required=True)),flush=True)
    if solved['status']!='feasible_model':
        raise ValueError('EXPERT_SURFACE_LAYOUT_UNRESOLVED: '+solved['status']+' '+solved.get('reason',''))
    return plan


def refine_layout_clearance(items,volumes,solved,halfspaces,*,margin_m=.015,expansion=1.3,device='cuda',lateral_inset=False):
    """Spread compacted groups into unused space without losing their constraints.

    Scaling XY centre separations by a factor >= 1 preserves axis-aligned box
    separation. A common translation preserves it as well. Convex halfspaces
    bound the interpolation analytically; this is only a native pose proposal.
    """
    import torch as th
    result=copy.deepcopy(solved)
    if solved.get('status')!='feasible_model':return result
    refinements={}
    for bin_name,size in volumes.items():
        names=[n for n,p in result['poses'].items() if p['bin']==bin_name]
        if not names:continue
        centers=th.tensor([result['poses'][n]['center_xy_m'] for n in names],dtype=th.float64,device=device)
        widths=th.tensor([result['poses'][n]['footprint_m'] for n in names],dtype=th.float64,device=device)
        capacity=th.tensor(size[:2],dtype=th.float64,device=device)
        origin=(centers.amin(0)+centers.amax(0))/2
        desired=(centers-origin)*max(1.,expansion)
        lower=(desired-widths/2).amin(0);upper=(desired+widths/2).amax(0)
        if not bool(((upper-lower)<=capacity-2*margin_m).all()):
            desired=centers.clone();lower=(desired-widths/2).amin(0);upper=(desired+widths/2).amax(0)
        desired+=(capacity-lower-upper)/2
        delta=desired-centers;alpha=th.ones((),dtype=th.float64,device=device)
        if halfspaces.get(bin_name):
            planes=th.tensor(halfspaces[bin_name],dtype=th.float64,device=device)
            z=th.tensor([items[n][2]/2 for n in names],dtype=th.float64,device=device)
            value=th.cat((centers,z[:,None]),-1)@planes[:,:3].T+planes[:,3]+.00001
            inset=((widths/2)@planes[:,:2].abs().T)*(th.linalg.vector_norm(planes[:,:2],dim=1)>.5) if lateral_inset else 0.
            value+=inset
            derivative=delta@planes[:,:2].T
            limits=th.where(derivative>0,(-value/derivative.clamp_min(1e-15)).clamp_min(0.),th.inf)
            alpha=th.minimum(alpha,limits.amin()).clamp(0.,1.)*.999
        updated=centers+alpha*delta
        inside=((updated-widths/2>=margin_m-1e-9)&(updated+widths/2<=capacity-margin_m+1e-9)).all()
        if not bool(inside):continue
        # All operations above retain non-overlap; verify convex constraints
        # again after finite-precision interpolation before changing any pose.
        if halfspaces.get(bin_name):
            query=th.cat((updated,z[:,None]),-1)@planes[:,:3].T+planes[:,3]+inset
            if not bool((query<-.00001).all()):continue
        host=updated.cpu().tolist();a=float(alpha)
        for name,center in zip(names,host):
            pose=result['poses'][name];old=pose['center_xy_m'];shift=[center[i]-old[i] for i in range(2)]
            pose['center_xy_m']=center;pose['lower_xy_m']=[pose['lower_xy_m'][i]+shift[i] for i in range(2)]
        refinements[bin_name]=dict(interpolation_fraction=a,max_expansion=expansion,objects=len(names))
    result['clearance_refinement']=refinements
    return result


def support_envelope_candidate(lo,hi,patches):
    """An extra placement candidate spanning real near-coplanar supports.

    This hull is a search heuristic, not an asserted floor or a stability
    certificate. No collision body is created; native physics must accept it.
    """
    import numpy as np
    from scipy.spatial import ConvexHull,QhullError
    measured=[p for p in patches if p.get('status')=='measured_patch']
    if len(measured)<2:return None
    heights=np.asarray([p['floor_z_m'] for p in measured],float)
    if not np.isfinite(heights).all() or np.ptp(heights)>.005:return None
    points=[]
    for patch in measured:
        lower,upper=np.asarray(patch['lo']),np.asarray(patch['hi'])
        points.extend([[x,y] for x in (lower[0],upper[0]) for y in (lower[1],upper[1])])
    try:hull=ConvexHull(np.asarray(points,float))
    except QhullError:return None
    lower=[float(lo[0]),float(lo[1]),float(heights.max())]
    if hi[2]<=lower[2]:return None
    return dict(status='support_hull_candidate',lo=lower,hi=np.asarray(hi).tolist(),floor_z_m=lower[2],
        center_halfspaces_world_xy=hull.equations.tolist(),measured_support_count=len(measured),
        measured_height_range_m=[float(heights.min()),float(heights.max())],
        physical_validation_required=True,scope='candidate support envelope; no synthetic floor')


def plan_native_contents(backend, scenario, receptacle_name,placed_entities=()):
    """Solve actual opened native cavity components, returning private proposals.

    Only initialization uses this plan. Native Inside sampling still performs
    its collision, support, container-drift and membership checks on each pose.
    """
    synchronize_authoring_geometry(backend)
    import numpy as np
    from omnigibson.object_states.contains import m as contains_macros
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float)
    placements=[p for p in scenario.placements if p.relation=='inside' and p.receptacle==receptacle_name]
    rec=backend._env.scene.object_registry('name',receptacle_name)
    raycast=None
    if placed_entities:
        from omnigibson.utils.sampling_utils import raytest
        ignored=[link.prim_path for name in placed_entities
                 for link in backend._env.scene.object_registry('name',name).links.values()]
        def raycast(start,end):return raytest(start,end,ignore_bodies=ignored)
    import json
    volumes={};bounds={};native={};membership_bottom={};support_reports={};halfspaces={};components={};overlap_groups={}
    from scipy.spatial import ConvexHull
    owners=[link.prim_path for link in rec.links.values()]
    from omnigibson.object_states import Open
    open_top=(getattr(rec,'category',None) in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray') and Open not in rec.states)
    # Perforated/open-bottom baskets can legitimately rest their contents on
    # the table supporting the basket. Reject unrelated foreign bodies, but
    # permit this actual native-verified direct support chain in proposals.
    parent=next((p for p in scenario.placements if p.entity==receptacle_name and p.relation=='on_top'),None)
    if open_top and parent is not None:
        from omnigibson.object_states import OnTop
        support_obj=backend._env.scene.object_registry('name',parent.receptacle)
        if OnTop in rec.states and bool(rec.states[OnTop].get_value(support_obj)):
            supported_by=[];seen=set()
            for other in [support_obj,*getattr(backend._env.scene,'objects',[])]:
                if other is rec or id(other) in seen:continue
                seen.add(id(other))
                if not bool(rec.states[OnTop].get_value(other)):continue
                owners.extend(link.prim_path for link in other.links.values())
                supported_by.append(getattr(other,'name',parent.receptacle))
            print('[expert-container-support] '+json.dumps(dict(receptacle=receptacle_name,
                native_on_top_parent=parent.receptacle,native_verified_supports=supported_by,
                scope='proposal ownership only; native physical checks unchanged')),flush=True)
    for name,link in rec.links.items():
        if not link.is_meta_link or link.meta_link_type not in contains_macros.CONTAINER_META_LINK_TYPES:continue
        for index,mesh in enumerate(link.visual_meshes.values()):
            points=array(mesh.points);transform=array(mesh.scaled_transform)
            world=points@transform[:3,:3].T+transform[:3,3]
            lo,hi=world.min(0),world.max(0)
            # Native Inside uses these convex visual-mesh boundaries, not the
            # mesh's world AABB. Keep its actual sloping faces in the GPU CSP.
            equations=ConvexHull(world).equations
            key=f'{name}#mesh:{index}'
            components[key]=dict(link=name,lo=lo,hi=hi,equations=equations)
            support=(measure_support_plane(lo,hi,owners,raycast=raycast) if raycast is not None
                     else measure_support_plane(lo,hi,owners));support_reports[key]=support
            print('[expert-support-plane] '+json.dumps(dict(receptacle=receptacle_name,component=key,**support)),flush=True)
            if support['status']=='measured_plane':
                domains=[dict(lo=[float(lo[0]),float(lo[1]),support['floor_z_m']],hi=hi.tolist(),**support)]
            else:
                domains=(measure_support_patches(lo,hi,owners,raycast=raycast) if raycast is not None
                         else measure_support_patches(lo,hi,owners))
                envelope=support_envelope_candidate(lo,hi,domains)
                if envelope is not None:domains=[*domains,envelope]
            for patch_index,domain in enumerate(domains):
                patch_key=(key if domain['status']=='measured_plane' else key+'#support_hull'
                           if domain['status']=='support_hull_candidate' else f'{key}#support_patch:{patch_index}')
                lower,upper=np.asarray(domain['lo']),np.asarray(domain['hi'])
                patch_planes=[]
                if domain['status']=='measured_patch':
                    # A box can bridge adjacent real floor patches. Constrain
                    # its centre to measured support, but fit its footprint
                    # in the native cavity; never invent a support surface.
                    patch_lo,patch_hi=lower.copy(),upper.copy()
                    lower=np.array([lo[0],lo[1],domain['floor_z_m']]);upper=hi.copy()
                    pad=min(.003,float(np.min(patch_hi[:2]-patch_lo[:2]))/4)
                    patch_planes=[[-1.,0.,0.,patch_lo[0]+pad-lower[0]],
                        [1.,0.,0.,lower[0]-patch_hi[0]+pad],[0.,-1.,0.,patch_lo[1]+pad-lower[1]],
                        [0.,1.,0.,lower[1]-patch_hi[1]+pad]]
                    overlap_groups[patch_key]=key
                elif domain['status']=='support_hull_candidate':
                    patch_planes=[[a,b,0.,d+a*lower[0]+b*lower[1]+.003]
                                  for a,b,d in domain['center_halfspaces_world_xy']]
                    overlap_groups[patch_key]=key
                    print('[expert-support-hull-candidate] '+json.dumps(dict(receptacle=receptacle_name,component=patch_key,**domain)),flush=True)
                if open_top:
                    # An open basket can contain a standing carton whose top
                    # protrudes. Native Inside checks its centre; there is no
                    # lid/ceiling. Native collision and membership still gate it.
                    upper=upper.copy();upper[2]=lower[2]+2*(upper[2]-lower[2])-.02
                    domain=dict(domain,open_top_centroid_height=True)
                if upper[2]<=lower[2]:continue
                membership_bottom[patch_key]=float(lo[2]);support_reports[patch_key]=domain
                volumes[patch_key]=(upper-lower).tolist();bounds[patch_key]=(lower,upper);native[patch_key]=name
                local=equations.copy();local[:,3]+=local[:,:3]@lower
                halfspaces[patch_key]=local.tolist()+patch_planes
                if domain['status']=='measured_patch':
                    print('[expert-support-patch] '+json.dumps(dict(receptacle=receptacle_name,component=patch_key,**domain)),flush=True)
    if not volumes:raise ValueError('EXPERT_SUPPORT_UNRESOLVED: no measured flat empty-cavity collision floor')
    context=dict(volumes=volumes,bounds=bounds,native=native,membership_bottom=membership_bottom,
                 support_reports=support_reports,halfspaces=halfspaces,open_top=open_top,components=components,overlap_groups=overlap_groups)
    if not hasattr(backend,'_findingbench_contents_contexts'):backend._findingbench_contents_contexts={}
    backend._findingbench_contents_contexts[receptacle_name]=context
    return replan_native_contents(backend,scenario,receptacle_name,placed_entities,domains_current=True)


def add_native_stacking_supports(backend,context,placed_entities):
    """Propose real, measured tops of already validated contained objects.

    No body or collision mesh is added. Child centres stay above an actual
    measured flat support and inside the unchanged native convex cavity.
    Native placement still decides stability, contact and membership.
    """
    import numpy as np,json
    def array(value):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value,'detach') else value,float)
    result={k:dict(context[k]) for k in ('volumes','bounds','native','membership_bottom','support_reports','halfspaces')}
    for entity in sorted(placed_entities):
        obj=backend._env.scene.object_registry('name',entity);lo,hi=(array(v) for v in obj.aabb)
        if not np.isfinite([*lo,*hi]).all() or min((hi-lo)[:2])<.025:continue
        center=(lo+hi)/2
        for component,geometry in context.get('components',{}).items():
            equations=geometry['equations'];lower0=geometry['lo'];upper=geometry['hi']
            if not np.all(equations[:,:3]@center+equations[:,3]<=.00001):continue
            if hi[2]+.002>=upper[2]:continue
            patch_lo=np.maximum(lo[:2],lower0[:2]);patch_hi=np.minimum(hi[:2],upper[:2])
            if np.any(patch_hi-patch_lo<.025):continue
            probe_lo=np.array([*patch_lo,hi[2]-.003]);probe_hi=np.array([*patch_hi,min(hi[2]+.05,upper[2])])
            # Already placed children can occlude their support's surface.
            # Ignore them only in this measurement query. They remain real
            # collidable bodies and remain obstacles in the contents CSP.
            ignored=[link.prim_path for other in sorted(placed_entities) if other!=entity
                     for link in backend._env.scene.object_registry('name',other).links.values()]
            owners=[link.prim_path for link in obj.links.values()]
            if ignored:
                def raycast(start,end):
                    from omnigibson.utils.sampling_utils import raytest
                    return raytest(start,end,ignore_bodies=ignored)
                support=measure_support_plane(probe_lo,probe_hi,owners,raycast=raycast)
            else:support=measure_support_plane(probe_lo,probe_hi,owners)
            if support['status']!='measured_plane':continue
            lower=np.array([*lower0[:2],support['floor_z_m']]);key=component+'#on:'+entity
            planes=equations.copy();planes[:,3]+=planes[:,:3]@lower
            # A flat paper/book may overhang its support while its centre of
            # mass remains supported. Cavity bounds and every native physical
            # check still apply to the complete object.
            pad=min(.005,float(np.min(patch_hi-patch_lo))/4)
            support_planes=np.array([[-1.,0.,0.,patch_lo[0]+pad-lower[0]],
                [1.,0.,0.,lower[0]-patch_hi[0]+pad],[0.,-1.,0.,patch_lo[1]+pad-lower[1]],
                [0.,1.,0.,lower[1]-patch_hi[1]+pad]])
            result['volumes'][key]=(upper-lower).tolist();result['bounds'][key]=(lower,upper)
            result['native'][key]=geometry['link'];result['membership_bottom'][key]=float(lower0[2])
            result['support_reports'][key]=dict(support,support_entity=entity,relation='native contained-object top')
            result['halfspaces'][key]=np.vstack((planes,support_planes)).tolist()
            print('[expert-stacking-support] '+json.dumps(dict(component=component,support_entity=entity,
                floor_z_m=support['floor_z_m'],ray_query_ignored_placed_bodies=ignored,
                placed_bodies_still_collidable=True,physical_validation_required=True)),flush=True)
    return result


def synchronize_authoring_geometry(backend):
    """Refresh graphics-space geometry from PhysX without stepping physics."""
    sync=getattr(getattr(backend,'_sim',None),'sync_physx_to_fabric',None)
    if callable(sync):sync();return True
    return False


def current_collision_aabb(obj):
    """Read collision geometry at current native body poses, without writes.

    SDK mesh world transforms come from Fabric, while dynamic body poses come
    from PhysX. During settling those can describe different frames. Planning
    must use the settled body pose rather than a previous graphics transform.
    Static local collision hulls and their authored scale are unchanged.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    def array(value):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value,'detach') else value,float)
    links=getattr(obj,'links',{}) or {};bounds=[]
    for link in links.values():
        if not hasattr(link,'collision_boundary_points_local'):return tuple(array(v) for v in obj.aabb)
        points=link.collision_boundary_points_local
        if points is None:continue
        points=array(points);matrix=array(link.scaled_transform)
        position,orientation=map(array,link.get_position_orientation())
        if (points.ndim!=2 or points.shape[1]!=3 or matrix.shape!=(4,4) or
                position.shape!=(3,) or orientation.shape!=(4,) or
                not np.isfinite(np.r_[points.ravel(),matrix.ravel(),position,orientation]).all()):
            raise ValueError('INVALID_NATIVE_COLLISION_GEOMETRY')
        linear=matrix[:3,:3];gram=linear.T@linear;scale=np.sqrt(np.diag(gram))
        if np.any(scale<=0):raise ValueError('INVALID_NATIVE_COLLISION_SCALE')
        if np.linalg.det(linear)<0 or not np.allclose(gram,np.diag(scale*scale),atol=1e-5,rtol=1e-5):
            # Unusual inherited shear gets a conservative enclosing sphere.
            # This can only reject an extra proposal, never erase an obstacle.
            radius=float(np.linalg.norm(points,axis=1).max()*np.linalg.norm(linear,ord=2))
            bounds.append((position-radius,position+radius));continue
        world=(points*scale)@Rotation.from_quat(orientation).as_matrix().T+position
        bounds.append((world.min(axis=0),world.max(axis=0)))
    if not bounds:return tuple(array(v) for v in obj.aabb)
    return np.min([b[0] for b in bounds],axis=0),np.max([b[1] for b in bounds],axis=0)


def remember_native_contents_plan(backend,receptacle_name,context,plan):
    """Keep a whole-layout candidate; this is never a physics certificate."""
    import numpy as np
    rec=backend._env.scene.object_registry('name',receptacle_name)
    joints=rec.get_joint_positions() if getattr(rec,'n_joints',0) else []
    if hasattr(joints,'detach'):joints=joints.detach().cpu().numpy()
    context['pending_plan']=copy.deepcopy(plan)
    context['pending_plan_joints']=np.asarray(joints,dtype=float).tolist()


def reusable_native_contents_plan(backend,receptacle_name,context,unplaced,placed_entities):
    """Reuse candidates only while joints and actual settled obstacles allow.

    A small XY clearance screens candidate reuse, not native acceptance. The
    original Inside sampler still checks every contact, support and final
    relation. Moving/tilting contents invalidate overlapping future poses.
    """
    import numpy as np
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float)
    pending=context.get('pending_plan',{})
    names=[p.entity for p in unplaced]
    if not names or not all(n in pending for n in names):return None
    rec=backend._env.scene.object_registry('name',receptacle_name)
    joints=array(rec.get_joint_positions() if getattr(rec,'n_joints',0) else [])
    previous=array(context.get('pending_plan_joints',[]))
    if joints.shape!=previous.shape or not np.isfinite(joints).all() or np.any(abs(joints-previous)>.001):return None
    obstacles=[tuple(array(v) for v in current_collision_aabb(backend._env.scene.object_registry('name',n)))
               for n in placed_entities]
    for name in names:
        pose=pending[name];obj=backend._env.scene.object_registry('name',name)
        center=array(pose['bbox_center_world'])
        footprint=array(pose['solver_pose']['footprint_m'])
        size=np.r_[footprint,pose['solver_pose'].get('oriented_dimensions_m',array(obj.native_bbox*obj.scale))[2]]
        lower,upper=center-size/2,center+size/2
        if not np.isfinite(np.r_[lower,upper]).all():return None
        for lo,hi in obstacles:
            # A real carrier touching from below remains valid support.
            if upper[2]<=lo[2]+1e-6 or hi[2]<=lower[2]+1e-6:continue
            if np.all(upper[:2]+.002>lo[:2]) and np.all(hi[:2]+.002>lower[:2]):return None
    return {n:copy.deepcopy(pending[n]) for n in names}


def support_center_lower_bound(floor_height,item_height,report):
    """Conservative centre height over the actually measured support interval."""
    spread=max(0.,float(report.get('height_spread_m',0.)))
    interval=report.get('measured_height_range_m')
    if interval:spread=max(spread,float(max(interval)-min(interval)))
    return float(floor_height)+float(item_height)/2-spread



def replan_native_contents(backend,scenario,receptacle_name,placed_entities,*,domains_current=False):
    """Pack unplaced items around the actual settled AABBs of placed contents.

    Empty-cavity domains are retained; probing again through placed contents
    would misidentify objects as foreign shelf support. No placed item moves.
    """
    synchronize_authoring_geometry(backend)
    import numpy as np,json
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float)
    rec=backend._env.scene.object_registry('name',receptacle_name)
    if placed_entities and not domains_current and not getattr(rec,'fixed_base',True):
        # A light basket can shift under its contents. Re-probe its actual
        # geometry while ignoring placed contents in ray queries only; those
        # bodies remain in the simulation and in every native collision check.
        return plan_native_contents(backend,scenario,receptacle_name,placed_entities)
    context=backend._findingbench_contents_contexts[receptacle_name]
    placements=[p for p in scenario.placements if p.relation=='inside' and p.receptacle==receptacle_name]
    unplaced=[p for p in placements if p.entity not in placed_entities]
    cached=reusable_native_contents_plan(backend,receptacle_name,context,unplaced,placed_entities)
    if cached is not None:
        print('[expert-layout-reuse] '+json.dumps(dict(receptacle=receptacle_name,
            entities=sorted(cached),source='previous whole-layout candidate',
            checked_against='actual settled AABBs and current joints',physical_validation_required=True)),flush=True)
        return cached
    leaf_names={p.entity for p in unplaced}-{p.receptacle for p in scenario.placements}
    thin={p.entity for p in unplaced if p.entity in leaf_names and float(placement_proposal_dimensions(
        backend._env.scene.object_registry('name',p.entity),p)[2])<.004}
    # Some native fillable volumes start above their real drawer floor.
    # Defer a leaf whenever its centre would fall below every compatible
    # native volume, irrespective of an arbitrary thickness threshold.
    # It may later rest on a measured top of an actually placed object.
    for p in unplaced:
        if p.entity not in leaf_names:continue
        obj=backend._env.scene.object_registry('name',p.entity)
        height=float(placement_proposal_dimensions(obj,p)[2])
        possible=any(support_center_lower_bound(context['bounds'][key][0][2],height,context['support_reports'][key])>=context['membership_bottom'][key]+.00001
            and (not p.link or link==p.link or link.startswith(f'meta__{p.link}_'))
            for key,link in context['native'].items())
        if not possible:thin.add(p.entity)
    deferred=thin if len(thin)<len(unplaced) else set()
    working=(add_native_stacking_supports(backend,context,placed_entities)
             if placed_entities and thin and not deferred else context)
    volumes=working['volumes'];bounds=working['bounds'];native=working['native']
    membership_bottom=working['membership_bottom'];support_reports=working['support_reports'];halfspaces=working['halfspaces']
    items={};fixed={};offsets={}
    for placement in placements:
        if placement.entity in placed_entities or placement.entity in deferred:continue
        obj=backend._env.scene.object_registry('name',placement.entity)
        items[placement.entity]=placement_proposal_dimensions(obj,placement).tolist()
        offsets[placement.entity]=array(obj.scaled_bbox_center_in_base_frame)
        # Thin objects may settle below an asset's inset fillable volume even
        # on a correct shelf. Exclude those item/bin pairs; never float objects
        # or change the native final-membership predicate to force acceptance.
        fixed[placement.entity]={key for key,name in native.items()
            if support_center_lower_bound(bounds[key][0][2],items[placement.entity][2],support_reports[key])>=membership_bottom[key]+.00001
            and (not placement.link or name==placement.link or name.startswith(f'meta__{placement.link}_'))}
    obstacles={}
    wall_boxes=[]
    if context.get('open_top'):
        for link in rec.links.values():
            for mesh in getattr(link,'collision_meshes',{}).values():
                from pxr import UsdPhysics
                approximation=UsdPhysics.MeshCollisionAPI(mesh.prim).GetApproximationAttr().Get()
                # A concave SDF/decomposed shell's AABB is not a solid wall.
                # The native baskets use separate convex wall components.
                if approximation not in ('convexHull','boundingCube'):continue
                low,high=(array(v) for v in mesh.aabb)
                if np.isfinite([*low,*high]).all() and np.all(high>low):wall_boxes.append((low,high))
    for key,(lower,upper) in bounds.items():
        boxes=[]
        for low,high in wall_boxes:
            # Near-floor skin is handled by the bounded native drop starts;
            # the collider remains present in every physical check.
            if high[2]<=lower[2]+.012:continue
            if np.all(high[:2]>lower[:2]) and np.all(low[:2]<upper[:2]):
                boxes.append([*(low-lower),*(high-lower)])
        for entity in placed_entities:
            if entity==support_reports[key].get('support_entity'):continue
            obj=backend._env.scene.object_registry('name',entity)
            low,high=(array(v)-lower for v in current_collision_aabb(obj))
            if np.all(high[:2]>0) and np.all(low[:2]<upper[:2]-lower[:2]):
                boxes.append([*low,*high])
        if boxes:obstacles[key]=boxes
    margin=.01 if context.get('open_top') else .015
    has_stacking=any('#on:' in key for key in volumes)
    print('[expert-layout-problem] '+json.dumps(dict(receptacle=receptacle_name,items=items,volumes=volumes,
        fixed={n:sorted(v) for n,v in fixed.items()},center_halfspaces=halfspaces,obstacles=obstacles,
        grid_m=.001,margin_m=margin,time_limit_s=2.,max_candidates=4000000,
        lateral_inset=False if context.get('overlap_groups') else not has_stacking,overlap_groups=context.get('overlap_groups'))),flush=True)
    solved=solve_authoring_layout(items,volumes,fixed,margin_m=margin,time_limit_s=2.,
                        max_candidates=4000000,center_halfspaces=halfspaces,obstacles=obstacles,
                        lateral_inset=False if context.get('overlap_groups') else not has_stacking,
                        overlap_groups=context.get('overlap_groups'))
    # With settled obstacles, retain the exact collision-free CSP positions.
    if not placed_entities and not context.get('overlap_groups'):
        solved=refine_layout_clearance(items,volumes,solved,halfspaces,margin_m=margin,lateral_inset=True)
    print('[expert-layout-plan] '+json.dumps(dict(receptacle=receptacle_name,owned_convex_wall_boxes=len(wall_boxes),**solved)),flush=True)
    if solved['status']!='feasible_model':
        raise ValueError('EXPERT_LAYOUT_UNRESOLVED: '+solved['status']+' '+solved.get('reason',''))
    plan={}
    for name,pose in solved['poses'].items():
        lo,_=bounds[pose['bin']]
        xy=np.asarray(pose['center_xy_m'])+lo[:2]
        center=[float(xy[0]),float(xy[1]),float(lo[2]+items[name][2]/2)]
        # gpu_volume_source retains the bbox-centre setter from
        # oriented_volume_source. Passing a root pose here applies the native
        # asset offset twice (and shifts eccentric meshes into nearby bodies).
        placement=next(p for p in placements if p.entity==name)
        hint=getattr(placement,'orientation_hint_xyzw',None)
        root_pose=bbox_center_root_pose(center,pose['yaw_degrees'],offsets[name],hint)
        pose=dict(pose,oriented_dimensions_m=items[name],orientation_hint_xyzw=hint)
        candidate=[*center,*root_pose[3:]]
        candidates=[candidate]
        if context.get('open_top'):
            # Sparse support rays can pass through a basket grid. Offer two
            # bounded drop starts above that measured plane; the unchanged
            # native sampler decides actual collision/support/settling.
            for lift in (.02,.04):
                local=np.asarray([*pose['center_xy_m'],items[name][2]/2+lift])
                planes=np.asarray(halfspaces[pose['bin']])
                inset=(np.abs(planes[:,:2])@(np.asarray(pose['footprint_m'])/2))*(np.linalg.norm(planes[:,:2],axis=1)>.5)
                if np.all(planes[:,:3]@local+planes[:,3]+inset<-.00001):
                    elevated=list(candidate);elevated[2]+=lift;candidates.append(elevated)
        else:
            candidates=native_drop_candidates(candidate,lo,halfspaces[pose['bin']])
        plan[name]=dict(native_link=native[pose['bin']],candidate=candidate,solver_pose=pose,
                        candidates=candidates,
                        bbox_center_world=center,scaled_root_offset=offsets[name].tolist(),
                        predicted_root_pose=root_pose,candidate_frame='world_bbox_center',
                        measured_support=support_reports[pose['bin']],
                        proposal_source='GPU_joint_layout_in_live_opened_component')
    remember_native_contents_plan(backend,receptacle_name,context,plan)
    return plan


def native_authoring_open_targets(container,fraction):
    """Native, direction-aware open poses away from joint end stops.

    This only selects an initialization pose. It changes no joint limit,
    collision shape, drive, openness threshold or execution skill.
    """
    import numpy as np
    from omnigibson.object_states.open_state import _get_relevant_joints,_compute_joint_threshold
    if not np.isfinite(fraction) or not .05<fraction<1.:raise ValueError('INVALID_AUTHORING_OPEN_FRACTION')
    both_sides,joints,directions=_get_relevant_joints(container)
    if both_sides or not joints:raise ValueError('UNSUPPORTED_AUTHORING_OPEN_METADATA')
    targets=[]
    for joint,direction in zip(joints,directions):
        threshold,opened,closed=(float(v) for v in _compute_joint_threshold(joint,direction))
        target=closed+fraction*(opened-closed)
        if not np.isfinite([threshold,opened,closed,target]).all() or (target-threshold)*(opened-threshold)<=0:
            raise ValueError('AUTHORING_POSE_DOES_NOT_SATISFY_NATIVE_OPEN')
        targets.append((joint,target))
    return targets


def trace_native_articulation_failure(backend,container):
    """Read-only evidence at the actual rejection frame, not pre-open contacts."""
    import json,numpy as np
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float).tolist()
    result=dict(entity=container.name,scope='private native rejection diagnostics; no acceptance change')
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI
        pairs=RigidContactAPI.get_contact_pairs(scene_idx=container.scene.idx,query_set=[container],with_set=None,current_only=True)
        result['contacts']=[list(pair) for pair in sorted(pairs)]
        paths={path for pair in pairs for path in pair};bodies={}
        for obj in backend._env.scene.objects:
            for link in obj.links.values():
                if link.prim_path not in paths:continue
                pos,ori=link.get_position_orientation()
                bodies[link.prim_path]=dict(entity=obj.name,category=obj.category,
                    position=array(pos),orientation=array(ori),aabb=[array(v) for v in link.aabb])
        result['bodies']=bodies
        view=container._articulation_view._physics_view
        result['direct_physx_positions']=array(view.get_dof_positions())
        result['direct_physx_velocities']=array(view.get_dof_velocities())
        result['entity_positions']=array(container.get_joint_positions())
        result['entity_velocities']=array(container.get_joint_velocities())
    except Exception as exc:result['diagnostic_error']=type(exc).__name__+': '+str(exc)
    print('[native-articulation-failure] '+json.dumps(result),flush=True)
    return result


def observe_articulation_motion(backend,container,steps=240):
    """Read every native substep without changing the original failed verdict."""
    import numpy as np
    sim=backend._sim;dt=float(sim.get_physics_dt())
    view=sim.physics_sim_view.create_articulation_view(container.prim_path)
    if view.count!=1:raise ValueError('NATIVE_ARTICULATION_VIEW_IDENTITY_MISMATCH')
    ranges=np.asarray([max(1e-6,abs(float(j.upper_limit)-float(j.lower_limit))) for j in container.joints.values()])
    def array(v):return v.detach().cpu().numpy().copy() if hasattr(v,'detach') else np.asarray(v).copy()
    frames=[]
    for index in range(steps+1):
        frames.append(dict(step=index,positions=array(view.get_dof_positions())[0].tolist(),
                           reported_velocities=array(view.get_dof_velocities())[0].tolist()))
        if index<steps:sim.step_physics()
    positions=np.asarray([f['positions'] for f in frames]);velocities=np.asarray([f['reported_velocities'] for f in frames])
    if positions.shape[1:]!=ranges.shape or not dt>0 or not np.isfinite([dt,*ranges]).all():
        raise ValueError('NATIVE_ARTICULATION_MOTION_TOPOLOGY_UNSUPPORTED')
    return dict(acceptance='diagnostic_only_original_failure_retained',entity=container.name,physics_dt=dt,
        physics_steps=steps,max_normalized_excursion=float((abs(positions-positions[0])/ranges).max()),
        max_normalized_measured_speed=float((abs(np.diff(positions,axis=0))/dt/ranges).max()),
        max_normalized_reported_speed=float((abs(velocities)/ranges).max()),
        all_finite=bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),frames=frames)


def articulation_motion_converged(report):
    """Apply the original .005 drift/.05 speed limits to measured motion."""
    return bool(report.get('all_finite') and report['max_normalized_excursion']<=.005
                and report['max_normalized_measured_speed']<=.05)


def settle_authoring_container(backend, container, *, max_windows=24, steps=15):
    """Wait for placed furniture's real joint motion to settle; never lock it."""
    import numpy as np,json
    if not getattr(container,'n_joints',0):return dict(extra_steps=0,reason='No articulated joints')
    from omnigibson.object_states import Open
    if Open in container.states and not bool(container.states[Open].get_value()):
        if not backend._set_open_carrying(container,True):raise ValueError('EXPERT_CONTAINER_OPEN_FAILED')
    ranges=np.asarray([max(1e-6,abs(float(j.upper_limit)-float(j.lower_limit))) for j in container.joints.values()])
    if not np.isfinite(ranges).all():raise ValueError('INVALID_CONTAINER_JOINT_LIMITS')
    previous=np.asarray([float(v) for v in container.get_joint_positions()]);quiet=0
    for window in range(max_windows):
        backend.settle(steps)
        current=np.asarray([float(v) for v in container.get_joint_positions()])
        speed=np.asarray([float(v) for v in container.get_joint_velocities()])
        drift=np.abs(current-previous)/ranges;relative_speed=np.abs(speed)/ranges
        stable=bool((drift<=.005).all() and (relative_speed<=.05).all())
        quiet=quiet+1 if stable else 0
        report=dict(entity=container.name,extra_steps=(window+1)*steps,normalized_joint_drift=drift.tolist(),
                    normalized_joint_speed=relative_speed.tolist(),quiet_windows=quiet)
        print('[container-settling] '+json.dumps(report),flush=True)
        if quiet>=3:
            if Open in container.states and not bool(container.states[Open].get_value()):raise ValueError('EXPERT_CONTAINER_CLOSED_DURING_SETTLING')
            return report
        previous=current
    trace_native_articulation_failure(backend,container)
    raise ValueError('EXPERT_CONTAINER_JOINTS_NOT_SETTLED')
