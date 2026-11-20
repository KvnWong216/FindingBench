"""Exact native-region selection; unit fixtures never count as scene data."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


def test_authored_side_orientation_rotates_real_dimensions_and_root_offset():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from rummagebench.sim.omnigibson.inside_volume import placement_proposal_dimensions,bbox_center_root_pose
    obj=SimpleNamespace(native_bbox=np.array([.029,.067,.228]),scale=np.ones(3))
    hint=Rotation.from_euler('y',90,degrees=True).as_quat()
    p=SimpleNamespace(orientation_hint_xyzw=hint)
    assert np.allclose(placement_proposal_dimensions(obj,p),[.228,.067,.029])
    center=np.array([1.,2.,3.]);offset=np.array([.01,-.02,.03])
    result=bbox_center_root_pose(center,90,offset,hint)
    rotation=Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat(hint)
    assert np.allclose(result[:3],center+rotation.apply(offset))
    assert np.allclose(Rotation.from_quat(result[3:]).as_matrix(),rotation.as_matrix())
    assert np.allclose(obj.native_bbox,[.029,.067,.228])


def test_cached_side_pose_uses_its_rotated_height_for_actual_obstacles(monkeypatch):
    import numpy as np
    import rummagebench.sim.omnigibson.inside_volume as volume
    objects=dict(tool=SimpleNamespace(native_bbox=np.array([.029,.067,.228]),scale=np.ones(3)),
        cabinet=SimpleNamespace(n_joints=0),upper=SimpleNamespace())
    backend=SimpleNamespace(_env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda _,name:objects[name])))
    pose=dict(bbox_center_world=[.5,.5,.015],solver_pose=dict(footprint_m=[.228,.067],oriented_dimensions_m=[.228,.067,.029]))
    context=dict(pending_plan={'tool':pose},pending_plan_joints=[])
    obstacle=[np.array([.4,.4,.06]),np.array([.6,.6,.08])]
    monkeypatch.setattr(volume,'current_collision_aabb',lambda _:obstacle)
    pending=[SimpleNamespace(entity='tool')]
    assert volume.reusable_native_contents_plan(backend,'cabinet',context,pending,{'upper'})
    obstacle[0][2]=.02
    assert volume.reusable_native_contents_plan(backend,'cabinet',context,pending,{'upper'}) is None


def test_contained_support_remains_measurable_when_placed_child_occludes_it(monkeypatch):
    import numpy as np,sys
    from rummagebench.sim.omnigibson.inside_volume import add_native_stacking_supports
    calls=[]
    def raytest(start,end,ignore_bodies=()):
        calls.append(ignore_bodies)
        body='/book/body' if '/pin/body' in ignore_bodies else '/pin/body'
        return dict(hit=True,rigidBody=body,position=[start[0],start[1],.2],normal=[0.,0.,1.])
    monkeypatch.setitem(sys.modules,'omnigibson.utils.sampling_utils',SimpleNamespace(raytest=raytest))
    objects={
        'book':SimpleNamespace(aabb=(np.array([.2,.2,.1]),np.array([.8,.8,.2])),links={'body':SimpleNamespace(prim_path='/book/body')}),
        'pin':SimpleNamespace(aabb=(np.array([.4,.4,.2]),np.array([.41,.45,.203])),links={'body':SimpleNamespace(prim_path='/pin/body')})}
    backend=SimpleNamespace(_env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda _,name:objects[name])))
    equations=np.array([[-1,0,0,0],[1,0,0,-1],[0,-1,0,0],[0,1,0,-1],[0,0,-1,0],[0,0,1,-1]],float)
    context=dict(volumes={'bin':[1.,1.,1.]},bounds={'bin':(np.zeros(3),np.ones(3))},native={'bin':'fillable'},
        membership_bottom={'bin':0.},support_reports={'bin':{}},halfspaces={'bin':[]},
        components={'bin':dict(equations=equations,lo=np.zeros(3),hi=np.ones(3),link='fillable')})
    result=add_native_stacking_supports(backend,context,{'book','pin'})
    assert 'bin#on:book' in result['volumes'] and 'bin#on:book' not in context['volumes']
    assert len(calls)==9 and all(ignored==['/pin/body'] for ignored in calls)
    assert result['support_reports']['bin#on:book']['support_entity']=='book'
    assert np.array_equal(objects['pin'].aabb[0],[.4,.4,.2])


def test_authoring_robot_isolation_restores_pose_and_controller_even_on_failure(monkeypatch):
    import numpy as np,pytest
    from rummagebench.sim.omnigibson.inside_volume import isolate_authoring_robot
    monkeypatch.setenv('YGBENCH_EXPERT_LAYOUT','1')
    for fail in [False,True]:
        initial=dict(position=[1.,2.,.4],orientation=[0.,0.,0.,1.],controller={'target':[.3,.2]})
        state=deepcopy(initial);calls=[]
        def load(saved,serialized=False):state.clear();state.update(deepcopy(saved));calls.append('restore')
        def move(position,orientation):state.update(position=list(position),orientation=list(orientation));calls.append('park')
        robot=SimpleNamespace(_ag_obj_in_hand={},dump_state=lambda serialized=False:deepcopy(state),load_state=load,
            get_position_orientation=lambda:(state['position'],state['orientation']),set_position_orientation=move)
        def author(backend):
            assert state['position'][:2]==[1000.,1000.]
            state['controller']['target']=[99.]
            if fail:raise RuntimeError('native placement rejected')
            return {'placements':[]}
        run=isolate_authoring_robot(author)
        if fail:
            with pytest.raises(RuntimeError,match='native placement rejected'):run(SimpleNamespace(_robot=robot))
        else:assert run(SimpleNamespace(_robot=robot))['robot_authoring_isolation']['spawn_changed'] is False
        assert state==initial and calls==['park','restore']


def test_gpu_component_proposals_avoid_partition_gap_and_keep_budget():
    import pytest,torch
    from rummagebench.sim.omnigibson.inside_volume import gpu_proposal_bank
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    boxes=torch.tensor([[[0,0,0],[.2,1,1]],[[.8,0,0],[1,1,1]]],device='cuda')
    bank=gpu_proposal_bank([.05]*3,[0]*3,[1]*3,[[0,0,0,1]],[1.],100,
        components=boxes,device='cuda',generator=torch.Generator(device='cuda').manual_seed(8))
    # Native proposals in this fixture are yaw-only. Test the entire rotated
    # box, including the sampler's +1 cm lift, not just candidate centers.
    z,w=bank[:50,5],bank[:50,6]
    width=.05*((1-2*z*z).abs()+(2*z*w).abs())
    extent=torch.stack([width,width,torch.full_like(width,.05)],dim=-1)
    centers=bank[:50,:3]+torch.tensor([0,0,.01],device='cuda')
    contained=((centers[:,None]-extent[:,None]/2>=boxes[None,:,0]-1e-6)&
               (centers[:,None]+extent[:,None]/2<=boxes[None,:,1]+1e-6)).all(-1)
    assert contained.any(-1).all() and contained.any(0).all()
    assert bank.shape==(100,7) and torch.isfinite(bank).all()
    # Full-bound fallback half is retained, including points in the gap.
    assert ((bank[50:,0]>.2)&(bank[50:,0]<.8)).any()


def test_component_bounds_do_not_reject_oversize_objects():
    import torch,pytest
    from rummagebench.sim.omnigibson.inside_volume import gpu_proposal_bank
    kwargs=dict(size=[2]*3,low=[0]*3,high=[1]*3,rotations=[[0,0,0,1]],probabilities=[1.],count=10,device='cpu')
    bank=gpu_proposal_bank(**kwargs,components=[[[0]*3,[.1]*3]])
    assert bank.shape==(10,7) and torch.isfinite(bank).all()
    with pytest.raises(ValueError,match='INVALID_GPU_VOLUME_COMPONENTS'):
        gpu_proposal_bank(**kwargs,components=[[[0]*3,[-1]*3]])


def test_sampling_progress_keeps_counters_and_explains_entry_timing(capsys):
    import json
    from rummagebench.sim.omnigibson.inside_volume import SamplingAudit
    audit=SamplingAudit('board','cabinet','volume')
    for attempt in range(1,42):
        audit['attempts']=attempt
        audit['collision']=audit.get('collision',0)+1
    messages=[json.loads(line.split(' ',1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [m['attempt_started'] for m in messages]==[1,20,40]
    assert [m['counts'].get('collision',0) for m in messages]==[0,19,39]
    assert dict(audit)=={'attempts':41,'collision':41}
    assert all(m['object']=='board' and m['elapsed_s']>=0 for m in messages)


def test_orientation_coverage_preserves_metadata_and_default():
    from rummagebench.sim.omnigibson.inside_volume import proposal_orientation_distribution as distribution
    assert distribution({}) == ([[0,0,0,1]], [1.])
    metadata={'side':{'rotation':[0,1,0,0],'prob':1.}}
    assert distribution(metadata,augment_empty=True) == distribution(metadata)


def test_axis_orientation_experiment_covers_narrow_cabinet_on_gpu():
    import pytest
    import torch
    from rummagebench.sim.omnigibson.inside_volume import gpu_proposal_bank,proposal_orientation_distribution
    if not torch.cuda.is_available():pytest.skip('CUDA required for actual proposal bank')
    # F331 chopping-board dimensions and the observed narrow storage volume.
    size=torch.tensor([.290024,.400032,.030223],device='cuda')
    extent=torch.tensor([.238533,.620769,.584417],device='cuda')
    fits=[]
    for augment in (False,True):
        rotations,weights=proposal_orientation_distribution({},augment_empty=augment)
        bank=gpu_proposal_bank(size,[0,0,0],extent,rotations,weights,100,
            device='cuda',generator=torch.Generator(device='cuda').manual_seed(0))
        x,y,z,w=bank[:,3:].unbind(-1)
        matrix=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
            2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
            2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
        fits.append(int(((matrix.abs()@size)<extent).all(-1).sum()))
        assert bank.shape==(100,7)
        assert torch.isfinite(bank).all()
    assert fits[0]==0 and fits[1]>0
    # This verifies proposal coverage only, not native containment or stability.


def test_early_stability_filter_retains_all_original_sampler_checks():
    from rummagebench.sim.omnigibson.inside_volume import oriented_volume_source,settled_candidate_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/object_states/inside.py'))
    text=path.read_text();node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    baseline=ast.parse(oriented_volume_source(ast.get_source_segment(text,node)))
    changed=ast.parse(settled_candidate_source(ast.unparse(baseline)))
    loop=next(n for n in ast.walk(changed) if isinstance(n,ast.For) and ast.unparse(n.target)=='attempt_idx')
    extra=[n for n in loop.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ('_findingbench_linear','_findingbench_angular')
        or isinstance(n,ast.If) and 'candidate_velocity_above_final_threshold' in ast.unparse(n)]
    assert len(extra)==3
    rejection=extra[-1]
    assert 'og.sim.load_state(state, serialized=False)' in ast.unparse(rejection)
    assert isinstance(rejection.body[-1],ast.Continue)
    loop.body=[n for n in loop.body if n not in extra]
    assert ast.dump(changed)==ast.dump(baseline)


def test_batched_native_parallel_rays_preserve_grid_rng_and_geometry():
    import torch
    from rummagebench.sim.omnigibson.inside_volume import vectorized_parallel_rays_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/utils/sampling_utils.py'))
    text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='get_parallel_rays')
    source=ast.get_source_segment(text,node)
    original={'th':torch};changed={'th':torch}
    exec(source,original);exec(vectorized_parallel_rays_source(source),changed)
    for size in (.02,.1,.5):
        for seed in range(5):
            args=(torch.tensor([3.1,-2.4,1.2]),torch.tensor([3.3,-2.1,-1.]),torch.tensor([size,size*.8]),.025)
            torch.manual_seed(seed);a=original['get_parallel_rays'](*args);state=torch.random.get_rng_state()
            torch.manual_seed(seed);b=changed['get_parallel_rays'](*args)
            assert torch.equal(state,torch.random.get_rng_state())
            assert torch.equal(a[2],b[2])
            for expected,actual in zip(a[:2],b[:2]):
                assert isinstance(actual,list) and len(expected)==len(actual)
                torch.testing.assert_close(torch.stack(expected),torch.stack(actual),rtol=0,atol=5e-7)


def test_dense_authoring_does_not_bypass_rejected_native_sampler(monkeypatch):
    import sys
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    class Inside: pass
    class OnTop: pass
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', SimpleNamespace(Inside=Inside, OnTop=OnTop))
    state=SimpleNamespace(set_value=lambda *a,**kw:False)
    obj=SimpleNamespace(name='item',states={Inside:state})
    rec=SimpleNamespace(name='basket')
    backend=OmniGibsonBackend()
    backend._scenario=SimpleNamespace(placements=[SimpleNamespace(receptacle='basket') for _ in range(2)])
    from rummagebench.sim.omnigibson import inside_volume
    monkeypatch.setattr(inside_volume,'fitting_volume_names',lambda *a:(['native_volume'],{}))
    monkeypatch.setattr(inside_volume,'sample_selected_volume',lambda *a:(False,{}))
    # No pose API is supplied: a center-drop fallback would raise here.
    assert backend._place(obj,rec,'inside') is False


@__import__('pytest').mark.parametrize('enabled,initial,raises',[
    (False,True,False),(True,True,False),(True,False,False),
    (True,True,True),(True,False,True)])
def test_deferred_authoring_restores_rendering_and_propagates_failure(monkeypatch,enabled,initial,raises):
    import pytest
    from rummagebench.sim.omnigibson.inside_volume import defer_authoring_rendering
    monkeypatch.setenv('YGBENCH_DEFER_AUTHORING_RENDER','1' if enabled else '0')
    calls=[]
    sim=SimpleNamespace(_render_on_step=initial,render=lambda:calls.append('render'))
    backend=SimpleNamespace(_sim=sim)
    @defer_authoring_rendering
    def placement(self):
        calls.append(('placement',self._sim._render_on_step))
        if raises:raise RuntimeError('native placement rejected')
        return {'verified':True}
    if raises:
        with pytest.raises(RuntimeError,match='native placement rejected'):placement(backend)
    else:assert placement(backend)=={'verified':True}
    assert sim._render_on_step is initial
    assert calls[0]==('placement',False if enabled else initial)
    assert calls.count('render')==(2 if enabled and initial and not raises else 0)


def test_unspecified_cupboard_skips_shallow_drawer(monkeypatch):
    import sys
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import fitting_volume_names
    monkeypatch.setitem(sys.modules,'omnigibson.object_states.contains',
                        SimpleNamespace(m=SimpleNamespace(CONTAINER_META_LINK_TYPES={'fillable'})))
    def volume(size):
        return SimpleNamespace(is_meta_link=True,meta_link_type='fillable',visual_aabb=(np.zeros(3),np.array(size)))
    obj=SimpleNamespace(native_bbox=np.array([.075,.075,.20]),scale=np.ones(3))
    rec=SimpleNamespace(links={'shallow_first':volume([.35,.30,.044]),'cabinet_second':volume([.44,1.01,.692])})
    names,audit=fitting_volume_names(obj,rec)
    assert names==['cabinet_second']
    assert audit['volumes'][0]['size_fit'] is False
    obj.native_bbox=np.array([2.,2.,2.])
    assert fitting_volume_names(obj,rec)[0]==[]

import pytest

from rummagebench.sim.omnigibson.inside_volume import selected_volume_source


@pytest.mark.local_assets
def test_actual_sdk_keeps_every_inside_check_except_region_selection():
    roots=(Path(__file__).resolve().parents[3],Path(__file__).resolve().parents[4])
    candidates=[p for root in roots for p in root.glob('BEHAVIOR-1K*/OmniGibson/omnigibson/object_states/inside.py')]
    if not candidates:pytest.skip('installed native Inside source unavailable')
    path=candidates[0]
    source=path.read_text();tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    original=ast.parse(ast.get_source_segment(source,method))
    adapted=ast.parse(selected_volume_source(ast.get_source_segment(source,method)))
    function=adapted.body[0]
    assert function.args.kwonlyargs[-1].arg=='_findingbench_volume_link'
    assert function.args.kw_defaults[-1] is None
    function.args.kwonlyargs.pop();function.args.kw_defaults.pop()
    loop=next(n for n in ast.walk(function) if isinstance(n,ast.For)
              and ast.unparse(n.iter)=='other.links.values()')
    condition=loop.body[0].test
    assert ast.unparse(condition.values[-1])=='link.name == _findingbench_volume_link'
    loop.body[0].test=condition.values[0]
    assert ast.dump(original)==ast.dump(adapted)


def test_selects_second_real_link_without_reordering_and_retains_failure():
    source='''
def _set_value(self,other,new_value,reset_before_sampling=False,use_trav_map=False):
    container_link=None
    for link in other.links.values():
        if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
            container_link=link
            break
    assert container_link is not None
    if not self.independent_physics_check:
        return False
    return container_link.name
'''
    namespace={'macros':SimpleNamespace(object_states=SimpleNamespace(contains=SimpleNamespace(CONTAINER_META_LINK_TYPES={'fillable'})))}
    exec(compile(selected_volume_source(source),'unit_only','exec'),namespace)
    links={name:SimpleNamespace(name=name,is_meta_link=True,meta_link_type='fillable') for name in ('first','second')}
    before=list(links)
    state=SimpleNamespace(independent_physics_check=True);other=SimpleNamespace(links=links)
    sampler=namespace['_set_value']
    assert sampler(state,other,True,_findingbench_volume_link='second')=='second'
    assert list(links)==before
    state.independent_physics_check=False
    assert sampler(state,other,True,_findingbench_volume_link='second') is False


def test_wrong_final_volume_rejects_and_restores_even_after_sampler_success(monkeypatch):
    import sys
    from rummagebench.sim.omnigibson import inside_volume as module
    events=[]
    class UnitInside:pass
    state=SimpleNamespace(_initialized=True,clear_cache=lambda:events.append('clear'))
    obj=SimpleNamespace(states={UnitInside:state},state_updated=lambda:events.append('updated'))
    rec=SimpleNamespace(links={'chosen':SimpleNamespace(is_meta_link=True,name='unit_bank:chosen')})
    sim=SimpleNamespace(dump_state=lambda **kw:'unit_saved',
                        load_state=lambda state,**kw:events.append(('restore',state)))
    monkeypatch.setitem(sys.modules,'omnigibson',SimpleNamespace(sim=sim))
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',SimpleNamespace(Inside=UnitInside))
    # Unit sampler stand-in deliberately claims success; final membership wins.
    type_state=type('UnitState',(),{'_set_value':lambda *a,**kw:True})
    state=type_state();state._initialized=True;state.clear_cache=lambda:events.append('clear')
    obj.states[UnitInside]=state
    def sampler(*a,**kwargs):
        assert kwargs['_findingbench_volume_link']=='unit_bank:chosen'
        return True
    monkeypatch.setattr(module,'_sampler',lambda method,**options:(sampler,'unit_hash'))
    monkeypatch.setattr(module,'selected_volume_contains',lambda *a:False)
    ok,_=module.sample_selected_volume(obj,rec,'chosen')
    assert ok is False and events==['clear','updated',('restore','unit_saved')]


@pytest.mark.parametrize('sampled,member,accepted',[(True,True,True),(False,True,False),(True,False,False)])
def test_existing_scene_probe_checks_selected_volume_and_sampler_success(monkeypatch,sampled,member,accepted):
    from rummagebench.authoring.tasks.probe import SceneProber
    from rummagebench.sim.omnigibson import probe_support as ps
    # Only unit stand-ins; the runtime predicate delegates to backend._in_link.
    backend=SimpleNamespace(settle=lambda steps:None,_collision_body_cache=None)
    probe=SimpleNamespace(backend=backend,probe='unit_target',cfg=SimpleNamespace(settle_steps=1))
    slot=SimpleNamespace(parent_entity='unit_bank',relation='inside',link='meta__drawer_fillable')
    monkeypatch.setattr(ps,'place_entity',lambda *a:sampled)
    monkeypatch.setattr(ps,'relation_holds',lambda *a:True)
    monkeypatch.setattr(ps,'containing_link',lambda *a:'physical_drawer')
    checked=[]
    monkeypatch.setattr(ps,'link_membership',lambda *a:checked.append(a[-1]) or member)
    result=SceneProber._place_probe(probe,slot)
    assert result['placed'] is accepted and checked==[slot.link]


def test_sampling_diagnostics_do_not_change_native_control_flow():
    from rummagebench.sim.omnigibson.inside_volume import diagnostic_volume_source
    source="""
def _set_value(self,other,new_value,reset_before_sampling=False,use_trav_map=False):
    for link in other.links.values():
        if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
            container_link=link
            break
    for attempt_idx in range(3):
        if attempt_idx < 2:
            continue
        return True
    return False
"""
    original=ast.parse(selected_volume_source(source))
    instrumented=ast.parse(diagnostic_volume_source(source))
    class StripCounters(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0]).startswith('_findingbench_audit['):return None
            return node
    instrumented=StripCounters().visit(instrumented)
    instrumented.body[0].args.kwonlyargs.pop();instrumented.body[0].args.kw_defaults.pop()
    assert ast.dump(original)==ast.dump(instrumented)


def test_centered_sampler_changes_only_pose_coordinate_setter():
    from rummagebench.sim.omnigibson.inside_volume import centered_volume_source,diagnostic_volume_source
    roots=(Path(__file__).resolve().parents[3],Path(__file__).resolve().parents[4])
    candidates=[p for root in roots for p in root.glob('BEHAVIOR-1K*/OmniGibson/omnigibson/object_states/inside.py')]
    if not candidates:pytest.skip('installed native Inside source unavailable')
    source=candidates[0].read_text()
    method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    source=ast.get_source_segment(source,method)
    original=ast.parse(diagnostic_volume_source(source));adapted=ast.parse(centered_volume_source(source))
    setters=[n for n in ast.walk(adapted) if isinstance(n,ast.Call)
        and ast.unparse(n.func)=='self.obj.set_bbox_center_position_orientation']
    assert len(setters)==1
    setters[0].func.attr='set_position_orientation'
    assert ast.dump(original)==ast.dump(adapted)


def test_authoring_failure_restores_gravity_for_staged_objects(monkeypatch):
    import sys
    from rummagebench.sim.omnigibson import backend as module
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',
        SimpleNamespace(Inside=object(),OnTop=object(),Open=object()))
    backend=module.OmniGibsonBackend();backend._env=SimpleNamespace(scene=object())
    backend._sim=object();backend._staged_authoring_names={'pending'}
    calls=[]
    obj=SimpleNamespace(enable_gravity=lambda:calls.append('gravity_restored'))
    monkeypatch.setattr(module,'resolve_object',lambda scene,name:obj)
    def fail(*args):raise RuntimeError('sim snapshot failure')
    monkeypatch.setattr(module,'dump_state',fail)
    with pytest.raises(RuntimeError,match='sim snapshot failure'):
        backend._apply_placements(SimpleNamespace(),attempts=1)
    assert calls==['gravity_restored']


def test_dependency_order_keeps_containers_before_larger_children():
    from rummagebench.sim.omnigibson.inside_volume import placement_order
    rows=[SimpleNamespace(entity='child',receptacle='box'),SimpleNamespace(entity='small',receptacle='cabinet'),SimpleNamespace(entity='box',receptacle='cabinet')]
    assert placement_order(rows,dict(child=10,small=1,box=2))==[2,1,0]
    rows[2].receptacle='child'
    with pytest.raises(ValueError,match='CYCLIC'):placement_order(rows,dict(child=10,small=1,box=2))


def test_gpu_transform_preserves_all_native_checks():
    from rummagebench.sim.omnigibson.inside_volume import oriented_volume_source,gpu_volume_source
    root=Path('/home/ubuntu/ygwang')
    path=next(root.glob('BEHAVIOR-1K*/OmniGibson/omnigibson/object_states/inside.py'))
    source=path.read_text();method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    source=ast.get_source_segment(source,method)
    tails=[]
    for transform in (oriented_volume_source,gpu_volume_source):
        tree=ast.parse(transform(source));loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='attempt_idx')
        index=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and 'container_link.check_points_in_volume' in ast.unparse(n.test))
        tails.append([ast.dump(n) for n in loop.body[index:]])
    assert tails[0]==tails[1]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batched_proposal_geometry(device):
    import torch
    from rummagebench.sim.omnigibson.inside_volume import gpu_proposal_bank
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    bank=gpu_proposal_bank([.1,.1,.1],[0,0,0],[1,1,1],[[0,0,0,1]],[1.],800,device=device)
    assert bank.shape==(800,7)
    assert torch.isfinite(bank).all()
    assert torch.allclose(torch.linalg.vector_norm(bank[:,3:],dim=-1),torch.ones(800,device=device),atol=1e-5)
    assert (bank[:400,:3]>=.05-1e-6).all() and (bank[:400,:3]<=.95+1e-6).all()
    assert (bank[400:,:3]>=0).all() and (bank[400:,:3]<=1).all()


def test_gpu_ordering_keeps_all_candidates_and_attempt_halves():
    import torch
    from rummagebench.sim.omnigibson.inside_volume import gpu_proposal_bank
    def sample(planes):
        return gpu_proposal_bank([.1]*3,[0]*3,[1]*3,[[0,0,0,1]],[1.],80,planes=planes,device='cpu',generator=torch.Generator().manual_seed(5))
    raw=sample([]);ordered=sample([[[1,0,0,-.5]]])
    for a,b in ((raw[:40],ordered[:40]),(raw[40:],ordered[40:])):
        assert sorted(map(tuple,a.tolist()))==sorted(map(tuple,b.tolist()))
        flags=(b[:,0]<=.5).int();assert bool((flags[:-1]>=flags[1:]).all())


def test_vector_fan_matches_native_for_mixed_polygons():
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import triangulate_faces
    counts=np.array([0,1,2,3,4,7,3]);indices=np.arange(counts.sum())[::-1]
    expected=[];start=0
    for count in counts:
        for j in range(count-2):expected.append([indices[start],indices[start+j+1],indices[start+j+2]])
        start+=count
    assert np.array_equal(triangulate_faces(counts,indices),expected)
    assert triangulate_faces([],[]).shape==(0,3)
    with pytest.raises(ValueError):triangulate_faces([3],[0,1])


def test_real_sdk_mesh_conversion_is_numerically_identical():
    import numpy as np
    import torch
    import trimesh
    from rummagebench.sim.omnigibson.inside_volume import triangulate_faces,vectorized_mesh_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/utils/usd_utils.py'))
    source=path.read_text();node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='mesh_prim_mesh_to_trimesh_mesh')
    source=ast.get_source_segment(source,node)
    attrs={'points':[[0.,0,0],[1.,0,0],[1.,1,0],[0.,1,0],[0.,0,1]],'faceVertexCounts':[4,3,3,3,3],
           'faceVertexIndices':[0,3,2,1,0,1,4,1,2,4,2,3,4,3,0,4]}
    mesh=SimpleNamespace(GetPrimTypeInfo=lambda:SimpleNamespace(GetTypeName=lambda:'Mesh'),GetAttribute=lambda name:SimpleNamespace(Get=lambda:attrs[name]))
    results=[]
    for code in (source,vectorized_mesh_source(source)):
        namespace={'th':torch,'trimesh':trimesh,'vtarray_to_torch':lambda value,dtype=torch.float32:torch.tensor(value,dtype=dtype),'_findingbench_triangulate':triangulate_faces}
        exec(code,namespace)
        results.append(namespace['mesh_prim_mesh_to_trimesh_mesh'](mesh,False,False))
    assert np.array_equal(results[0].vertices,results[1].vertices)
    assert np.array_equal(results[0].faces,results[1].faces)
    assert results[0].volume==results[1].volume
    assert np.array_equal(results[0].center_mass,results[1].center_mass)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gpu_occupancy_order_does_not_remove_candidates_or_cross_halves(device):
    import torch
    from rummagebench.sim.omnigibson.inside_volume import gpu_proposal_bank
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    def sample(obstacles):
        return gpu_proposal_bank([.1]*3,[0]*3,[1]*3,[[0,0,0,1]],[1.],80,obstacles=obstacles,device=device,generator=torch.Generator(device=device).manual_seed(17))
    raw=sample([]);ranked=sample([[[0,0,0],[.5,1,1]]])
    for a,b in ((raw[:40],ranked[:40]),(raw[40:],ranked[40:])):
        assert sorted(map(tuple,a.tolist()))==sorted(map(tuple,b.tolist()))
        # Candidates wholly to the right of the obstacle are tried first.
        assert b[0,0]>.58
    with pytest.raises(ValueError):sample([[[1,0,0],[0,1,1]]])


def _cache_test_mesh_convert(mesh):
    face_vertex_counts=[3]
    face_indices=[0,1,2]
    faces = []
    i = 0
    for count in face_vertex_counts:
        for j in range(count-2):
            faces.append([face_indices[i],face_indices[i+j+1],face_indices[i+j+2]])
        i += count
    return faces


def _cache_test_mass(mesh,world_frame=False):
    import torch
    mesh.calls+=1
    return float(sum(mesh.attributes['points'][0])),torch.tensor([1.,2.,3.])


def test_mass_cache_tracks_geometry_and_bypasses_world_coordinates(monkeypatch,tmp_path):
    import sys,types,torch,sqlite3
    from functools import cached_property
    from rummagebench.sim.omnigibson.inside_volume import install_mesh_acceleration
    usd=types.ModuleType('omnigibson.utils.usd_utils')
    usd.mesh_prim_mesh_to_trimesh_mesh=_cache_test_mesh_convert
    usd.get_mesh_volume_and_com=_cache_test_mass
    class Geom:
        points_and_faces=cached_property(_cache_test_mesh_convert)
    og=types.ModuleType('omnigibson');og.__path__=[]
    utils=types.ModuleType('omnigibson.utils');utils.__path__=[];utils.usd_utils=usd
    prims=types.ModuleType('omnigibson.prims');prims.__path__=[]
    geom=types.ModuleType('omnigibson.prims.geom_prim');geom.GeomPrim=Geom
    states=types.ModuleType('omnigibson.object_states')
    class Inside:
        def _get_value(self,other):return False
        def get_value(self,other):return self._get_value(other)
    states.Inside=Inside
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',states)
    for name,value in [('omnigibson',og),('omnigibson.utils',utils),('omnigibson.utils.usd_utils',usd),('omnigibson.prims',prims),('omnigibson.prims.geom_prim',geom)]:
        monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setenv('YGBENCH_GEOMETRY_CACHE',str(tmp_path))
    mesh=SimpleNamespace(calls=0,attributes={'points':[[1.,2.,3.]],'faceVertexCounts':[3],'faceVertexIndices':[0,0,0]},GetTypeName=lambda:'Mesh')
    mesh.GetAttribute=lambda key:SimpleNamespace(Get=lambda:mesh.attributes[key])
    install_mesh_acceleration()
    one=usd.get_mesh_volume_and_com(mesh);two=usd.get_mesh_volume_and_com(mesh)
    assert mesh.calls==1 and one[0]==two[0] and torch.equal(one[1],two[1])
    mesh.attributes['points'][0][0]=4.
    assert usd.get_mesh_volume_and_com(mesh)[0]==9. and mesh.calls==2
    usd.get_mesh_volume_and_com(mesh,world_frame=True)
    assert mesh.calls==3
    with sqlite3.connect(tmp_path/'local_mass.sqlite3') as connection:
        connection.execute("UPDATE mass SET payload='broken JSON'")
    assert usd.get_mesh_volume_and_com(mesh)[0]==9. and mesh.calls==4
    assert usd._findingbench_mesh_cache_stats=={'hits':1,'misses':2,'fallbacks':1}


@pytest.mark.parametrize('binding',['none','valid','missing'])
def test_material_read_probe_matches_usd_without_applying_schema(binding):
    from pxr import Usd,UsdShade
    from rummagebench.sim.omnigibson.inside_volume import readonly_material_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/prims/xform_prim.py'))
    source=path.read_text();node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='has_material')
    source=ast.get_source_segment(source,node)
    stage=Usd.Stage.CreateInMemory();prim=stage.DefinePrim('/Object','Xform')
    if binding!='none':
        material=UsdShade.Material.Define(stage,'/Material')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        prim.RemoveAPI(UsdShade.MaterialBindingAPI)
        if binding=='missing':stage.RemovePrim('/Material')
    class Object:
        @property
        def _binding_api(self):return UsdShade.MaterialBindingAPI.Apply(self.prim)
    obj=Object();obj.prim=prim
    namespace={'lazy':SimpleNamespace(pxr=SimpleNamespace(UsdShade=UsdShade),isaacsim=SimpleNamespace(core=SimpleNamespace(utils=SimpleNamespace(prims=SimpleNamespace(is_prim_path_valid=lambda path:bool(stage.GetPrimAtPath(path)))))))}
    exec(readonly_material_source(source),namespace)
    fast=namespace['has_material'](obj)
    assert not prim.HasAPI(UsdShade.MaterialBindingAPI)
    exec(source,namespace)
    original=namespace['has_material'](obj)
    assert fast==original==(binding=='valid')


@pytest.mark.parametrize('quat_type',['quatf','quatd'])
def test_batched_pose_preserves_usd_pose_and_cache_invalidation(monkeypatch,quat_type):
    import sys,types,math,torch,contextlib
    from pxr import Usd,UsdGeom,Gf,Sdf
    from rummagebench.sim.omnigibson.inside_volume import batched_pose_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/prims/xform_prim.py'))
    source=path.read_text();node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='set_position_orientation')
    source=ast.get_source_segment(source,node)
    state=SimpleNamespace(caches_dirty=False)
    module=types.ModuleType('omnigibson.object_states.tensorized_state');module.TensorizedState=state
    monkeypatch.setitem(sys.modules,module.__name__,module)
    results=[]
    for code in [source,batched_pose_source(source)]:
        stage=Usd.Stage.CreateInMemory();prim=stage.DefinePrim('/Object','Xform')
        prim.CreateAttribute('xformOp:translate',Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0))
        prim.CreateAttribute('xformOp:orient',Sdf.ValueTypeNames.Quatf if quat_type=='quatf' else Sdf.ValueTypeNames.Quatd)
        events=[]
        @contextlib.contextmanager
        def edit():
            events.append('enter');yield;events.append('sync')
        sim=SimpleNamespace(editing_usd=edit,fabric_hierarchy=SimpleNamespace(update_world_xforms=lambda:events.append('world')))
        namespace=dict(Literal=__import__('typing').Literal,th=torch,math=math,og=SimpleNamespace(sim=sim),lazy=SimpleNamespace(pxr=SimpleNamespace(Gf=Gf)))
        exec(code,namespace)
        class Obj:
            def set_attribute(self,name,value):
                with edit():self._prim.GetAttribute(name).Set(value)
        obj=Obj();obj.prim=obj._prim=prim;obj.prim_path='/Object';state.caches_dirty=False
        namespace['set_position_orientation'](obj,torch.tensor([1.,2.,3.]),torch.tensor([0.,0.,0.6,0.8]),frame='parent')
        results.append((prim.GetAttribute('xformOp:translate').Get(),prim.GetAttribute('xformOp:orient').Get()))
        assert state.caches_dirty and events[-1]=='world'
        assert events.count('sync')==(2 if code==source else 1)
    assert results[0]==results[1]
    with pytest.raises(ValueError,match='SDK_POSE_SETTER_CHANGED'):
        batched_pose_source(source.replace('orientation = orientation[[3, 0, 1, 2]].tolist()',"self.get_position_orientation()\n        orientation = orientation[[3, 0, 1, 2]].tolist()"))


def test_isaac_thread_limit_uses_native_launcher_config(monkeypatch):
    import sys,types
    from rummagebench.sim.omnigibson.inside_volume import configure_isaac_thread_limit
    module=types.ModuleType('isaacsim');module.SimulationApp=SimpleNamespace(DEFAULT_LAUNCHER_CONFIG={'limit_cpu_threads':32,'physics_gpu':0})
    monkeypatch.setitem(sys.modules,'isaacsim',module)
    monkeypatch.delenv('YGBENCH_ISAAC_THREADS',raising=False)
    configure_isaac_thread_limit();assert module.SimulationApp.DEFAULT_LAUNCHER_CONFIG['limit_cpu_threads']==32
    monkeypatch.setenv('YGBENCH_ISAAC_THREADS','4');configure_isaac_thread_limit()
    assert module.SimulationApp.DEFAULT_LAUNCHER_CONFIG=={'limit_cpu_threads':4,'physics_gpu':0}
    monkeypatch.setenv('YGBENCH_ISAAC_THREADS','0')
    with pytest.raises(ValueError):configure_isaac_thread_limit()


@pytest.mark.parametrize('case',['canonical','missing','extra','order','animated','nonunit','negative_scale','parent_nonuniform','parent_shear','reset'])
def test_canonical_transform_guard_rejects_ambiguous_usd(case):
    from pxr import Usd,UsdGeom,Gf
    from rummagebench.sim.omnigibson.inside_volume import canonical_xform
    stage=Usd.Stage.CreateInMemory();parent=UsdGeom.Xform.Define(stage,'/Parent')
    x=UsdGeom.Xform.Define(stage,'/Parent/Object')
    translation=x.AddTranslateOp();translation.Set(Gf.Vec3d(1,2,3))
    rotation=x.AddOrientOp();rotation.Set(Gf.Quatf(1))
    scale=x.AddScaleOp();scale.Set(Gf.Vec3f(1,2,3))
    if case=='missing':x.GetPrim().RemoveProperty('xformOp:scale')
    elif case=='extra':x.AddRotateXOp().Set(0)
    elif case=='order':x.SetXformOpOrder([rotation,translation,scale])
    elif case=='animated':translation.Set(Gf.Vec3d(1),Usd.TimeCode(1))
    elif case=='nonunit':rotation.Set(Gf.Quatf(2))
    elif case=='negative_scale':scale.Set(Gf.Vec3f(-1,1,1))
    elif case=='parent_nonuniform':parent.AddScaleOp().Set(Gf.Vec3f(1,2,1))
    elif case=='parent_shear':
        mat=Gf.Matrix4d(1);mat.SetRow(0,Gf.Vec4d(1,.2,0,0));parent.AddTransformOp().Set(mat)
    elif case=='reset':x.SetResetXformStack(True)
    before=stage.GetRootLayer().ExportToString()
    assert canonical_xform(x.GetPrim())==(case=='canonical')
    assert stage.GetRootLayer().ExportToString()==before


def test_canonical_transform_keeps_native_slow_path_and_pose_assertions():
    from rummagebench.sim.omnigibson.inside_volume import canonical_xform_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/prims/xform_prim.py'))
    source=path.read_text();node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_set_xform_properties')
    original=ast.parse(ast.get_source_segment(source,node)).body[0]
    fast=ast.parse(canonical_xform_source(ast.get_source_segment(source,node))).body[0]
    conditional=next(n for n in fast.body if isinstance(n,ast.If))
    flattened=[]
    for n in fast.body:flattened.extend(n.body if n is conditional else [n])
    assert [ast.dump(n) for n in flattened]==[ast.dump(n) for n in original.body]


@pytest.mark.parametrize('playing,valid,guard,lifecycle,expected_sync',[
    (True,True,True,0,0),(False,True,True,0,1),(True,False,True,0,1),
    (True,True,False,0,1),(True,True,True,1,1)])
def test_physics_velocity_scope_preserves_tensor_write_and_fallback(playing,valid,guard,lifecycle,expected_sync):
    import contextlib,torch
    from rummagebench.sim.omnigibson.inside_volume import physics_velocity_scope
    calls=[]
    @contextlib.contextmanager
    def edit():
        yield;calls.append('sync')
    sim=SimpleNamespace(is_playing=lambda:playing,_usd_guard_enabled=guard,_in_sim_lifecycle=lifecycle,
        _editing_usd=False,currently_in_isaac_step=False,_check_usd_guard=lambda:None,editing_usd=edit)
    view=SimpleNamespace(_physics_view=object(),is_physics_handle_valid=lambda:valid,velocity=torch.zeros(3))
    velocity=torch.tensor([1.5,-2.,0.3])
    with physics_velocity_scope(sim,view):view.velocity.copy_(velocity)
    assert torch.equal(view.velocity,velocity) and calls.count('sync')==expected_sync


def test_physics_velocity_scope_rejects_unexpected_real_usd_writes():
    from pxr import Usd,Tf,Sdf
    from rummagebench.sim.omnigibson.inside_volume import physics_velocity_scope
    stage=Usd.Stage.CreateInMemory();prim=stage.DefinePrim('/Object','Xform');errors=[]
    listener=Tf.Notice.Register(Usd.Notice.ObjectsChanged,lambda notice,sender:errors.append('unexpected USD write'),stage)
    def check():
        if errors:raise RuntimeError(errors[0])
    sim=SimpleNamespace(is_playing=lambda:True,_usd_guard_enabled=True,_in_sim_lifecycle=0,
        _editing_usd=False,currently_in_isaac_step=False,_check_usd_guard=check)
    view=SimpleNamespace(_physics_view=object(),is_physics_handle_valid=lambda:True)
    try:
        with pytest.raises(RuntimeError,match='unexpected USD write'):
            with physics_velocity_scope(sim,view):prim.CreateAttribute('test',Sdf.ValueTypeNames.Float).Set(1.)
    finally:listener.Revoke()
    errors.clear();sim._editing_usd=True
    with pytest.raises(AssertionError,match='nest'):
        with physics_velocity_scope(sim,view):pass


@pytest.mark.parametrize('filename,method',[('rigid_dynamic_prim.py','set_linear_velocity'),('rigid_dynamic_prim.py','set_angular_velocity'),('entity_prim.py','set_joint_velocities')])
def test_velocity_rewrite_retains_all_native_setter_operations(filename,method):
    from rummagebench.sim.omnigibson.inside_volume import physics_velocity_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/prims/'+filename))
    source=path.read_text();node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==method)
    original=ast.parse(ast.get_source_segment(source,node));changed=ast.parse(physics_velocity_source(ast.get_source_segment(source,node)))
    replaced=0
    for part in ast.walk(changed):
        if isinstance(part,ast.With) and ast.unparse(part.items[0].context_expr).startswith('_findingbench_velocity_scope'):
            part.items[0].context_expr=ast.parse('og.sim.editing_usd()',mode='eval').body;replaced+=1
    assert replaced==1 and ast.dump(changed)==ast.dump(original)


def test_prepared_usd_reuse_preserves_native_state_and_hash(tmp_path):
    from rummagebench.sim.omnigibson.inside_volume import reuse_prepared_usd
    import torch
    source = tmp_path/'source.usd'
    source.write_text('asset')
    obj = SimpleNamespace(_load_config={'scale': None}, _relative_prim_path='/test',
        fixed_base=True, name='test', category='cabinet', model='model',
        _usd_path=str(source), _encrypted=False, _expected_file_hash=None)
    calls, checks = [], []
    obj.check_hash = lambda path: checks.append(path)
    def native(item):
        calls.append(item)
        item.check_hash(item._usd_path)
        item._load_config['scale'] = torch.ones(3)
        item._load_config['kinematic_only'] = False
        item._relative_prim_path = '/articulated__test'
        path = tmp_path/f'prepared{len(calls)}.usd'
        path.write_text('prepared')
        return str(path)
    stats = dict(prepare_hits=0, prepare_misses=0, prepare_fallbacks=0)
    wrapped = reuse_prepared_usd(native, lambda item: item is obj, stats)
    first = wrapped(obj)
    assert wrapped(obj) == first
    assert len(calls) == 1 and checks == [str(source)]*2
    assert obj._relative_prim_path == '/articulated__test'
    assert stats['prepare_hits'] == 1
    # A hit consumes the entry; the next preparation must run natively.
    assert wrapped(obj) != first and len(calls) == 2
    obj._load_config['scale'][0] = 2
    wrapped(obj)
    assert len(calls) == 3
    source.write_text('changed asset')
    wrapped(obj)
    assert len(calls) == 4
    cached = obj.__dict__['_findingbench_prepared_reuse'][1]
    Path(cached).unlink()
    wrapped(obj)
    assert len(calls) == 5
    obj._relative_prim_path = '/changed'
    wrapped(obj)
    assert len(calls) == 6
    obj._load_config['unsupported'] = object()
    wrapped(obj); wrapped(obj)
    assert len(calls) == 8 and '_findingbench_prepared_reuse' not in obj.__dict__


def test_prepared_usd_hash_failure_not_suppressed(tmp_path):
    from rummagebench.sim.omnigibson.inside_volume import reuse_prepared_usd
    import pytest
    path = tmp_path/'asset.usd'; path.write_text('asset')
    obj = SimpleNamespace(_load_config={}, _relative_prim_path='/x', fixed_base=False,
        name='x', category='x', model='x', _usd_path=str(path), _encrypted=False,
        _expected_file_hash=None, check_hash=lambda path: None)
    calls = []
    def native(item):
        calls.append(item)
        return str(path)
    stats = dict(prepare_hits=0, prepare_misses=0, prepare_fallbacks=0)
    wrapped = reuse_prepared_usd(native, lambda item: True, stats)
    wrapped(obj)
    def reject(path): raise RuntimeError('HASH_MISMATCH')
    obj.check_hash = reject
    with pytest.raises(RuntimeError, match='HASH_MISMATCH'): wrapped(obj)
    assert len(calls) == 1 and stats['prepare_hits'] == 0


def test_contact_diagnostics_capture_rejection_before_state_restore(capsys):
    import json
    from rummagebench.sim.omnigibson.inside_volume import SamplingAudit
    calls=[]
    def probe():
        calls.append(True)
        return dict(contact_pairs=[['/scene/item/base','/scene/cabinet/base']])
    audit=SamplingAudit('item','cabinet','volume',contact_probe=probe)
    audit['attempts']=1
    audit['RigidContactAPI.is_in_contact(test)']=1
    audit['RigidContactAPI.is_in_contact(test)']=2
    audit['RigidContactAPI.is_in_contact(test)']=20
    assert len(calls)==2 and len(audit.contact_events)==2
    assert audit['RigidContactAPI.is_in_contact(test)']==20
    events=[json.loads(s.split(' ',1)[1]) for s in capsys.readouterr().out.splitlines() if s.startswith('[inside-contact-audit]')]
    assert events[0]['contact_pairs'][0][1]=='/scene/cabinet/base'
    def broken():raise RuntimeError('diagnostic only')
    audit.contact_probe=broken;audit['RigidContactAPI.is_in_contact(test)']=40
    assert audit['RigidContactAPI.is_in_contact(test)']==40
    assert 'diagnostic_error' in audit.contact_events[-1]


def test_local_search_reduces_overlap_without_losing_native_fallback():
    import torch
    from rummagebench.sim.omnigibson.inside_volume import refine_proposal_positions
    bank=torch.tensor([[.5,.5,.5,0,0,0,1.]]*100)
    obstacles=[[[.35,.35,.35],[.65,.65,.65]]]
    refined=refine_proposal_positions(bank,[.1]*3,[0]*3,[1]*3,
        obstacles=obstacles,iterations=40,generator=torch.Generator().manual_seed(4))
    assert torch.equal(refined[50:],bank[50:])
    assert torch.equal(refined[:,3:],bank[:,3:])
    center=refined[:50,:3]+torch.tensor([0,0,.01])
    overlap=torch.relu(torch.minimum(center+.05,torch.tensor([.65]*3))
                        -torch.maximum(center-.05,torch.tensor([.35]*3))).prod(-1)
    assert (overlap<1e-8).sum()>=40
    assert ((refined[:50,:3]>=.05)&(refined[:50,:3]<=torch.tensor([.95,.95,.94])+1e-6)).all()


def test_local_search_does_not_assume_disconnected_cavity_gap_is_free():
    import torch
    from rummagebench.sim.omnigibson.inside_volume import refine_proposal_positions
    def boxplanes(lo,hi):
        return [[1,0,0,-hi[0]],[-1,0,0,lo[0]],[0,1,0,-hi[1]],
                [0,-1,0,lo[1]],[0,0,1,-hi[2]],[0,0,-1,lo[2]]]
    bank=torch.tensor([[.5,.5,.5,0,0,0,1.]]*100)
    refined=refine_proposal_positions(bank,[.05]*3,[0]*3,[1]*3,
        planes=[boxplanes([0,0,0],[.25,1,1]),boxplanes([.75,0,0],[1,1,1])],
        iterations=50,generator=torch.Generator().manual_seed(3))
    assert ((refined[:50,0]<=.2251)|(refined[:50,0]>=.7749)).sum()>=40
    assert torch.equal(refined[50:],bank[50:])


def test_local_search_keeps_oversize_candidates_for_native_rejection():
    import torch
    from rummagebench.sim.omnigibson.inside_volume import refine_proposal_positions
    bank=torch.tensor([[.5,.5,.5,0,0,0,1.]]*10)
    result=refine_proposal_positions(bank,[2]*3,[0]*3,[1]*3,obstacles=[[[.3]*3,[.6]*3]])
    assert torch.equal(result,bank)


def test_expert_pose_provider_returns_one_deterministic_candidate_without_rng():
    import torch
    from types import SimpleNamespace
    from rummagebench.sim.omnigibson.inside_volume import native_gpu_proposals
    pose=[1.,2.,3.,0.,0.,0.,1.]
    obj=SimpleNamespace(_findingbench_layout_candidate=pose)
    before=torch.random.get_rng_state().clone()
    bank=native_gpu_proposals(obj,None,None,None,100)
    assert bank.shape==(1,7) and bank.tolist()==[pose]
    assert torch.equal(torch.random.get_rng_state(),before)


def test_expert_candidate_keeps_native_rejection_checks_and_bounds_attempts():
    from rummagebench.sim.omnigibson.inside_volume import gpu_volume_source
    path=next(Path('/home/ubuntu/ygwang').glob('BEHAVIOR-1K*/OmniGibson/omnigibson/object_states/inside.py'))
    text=path.read_text();node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    source=gpu_volume_source(ast.get_source_segment(text,node))
    for predicate in ['RigidContactAPI.is_in_contact','container_link.check_points_in_volume','container_joint_positions_delta','n_steps_max']:
        assert predicate in source
    assert "getattr(self.obj, '_findingbench_layout_candidate', None) is not None" in source
    assert 'total_attempts = 1' in source
    assert 'self.obj.set_bbox_center_position_orientation(position=pos, orientation=orientation)' in source


def test_expert_live_plan_preserves_separate_moving_components(monkeypatch,capsys):
    import numpy as np,torch,pytest,sys,types
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from types import SimpleNamespace as NS
    from rummagebench.sim.omnigibson.inside_volume import plan_native_contents
    fake=types.ModuleType('omnigibson.object_states.contains')
    fake.m=NS(CONTAINER_META_LINK_TYPES={'fillable'})
    monkeypatch.setitem(sys.modules,'omnigibson.object_states.contains',fake)
    states=types.ModuleType('omnigibson.object_states');states.Open=object()
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',states)
    def mesh(x):
        transform=np.eye(4);transform[:3,3]=[x,20.,1.]
        return NS(points=np.array([[x,y,z] for x in [0.,.2] for y in [0.,.2] for z in [0.,.2]]),scaled_transform=transform)
    link=NS(is_meta_link=True,meta_link_type='fillable',visual_meshes={'a':mesh(10.),'b':mesh(12.)})
    link.prim_path='/cabinet/fillable'
    objects={'cabinet':NS(links={'meta__link_0_fillable_0_0_link':link}),
        'a':NS(native_bbox=np.array([.15]*3),scale=np.ones(3),scaled_bbox_center_in_base_frame=np.array([.01,-.02,.03])),
        'b':NS(native_bbox=np.array([.15]*3),scale=np.ones(3),scaled_bbox_center_in_base_frame=np.array([-.02,.01,-.01]))}
    backend=NS(_env=NS(scene=NS(object_registry=lambda _,name:objects[name])))
    monkeypatch.setattr('rummagebench.sim.omnigibson.inside_volume.measure_support_plane',
                        lambda lo,hi,owners:dict(status='measured_plane',floor_z_m=float(lo[2])))
    spec=NS(placements=[NS(entity=n,receptacle='cabinet',relation='inside',link='link_0') for n in ['a','b']])
    plan=plan_native_contents(backend,spec,'cabinet')
    xs=sorted(p['candidate'][0] for p in plan.values())
    assert np.allclose(xs,[10.1,12.1],atol=.00002)
    assert {p['native_link'] for p in plan.values()}=={'meta__link_0_fillable_0_0_link'}
    for name,p in plan.items():
        # The sampler must receive the desired bbox centre, not an already
        # converted root. Exercise nonzero native offsets end to end.
        assert p['candidate'][:3]==p['bbox_center_world']
        assert p['candidate_frame']=='world_bbox_center'
        yaw=p['solver_pose']['yaw_degrees'];angle=np.deg2rad(yaw)
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        expected=np.array(p['candidate'][:3])+rotation@objects[name].scaled_bbox_center_in_base_frame
        assert np.allclose(expected,p['predicted_root_pose'][:3])


def test_bbox_plan_uses_native_root_offset_and_rotates_it():
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import bbox_center_root_pose
    # Measured baking-powder geometry at identity: bbox centre = root + offset.
    model_center_offset=np.array([.0059521,-.0001487,.0123555])
    for yaw,rotation in [(0,np.eye(3)),(90,np.array([[0,-1,0],[1,0,0],[0,0,1]]))]:
        desired=np.array([1.,2.,.3]);pose=bbox_center_root_pose(desired,yaw,-model_center_offset)
        assert np.allclose(np.array(pose[:3])+rotation@model_center_offset,desired)


def test_collision_floor_measurement_handles_metadata_offsets_and_fails_closed():
    from rummagebench.sim.omnigibson.inside_volume import measure_support_plane
    def ray(body='/cabinet/base',height=.98,normal=(0,0,1)):
        return lambda start,end:dict(hit=True,rigidBody=body,position=[start[0],start[1],height],normal=normal)
    report=measure_support_plane([0,0,1],[1,1,1.4],['/cabinet'],ray())
    assert report['status']=='measured_plane' and report['floor_z_m']==.98
    assert abs(report['floor_minus_fillable_bottom_m']+.02)<1e-9
    for cast in [ray('/cabinet_neighbor/base'),ray(normal=(0,0,.5)),lambda *args:dict(hit=False),
                 lambda start,end:dict(hit=True,rigidBody='/cabinet/base',position=[start[0],start[1],.95+start[0]*.1],normal=[0,0,1])]:
        assert measure_support_plane([0,0,1],[1,1,1.4],['/cabinet'],cast)['status']=='unknown'


def test_support_patches_preserve_usable_region_without_bridging_foreign_hits():
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import measure_support_patches
    def ray(start,end):
        x,y,z=start
        return dict(hit=True,rigidBody='/foreign' if .4<x<.6 else '/cabinet/base',
                    position=[x,y,1. if x<.5 else 1.02],normal=[0,0,1])
    patches=measure_support_patches([0,0,1],[1,1,1.4],['/cabinet'],ray)
    assert patches and {round(p['floor_z_m'],2) for p in patches}=={1.,1.02}
    for p in patches:
        assert p['hi'][0]<=.4 or p['lo'][0]>=.6
        assert 0<=p['lo'][0]<p['hi'][0]<=1
        assert p['physical_validation_required']
    for i,a in enumerate(patches):
        for b in patches[i+1:]:
            assert any(a['hi'][axis]<=b['lo'][axis] or b['hi'][axis]<=a['lo'][axis] for axis in [0,1])
    assert not measure_support_patches([0,0,1],[1,1,1.4],['/cabinet'],lambda *args:dict(hit=False))


def test_long_shallow_drawer_clips_domains_before_cartesian_allocation():
    import pytest,torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    items={str(i):[length,width,.004] for i,(length,width) in enumerate(zip(
        [.103,.127,.151,.173,.197,.211,.233,.257],[.003,.004,.005,.006,.007,.008,.009,.010]))}
    result=solve_layout(items,{'drawer':[1.6,.08,.03]},margin_m=.001,time_limit_s=2.,max_candidates=100000)
    assert result['status']=='feasible_model'
    poses=list(result['poses'].values())
    for p in poses:
        x,y=p['lower_xy_m'];w,h=p['footprint_m']
        assert 0<=x and 0<=y and x+w<=1.600001 and y+h<=.080001
    for i,a in enumerate(poses):
        for b in poses[i+1:]:
            assert any(a['lower_xy_m'][axis]+a['footprint_m'][axis]<=b['lower_xy_m'][axis]+1e-8 or
                       b['lower_xy_m'][axis]+b['footprint_m'][axis]<=a['lower_xy_m'][axis]+1e-8 for axis in [0,1])


def test_open_vessel_proposals_allow_overhang_but_retain_native_centre_membership(monkeypatch):
    import numpy as np,torch,pytest,sys,types
    from types import SimpleNamespace as NS
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import rummagebench.sim.omnigibson.inside_volume as m
    contains=types.ModuleType('omnigibson.object_states.contains');contains.m=NS(CONTAINER_META_LINK_TYPES={'fillable'})
    states=types.ModuleType('omnigibson.object_states');states.Open=object()
    monkeypatch.setitem(sys.modules,'omnigibson.object_states.contains',contains)
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',states)
    mesh=NS(points=np.array([[x,y,z] for x in [0.,.4] for y in [0.,.4] for z in [1.,1.16]]),scaled_transform=np.eye(4))
    rec=NS(category='shopping_basket',states={},links={'fillable':NS(is_meta_link=True,meta_link_type='fillable',
        prim_path='/basket/fillable',visual_meshes={'a':mesh})})
    obj=NS(native_bbox=np.array([.1,.1,.25]),scale=np.ones(3),scaled_bbox_center_in_base_frame=np.zeros(3))
    backend=NS(_env=NS(scene=NS(object_registry=lambda _,name:rec if name=='basket' else obj)))
    spec=NS(placements=[NS(entity='carton',receptacle='basket',relation='inside',link=None)])
    monkeypatch.setattr(m,'measure_support_plane',lambda lo,hi,owners:dict(status='measured_plane',floor_z_m=float(lo[2])))
    proposal=m.plan_native_contents(backend,spec,'basket')['carton']
    assert 1.<proposal['candidate'][2]<1.16
    assert proposal['candidate'][2]+.25/2>1.16
    assert proposal['measured_support']['open_top_centroid_height']
    rec.states[states.Open]=object()
    with pytest.raises(ValueError,match='EXPERT_LAYOUT_UNRESOLVED'):m.plan_native_contents(backend,spec,'basket')
    # A thin utensil 0.7 mm above the fillable lower boundary is a legal native
    # centre; an extra authoring-only 2 mm inset must not forbid it.
    rec.category='bottom_cabinet';obj.native_bbox=np.array([.1,.1,.005])
    monkeypatch.setattr(m,'measure_support_plane',lambda lo,hi,owners:dict(status='measured_plane',floor_z_m=float(lo[2]-.0018)))
    thin=m.plan_native_contents(backend,spec,'basket')['carton']
    assert 1.<thin['candidate'][2]<1.002


def test_expert_pose_log_accepts_native_list_values(capsys):
    import ast,json
    from pathlib import Path
    from types import SimpleNamespace as NS
    backend=Path(__file__).resolve().parents[2]/'src/rummagebench/sim/omnigibson/backend.py'
    tree=ast.parse(backend.read_text())
    statement=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and any(isinstance(v,ast.Constant) and v.value=='[expert-pose-result] ' for v in ast.walk(n)))
    obj=NS(get_position_orientation=lambda:([1.,2.,3.],[0.,0.,0.,1.]),aabb=([0.,1.,2.],[2.,3.,4.]))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[statement],type_ignores=[])),'pose-log','exec'),
         dict(obj=obj,p=NS(entity='jar'),ok=True,json=json))
    result=json.loads(capsys.readouterr().out.split('[expert-pose-result] ',1)[1])
    assert result['position']==[1.,2.,3.] and result['aabb']==[[0.,1.,2.],[2.,3.,4.]]


def test_coarse_proposal_fallback_retains_constraints_and_reports_its_grid(monkeypatch):
    import rummagebench.sim.omnigibson.inside_volume as volume
    calls=[]
    def solve(items,bins,fixed,**kwargs):
        calls.append((items,bins,fixed,kwargs))
        return dict(status='unknown',reason='Candidate tensor memory budget exhausted') if len(calls)==1 else dict(status='feasible_model',grid_m=kwargs['grid_m'])
    monkeypatch.setattr(volume,'solve_layout',solve)
    options=dict(margin_m=.015,center_halfspaces={'box':[[1,0,0,-1]]},obstacles={'box':[[0,0,0,.1,.1,.1]]})
    result=volume.solve_layout_with_coarse_fallback({'jar':[.09,.09,.1]},{'box':[1,1,1]},{'jar':['box']},**options)
    assert result['status']=='feasible_model' and result['grid_m']==.005
    assert calls[1][3]==dict(options,grid_m=.005)
    assert calls[0][:3]==calls[1][:3]


def test_coarse_proposal_failure_never_certifies_fine_model_infeasible(monkeypatch):
    import rummagebench.sim.omnigibson.inside_volume as volume
    def solve(*args,**kwargs):
        return dict(status='infeasible_model') if kwargs.get('grid_m')==.005 else dict(status='unknown',reason='Compact corner proposal subset exhausted; complete grid not exhausted')
    monkeypatch.setattr(volume,'solve_layout',solve)
    result=volume.solve_layout_with_coarse_fallback({}, {})
    assert result['status']=='unknown' and result['coarse_proposal_attempt']['status']=='infeasible_model'


def test_container_preparation_waits_for_real_joint_convergence(monkeypatch):
    import sys,types,pytest
    from types import SimpleNamespace as NS
    from rummagebench.sim.omnigibson.inside_volume import settle_authoring_container
    marker=object();fake=types.ModuleType('omnigibson.object_states');fake.Open=marker
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',fake)
    state={'step':0};trajectory=[0.,.1,.12,.12,.12,.12]
    container=NS(name='cabinet',n_joints=1,joints={'door':NS(lower_limit=0.,upper_limit=1.)},
        states={marker:NS(get_value=lambda:True)},get_joint_positions=lambda:[trajectory[min(state['step'],len(trajectory)-1)]],
        get_joint_velocities=lambda:[0.])
    backend=NS(settle=lambda steps:state.update(step=state['step']+1))
    result=settle_authoring_container(backend,container)
    assert result['quiet_windows']==3 and result['extra_steps']==75
    state['step']=0;trajectory[:]=[i*.1 for i in range(8)]
    with pytest.raises(ValueError,match='EXPERT_CONTAINER_JOINTS_NOT_SETTLED'):
        settle_authoring_container(backend,container,max_windows=4)


def test_articulation_audit_records_substep_motion_without_changing_failure_decision():
    import numpy as np
    from types import SimpleNamespace as NS
    from rummagebench.sim.omnigibson.inside_volume import observe_articulation_motion
    step=[0]
    def advance():step[0]+=1
    view=NS(count=1,get_dof_positions=lambda:np.array([[.002*np.sin(step[0]*np.pi/2)]]),
            get_dof_velocities=lambda:np.array([[8.]]))
    backend=NS(_sim=NS(get_physics_dt=lambda:1/120,step_physics=advance,
        physics_sim_view=NS(create_articulation_view=lambda path:view)))
    container=NS(name='cabinet',prim_path='/cabinet',joints={'door':NS(lower_limit=0.,upper_limit=1.)})
    report=observe_articulation_motion(backend,container,steps=32)
    assert report['acceptance']=='diagnostic_only_original_failure_retained'
    assert report['max_normalized_measured_speed']>.05
    assert report['max_normalized_reported_speed']==8. and step[0]==32
    from rummagebench.sim.omnigibson.inside_volume import articulation_motion_converged
    assert not articulation_motion_converged(report)
    stationary=dict(report,max_normalized_measured_speed=.0001,max_normalized_excursion=.000001)
    assert articulation_motion_converged(stationary)
    assert not articulation_motion_converged(dict(stationary,all_finite=False))


def test_authoring_open_targets_follow_native_direction_without_touching_joints(monkeypatch):
    import sys,types,pytest
    from types import SimpleNamespace as NS
    from rummagebench.sim.omnigibson.inside_volume import native_authoring_open_targets
    fake=types.ModuleType('omnigibson.object_states.open_state')
    positive,negative=NS(name='positive'),NS(name='negative')
    fake._get_relevant_joints=lambda obj:(False,[positive,negative],[1,-1])
    fake._compute_joint_threshold=lambda joint,direction:(.05,1.,0.) if direction==1 else (.95,0.,1.)
    monkeypatch.setitem(sys.modules,'omnigibson.object_states.open_state',fake)
    targets=native_authoring_open_targets(object(),.9)
    assert targets[0][0] is positive and targets[0][1]==pytest.approx(.9)
    assert targets[1][0] is negative and targets[1][1]==pytest.approx(.1)
    for invalid in [0.,1.,float('nan')]:
        with pytest.raises(ValueError,match='INVALID_AUTHORING'):native_authoring_open_targets(object(),invalid)
    fake._get_relevant_joints=lambda obj:(True,[positive],[1])
    with pytest.raises(ValueError,match='UNSUPPORTED_AUTHORING'):native_authoring_open_targets(object(),.9)


def test_cuda_layout_respects_actual_convex_centre_not_only_aabb():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    # A tilted native cavity bottom is z > .2*x. An AABB-only proposal at
    # the lower corner for a 4 mm object can lie below that actual face.
    items={'tool':[.04,.02,.004]};bins={'drawer':[.1,.1,.1]}
    plain=solve_layout(items,bins,margin_m=.005)
    assert plain['status']=='feasible_model'
    rejected=solve_layout(items,bins,margin_m=.005,center_halfspaces={'drawer':[[.2,0.,-1.,0.]]})
    assert rejected['status']=='infeasible_model'
    # A 20 mm object has legal centres; exact halfspaces remain satisfiable.
    feasible=solve_layout({'tool':[.04,.02,.02]},bins,margin_m=.005,
                          center_halfspaces={'drawer':[[.2,0.,-1.,0.]]})
    assert feasible['status']=='feasible_model'
    x=feasible['poses']['tool']['center_xy_m'][0]
    assert .2*x-.01 < -.00001


def test_constrained_layout_searches_grid_when_normal_patterns_cannot_compact():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    # The only legal centre band is x in (.031,.034); neither unconstrained
    # compacted normal-pattern centre lies there. A full 1 mm grid can find it.
    result=solve_layout({'a':[.02,.02,.02]},{'bin':[.08,.08,.04]},margin_m=.005,
        center_halfspaces={'bin':[[-1.,0.,0.,.031],[1.,0.,0.,-.034]]})
    assert result['status']=='feasible_model'
    assert .031 < result['poses']['a']['center_xy_m'][0] < .034


def test_clearance_refinement_preserves_separation_bounds_and_convex_membership():
    import torch,copy
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout,refine_layout_clearance
    items={'a':[.12,.1,.08],'b':[.1,.08,.08],'c':[.06,.05,.08]};bins={'cabinet':[.6,.5,.3]}
    planes={'cabinet':[[1.,0.,0.,-.5],[0.,-1.,0.,.04]]}
    solved=solve_layout(items,bins,margin_m=.015,center_halfspaces=planes)
    before=copy.deepcopy(solved)
    refined=refine_layout_clearance(items,bins,solved,planes)
    assert solved==before and refined['clearance_refinement']
    poses=list(refined['poses'].values())
    for p in poses:
        x,y=p['center_xy_m'];w,h=p['footprint_m']
        assert .015-1e-9<=x-w/2 and x+w/2<=.585+1e-9
        assert .015-1e-9<=y-h/2 and y+h/2<=.485+1e-9
        assert x<.5-.00001 and y>.04+.00001
    for i,a in enumerate(poses):
        for b in poses[i+1:]:
            assert any(abs(a['center_xy_m'][k]-b['center_xy_m'][k]) >= (a['footprint_m'][k]+b['footprint_m'][k])/2+.015-1e-9 for k in range(2))
    oldgap=min(p['lower_xy_m'][0] for p in solved['poses'].values())
    assert min(p['lower_xy_m'][0] for p in poses)>oldgap


def test_layout_replans_around_observed_settled_obstacle_not_its_old_pose():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    bin={'cavity':[.4,.2,.2]};item={'next':[.08,.08,.1]}
    initial=solve_layout(item,bin,margin_m=.015)
    assert initial['poses']['next']['lower_xy_m']==[.015,.015]
    # The previous object tilted into the old next-object slot. Its observed
    # box now extends to x=.2; keep 15 mm clearance from that actual extent.
    changed=solve_layout(item,bin,margin_m=.015,obstacles={'cavity':[[0.,0.,0.,.2,.2,.15]]})
    assert changed['status']=='feasible_model'
    assert changed['poses']['next']['lower_xy_m'][0]>=.215-1e-9
    # A box on a different shelf must not exclude an otherwise legal slot.
    overhead=solve_layout(item,bin,margin_m=.015,obstacles={'cavity':[[0.,0.,.16,.2,.2,.19]]})
    assert overhead['status']=='feasible_model'
    assert overhead['poses']['next']['lower_xy_m']==[.015,.015]


def test_whole_layout_candidate_reuse_observes_settled_obstacles_and_joints():
    import numpy as np
    from types import SimpleNamespace as NS
    from rummagebench.sim.omnigibson.inside_volume import remember_native_contents_plan,reusable_native_contents_plan
    position=np.array([.3])
    rec=NS(n_joints=1,get_joint_positions=lambda:position)
    first=NS(aabb=(np.array([.0,.0,.0]),np.array([.1,.1,.1])))
    next_obj=NS(native_bbox=np.array([.08,.08,.08]),scale=1.)
    objects={'cabinet':rec,'first':first,'next':next_obj}
    backend=NS(_env=NS(scene=NS(object_registry=lambda kind,name:objects[name])))
    plan={'next':dict(bbox_center_world=[.24,.08,.04],solver_pose={'footprint_m':[.08,.08]},
                      native_link='fillable',candidate=[.24,.08,.04,0,0,0,1])}
    context={};remember_native_contents_plan(backend,'cabinet',context,plan)
    placements=[NS(entity='next')]
    reused=reusable_native_contents_plan(backend,'cabinet',context,placements,{'first'})
    assert reused==plan
    reused['next']['candidate'][0]=999
    assert context['pending_plan']['next']['candidate'][0]==.24
    first.aabb=(np.array([.15,.0,.0]),np.array([.3,.1,.1]))
    assert reusable_native_contents_plan(backend,'cabinet',context,placements,{'first'}) is None
    first.aabb=(np.array([.0,.0,.0]),np.array([.1,.1,.1]));position[0]=.32
    assert reusable_native_contents_plan(backend,'cabinet',context,placements,{'first'}) is None
    position[0]=.3
    assert reusable_native_contents_plan(backend,'cabinet',context,[NS(entity='unplanned')],{'first'}) is None


def test_open_basket_accepts_only_its_native_verified_direct_support(monkeypatch):
    import numpy as np,sys,types,pytest
    from types import SimpleNamespace as NS
    import rummagebench.sim.omnigibson.inside_volume as m
    states=types.ModuleType('omnigibson.object_states');states.Open=object();states.OnTop=object()
    contains=types.ModuleType('omnigibson.object_states.contains');contains.m=NS(CONTAINER_META_LINK_TYPES={'fillable'})
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',states)
    monkeypatch.setitem(sys.modules,'omnigibson.object_states.contains',contains)
    mesh=NS(points=np.array([[x,y,z] for x in [0.,.4] for y in [0.,.3] for z in [0.,.2]]),scaled_transform=np.eye(4))
    verified=[True];seen=[]
    basket=NS(category='shopping_basket',states={states.OnTop:NS(get_value=lambda other:verified[0] and other is not unrelated)},
        links={'fillable':NS(is_meta_link=True,meta_link_type='fillable',prim_path='/basket/base',visual_meshes={'a':mesh})})
    table=NS(links={'base':NS(prim_path='/table/base')})
    neighbour=NS(links={'base':NS(prim_path='/adjacent_counter/base')})
    unrelated=NS(links={'base':NS(prim_path='/unrelated/base')})
    backend=NS(_env=NS(scene=NS(objects=[basket,table,neighbour,unrelated],object_registry=lambda _,name:basket if name=='basket' else table)))
    spec=NS(placements=[NS(entity='basket',relation='on_top',receptacle='table'),NS(entity='milk',relation='inside',receptacle='basket')])
    def stop_after_ownership(lo,hi,owners):
        seen.append(list(owners));raise RuntimeError('ownership captured')
    monkeypatch.setattr(m,'measure_support_plane',stop_after_ownership)
    for value in [True,False]:
        verified[0]=value
        with pytest.raises(RuntimeError,match='ownership captured'):m.plan_native_contents(backend,spec,'basket')
    assert '/table/base' in seen[0] and '/table/base' not in seen[1]
    assert '/basket/base' in seen[0] and '/basket/base' in seen[1]
    assert '/adjacent_counter/base' in seen[0] and '/unrelated/base' not in seen[0]


def test_rigid_container_diagnostics_never_call_articulation_api(monkeypatch,capsys):
    import sys,types
    from types import SimpleNamespace as NS
    import rummagebench.sim.omnigibson.inside_volume as m
    inside=object();states=types.ModuleType('omnigibson.object_states');states.Inside=inside
    og=types.ModuleType('omnigibson');og.sim=NS(dump_state=lambda **kw:{},load_state=lambda *args,**kw:None)
    utils=types.ModuleType('omnigibson.utils.usd_utils');utils.RigidContactAPI=NS(get_contact_pairs=lambda **kw:set())
    monkeypatch.setitem(sys.modules,'omnigibson',og);monkeypatch.setitem(sys.modules,'omnigibson.object_states',states)
    monkeypatch.setitem(sys.modules,'omnigibson.utils.usd_utils',utils)
    monkeypatch.setenv('YGBENCH_AUTHORING_CONTACT_AUDIT','1')
    class State:
        _initialized=True
        def _set_value(self,*args):pass
        def clear_cache(self):pass
    def forbidden():raise AssertionError('joint API must not run for rigid basket')
    rec=NS(name='basket',n_joints=0,links={'fillable':NS(is_meta_link=True,name='basket:fillable')},
           get_joint_positions=forbidden,get_joint_velocities=forbidden)
    obj=NS(name='milk',states={inside:State()},scene=NS(idx=0),get_position_orientation=lambda:([0,0,0],[0,0,0,1]),
           aabb=[[0,0,0],[.1,.1,.1]],state_updated=lambda:None)
    def sampler(state,other,value,**kw):
        kw['_findingbench_audit']['final_membership_or_reachability_failed']=1
        return False
    monkeypatch.setattr(m,'_sampler',lambda *args,**kw:(sampler,'digest'))
    success,evidence=m.sample_selected_volume(obj,rec,'fillable')
    assert not success and evidence['counts']['final_membership_or_reachability_failed']==1
    assert '"current_container_joints": []' in capsys.readouterr().out


@pytest.mark.parametrize('foreign,hole,normal_z,accepted',[
    (False,False,1.,True),(True,False,1.,False),
    (False,True,1.,False),(False,False,.5,False)])
def test_shelf_drop_height_requires_owned_complete_horizontal_support(foreign,hole,normal_z,accepted):
    from rummagebench.sim.omnigibson.inside_volume import measure_shelf_drop_height
    def ray(start,end):
        corner=start[0]<.5 and start[1]<.5
        return dict(hit=not (hole and corner),rigidBody='/neighbour' if foreign and corner else '/shelf/base',
                    position=[start[0],start[1],.04],normal=[0,0,normal_z])
    result=measure_shelf_drop_height([.5,.5],[.2,.2],[[0,0,0],[1,1,.6]],['/shelf/base'],ray)
    assert (result is not None)==accepted
    if result:assert result['height_m']==.04 and result['physical_validation_required']


def test_cuda_strip_witness_avoids_sampling_memory_for_thin_utensils():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    items={f'utensil_{i}':[.15+i*.01,.006+i*.001,.01] for i in range(9)}
    result=solve_layout(items,{'drawer':[.6,.5,.1]},margin_m=.005,max_candidates=16)
    assert result['status']=='feasible_model' and result['constructive_witness']
    poses=list(result['poses'].values())
    for i,a in enumerate(poses):
        for b in poses[i+1:]:
            assert any(abs(a['center_xy_m'][k]-b['center_xy_m'][k]) >= (a['footprint_m'][k]+b['footprint_m'][k])/2+.005-1e-9 for k in range(2))


def test_rotated_container_checks_horizontal_box_corners_not_only_centre():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    bins={'diamond':[.2,.2,.15]};planes={'diamond':[[1.,1.,0.,-.12],[-1.,-1.,0.,.04],[1.,-1.,0.,-.06],[-1.,1.,0.,-.06]]}
    item={'box':[.08,.08,.05]}
    assert solve_layout(item,bins,center_halfspaces=planes)['status']=='feasible_model'
    assert solve_layout(item,bins,center_halfspaces=planes,lateral_inset=True)['status']=='infeasible_model'
    assert solve_layout({'small':[.02,.02,.05]},bins,center_halfspaces=planes,lateral_inset=True)['status']=='feasible_model'


def test_static_native_domain_rejection_does_not_backtrack_through_large_item():
    import torch,pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    # The card is below the native volume's floor at every XY. Moving the
    # larger object cannot help, so this is provably infeasible without DFS.
    result=solve_layout({'large':[.13,.13,.02],'thin_card':[.02,.02,.0005]},
        {'drawer':[.25,.25,.1]},grid_m=.01,margin_m=0.,max_nodes=1,
        center_halfspaces={'drawer':[[0.,0.,-1.,.002]]})
    assert result['status']=='infeasible_model' and result['blocking_entities']==['thin_card']
    assert result['nodes']==0


def test_layout_obeys_the_orientation_the_surface_placer_can_execute():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    items={'package':[.08,.02,.01]};volumes={'surface':[.04,.1,.1]}
    assert solve_layout(items,volumes,margin_m=0.,allowed_yaws={'package':[0]})['status']=='infeasible_model'
    rotated=solve_layout(items,volumes,margin_m=0.,allowed_yaws={'package':[90]})
    assert rotated['status']=='feasible_model' and rotated['poses']['package']['yaw_degrees']==90
    square=solve_layout({'square':[.02,.02,.01]},volumes,allowed_yaws={'square':[90]})
    assert square['status']=='feasible_model' and square['poses']['square']['yaw_degrees']==90


def test_shared_floor_patches_cannot_hide_overlapping_objects_in_separate_bins():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    items={'a':[.1,.1,.02],'b':[.1,.1,.02]};bins={'left_patch':[.15,.1,.1],'right_patch':[.15,.1,.1]}
    fixed={'a':['left_patch'],'b':['right_patch']}
    independent=solve_layout(items,bins,fixed,grid_m=.01,margin_m=0.)
    assert independent['status']=='feasible_model'
    shared=solve_layout(items,bins,fixed,grid_m=.01,margin_m=0.,time_limit_s=2.,
        overlap_groups={'left_patch':'one_drawer','right_patch':'one_drawer'})
    assert shared['status']=='infeasible_model'
    larger={n:[.25,.1,.1] for n in bins}
    shared=solve_layout(items,larger,fixed,grid_m=.01,margin_m=0.,time_limit_s=2.,
        overlap_groups={'left_patch':'one_drawer','right_patch':'one_drawer'})
    assert shared['status']=='feasible_model'
    a,b=(shared['poses'][n]['lower_xy_m'] for n in ['a','b'])
    assert abs(a[0]-b[0])>=.1-1e-9


def test_thin_native_contents_can_use_real_measured_object_top(monkeypatch):
    import numpy as np,torch,pytest
    from types import SimpleNamespace as NS
    import rummagebench.sim.omnigibson.inside_volume as m
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    planes=np.array([[-1.,0,0,0],[1.,0,0,-.5],[0,-1.,0,0],[0,1.,0,-.5],[0,0,-1.,.01],[0,0,1.,-.2]])
    context=dict(volumes={'floor':[.5,.5,.2]},bounds={'floor':(np.zeros(3),np.array([.5,.5,.2]))},
        native={'floor':'fillable'},membership_bottom={'floor':.01},support_reports={'floor':{}},
        halfspaces={'floor':planes.tolist()},components={'cavity':dict(link='fillable',lo=np.array([0.,0.,.01]),hi=np.array([.5,.5,.2]),equations=planes)},open_top=False)
    parent=NS(aabb=(np.array([.1,.1,.01]),np.array([.2,.18,.05])),links={'body':NS(prim_path='/box')})
    card=NS(native_bbox=np.array([.16,.08,.0005]),scale=1.,scaled_bbox_center_in_base_frame=np.zeros(3))
    rec=NS(fixed_base=True,links={});objects={'box':parent,'card':card,'cabinet':rec}
    backend=NS(_env=NS(scene=NS(object_registry=lambda kind,name:objects[name])),_findingbench_contents_contexts={'cabinet':context})
    scenario=NS(placements=[NS(entity=name,relation='inside',receptacle='cabinet',link=None) for name in ['box','card']])
    calls=[]
    def measured(lo,hi,owners):
        calls.append(owners);return dict(status='measured_plane',floor_z_m=.05,physical_validation_required=True)
    monkeypatch.setattr(m,'measure_support_plane',measured)
    plan=m.replan_native_contents(backend,scenario,'cabinet',{'box'},domains_current=True)
    assert set(plan)=={'card'} and calls==[['/box']]
    assert plan['card']['measured_support']['support_entity']=='box'
    x,y,z=plan['card']['bbox_center_world']
    assert .1<x<.2 and .1<y<.18 and z==pytest.approx(.05025)
    assert card.native_bbox[0]>.1  # a supported overhang, checked again by physics
    assert set(context['volumes'])=={'floor'}  # do not mutate empty-cavity domains
    monkeypatch.setattr(m,'measure_support_plane',lambda *args,**kw:dict(status='unknown'))
    assert set(m.add_native_stacking_supports(backend,context,{'box'})['volumes'])=={'floor'}


def test_interval_projected_native_grid_matches_exhaustive_coupled_plane_search():
    import torch,pytest,itertools
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_layout
    shapes=[(2,1),(1,1),(2,2)]
    for slope,threshold in [(0.3,.035),(-.4,.025),(1.,.025)]:
        planes=[[1.,slope,0.,-threshold],[-1.,-.2,0.,.008]]
        for first in shapes:
            items={'a':[first[0]*.01,first[1]*.01,.01],'b':[.02,.01,.01]}
            domains=[]
            for shape in [first,(2,1)]:
                choices=[]
                for w,h in set([shape,shape[::-1]]):
                    for x,y in itertools.product(range(5-w),range(5-h)):
                        point=((x+w/2)*.01,(y+h/2)*.01,.005)
                        if all(sum(p[k]*point[k] for k in range(3))+p[3]<-.00001 for p in planes):choices.append((x,y,w,h))
                domains.append(choices)
            expected=any(a[0]+a[2]<=b[0] or b[0]+b[2]<=a[0] or a[1]+a[3]<=b[1] or b[1]+b[3]<=a[1]
                         for a,b in itertools.product(*domains))
            result=solve_layout(items,{'bin':[.04,.04,.05]},grid_m=.01,margin_m=0.,time_limit_s=1.,center_halfspaces={'bin':planes})
            assert result['status']!='unknown' and (result['status']=='feasible_model')==expected


def test_native_drop_starts_keep_xy_and_fit_the_original_cavity():
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import native_drop_candidates
    planes=[[-1.,0,0,0],[1.,0,0,-.2],[0,-1.,0,0],[0,1.,0,-.2],[0,0,-1.,0],[0,0,1.,-.1]]
    candidate=[.1,.1,.02,0,0,0,1]
    poses=native_drop_candidates(candidate,[0,0,0],planes)
    assert len(poses)==3 and poses[0]==candidate
    assert all(p[:2]==candidate[:2] and p[3:]==candidate[3:] for p in poses)
    assert [p[2] for p in poses]==pytest.approx([.02,.035,.05])
    # The native sampler's own 1-cm lift is included in the ceiling screen.
    candidate[2]=.08
    assert native_drop_candidates(candidate,[0,0,0],planes)==[candidate]


def test_support_envelope_is_only_a_candidate_and_does_not_join_different_shelves():
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import support_envelope_candidate
    patches=[dict(status='measured_patch',lo=[0,0,.01],hi=[.05,.2,.2],floor_z_m=.01),
             dict(status='measured_patch',lo=[.15,0,.011],hi=[.2,.2,.2],floor_z_m=.011)]
    candidate=support_envelope_candidate(np.zeros(3),np.array([.2,.2,.2]),patches)
    assert candidate['status']=='support_hull_candidate' and candidate['physical_validation_required']
    assert candidate['floor_z_m']==.011 and candidate['measured_support_count']==2
    # A box may bridge the gap; the native sampler, not this hull, must prove
    # actual support. Distinct shelf heights must not become a fictional floor.
    assert all(a*.1+b*.1+d<0 for a,b,d in candidate['center_halfspaces_world_xy'])
    patches[1]['floor_z_m']=.05
    assert support_envelope_candidate(np.zeros(3),np.array([.2,.2,.2]),patches) is None


def test_bounded_drop_candidates_do_not_restore_random_sampling_budget():
    import torch,pytest
    from types import SimpleNamespace as NS
    from rummagebench.sim.omnigibson.inside_volume import native_gpu_proposals
    poses=[[1.,2.,3.+dz,0.,0.,0.,1.] for dz in [0.,.02,.04]]
    before=torch.random.get_rng_state().clone()
    assert native_gpu_proposals(NS(_findingbench_layout_candidate=poses),None,None,None,100).shape==(3,7)
    assert torch.equal(before,torch.random.get_rng_state())
    with pytest.raises(ValueError,match='INVALID_EXPERT'):native_gpu_proposals(NS(_findingbench_layout_candidate=poses*2),None,None,None,100)


def test_degenerate_native_floor_repair_preserves_xy_midplane_and_exterior_bbox():
    import numpy as np,pytest,trimesh
    from rummagebench.sim.omnigibson.inside_volume import regularize_native_box_floor
    mesh=trimesh.creation.box(extents=[.50,.29,8e-8]);points=mesh.vertices.copy()
    points[:,2]+=-.1607+points[:,0]*4e-6+points[:,1]*2e-6
    before=points.copy();fixed=regularize_native_box_floor(points,mesh.faces)
    np.testing.assert_array_equal(points,before)
    np.testing.assert_array_equal(fixed[:,:2],points[:,:2])
    assert np.ptp(fixed[:,2])==pytest.approx(.002)
    assert np.mean([fixed[:,2].min(),fixed[:,2].max()])==pytest.approx(np.mean([points[:,2].min(),points[:,2].max()]))
    assert fixed[:,2].min()>-.163 and fixed[:,2].max()<.17
    with pytest.raises(ValueError,match='DEGENERATE'):regularize_native_box_floor(trimesh.creation.box(extents=[.50,.29,.02]).vertices,mesh.faces)
    with pytest.raises(ValueError,match='THICKNESS'):regularize_native_box_floor(points,mesh.faces,thickness_m=-.002)


def test_frozen_contract_checks_provider_and_sdk_loop_consumer(tmp_path):
    import hashlib,json,runpy,importlib.util
    if importlib.util.find_spec('omnigibson') is None:pytest.skip('Native SDK source unavailable')
    root=Path(__file__).resolve().parents[2]
    relative='src/rummagebench/sim/omnigibson/inside_volume.py'
    source=(root/relative).read_text();frozen=tmp_path/relative;frozen.parent.mkdir(parents=True)
    check=runpy.run_path(str(root/'scripts/v2/check_frozen_layout_runtime.py'))['check']
    def install(text):
        frozen.write_text(text)
        (tmp_path/'release.json').write_text(json.dumps(dict(source_sha256={relative:hashlib.sha256(frozen.read_bytes()).hexdigest()})))
    install(source)
    assert check(tmp_path)['candidate_count']==3
    # The provider still accepts all three poses, but the sampler silently
    # discards two. The frozen contract must reject this integration bug.
    install(source.replace('total_attempts = len(_gpu_bank)\n',''))
    with pytest.raises(AssertionError,match='consumer dropped'):check(tmp_path)


def test_surface_layout_keeps_full_footprints_in_real_patches():
    import torch,numpy as np
    from rummagebench.sim.omnigibson.inside_volume import surface_patch_layout
    if not torch.cuda.is_available():pytest.skip('CUDA layout solver required')
    patches=[dict(status='measured_patch',lo=[0.,0.,.8],hi=[.5,.35,1.1],floor_z_m=.8),
             dict(status='measured_patch',lo=[.6,.5,.9],hi=[1.1,.85,1.2],floor_z_m=.9)]
    items={f'bottle{i}':[.12,.12,.2] for i in range(8)}
    solved,plan=surface_patch_layout(items,patches)
    assert solved['status']=='feasible_model' and len(plan)==8
    for p in plan.values():
        xy=np.asarray(p['position_xy']);half=np.asarray(p['footprint_m'])/2;patch=p['support_patch']
        assert np.all(xy-half>=np.asarray(patch['lo'][:2]))
        assert np.all(xy+half<=np.asarray(patch['hi'][:2]))
        assert p['support_height_m']==patch['floor_z_m']
    # A broad outer AABB must not let an oversized object bridge the gap.
    solved,plan=surface_patch_layout({'wide_box':[.6,.4,.2]},patches)
    assert solved['status']=='infeasible_model' and not plan


def test_surface_group_leaves_authored_containers_and_fixed_furniture_in_place():
    from types import SimpleNamespace as N
    from rummagebench.sim.omnigibson.inside_volume import native_surface_group
    placements=[N(entity=x,relation='on_top',receptacle='counter') for x in ['bottle','basket','fixture']]
    placements.append(N(entity='fruit',relation='inside',receptacle='basket'))
    scenario=N(objects=[N(name='fixture',fixed_base=True)],placements=placements)
    assert [p.entity for p in native_surface_group(scenario,'counter')]==['bottle']


def test_surface_group_reserves_bottom_books_in_real_piles():
    from types import SimpleNamespace as N
    from rummagebench.sim.omnigibson.inside_volume import native_surface_group
    scenario=N(objects=[],placements=[N(entity='bottom_book',relation='on_top',receptacle='desk'),
        N(entity='top_book',relation='on_top',receptacle='bottom_book')])
    assert [p.entity for p in native_surface_group(scenario,'desk')]==['bottom_book']
    scenario.goal=[N(receptacle='desk')]
    assert native_surface_group(scenario,'desk')==[]


def test_surface_convex_candidate_is_an_alternative_and_insets_full_footprints():
    import torch,numpy as np
    from rummagebench.sim.omnigibson.inside_volume import surface_patch_layout
    if not torch.cuda.is_available():pytest.skip('CUDA layout solver required')
    # A measured narrow rectangular subpatch cannot hold this box. The convex
    # envelope can propose a pose; native support rays must still validate it.
    patches=[dict(status='measured_patch',lo=[2.,3.,.8],hi=[2.15,3.15,1.1],floor_z_m=.8),
        dict(status='support_hull_candidate',lo=[2.,3.,.8],hi=[3.,4.,1.1],floor_z_m=.8,
             center_halfspaces_world_xy=[[1.,0.,-3.],[-1.,0.,2.],[0.,1.,-4.],[0.,-1.,3.]])]
    solved,plan=surface_patch_layout({'box':[.5,.4,.2]},patches)
    assert solved['status']=='feasible_model'
    pose=plan['box'];xy=np.asarray(pose['position_xy']);half=np.asarray(pose['footprint_m'])/2
    assert pose['support_patch']['status']=='support_hull_candidate'
    assert np.all(xy-half>=np.array([2.,3.])) and np.all(xy+half<=np.array([3.,4.]))
    assert pose['physical_validation_required']


def test_thin_object_membership_uses_lowest_measured_support_not_highest():
    from rummagebench.sim.omnigibson.inside_volume import support_center_lower_bound
    # A 2-mm comb can settle 3 mm below the optimistic probe height.
    # The old highest-ray test would falsely admit it to the native volume.
    assert support_center_lower_bound(.102,.002,{'height_spread_m':.003})==pytest.approx(.100)
    assert support_center_lower_bound(.102,.002,{'measured_height_range_m':[.098,.102]})==pytest.approx(.099)
    # An actual raised support leaves the complete uncertainty interval above
    # the native bottom, without changing the membership predicate.
    assert support_center_lower_bound(.120,.002,{'height_spread_m':.001})>.102


def test_inclined_support_requires_one_measured_plane_and_matching_normals():
    import numpy as np
    from rummagebench.sim.omnigibson.inside_volume import fit_inclined_support
    normal=np.array([.002,.02,1.]);normal/=np.linalg.norm(normal)
    hits=[dict(valid=True,body='/cabinet/base',position=[x,y,.1-.002*x-.02*y],normal=normal.tolist())
          for x in [-.3,0,.3] for y in [-.4,0,.4]]
    result=fit_inclined_support(hits)
    assert result and result['plane_fit_max_residual_m']<1e-10
    assert result['height_spread_m']>.003
    for kind in ['step','foreign','missing','wrong_normal','two_colliders']:
        changed=deepcopy(hits)
        if kind=='step':changed[4]['position'][2]+=.004
        elif kind=='foreign':changed[4]['valid']=False
        elif kind=='missing':changed.pop()
        elif kind=='wrong_normal':changed[4]['normal']=[0,.2,.98]
        else:changed[4]['body']='/cabinet/other_shelf'
        assert fit_inclined_support(changed) is None


def test_collision_obstacles_use_current_native_pose_instead_of_stale_fabric():
    import itertools,numpy as np
    from types import SimpleNamespace as NS
    from scipy.spatial.transform import Rotation
    from rummagebench.sim.omnigibson.inside_volume import current_collision_aabb,reusable_native_contents_plan
    points=np.array(list(itertools.product([-.1,.1],[-.2,.2],[-.3,.3])))
    matrix=np.diag([2.,3.,4.,1.]);matrix[:3,3]=[100.,100.,100.]
    link=NS(collision_boundary_points_local=points,scaled_transform=matrix,
        get_position_orientation=lambda:([1.,2.,3.],Rotation.from_euler('z',90,degrees=True).as_quat()))
    moving=NS(links={'body':link},aabb=([99,99,99],[101,101,101]))
    lo,hi=current_collision_aabb(moving)
    assert np.allclose(lo,[.4,1.8,1.8]) and np.allclose(hi,[1.6,2.2,4.2])
    # A pending item fits the obsolete graphic-frame bounds but intersects
    # the settled collision shape. Reuse must be rejected before native work.
    pending=NS(native_bbox=np.array([.1,.1,.1]),scale=np.ones(3))
    rec=NS(n_joints=0);objects={'cabinet':rec,'placed':moving,'pending':pending}
    backend=NS(_env=NS(scene=NS(object_registry=lambda _,n:objects[n])))
    context=dict(pending_plan_joints=[],pending_plan={'pending':dict(bbox_center_world=[1,2,3],
        solver_pose=dict(footprint_m=[.1,.1]))})
    assert reusable_native_contents_plan(backend,'cabinet',context,[NS(entity='pending')],{'placed'}) is None
    # Conservative fallback for a lightweight object without collision links.
    assert np.allclose(current_collision_aabb(NS(aabb=([0,0,0],[1,1,1]))),[[0,0,0],[1,1,1]])


def test_dense_authoring_spacing_keeps_real_dimensions_and_fixed_minimum():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rummagebench.sim.omnigibson.inside_volume import solve_authoring_layout
    items={'part':[.1,.1,.05]}
    result=solve_authoring_layout(items,{'bin':[.12,.12,.1]},margin_m=.015,time_limit_s=1.)
    assert result['status']=='feasible_model' and result['margin_m']==.005
    pose=result['poses']['part'];low=pose['lower_xy_m'];size=pose['footprint_m']
    assert all(size[i]>=items['part'][i] for i in range(2))
    assert all(low[i]>=.005 and low[i]+size[i]<=.115+1e-9 for i in range(2))
    assert result['authoring_spacing_fallback']['native_physical_checks'].startswith('unchanged')
    # Five millimetres is a fixed lower bound, not an adaptive acceptance bypass.
    assert solve_authoring_layout(items,{'bin':[.105,.105,.1]},margin_m=.015,time_limit_s=1.)['status']!='feasible_model'


def test_tensorized_relation_reads_do_not_reuse_pose_only_result_cache():
    from rummagebench.sim.omnigibson.inside_volume import fresh_relative_state_getter
    class State:
        def __init__(self):self.cache={};self.native_value=True;self.initialized=True
        def clear_cache(self):self.cache.clear()
        def native_get(self, other):
            assert self.initialized
            return self.cache.setdefault(other,self.native_value)
    state=State();state.cache['surface']=False
    get=fresh_relative_state_getter(State.native_get)
    assert get(state,'surface') is True
    state.native_value=False
    assert get(state,'surface') is False
    state.initialized=False
    with pytest.raises(AssertionError):get(state,'surface')


def test_native_inside_reference_retains_both_aabb_and_actual_volume_gates(monkeypatch):
    import sys,types,torch
    from rummagebench.sim.omnigibson.inside_volume import native_inside_reference_getter
    for name in ['omnigibson','omnigibson.utils','omnigibson.object_states']:
        package=types.ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    monkeypatch.setitem(sys.modules,'omnigibson.utils.constants',SimpleNamespace(
        PrimType=SimpleNamespace(CLOTH='cloth')))
    monkeypatch.setitem(sys.modules,'omnigibson.object_states.contains',SimpleNamespace(
        m=SimpleNamespace(CONTAINER_META_LINK_TYPES={'fillable'})))
    center=torch.tensor([.5,.5,.5]);obj=SimpleNamespace(aabb_center=center)
    queries=[]
    def native_volume(points):
        queries.append(points.clone());return (points[:,0]<.6)&(points[:,1]>.3)
    link=SimpleNamespace(is_meta_link=True,meta_link_type='fillable',check_points_in_volume=native_volume)
    container=SimpleNamespace(prim_type='rigid',aabb=(torch.zeros(3),torch.ones(3)),links={'native':link})
    state=SimpleNamespace(obj=obj)
    def original(_,other):raise ValueError('native cloth error')
    get=native_inside_reference_getter(original)
    assert get(state,container) is True
    center[0]=.7
    assert get(state,container) is False  # Inside AABB alone is insufficient.
    count=len(queries);center[0]=2.
    assert get(state,container) is False and len(queries)==count
    center[0]=float('nan')
    with pytest.raises(ValueError,match='NONFINITE'):get(state,container)
    container.prim_type='cloth'
    with pytest.raises(ValueError,match='native cloth'):get(state,container)
def test_declared_door_state_requests_native_open_without_mutating_taxonomy():
    from types import SimpleNamespace
    from rummagebench.sim.omnigibson.backend import declared_open_abilities
    defaults={'fillable':{},'coldSource':{'temperature':-18.}}
    updated=declared_open_abilities(SimpleNamespace(open=False),defaults)
    assert updated==dict(defaults,openable={})
    updated['coldSource']['temperature']=0.
    assert defaults['coldSource']['temperature']==-18.
    assert 'openable' not in defaults
    assert declared_open_abilities(None,defaults) is None
    assert declared_open_abilities(SimpleNamespace(open=None),defaults) is None
    assert declared_open_abilities(SimpleNamespace(open=True),{'openable':{}}) is None


