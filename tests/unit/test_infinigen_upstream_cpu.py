"""Optional real pinned SDK contract checks; no generated data or GPU imports."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from rummagebench.authoring.tasks.infinigen import CPU_ENV, cpu_sandbox_argv, source_audit


@pytest.mark.local_assets
def test_real_sink_uses_declared_parameters_and_upstream_front_relation(tmp_path):
    root=os.environ.get('FINDINGBENCH_INFINIGEN_ROOT');python=os.environ.get('FINDINGBENCH_INFINIGEN_CPU_PYTHON')
    if not root or not python:pytest.skip('real pinned SDK environment required')
    source_audit(Path(root))
    script='''
import json,numpy as np
from infinigen.assets.objects import shelves,table_decorations
from infinigen_examples.constraints import home,util as cu
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.tags import Semantics,Subpart
from rummagebench.authoring.tasks.infinigen import apply_source_sink_parameters,apply_sink_front_margin,apply_closed_kitchen_cabinets
apply_source_sink_parameters(table_decorations.SinkFactory,{'Width':[.4,.45],'Margin':[.02,.03]})
a=table_decorations.SinkFactory(1234);b=table_decorations.SinkFactory(1234)
assert a.params['Width']==b.params['Width'] and .4<=a.params['Width']<=.45
assert .02<=a.params['Margin']<=.03
graph=home.home_furniture_constraints();original=dict(graph.constraints)
apply_sink_front_margin(graph,.18,cl=cl,cu=cu,Semantics=Semantics,Subpart=Subpart,
                        shelves=shelves,table_decorations=table_decorations)
assert all(graph.constraints[k] is v for k,v in original.items() if k!='kitchen_sink')
relations=[value for node in graph.constraints['kitchen_sink'].traverse()
           for value in vars(node).values() if isinstance(value,cl.CoPlanar)]
assert relations and all(rel.margin==.18 and rel.rev_normal for rel in relations)
assert 'findingbench_complete_kitchen_sink' in graph.constraints
apply_closed_kitchen_cabinets(shelves.KitchenCabinetFactory)
closed=shelves.KitchenCabinetFactory(421,dimensions=(.3,1.1,.7),drawer_only=False)
assert closed.drawer_only is True
components=closed.create_cabinet_components(i=0,drawer_only=closed.drawer_only)
assert components and all(c[1] in ('door','drawer') for c in components)
try:apply_source_sink_parameters(table_decorations.SinkFactory,{'Width':[.1,.2]})
except ValueError:pass
else:raise AssertionError('outside upstream parameter domain accepted')
print(json.dumps({'real_factory_parameters_only':True,'Width':a.params['Width'],'Margin':a.params['Margin']}))
'''
    repo=Path(__file__).resolve().parents[2]
    env=dict(os.environ,**CPU_ENV,PYTHONPATH=os.pathsep.join([str(repo/'src'),root]))
    result=subprocess.run(cpu_sandbox_argv([python,'-c',script],tmp_path),cwd=root,env=env,
                          capture_output=True,text=True,timeout=60,check=False)
    assert result.returncode==0,result.stderr[-5000:]


@pytest.mark.local_assets
def test_real_cabinet_factory_consumes_declared_dimensions_with_its_own_seed(tmp_path):
    root=os.environ.get('FINDINGBENCH_INFINIGEN_ROOT');python=os.environ.get('FINDINGBENCH_INFINIGEN_CPU_PYTHON')
    if not root or not python:pytest.skip('real pinned SDK environment not configured')
    source_audit(Path(root))
    script='''
import json
from infinigen.assets.objects.shelves import KitchenCabinetFactory
from rummagebench.authoring.tasks.infinigen import apply_source_storage_dimensions
bounds=[[.25,.35],[1.,1.2],[.5,.9]]
apply_source_storage_dimensions(KitchenCabinetFactory,bounds)
a=KitchenCabinetFactory(1234);b=KitchenCabinetFactory(1234);c=KitchenCabinetFactory(1235)
assert a.dimensions==b.dimensions and a.dimensions!=c.dimensions
assert all(lo<=x<=hi for x,(lo,hi) in zip(a.dimensions,bounds))
assert a.frame_params['Dimensions']==a.dimensions
placeholder=a.create_placeholder()
assert 1.<=placeholder.dimensions.y<1.4
explicit=KitchenCabinetFactory(1234,dimensions=(.7,3.8,.9),drawer_only=True)
assert explicit.dimensions==(.7,3.8,.9)
# KitchenSpace's explicit nested dimensions must remain untouched. They are
# separately audited, not falsely counted as standalone state candidates.
print(json.dumps({'real_factory':True,'dimensions':list(a.dimensions),'unit_placeholder_only':True}))
'''
    repo=Path(__file__).resolve().parents[2]
    env=dict(os.environ,**CPU_ENV,PYTHONPATH=os.pathsep.join([str(repo/'src'),root]))
    result=subprocess.run(cpu_sandbox_argv([python,'-c',script],tmp_path),cwd=root,env=env,
                          capture_output=True,text=True,timeout=60,check=False)
    assert result.returncode==0,result.stderr[-3000:]
    report=json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
    assert report['real_factory'] is True


@pytest.mark.local_assets
def test_populated_storage_repair_rechecks_meshes_and_protects_inventory(tmp_path):
    """Real Blender/FCL evaluation; repair callbacks are UNIT FIXTURES ONLY."""
    root=os.environ.get('FINDINGBENCH_INFINIGEN_ROOT')
    python=os.environ.get('FINDINGBENCH_INFINIGEN_CPU_PYTHON')
    if not root or not python:pytest.skip('real pinned SDK environment not configured')
    source_audit(Path(root))
    script='''
import bpy,json,numpy as np
from types import SimpleNamespace
from infinigen_examples.constraints import home,util as cu
from infinigen.core.constraints import constraint_language as cl,reasoning as r,usage_lookup
from infinigen.core.constraints.example_solver.state_def import State,ObjectState,RelationState
from infinigen.core.constraints.constraint_language.util import sync_trimesh
from infinigen.core.tags import Semantics,FromGenerator,SpecificObject
from infinigen.assets.objects import shelves
from rummagebench.authoring.tasks.infinigen import apply_search_constraints,refine_populated_storage
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def cube(name,x):
    bpy.ops.mesh.primitive_cube_add(size=.2,location=(x,0,0))
    obj=bpy.context.object;obj.name=name
    return obj
room=ObjectState(obj=cube('unit_room',4),tags={Semantics.Room,Semantics.Kitchen,SpecificObject('room')})
graph=apply_search_constraints(home.home_furniture_constraints(),2,.25,cl=cl,cu=cu,Semantics=Semantics,shelves=shelves)
tags=set(usage_lookup.usages_of_factory(shelves.KitchenCabinetFactory))
tags.update({Semantics.Object,FromGenerator(shelves.KitchenCabinetFactory)})
objects={'room':room}
for key,x in [('cab_a',0.),('cab_b',.3)]:
    objects[key]=ObjectState(obj=cube(key,x),tags=set(tags)|{SpecificObject(key)},
        relations=[RelationState(cu.on_floor,'room'),RelationState(cu.against_wall,'room')])
state=State(objs=objects);solver=SimpleNamespace(state=state)
domain=r.Domain({Semantics.Object,FromGenerator(shelves.KitchenCabinetFactory)})
assignments={cu.variable_room:'room'};calls=[]
def repair(solver,problem,domain,assignments,**kwargs):
    calls.append(kwargs)
    assert kwargs['restrict_moves']==['plane_change','reinit_pose','translate','rotate']
    state.objs['cab_b'].obj.location.x=1.
    bpy.context.view_layer.update();sync_trimesh(state.trimesh_scene,'cab_b')
result=refine_populated_storage(solver,graph,domain,assignments,solve_objects=repair,n_steps=7)
assert result['status']=='passed' and result['attempts']==1 and len(calls)==1
assert result['violations_before']['findingbench_storage_separation']>0
assert result['violations_after']['findingbench_storage_separation']==0
assert result['physical_acceptance'] is False and calls[0]['n_steps']==7
assert refine_populated_storage(solver,graph,domain,assignments,solve_objects=repair)['attempts']==0
assert len(calls)==1
def reset_gap():
    state.objs['cab_b'].obj.location.x=.3
    bpy.context.view_layer.update();sync_trimesh(state.trimesh_scene,'cab_b')
reset_gap()
failed=refine_populated_storage(solver,graph,domain,assignments,solve_objects=lambda *a,**k:None)
assert failed['status']=='failed' and failed['attempts']==1
def disturb(*a,**k):room.obj.location.x+=.1
try:refine_populated_storage(solver,graph,domain,assignments,solve_objects=disturb)
except ValueError as exc:assert str(exc)=='POPULATED_STORAGE_REPAIR_MOVED_NON_CANDIDATE'
else:raise AssertionError('non-candidate movement accepted')
def remove(*a,**k):del state.objs['cab_b']
try:refine_populated_storage(solver,graph,domain,assignments,solve_objects=remove)
except ValueError as exc:assert str(exc)=='POPULATED_STORAGE_REPAIR_CHANGED_OBJECT_INVENTORY'
else:raise AssertionError('candidate removal accepted')
assert room.active is True
print(json.dumps({'unit_fixtures_only':True,'expanded_geometry_rechecked':True}))
'''
    repo=Path(__file__).resolve().parents[2]
    env=dict(os.environ,**CPU_ENV,PYTHONPATH=os.pathsep.join([str(repo/'src'),root]))
    result=subprocess.run(cpu_sandbox_argv([python,'-c',script],tmp_path),cwd=root,env=env,
                          capture_output=True,text=True,timeout=60,check=False)
    assert result.returncode==0,result.stderr[-5000:]
    report=json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
    assert report=={'unit_fixtures_only':True,'expanded_geometry_rechecked':True}


@pytest.mark.local_assets
def test_pinned_export_compatibility_preserves_all_geometry_and_bake_operations():
    import ast
    from rummagebench.authoring.tasks.infinigen import export_compatibility_source
    root=os.environ.get('FINDINGBENCH_INFINIGEN_ROOT')
    if not root:pytest.skip('set FINDINGBENCH_INFINIGEN_ROOT for real pinned exporter source check')
    source_audit(Path(root))
    source=(Path(root)/'infinigen/tools/export.py').read_text()
    adapted,count=export_compatibility_source(source,modern_cycles=True)
    original=ast.parse(source)
    # Compare all non-tile statements; four obsolete performance hints are
    # the only allowable change in the real pinned source. No bpy import.
    class WithoutTiles(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and ast.unparse(node.targets[0]) in (
                'bpy.context.scene.cycles.tile_x','bpy.context.scene.cycles.tile_y'):return None
            return self.generic_visit(node)
    assert count==4 and ast.dump(WithoutTiles().visit(original))==ast.dump(ast.parse(adapted))
    compile(adapted,'pinned_export_CPU_compile_only','exec')


@pytest.mark.local_assets
def test_real_search_constraint_relevance_in_all_upstream_greedy_stages(tmp_path):
    root = os.environ.get("FINDINGBENCH_INFINIGEN_ROOT")
    python = os.environ.get("FINDINGBENCH_INFINIGEN_CPU_PYTHON")
    if not root or not python:
        pytest.skip("set FINDINGBENCH_INFINIGEN_ROOT and FINDINGBENCH_INFINIGEN_CPU_PYTHON for real SDK checks")
    source_audit(Path(root))
    script = '''
import json,os
import infinigen.terrain as terrain
from infinigen.terrain.core import Terrain
terrain.Terrain=Terrain
from infinigen_examples.constraints import home,util as cu
from infinigen_examples.generate_indoors import default_greedy_stages
from infinigen.core.constraints import constraint_language as cl,reasoning as r
from infinigen.core.constraints.evaluator.evaluate import relevant
from infinigen.core.tags import Semantics,SpecificObject,FromGenerator
from infinigen.assets.objects import shelves
from rummagebench.authoring.tasks.infinigen import apply_search_constraints
stages=default_greedy_stages()
room=r.Domain({Semantics.Room,-Semantics.Object,Semantics.Kitchen,SpecificObject('kitchen_0/0')})
reports={}
assert len(os.sched_getaffinity(0))<=2
for count in (2,4):
    graph=apply_search_constraints(home.home_furniture_constraints(),count,.25,
                                  cl=cl,cu=cu,Semantics=Semantics,shelves=shelves)
    whole=graph.constraints['findingbench_complete_storage_count']
    assert len(r.constraint_bounds(whole))==0
    bounds=r.constraint_bounds(graph.constraints['findingbench_cabinet_count'])
    assert bounds and all(b.low==count and b.high==count for b in bounds)
    assert not r.constraint_bounds(graph.constraints['storage'])
    assert not r.constraint_bounds(graph.constraints['findingbench_storage_count'])
    cabinet_stage=r.substitute_all(stages['on_floor_and_wall'],{cu.variable_room:room}).with_tags({FromGenerator(shelves.KitchenCabinetFactory)})
    assert relevant(graph.constraints['findingbench_storage_separation'],cabinet_stage)
    reports[count]={}
    for name in ('on_floor_and_wall','on_floor_freestanding','on_wall','on_ceiling'):
        stage=r.substitute_all(stages[name],{cu.variable_room:room})
        reports[count][name]=relevant(whole,stage)
        assert reports[count][name]
print(json.dumps({'real_sdk_only':True,'relevance':reports}))
'''
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, **CPU_ENV, PYTHONPATH=os.pathsep.join([str(repo/"src"),root]))
    result = subprocess.run(cpu_sandbox_argv([python,"-c",script],tmp_path),cwd=root,env=env,
                            capture_output=True,text=True,timeout=60,check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
    assert report['real_sdk_only'] is True


@pytest.mark.local_assets
def test_optional_spacing_guards_use_real_sdk_and_keep_existing_pair_rules(tmp_path):
    """Blender cubes are explicitly unit fixtures, never benchmark data."""
    root=os.environ.get('FINDINGBENCH_INFINIGEN_ROOT')
    python=os.environ.get('FINDINGBENCH_INFINIGEN_CPU_PYTHON')
    if not root or not python:
        pytest.skip('real pinned SDK environment not configured')
    script='''
import bpy, json, numpy as np, trimesh
from infinigen_examples.constraints import home,util as cu
from infinigen.core.constraints import constraint_language as cl,usage_lookup
from infinigen.core.constraints.evaluator.evaluate import evaluate_problem
from infinigen.core.constraints.example_solver.state_def import State,ObjectState,RelationState
from infinigen.core.constraints.example_solver.geometry.parse_scene import add_to_scene
from infinigen.core.constraints.constraint_language.util import sync_trimesh
from infinigen.core.tags import Semantics,FromGenerator,SpecificObject
from infinigen.assets.objects import shelves,elements
from rummagebench.authoring.tasks.infinigen import apply_optional_geometry_guards
graph=home.home_furniture_constraints()
raw=apply_optional_geometry_guards(graph,cl=cl,cu=cu,Semantics=Semantics,shelves=shelves,elements=elements)
def cube(name,x):
    bpy.ops.mesh.primitive_cube_add(size=.1,location=(x,0,0))
    obj=bpy.context.object;obj.name=name
    return obj
state=State(trimesh_scene=trimesh.Scene())
state.objs['room']=ObjectState(obj=cube('unit_room',0),tags={Semantics.Room,Semantics.Kitchen,SpecificObject('room')})
def violation(rule,original=False):
    expr=(raw if original else graph.constraints)[rule]
    return evaluate_problem(cl.Problem({rule:expr},{}),state,enable_loss=False).violations[rule]
for rule,factory in [('rugs',elements.RugFactory),('lighting',None)]:
    assert violation(rule)==0 and violation(rule,True)>0
    names=[]
    for i,x in enumerate((0.,.3)):
        name=f'unit_{rule}_{i}'; obj=cube(name,x); names.append(name)
        tags={Semantics.Object,SpecificObject(name)}
        if factory:tags.update(usage_lookup.usages_of_factory(factory));tags.add(FromGenerator(factory))
        else:tags.add(Semantics.Lighting)
        state.objs[name]=ObjectState(obj=obj,tags=tags,relations=[RelationState(cu.on_floor,'room')])
        bpy.context.view_layer.update()
        add_to_scene(state.trimesh_scene,obj)
        if i==0:assert violation(rule)==0 and violation(rule,True)>0
    assert violation(rule)>0 and violation(rule)==violation(rule,True)
    state.objs[names[1]].obj.location.x=2.
    bpy.context.view_layer.update()
    sync_trimesh(state.trimesh_scene,names[1])
    assert violation(rule)==0 and violation(rule,True)==0
    for name in names:del state.objs[name]
# A wall counter without an optional island must not invent a .7m violation.
name='unit_counter';tags=set(usage_lookup.usages_of_factory(shelves.KitchenSpaceFactory))
tags.update({Semantics.Object,FromGenerator(shelves.KitchenSpaceFactory),SpecificObject(name)})
state.objs[name]=ObjectState(obj=cube(name,0),tags=tags,
    relations=[RelationState(cu.on_floor,'room'),RelationState(cu.against_wall,'room')])
assert violation('kitchen_island_placement')==0
assert violation('kitchen_island_placement',True)==.7
print(json.dumps({'unit_fixtures_only':True,'optional_domain_guards_verified':True}))
'''
    repo=Path(__file__).resolve().parents[2]
    env=dict(os.environ,**CPU_ENV,PYTHONPATH=os.pathsep.join([str(repo/'src'),root]))
    result=subprocess.run(cpu_sandbox_argv([python,'-c',script],tmp_path),cwd=root,env=env,
                          capture_output=True,text=True,timeout=60,check=False)
    assert result.returncode==0,result.stderr[-4000:]
