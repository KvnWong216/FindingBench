"""Exact native-region selection; unit fixtures never count as scene data."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from rummagebench.sim.omnigibson.inside_volume import selected_volume_source


@pytest.mark.local_assets
def test_actual_sdk_keeps_every_inside_check_except_region_selection():
    path=Path(__file__).resolve().parents[3]/'BEHAVIOR-1K/OmniGibson/omnigibson/object_states/inside.py'
    if not path.is_file():pytest.skip('installed native Inside source unavailable')
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
    monkeypatch.setattr(module,'_sampler',lambda method:(sampler,'unit_hash'))
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
