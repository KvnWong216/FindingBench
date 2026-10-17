"""Synthetic native state-cache fixtures; not physics evidence."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from rummagebench.sim.omnigibson import spatial_snapshot as s


class OnTop: pass
class Obj:
    def __init__(self,path): self.prim_path=path;self.states={}
class State:
    def __init__(self): self._cache={}
    @property
    def cache(self): return self._cache
    def clear_cache(self): self._cache={}


def test_native_predicate_history_roundtrips_without_live_object_references():
    a,b=Obj('/a'),Obj('/b');state=State();a.states={OnTop:state}
    state._cache[(b,)]=dict(value=True,t=10,info={a:dict(p=(np.zeros(3),np.array([0,0,0,1])),q=np.array([]))})
    saved=s.capture(NS(objects=[a,b]),12)
    x,y=Obj('/a'),Obj('/b');new=State();x.states={OnTop:new}
    s.restore(NS(objects=[x,y]),100,saved)
    rec=new.cache[(y,)]
    assert rec['value'] is True and rec['t']==98 and x in rec['info'] and a not in rec['info']
    rec['info'][x]['p'][0][0]=2
    assert state.cache[(b,)]['info'][a]['p'][0][0]==0


def test_unknown_world_or_duplicate_predicate_rejects_before_cache_mutation():
    a,b=Obj('/a'),Obj('/b');state=State();a.states={OnTop:state}
    state._cache[(b,)]=dict(value=False,t=1,info={})
    scene=NS(objects=[a,b]);saved=s.capture(scene,1)
    with pytest.raises(ValueError):s.restore(NS(objects=[a]),2,saved)
    saved['records']*=2
    with pytest.raises(ValueError):s.restore(scene,2,saved)
    assert state.cache[(b,)]['value'] is False


def test_same_path_different_asset_cannot_reuse_cached_predicate():
    a=Obj('/a');a.model='one';scene=NS(objects=[a]);saved=s.capture(scene,1)
    a.model='two'
    with pytest.raises(ValueError,match='INVENTORY'):s.restore(scene,1,saved)


def test_nonfinite_cached_geometry_rejects():
    a,b=Obj('/a'),Obj('/b');state=State();a.states={OnTop:state}
    state._cache[(b,)]=dict(value=True,t=1,info={a:np.array([float('nan')])})
    with pytest.raises(ValueError,match='CACHE_ARRAY'):s.capture(NS(objects=[a,b]),1)
