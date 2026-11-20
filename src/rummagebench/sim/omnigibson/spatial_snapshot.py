"""Portable snapshot of OG's existing cached spatial predicates.

Sleeping bodies retain their last valid kinematic predicate even when PhysX
stops reporting contact impulses. Those predicates are part of observable OG
state, but are absent from its flat object-state serialization. Copy existing
cache entries, with object references encoded as prim paths; never infer a new
relation or write contact matrices.
"""
from copy import deepcopy
import math
import numpy as np

STATES = frozenset(('OnTop', 'Inside', 'Touching'))


def _objects(scene):
    objects = {o.prim_path:o for o in scene.objects}
    if len(objects) != len(scene.objects): raise ValueError('SPATIAL_SNAPSHOT_DUPLICATE_PATH')
    return objects


def _inventory(objects):
    return [dict(path=p,category=getattr(o,'category',None),model=getattr(o,'model',None))
            for p,o in sorted(objects.items())]


def _encode(value, refs):
    if id(value) in refs: return ('object', refs[id(value)])
    if isinstance(value, dict): return ('dict', [(_encode(k,refs),_encode(v,refs)) for k,v in value.items()])
    if isinstance(value, (tuple,list)): return ('tuple' if isinstance(value,tuple) else 'list', [_encode(v,refs) for v in value])
    if value is None or isinstance(value,(bool,int,float,str)):
        if isinstance(value,float) and not math.isfinite(value): raise ValueError('NONFINITE_SPATIAL_CACHE')
        return ('scalar',value)
    if isinstance(value,np.ndarray) or hasattr(value,'detach'):
        array=value.detach().cpu().numpy() if hasattr(value,'detach') else value
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():raise ValueError('INVALID_SPATIAL_CACHE_ARRAY')
        return ('tensor',value.detach().cpu().clone()) if hasattr(value,'detach') else ('array',value.copy())
    raise ValueError('UNSUPPORTED_SPATIAL_CACHE_VALUE:'+type(value).__name__)


def _decode(value, objects):
    kind, data = value
    if kind=='object': return objects[data]
    if kind=='dict': return {_decode(k,objects):_decode(v,objects) for k,v in data}
    if kind in ('tuple','list'):
        items=[_decode(v,objects) for v in data]
        return tuple(items) if kind=='tuple' else items
    if kind in ('scalar','array','tensor'): return deepcopy(data)
    raise ValueError('UNSUPPORTED_SPATIAL_CACHE_TAG')


def capture(scene, step):
    objects = _objects(scene);refs={id(o):path for path,o in objects.items()}
    records=[]
    for path,obj in objects.items():
        for cls,state in obj.states.items():
            if cls.__name__ not in STATES: continue
            for args,record in state.cache.items():
                if len(args)!=1 or id(args[0]) not in refs:
                    raise ValueError('UNSUPPORTED_SPATIAL_PREDICATE_ARGUMENTS')
                record=deepcopy({k:v for k,v in record.items() if k not in ('info','t')}) | {
                    'info':state.cache[args]['info'], 'age':int(step-state.cache[args]['t'])}
                records.append(dict(owner=path,state=cls.__name__,other=refs[id(args[0])],
                                    record=_encode(record,refs)))
    return dict(version=1, inventory=_inventory(objects),records=records)


def restore(scene, step, saved):
    objects=_objects(scene)
    if saved.get('version')!=1 or saved.get('inventory')!=_inventory(objects):
        raise ValueError('SPATIAL_SNAPSHOT_INVENTORY_MISMATCH')
    prepared=[];seen=set()
    for row in saved['records']:
        key=(row['owner'],row['state'],row['other'])
        if key in seen or row['state'] not in STATES: raise ValueError('SPATIAL_SNAPSHOT_DUPLICATE_OR_UNKNOWN_STATE')
        seen.add(key)
        states={cls.__name__:state for cls,state in objects[row['owner']].states.items()}
        record=_decode(row['record'],objects);age=record.pop('age')
        if type(age) is not int or age<0: raise ValueError('SPATIAL_SNAPSHOT_INVALID_AGE')
        record['t']=int(step-age)
        prepared.append((states[row['state']],(objects[row['other']],),record))
    # Validate everything before changing any native cache.
    for obj in objects.values():
        for cls,state in obj.states.items():
            if cls.__name__ in STATES: state.clear_cache()
    for state,args,record in prepared:
        state._cache[args]=record
