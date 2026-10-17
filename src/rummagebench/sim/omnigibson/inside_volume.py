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


@lru_cache(maxsize=4)
def _sampler(method):
    source=inspect.getsource(method)
    adapted=selected_volume_source(source)
    namespace=dict(vars(inspect.getmodule(method)))
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


def sample_selected_volume(obj,receptacle,link_name):
    import omnigibson as og
    from omnigibson.object_states import Inside
    state=obj.states.get(Inside)
    volume=receptacle.links.get(link_name)
    if state is None or volume is None or not volume.is_meta_link:
        raise ValueError('ACTUAL_NATIVE_INSIDE_STATE_AND_FILLABLE_LINK_REQUIRED')
    sampler,digest=_sampler(type(state)._set_value)
    if not state._initialized:raise ValueError('NATIVE_INSIDE_STATE_NOT_INITIALIZED')
    before=og.sim.dump_state(serialized=False)
    try:
        # Same wrapper lifecycle as BaseObjectState.set_value().
        state.clear_cache()
        # Native links are keyed by short names, but RigidPrim.name is
        # qualified as "<object>:<link>" in the installed SDK.
        success=bool(sampler(state,receptacle,True,use_trav_map=False,_findingbench_volume_link=volume.name))
        obj.state_updated()
        if success and selected_volume_contains(obj,receptacle,link_name):
            return True,{'sampler_source_sha256':digest,'fillable_link':link_name,
                         'checks':'all_original_Inside_checks_and_selected_native_volume_membership'}
    except Exception:
        og.sim.load_state(before,serialized=False)
        raise
    og.sim.load_state(before,serialized=False)
    return False,{'sampler_source_sha256':digest,'fillable_link':link_name,'checks':'placement_rejected'}
