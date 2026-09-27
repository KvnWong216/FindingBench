"""Entity resolution and privileged EntityInfo construction.

EntityInfo is evaluator-side metadata (category, openability, fixed base,
rooms). It feeds validators; it is never placed into an Observation.
"""

from __future__ import annotations

from rummagebench.core.errors import UnresolvableTargetError
from rummagebench.sim.base import EntityInfo


def resolve_object(scene, name: str):
    """Look up a scene object by its unique name (Registry unique key 'name')."""
    obj = scene.object_registry("name", name)
    if obj is None:
        raise UnresolvableTargetError(f"no entity named {name!r} in the scene")
    return obj


def build_entity_info(obj) -> EntityInfo:
    abilities = obj.abilities or {}
    openable = "openable" in abilities
    fixed_base = bool(getattr(obj, "fixed_base", False))
    is_receptacle = "fillable" in abilities or "inside" in abilities
    return EntityInfo(
        name=obj.name,
        category=obj.category,
        model=getattr(obj, "model", None),
        fixed_base=fixed_base,
        openable=openable,
        graspable=not fixed_base,
        is_receptacle=is_receptacle,
        in_rooms=list(getattr(obj, "in_rooms", None) or []),
    )
