"""Evaluator-only placement helpers for task-authoring probes/certification.

Thin wrappers that expose the builder's OWN placement + verification
routines (OmniGibsonBackend._place / _verify_relation /
_containing_moving_link) by entity name, so authoring code never imports
omnigibson (README architecture rule 2).
"""
from __future__ import annotations

from typing import Optional

from rummagebench.sim.omnigibson.entity_resolver import resolve_object


def _obj(backend, name: str):
    return resolve_object(backend._env.scene, name)


def place_entity(backend, entity: str, receptacle: str, relation: str,
                 link: Optional[str]) -> bool:
    """Builder placement (receptacle must already be open for ``inside``).
    Returns the sampler / drop result; verify separately after settling."""
    ok = bool(backend._place(_obj(backend, entity), _obj(backend, receptacle), relation, link))
    backend._collision_body_cache = None
    return ok


def relation_holds(backend, entity: str, receptacle: str, relation: str) -> bool:
    from omnigibson.object_states import Inside, OnTop

    return bool(backend._verify_relation(_obj(backend, entity), _obj(backend, receptacle),
                                         relation, Inside, OnTop))


def containing_link(backend, receptacle: str, entity: str) -> Optional[str]:
    """Smallest moving link of ``receptacle`` whose AABB holds the entity
    centre (the builder's link-membership test), or None."""
    rec = _obj(backend, receptacle)
    link = backend._containing_moving_link(rec, _obj(backend, entity))
    if link is None:
        return None
    # the links-dict KEY (what scenarios name), not the prim name
    # ("<object>:<link>")
    return next((k for k, v in (rec.links or {}).items() if v is link), link.name)


def link_names(backend, entity: str) -> list[str]:
    return sorted((_obj(backend, entity).links or {}).keys())
