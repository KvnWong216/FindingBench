"""Oracle / debug entity grounding (MVP mode).

The agent names an entity directly; the backend resolves the name. This mode
exists for scripted tests, oracle baselines, unit tests and benchmark debugging.
Pixel grounding will replace this at the evaluated-agent boundary later without
changing the action schema or validators.
"""

from __future__ import annotations

from rummagebench.core.errors import UnresolvableTargetError
from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import TargetKind, TargetRef
from rummagebench.sim.base import ResolvedTarget, SimBackend


class OracleEntityGrounding:
    """Resolves PLACE refs against scenario anchors and ENTITY refs against the backend."""

    def __init__(self, backend: SimBackend, scenario: ScenarioSpec):
        self._backend = backend
        self._scenario = scenario

    def resolve(self, target: TargetRef) -> ResolvedTarget:
        if target.type == TargetKind.PLACE:
            anchor = self._scenario.anchors.get(target.value or "")
            if anchor is None:
                raise UnresolvableTargetError(f"unknown place {target.value!r}")
            return ResolvedTarget(
                kind=TargetKind.PLACE, place=target.value, anchor=anchor
            )
        if target.type == TargetKind.ENTITY:
            resolved = self._backend.resolve_entity(target.value or "")
            return resolved
        raise UnresolvableTargetError(
            f"target kind {target.type!r} is not supported by the active grounding mode"
        )
