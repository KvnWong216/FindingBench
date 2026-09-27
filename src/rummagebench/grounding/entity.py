"""Oracle / debug entity grounding (MVP mode).

The agent names an entity directly; the backend resolves the name. This mode
exists for scripted tests, oracle baselines, unit tests and benchmark debugging.
Pixel grounding will replace this at the evaluated-agent boundary later without
changing the action schema or validators.

NAV places resolve through a NavigationTargetProvider (NamedAnchorProvider
today; see core/navigation.py).
"""

from __future__ import annotations

from rummagebench.core.errors import UnresolvableTargetError
from rummagebench.core.navigation import NamedAnchorProvider
from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import TargetKind, TargetRef
from rummagebench.sim.base import ResolvedTarget, SimBackend


class OracleEntityGrounding:
    """Resolves PLACE refs through the nav provider and ENTITY refs against the backend."""

    def __init__(self, backend: SimBackend, scenario: ScenarioSpec, nav_provider=None):
        self._backend = backend
        self._scenario = scenario
        self._nav_provider = nav_provider or NamedAnchorProvider(scenario.anchors)

    def resolve(self, target: TargetRef) -> ResolvedTarget:
        if target.type == TargetKind.PLACE:
            anchor = self._nav_provider.get_anchor(target.value or "")
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
