"""Navigation target providers (prompt §18): NAV stays a perfect executor.

Navigation is a global skill over build-time-verified targets. No nav-mesh
planning, no locomotion control, no execution failure source. This module
formalizes the target-provider interface: today only NamedAnchorProvider is
implemented (scenario YAML anchors); RoomProvider / ObjectVicinityProvider /
ContinuousPoseProvider are planned extensions and deliberately NOT built in
this round.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rummagebench.core.scenario import AnchorSpec


@runtime_checkable
class NavigationTargetProvider(Protocol):
    """Grounds a semantic NAV place to a verified world-frame anchor pose."""

    def get_anchor(self, place: str) -> AnchorSpec | None: ...

    def places(self) -> list[str]: ...


class NamedAnchorProvider:
    """Scenario-YAML named anchors (build-time verified by pick_anchors.py)."""

    def __init__(self, anchors: dict[str, AnchorSpec]):
        self._anchors = anchors

    def get_anchor(self, place: str) -> AnchorSpec | None:
        return self._anchors.get(place)

    def places(self) -> list[str]:
        return list(self._anchors.keys())
