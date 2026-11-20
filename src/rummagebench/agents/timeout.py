"""TimeoutAgent: performs legal but useless navigation until FAIL_MAX_STEPS."""

from __future__ import annotations

from rummagebench.core.types import Action, Observation, TargetKind, TargetRef


class TimeoutAgent:
    """Cycles NAV between two scenario places forever (legal, useless actions)."""

    name = "timeout"

    def __init__(self, places: list[str]):
        if len(places) < 2:
            raise ValueError("TimeoutAgent needs at least two places to alternate")
        self._places = places
        self._i = 0

    def reset(self, instruction: str) -> None:
        self._i = 0

    def act(self, observation: Observation) -> Action:
        place = self._places[self._i % len(self._places)]
        self._i += 1
        return Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value=place))
