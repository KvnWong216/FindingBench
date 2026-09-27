"""Agent interface.

An agent sees only the Observation contract and returns canonical Actions.
Agents never import the simulator and never see privileged state.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rummagebench.core.types import Action, Observation


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, instruction: str) -> None: ...

    def act(self, observation: Observation) -> Action: ...
