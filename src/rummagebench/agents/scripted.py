"""ScriptedSuccessAgent: replays a scenario-authored oracle action sequence.

The sequence lives in the scenario YAML (agent.scripted_success), so no
scenario-specific Python exists here (Architecture Rule 1).
"""

from __future__ import annotations

from rummagebench.agents.base import Agent  # noqa: F401 - re-exported protocol
from rummagebench.core.types import Action, Observation


class ScriptedSuccessAgent:
    name = "scripted"

    def __init__(self, sequence: list[dict]):
        self._sequence = [Action.from_dict(a) for a in sequence]
        self._i = 0

    def reset(self, instruction: str) -> None:
        self._i = 0

    def act(self, observation: Observation) -> Action:
        if self._i >= len(self._sequence):
            raise RuntimeError("scripted sequence exhausted before episode end")
        action = self._sequence[self._i]
        self._i += 1
        return action
