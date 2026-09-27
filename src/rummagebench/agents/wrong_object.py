"""WrongObjectAgent: searches correctly but finally grasps a distractor.

Must terminate with FAIL_WRONG_TARGET. The distractor entity is authored in
the scenario YAML (agent.scripted_wrong_object); the class only replays it.
"""

from __future__ import annotations

from rummagebench.core.types import Action, Observation


class WrongObjectAgent:
    name = "wrong_object"

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
