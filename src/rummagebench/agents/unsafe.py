"""UnsafeActionAgent: test-support agent that performs an unsafe interaction.

Used to exercise the FAIL_UNSAFE_ACTION branch end to end (the action is
never executed — the safety validator must terminate the episode first).
The unsafe action sequence is authored in the scenario YAML
(agent.unsafe_sequence), typically GRASP on fixed furniture.
"""

from __future__ import annotations

from rummagebench.core.types import Action, Observation


class UnsafeActionAgent:
    name = "unsafe"

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
