"""Agent construction from scenario data (no scenario-specific Python)."""

from __future__ import annotations

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import Action


def make_agent(name: str, scenario: ScenarioSpec):
    if name == "scripted":
        from rummagebench.agents.scripted import ScriptedSuccessAgent

        return ScriptedSuccessAgent(scenario.agent.scripted_success)
    if name == "wrong_object":
        from rummagebench.agents.wrong_object import WrongObjectAgent

        return WrongObjectAgent(scenario.agent.scripted_wrong_object)
    if name == "timeout":
        from rummagebench.agents.timeout import TimeoutAgent

        places = list(scenario.anchors.keys())
        return TimeoutAgent(places)
    if name == "unsafe":
        from rummagebench.agents.unsafe import UnsafeActionAgent

        return UnsafeActionAgent(scenario.agent.unsafe_sequence)
    raise ValueError(
        f"unknown agent {name!r}; expected scripted|wrong_object|timeout|unsafe"
    )


def validate_agent_sequence(name: str, scenario: ScenarioSpec) -> None:
    """Fail fast when the scenario lacks the sequence an agent needs."""
    if name == "scripted" and not scenario.agent.scripted_success:
        raise ValueError(f"scenario {scenario.id} has no agent.scripted_success sequence")
    if name == "wrong_object" and not scenario.agent.scripted_wrong_object:
        raise ValueError(f"scenario {scenario.id} has no agent.scripted_wrong_object sequence")
    if name == "unsafe" and not scenario.agent.unsafe_sequence:
        raise ValueError(f"scenario {scenario.id} has no agent.unsafe_sequence")
