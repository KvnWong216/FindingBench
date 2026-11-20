"""Agent interface — the frozen baseline agent API (§18, Core v0.1).

An agent sees ONLY the Observation contract and returns canonical Actions.
Agents never import the simulator and never see privileged state.

Input  (rummagebench.core.types.Observation):
    instruction            task text
    rgb                    head-camera frame (HxWx3 uint8, or None)
    planning_step          current semantic planning step
    max_planning_steps     episode horizon
    previous_action_result feedback for the LAST action:
                           {executed, postcondition_satisfied, failure_reason}
                           — in candidate mode this is the environment's
                           structured oracle answer (SUCCESS / UNREACHABLE /
                           COLLISION / INVALID_STATE / ...)
    available_skills       A_t^admissible  (action_interface.mode=admissible)
    candidate_skills       A_t^candidate   (action_interface.mode=candidate;
                           semantic+state-valid, VISIBLE objects only, no
                           feasibility metadata)

Output (rummagebench.core.types.Action):
    {"skill": ..., "target": {type, value}} — the canonical schema.

NEVER exposed to agents: target ground-truth location, IK results, collision
pairs, feasibility scores, entity lists behind closed containers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rummagebench.core.types import Action, Observation


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, instruction: str) -> None: ...

    def act(self, observation: Observation) -> Action: ...
