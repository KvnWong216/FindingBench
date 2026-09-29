"""Environment API: the JSON facade over the benchmark.

    env.reset()  -> {episode_id, observation: PublicObservation}
    env.step(action_json) -> {episode_status, planning_step, feedback, observation}

AGENT mode (default, §5/§19): the AGENT contract exposes ONLY the public
protocol — fixed 8-skill library, signed MOVE/TURN parameters, 2D-point
object targeting on the CURRENT frame, four-class feedback. No robot state,
object states, entity names, admissible action lists or state_update ever
cross this boundary.

ORACLE mode (mode="oracle") preserves the legacy privileged contract for
scripted acceptance agents, certification and the oracle planner.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from rummagebench.core.public_types import PublicActionFeedback, SessionMode
from rummagebench.core.scenario import load_scenario

def _rgb_to_png_b64(rgb) -> str | None:
    if rgb is None:
        return None
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class InteractiveSearchEnv:
    """JSON facade over one benchmark scenario."""

    def __init__(self, scenario_path: str | Path, seed: int = 0,
                 mode: str | SessionMode = SessionMode.AGENT,
                 trace_path: str | Path | None = None):
        self.mode = SessionMode(mode)
        self._scenario = load_scenario(scenario_path)
        from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

        self._backend = OmniGibsonBackend(seed=seed)
        self._backend.setup(self._scenario)
        if self.mode is SessionMode.AGENT:
            from rummagebench.core.visual_session import VisualProtocolSession

            self._session = VisualProtocolSession(self._backend, self._scenario)
            self._session.set_trace(trace_path)
        else:
            from rummagebench.core.session import BenchmarkSession

            self._session = BenchmarkSession(self._backend, self._scenario)

    # --------------------------------------------------------------- contract

    def reset(self) -> dict[str, Any]:
        observation = self._session.reset()
        if self.mode is SessionMode.AGENT:
            return {"episode_id": "episode_0", "observation": observation.model_dump()}
        return {
            "episode_id": self._scenario.id,
            "observation": self._oracle_observation(),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.mode is SessionMode.AGENT:
            result = self._session.step(action)
            return result.model_dump()
        result = self._session.act(action)
        return {
            "status": "SUCCESS"
            if (result.executed and result.observation.previous_action_result
                and result.observation.previous_action_result.get("postcondition_satisfied"))
            else "FAILURE",
            "reason": result.failure_reason.value,
            "episode_status": result.episode_status.value,
            "planning_step": result.planning_step,
            "state_update": {
                "held": self._session.world_state.held_object,
                **{
                    entity: {"open": self._backend.is_open(entity)}
                    for entity in self._scenario.initial_states
                },
            },
            "observation": self._oracle_observation(),
        }

    def status(self) -> dict[str, Any]:
        return {"episode_status": self._session.status().value
                if hasattr(self._session.status(), "value")
                else self._session.status()}

    # -------------------------------------------------------- ORACLE helper

    def _oracle_observation(self) -> dict[str, Any]:
        obs = self._session.observe()
        return {
            "instruction": obs.instruction,
            "planning_step": obs.planning_step,
            "max_planning_steps": obs.max_planning_steps,
            "previous_action_result": obs.previous_action_result,
            "available_skills": list(obs.available_skills),
            "candidate_skills": list(obs.candidate_skills),
            "image_png_b64": _rgb_to_png_b64(obs.rgb),
        }

    def close(self) -> None:
        self._backend.close()
