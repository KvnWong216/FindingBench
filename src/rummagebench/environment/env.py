"""Environment API: the JSON facade over BenchmarkSession.

Transport-independent, matching the benchmark contract:

    env.reset()  -> {robot_state, object_states, observation}
    env.step(action_json) -> {status: SUCCESS|FAILURE, reason, state_update, observation}

Zero benchmark logic lives here: it only translates between dicts and the
canonical session types (Rule 7 — one schema everywhere).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import EpisodeStatus


def _rgb_to_png_b64(rgb) -> str | None:
    if rgb is None:
        return None
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class InteractiveSearchEnv:
    """JSON facade over one benchmark scenario."""

    def __init__(self, scenario_path: str | Path, seed: int = 0):
        self._scenario = load_scenario(scenario_path)
        from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

        self._backend = OmniGibsonBackend(seed=seed)
        self._backend.setup(self._scenario)
        self._session = BenchmarkSession(self._backend, self._scenario)

    # ---------------------------------------------------------------- helpers

    def _robot_state(self) -> dict[str, Any]:
        pose, quat = self._backend.robot_pose()
        return {
            "name": self._scenario.robot.name,
            "position": [round(v, 4) for v in pose],
            "orientation": [round(v, 4) for v in quat],
            # benchmark-owned holding state (never the backend's grasp joints)
            "holding": self._session.world_state.held_object,
        }

    def _object_states(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for entity in self._scenario.initial_states:
            states[entity] = {"open": self._backend.is_open(entity)}
        return states

    def _observation(self) -> dict[str, Any]:
        obs = self._session.observe()
        return {
            "instruction": obs.instruction,
            "planning_step": obs.planning_step,
            "max_planning_steps": obs.max_planning_steps,
            "previous_action_result": obs.previous_action_result,
            "available_skills": list(obs.available_skills),
            # candidate protocol (action_interface.mode == candidate):
            # semantic+state-valid, visible objects only, no feasibility info
            "candidate_skills": list(obs.candidate_skills),
            "image_png_b64": _rgb_to_png_b64(obs.rgb),
        }

    # --------------------------------------------------------------- contract

    def reset(self) -> dict[str, Any]:
        self._session.reset()
        return {
            "episode_id": self._scenario.id,
            "robot_state": self._robot_state(),
            "object_states": self._object_states(),
            "observation": self._observation(),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
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
            "observation": self._observation(),
        }

    def status(self) -> dict[str, Any]:
        return {"episode_status": self._session.status().value}

    def close(self) -> None:
        self._backend.close()
