"""Model-agnostic adapter: plug a VLA / VLM policy into run_visual_episode.

The policy is any callable

    policy(observation: dict, history: list[dict]) -> dict | str | None

that receives ONLY the public observation (PublicObservation.model_dump():
instruction, frame_id, image_png_b64, planning_step, max_planning_steps,
skill_library, feedback) and the agent's own previous (action, feedback)
pairs. It returns a public action (agent-visual-v2, configs/protocol/
action_v2.schema.json) as a dict or a JSON string; None abstains (ends the
episode, scored as budget exhaustion).

The adapter never touches the simulator. Unparseable output is passed on as
an empty action so the session rejects it with INVALID_ACTION and the step
counts — malformed outputs are model failures, not harness failures.

Loading from the command line (scripts/run_vla_variant.py --agent):
    --agent my_package.my_module:make_policy
where ``make_policy()`` returns the policy callable (or a PolicyAgent).
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Callable, Optional

Policy = Callable[[dict, list], Any]

class PolicyInfrastructureError(RuntimeError):
    """Inference transport/runtime failure; exclude the run from model scores."""


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_action(raw: Any) -> Optional[dict]:
    """dict / JSON text (optionally wrapped in prose or a code fence) -> dict.
    None stays None (abstain); anything unparseable becomes {} (INVALID_ACTION)."""
    if raw is None or isinstance(raw, dict):
        return raw
    text = _THINK.sub('', str(raw)).strip()
    if '<think>' in text:  # truncated reasoning has no final action
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    objects = []
    offset = 0
    while offset < len(text):
        start = text.find('{', offset)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            offset = start + 1
            continue
        objects.append(value)
        offset = start + length
    # Never choose one of several proposed actions for the model.
    return objects[0] if len(objects) == 1 else {}


class PolicyAgent:
    """Agent contract of evaluation.episode_log.run_visual_episode."""

    def __init__(self, policy: Policy, name: str = "policy"):
        self._policy = policy
        self.name = name
        self.history: list[dict[str, Any]] = []
        self._last_action: Optional[dict] = None

    def reset(self, instruction: str) -> None:
        self.history = []
        self._last_action = None

    def act(self, observation) -> Optional[dict]:
        obs = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        if self._last_action is not None:
            self.history.append({"action": self._last_action, "feedback": obs.get("feedback")})
        action = parse_action(self._policy(obs, list(self.history)))
        self._last_action = action
        return action


def load_agent(spec: str) -> PolicyAgent:
    """'package.module:factory' -> PolicyAgent (factory() may return either a
    PolicyAgent or a bare policy callable)."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"agent spec must be 'module:factory', got {spec!r}")
    made = getattr(importlib.import_module(module_name), attr)()
    if isinstance(made, PolicyAgent):
        return made
    if callable(made):
        return PolicyAgent(made, name=spec)
    raise TypeError(f"{spec} returned {type(made).__name__}, not a policy")


def make_turn_in_place_policy() -> Policy:
    """Trivial reference policy (wiring check, never a baseline): looks around
    with VIEW, then reports done — must always FAIL_FALSE_COMPLETION."""
    def policy(obs: dict, history: list) -> dict:
        if len(history) < 4:
            return {"skill": "VIEW", "yaw": [90, 180, -90, 0][len(history)], "pitch": 0}
        return {"skill": "REPORT_DONE"}
    return policy
