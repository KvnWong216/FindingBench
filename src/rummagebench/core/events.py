"""Structured event log. Every semantic transition appends one event.

Evaluation data must live here (JSONL), never only in console output.
The format intentionally leaves room for later metrics: geodesic navigation
cost, revisits, safety violations, token usage, latency.

Remediation R03: versioned evaluation events. Oracle-mode and agent(visual)-
mode logs share one explicit schema so metrics never has to guess which
protocol produced a trajectory — and the two can never be silently mixed in
one ranking. Every event carries ``schema_version`` plus
``session_mode``/``action_protocol``/``feedback_protocol``; action events
additionally carry the parsed action, execution verdicts, binding evidence
and state fingerprints; reset / terminal / infrastructure failures are
explicit ``event_type``s, never action rows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

EVALUATION_EVENTS_SCHEMA_VERSION = "eval-events/1"

SESSION_MODE_ORACLE = "oracle"
SESSION_MODE_AGENT = "agent"

ORACLE_ACTION_PROTOCOL = "oracle_symbolic_v1"
ORACLE_FEEDBACK_PROTOCOL = "structured_step_feedback_v1"
AGENT_ACTION_PROTOCOL = "visual_pixel_v1"
AGENT_FEEDBACK_PROTOCOL = "public_feedback_v1"

# explicit non-action event types (R03: reset / terminal / infrastructure
# errors are never counted as skill steps)
EVENT_TYPE_ACTION = "action"
EVENT_TYPE_RESET = "reset"
EVENT_TYPE_TERMINAL = "terminal"
EVENT_TYPE_INFRASTRUCTURE_ERROR = "infrastructure_error"


def make_event(
    episode_id: str,
    step: int,
    action: dict[str, Any],
    validation: dict[str, Any],
    execution: dict[str, Any],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    """Legacy oracle-mode event, labelled with the versioned schema fields
    (R03) so oracle logs cannot pass unlabelled into agent rankings."""
    event = {
        "event_type": EVENT_TYPE_ACTION,
        "schema_version": EVALUATION_EVENTS_SCHEMA_VERSION,
        "session_mode": SESSION_MODE_ORACLE,
        "action_protocol": ORACLE_ACTION_PROTOCOL,
        "feedback_protocol": ORACLE_FEEDBACK_PROTOCOL,
        "episode_id": episode_id,
        "step": step,
        "timestamp": time.time(),
        "action": action,
        "validation": validation,
        "execution": execution,
        "status": status,
    }
    event.update(extra)
    return event


def run_metadata_event(
    episode_id: str,
    scenario: Any,
    *,
    session_mode: str,
    action_protocol: str,
    feedback_protocol: str,
    model_version: str | None = None,
    config_hash: str | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``run_metadata`` event per episode: scene/layout/seed, embodiment
    and camera, model identity and config hash, timing/token placeholders
    (null when the upstream harness does not provide them)."""
    robot = getattr(scenario, "robot", None)
    return {
        "event_type": "run_metadata",
        "schema_version": EVALUATION_EVENTS_SCHEMA_VERSION,
        "episode_id": episode_id,
        "session_mode": session_mode,
        "action_protocol": action_protocol,
        "feedback_protocol": feedback_protocol,
        "scene": getattr(getattr(scenario, "scene", None), "model", None),
        "layout_id": getattr(scenario, "layout_id", None),
        "pair_group_id": getattr(scenario, "pair_id", None),
        "seed": seed,
        "robot_model": getattr(robot, "model", None),
        "camera": (
            (getattr(robot, "include_sensor_names") or None)
            if getattr(robot, "include_sensor_names", None) else "default"
        ),
        "model_version": model_version,  # null when the harness does not supply it
        "config_hash": config_hash,
        "timing_s": None,
        "tokens": None,
        **(extra or {}),
    }


def assert_single_session_mode(events: list[dict[str, Any]]) -> str:
    """Refuse to score a trajectory that mixes oracle and agent events."""
    modes = {
        e.get("session_mode", SESSION_MODE_ORACLE)
        for e in events
        if e.get("event_type", EVENT_TYPE_ACTION) == EVENT_TYPE_ACTION
    }
    if len(modes) > 1:
        raise ValueError(
            f"event log mixes session modes {sorted(modes)}; oracle logs must "
            "never enter the agent ranking unlabelled"
        )
    return modes.pop() if modes else SESSION_MODE_ORACLE


class EpisodeLogWriter:
    """Appends events as JSONL. One file per episode under runs/<run_id>/."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def load_events(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def write_summary(run_dir: str | Path, summary: dict[str, Any]) -> Optional[Path]:
    out = Path(run_dir) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
    return out
