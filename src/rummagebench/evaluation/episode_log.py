"""Episode evaluation loop: agent <-> session, JSONL trajectory, summary.

The loop is transport-independent; MCP and future HTTP adapters wrap the same
BenchmarkSession, never this file's logic.

Remediation R03: ``run_visual_episode`` is the real AGENT-protocol batch
entry — it drives a VisualProtocolSession through ``.step(raw)`` with raw
public actions (never the legacy ``.act()``), writing the versioned
evaluation event log (session_mode/action_protocol/feedback_protocol are
recorded by the session itself). ``run_episode`` stays the ORACLE/scripted
acceptance loop and is explicitly auxiliary: its events are labelled
oracle-mode and can never silently enter the agent ranking.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from rummagebench.core.events import (
    AGENT_ACTION_PROTOCOL,
    AGENT_FEEDBACK_PROTOCOL,
    SESSION_MODE_AGENT,
    load_events,
    write_summary,
)
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import EpisodeStatus, Observation

logger = logging.getLogger(__name__)


def _save_rgb(run_dir: Path, step: int, obs: Observation) -> str | None:
    if obs.rgb is None:
        return None
    from PIL import Image

    path = run_dir / f"step_{step:03d}.png"
    Image.fromarray(obs.rgb).save(path)
    return path.name


def run_episode(
    session: BenchmarkSession,
    agent,
    run_dir: str | Path,
    save_images: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    observation = session.reset()
    agent.reset(observation.instruction)

    if save_images:
        _save_rgb(run_dir, 0, observation)

    error: str | None = None
    while session.status() == EpisodeStatus.RUNNING:
        try:
            action = agent.act(observation)
        except Exception as e:
            error = f"agent error: {e}"
            logger.exception("agent failed to produce an action")
            break
        result = session.act(action)
        observation = result.observation
        if save_images:
            _save_rgb(run_dir, result.planning_step, observation)
        logger.info(
            "step %d %s -> %s (%s)",
            result.planning_step,
            result.action,
            result.episode_status.value,
            result.failure_reason.value,
        )

    status = session.status()
    if error is not None and status == EpisodeStatus.RUNNING:
        status = EpisodeStatus.FAIL_MAX_STEPS  # treat agent breakdown as a failed episode

    summary = {
        "episode_id": session.episode_id,
        "agent": getattr(agent, "name", type(agent).__name__),
        "status": status.value,
        "planning_steps_used": observation.planning_step,
        "max_planning_steps": observation.max_planning_steps,
        "duration_s": round(time.time() - started, 2),
        "error": error,
        "instruction": observation.instruction,
    }
    write_summary(run_dir, summary)
    return summary


def run_visual_episode(
    session,
    agent,
    run_dir: str | Path,
    save_images: bool = False,
    model_version: str | None = None,
) -> dict[str, Any]:
    """AGENT-protocol evaluation loop (R03): drives a VisualProtocolSession
    with RAW public actions through ``.step()`` — never the legacy ``.act()``.

    The agent contract is ``reset(instruction)`` and
    ``act(observation: PublicObservation) -> raw dict | None`` (``None`` ends
    the loop as an unreported budget exhaustion or, when the report was
    already terminal, a no-op). The session writes the versioned evaluation
    event log itself (``set_trace``); infrastructure failures from
    ``.step()`` mark the run invalid — they are never model failures.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    session.set_trace(run_dir / "events.jsonl")
    session.model_version = model_version
    observation = session.reset()
    agent.reset(observation.instruction)
    # the agent's reset-time setup (episode spawn fixes) may have advanced
    # the world: act on the session's CURRENT snapshot, not the reset frame
    observation = session.observe()

    error: str | None = None
    run_invalid = False
    while session.status() == "RUNNING":
        try:
            raw = agent.act(observation)
        except Exception as e:
            error = f"agent error: {e}"
            logger.exception("visual agent failed to produce an action")
            break
        if raw is None:
            break  # agent abstains; the budget rule scores the outcome
        try:
            result = session.step(raw)
        except RuntimeError as e:
            # ENGINE_ERROR: infrastructure fault — run invalid, not model failure
            error = str(e)
            run_invalid = True
            logger.error("visual episode invalidated by infrastructure failure")
            break
        observation = result.observation
        if save_images:
            _save_rgb_png(run_dir, result.planning_step, observation)
        logger.info(
            "visual step %d -> %s (%s)",
            result.planning_step,
            result.episode_status,
            result.feedback.code.value,
        )

    status = session.status()
    if status == "RUNNING":
        # agent abstained or broke down without a terminal status: an
        # episode without a successful report is a budget-exhaustion failure
        status = "FAIL_MAX_STEPS"

    summary = {
        "episode_id": session.episode_id,
        "agent": getattr(agent, "name", type(agent).__name__),
        "session_mode": SESSION_MODE_AGENT,
        "action_protocol": AGENT_ACTION_PROTOCOL,
        "feedback_protocol": AGENT_FEEDBACK_PROTOCOL,
        "run_valid": not run_invalid,
        "status": status,
        "planning_steps_used": observation.planning_step,
        "max_planning_steps": observation.max_planning_steps,
        "duration_s": round(time.time() - started, 2),
        "error": error,
        "model_version": model_version,
        "instruction": observation.instruction,
    }
    write_summary(run_dir, summary)
    return summary


def _save_rgb_png(run_dir: Path, step: int, obs) -> str | None:
    """Save the public observation image (already base64 PNG)."""
    import base64 as _b64

    if getattr(obs, "image_png_b64", None) is None:
        return None
    path = run_dir / f"step_{step:03d}.png"
    path.write_bytes(_b64.b64decode(obs.image_png_b64))
    return path.name


class ScriptedPixelAgent:
    """Evaluator-side scripted driver for the visual protocol (smoke/CI).

    Replays ``scenario.agent.scripted_success``-style legacy entries
    ({"skill": ..., "target": {"value": ...}}) or raw public action dicts
    through the public schema. Legacy entity actions are resolved to the
    entity's CURRENT-frame centre pixel via the session's evaluator-private
    ``entity_pixel`` helper — the driver runs evaluator-side, so this is not
    part of any model-facing information boundary. Entries without a
    ``skill`` key are passed through verbatim as raw public actions.

    ``setup`` is an optional evaluator-side callable invoked after every
    session reset (episode spawn anchors, forced initial states) — it must
    not hand the model anything outside the public observation.
    """

    name = "scripted-pixel"

    def __init__(self, script: list[dict[str, Any]], session, setup=None):
        self._script = list(script)
        self._session = session
        self._setup = setup
        self._index = 0

    def reset(self, instruction: str) -> None:
        self._index = 0
        if self._setup is not None:
            self._setup(self._session)

    def act(self, observation) -> dict[str, Any] | None:
        if self._index >= len(self._script):
            return None
        entry = self._script[self._index]
        self._index += 1
        # only entity-targeted interaction skills need pixel resolution;
        # REPORT_DONE / MOVE / TURN pass through verbatim
        if "skill" not in entry or "point" in entry:
            return dict(entry)
        entity = (entry.get("target") or {}).get("value")
        if entity is None or entry["skill"] not in (
                "OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE"):
            return dict(entry)
        pixel = self._session.entity_pixel(entity)
        if pixel is None:
            raise ValueError(
                f"scripted driver cannot resolve entity {entity!r} to a pixel"
            )
        frame_id = observation.frame_id
        return {"skill": entry["skill"], "point": {
            "frame_id": frame_id, "x": pixel[0], "y": pixel[1],
        }}


def load_trajectory(run_dir: str | Path) -> list[dict[str, Any]]:
    events_path = Path(run_dir) / "events.jsonl"
    if not events_path.exists():
        return []
    return load_events(events_path)
