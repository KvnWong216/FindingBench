"""Episode evaluation loop: agent <-> session, JSONL trajectory, summary.

The loop is transport-independent; MCP and future HTTP adapters wrap the same
BenchmarkSession, never this file's logic.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from rummagebench.core.events import load_events, write_summary
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


def load_trajectory(run_dir: str | Path) -> list[dict[str, Any]]:
    events_path = Path(run_dir) / "events.jsonl"
    if not events_path.exists():
        return []
    return load_events(events_path)
