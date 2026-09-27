"""Structured event log. Every semantic transition appends one event.

Evaluation data must live here (JSONL), never only in console output.
The format intentionally leaves room for later metrics: geodesic navigation
cost, revisits, safety violations, token usage, latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def make_event(
    episode_id: str,
    step: int,
    action: dict[str, Any],
    validation: dict[str, Any],
    execution: dict[str, Any],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    event = {
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
