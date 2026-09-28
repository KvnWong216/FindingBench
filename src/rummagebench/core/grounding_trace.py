"""Grounding trace (§4): evaluator-side JSONL log of the dynamic skill
grounding process — the answer to 'why does THIS action exist / not exist'.

One record per grounding pass:

    {
      "step": 3,
      "mode": "admissible",
      "robot_base_pose": [x, y, z, qx, qy, qz, qw],
      "world_state": {"held_object": null, "opens": [...]},
      "candidates": ["NAV(kitchen)", "OPEN(cabinet_A)", ...],
      "grounded": ["NAV(kitchen)", ...],
      "verdicts": [
        {
          "skill": "OPEN", "target": "cabinet_A",
          "result": "AVAILABLE" | "UNREACHABLE" | "COLLISION" | "INVALID_STATE",
          "feasible": true,
          "interaction_targets": [...],     # interface anchors (source, position, link)
          "ik": {...},                      # position/orientation error or reason
          "collision": {...},               # self/world, pairs
          "candidates_evaluated": n
        }, ...
      ]
    }

Traces are evaluator artifacts (runs/<run_id>/grounding_trace.jsonl) and may
contain oracle knowledge (hidden objects, IK internals); they never reach the
agent observation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GroundingTracer:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def record(
        self,
        step: int,
        mode: str,
        base_pose: list[float],
        world_state: dict[str, Any],
        candidates: list[str],
        grounded: list[str],
        verdicts: list[dict[str, Any]],
    ) -> None:
        record = {
            "step": step,
            "mode": mode,
            "robot_base_pose": [round(float(v), 6) for v in base_pose],
            "world_state": world_state,
            "candidates": candidates,
            "grounded": grounded,
            "verdicts": verdicts,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._count += 1


def summarize_verdict(verdict: dict[str, Any]) -> str:
    """One-line human-readable block for CLI trace/inspect output."""
    lines = [f"{verdict['skill']}({verdict['target']})"]
    for target in verdict.get("interaction_targets", []) or []:
        lines.append(
            f"  interaction_target: source={target.get('source')} "
            f"position={target.get('position')} link={target.get('link')}"
        )
    ik = verdict.get("ik") or {}
    if ik:
        if ik.get("success") is True or "position_error" in ik:
            lines.append(
                "  IK: success "
                f"pos_err={ik.get('position_error')} rot_err={ik.get('orientation_error')}"
            )
        else:
            lines.append(f"  IK: success=false reason={ik.get('reason')}")
    coll = verdict.get("collision") or {}
    if coll:
        lines.append(
            "  collision: "
            f"self_collision={coll.get('self_collision')} "
            f"world_collision={coll.get('world_collision')}"
        )
    lines.append(f"  result: {verdict['result']}")
    return "\n".join(lines)
