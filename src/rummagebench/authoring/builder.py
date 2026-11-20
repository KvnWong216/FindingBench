"""Scenario builder: compiles a scenario YAML into a reproducible initial state.

    python -m rummagebench.authoring.build scenarios/knife_search_001/scenario.yaml

Produces under build/scenarios/<id>/:
    initial_state.pt      simulator snapshot (deterministic reset)
    preview.png           head-camera render at the initial anchor
    resolved_entities.json  chosen dataset models + entity metadata
    build_report.json     per-predicate pass/fail report
"""

from __future__ import annotations

import logging
from pathlib import Path

from rummagebench.core.errors import ScenarioValidationError, SimBackendError
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
from rummagebench.sim.omnigibson.state_io import save_json, save_snapshot
from rummagebench.sim.omnigibson.dataset import behavior_objects_dir

logger = logging.getLogger(__name__)


def build_scenario(
    scenario_path: str | Path,
    out_root: str | Path | None = None,
    seed: int = 0,
) -> dict:
    scenario = load_scenario(scenario_path)
    out_root = Path(out_root) if out_root else Path("build/scenarios")
    out_dir = out_root / scenario.id
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = OmniGibsonBackend(seed=seed)
    try:
        report = backend.setup(scenario)

        from rummagebench.authoring.validation import verify_predicates

        verdict = verify_predicates(scenario, report)
        if not verdict["passed"]:
            raise SimBackendError(f"scenario build failed verification: {verdict}")

        preview_path = backend.render_snapshot(str(out_dir / "preview.png"))
        snapshot_path = save_snapshot(
            out_dir / "initial_state.pt",
            backend.dump_state(),
            meta={"scenario_id": scenario.id, "seed": seed},
        )
        resolved = {
            "scenario_id": scenario.id,
            "scene": scenario.scene.model,
            "spawned": report["spawned"],
            "anchors": scenario.anchors,
            "entity_infos": {
                name: vars(info) for name, info in backend._entity_infos.items()
            },
        }
        resolved_path = save_json(out_dir / "resolved_entities.json", resolved)
        build_report_path = save_json(
            out_dir / "build_report.json",
            {"build": report, "verification": verdict, "seed": seed},
        )
        logger.info("scenario %s built at %s", scenario.id, out_dir)
        return {
            "scenario_id": scenario.id,
            "out_dir": str(out_dir),
            "preview": str(preview_path),
            "snapshot": str(snapshot_path),
            "resolved_entities": str(resolved_path),
            "build_report": str(build_report_path),
            "report": report,
        }
    finally:
        backend.close()
