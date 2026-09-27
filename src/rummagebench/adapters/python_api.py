"""Python API adapter: the thinnest possible entry point into the benchmark.

Everything else (CLI, MCP) goes through the same BenchmarkSession; this
module only wires the OmniGibson backend to it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.robots.model_loader import resolve_urdf_path
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

logger = logging.getLogger(__name__)


def _ensure_kinematics_urdf(backend: OmniGibsonBackend, scenario, scenario_path) -> None:
    """Export the robot URDF from the loaded articulation when the scenario
    configures kinematics whose urdf_path does not exist yet.

    The URDF is always derived from the simulator asset (never hand-written)
    and FK-cross-validated by the exporter; a failed export aborts the run
    loudly instead of degrading to the reach-radius proxy.
    """
    kin = scenario.robot.kinematics
    if kin is None:
        return
    try:
        urdf = resolve_urdf_path(kin.urdf_path)
    except FeasibilityBackendError:
        logger.info(
            "kinematics URDF %s not found; exporting from the OmniGibson "
            "articulation of %s",
            kin.urdf_path, scenario.robot.model,
        )
        # default export location: build/robots/<model>.urdf (gitignored)
        out = Path("build/robots") / f"{scenario.robot.model}.urdf"
        manifest = backend.export_kinematics_urdf(str(out))
        logger.info(
            "exported %s (validation: %s)", out, manifest.get("validation")
        )


def create_session(
    scenario_path: str | Path,
    run_dir: str | Path | None = None,
    seed: int = 0,
) -> BenchmarkSession:
    """Build the simulator for this scenario and return a ready session.

    Call session.reset() before acting.
    """
    scenario = load_scenario(scenario_path)
    backend = OmniGibsonBackend(seed=seed)
    backend.setup(scenario)
    _ensure_kinematics_urdf(backend, scenario, scenario_path)
    log_path = None
    if run_dir is not None:
        log_path = str(Path(run_dir) / "events.jsonl")
    return BenchmarkSession(backend, scenario, log_path=log_path)
