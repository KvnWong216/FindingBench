"""Python API adapter: the thinnest possible entry point into the benchmark.

Everything else (CLI, MCP) goes through the same BenchmarkSession; this
module only wires the OmniGibson backend to it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.scenario import load_scenario
from rummagebench.core.public_types import SessionMode
from rummagebench.core.session import BenchmarkSession
from rummagebench.robots.model_loader import resolve_urdf_path
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
from rummagebench.sim.omnigibson.raw_instance import GROUNDING_MODALITIES

logger = logging.getLogger(__name__)


def prepare_renderer_grounding(scenario) -> bool:
    """Pre-build preparation for renderer instance grounding.

    Returns True when the scenario requests renderer instance segmentation
    (fulfilled post-setup by the raw instance_id_segmentation_fast capture).
    The verified recipe restricts the R1Pro sensor set to its head ZED before
    the environment is built: fewer render products on the crash-prone host,
    and the AGENT protocol only ever reads the head camera.
    """
    requested = set(getattr(scenario.robot, "obs_modalities", None) or [])
    if not requested & GROUNDING_MODALITIES:
        return False
    if (getattr(scenario.robot, "model", "") == "r1pro"
            and scenario.robot.include_sensor_names is None):
        scenario.robot.include_sensor_names = ["zed_link"]
    return True


def install_renderer_grounding(backend) -> str:
    """Post-setup installation of the verified raw renderer grounding path.

    Shared by the Python API, the MCP worker and the localhost UI so every
    default adapter produces real renderer instance IDs, real depth and real
    RGB — no raycast substitute, no synthetic modality, and the strict AGENT
    reset gate still rejects incomplete frames.
    """
    from rummagebench.sim.omnigibson.raw_instance import (
        enable_renderer_grounding,
        install_physics_only_settle,
    )

    install_physics_only_settle(backend)
    return enable_renderer_grounding(backend)


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
    trace_path: str | Path | None = None,
    mode: str = "agent",
):
    """Build the simulator for this scenario and return a ready session.

    Call session.reset() before acting. ``trace_path`` enables the §4
    grounding trace (runs/<run_id>/grounding_trace.jsonl).

    mode="agent" (default): the final visual interaction protocol — fixed
    8-skill library, 2D-point targeting, four-class feedback, public
    observation only (§5). mode="oracle": the legacy privileged
    BenchmarkSession for scripted acceptance agents, certification and the
    oracle planner.
    """
    mode = SessionMode(mode)  # reject typos before launching the simulator
    scenario = load_scenario(scenario_path)
    grounding = prepare_renderer_grounding(scenario)
    backend = OmniGibsonBackend(seed=seed)
    backend.setup(scenario)
    if grounding:
        sensor_name = install_renderer_grounding(backend)
        logger.info("renderer instance grounding installed on %s", sensor_name)
    _ensure_kinematics_urdf(backend, scenario, scenario_path)
    log_path = None
    if run_dir is not None:
        log_path = str(Path(run_dir) / "events.jsonl")
    if mode is SessionMode.AGENT:
        from rummagebench.core.visual_session import VisualProtocolSession

        session = VisualProtocolSession(backend, scenario)
        # Public trace schema is distinct from legacy evaluator events.
        session.set_trace(trace_path if trace_path is not None else log_path)
        return session
    return BenchmarkSession(backend, scenario, log_path=log_path, trace_path=trace_path)
