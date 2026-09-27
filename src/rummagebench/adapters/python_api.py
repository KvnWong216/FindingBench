"""Python API adapter: the thinnest possible entry point into the benchmark.

Everything else (CLI, MCP) goes through the same BenchmarkSession; this
module only wires the OmniGibson backend to it.
"""

from __future__ import annotations

from pathlib import Path

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend


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
    log_path = None
    if run_dir is not None:
        log_path = str(Path(run_dir) / "events.jsonl")
    return BenchmarkSession(backend, scenario, log_path=log_path)
