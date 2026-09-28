"""Environment factory: turns a validated ScenarioSpec into an OmniGibson env.

Environment variables that OmniGibson reads at import time are applied here,
before the first omnigibson import, so the rest of the package can just
import it.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import yaml

from rummagebench.core.errors import SimBackendError
from rummagebench.core.scenario import ScenarioSpec


def repo_root() -> Path:
    """Locate the RummageBench repo root (contains configs/ and scenarios/)."""
    override = os.environ.get("RUMMAGEBENCH_ROOT")
    if override:
        return Path(override)
    # src/rummagebench/sim/omnigibson/env_factory.py -> repo root is 5 levels up
    return Path(__file__).resolve().parents[4]


def apply_runtime_env(gpu_id: int | None = None) -> None:
    """Set OmniGibson import-time environment knobs.

    Must be called BEFORE `import omnigibson` anywhere in the process.
    """
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
    if gpu_id is not None:
        os.environ["OMNIGIBSON_GPU_ID"] = str(gpu_id)


def apply_sim_settings() -> None:
    """Set OmniGibson global macros that must be applied AFTER
    `import omnigibson` (which exposes `omnigibson.macros.gm`) but BEFORE
    the simulator launches.

    USE_GPU_DYNAMICS enables the GPU particle systems (sludge etc.) that the
    Inside-placement sampler initializes when re-applying episode
    placements; without it the sampler raises
    "Failed to initialize sludge system".
    """
    import omnigibson as og

    og.macros.gm.USE_GPU_DYNAMICS = True


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_config() -> dict:
    return _load_yaml(repo_root() / "configs" / "simulator" / "omnigibson.yaml")


def load_robot_config(model: str) -> dict:
    path = repo_root() / "configs" / "robots" / f"{model.lower()}_symbolic.yaml"
    if not path.exists():
        raise SimBackendError(f"no robot config for {model!r} at {path}")
    return _load_yaml(path)


def build_env_config(scenario: ScenarioSpec) -> dict:
    """Merge the simulator template, the robot config, and the scenario."""
    config = deepcopy(load_env_config())
    config["scene"]["scene_model"] = scenario.scene.model

    robot_cfg = deepcopy(load_robot_config(scenario.robot.model))
    robot_cfg["model"] = scenario.robot.model
    robot_cfg["name"] = scenario.robot.name
    robot_cfg["obs_modalities"] = list(scenario.robot.obs_modalities)
    robot_cfg["grasping_mode"] = scenario.robot.grasping_mode
    robot_cfg.setdefault("sensor_config", {}).setdefault("VisionSensor", {}).setdefault(
        "sensor_kwargs", {}
    ).update(
        {
            "image_height": scenario.robot.image_height,
            "image_width": scenario.robot.image_width,
        }
    )
    # Robot initial pose comes from the scenario's init anchor; the backend
    # teleports via the same anchor mechanism used by NAV to keep one code path.
    config["robots"] = [robot_cfg]
    return config


def build_env(scenario: ScenarioSpec):
    """Create the OmniGibson Environment (launches the simulator on first call)."""
    import omnigibson as og
    from omnigibson.envs import Environment

    config = build_env_config(scenario)
    env = Environment(configs=config)
    return env, og
