"""Level-1 configuration contracts (spec §40).

All knobs live in configs/level1/*.yaml; the Python source never duplicates
the dataset distributions. GPU-requiring values are recorded here as data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[4]
LEVEL1_DIR = REPO / "configs" / "level1"


class ConfigError(RuntimeError):
    pass


def _load(name: str) -> dict[str, Any]:
    path = LEVEL1_DIR / name
    if not path.exists():
        raise ConfigError(f"missing Level-1 config: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return doc


@dataclass(frozen=True)
class DatasetConfig:
    batch_label: str
    objects_root: Path
    accepted_environment_target: int
    candidate_budget: int
    paradigms: dict[str, int]
    robot_count: int
    voxel_size_m: float
    drop_batch_size: int
    object_count_distribution: dict[int, float]
    rummage_inside_distribution: dict[int, float]
    rummage_outside_distribution: dict[int, float]
    visibility_bands: dict[str, tuple[float, float]]
    split: dict[str, int]
    ordinary_object_limits: dict[str, float]
    pilot_safe_categories: list[str]
    pilot_environments_per_paradigm: int
    camera: dict[str, Any]

    @staticmethod
    def _counts(raw: dict[str, Any]) -> dict[int, float]:
        out = {int(k): float(v) for k, v in raw.items()}
        total = sum(out.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"distribution must sum to 1 (got {total})")
        return dict(sorted(out.items()))

    @classmethod
    def load(cls, path: Path | None = None) -> "DatasetConfig":
        doc = _load("dataset_v1.yaml" if path is None else str(path))
        bands = {k: (float(v[0]), float(v[1]))
                 for k, v in doc["visibility_bands"].items()}
        return cls(
            batch_label=str(doc["batch_label"]),
            objects_root=Path(doc["objects_root"]),
            accepted_environment_target=int(doc["accepted_environment_target"]),
            candidate_budget=int(doc["candidate_budget"]),
            paradigms={k: int(v) for k, v in doc["paradigms"].items()},
            robot_count=int(doc["robot_count"]),
            voxel_size_m=float(doc["voxel_size_m"]),
            drop_batch_size=int(doc["drop_batch_size"]),
            object_count_distribution=cls._counts(doc["object_count_distribution"]),
            rummage_inside_distribution=cls._counts(doc["rummage_inside_distribution"]),
            rummage_outside_distribution=cls._counts(doc["rummage_outside_distribution"]),
            visibility_bands=bands,
            split={k: int(v) for k, v in doc["split"].items()},
            ordinary_object_limits={k: float(v) for k, v in doc["ordinary_object_limits"].items()},
            pilot_safe_categories=list(doc.get("pilot_safe_categories", [])),
            pilot_environments_per_paradigm=int(doc["pilot"]["environments_per_paradigm"]),
            camera=dict(doc["camera"]),
        )

    def sample_count(self, distribution: dict[int, float], rng) -> int:
        """Inverse-CDF draw over the discrete count distribution."""
        u = rng.random()
        acc = 0.0
        for count, prob in distribution.items():
            acc += prob
            if u <= acc + 1e-12:
                return count
        return next(reversed(distribution.keys()))  # numeric tail


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    return DatasetConfig.load(path)


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float
    drop_batch_size: int
    stable_linear_velocity_mps: float
    stable_angular_velocity_rps: float
    stable_window_sim_seconds: float
    max_settle_sim_seconds: float
    physics_hz: float
    penetration_tolerance_m: float
    escape_radius_margin_m: float

    @property
    def stable_window_steps(self) -> int:
        return max(1, round(self.stable_window_sim_seconds * self.physics_hz))

    @property
    def max_settle_steps(self) -> int:
        return max(self.stable_window_steps,
                   round(self.max_settle_sim_seconds * self.physics_hz))

    @classmethod
    def load(cls, path: Path | None = None) -> "PhysicsConfig":
        doc = _load("physics_v1.yaml" if path is None else str(path))
        settle = doc["settle"]
        return cls(
            gravity=float(doc["gravity"]),
            drop_batch_size=int(doc["drop_batch_size"]),
            stable_linear_velocity_mps=float(settle["stable_linear_velocity_mps"]),
            stable_angular_velocity_rps=float(settle["stable_angular_velocity_rps"]),
            stable_window_sim_seconds=float(settle["stable_window_sim_seconds"]),
            max_settle_sim_seconds=float(settle["max_settle_sim_seconds"]),
            physics_hz=float(doc.get("physics_hz", 120.0)),
            penetration_tolerance_m=float(doc["penetration_tolerance_m"]),
            escape_radius_margin_m=float(doc["escape_radius_margin_m"]),
        )


def load_physics_config(path: Path | None = None) -> PhysicsConfig:
    return PhysicsConfig.load(path)


def load_appearance_config(path: Path | None = None) -> dict[str, Any]:
    """Appearance ranges (§15). Applying them is a GPU-stage responsibility;
    this module never mutates geometry."""
    return _load("appearance_v1.yaml" if path is None else str(path))


def load_support_pool(path: Path | None = None) -> dict[str, Any]:
    return _load("support_pool_v1.yaml" if path is None else str(path))


def load_container_pool(path: Path | None = None) -> dict[str, Any]:
    return _load("container_pool_v1.yaml" if path is None else str(path))


def load_robot_pool(path: Path | None = None) -> dict[str, Any]:
    return _load("robot_pool_v1.yaml" if path is None else str(path))
