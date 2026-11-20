"""Tier contracts (TierSpec) and per-tier planner rules."""
from __future__ import annotations

from pathlib import Path

from rummagebench.authoring.tasks.tiers.base import TierSpec, load_tier_spec

REPO = Path(__file__).resolve().parents[5]
TIERS_DIR = REPO / "assets" / "tasks" / "tiers"


def tier_spec_path(name: str) -> Path:
    """name like 'easy_v1'."""
    return TIERS_DIR / f"{name}.yaml"


def load_tier(name: str) -> TierSpec:
    return load_tier_spec(tier_spec_path(name))


def load_all_tiers(version: int = 1) -> list[TierSpec]:
    return [load_tier(f"{t}_v{version}") for t in ("easy", "medium")]
