"""§13/§14: verified support pool handling (data lives in the config)."""
from __future__ import annotations

from rummagebench.authoring.level1.config import load_support_pool
from rummagebench.authoring.occupancy.support import (
    build_support_grid,
    sample_cluster_centers,
)


def verified_support_families(pool: dict | None = None) -> list[str]:
    """Families whose models passed GPU geometry verification."""
    doc = pool if pool is not None else load_support_pool()
    return sorted(name for name, entry in doc["families"].items()
                  if entry.get("verified"))


def candidate_support_categories(pool: dict | None = None) -> list[str]:
    """All dataset categories the pool is willing to consider."""
    doc = pool if pool is not None else load_support_pool()
    return [entry.get("dataset_category", name)
            for name, entry in sorted(doc["families"].items())]
