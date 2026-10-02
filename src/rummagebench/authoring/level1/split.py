"""§36/rev§1: dev/test split — by ENVIRONMENT id, stratified by paradigm.

All five robot variants of one environment share the split; pilot/debug
scenes never become held-out test scenes. Deterministic given the registry
contents and the dataset seed (no worker-order dependence).
"""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.level1.registry import DatasetRegistry


def assign_splits(registry: DatasetRegistry, dataset_seed: int,
                  dev_environments: int, test_environments: int) -> dict[str, str]:
    """Deterministic stratified assignment. Returns {environment_id: split}."""
    by_paradigm: dict[str, list[str]] = {}
    for entry in sorted(registry.accepted_environments(),
                        key=lambda e: e.environment_id):
        by_paradigm.setdefault(entry.paradigm, []).append(entry.environment_id)

    rng = np.random.default_rng(dataset_seed)
    assignment: dict[str, str] = {}
    remaining_dev, remaining_test = dev_environments, test_environments
    paradigms = sorted(by_paradigm)
    for paradigm in paradigms:
        ids = by_paradigm[paradigm]
        rng.shuffle(ids)
        total = len(ids)
        # proportional allocation of the dev budget
        share = min(total, round(dev_environments * total
                                 / max(1, dev_environments + test_environments)))
        share = min(share, remaining_dev)
        for environment_id in ids[:share]:
            assignment[environment_id] = "dev"
        for environment_id in ids[share:]:
            assignment[environment_id] = "test"
        remaining_dev -= share
        remaining_test -= total - share
    return assignment
