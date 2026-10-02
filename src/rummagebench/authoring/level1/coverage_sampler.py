"""§11: coverage-aware hierarchical object sampling.

Two stages — sample category, then sample model within the category — with
deterministic under-representation weights 1/sqrt(1 + usage_count) plus a
small seeded jitter from the object stream. The sampler tracks
category/model/target usage separately and reports coverage statistics.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from rummagebench.authoring.level1.catalog import CatalogEntry


class CoverageSampler:
    def __init__(self, entries: Iterable[CatalogEntry], object_rng):
        self._rng = object_rng
        self._by_category: dict[str, list[str]] = {}
        for e in entries:
            if e.category_excluded_reason is None:
                self._by_category.setdefault(e.category, []).append(e.model_id)
        if not self._by_category:
            raise ValueError("coverage sampler received no eligible categories")
        self._category_usage: Counter[str] = Counter()
        self._model_usage: Counter[str] = Counter()
        self._target_category_usage: Counter[str] = Counter()
        self._target_model_usage: Counter[str] = Counter()
        self._distractor_usage: Counter[str] = Counter()

    @property
    def categories(self) -> list[str]:
        return sorted(self._by_category)

    def _jitter(self) -> float:
        # small seeded component on top of the deterministic weight
        return float(self._rng.random()) * 1e-3

    def sample_category(self) -> str:
        weights = {c: 1.0 / math.sqrt(1.0 + self._category_usage[c]) + self._jitter()
                   for c in self._by_category}
        return _weighted_choice(weights, self._rng)

    def sample_model(self, category: str) -> str:
        weights = {m: 1.0 / math.sqrt(1.0 + self._model_usage[(category, m)])
                   + self._jitter()
                   for m in self._by_category[category]}
        return _weighted_choice(weights, self._rng)

    def sample(self) -> tuple[str, str]:
        """Sample (category, model) and record a distractor usage."""
        category = self.sample_category()
        model = self.sample_model(category)
        self.record_distractor(category, model)
        return category, model

    def record_target(self, category: str, model: str) -> None:
        self._category_usage[category] += 1
        self._model_usage[(category, model)] += 1
        self._target_category_usage[category] += 1
        self._target_model_usage[(category, model)] += 1

    def record_distractor(self, category: str, model: str) -> None:
        self._category_usage[category] += 1
        self._model_usage[(category, model)] += 1
        self._distractor_usage[(category, model)] += 1

    def stats(self) -> dict:
        return {
            "eligible_categories": len(self._by_category),
            "eligible_models": sum(len(v) for v in self._by_category.values()),
            "unique_target_categories": len(self._target_category_usage),
            "unique_target_models": len(self._target_model_usage),
            "category_usage": dict(self._category_usage),
            "model_usage": {f"{c}::{m}": n
                            for (c, m), n in sorted(self._model_usage.items())},
        }


def _weighted_choice(weights: dict, rng) -> str:
    total = sum(weights.values())
    u = rng.random() * total
    acc = 0.0
    for key, w in sorted(weights.items()):
        acc += w
        if u <= acc + 1e-12:
            return key
    return sorted(weights.items())[-1][0]
