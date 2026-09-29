"""Asset catalog: curated BEHAVIOR-1K model pool with exact model ids (§26).

Never "first sorted model in category". Every entry is an explicit allowed
model with its AABB size (meters). Validation fails LOUDLY on missing
category/model and never silently substitutes.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AssetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    aabb_size: list[float] = Field(..., min_length=3, max_length=3)


class AssetCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    models: list[AssetEntry]


class AssetCatalog:
    def __init__(self, entries: dict[str, list[AssetCategory]]):
        # entries: role -> list of categories
        self._by_role: dict[str, dict[str, AssetEntry]] = {}
        for role, categories in entries.items():
            flat: dict[str, AssetEntry] = {}
            for cat in categories:
                for entry in cat.models:
                    if entry.model_id in flat:
                        raise ValueError(f"duplicate model id {entry.model_id!r}")
                    flat[entry.model_id] = entry
            self._by_role[role] = flat

    @classmethod
    def from_yaml(cls, path) -> "AssetCatalog":
        from pathlib import Path

        raw = yaml.safe_load(Path(path).read_text())
        entries: dict[str, list[AssetCategory]] = {}
        for role, categories in raw.items():
            if isinstance(categories, dict):
                # clutter: {subcategory: [models...]} -> category = subcategory
                categories = [
                    {"category": sub, "models": models}
                    for sub, models in categories.items()
                ]
            entries[role] = [AssetCategory.model_validate(c) for c in categories]
        return cls(entries)

    def models_for(self, role: str) -> dict[str, AssetEntry]:
        if role not in self._by_role or not self._by_role[role]:
            raise ValueError(f"asset catalog: no models for role {role!r}")
        return self._by_role[role]

    def get(self, role: str, model_id: str) -> AssetEntry:
        entry = self._by_role.get(role, {}).get(model_id)
        if entry is None:
            raise ValueError(
                f"asset catalog: model {model_id!r} not allowed for role {role!r} — never substitute"
            )
        return entry
