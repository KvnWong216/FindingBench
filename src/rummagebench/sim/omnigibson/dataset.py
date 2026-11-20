"""Dataset-side helpers: deterministic model selection for spawned categories."""

from __future__ import annotations

from pathlib import Path

from rummagebench.core.errors import SimBackendError


def behavior_objects_dir() -> Path:
    """Root of the BEHAVIOR-1K object assets: <DATA_PATH>/behavior-1k-assets/objects."""
    from omnigibson.macros import gm

    path = Path(gm.DATA_PATH) / "behavior-1k-assets" / "objects"
    if not path.exists():
        raise SimBackendError(
            f"BEHAVIOR object assets not found at {path}; "
            "run the BEHAVIOR-1K dataset download first"
        )
    return path


def available_models_for_category(category: str) -> list[str]:
    root = behavior_objects_dir() / category
    if not root.exists():
        raise SimBackendError(f"category {category!r} not present in the dataset")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def pick_model_for_category(category: str) -> str:
    """Deterministically pick the first sorted model for a category."""
    models = available_models_for_category(category)
    if not models:
        raise SimBackendError(f"category {category!r} has no models")
    return models[0]
