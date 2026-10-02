"""§32/§36: build per-robot episode directories + split files.

    PYTHONPATH=src python scripts/build_level1_episodes.py
    PYTHONPATH=src python scripts/build_level1_split.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rummagebench.authoring.level1.config import load_dataset_config
from rummagebench.authoring.level1.registry import DatasetRegistry
from rummagebench.authoring.level1.snapshot import (
    atomic_write_json,
    canonical_hash,
)
from rummagebench.authoring.level1.split import assign_splits

REPO = Path(__file__).resolve().parents[1]
BUDGET = 16

INSTRUCTION_TEMPLATES = {
    "container_rummage": "Rummage the container and find the {display}. Pick it up.",
    "tabletop_clutter": "Find the {display} on the table and pick it up.",
    "deliberate_cover": "Something is covering the {display}. Uncover it and pick it up.",
}


def _display_name(category: str) -> str:
    return category.replace("_", " ")


def episodes() -> int:
    dataset_cfg = load_dataset_config()
    registry_path = REPO / "build" / "level1" / "dataset_registry.json"
    registry = DatasetRegistry.load(registry_path)
    built = 0
    for entry in registry.accepted_environments():
        for robot_id in entry.robot_ids:
            episode_dir = (REPO / "build" / "level1" / "episodes"
                           / entry.environment_id / robot_id)
            episode = {
                "environment_id": entry.environment_id,
                "environment_manifest_hash": entry.canonical_manifest_hash,
                "robot_id": robot_id,
                "target_category": entry.target_category,
                "target_display": _display_name(entry.target_category or ""),
                "instruction": (INSTRUCTION_TEMPLATES[entry.paradigm]
                                .format(display=_display_name(
                                    entry.target_category or "object"))),
                "paradigm": entry.paradigm,
                "max_planning_steps": BUDGET,
                "split": entry.split,
            }
            atomic_write_json(episode_dir / "episode.yaml", episode)
            built += 1
    print(f"episode definitions written: {built}")
    return 0


def split() -> int:
    dataset_cfg = load_dataset_config()
    registry_path = REPO / "build" / "level1" / "dataset_registry.json"
    registry = DatasetRegistry.load(registry_path)
    assignment = assign_splits(registry, dataset_seed=1,
                               dev_environments=dataset_cfg.split["dev_environments"],
                               test_environments=dataset_cfg.split["test_environments"])
    for environment_id, split in assignment.items():
        entry = registry.entries[environment_id]
        entry.split = split
    registry.save(registry_path)
    for split_name in ("dev", "test"):
        ids = sorted(k for k, v in assignment.items() if v == split_name)
        doc = {
            "split": split_name,
            "hash": canonical_hash(ids),
            "environments": ids,
            "episodes": len(ids) * dataset_cfg.robot_count,
        }
        target = (REPO / "benchmark_splits" / f"level1_v1_{split_name}.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        target.write_text(yaml.safe_dump(doc, sort_keys=True),
                          encoding="utf-8")
        print(f"{split_name}: {len(ids)} environments "
              f"({len(ids) * dataset_cfg.robot_count} episodes) hash="
              f"{doc['hash'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(episodes())
