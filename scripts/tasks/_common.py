"""Shared helpers for scripts/tasks/* (path setup + host config)."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

ASSETS = REPO / "assets" / "tasks"


def host_config(path: str | None = None) -> dict:
    p = Path(path) if path else REPO / "configs" / "tasks" / "host_v1.yaml"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    doc["build_root"] = REPO / doc.get("build_root", "build/tasks")
    doc["behavior_assets_root"] = Path(doc["behavior_assets_root"])
    doc["bddl_activity_root"] = Path(doc["bddl_activity_root"])
    return doc


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
