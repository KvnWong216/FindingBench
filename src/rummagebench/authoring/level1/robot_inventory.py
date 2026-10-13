"""Static inventory for the installed YAML-based OG robot registry (no GPU)."""
from __future__ import annotations

from pathlib import Path

import yaml

from rummagebench.authoring.tasks.plan import sha256_path

ROBOT_MODELS = ("r1pro", "r1", "fetch", "tiago", "stretch")


def discover_installed_robots(assets_root: Path) -> dict[str, dict]:
    root = Path(assets_root)
    if not (root / "models").is_dir():
        raise FileNotFoundError(f"robot models root missing: {root / 'models'}")
    result = {}
    for model in ROBOT_MODELS:
        definition = root / "models" / model / f"{model}.yaml"
        if not definition.is_file():
            result[model] = {"status": "unavailable", "reason": "MODEL_DEFINITION_MISSING"}
            continue
        doc = yaml.safe_load(definition.read_text())
        manipulation = doc.get("manipulation") or {}
        files = sorted(p for p in (definition.parent / "usd").glob("*")
                       if p.is_file() and p.suffix in {".usd", ".usda", ".usdc"})
        reasons = []
        if not manipulation.get("arm_joint_names") or not manipulation.get("eef_link_names"):
            reasons.append("ARM_DEFINITION_MISSING")
        if not (doc.get("locomotion") or {}).get("base_joint_names"):
            reasons.append("BASE_DEFINITION_MISSING")
        if not files:
            reasons.append("USD_ASSET_MISSING")
        result[model] = {"model": model, "status": "candidate" if not reasons else "unavailable",
                         "definition": str(definition), "definition_sha256": sha256_path(definition),
                         "usd_files": [str(p) for p in files], "reasons": reasons,
                         "arm_joints": manipulation.get("arm_joint_names", {}),
                         "eef_links": manipulation.get("eef_link_names", {}),
                         "camera_verification": "pending", "skill_verification": "pending",
                         "scene_verification": "pending"}
    return result
