"""Layout manifest (§30): exact model ids, transforms, config, validation
summary and a content hash. Identical inputs => identical manifest hash."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def build_manifest(layout, validation: dict, generator_version: str = "1.0.0",
                   behavior_version: str = "3.9.3") -> dict:
    manifest = {
        "generator_version": generator_version,
        "behavior_version": behavior_version,
        "git_commit": _git_commit(),
        "base_scene": layout.config.base_scene,
        "layout_seed": layout.layout_seed,
        "asset_categories": sorted({p.role for p in layout.placements}),
        "objects": [
            {
                "name": p.name, "category": p.category, "model_id": p.model_id,
                "role": p.role,
                "position": [round(v, 6) for v in p.position],
                "orientation_deg": p.orientation_deg,
                "aabb_size": [round(v, 6) for v in p.aabb_size],
            }
            for p in sorted(layout.placements, key=lambda p: p.name)
        ],
        "generator_config": layout.config.model_dump(),
        "rejected": layout.rejected,
        "validation": validation,
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return manifest


def write_layout_outputs(layout, manifest: dict, out_dir: Path) -> Path:
    """§30: layout.yaml + manifest.json + object_transforms.json under
    build/layouts/<id>_seedNNN/. occupancy/topdown previews are rendered by
    the sim-side pipeline when available (not required for validity)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "object_transforms.json").write_text(json.dumps(
        {p.name: {"position": p.position, "orientation_deg": p.orientation_deg,
                  "model_id": p.model_id}
         for p in layout.placements}, indent=2))
    (out_dir / "layout.yaml").write_text(json.dumps(
        {"layout": manifest["generator_config"],
         "objects": manifest["objects"]}, indent=2))
    return out_dir


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
    except Exception:
        return "unknown"
