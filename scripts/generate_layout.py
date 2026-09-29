#!/usr/bin/env python3
"""Generate a deterministic BEHAVIOR layout (protocol §23-§31).

    python scripts/generate_layout.py --config configs/layouts/kitchen_search_v1.yaml \
        --catalog configs/layouts/asset_pool_v1.yaml --seed 0 [--out build/layouts]

Deterministic: same repo + config + seed => identical manifest hash.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from rummagebench.authoring.layout.asset_catalog import AssetCatalog  # noqa: E402
from rummagebench.authoring.layout.generator import LayoutGenerator  # noqa: E402
from rummagebench.authoring.layout.manifest import build_manifest, write_layout_outputs  # noqa: E402
from rummagebench.authoring.layout.spec import LayoutConfig  # noqa: E402
from rummagebench.authoring.layout.validator import LayoutValidator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO / "build" / "layouts"))
    args = parser.parse_args()

    config = LayoutConfig.model_validate(yaml.safe_load(Path(args.config).read_text())["layout"])
    catalog = AssetCatalog.from_yaml(args.catalog)
    layout = LayoutGenerator(catalog, config).generate(args.seed)
    validation = LayoutValidator(config).validate(layout, catalog)
    manifest = build_manifest(layout, validation)

    seed_dir = Path(args.out) / f"{config.id}_seed{args.seed:03d}"
    write_layout_outputs(layout, manifest, seed_dir)
    print(json.dumps({
        "out": str(seed_dir),
        "valid": validation["valid"],
        "objects": len(layout.placements),
        "rejected": len(layout.rejected),
        "manifest_hash": manifest["manifest_hash"][:16],
    }, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
