"""§7/§8: build the Level-1 ordinary-object catalog (inventory stage, CPU).

Scans the installed BEHAVIOR object dataset into
build/level1/catalog/ordinary_objects.json with exact (category, model_id,
asset_path) records and documented category exclusions. Geometry/physics
fields stay pending until the GPU verification stage (revision §15).

    PYTHONPATH=src python scripts/build_level1_object_catalog.py [--objects-root PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rummagebench.authoring.level1.catalog import (
    eligible_inventory,
    save,
    scan_inventory,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-root", default=None,
                        help="BEHAVIOR objects dataset root (default: the "
                             "objects_root in configs/level1/dataset_v1.yaml)")
    parser.add_argument("--out",
                        default="build/level1/catalog/ordinary_objects.json")
    args = parser.parse_args()

    from rummagebench.authoring.level1.config import load_dataset_config
    objects_root = (Path(args.objects_root) if args.objects_root
                    else load_dataset_config().objects_root)
    entries = scan_inventory(objects_root)
    eligible = eligible_inventory(entries)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    save(entries, out)
    excluded = len(entries) - len(eligible)
    print(f"inventory: {len(entries)} (category, model) pairs; "
          f"{len(eligible)} inventory-eligible, {excluded} excluded by "
          f"documented denylist -> {out}")
    print("geometry/physics verification is a GPU-stage responsibility; "
          "fields remain pending until verified inside the simulator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
