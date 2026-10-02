"""§18/rev§7: precompute conservative voxel proxies.

CPU stage: axis-aligned AABB proxies per eligible catalog entry (thin-object
dilation applied). The GPU stage may upgrade entries to collision-geometry
voxelizations with proxy_source="collision"; cache identity binds source,
scale, orientation bin, resolution and preprocessing version, so a mask is
never reused across an orientation change.

    PYTHONPATH=src python scripts/precompute_level1_voxels.py [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rummagebench.authoring.level1.catalog import (
    eligible_inventory,
    load,
)
from rummagebench.authoring.level1.config import load_dataset_config
from rummagebench.authoring.occupancy.proxy import (
    aabb_proxy,
    proxy_cache_path,
    save_proxy,
)
from rummagebench.authoring.occupancy.voxelize import voxelize_dims

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",
                        default="build/level1/catalog/ordinary_objects.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_cfg = load_dataset_config()
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = REPO / catalog_path
    entries = eligible_inventory(load(catalog_path))
    if args.limit:
        entries = entries[:args.limit]

    cache_root = REPO / "build" / "level1" / "catalog"
    written = 0
    for entry in entries:
        # native dims are pending GPU verification: the CPU proxy uses the
        # documented default small-object envelope until then
        proxy = aabb_proxy(entry.category, entry.model_id,
                           entry.native_aabb_xyz or [0.12, 0.12, 0.12],
                           dataset_cfg.voxel_size_m)
        proxy["dims_pending_gpu_verification"] = entry.native_aabb_xyz is None
        path = proxy_cache_path(cache_root, entry.category, entry.model_id)
        if path.exists():
            continue
        save_proxy(path, voxelize_dims(proxy["physical_dims_m"],
                                       dataset_cfg.voxel_size_m),
                   dataset_cfg.voxel_size_m, "aabb",
                   {"world_volume_m3": proxy["world_volume_m3"],
                    "flatness": proxy["flatness"],
                    "elongation": proxy["elongation"],
                    "compactness": proxy["compactness"]})
        written += 1
    print(f"voxel proxies written: {written} "
          f"(source=aabb, proposal-only) -> {cache_root / 'voxels'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
