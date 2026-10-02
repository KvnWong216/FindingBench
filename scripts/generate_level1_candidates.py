"""§17/§35/§45: Stage-A candidate generation (CPU) with atomic resume.

For every candidate seed: scene grammar -> coverage-aware sampling ->
occupancy packing -> lifted placements -> occupancy_spec + manifest. Output
is written atomically per candidate; --resume skips complete candidates and
never overwrites accepted ones.

    PYTHONPATH=src python scripts/generate_level1_candidates.py \
        [--candidates N] [--resume] [--start-seed S]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rummagebench.authoring.level1.catalog import (
    eligible_inventory,
    load,
)
from rummagebench.authoring.level1.compiler import (
    StructuralFailure,
    compile_candidate,
)
from rummagebench.authoring.level1.config import load_dataset_config
from rummagebench.authoring.level1.coverage_sampler import CoverageSampler
from rummagebench.authoring.level1.grammar import build_scene_plan
from rummagebench.authoring.level1.snapshot import (
    atomic_write_json,
    canonical_hash,
)

REPO = Path(__file__).resolve().parents[1]

# provisional verified-support choice: breakfast_table is the family used by
# the successful closed-loop run; per-candidate variety arrives after the GPU
# support-pool verification stage.
SUPPORT_MODEL = "breakfast_table"
SUPPORT_CENTER = [1.6, 1.6]
SUPPORT_SIZE = [1.6, 0.9]
SUPPORT_TOP_Z = 0.75
CONTAINER_INTERIOR = {"origin": [1.35, 1.45, 0.755], "size": [0.42, 0.30, 0.20]}
COVER_MODEL = ("bowl", "adciys")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=800)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--catalog",
                        default="build/level1/catalog/ordinary_objects.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="build/level1/candidates")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config()
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = REPO / catalog_path
    entries = eligible_inventory(load(catalog_path))
    sampler = CoverageSampler(entries, __import__("numpy").random.default_rng(
        dataset_seed := 1))

    progress_path = out_root / "progress.json"
    attempted = accepted = rejected = 0
    if args.resume and progress_path.exists():
        import json
        doc = json.loads(progress_path.read_text())
        attempted = doc["attempted"]
        accepted = doc["accepted"]
        rejected = doc["rejected"]

    paradigm_cycle = ["container_rummage", "tabletop_clutter",
                      "deliberate_cover"]
    for offset in range(attempted, args.candidates):
        seed = args.start_seed + offset
        candidate_id = seed
        paradigm = paradigm_cycle[offset % 3]
        env_id = f"level1_{paradigm}_{seed:06d}"
        candidate_dir = out_root / env_id
        if args.resume and (candidate_dir / "occupancy_spec.json").exists():
            continue
        try:
            plan = build_scene_plan(seed, paradigm, sampler, dataset_cfg,
                                    SUPPORT_MODEL, CONTAINER_INTERIOR and
                                    "tray", COVER_MODEL)
            candidate = compile_candidate(
                plan, dataset_cfg, SUPPORT_CENTER, SUPPORT_SIZE, SUPPORT_TOP_Z,
                CONTAINER_INTERIOR if paradigm == "container_rummage" else None,
                cover_dims=({plan.target().name: [0.10, 0.10, 0.08]}
                            | {o.name: [0.22, 0.22, 0.11]
                               for o in plan.objects if o.role == "cover"}
                            if paradigm == "deliberate_cover" else None),
            )
            manifest_hash = canonical_hash({
                "plan": {"paradigm": paradigm,
                         "support_model": SUPPORT_MODEL,
                         "counts": plan.counts},
                "placements": candidate.placements,
            })
            atomic_write_json(candidate_dir / "occupancy_spec.json",
                              {"environment_id": env_id,
                               "environment_seed": seed,
                               "manifest_hash": manifest_hash,
                               "stage": "A",
                               "candidate": {
                                   "paradigm": paradigm,
                                   "support_model": SUPPORT_MODEL,
                                   "container_model": "tray" if paradigm ==
                                   "container_rummage" else None,
                                   "surface": candidate.surface,
                                   "occupancy_spec": candidate.occupancy_spec,
                                   "placements": candidate.placements,
                                   "objects": candidate.objects,
                                   "relations": candidate.relations,
                               }})
            accepted += 1
        except StructuralFailure as e:
            atomic_write_json(candidate_dir / "stage_a_rejection.json",
                              {"environment_id": env_id, "code": str(e)})
            rejected += 1
        attempted += 1
        if offset % 25 == 0:
            atomic_write_json(progress_path,
                              {"attempted": attempted, "accepted": accepted,
                               "rejected": rejected})
            print(f"progress: {attempted} attempted, {accepted} stage-A ok",
                  flush=True)
    atomic_write_json(progress_path, {"attempted": attempted,
                                      "accepted": accepted,
                                      "rejected": rejected})
    print(f"stage-A generation done: {attempted} attempted, {accepted} ok, "
          f"{rejected} structural rejections (dataset seed {dataset_seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
