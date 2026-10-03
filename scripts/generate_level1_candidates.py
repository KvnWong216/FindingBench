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
import sys
from pathlib import Path

import yaml

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
    # rev §15 staged preprocessing: the pilot samples only from categories
    # proven to load in FindingBench GPU runs; --full-catalog lifts the gate
    if "--full-catalog" not in sys.argv:
        safe = set(load_dataset_config().pilot_safe_categories)
        entries = [e for e in entries if e.category in safe]
        print(f"pilot gate: {len(entries)} entries from "
              f"{len(safe)} proven-safe categories")
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
                                    SUPPORT_MODEL,
                                    "adciys" if paradigm == "container_rummage"
                                    else None, COVER_MODEL,
                                    exclude_pairs={("bowl", "adciys")})

            candidate = compile_candidate(
                plan, dataset_cfg, SUPPORT_SIZE, SUPPORT_TOP_Z,
                container_interior_size=([0.42, 0.30, 0.20]
                                         if paradigm == "container_rummage"
                                         else None),
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
                                   "container_model": "recycling_bin" if paradigm ==
                                   "container_rummage" else None,
                                   "surface": candidate.surface,
                                   "occupancy_spec": candidate.occupancy_spec,
                                   "placements": candidate.placements,
                                   "objects": candidate.objects,
                                   "relations": candidate.relations,
                               }})
            # Level-1 scenario: the candidate through the FROZEN protocol —
            # objects on the verified support via on_top placements; the
            # standard driver runs the witness (rev §10).
            target_name = next(o["name"] for o in candidate.objects
                               if o["role"] == "target")
            target_cat = next(o["category"] for o in candidate.objects
                              if o["role"] == "target")
            display = target_cat.replace("_", " ")
            instruction = {
                "container_rummage": "Rummage the container on the table and find the {d}. Pick it up.",
                "tabletop_clutter": "Find the {d} on the table and pick it up.",
                "deliberate_cover": "Something on the table may be covering the {d}. Uncover it and pick it up.",
            }[paradigm].format(d=display)
            level1_scenario = {
                "id": env_id,
                "instruction": instruction,
                "scene": {"model": "Beechwood_0_int"},
                "robot": {"model": "r1pro",
                          "obs_modalities": ["rgb", "depth_linear",
                                             "seg_instance"],
                          "name": "robot_0",
                          "image_width": 854, "image_height": 480,
                          "focal_length_mm": 10.5,
                          "init_anchor": "table_side",
                          "kinematics": {"urdf_path": "build/robots/r1pro.urdf",
                                         "base_link": "base_link",
                                         "end_effector_link": "right_eef_link",
                                         "controlled_joints": "auto",
                                         "joint_limits_source": "urdf",
                                         "ik_seed": 0},
                          "reach_radius": 1.0, "z_min": 0.0, "z_max": 1.6,
                          "hand_capacity": 1},
                "feasibility": {"backend": "pinocchio", "mode": "endpoint"},
                "anchors": {"table_side": {"position": [1.6, -5.15, 0.0053],
                                           "orientation": [0.0, 0.0, 0.0, 1.0]}},
                "target": {"entity": target_name, "category": target_cat},
                "objects": [{"name": o["name"], "category": o["category"],
                             "model": o["model"]}
                            for o in candidate.objects],
                "placements": [{"entity": o["name"], "relation": "on_top",
                                "receptacle": "breakfast_table_uhrsex_0"}
                               for o in candidate.objects],
                "termination": {"max_planning_steps": 16,
                                "fail_on_wrong_grasp": True,
                                "fail_on_unsafe_action": True,
                                "succeed_when_holding_target": True},
                "skills": ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"],
                "safety": {"forbidden_categories": [],
                           "grasping_fixed_base_unsafe": True},
            }
            atomic_write_json(candidate_dir / "level1_scenario.json",
                              level1_scenario)
            level1_dir = REPO / "scenarios" / "level1" / env_id
            level1_dir.mkdir(parents=True, exist_ok=True)
            (level1_dir / "scenario.yaml").write_text(
                yaml.safe_dump(level1_scenario, sort_keys=False),
                encoding="utf-8")
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
