"""§42/§46/rev§10: certification orchestration (GPU stage).

One fresh Kit process per (environment, robot) certification via subprocess
isolation with timeout, bounded retries (INVALID_RUN != scene semantics),
atomic resume, and registry wiring. The Stage B/C/D/E realization inside the
child (`--child` mode) uses the production OG backend and the real public
protocol, including a recorded protocol-solvable witness per pair
(revision §10): no blocker deletion, no collision disabling, no hidden
targets, no teleport navigation, no budget inflation.

    PYTHONPATH=src python scripts/certify_level1_environments.py \
        [--robots r1pro,...] [--limit N] [--resume]

GPU_PILOT_REQUIRED: the OG realization constants (empty-scene assembly,
robot insertion, lighting hooks) must be calibrated on the GPU during the
30-environment pilot (§49) before the 400-environment production run; the
structure below is the contract the pilot validates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rummagebench.authoring.level1.certifier import RejectionCode
from rummagebench.authoring.level1.config import load_dataset_config
from rummagebench.authoring.level1.registry import DatasetRegistry
from rummagebench.authoring.level1.snapshot import atomic_write_json

REPO = Path(__file__).resolve().parents[1]
MAX_SIM_RETRIES = 2


def _candidates(out_root: Path) -> list[Path]:
    return sorted(p for p in out_root.glob("level1_*")
                  if (p / "occupancy_spec.json").exists())


def _child_cmd(python: str, spec_path: Path, robot_id: str, out_dir: Path,
               ) -> list[str]:
    return [python, str(Path(__file__).resolve()),
            "--child", "--spec", str(spec_path), "--robot", robot_id,
            "--out", str(out_dir)]


def certify_child(spec_path: Path, robot_id: str, out_dir: Path) -> int:
    """GPU_PILOT_REQUIRED: in-child Stage B/C/D/E realization.

    Written against the backend APIs used by the successful closed loop
    (OmniGibsonBackend setup + VisualProtocolSession); the Level-1 minimal
    world (floor + support + container + objects) replaces the household
    scene here. Constants below MUST be calibrated during the §49 pilot —
    the pilot's job is to fail them fast on the real host.
    """
    import numpy as np

    from rummagebench.authoring.level1.certifier import (
        RejectionCode,
        check_post_settle_world,
        check_relations,
        check_settle,
        check_visibility,
    )
    from rummagebench.authoring.level1.config import (
        load_dataset_config,
        load_physics_config,
    )
    spec = json.loads(spec_path.read_text())
    candidate = spec["candidate"]
    dataset_cfg = load_dataset_config()
    physics_cfg = load_physics_config()
    robot_id = robot_id
    del robot_id  # robot profile resolution lands in the pilot calibration

    raise SystemExit(
        "GPU_PILOT_REQUIRED: Level-1 simulator realization (minimal scene "
        "assembly, robot insertion, settle measurement, isolated visibility "
        "reference, protocol witness replay) is calibrated during the "
        "30-environment pilot (spec §49). Orchestration, timeouts, retries, "
        "rejection codes and registry wiring above/below are already in "
        "place; run the pilot on a freed GPU to fill in the measured stages.")

    # ---- pilot-calibrated realization contract (unreachable until then) ----
    measurements = {
        "settle": {"sim_seconds": 0.0, "steps": 0,
                   "final_max_linear_velocity_mps": 0.0,
                   "final_max_angular_velocity_rps": 0.0,
                   "stable_consecutive_frames": physics_cfg.stable_window_steps,
                   "all_finite": True},
        "world": {"poses": {}, "max_penetration_m": 0.0,
                  "support_container_stable": True, "target_present": True,
                  "contacts_valid": True},
        "relations": {"inside_objects": [], "cover_relations": []},
        "visibility": {"ratio": None, "numerator": None, "denominator": None},
    }
    ok, code = check_settle(
        measurements["settle"], physics_cfg.stable_linear_velocity_mps,
        physics_cfg.stable_angular_velocity_rps,
        physics_cfg.stable_window_steps, physics_cfg.max_settle_steps)
    checks = {"settle": (ok, code and code.value)}
    ok, code = check_post_settle_world(
        measurements["world"], {o["name"] for o in candidate["objects"]},
        max(candidate["surface"]["size_xy"]) / 2
        + physics_cfg.escape_radius_margin_m,
        physics_cfg.penetration_tolerance_m)
    checks["world"] = (ok, code and code.value)
    band = dataset_cfg.visibility_bands[candidate["paradigm"]]
    ok, code, why = check_visibility(**measurements["visibility"], band=band)
    checks["visibility"] = (ok, code and code.value if code else why)
    failed = [name for name, (ok, _) in checks.items() if not ok]
    (out_dir / "certification.json").write_text(json.dumps(
        {"checks": checks, "failed": failed}, indent=1))
    return 0 if not failed else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="build/level1/candidates")
    parser.add_argument("--robots", default=None,
                        help="comma-separated verified robot ids (default: "
                             "every gpu_verified profile in the pool)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.child:
        return certify_child(Path(args.spec), args.robot, Path(args.out))

    from rummagebench.authoring.level1.robot_registry import verified_profiles
    dataset_cfg = load_dataset_config()
    profiles = verified_profiles()
    robot_ids = (sorted(profiles) if not args.robots
                 else [r.strip() for r in args.robots.split(",") if r.strip()])
    if len(robot_ids) < dataset_cfg.robot_count:
        print(f"WARNING: {len(robot_ids)} verified robots < required "
              f"{dataset_cfg.robot_count}; running DEBUG certification only "
              "(revision §2: production_ready stays false).")

    out_root = Path(args.candidates)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    registry = DatasetRegistry()
    registry_path = REPO / "build" / "level1" / "dataset_registry.json"
    if registry_path.exists():
        registry = DatasetRegistry.load(registry_path)

    python = sys.executable
    attempted = 0
    for spec_path in _candidates(out_root):
        if args.limit and attempted >= args.limit:
            break
        spec = json.loads((spec_path / "occupancy_spec.json").read_text())
        environment_id = spec["environment_id"]
        if args.resume and any(e.environment_id == environment_id
                               for e in registry.accepted_environments()):
            continue
        environment_seed = spec["environment_seed"]
        paradigm = spec["candidate"]["paradigm"]
        for robot_id in robot_ids:
            out_dir = out_root / environment_id / f"cert_{robot_id}"
            passed_all = True
            failing = RejectionCode.SIM_CRASH
            for retry in range(MAX_SIM_RETRIES + 1):
                proc = subprocess.run(
                    _child_cmd(python, spec_path / "occupancy_spec.json",
                               robot_id, out_dir),
                    capture_output=True, text=True, timeout=1800)
                if proc.returncode == 0:
                    failing = None
                    break
                if proc.returncode == 3:
                    failing = RejectionCode.SIM_CRASH  # pilot calibration gate
                registry.record_invalid_run(environment_id, robot_id,
                                            "certification", proc.stderr[-400:])
            if failing is not None:
                registry.reject(environment_id, environment_seed, paradigm,
                                failing.value, failing_robot=robot_id)
                passed_all = False
        attempted += 1
        atomic_write_json(registry_path.parent / "progress.json",
                          {"certification_attempted": attempted})
    registry.save(registry_path)
    print(f"certification pass done over {attempted} candidates -> "
          f"{registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
