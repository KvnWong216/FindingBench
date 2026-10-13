"""One household pipeline entry: CPU audit, or explicit GPU native capture.

Missing family sources are blockers. This command never fabricates family
members, applies appearance requests on CPU, or promotes oracle certificates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO, host_config

from rummagebench.authoring.level1.robot_inventory import discover_installed_robots
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path

MISSING_BASELINE = (
    "src/rummagebench/authoring/tasks/difficulty.py",
    "src/rummagebench/authoring/tasks/families.py",
    "src/rummagebench/authoring/tasks/family_binding.py",
    "src/rummagebench/authoring/tasks/household.py",
    "scripts/tasks/generate_families.py",
    "assets/tasks/household_families_v1.yaml",
    "assets/tasks/household_background_v1.yaml",
    "assets/tasks/household_robots_v1.yaml",
    "docs/household_task_families.md",
    "docs/household_background_pipeline.md",
    "tests/unit/test_household_background.py",
    "tests/unit/test_household_families.py",
)


def write_report(path: Path | None, report: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: reruns may reuse identical evidence, never
        # overwrite a conflicting run or a partial previous write.
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(canonical_json(report).decode())
        except FileExistsError:
            if json.loads(path.read_text()) != report:
                raise ValueError(f"conflicting report: {path}; use a new output")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cpu_audit(host: dict, robot_assets: Path) -> dict:
    missing = [p for p in MISSING_BASELINE if not (REPO / p).is_file()]
    robots = discover_installed_robots(robot_assets)
    # Count only state bundles actually emitted by this pipeline, never
    # configurations, seeds, historical certificates or .pt files alone.
    measured_native = 0
    for path in (host["build_root"] / "household" / "native").glob("*/inventory.json"):
        try:
            doc = json.loads(path.read_text())
            snapshot = path.parent / "native_state.pt"
            payload = {k: v for k, v in doc.items() if k != "inventory_sha256"}
            measured_native += int(doc.get("status") == "measured_native_occupancy"
                                   and snapshot.is_file()
                                   and sha256_bytes(canonical_json(payload)) == doc.get("inventory_sha256")
                                   and sha256_path(snapshot) == doc.get("snapshot_sha256"))
        except (OSError, ValueError):
            continue
    return {"stage": "cpu_audit", "status": "blocked" if missing else "pending_simulation",
            "missing_baseline_files": missing,
            "failure_reasons": {"MISSING_BASELINE_SOURCE": len(missing)} if missing else {},
            "counts": {"native_state_bundles": measured_native,
                       "generated_family_candidates_this_run": 0,
                       "simulation_validated_families_this_run": 0},
            "robots": robots, "camera_protocol": "current_code_head_rgb; wrist_only_not_certified",
            "stages": {"native_occupancy": "pending_gpu" if not measured_native else "measured",
                       "frozen_background": "pending", "family_binding": "blocked" if missing else "pending",
                       "appearance_application_and_images": "pending_gpu",
                       "robot_scene_skills": "pending_gpu", "search_cost_measurement": "pending_gpu"}}


def native_capture(args) -> dict:
    # Imports capable of loading simulator dependencies stay behind the
    # explicit GPU subcommand. --help and CPU audit never import OG.
    from rummagebench.authoring.tasks.native_occupancy import capture_native_occupancy
    from rummagebench.authoring.tasks.plan import sha256_bytes
    from rummagebench.authoring.tasks.slots import SceneSlots
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.state_io import save_snapshot

    slots = SceneSlots.load(args.slots)
    scenario = load_scenario(args.scenario)
    if scenario.scene.model != slots.scene:
        raise ValueError("scenario and SceneSlots refer to different scenes")
    # Reserve a fresh bundle before loading; incomplete bundles have no
    # inventory commit marker and are never counted or resumed as accepted.
    args.out.mkdir(parents=True, exist_ok=False)
    backend = OmniGibsonBackend(seed=args.seed)
    result = {}

    def capture(b):
        doc, state = capture_native_occupancy(b, slots, source_scene_path=args.source_scene)
        snapshot = save_snapshot(args.out / "native_state.pt", state,
                                 {"source_state_sha256": doc["source_state_sha256"]})
        doc["snapshot_sha256"] = sha256_path(snapshot)
        doc["scenario_sha256"] = sha256_path(args.scenario)
        # Rehash after binding the on-disk restore artifact and invocation.
        doc.pop("inventory_sha256")
        doc["inventory_sha256"] = sha256_bytes(canonical_json(doc))
        result.update(doc)
        return {"inventory_sha256": doc["inventory_sha256"], "counts": doc["counts"]}

    try:
        backend.setup(scenario, native_scene_capture=capture, native_scene_only=True)
        atomic_write_json(args.out / "inventory.json", result)
        return {"stage": "native_capture", "status": "pending_simulation_acceptance",
                "output": str(args.out), "counts": result["counts"],
                "generated_family_candidates_this_run": 0, "validated_families_this_run": 0}
    except Exception as exc:
        atomic_write_json(args.out / "failure.json", {"reason": type(exc).__name__, "detail": str(exc)})
        raise
    finally:
        backend.close()


def verify_native_restore(args) -> dict:
    """GPU check for a locally exported state, without target import or stepping."""
    from rummagebench.authoring.tasks.native_occupancy import require_known_slots, verify_source
    from rummagebench.authoring.tasks.slots import SceneSlots
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.state_io import load_snapshot

    doc = json.loads((args.bundle / "inventory.json").read_text())
    snapshot = args.bundle / "native_state.pt"
    if sha256_path(snapshot) != doc["snapshot_sha256"]:
        raise ValueError("native snapshot file hash mismatch")
    if sha256_path(args.scenario) != doc["scenario_sha256"]:
        raise ValueError("native invocation scenario hash mismatch")
    slots = SceneSlots.load(args.slots)
    # This is a trusted LOCAL snapshot emitted by export-native, never an
    # untrusted downloaded pickle or a redistributed BEHAVIOR asset.
    state, meta = load_snapshot(snapshot)
    if meta.get("source_state_sha256") != doc["source_state_sha256"]:
        raise ValueError("native snapshot metadata mismatch")
    backend = OmniGibsonBackend(seed=args.seed)

    def verify(b):
        for _ in range(2):
            b.restore_observe_state({"sim": state, "commanded_pose": None})
            verify_source(doc, b, slots)
        require_known_slots(doc, args.require_slots)
        return {"source_matches": True, "restore_repeats": 2}

    try:
        backend.setup(load_scenario(args.scenario), native_scene_capture=verify, native_scene_only=True)
        return {"stage": "verify_native_restore", "status": "native_restore_checked",
                "restore_repeats": 2, "validated_families_this_run": 0,
                "physical_acceptance": "pending", "required_slots": args.require_slots}
    finally:
        backend.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    cpu = sub.add_parser("cpu", help="static audit only; no simulator imports")
    cpu.add_argument("--host-config")
    cpu.add_argument("--robot-assets", type=Path)
    cpu.add_argument("--report", type=Path)
    gpu = sub.add_parser("export-native", help="GPU: capture builder-native state before task insertion")
    gpu.add_argument("--scenario", type=Path, required=True)
    gpu.add_argument("--slots", type=Path, required=True)
    gpu.add_argument("--source-scene", type=Path, required=True)
    gpu.add_argument("--out", type=Path, required=True)
    gpu.add_argument("--seed", type=int, default=0)
    restore = sub.add_parser("verify-native", help="GPU: restore an exported local native state twice")
    restore.add_argument("--scenario", type=Path, required=True)
    restore.add_argument("--slots", type=Path, required=True)
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--require-slots", nargs="*", default=[])
    restore.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.stage == "cpu":
        host = host_config(args.host_config)
        root = args.robot_assets or host["behavior_assets_root"].parent / "omnigibson-robot-assets"
        report = cpu_audit(host, root)
        write_report(args.report, report)
        return 2 if report["status"] == "blocked" else 0
    report = native_capture(args) if args.stage == "export-native" else verify_native_restore(args)
    write_report(None, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
