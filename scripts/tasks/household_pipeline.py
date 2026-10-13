"""One household pipeline entry: CPU audit, or explicit GPU native capture.

Pinned Infinigen proposals and geometry use CPU stages. Native capture and
restore are explicitly GPU-only stages. No stage promotes oracle certificates.
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
    for name, profile in robots.items():
        profile["integration_status"] = ("configured" if name == "r1pro" and
            (REPO / "configs/tasks/robots/r1pro.yaml").is_file() and profile["status"] == "candidate" else
            "integration_pending" if profile["status"] == "candidate" else "unavailable")
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
    return {"stage": "cpu_audit", "status": "pending_simulation",
            "missing_baseline_files": missing,
            "failure_reasons": {}, "baseline_note": "historical cloud sources absent; current Infinigen implementation is independent",
            "counts": {"native_state_bundles": measured_native,
                       "generated_family_candidates_this_run": 0,
                       "simulation_validated_families_this_run": 0},
            "robots": robots, "camera_protocol": "current_code_head_rgb; wrist_only_not_certified",
            "stages": {"native_occupancy": "pending_gpu" if not measured_native else "measured",
                       "frozen_background": "pending", "family_binding": "pending",
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


def load_household_spec(path: Path, policy: Path):
    from rummagebench.authoring.tasks.household import HouseholdTaskSpec, validate_spec_assets
    spec = HouseholdTaskSpec.model_validate_json(path.read_text())
    validate_spec_assets(spec, REPO, policy)
    return spec


def plan_infinigen(args) -> dict:
    from rummagebench.authoring.tasks.household import sample_spec, family_members, grouped_split, validate_spec_assets
    from rummagebench.authoring.tasks.infinigen import generation_command, export_command
    if not 1 <= args.count <= 8:
        raise ValueError("pilot count must be 1..8; first verify one actual scene")
    args.out.mkdir(parents=True, exist_ok=True)
    all_specs, requests = [], []
    for index in range(args.count):
        base = sample_spec(args.policy, REPO, args.seed, index)
        validate_spec_assets(base, REPO, args.policy)
        members = family_members(base)
        for spec in members:
            path = args.out / (spec.task_id + ".json")
            if path.exists():
                if path.read_bytes() != spec.canonical_bytes():
                    raise ValueError(f"SPEC_OUTPUT_CONFLICT:{path}")
            else:
                with path.open("xb") as f:
                    f.write(spec.canonical_bytes())
        all_specs.extend(members)
        requests.append({"family_id": base.family_id, "specs": [s.task_id + ".json" for s in members],
                         "base_layout_id": base.base_layout_id,
                         "generation": generation_command(base, args.python, args.upstream_root,
                                                          args.out / base.base_layout_id / "geometry"),
                         "export": export_command(args.python, args.upstream_root,
                                                  args.out / base.base_layout_id / "geometry",
                                                  args.out / base.base_layout_id / "export")})
    report = {"version": 1, "stage": "plan_infinigen", "status": "candidate",
              "seed": args.seed, "counts": {"task_specs": len(all_specs), "families": args.count,
              "geometry_scenes": 0, "exported_scene_files": 0, "evaluation_ready_episodes": 0},
              "split": grouped_split(all_specs, split_seed=args.split_seed),
              "split_scope": "layout_grouped_dev_test; not a claim of asset or family generalization",
              "requests": requests, "failure_reasons": {}, "gpu_started": False,
              "appearance_status": "requested_pending_application_and_images",
              "robots": {"r1pro": "configured_pending_scene_skill_validation"},
              "camera_protocol": "current_head_rgb; no wrist-only certification",
              "difficulty_status": "requested_only; actual geometry and search not measured"}
    write_report(args.out / "batch.json", report)
    return report


def cpu_generate(args) -> dict:
    from rummagebench.authoring.tasks.infinigen import run_cpu_generation
    spec = load_household_spec(args.spec, args.policy)
    return run_cpu_generation(spec, args.upstream_root, args.python, args.out,
                              timeout_s=args.timeout, retries=args.retries)


def prepare_infinigen(args) -> dict:
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly, scene_slots_from_assembly
    spec = load_household_spec(args.spec, args.policy)
    try:
        doc = prepare_assembly(spec, args.scene, args.solve_state, json.loads(args.bindings.read_text()),
                               args.assets_root, args.out)
    except Exception as exc:
        # Keep partial preparation evidence; absence of assembly.json is the
        # commit marker distinguishing a failed bundle from a candidate.
        if not (args.out / "assembly.json").exists():
            args.out.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.out / "failure.json", {"status": "assembly_failed", "family": spec.family,
                "requested_difficulty": spec.requested_difficulty, "reason": f"{type(exc).__name__}: {exc}",
                "evaluation_ready_episodes": 0})
        raise
    slots = scene_slots_from_assembly(args.out / "assembly.json")
    slots.save(args.out / "scene_slots.json")
    return {"stage": "prepare_infinigen", "status": doc["status"],
            "assembly": str(args.out / "assembly.json"), "slots": str(args.out / "scene_slots.json"),
            "counts": {"objects": len(doc["objects"]), "required_candidates": len(doc["candidate_bindings"]),
                       "evaluation_ready_episodes": 0}, "pending_checks": doc["pending_checks"]}


def bind_infinigen(args) -> dict:
    from rummagebench.authoring.tasks.household import family_members
    from rummagebench.authoring.tasks.household_binding import bind_family
    spec = load_household_spec(args.spec, args.policy)
    return bind_family(family_members(spec), args.assembly, args.overlay, args.out)



def gpu_infinigen_preflight(args) -> dict:
    """GPU ONLY, never invoked by CPU commands. Partial checks cannot release data."""
    import yaml
    import numpy as np
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.authoring.tasks.native_occupancy import capture_native_occupancy, verify_source
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly, runtime_articulation_checks
    from rummagebench.sim.omnigibson.state_io import save_snapshot

    spec = load_household_spec(args.spec, args.policy)
    manifest = load_assembly_manifest(args.assembly)
    if spec.base_layout_id != manifest["scene_id"] or spec.family_id != manifest["family_id"]:
        raise ValueError("ASSEMBLY_SPEC_MISMATCH")
    slots = scene_slots_from_assembly(args.assembly)
    robot_doc = yaml.safe_load((REPO / spec.robot_config_ref).read_text())
    start = json.loads(args.start.read_text())
    scenario = ScenarioSpec.model_validate({"id": "preflight_" + spec.base_layout_id,
        "instruction": "evaluator-only assembly preflight", "scene": {"model": spec.base_layout_id,
            "source": "infinigen", "assembly_manifest": str(args.assembly.resolve())},
        "robot": {**robot_doc["robot"], "init_anchor": "start"}, "feasibility": robot_doc["feasibility"],
        "anchors": {"start": start}, "target": {"entity": spec.target.internal_entity, "category": spec.target.category},
        "objects": [], "placements": [], "skills": ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"]})
    args.out.mkdir(parents=True, exist_ok=False)
    backend = OmniGibsonBackend(seed=spec.seed)
    evidence = {"stage": "infinigen_gpu_preflight", "status": "pending_simulation_acceptance",
                "assembly_manifest_sha256": sha256_path(args.assembly), "evaluation_ready_episodes": 0,
                "pending_checks": list(manifest["pending_checks"]), "failures": []}
    try:
        # Callback intentionally captures no task entities; this is the native
        # occupancy bundle. It is NOT yet a topology-compatible family snapshot.
        def capture(b):
            for item in manifest["candidate_bindings"]:
                if not b.set_open(item["name"], False):
                    raise ValueError(f"INITIAL_CLOSE_FAILED:{item['name']}")
            b.settle(30); b.validate_physics_state(); b._validate_spawn_clearance()
            evidence["failures"] += runtime_articulation_checks(b, manifest)
            doc, state = capture_native_occupancy(b, slots, source_scene_path=args.assembly)
            for _ in range(2):
                b.restore_observe_state({"sim": state, "commanded_pose": None})
                verify_source(doc, b, slots)
            snapshot = args.out / "native_state.pt"
            save_snapshot(snapshot, state, {"source_state_sha256": doc["source_state_sha256"]})
            doc["snapshot_sha256"] = sha256_path(snapshot)
            doc.pop("inventory_sha256")
            doc["inventory_sha256"] = sha256_bytes(canonical_json(doc))
            atomic_write_json(args.out / "inventory.json", doc)
            evidence["native_occupancy_counts"] = doc["counts"]
            evidence["native_restore_repeats"] = 2
            return {"inventory_sha256": doc["inventory_sha256"]}
        backend.setup(scenario, native_scene_capture=capture, native_scene_only=True)
        image = backend.get_observation()
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("CAMERA_RGB_CONTRACT_FAILED")
        backend.render_snapshot(str(args.out / "initial_rgb.png"))
        camera = backend._camera_prim()
        evidence["camera"] = {"prim": str(camera.GetPath()), "shape": list(image.shape),
                              "protocol": "head_rgb_current_code; wrist_only_unverified"}
        evidence["appearance"] = getattr(backend._env, "_findingbench_appearance_evidence", {"status": "unavailable"})
        evidence["completed_checks"] = ["native_inventory", "native_restore_twice", "camera_rgb_shape",
                                        "loaded_articulation_query", "spawn_clearance"]
        # All checks requiring inserted targets, images, skills or reference
        # search remain pending even if this limited preflight passes.
        if evidence["failures"]:
            evidence["status"] = "rejected_preflight"
        atomic_write_json(args.out / "preflight.json", evidence)
        return evidence
    except Exception as exc:
        evidence.update(status="failed_preflight", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(args.out / "preflight.json", evidence)
        raise
    finally:
        backend.close()


def gpu_freeze_infinigen(args) -> dict:
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.infinigen_adapter import freeze_family_background
    family = json.loads((args.family_root / "family.json").read_text())
    if family.get("status") != "pending_simulation":
        raise ValueError("FAMILY_NOT_PENDING_ASSEMBLY_CHECK")
    scenarios = [load_scenario(args.family_root / "candidates" / f"{tid}.yaml") for tid in family["task_ids"]]
    if sha256_path(args.assembly) != family["assembly_manifest_sha256"]:
        raise ValueError("FAMILY_ASSEMBLY_CHANGED")
    backend = OmniGibsonBackend(seed=0)
    try:
        return freeze_family_background(backend, scenarios, args.assembly, args.out)
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
    default_policy = REPO / "assets/tasks/household_families_v1.yaml"
    plan = sub.add_parser("plan-infinigen", help="CPU: one or a few complete proposal families")
    plan.add_argument("--policy", type=Path, default=default_policy)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--split-seed", type=int, default=0)
    plan.add_argument("--count", type=int, default=1)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--upstream-root", type=Path, required=True)
    plan.add_argument("--python", required=True)
    run = sub.add_parser("run-infinigen-cpu", help="CPU device sandbox: actual pinned upstream geometry")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--policy", type=Path, default=default_policy)
    run.add_argument("--upstream-root", type=Path, required=True)
    run.add_argument("--python", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=300.)
    run.add_argument("--retries", type=int, default=0)
    inspect = sub.add_parser("inspect-infinigen", help="CPU standalone OpenUSD: real export inspection")
    inspect.add_argument("--scene", type=Path, required=True)
    inspect.add_argument("--solve-state", type=Path, required=True)
    inspect.add_argument("--report", type=Path)
    prepare = sub.add_parser("prepare-infinigen", help="CPU: real asset replacements and retained background")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--policy", type=Path, default=default_policy)
    prepare.add_argument("--scene", type=Path, required=True)
    prepare.add_argument("--solve-state", type=Path, required=True)
    prepare.add_argument("--bindings", type=Path, required=True)
    prepare.add_argument("--assets-root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    bind = sub.add_parser("bind-infinigen", help="CPU: compile full family using a separately GPU-probed overlay")
    bind.add_argument("--spec", type=Path, required=True)
    bind.add_argument("--policy", type=Path, default=default_policy)
    bind.add_argument("--assembly", type=Path, required=True)
    bind.add_argument("--overlay", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    preflight = sub.add_parser("gpu-preflight-infinigen", help="GPU ONLY: native, articulation, RGB and restore preflight; no release")
    preflight.add_argument("--spec", type=Path, required=True)
    preflight.add_argument("--policy", type=Path, default=default_policy)
    preflight.add_argument("--assembly", type=Path, required=True)
    preflight.add_argument("--start", type=Path, required=True, help="explicit initial pose JSON; collision checked at runtime")
    preflight.add_argument("--out", type=Path, required=True)
    freeze = sub.add_parser("gpu-freeze-infinigen", help="GPU ONLY: freeze background once and compare all target insertions")
    freeze.add_argument("--family-root", type=Path, required=True)
    freeze.add_argument("--assembly", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.stage == "gpu-freeze-infinigen":
        write_report(None, gpu_freeze_infinigen(args))
        return 0
    if args.stage == "gpu-preflight-infinigen":
        write_report(None, gpu_infinigen_preflight(args))
        return 0
    if args.stage == "plan-infinigen":
        plan_infinigen(args)
        return 0
    if args.stage == "run-infinigen-cpu":
        report = cpu_generate(args)
        write_report(None, report)
        return 0 if report["status"] == "geometry_generated" else 2
    if args.stage == "inspect-infinigen":
        from rummagebench.authoring.tasks.infinigen import inspect_export
        write_report(args.report, inspect_export(args.scene, args.solve_state))
        return 0
    if args.stage == "prepare-infinigen":
        write_report(None, prepare_infinigen(args))
        return 0
    if args.stage == "bind-infinigen":
        write_report(None, bind_infinigen(args))
        return 0

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
    try:
        result = main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                          "evaluation_ready_episodes": 0}, ensure_ascii=False))
        result = 2
    raise SystemExit(result)
