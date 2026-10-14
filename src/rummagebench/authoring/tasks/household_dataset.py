"""Family-level validation, local dataset publication and truthful stage counts.

Reuses TaskPlan, ScenarioSpec, frozen snapshots and the existing certifier.
Candidate files and individual oracle certificates never release a family.
The dataset is an evaluator-local index; restricted assets remain in place.
"""
from __future__ import annotations

import json
from pathlib import Path

from rummagebench.authoring.level1.snapshot import write_environment_snapshot
from rummagebench.authoring.tasks.household import HouseholdTaskSpec, family_members
from rummagebench.authoring.tasks.household_binding import validate_family_index
from rummagebench.authoring.tasks.household_state import bundle_file, load_frozen_member
from rummagebench.authoring.tasks.plan import TaskPlan, canonical_json, sha256_bytes, sha256_path
from rummagebench.core.scenario import load_scenario


def validate_authoring_spec(spec) -> None:
    from rummagebench.authoring.tasks.household import validate_spec_assets
    repo = Path(__file__).resolve().parents[4]
    validate_spec_assets(spec, repo, repo / "assets/tasks/household_families_v1.yaml")


def load_compiled_family(root: Path) -> tuple[dict, dict, dict]:
    root = Path(root).resolve()
    family = json.loads((root / "family.json").read_text())
    validate_family_index(family)
    if family.get("version") != 2 or family.get("status") != "pending_simulation":
        raise ValueError("PENDING_COMPILED_FAMILY_REQUIRED")
    expected = {"overlay.json", *[f"plans/{e}.json" for e in family["task_ids"]],
                *[f"candidates/{e}.yaml" for e in family["task_ids"]],
                *[f"specs/{s}.json" for s in family["specification_ids"]]}
    if set(family.get("artifact_hashes", {})) != expected:
        raise ValueError("COMPLETE_COMPILED_FAMILY_ARTIFACT_LEDGER_REQUIRED; rebind legacy families")
    for ref, digest in family["artifact_hashes"].items():
        bundle_file(root, ref, digest)
    for folder, suffix in (("plans", ".json"), ("candidates", ".yaml"), ("specs", ".json")):
        expected_ids = family["specification_ids"] if folder == "specs" else family["task_ids"]
        if {p.stem for p in (root / folder).glob("*" + suffix)} != set(expected_ids):
            raise ValueError("COMPILED_FAMILY_EXTRA_OR_MISSING_ARTIFACT")
    specs = {e: HouseholdTaskSpec.model_validate_json((root / "specs" / f"{s}.json").read_text())
             for s, e in family["specification_to_episode_id"].items()}
    plans = {e: TaskPlan.load(root / "plans" / f"{e}.json") for e in family["task_ids"]}
    planned = family_members(next(iter(specs.values())))
    if {s.task_id: s for s in planned} != {s.task_id: s for s in specs.values()}:
        raise ValueError("COMPILED_SPECIFICATIONS_NOT_COMPLETE_MATCHED_FAMILY")
    for episode, spec in specs.items():
        validate_authoring_spec(spec)
        plan = plans[episode]
        scenario = load_scenario(root / "candidates" / f"{episode}.yaml")
        if (plan.task_id != episode or scenario.id != episode or spec.task_id != family["episode_to_specification_id"][episode]
                or spec.split_assignment.model_dump(mode="json") != family["split_assignment"]
                or plan.provenance.get("household_spec_id") != spec.task_id
                or plan.provenance.get("family_id") != family["family_id"]
                or scenario.scene.frozen_member_manifest is not None
                or scenario.scene.source != "infinigen"
                or sha256_path(scenario.scene.assembly_manifest) != family["assembly_manifest_sha256"]):
            raise ValueError("COMPILED_FAMILY_SCENARIO_PLAN_OR_SPECIFICATION_MISMATCH")
    return family, specs, plans


def implementation_fingerprint() -> dict:
    from rummagebench.authoring.level1.robot_registry import runtime_protocol_hash
    package = Path(__file__).resolve().parents[2]
    # Covers builder/reset, public actions, feasibility, reference policy,
    # oracle and family gates. A changed implementation requires revalidation.
    return {"protocol_sha256": runtime_protocol_hash(),
            "sources": {str(p.relative_to(package)): sha256_path(p)
                        for p in sorted(package.rglob("*.py"))}}


def verify_generation_admission(spec, receipt: Path, solve: Path) -> dict:
    from rummagebench.authoring.tasks.infinigen import inspect_geometry_receipt
    inspection = inspect_geometry_receipt(spec, receipt)
    matched = [c for c in inspection["completed"]
               if c["artifacts"].get(str(Path(c["folder"]) / "solve_state.json")) == sha256_path(solve)]
    if len(matched) != 1 or matched[0]["measurement"]["proxy_filter_passed"] is not True:
        raise ValueError("ASSEMBLY_REQUIRES_ACCEPTED_COMPLETE_GENERATION_FOR_THIS_SOLVE_STATE")
    return {"receipt": str(receipt.resolve()), "receipt_sha256": sha256_path(receipt),
            "solve_state_sha256": sha256_path(solve), "measurement": matched[0]["measurement"],
            "scope": "cpu_proxy_admitted; physical_acceptance_pending"}


def validation_inputs(family_root: Path, frozen_root: Path) -> dict:
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
    from rummagebench.authoring.tasks.infinigen_probe import verify_overlay_evidence
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
    family, specs, _ = load_compiled_family(family_root)
    episode = family["task_ids"][0]
    scenario = load_scenario(family_root / "candidates" / f"{episode}.yaml")
    manifest = load_assembly_manifest(Path(scenario.scene.assembly_manifest))
    admission = manifest.get("generation_admission")
    if not admission:
        raise ValueError("ASSEMBLY_GENERATION_ADMISSION_REQUIRED; prepare using an accepted generation receipt")
    first_spec = next(iter(specs.values()))
    if verify_generation_admission(first_spec, Path(admission["receipt"]), Path(manifest["source"]["solve_state"])) != admission:
        raise ValueError("ASSEMBLY_GENERATION_ADMISSION_CHANGED")
    overlay = SlotEmbodimentOverlay.load(family_root / "overlay.json")
    probe = verify_overlay_evidence(overlay)
    if probe["fingerprint"]["assembly_sha256"] != family["assembly_manifest_sha256"]:
        raise ValueError("VALIDATION_PROBE_ASSEMBLY_MISMATCH")
    preflight = json.loads((frozen_root / "family_preflight.json").read_text())
    payload = {k: v for k, v in preflight.items() if k != "receipt_sha256"}
    if (preflight.get("receipt_sha256") != sha256_bytes(canonical_json(payload))
            or preflight.get("gpu_executed") is not True
            or preflight.get("background_and_members") != "frozen_and_restore_checked"
            or preflight.get("assembly_manifest_sha256") != family["assembly_manifest_sha256"]
            or preflight.get("status") != "pending_simulation_acceptance"):
        raise ValueError("COMPLETE_GPU_FROZEN_FAMILY_RECEIPT_REQUIRED")
    for ref, digest in preflight["artifacts"].items():
        bundle_file(frozen_root, ref, digest)
    members = preflight["members"]
    if (len(members) != len(family["task_ids"])
            or {m["scenario_id"] for m in members} != set(family["task_ids"])
            or any(m["background_match"] is not True or m["restore_repeats"] < 2 for m in members)):
        raise ValueError("FROZEN_FAMILY_COMPLETE_MEMBER_SET_REQUIRED")
    states = {}
    for m in members:
        scenario = load_scenario(family_root / "candidates" / f"{m['scenario_id']}.yaml")
        member_path = bundle_file(frozen_root, m["member_manifest"], preflight["artifacts"][m["member_manifest"]])
        load_frozen_member(member_path, scenario)
        states[m["scenario_id"]] = str(member_path)
    return {"family_root": str(family_root.resolve()), "frozen_root": str(frozen_root.resolve()),
            "family_sha256": sha256_path(family_root / "family.json"),
            "frozen_receipt_sha256": sha256_path(frozen_root / "family_preflight.json"),
            "assembly": scenario.scene.assembly_manifest, "assembly_sha256": family["assembly_manifest_sha256"],
            "generation_admission": admission,
            "probe_fingerprint": probe["fingerprint"], "implementation": implementation_fingerprint(),
            "member_manifests": states, "required_checks": manifest["pending_checks"],
            "split_assignment": next(iter(specs.values())).split_assignment.model_dump(mode="json")}


def prepare_family_validation(family_root: Path, frozen_root: Path, output: Path, python: str, entry: Path) -> dict:
    inputs = validation_inputs(family_root.resolve(), frozen_root.resolve())
    family, _, _ = load_compiled_family(family_root)
    request = {"version": 1, "status": "pending_simulation", "gpu_executed": False,
               "family_id": family["family_id"], "episode_ids": family["task_ids"], "inputs": inputs}
    request["request_sha256"] = sha256_bytes(canonical_json(request))
    commands = [[python, str(entry.resolve()), "gpu-certify-infinigen-member", "--request",
                 str((output / "request.json").resolve()), "--episode", e,
                 "--out", str((output / "results" / e).resolve())] for e in family["task_ids"]]
    write_environment_snapshot(output, {"request.json": request,
        "commands.json": {"commands": commands, "execution_status": "pending_gpu",
                          "one_process_per_member": True, "evaluation_ready_episodes": 0}})
    return {"status": "pending_simulation", "request": str(output / "request.json"),
            "counts": {"compiled_candidates": len(commands), "simulator_validated_episodes": 0,
                       "evaluation_ready_episodes": 0}, "gpu_started": False, "commands": commands}


def load_validation_request(path: Path) -> dict:
    doc = json.loads(path.read_text())
    payload = {k: v for k, v in doc.items() if k != "request_sha256"}
    if (doc.get("version") != 1 or doc.get("status") != "pending_simulation"
            or doc.get("gpu_executed") is not False
            or doc.get("request_sha256") != sha256_bytes(canonical_json(payload))):
        raise ValueError("VALIDATION_REQUEST_HASH_OR_STATUS_INVALID")
    inputs = doc["inputs"]
    if validation_inputs(Path(inputs["family_root"]), Path(inputs["frozen_root"])) != inputs:
        raise ValueError("VALIDATION_INPUT_OR_IMPLEMENTATION_CHANGED")
    family, _, _ = load_compiled_family(Path(inputs["family_root"]))
    if doc["episode_ids"] != family["task_ids"] or doc["family_id"] != family["family_id"]:
        raise ValueError("VALIDATION_REQUEST_COMPLETE_FAMILY_REQUIRED")
    return doc


def collect_family_validation(request_path: Path, results: Path) -> dict:
    """CPU verifier: recompute gates and costs, never trust a certificate label."""
    from rummagebench.authoring.tasks import gates
    from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate
    from rummagebench.authoring.tasks.household_search import summarize_reference_search
    from rummagebench.authoring.tasks.tiers import load_tier
    request = load_validation_request(request_path)
    family_root = Path(request["inputs"]["family_root"])
    family, specs, _ = load_compiled_family(family_root)
    from rummagebench.authoring.tasks.household_search import reference_order
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly
    overlay = SlotEmbodimentOverlay.load(family_root / "overlay.json")
    manifest = load_assembly_manifest(Path(request["inputs"]["assembly"]))
    slots = scene_slots_from_assembly(Path(request["inputs"]["assembly"]))
    location_for_slot = {next(s.slot_id for s in slots.slots if s.parent_entity == b["name"]): b["location_id"]
                         for b in manifest["candidate_bindings"]}
    scenario0 = load_scenario(family_root / "candidates" / f"{family['task_ids'][0]}.yaml")
    start = scenario0.anchors[scenario0.robot.init_anchor].position[:2]
    blind_order = [location_for_slot[s] for s in reference_order(start,
                    {sid: list(overlay.slots[sid].navigation_anchor.xy) for sid in location_for_slot})]
    failures, pending, records, runs, certified_members = [], [], {}, {}, set()
    expected = set(request["episode_ids"])
    extras = {p.parent.name for p in results.glob("*/result.json")} - expected
    if extras:
        raise ValueError("VALIDATION_RESULTS_CONTAIN_EXTRA_EPISODES")
    for episode in request["episode_ids"]:
        path = results / episode / "result.json"
        if not path.is_file():
            pending.append(f"member:{episode}")
            continue
        record = json.loads(path.read_text())
        payload = {k: v for k, v in record.items() if k != "receipt_sha256"}
        if (record.get("receipt_sha256") != sha256_bytes(canonical_json(payload))
                or record.get("episode_id") != episode or record.get("request_sha256") != request["request_sha256"]
                or record.get("gpu_executed") is not True):
            raise ValueError("MEMBER_RECEIPT_NOT_BOUND_TO_VALIDATION_REQUEST")
        for ref, digest in record.get("artifacts", {}).items():
            bundle_file(path.parent, ref, digest)
        cert = SearchStructureCertificate.from_dict(record["search_certificate"])
        tier = load_tier("easy_v1")
        tier.budget.max_planning_steps = specs[episode].max_planning_steps
        violations = gates.certify(cert, tier)
        if (violations or record.get("error") or record.get("status") != "certified"
                or cert.candidate_slot_count != len(specs[episode].locations)
                or set(cert.candidate_slots) != set(location_for_slot)
                or len(cert.candidate_slots) != len(location_for_slot)
                or cert.reachable_candidate_slot_count != len(specs[episode].locations)):
            failures.append(f"MEMBER_CERTIFICATION_FAILED:{episode}")
        else:
            certified_members.add(episode)
        for check in request["inputs"]["required_checks"]:
            status = record.get("checks", {}).get(check, "pending")
            if status == "failed":
                failures.append(f"{check}:{episode}")
            elif status != "passed":
                pending.append(f"{check}:{episode}")
        records[episode] = {"path": str(path.resolve()), "sha256": sha256_path(path)}
        if record["reference_search"].get("candidate_order") != blind_order:
            failures.append(f"REFERENCE_ORDER_DIFFERS_FROM_TARGET_BLIND_POLICY:{episode}")
        if len(record["reference_search"].get("trace", [])) > specs[episode].max_planning_steps:
            failures.append(f"REFERENCE_EXCEEDS_PUBLIC_HORIZON:{episode}")
        runs[specs[episode].target_location] = record["reference_search"]
    search = None
    if len(runs) == len(expected):
        first = next(iter(specs.values()))
        try:
            search = summarize_reference_search(runs, {l.location_id: l.prior for l in first.locations},
                                                first.generation_constraints["difficulty_intervals"])
            failures.extend(search["rejections"])
        except ValueError as exc:
            failures.append(f"REFERENCE_SEARCH_EVIDENCE_INVALID:{exc}")
    ready = not failures and not pending and search is not None
    # Simulator episode count requires all member gates, separately from
    # additional family/image/semantic checks needed to release data.
    return {"version": 1, "family_id": family["family_id"],
            "status": "validated" if ready else "rejected" if failures else "pending_simulation",
            "request": str(request_path.resolve()), "request_sha256": sha256_path(request_path),
            "members": records, "rejections": sorted(set(failures)), "pending_checks": sorted(set(pending)),
            "search_measurement": search, "split_assignment": family["split_assignment"],
            "counts": {"compiled_candidates": len(expected),
                       "member_receipts": len(records), "validated_families": int(ready),
                       "simulator_validated_episodes": len(certified_members),
                       "evaluation_ready_episodes": len(expected) if ready else 0}}


def publish_family(request: Path, results: Path, output: Path) -> dict:
    report = collect_family_validation(request, results)
    if report["status"] != "validated":
        raise ValueError(f"FAMILY_PUBLICATION_BLOCKED:{report['rejections'] + report['pending_checks']}")
    doc = load_validation_request(request)
    root = Path(doc["inputs"]["family_root"])
    family, specs, _ = load_compiled_family(root)
    artifacts = {"acceptance.json": report}
    public, episodes = [], []
    for episode in family["task_ids"]:
        scenario = load_scenario(root / "candidates" / f"{episode}.yaml")
        scenario.scene.frozen_member_manifest = doc["inputs"]["member_manifests"][episode]
        artifacts[f"scenarios/{episode}.yaml"] = scenario.model_dump(mode="json")
        public.append({"episode_id": episode, "split": family["split"][episode], **specs[episode].public_task()})
        episodes.append({"episode_id": episode, "scenario": f"scenarios/{episode}.yaml",
                         "split": family["split"][episode], "robot": "r1pro",
                         "difficulty": report["search_measurement"]["reference_difficulty"],
                         "difficulty_basis": "measured_reference; optimality_not_proved"})
    artifacts["public_tasks.json"] = {"tasks": public}
    dataset = {"version": 1, "status": "validated", "family_id": family["family_id"],
               "episodes": episodes, "split_assignment": family["split_assignment"],
               "counts": report["counts"], "scope": "evaluator_local; no licensed asset redistribution",
               "validation_request": str(request.resolve()), "validation_results": str(results.resolve()),
               "artifact_hashes": {ref: sha256_bytes(canonical_json(value)) for ref, value in artifacts.items()}}
    artifacts["dataset.json"] = dataset
    write_environment_snapshot(output, artifacts)
    return dataset


def load_dataset(root: Path) -> dict:
    """Recheck live acceptance and transitive dependencies before consuming data."""
    doc = json.loads((root / "dataset.json").read_text())
    report = collect_family_validation(Path(doc["validation_request"]), Path(doc["validation_results"]))
    if (doc.get("version") != 1 or doc.get("status") != "validated" or report["status"] != "validated"
            or doc["counts"] != report["counts"] or doc["family_id"] != report["family_id"]):
        raise ValueError("DATASET_ACCEPTANCE_NO_LONGER_VALID")
    for ref, digest in doc["artifact_hashes"].items():
        bundle_file(root, ref, digest)
    episodes = doc["episodes"]
    if len({e["episode_id"] for e in episodes}) != len(episodes) or len(episodes) != doc["counts"]["evaluation_ready_episodes"]:
        raise ValueError("DATASET_EPISODE_INDEX_INVALID")
    request = load_validation_request(Path(doc["validation_request"]))
    family_root = Path(request["inputs"]["family_root"])
    family, specs, _ = load_compiled_family(family_root)
    required = {"public_tasks.json", "acceptance.json", *[f"scenarios/{e}.yaml" for e in family["task_ids"]]}
    if set(doc["artifact_hashes"]) != required or {e["episode_id"] for e in episodes} != set(family["task_ids"]):
        raise ValueError("DATASET_COMPLETE_FAMILY_LEDGER_REQUIRED")
    for episode in episodes:
        tid = episode["episode_id"]
        scenario = load_scenario(family_root / "candidates" / f"{tid}.yaml")
        scenario.scene.frozen_member_manifest = request["inputs"]["member_manifests"][tid]
        if (episode["scenario"] != f"scenarios/{tid}.yaml" or episode["split"] != family["split"][tid]
                or episode["difficulty"] != report["search_measurement"]["reference_difficulty"]
                or load_scenario(root / episode["scenario"]) != scenario):
            raise ValueError("DATASET_SCENARIO_OR_SPLIT_DIFFERS_FROM_ACCEPTED_FAMILY")
    expected_public = [{"episode_id": tid, "split": family["split"][tid], **specs[tid].public_task()}
                       for tid in family["task_ids"]]
    if json.loads((root / "public_tasks.json").read_text()) != {"tasks": expected_public}:
        raise ValueError("DATASET_PUBLIC_TASK_CONTENT_INVALID")
    return doc


def pipeline_inventory(root: Path) -> dict:
    """Inspect recognized, hash-checked artifacts; do not count arbitrary JSON."""
    from collections import Counter
    from rummagebench.authoring.tasks.household import UPSTREAM_COMMIT, source_layout_group_id
    from rummagebench.authoring.tasks.infinigen import _blend_valid, read_solve_state
    root = root.resolve()
    specs, sources, layouts, geometry, accepted, exports, compiled, simulated, validated, ready = (set() for _ in range(10))
    failures = Counter()
    for path in sorted(root.glob("**/specs/task_*.json")):
        try:
            spec = HouseholdTaskSpec.model_validate_json(path.read_text())
            if spec.task_id != path.stem:
                raise ValueError("SPECIFICATION_FILENAME_ID_MISMATCH")
            specs.add(spec.task_id)
            layouts.add(spec.base_layout_id)
            sources.add(source_layout_group_id(spec.upstream_commit, spec.scene_seed))
        except (OSError, ValueError) as exc:
            failures[f"invalid_spec:{type(exc).__name__}"] += 1
    for path in sorted(root.glob("**/generation.json")):
        try:
            receipt = json.loads(path.read_text())
            source = json.loads((path.parent / "input.json").read_text())
            inputs = source["inputs"]
            if (receipt.get("input_sha256") != sha256_bytes(canonical_json(inputs))
                    or source["input_sha256"] != receipt["input_sha256"]
                    or receipt.get("source", {}).get("commit") != UPSTREAM_COMMIT
                    or inputs["runtime"]["packages"].get("bpy") != "4.2.0"
                    or not inputs["runtime"].get("executable_sha256")
                    or receipt.get("gpu_started") is not False
                    or receipt.get("gpu_device_isolation") != "bwrap_private_dev_no_nvidia_or_drm"):
                raise ValueError("REAL_PINNED_CPU_GENERATION_RECEIPT_REQUIRED")
            for ref, digest in receipt["artifacts"].items():
                bundle_file(path.parent, ref, digest)
            for attempt in receipt["attempts"]:
                if attempt.get("reason"):
                    failures[attempt["reason"]] += 1
                if attempt.get("exit_code") != 0:
                    continue
                folder = path.parent / f"attempt_{attempt['retry']}" / "geometry"
                blend, solve, sidecar = (folder / name for name in ("scene.blend", "solve_state.json", "findingbench_geometry.json"))
                if not _blend_valid(blend) or not sidecar.is_file() or not read_solve_state(solve)["kitchens"]:
                    raise ValueError("GENERATION_ACTUAL_GEOMETRY_INCOMPLETE")
                # Each counted file must have been protected by the original
                # invocation; fixtures without runtime/artifact evidence fail.
                for file in (blend, solve, sidecar):
                    rel = str(file.relative_to(path.parent))
                    if receipt["artifacts"].get(rel) != sha256_path(file):
                        raise ValueError("GENERATION_ARTIFACT_LEDGER_INCOMPLETE")
                digest = sha256_path(blend)
                geometry.add(digest)
                if attempt.get("reason") is None and attempt.get("measurement", {}).get("proxy_filter_passed") is True:
                    accepted.add(digest)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[f"invalid_generation:{type(exc).__name__}"] += 1
    # Older runs rejected by the former tag parser have separate immutable
    # reinspection evidence. Count those real files once, retaining the old
    # failure reason; do not rewrite the original generation receipt.
    for path in sorted(root.glob("**/geometry_inspection*.json")):
        try:
            inspection = json.loads(path.read_text())
            original = Path(inspection["original_receipt"])
            receipt = json.loads(original.read_text())
            source = json.loads((original.parent / "input.json").read_text())
            if (not original.resolve().is_relative_to(root) or sha256_path(original) != inspection["original_receipt_sha256"]
                    or inspection.get("stage") != "inspect_infinigen_geometry" or inspection.get("gpu_started") is not False
                    or receipt.get("source", {}).get("commit") != UPSTREAM_COMMIT
                    or source["inputs"]["runtime"]["packages"].get("bpy") != "4.2.0"
                    or receipt["input_sha256"] != source["input_sha256"]):
                raise ValueError("GEOMETRY_REINSPECTION_SOURCE_REQUIRED")
            for completed in inspection["completed"]:
                folder = Path(completed["folder"])
                if not folder.resolve().is_relative_to(original.parent):
                    raise ValueError("GEOMETRY_REINSPECTION_FOLDER_INVALID")
                if not any(a.get("exit_code") == 0 and a["command"]["request"] == inspection["background_request"]
                           and folder == original.parent / f"attempt_{a['retry']}" / "geometry" for a in receipt["attempts"]):
                    raise ValueError("GEOMETRY_REINSPECTION_COMPLETED_INVOCATION_REQUIRED")
                for ref, digest in completed["artifacts"].items():
                    if not Path(ref).resolve().is_relative_to(folder) or sha256_path(ref) != digest:
                        raise ValueError("GEOMETRY_REINSPECTION_ARTIFACT_CHANGED")
                blend = folder / "scene.blend"
                if not _blend_valid(blend) or not read_solve_state(folder / "solve_state.json")["kitchens"]:
                    raise ValueError("GEOMETRY_REINSPECTION_ACTUAL_KITCHEN_REQUIRED")
                geometry.add(sha256_path(blend))
                if completed["measurement"]["proxy_filter_passed"] is True:
                    accepted.add(sha256_path(blend))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[f"invalid_reinspection:{type(exc).__name__}"] += 1
    for path in sorted(root.glob("**/family.json")):
        try:
            family, _, _ = load_compiled_family(path.parent)
            compiled.update(family["task_ids"])
            # Count exports only if a complete assembly is actually loaded.
            scenario = load_scenario(path.parent / "candidates" / f"{family['task_ids'][0]}.yaml")
            from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
            manifest = load_assembly_manifest(Path(scenario.scene.assembly_manifest))
            exports.add(manifest["source"]["scene_sha256"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[f"invalid_compiled_family:{type(exc).__name__}"] += 1
    for path in sorted(root.glob("**/assembly.json")):
        try:
            from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
            exports.add(load_assembly_manifest(path)["source"]["scene_sha256"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[f"invalid_assembly:{type(exc).__name__}"] += 1
    for path in sorted(root.glob("**/dataset.json")):
        try:
            dataset = load_dataset(path.parent)
            ready.update(e["episode_id"] for e in dataset["episodes"])
            validated.add(dataset["family_id"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[f"invalid_dataset:{type(exc).__name__}"] += 1
    for path in sorted(root.glob("**/request.json")):
        try:
            request = json.loads(path.read_text())
            if "episode_ids" not in request or "family_root" not in request.get("inputs", {}):
                continue  # probe requests are a different stage, not episodes
            report = collect_family_validation(path, path.parent / "results")
            failed = {r.split(":",1)[1] for r in report["rejections"] if r.startswith("MEMBER_CERTIFICATION_FAILED:")}
            simulated.update(set(report["members"]) - failed)
            if report["status"] == "validated":
                validated.add(report["family_id"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures[f"invalid_validation:{type(exc).__name__}"] += 1
    return {"stage": "infinigen_dataset_status", "status": "validated" if ready else "pending_simulation",
            "counts": {"task_specifications": len(specs), "source_layout_groups": len(sources),
                       "requested_base_layouts": len(layouts), "actual_geometry_outputs": len(geometry),
                       "proxy_accepted_geometry_outputs": len(accepted), "verified_exported_scenes": len(exports),
                       "compiled_candidates": len(compiled), "validated_families": len(validated),
                       "simulator_validated_episodes": len(simulated),
                       "evaluation_ready_episodes": len(ready)},
            "failure_reasons": dict(sorted(failures.items())),
            "counting_scope": "content-deduplicated actual artifacts; geometry variants are not independent source layouts",
            "gpu_started_by_this_command": False, "training_started_by_this_command": False}
