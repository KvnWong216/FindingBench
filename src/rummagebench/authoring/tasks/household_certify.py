"""Explicit GPU worker for one frozen Infinigen member; no eager OG imports."""
from __future__ import annotations

from pathlib import Path

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path


def certify_member(request_path: Path, episode: str, output: Path) -> dict:
    from rummagebench.authoring.tasks.household_dataset import load_validation_request, load_compiled_family
    request = load_validation_request(request_path)
    if episode not in request["episode_ids"]:
        raise ValueError("MEMBER_NOT_IN_COMPLETE_VALIDATION_REQUEST")
    # Nothing capable of starting the simulator is imported before verifying
    # all input files, frozen states, probe evidence and implementations.
    from rummagebench.authoring.tasks.certify import TaskCertifier
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay, Pose2
    from rummagebench.authoring.tasks.infinigen_probe import GeometryRoomMap, verify_overlay_evidence
    from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate, navigation_lower_bound, minimum_required_interactions
    from rummagebench.authoring.tasks.household_search import run_reference_search
    from rummagebench.authoring.tasks.household_state import check_background, load_frozen_member
    from rummagebench.authoring.tasks.household_physics import verify_door_sweeps, verify_temporary_place
    from rummagebench.authoring.tasks.household_locations import inspect_loaded_search_locations
    from rummagebench.authoring.tasks.tiers import load_tier
    from rummagebench.authoring.tasks import gates
    from rummagebench.sim.omnigibson.infinigen_adapter import scene_slots_from_assembly, load_assembly_manifest, runtime_articulation_checks
    from rummagebench.core.scenario import load_scenario

    root = Path(request["inputs"]["family_root"])
    _, specs, plans = load_compiled_family(root)
    spec, plan = specs[episode], plans[episode]
    assembly = Path(request["inputs"]["assembly"])
    manifest = load_assembly_manifest(assembly)
    slots = scene_slots_from_assembly(assembly)
    overlay = SlotEmbodimentOverlay.load(root / "overlay.json")
    probe = verify_overlay_evidence(overlay)
    room_map = GeometryRoomMap.from_dict(probe["room_map"])
    scenario = load_scenario(root / "candidates" / f"{episode}.yaml")
    scenario.scene.frozen_member_manifest = request["inputs"]["member_manifests"][episode]
    required = sorted(s.slot_id for s in slots.slots)
    if required != sorted(plan.candidate_slots):
        raise ValueError("MEMBER_COMPLETE_CANDIDATE_SET_MISMATCH")
    target_slot = slots.by_id()[plan.target.slot_id]
    start, goal = Pose2.from_dict(plan.start), overlay.slots[target_slot.slot_id].navigation_anchor
    protocol = scenario.agent_protocol
    nav = navigation_lower_bound(start.xy, start.yaw, goal.xy, goal.yaw,
                                 protocol.max_move_cm / 100., protocol.max_turn_deg)
    cert = SearchStructureCertificate(plan.tier, plan.mode, 1, len(required), 1, 1, 0, 0,
        target_slot.slot_type, nav, minimum_required_interactions(1, 0, nav),
        candidate_slots=required, rooms=[plan.room])
    tier = load_tier("easy_v1")
    tier.budget.max_planning_steps = spec.max_planning_steps
    output.mkdir(parents=True, exist_ok=False)
    scenario_path = output / "frozen_scenario.yaml"
    atomic_write_json(scenario_path, scenario.model_dump(mode="json"))
    record = {"version": 1, "episode_id": episode, "request_sha256": request["request_sha256"],
              "gpu_executed": False, "status": "rejected", "checks": {}, "error": None}
    certifier = TaskCertifier(slots, overlay, room_map,
                             reveal_min=tier.gates.modes[plan.mode].generator_visibility_margin)
    try:
        record["gpu_executed"] = True  # explicit live worker invocation, including failed builds
        result = certifier.certify(plan, cert, scenario_path)
        cert.attach_evidence(result.evidence)
        violations = gates.certify(cert, tier)
        record.update(status=cert.status if not result.error else "rejected", error=result.error,
                      search_certificate=cert.to_dict(), rejections=[r.to_dict() for r in violations],
                      witness_trace=result.witness_trace, oracle_certificate=result.oracle_certificate,
                      details=result.details)
        checks = {name: "pending" for name in request["inputs"]["required_checks"]}
        record["checks"] = checks
        if result.evidence.builder_ok is True:
            backend = certifier.backend
            backend.reset()
            state_path = Path(scenario.scene.frozen_member_manifest)
            member = load_frozen_member(state_path, scenario)
            check_background(backend, member, state_path.parent.parent)
            # These narrow checks do not stand in for door sweep, temporary
            # placement space, replacement contact or visual semantic review.
            for name in ("native_occupancy", "background_stability", "target_insertion_background_match", "reset"):
                checks[name] = "passed"
            articulation_failures = runtime_articulation_checks(backend, manifest)
            checks["real_joints_limits_handles"] = "failed" if articulation_failures else "passed"
            record["articulation_failures"] = articulation_failures
            record["appearance"] = getattr(backend._env, "_findingbench_appearance_evidence", {})
            backend.render_snapshot(str(output / "initial_rgb.png"))
            from rummagebench.authoring.tasks.household_appearance import image_quality
            from PIL import Image
            import numpy as np
            with Image.open(output / "initial_rgb.png") as saved_image:
                record["image_quality"]=image_quality(np.asarray(saved_image),[scenario.robot.image_height,scenario.robot.image_width,3])
            # Passing numeric QC does not certify semantic appearance or
            # unimplemented texture swaps. Failed QC is an actual rejection.
            if record["image_quality"]["status"]=="failed":checks["appearance_images"]="failed"
            record["door_sweep"] = verify_door_sweeps(backend,manifest)
            checks["door_sweep"] = record["door_sweep"]["status"]
            check_background(backend,member,state_path.parent.parent)
            record["search_locations"] = inspect_loaded_search_locations(backend,manifest)
            checks["all_additional_counter_drawer_search_locations_classified"] = record["search_locations"]["status"]
            search = run_reference_search(backend, scenario, plan, slots, overlay, room_map)
            # Compiler slot IDs are distinct from household location IDs.
            location_for_slot = {next(s.slot_id for s in slots.slots if s.parent_entity == b["name"]): b["location_id"]
                                 for b in manifest["candidate_bindings"]}
            search["candidate_order"] = [location_for_slot[s] for s in search["candidate_order"]]
            search["visited_slots"] = [location_for_slot[s] for s in search["visited_slots"]]
            record["reference_search"] = search
            checks["search_cost"] = "passed" if search["status"] == "SUCCESS" else "failed"
            if cert.status == "certified" and search["status"] == "SUCCESS":
                checks["robot_skills"] = "passed"
                checks["visibility_and_observe"] = "failed" if search["closed_observe_bypass"] else "passed"
            backend.reset()
            record["temporary_place"] = verify_temporary_place(backend,scenario,plan,slots,overlay,room_map,manifest)
            checks["temporary_place_space"] = record["temporary_place"]["status"]
            check_background(backend,member,state_path.parent.parent)
            record["checks"] = checks
        else:
            record["reference_search"] = {"policy": "nearest-uninspected-anchor-v1", "gpu_executed": False,
                "candidate_order": [l.location_id for l in spec.locations], "visited_slots": [],
                "trace": [], "status": "ENGINE_ERROR"}
    except Exception as exc:
        record.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
        record.setdefault("search_certificate", cert.to_dict())
        record.setdefault("reference_search", {"policy": "nearest-uninspected-anchor-v1", "gpu_executed": False,
            "candidate_order": [l.location_id for l in spec.locations], "visited_slots": [], "trace": [], "status": "ENGINE_ERROR"})
    # Write before Kit shutdown, which may terminate the interpreter. Missing
    # records/crashes stay pending and never count as verified episodes.
    record["artifacts"] = {p.name: sha256_path(p) for p in output.iterdir() if p.is_file()}
    record["receipt_sha256"] = sha256_bytes(canonical_json(record))
    atomic_write_json(output / "result.json", record)
    return record
