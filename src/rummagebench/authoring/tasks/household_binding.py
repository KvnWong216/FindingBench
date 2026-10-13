"""Bind complete household families to existing TaskPlan/compiler contracts.

This is preparation, never release acceptance. Requires a separately probed
embodiment overlay bound to this assembly. No location is silently discarded.
"""
from __future__ import annotations

import json
from pathlib import Path

from rummagebench.authoring.level1.snapshot import write_environment_snapshot
from rummagebench.authoring.tasks.compile import compile_plan
from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
from rummagebench.authoring.tasks.household import HouseholdTaskSpec, grouped_split
from rummagebench.authoring.tasks.plan import PlacedObject, TaskPlan, canonical_json, sha256_bytes, sha256_path
from rummagebench.authoring.tasks.tiers import load_tier
from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly


def bind_family(members: list[HouseholdTaskSpec], assembly: Path, overlay_path: Path, output: Path) -> dict:
    if not members:
        raise ValueError("EMPTY_FAMILY")
    doc = load_assembly_manifest(assembly)
    if doc.get("target_category_confounds"):
        raise ValueError("NATIVE_TARGET_CATEGORY_AMBIGUOUS; preserve background and reject family or supply verified descriptive instruction")
    slots = scene_slots_from_assembly(assembly)
    overlay = SlotEmbodimentOverlay.load(overlay_path)
    first = members[0]
    locations = {s.location_id for s in first.locations}
    if len(members) != len(locations) or {s.target_location for s in members} != locations:
        raise ValueError("INCOMPLETE_TARGET_POSITION_FAMILY")
    if any(s.family_id != first.family_id or s.base_layout_id != first.base_layout_id
           or s.target != first.target or s.locations != first.locations
           or s.appearance != first.appearance or s.robot_config_ref != first.robot_config_ref for s in members):
        raise ValueError("FAMILY_BACKGROUND_OR_VARIABLES_DIFFER")
    if doc["scene_id"] != first.base_layout_id or doc["family_id"] != first.family_id:
        raise ValueError("ASSEMBLY_FAMILY_MISMATCH")
    if (overlay.scene != slots.scene or overlay.robot_id != first.robot_id or not overlay.probed
            or f"assembly_manifest_sha256:{sha256_path(assembly)}" not in overlay.notes
            or not overlay.urdf_hash or sha256_path(overlay.urdf_path) != overlay.urdf_hash):
        raise ValueError("EMBODIMENT_OVERLAY_NOT_PROBED_FOR_THIS_ASSEMBLY")
    by_location = {b["location_id"]: next(s for s in slots.slots if s.parent_entity == b["name"])
                   for b in doc["candidate_bindings"]}
    if set(by_location) != locations:
        raise ValueError("CANDIDATE_MAPPING_INCOMPLETE")
    for slot in by_location.values():
        inter = overlay.slots.get(slot.slot_id)
        if inter is None or not inter.probed or not inter.usable(True) or inter.open_causal is not True:
            raise ValueError(f"CANDIDATE_REQUIRES_REAL_OPEN_GRASP_PROBE:{slot.slot_id}")
    starts = sorted(overlay.start_poses.get("kitchen_0", []), key=lambda p: p.name)
    if not starts:
        raise ValueError("ROBOT_START_NOT_PROBED")
    start = starts[0]  # independent of actual target location
    tier = load_tier("easy_v1")
    tier = tier.model_copy(update={"budget": tier.budget.model_copy(update={"max_planning_steps": first.max_planning_steps})})
    artifacts, ids = {}, []
    for spec in members:
        target_slot = by_location[spec.target_location]
        plan = TaskPlan("easy", 1, "in_container", spec.seed, spec.index, slots.scene, "kitchen_0", "kitchen",
            spec.robot_id, PlacedObject(spec.target.internal_entity, spec.target.category, spec.target.model,
                                        target_slot.slot_id, "target"), [], start.name, start.pose.to_anchor(),
            spec.instruction, ["kitchen_0"], sorted(s.slot_id for s in by_location.values()),
            {"rearrangement_depth": 0, "open_depth": 1, "containment_depth": 1, "lookalike_count": 0},
            provenance={"family_id": spec.family_id, "background_id": spec.base_layout_id,
                        "family_binding": "infinigen_pending_simulation", "household_spec_id": spec.task_id,
                        "inputs": {**spec.input_hashes, "assembly_manifest": sha256_path(assembly),
                                   "overlay_hash": sha256_path(overlay_path)},
                        "robot": {"id": spec.robot_id, "urdf_hash": overlay.urdf_hash},
                        "search_difficulty": "unmeasured"})
        scenario = compile_plan(plan, slots, overlay, tier)
        ids.append(plan.task_id)
        artifacts[f"plans/{plan.task_id}.json"] = plan.to_dict()
        # JSON is valid YAML and load_scenario accepts the existing .yaml paths.
        artifacts[f"candidates/{plan.task_id}.yaml"] = scenario
    report = {"version": 1, "status": "pending_simulation", "family_id": first.family_id,
              "task_ids": ids, "counts": {"compiled_candidates": len(ids), "evaluation_ready_episodes": 0},
              "assembly_manifest_sha256": sha256_path(assembly), "split": grouped_split(members),
              "required_checks": doc["pending_checks"],
              "publication": "blocked; legacy single-episode certificates cannot release this family"}
    artifacts["family.json"] = report
    write_environment_snapshot(output, artifacts)
    return report
