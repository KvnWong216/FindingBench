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
from rummagebench.authoring.tasks.household import HouseholdTaskSpec, source_layout_group_id
from rummagebench.authoring.tasks.plan import PlacedObject, TaskPlan, canonical_json, sha256_bytes, sha256_path
from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly


def resolve_family_slots(bindings, slots):
    """Bind actual cavities explicitly; never take the first drawer of an asset."""
    result = {}
    for binding in bindings:
        location = binding['location_id']
        candidates = [s for s in slots.slots if s.parent_entity == binding['name']
                      and (binding.get('slot_id') is None or s.slot_id == binding['slot_id'])]
        if location in result or len(candidates) != 1:
            raise ValueError(f'FAMILY_LOCATION_REQUIRES_UNIQUE_EXPLICIT_SLOT:{location}')
        result[location] = candidates[0]
    if len({s.slot_id for s in result.values()}) != len(result):
        raise ValueError('DISTINCT_FAMILY_LOCATIONS_CANNOT_ALIAS_ONE_SLOT')
    return result


def bind_family(members: list[HouseholdTaskSpec], assembly: Path, overlay_path: Path, output: Path) -> dict:
    if not members:
        raise ValueError("EMPTY_FAMILY")
    doc = load_assembly_manifest(assembly)
    if doc.get('component_context',{}).get('complete_leaf_binding') == 'pending':
        raise ValueError('COMPLETE_KITCHEN_LEAF_BINDING_PENDING')
    if doc.get("target_category_confounds"):
        raise ValueError("NATIVE_TARGET_CATEGORY_AMBIGUOUS; preserve background and reject family or supply verified descriptive instruction")
    slots = scene_slots_from_assembly(assembly)
    overlay = SlotEmbodimentOverlay.load(overlay_path)
    first = members[0]
    if first.split_assignment is None or any(s.split_assignment != first.split_assignment for s in members):
        raise ValueError("PLANNED_SPLIT_REQUIRED_FOR_COMPLETE_FAMILY")
    if first.split_assignment.source_layout_group != source_layout_group_id(first.upstream_commit,first.scene_seed):
        raise ValueError("PLANNED_SOURCE_LAYOUT_SPLIT_GROUP_REQUIRED; replan legacy candidate specs")
    specification_ids = [s.task_id for s in members]
    if len(set(specification_ids)) != len(specification_ids):
        raise ValueError("DUPLICATE_SPECIFICATION_ID")
    locations = {s.location_id for s in first.locations}
    if len(members) != len(locations) or {s.target_location for s in members} != locations:
        raise ValueError("INCOMPLETE_TARGET_POSITION_FAMILY")
    if any(s.family_id != first.family_id or s.base_layout_id != first.base_layout_id
           or s.target != first.target or s.locations != first.locations
           or s.appearance != first.appearance or s.robot_config_ref != first.robot_config_ref
           or s.scene_seed != first.scene_seed or s.upstream_commit != first.upstream_commit
           or s.generation_constraints != first.generation_constraints or s.input_hashes != first.input_hashes
           or s.required_skills != first.required_skills or s.max_planning_steps != first.max_planning_steps
           or s.instruction != first.instruction or s.requested_difficulty != first.requested_difficulty for s in members):
        raise ValueError("FAMILY_BACKGROUND_OR_VARIABLES_DIFFER")
    if doc["scene_id"] != first.base_layout_id or doc["family_id"] != first.family_id:
        raise ValueError("ASSEMBLY_FAMILY_MISMATCH")
    if (overlay.scene != slots.scene or overlay.robot_id != first.robot_id or not overlay.probed
            or f"assembly_manifest_sha256:{sha256_path(assembly)}" not in overlay.notes
            or not overlay.urdf_hash or sha256_path(overlay.urdf_path) != overlay.urdf_hash):
        raise ValueError("EMBODIMENT_OVERLAY_NOT_PROBED_FOR_THIS_ASSEMBLY")
    from rummagebench.authoring.tasks.infinigen_probe import verify_overlay_evidence
    probe_request = verify_overlay_evidence(overlay)
    if probe_request["fingerprint"]["assembly_sha256"] != sha256_path(assembly):
        raise ValueError("PROBE_ASSEMBLY_MISMATCH")
    by_location = resolve_family_slots(doc.get('native_region_bindings',doc['candidate_bindings']), slots)
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
    from rummagebench.authoring.tasks.household import execution_tier
    tier=execution_tier(first)
    artifacts, ids, mapping = {}, [], {}
    for spec in members:
        target_slot = by_location[spec.target_location]
        plan = TaskPlan(tier.tier, tier.version, "in_container", spec.seed, spec.index, slots.scene, "kitchen_0", "kitchen",
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
        if plan.task_id in ids:
            raise ValueError("DUPLICATE_COMPILED_EPISODE_ID")
        ids.append(plan.task_id)
        mapping[spec.task_id] = plan.task_id
        artifacts[f"plans/{plan.task_id}.json"] = plan.to_dict()
        # JSON is valid YAML and load_scenario accepts the existing .yaml paths.
        artifacts[f"candidates/{plan.task_id}.yaml"] = scenario
    report = {"version": 2, "status": "pending_simulation", "family_id": first.family_id,
              "task_ids": ids, "counts": {"compiled_candidates": len(ids), "evaluation_ready_episodes": 0},
              "assembly_manifest_sha256": sha256_path(assembly),
              "specification_ids": specification_ids, "specification_to_episode_id": mapping,
              "episode_to_specification_id": {v: k for k, v in mapping.items()},
              "specification_split": {s.task_id: s.split_assignment.label for s in members},
              "split": {mapping[s.task_id]: s.split_assignment.label for s in members},
              "split_assignment": first.split_assignment.model_dump(mode="json"),
              "required_checks": doc["pending_checks"],
              "publication": "blocked; legacy single-episode certificates cannot release this family"}
    validate_family_index(report)
    artifacts["family.json"] = report
    artifacts["overlay.json"] = overlay.to_dict()
    for member in members:
        artifacts[f"specs/{member.task_id}.json"] = member.model_dump(mode="json")
    report["artifact_hashes"] = {name: sha256_bytes(canonical_json(value))
                                 for name, value in artifacts.items() if name != "family.json"}
    # family.json is the immutable commit marker for all compiled inputs.
    write_environment_snapshot(output, artifacts)
    return report


def validate_family_index(doc: dict) -> None:
    """Reject duplicate, missing, extra or non-bijective IDs before GPU loading."""
    required = {"specification_ids","task_ids","specification_to_episode_id","episode_to_specification_id",
                "split","specification_split","split_assignment"}
    if not required <= set(doc):
        raise ValueError("COMPILED_FAMILY_ID_OR_SPLIT_MAPPING_MISSING")
    specifications, episodes = doc["specification_ids"], doc["task_ids"]
    mapping, reverse = doc["specification_to_episode_id"], doc["episode_to_specification_id"]
    if (len(specifications) != len(set(specifications)) or len(episodes) != len(set(episodes))
            or set(mapping) != set(specifications) or set(mapping.values()) != set(episodes)
            or len(set(mapping.values())) != len(mapping) or reverse != {v: k for k, v in mapping.items()}
            or set(doc["split"]) != set(episodes) or set(doc["specification_split"]) != set(specifications)
            or any(doc["split"][e] != doc["specification_split"][s] for s, e in mapping.items())
            or set(doc["split"].values()) != {doc["split_assignment"]["label"]}):
        raise ValueError("COMPILED_FAMILY_ID_OR_SPLIT_MAPPING_INVALID")
    from rummagebench.authoring.tasks.household import SplitAssignment
    SplitAssignment.model_validate(doc["split_assignment"])
