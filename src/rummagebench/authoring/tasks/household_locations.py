"""Complete source-backed closed-storage audit for the kitchen pilot.

Factory semantics identify locations to inspect, never prove that a cavity is
empty or absent. Static counter meshes without recoverable articulation or
fillable-volume evidence remain unknown, and cannot disappear from acceptance.
"""
from __future__ import annotations

import re

from rummagebench.authoring.tasks.plan import sha256_path


def kitchen_sources(metadata: dict) -> set[str]:
    objects = metadata["objects"]
    result = set()
    for key in objects:
        seen, frontier = set(), [key]
        while frontier:
            current = frontier.pop()
            if current in seen: continue
            if current not in objects:
                raise ValueError("SEARCH_LOCATION_SOURCE_RELATION_UNRESOLVED")
            seen.add(current)
            frontier.extend(r["parent"] for r in objects[current]["relations"])
        if any("Semantics.Kitchen" in objects[k]["tags"] for k in seen):
            result.add(key)
    return result


def search_location_inventory(manifest: dict, metadata: dict, loaded: dict | None = None) -> dict:
    kitchen = kitchen_sources(metadata)
    bindings = manifest["candidate_bindings"]
    mapping = {b["source_id"]: b["location_id"] for b in bindings}
    if len(mapping)!=len(bindings) or len({b["location_id"] for b in bindings})!=len(bindings):
        raise ValueError("SEARCH_LOCATION_BINDING_NOT_BIJECTIVE")
    if not set(mapping)<=kitchen:
        raise ValueError("SEARCH_LOCATION_CANDIDATE_NOT_IN_ACTUAL_KITCHEN")
    entities = {}
    for obj in manifest["objects"]:
        entities.setdefault(obj["source_id"],[]).append(obj["name"])
    entries, failures, unknown = [], [], []
    for key in sorted(kitchen):
        obj = metadata["objects"][key]
        tags = set(obj["tags"])
        if "Semantics.Room" in tags: continue
        generator = obj.get("generator") or ""
        suspect = bool(tags & {"Semantics.Storage","Semantics.KitchenCounter"}
                       or re.search(r"(?:Cabinet|Drawer|Cupboard|Pantry|KitchenSpace).*Factory",generator))
        if key in mapping:
            suspect = True
        if not suspect and "Semantics.Furniture" not in tags:
            continue
        if key not in entities:
            failures.append(f"SEARCH_FURNITURE_SOURCE_NOT_LOADED:{key}")
            entries.append({"source_id":key,"status":"missing","entities":[]})
            continue
        if key in mapping:
            status = "required_candidate"
            reason = "complete explicit mapping; physical cavity evidence is required by the separate full probe"
            if loaded is not None and any(loaded.get(name) is None or loaded[name].get("error")
                                           or not loaded[name].get("openable_joints") for name in entities[key]):
                status,reason="unknown","REQUIRED_CONTAINER_LIVE_ARTICULATION_UNRESOLVED"
                unknown.append(key)
        elif not suspect and generator:
            status = "outside_closed_storage_scope"
            reason = "source semantics under cabinets-only instruction; not a mixed-location completeness certificate"
        else:
            status, reason = "unknown", "STATIC_OR_UNANNOTATED_GEOMETRY_CANNOT_EXCLUDE_SEARCH_CAVITIES"
            measured = [loaded.get(name) for name in entities[key]] if loaded is not None else []
            if measured and all(m is not None and not m.get("error") for m in measured):
                if any(m.get("openable_joints") or m.get("fillable_volumes") for m in measured):
                    status, reason = "additional_candidate", "LOADED_ARTICULATION_OR_FILLABLE_VOLUME_OUTSIDE_PLANNED_DISTRIBUTION"
                    failures.append(f"ADDITIONAL_SEARCH_LOCATION:{key}")
            if status=="unknown": unknown.append(key)
        entries.append({"source_id":key,"entities":sorted(entities[key]),"status":status,"reason":reason,
                        "location_id":mapping.get(key),"source_tags":sorted(tags),"source_generator":generator,
                        "loaded_evidence":{name:loaded.get(name) for name in entities[key]} if loaded is not None else None})
    return {"version":1,"scope":"direct_closed_kitchen_storage_complete_source_audit",
            "status":"failed" if failures else "pending" if unknown or loaded is None else "passed",
            "required_location_ids":sorted(mapping.values()),"entries":entries,"rejections":failures,
            "unknown_sources":unknown,"candidate_distribution_changed":False,
            "gpu_executed":loaded is not None,
            "note":"No unknown or additional source is silently excluded or reweighted."}


def inspect_search_locations(assembly):
    from pathlib import Path
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
    from rummagebench.authoring.tasks.infinigen import read_solve_state
    manifest = load_assembly_manifest(assembly)
    report = search_location_inventory(manifest,read_solve_state(Path(manifest["source"]["solve_state"])))
    report["assembly_sha256"] = sha256_path(assembly)
    report["status"] = "pending_simulation" if report["status"]=="pending" else report["status"]
    report["evaluation_ready_episodes"] = 0
    return report


def inspect_loaded_search_locations(backend, manifest):
    """GPU only: consume loaded joints/metadata, not room-level category lists."""
    from pathlib import Path
    from rummagebench.authoring.tasks.infinigen import read_solve_state
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    loaded = {}
    for item in manifest["objects"]:
        name = item["name"]
        try:
            obj = resolve_object(backend._env.scene,name)
            info = backend.articulation_info(name)
            actual_links = sorted((obj.links or {}).keys())
            volumes = [link for link in actual_links if link.startswith("meta__") and "fillable" in link]
            metadata = getattr(obj,"metadata",None) or {}
            annotation = metadata.get("openable_joint_ids") or {}
            annotated = list(annotation.values()) if isinstance(annotation,dict) else list(annotation)
            loaded[name] = {"category":obj.category,"model":getattr(obj,"model",None),
                "openable_joints":sorted(j.name for j in info.joints if j.joint_type in ("revolute","prismatic")) if info else [],
                "openable_annotation":sorted(str(v) for v in annotated),"fillable_volumes":volumes,
                "world_aabb":backend.entity_aabb(name),"error":None}
        except Exception as exc:
            loaded[name] = {"error":f"{type(exc).__name__}: {exc}"}
    report = search_location_inventory(manifest,read_solve_state(Path(manifest["source"]["solve_state"])),loaded)
    report["loaded_entities"] = loaded
    return report
