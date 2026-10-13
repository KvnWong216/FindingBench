"""Evaluator-only native occupancy from the loaded builder scene.

AABB envelopes are conservative exclusion tests, never free-space, packing,
collision or stability certificates. Per-link membership that the installed
Inside state cannot resolve remains unknown. No room/category inference.
This module imports neither OmniGibson nor torch at import time.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.authoring.tasks.slots import SceneSlots


def plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def state_hash(state: Any) -> str:
    """Hash serialized simulator tensor values, shape and dtype, not a .pt wrapper."""
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    a = np.asarray(state)
    if a.dtype.kind not in "biuf" or not np.isfinite(a).all():
        raise ValueError("invalid serialized simulator state")
    return sha256_bytes(canonical_json({"dtype": a.dtype.str, "shape": list(a.shape)})
                        + np.ascontiguousarray(a).tobytes())


def checked_box(box):
    a = np.asarray(plain(box), dtype=float)
    if a.shape != (2, 3) or not np.isfinite(a).all() or np.any(a[1] < a[0]):
        raise ValueError("missing or invalid world AABB")
    return a.tolist()


def overlaps(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    # Contact and a 1 cm uncertainty band both count as possible occupancy.
    return bool(np.all(a[1] >= b[0] - 0.01) and np.all(b[1] >= a[0] - 0.01))


def summarize_slot(slot, objects: dict, relations: list, envelope,
                   global_unknown: list[str]) -> dict:
    """Three-way classification; retain uncertainty even when occupancy is proven."""
    reasons = list(global_unknown)
    definite, possible = set(), set()
    for r in relations:
        if r["parent"] == slot.parent_entity and r["relation"] == slot.relation:
            # OG Inside is object-relative; it does not identify a drawer link.
            (possible if slot.link is not None else definite).add(r["entity"])
    if envelope is None:
        reasons.append("SLOT_ENVELOPE_UNAVAILABLE")
    for name, record in objects.items():
        if name == slot.parent_entity:
            continue
        if record.get("aabb") is None:
            reasons.append(f"OBJECT_GEOMETRY_UNAVAILABLE:{name}")
        elif envelope is not None and overlaps(record["aabb"], envelope):
            possible.add(name)
    # Descendants of a native bowl/tray/container stay in the same occupancy
    # record as their parent. IDs appear once; their volumes are never summed.
    known = definite | possible
    while True:
        children = {r["entity"] for r in relations if r["parent"] in known}
        if children <= known:
            break
        known |= children
    possible = known - definite
    if possible:
        reasons.append("GEOMETRY_OR_LINK_MEMBERSHIP_UNRESOLVED")
    status = ("confirmed_occupied" if definite else
              "unknown" if reasons else "confirmed_empty")
    return {"slot_id": slot.slot_id, "status": status,
            "occupants": sorted(definite), "possible_occupants": sorted(possible),
            "world_envelope": envelope, "envelope_source": "loaded_world_aabb",
            "reasons": sorted(set(reasons)), "geometry_complete": not reasons}


def capture_native_occupancy(backend, slots: SceneSlots, *, source_scene_path: Path) -> tuple[dict, Any]:
    """Called before task import on the builder backend; returns manifest + state.

    Query errors become unknown, never empty. A changed source, filtered scene,
    unexpected task entity or state mutation aborts instead of producing an
    inventory for a different scene. Relations cover all rooms and nested
    containers. Particle systems block claims of complete occupancy.
    """
    import json
    from omnigibson.object_states import Inside, OnTop

    source_scene_path = Path(source_scene_path).resolve()
    source_hash = sha256_path(source_scene_path)
    if source_hash != slots.source.get("scene_json_sha256"):
        raise ValueError("scene source differs from SceneSlots")
    if backend._scenario.scene.model != slots.scene:
        raise ValueError("loaded scene differs from SceneSlots")
    from rummagebench.sim.omnigibson.env_factory import build_env_config

    config = build_env_config(backend._scenario)
    cfg = config["scene"]
    for key in ("load_object_categories", "not_load_object_categories",
                "load_room_types", "load_room_instances", "load_task_relevant_only"):
        if cfg.get(key):
            raise ValueError(f"native occupancy requires a complete scene: {key}")
    loaded_path = Path(backend._env.scene.scene_file).resolve()
    if loaded_path != source_scene_path:
        raise ValueError("actual loaded scene file differs from supplied source")
    scene = backend._env.scene
    native = {o.name: o for o in scene.objects if o not in scene.robots}
    source = json.loads(source_scene_path.read_text())
    expected = {name for name, info in source["objects_info"]["init_info"].items()
                if info.get("class_name") != "Robot"}
    if set(native) != expected:
        raise ValueError(f"native entity inventory mismatch: missing={sorted(expected-set(native))}, "
                         f"extra={sorted(set(native)-expected)}")
    before = backend.dump_state()
    digest = state_hash(before)
    objects, errors, relations, asset_sources = {}, [], [], {}
    for name, obj in sorted(native.items()):
        rec = {"entity": name, "category": getattr(obj, "category", None),
               "model": getattr(obj, "model", None), "origin": "native",
               "rooms": sorted(getattr(obj, "in_rooms", []) or [])}
        asset = getattr(obj, "_usd_path", None)
        if asset is not None:
            path = Path(asset)
            if getattr(obj, "_encrypted", False):
                path = Path(str(path).replace(".usd", ".encrypted.usd"))
            if path.is_file():
                key = str(path.resolve())
                if key not in asset_sources:
                    asset_sources[key] = sha256_path(path)
                rec["asset_source"] = key
            else:
                errors.append(f"ASSET_SOURCE_UNAVAILABLE:{name}")
        else:
            errors.append(f"ASSET_SOURCE_UNAVAILABLE:{name}")
        try:
            rec["aabb"] = checked_box(backend.entity_aabb(name))
        except Exception as exc:
            rec["aabb"] = None
            rec["geometry_error"] = f"{type(exc).__name__}: {exc}"
        try:
            rec["state"] = plain(obj.dump_state(serialized=False))
            rec["state_sha256"] = sha256_bytes(canonical_json(rec["state"]))
        except Exception as exc:
            errors.append(f"OBJECT_STATE_UNAVAILABLE:{name}:{type(exc).__name__}")
        objects[name] = rec
    for name, obj in sorted(native.items()):
        for parent, rec in sorted(native.items()):
            if name == parent:
                continue
            a, b = objects[name]["aabb"], objects[parent]["aabb"]
            if a is not None and b is not None and not overlaps(a, b):
                continue
            for relation, state_type in (("inside", Inside), ("on_top", OnTop)):
                state = obj.states.get(state_type)
                # Unsupported states do not imply geometric emptiness; the
                # conservative envelope test below still retains the object.
                if state is None:
                    continue
                try:
                    if bool(state.get_value(rec)):
                        relations.append({"entity": name, "parent": parent,
                                          "relation": relation, "source": "og_relative_state"})
                except Exception as exc:
                    errors.append(f"RELATION_QUERY_FAILED:{name}:{parent}:{relation}:{type(exc).__name__}")
    if getattr(scene, "systems", {}):
        errors.append("PARTICLE_SYSTEM_OCCUPANCY_UNRESOLVED")
    results = []
    for slot in sorted(slots.slots, key=lambda s: s.slot_id):
        envelope = objects.get(slot.parent_entity, {}).get("aabb")
        if envelope is not None and slot.relation == "on_top":
            # Query the same live support interface the skill uses. Its top
            # is an exclusion envelope, not a verified usable surface.
            try:
                region = backend.receptacle_region(slot.parent_entity)
                box = checked_box([region.lo, region.hi])
                envelope = [box[0][:2] + [box[1][2] - 0.01],
                            box[1][:2] + [box[1][2] + 0.02]]
            except Exception:
                envelope = None
        results.append(summarize_slot(slot, objects, relations, envelope, errors))
    if state_hash(backend.dump_state()) != digest:
        raise ValueError("native occupancy query mutated source state")
    if sha256_path(source_scene_path) != source_hash:
        raise ValueError("scene source changed during capture")
    if any(sha256_path(p) != digest for p, digest in asset_sources.items()):
        raise ValueError("native asset source changed during capture")
    doc = {"version": 1, "scene": slots.scene, "status": "measured_native_occupancy",
           "source_scene_sha256": source_hash, "source_state_sha256": digest,
           "scene_slots_sha256": sha256_bytes(canonical_json(slots.to_dict())),
           "builder_config_sha256": sha256_bytes(canonical_json(config)),
           "objects": objects, "asset_sources": asset_sources, "relations": relations, "slots": results,
           "counts": dict(Counter(s["status"] for s in results)),
           "uncertainties": sorted(set(errors)), "physical_acceptance": "pending"}
    doc["inventory_sha256"] = sha256_bytes(canonical_json(doc))
    return doc, before


def verify_source(doc: dict, backend, slots: SceneSlots) -> None:
    """A reused inventory must refer to this restored state and these slots."""
    payload = {k: v for k, v in doc.items() if k != "inventory_sha256"}
    if sha256_bytes(canonical_json(payload)) != doc.get("inventory_sha256"):
        raise ValueError("native inventory hash mismatch")
    if state_hash(backend.dump_state()) != doc["source_state_sha256"]:
        raise ValueError("native source state mismatch")
    if sha256_bytes(canonical_json(slots.to_dict())) != doc["scene_slots_sha256"]:
        raise ValueError("native SceneSlots mismatch")
    if sha256_path(backend._env.scene.scene_file) != doc["source_scene_sha256"]:
        raise ValueError("native scene source mismatch")
    from rummagebench.sim.omnigibson.env_factory import build_env_config

    if sha256_bytes(canonical_json(build_env_config(backend._scenario))) != doc["builder_config_sha256"]:
        raise ValueError("native builder configuration mismatch")
    if any(sha256_path(p) != digest for p, digest in doc["asset_sources"].items()):
        raise ValueError("native asset source mismatch")
    scene = backend._env.scene
    native = {o.name: o for o in scene.objects if o not in scene.robots}
    if set(native) != set(doc["objects"]):
        raise ValueError("native restored entity inventory mismatch")
    for name, record in doc["objects"].items():
        obj = native[name]
        if record.get("aabb") is not None and checked_box(backend.entity_aabb(name)) != record["aabb"]:
            raise ValueError(f"native restored geometry mismatch: {name}")
        if record.get("state_sha256") and sha256_bytes(canonical_json(plain(obj.dump_state(serialized=False)))) != record["state_sha256"]:
            raise ValueError(f"native restored object state mismatch: {name}")


def require_known_slots(doc: dict, slot_ids: list[str]) -> None:
    """Reject the whole requested candidate set; never prune unknown locations."""
    entries = {s["slot_id"]: s for s in doc["slots"]}
    unknown = [sid for sid in slot_ids if sid not in entries or entries[sid]["status"] == "unknown"
               or not entries[sid]["geometry_complete"]]
    if unknown:
        raise ValueError(f"native occupancy unresolved for candidate slots: {sorted(unknown)}")


def background_mismatches(frozen: dict, current: dict) -> list[str]:
    """Compare an explicitly captured background, excluding no IDs implicitly.

    Target insertion is allowed only outside these background records; native
    and added objects, poses, velocities and states must all match exactly.
    Callers may rerun physical placement on mismatch, never silently pass it.
    """
    out = []
    for name in sorted(set(frozen) | set(current)):
        if name not in frozen or name not in current or canonical_json(frozen[name]) != canonical_json(current[name]):
            out.append(name)
    return out
