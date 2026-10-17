"""Integrity and restore contract for LOCAL household simulator snapshots.

No simulator, torch or pickle imports on CPU. Loading a tensor snapshot is
restricted to the explicit runtime restore, after checking its bundle hashes.
"""
from __future__ import annotations

import json
from pathlib import Path

from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path


def scenario_state_identity(scenario) -> str:
    doc = scenario.model_dump(mode="json")
    doc["scene"].pop("frozen_member_manifest", None)
    return sha256_bytes(canonical_json(doc))


def runtime_topology(backend) -> list[dict]:
    # Serialized OG tensors depend on object order, not just the name set.
    return [{"name": o.name, "links": list((o.links or {}).keys())}
            for o in backend._env.scene.objects]


def bundle_file(root: Path, ref: str, digest: str) -> Path:
    path = (root / ref).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file()
            or not digest or sha256_path(path) != digest):
        raise ValueError(f"FROZEN_BUNDLE_FILE_MISSING_OR_CHANGED:{ref}")
    return path


def load_frozen_member(path: Path, scenario) -> dict:
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    payload = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    if (doc.get("version") not in (1, 2) or doc.get("status") != "frozen_member"
            or doc.get("manifest_sha256") != sha256_bytes(canonical_json(payload))
            or doc.get("scenario_id") != scenario.id
            or doc.get("scenario_sha256") != scenario_state_identity(scenario)):
        raise ValueError("FROZEN_MEMBER_SCENARIO_OR_HASH_MISMATCH")
    if sha256_path(scenario.scene.assembly_manifest) != doc["assembly_manifest_sha256"]:
        raise ValueError("FROZEN_MEMBER_ASSEMBLY_CHANGED")
    root = path.parent.parent
    for ref, digest in doc["artifacts"].items():
        bundle_file(root, ref, digest)
    if not {doc["state"], doc["background_records"]} <= set(doc["artifacts"]):
        raise ValueError("FROZEN_MEMBER_ARTIFACT_LEDGER_INCOMPLETE")
    if doc['version']==2:
        recipe=doc.get('restore_recipe',{})
        if (set(recipe)!={'contract','checkpoint_input_state_sha256','observed_world_state_sha256'}
                or recipe['contract']!='same-checkpoint-exact-world-v1'
                or any(len(recipe[k])!=64 or any(c not in '0123456789abcdef' for c in recipe[k])
                       for k in ('checkpoint_input_state_sha256','observed_world_state_sha256'))):
            raise ValueError('FROZEN_MEMBER_RECONSTRUCTION_RECIPE_REQUIRED')
    return doc


def reconstruct_checkpoint(backend, state, *, expected_world_hash=None):
    """Load the same immutable input, refresh spatial queries without stepping."""
    from copy import deepcopy
    from rummagebench.authoring.tasks.native_occupancy import state_hash
    digest=state_hash(state)
    backend.restore_observe_state({'sim':deepcopy(state),'commanded_pose':None})
    backend.validate_physics_state()
    backend._validate_spawn_clearance()
    observed=state_hash(backend.dump_state())
    if state_hash(state)!=digest:
        raise ValueError('FROZEN_CHECKPOINT_INPUT_MUTATED')
    if expected_world_hash is not None and observed!=expected_world_hash:
        raise ValueError('FROZEN_MEMBER_EXACT_WORLD_RECONSTRUCTION_MISMATCH')
    return {'contract':'same-checkpoint-exact-world-v1','checkpoint_input_state_sha256':digest,
            'observed_world_state_sha256':observed}


def merge_checkpoint_objects(native, complete, target_entity):
    """SDK structured state composition, before native sim.serialize, runtime only.

    Preserve the original background INPUT states rather than reserializing
    the reconstructed background as a new input. The sole added entity is
    the parked/placed target; unexpected topology and frame changes reject.
    No tensor offsets, rounding, simulator objects or room-level inference.
    """
    from copy import deepcopy
    from rummagebench.authoring.tasks.native_occupancy import state_hash
    if set(native)!={0} or set(complete)!={0}:
        raise ValueError('FROZEN_CHECKPOINT_SINGLE_SCENE_REQUIRED')
    source,current=native[0],complete[0]
    if (set(source)!=set(current) or set(source)!={'pos','ori','registry'}
            or set(source['registry'])!=set(current['registry'])):
        raise ValueError('FROZEN_CHECKPOINT_SCENE_SCHEMA_CHANGED')
    if any(state_hash(source[k])!=state_hash(current[k]) for k in ('pos','ori')):
        raise ValueError('FROZEN_CHECKPOINT_SCENE_FRAME_CHANGED')
    original=source['registry']['object_registry']
    actual=current['registry']['object_registry']
    if target_entity in original or set(actual)!=set(original)|{target_entity}:
        raise ValueError('FROZEN_CHECKPOINT_NATIVE_OR_TARGET_TOPOLOGY_CHANGED')
    result=deepcopy(native)
    # Retain current registry order: the SDK serializer and runtime topology
    # must agree. Every original object's INPUT is preserved byte for byte.
    result[0]['registry']['object_registry']={name:deepcopy(actual[name] if name==target_entity else original[name])
                                              for name in actual}
    return result


def check_background(backend, doc: dict, root: Path) -> None:
    from rummagebench.sim.omnigibson.infinigen_adapter import background_records
    from rummagebench.authoring.tasks.native_occupancy import background_mismatches
    records = json.loads((root / doc["background_records"]).read_text())
    mismatch = background_mismatches(records, background_records(backend, list(records)))
    if mismatch:
        raise ValueError(f"FROZEN_BACKGROUND_RUNTIME_MISMATCH:{mismatch}")


def restore_frozen_member(backend, path: Path, scenario) -> dict:
    """Runtime only: restore a trusted locally generated state without resampling."""
    from rummagebench.sim.omnigibson.state_io import load_snapshot
    doc = load_frozen_member(path, scenario)
    if runtime_topology(backend) != doc["runtime_topology"]:
        raise ValueError("FROZEN_MEMBER_SERIALIZED_TOPOLOGY_OR_ORDER_MISMATCH")
    root = Path(path).resolve().parent.parent
    state, meta = load_snapshot(root / doc["state"])
    expected_meta={"assembly_manifest_sha256": doc["assembly_manifest_sha256"],
                   "scenario_sha256": doc["scenario_sha256"]}
    if doc['version']==2:expected_meta['restore_recipe']=doc['restore_recipe']
    if meta != expected_meta:
        raise ValueError("FROZEN_MEMBER_STATE_METADATA_MISMATCH")
    if doc['version']==2:
        from rummagebench.authoring.tasks.native_occupancy import state_hash
        if state_hash(state)!=doc['restore_recipe']['checkpoint_input_state_sha256']:
            raise ValueError('FROZEN_MEMBER_CHECKPOINT_INPUT_HASH_MISMATCH')
        reconstruct_checkpoint(backend,state,expected_world_hash=doc['restore_recipe']['observed_world_state_sha256'])
    else:
        backend.restore_observe_state({"sim": state, "commanded_pose": None})
    check_background(backend, doc, root)
    return doc
