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
    if (doc.get("version") != 1 or doc.get("status") != "frozen_member"
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
    return doc


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
    if meta != {"assembly_manifest_sha256": doc["assembly_manifest_sha256"],
                "scenario_sha256": doc["scenario_sha256"]}:
        raise ValueError("FROZEN_MEMBER_STATE_METADATA_MISMATCH")
    backend.restore_observe_state({"sim": state, "commanded_pose": None})
    check_background(backend, doc, root)
    return doc
