"""Evaluator-side, hash-checked scene variants with unchanged official task bindings."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_scene_derivation(manifest_path, scene_model, activity_name):
    manifest_path=Path(manifest_path).resolve()
    d=json.loads(manifest_path.read_text())
    if d['scene_model']!=scene_model or d['activity_name']!=activity_name:
        raise ValueError('Derived scene/task binding mismatch')
    source=Path(d['source_template']);derived=Path(d['derived_template'])
    if digest(source)!=d['source_sha256'] or digest(derived)!=d['derived_sha256']:
        raise ValueError('Derived scene provenance hash mismatch')
    original=json.loads(source.read_text());edited=json.loads(derived.read_text())
    removed=set(d['removed_objects'])
    bindings=set(original['metadata']['task']['inst_to_name'].values())
    if removed & bindings:raise ValueError('Cannot remove task-bound objects')
    expected=json.loads(source.read_text())
    for name in removed:
        del expected['objects_info']['init_info'][name]
        del expected['state']['registry']['object_registry'][name]
    if edited!=expected:raise ValueError('Derived scene differs beyond declared object removals')
    return str(derived),d
