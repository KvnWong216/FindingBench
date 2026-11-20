"""Local render-only USD caches, derived from integrity-checked frozen worlds.

This never changes a dataset member or its pinned runtime. Physical validation
still belongs to the native snapshot; cached viewing is not physical execution.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

VERSION = 1


def optimize_render_stage(stage):
    """Drop invisible geometry and physics definitions from a derived stage."""
    from pxr import UsdGeom
    removed, schemas = 0, 0
    for prim in list(stage.Traverse()):
        if not prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(prim)
        if (prim.GetTypeName().startswith(('Physics', 'Physx', 'OmniGraph')) or
                imageable and imageable.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible):
            stage.RemovePrim(prim.GetPath())
            removed += 1
            continue
        for schema in prim.GetAppliedSchemas():
            if schema.startswith(('Physics', 'Physx')):
                prim.RemoveAppliedSchema(schema)
                schemas += 1
    return dict(removed_invisible_or_physics_prims=removed, removed_physics_schemas=schemas)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(task):
    from rummagebench.core.scenario import load_scenario
    from rummagebench.adapters.python_api import prepare_renderer_grounding
    from rummagebench.authoring.tasks.household_state import load_frozen_member
    scenario = Path(task['play_scenario'])
    spec = load_scenario(scenario)
    prepare_renderer_grounding(spec)
    frozen = load_frozen_member(Path(spec.scene.frozen_member_manifest), spec)
    source = dict(version=VERSION, scenario_sha256=digest(scenario),
                  frozen_manifest_sha256=frozen['manifest_sha256'],
                  runtime_root=task['runtime_root'], runtime_environment=task.get('runtime_environment', {}))
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return key, source


def lookup(directory, task):
    key, source = identity(task)
    root = Path(directory) / key
    try:
        meta = json.loads((root / 'manifest.json').read_text())
        if meta['source'] != source or meta['version'] != VERSION:
            return None
        if digest(root / 'scene.usdc') != meta['scene_sha256']:
            return None
        for path, stamp in meta.get('external_files', {}).items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != stamp:
                return None
        return root, meta
    except (OSError, ValueError, KeyError):
        return None


def export(directory, task, backend, camera, report):
    """Bake real Fabric link transforms into a separate, flattened USD layer.

    Native physics writes to Fabric, not USD. Exporting the USD stage without
    this bake silently exports the wrong object poses / articulation states.
    Every baked link's complete world matrix (including scale) is checked.
    """
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom
    from omnigibson.utils.usd_utils import (
        _get_local_pose_with_scale_from_fabric_hierarchy as local_matrix,
        _get_world_pose_with_scale_from_fabric_hierarchy as world_matrix,
    )
    started = time.monotonic()
    key, source = identity(task)
    root = Path(directory) / key
    root.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(backend._sim.stage.Flatten())
    nodes = {}
    for obj in backend._env.scene.objects:
        for item in (obj, *obj.links.values()):
            path = str(item.prim_path)
            nodes[path] = (np.asarray(local_matrix(path), dtype=float),
                           np.asarray(world_matrix(path), dtype=float))
    for path, (local, _) in nodes.items():
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        reset = xform.GetResetXformStack()
        xform.ClearXformOpOrder()
        xform.MakeMatrixXform().Set(Gf.Matrix4d(*local.ravel().tolist()))
        xform.SetResetXformStack(reset)
    xforms = UsdGeom.XformCache()
    max_error = 0.
    for path, (_, expected) in nodes.items():
        actual = np.asarray(xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        if not np.allclose(actual, expected, rtol=0, atol=1e-5):
            raise RuntimeError(f'Render-cache world transform mismatch: {path}: {error}')
    # Runtime graph/render products must be rebuilt by the viewer, not replayed.
    for path in ('/Render', '/Replicator'):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    optimization = optimize_render_stage(stage)
    external = {}
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                continue
            values = attr.Get()
            values = values if isinstance(values, (tuple, list)) or attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray else [values]
            for value in values or []:
                if value is None:
                    continue
                ref = value.resolvedPath or value.path
                if os.path.isabs(ref):
                    stat = Path(ref).stat()
                    external[ref] = [stat.st_size, stat.st_mtime_ns]
    # Atomic publication: a partial stage can never become a valid cache.
    temporary = root / f'.scene.{os.getpid()}.usdc'
    try:
        stage.GetRootLayer().Export(str(temporary))
        os.replace(temporary, root / 'scene.usdc')
    finally:
        temporary.unlink(missing_ok=True)
    meta = dict(version=VERSION, task_id=task['id'], source=source, scene_sha256=digest(root / 'scene.usdc'),
                external_files=external, home=camera.home, source_physics_tick=report['physics_tick'],
                native_physical_sha256=report['initial_physical_sha256'],
                baked_nodes=len(nodes), max_world_matrix_error=max_error,
                export_seconds=time.monotonic() - started,
                optimization=optimization,
                mode='render-only cache; no physics or task execution')
    with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as stream:
        json.dump(meta, stream, indent=2)
        tmp = Path(stream.name)
    os.replace(tmp, root / 'manifest.json')
    return root, meta
