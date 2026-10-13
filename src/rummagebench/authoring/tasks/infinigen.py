"""Pinned Infinigen-Indoors boundary: commands, CPU runner and real JSON/USD inspection.

No Blender, USD or simulator is imported by importing this module. Upstream
code stays outside FindingBench. Whole-scene export is deferred because the
pinned exporter unconditionally selects Cycles GPU for texture baking.
"""
from __future__ import annotations

import gzip
import fcntl
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.household import HouseholdTaskSpec, UPSTREAM_COMMIT
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path

CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void", "HIP_VISIBLE_DEVICES": "",
           "ROCR_VISIBLE_DEVICES": "", "DISPLAY": "", "LIBGL_ALWAYS_SOFTWARE": "1",
           "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/findingbench_mpl"}


def cpu_sandbox_argv(argv: list[str], writable: Path) -> list[str]:
    """Kernel device isolation: private /dev contains no NVIDIA or DRM nodes.

    The entire filesystem is read-only except this run directory and private
    /tmp; no network. --unshare-pid also ensures a timeout tears down children.
    Refuse to execute if this mechanism is unavailable, rather than falling
    back to environment flags alone.
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("CPU_DEVICE_SANDBOX_UNAVAILABLE")
    return [bwrap, "--ro-bind", "/", "/", "--bind", str(writable), str(writable),
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--unshare-pid",
            "--unshare-net", "--unshare-user", "--die-with-parent", "--", *argv]


def source_audit(root: Path) -> dict:
    root = Path(root).resolve()
    files = ["infinigen_examples/generate_indoors.py", "infinigen/core/init.py",
             "infinigen/core/constraints/example_solver/state_def.py", "infinigen/tools/export.py",
             "infinigen_examples/constraints/home.py", "infinigen_examples/constraints/semantics.py",
             "infinigen_examples/configs_indoor/singleroom.gin"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                           capture_output=True, text=True, check=True).stdout.strip()
    missing = [p for p in files if not (root / p).is_file()]
    if commit != UPSTREAM_COMMIT or dirty or missing:
        raise ValueError(f"UPSTREAM_VERSION_OR_SOURCE_MISMATCH: commit={commit}, dirty={bool(dirty)}, missing={missing}")
    init = (root / "infinigen/core/init.py").read_text()
    # Version is pinned and source pristine. Confirm the branch which returns
    # before GPU device enumeration exists; the mandatory Gin binding below
    # selects that branch, not merely headless mode or an empty CUDA list.
    if 'def configure_cycles_devices(use_gpu=True)' not in init or 'if use_gpu is False:' not in init:
        raise ValueError("CPU_DEVICE_GUARD_UNAVAILABLE")
    return {"commit": commit, "root": str(root), "files": {p: sha256_path(root / p) for p in files},
            "cpu_generation_supported": True, "whole_scene_articulation_preserved": False,
            "whole_scene_export_cpu_supported": False, "license": "BSD-3-Clause"}



def runtime_audit(python: str) -> dict:
    """Stdlib metadata only; never import bpy, CUDA or simulator packages."""
    script = (
        "import sys,json,importlib.metadata as m; "
        "names=['bpy','gin-config','numpy','shapely','python-fcl','rtree','OpenEXR','usd-core']; "
        "versions={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
        "print(json.dumps({'python':sys.version,'packages':{n:versions.get(n.lower()) for n in names},'installed_packages':versions}))"
    )
    result = subprocess.run([python, "-I", "-c", script], capture_output=True, text=True,
                            timeout=15., check=True, env=dict(os.environ, **CPU_ENV))
    binary = Path(shutil.which(python) or python).resolve()
    return {**json.loads(result.stdout), "executable": str(binary), "executable_sha256": sha256_path(binary)}

def generation_request(spec: HouseholdTaskSpec) -> dict:
    """Only environment information enters upstream generation, never the hidden target."""
    return {"schema_version": 1, "upstream_commit": spec.upstream_commit, "scene_seed": spec.scene_seed,
            "base_layout_id": spec.base_layout_id, "room_type": spec.room_type,
            "background_policy": spec.background_policy,
            "postchecks": ["kitchen_present", "all_required_candidates_mapped", "source_geometry_complete",
                           "background_support_relations", "real_articulation_for_open", "target_fit",
                           "initial_visibility", "robot_collision_and_reach", "reset_consistency"],
            "target_conditioned_generation": False}


def generation_command(spec: HouseholdTaskSpec, python: str, root: Path, output: Path) -> dict:
    if spec.family != "kitchen_cabinet_search":
        raise ValueError("UNSUPPORTED_TASK_FAMILY")
    overrides = ["configure_cycles_devices.use_gpu=False", "compose_indoors.terrain_enabled=False",
                 "compose_indoors.animate_cameras_enabled=False", "compose_indoors.floating_objs_enabled=False",
                 "compose_indoors.invisible_room_ceilings_enabled=False", "compose_indoors.hide_other_rooms_enabled=False",
                 "compose_indoors.lights_off_enabled=False", "restrict_solving.restrict_parent_rooms=['Kitchen']",
                 "restrict_solving.solve_max_rooms=1", "compose_indoors.abort_unsatisfied_large=True",
                 "compose_indoors.abort_unsatisfied_medium=True", "compose_indoors.abort_unsatisfied_small=True"]
    return {"argv": [python, "-m", "infinigen_examples.generate_indoors", "--seed", str(spec.scene_seed),
                     "--task", "coarse", "--output_folder", str(Path(output).resolve()),
                     "-g", "singleroom.gin", "-p", *overrides],
            "cwd": str(Path(root).resolve()), "environment": dict(CPU_ENV),
            "request": generation_request(spec), "stage": "cpu_geometry"}


def export_command(python: str, root: Path, input_folder: Path, output: Path, resolution: int = 512) -> dict:
    if not 64 <= resolution <= 4096:
        raise ValueError("invalid texture resolution")
    return {"argv": [python, "-m", "infinigen.tools.export", "--input_folder", str(input_folder.resolve()),
                     "--output_folder", str(output.resolve()), "-f", "usdc", "-r", str(resolution), "--omniverse"],
            "cwd": str(root.resolve()), "execution_status": "deferred",
            "reason": "PINNED_EXPORTER_FORCES_CYCLES_GPU", "whole_scene_articulation_preserved": False}


def read_solve_state(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    objs = raw.get("objs")
    if not isinstance(objs, dict) or not objs:
        raise ValueError("INVALID_SOLVE_STATE")
    normalized, names = {}, {}
    for key, obj in sorted(objs.items()):
        if not isinstance(obj, dict) or not isinstance(obj.get("tags", []), list):
            raise ValueError(f"INVALID_OBJECT_METADATA:{key}")
        name = obj.get("obj")
        if name is not None and not isinstance(name, str):
            raise ValueError("INVALID_SOURCE_OBJECT_NAME")
        usd_name = None if name is None else name.replace("(", "_").replace(")", "_").replace(".", "_")
        if usd_name in names and usd_name is not None:
            raise ValueError(f"AMBIGUOUS_EXPORTED_OBJECT_NAME:{usd_name}")
        names[usd_name] = key
        relations = []
        for rel in obj.get("relations", []):
            if not isinstance(rel, dict) or rel.get("target_name") not in objs:
                raise ValueError(f"UNRESOLVED_RELATION_TARGET:{key}")
            relation = rel.get("relation")
            if not isinstance(relation, dict) or not isinstance(relation.get("relation_type"), str):
                raise ValueError(f"INVALID_RELATION:{key}")
            relations.append({"parent": rel["target_name"], "type": relation["relation_type"],
                              "parent_tags": relation.get("parent_tags", []),
                              "child_tags": relation.get("child_tags", []),
                              "source": "infinigen_solver_relation_unverified_in_sim"})
        normalized[key] = {"source_id": key, "blender_name": name, "usd_name": usd_name,
                           "tags": sorted(obj.get("tags", [])), "generator": obj.get("generator"),
                           "active": obj.get("active", True), "relations": relations,
                           "articulation": "not_exported_by_whole_scene_exporter"}
    return {"source_sha256": sha256_path(path), "objects": normalized,
            "kitchens": [k for k, o in normalized.items() if "Semantics.Kitchen" in o["tags"]],
            "physics_verified": False}


def _blend_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as f:
        prefix = f.read(7)
    if prefix[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            prefix = f.read(7)
    return prefix == b"BLENDER"


def run_cpu_generation(spec: HouseholdTaskSpec, root: Path, python: str, output: Path,
                       *, timeout_s: float = 300., retries: int = 0) -> dict:
    if timeout_s <= 0 or not 0 <= retries <= 2:
        raise ValueError("invalid timeout/retry bound")
    audit = source_audit(root)
    output = Path(output).resolve()
    command = generation_command(spec, python, root, output / "geometry")
    inputs = {"command": command, "source": audit, "runtime": runtime_audit(python)}
    identity = sha256_bytes(canonical_json(inputs))
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    # Advisory locks are released by the kernel on crashes, unlike a stale
    # O_EXCL lock file. Keep the inode so concurrent resumptions cannot race.
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("RUN_ALREADY_ACTIVE") from exc
        input_path = output / "input.json"
        if input_path.exists():
            if json.loads(input_path.read_text())["input_sha256"] != identity:
                raise ValueError("RESUME_INPUT_CONFLICT")
        else:
            if list(output.glob("attempt_*")):
                raise ValueError("RESUME_INPUT_RECORD_MISSING")
            atomic_write_json(input_path, {"input_sha256": identity, "inputs": inputs})
        receipt_path = output / "generation.json"
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            if old["input_sha256"] != identity:
                raise ValueError("RESUME_INPUT_CONFLICT")
            for rel, digest in old["artifacts"].items():
                if sha256_path(output / rel) != digest:
                    raise ValueError("RESUME_ARTIFACT_CONFLICT")
            return old
        attempts, artifacts = [], {}
        for retry in range(retries + 1):
            # SAME seed/config. Incomplete attempts are recorded as interrupted
            # and consume the finite retry budget, never counted as successful.
            attempt_dir = output / f"attempt_{retry}"
            checkpoint = attempt_dir / "attempt.json"
            if checkpoint.exists():
                attempt = json.loads(checkpoint.read_text())
                if attempt.get("input_sha256") != identity:
                    raise ValueError("RESUME_ATTEMPT_CONFLICT")
                attempts.append(attempt)
                if attempt["reason"] is None:
                    artifacts = attempt["artifacts"]
                    for rel, digest in artifacts.items():
                        if sha256_path(output / rel) != digest:
                            raise ValueError("RESUME_ARTIFACT_CONFLICT")
                    break
                continue
            if attempt_dir.exists():
                attempt = {"retry": retry, "input_sha256": identity, "exit_code": None,
                           "reason": "GENERATION_INTERRUPTED", "artifacts": {}}
                atomic_write_json(checkpoint, attempt)
                attempts.append(attempt)
                continue
            attempt_dir.mkdir()
            geometry = attempt_dir / "geometry"; geometry.mkdir()
            cmd = generation_command(spec, python, root, geometry)
            env = dict(os.environ, **CPU_ENV, PYTHONPATH=str(root.resolve()))
            started = time.monotonic()
            code, reason = None, None
            with (attempt_dir / "stdout.log").open("w") as stdout, (attempt_dir / "stderr.log").open("w") as stderr:
                try:
                    cmd["sandbox_argv"] = cpu_sandbox_argv(cmd["argv"], attempt_dir)
                    proc = subprocess.run(cmd["sandbox_argv"], cwd=cmd["cwd"], env=env, stdout=stdout,
                                          stderr=stderr, timeout=timeout_s, check=False, shell=False)
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    reason = "GENERATION_TIMEOUT"
                except (OSError, RuntimeError) as exc:
                    reason = "CPU_EXECUTION_UNAVAILABLE"
                    stderr.write(str(exc))
            error_text = (attempt_dir / "stdout.log").read_text() + (attempt_dir / "stderr.log").read_text()
            if reason is None and code != 0:
                if "ModuleNotFoundError" in error_text or "ImportError" in error_text:
                    reason = "UPSTREAM_DEPENDENCY_MISSING"
                elif re.search(r"No configurable matching|Unknown configurable|gin.*binding", error_text, re.I):
                    reason = "CONFIGURATION_INVALID"
                elif re.search(r"unsatisf|infeasible constraint", error_text, re.I):
                    reason = "CONSTRAINT_UNSATISFIABLE"
                else:
                    reason = "GENERATION_FAILED"
            state_path, blend = geometry / "solve_state.json", geometry / "scene.blend"
            if reason is None:
                try:
                    metadata = read_solve_state(state_path)
                    if not metadata["kitchens"] or not _blend_valid(blend):
                        raise ValueError("incomplete generated kitchen/Blender artifact")
                except (OSError, ValueError) as exc:
                    reason = "GENERATION_OUTPUT_INVALID"
                    error_text += f"\n{type(exc).__name__}: {exc}"
            if reason is None:
                artifacts = {str(p.relative_to(output)): sha256_path(p) for p in (state_path, blend)}
            attempt = {"retry": retry, "input_sha256": identity, "command": cmd, "exit_code": code,
                       "reason": reason, "elapsed_s": round(time.monotonic() - started, 3),
                       "error_excerpt": error_text[-3000:], "artifacts": artifacts}
            atomic_write_json(checkpoint, attempt)
            attempts.append(attempt)
            if reason is None:
                break
        success = attempts[-1]["reason"] is None
        receipt = {"version": 1, "input_sha256": identity, "source": audit, "attempts": attempts,
                   "status": "geometry_generated" if success else "generation_failed",
                   "generated_geometry_scenes": int(success), "exported_scene_files": 0,
                   "evaluation_ready_episodes": 0, "artifacts": artifacts,
                   "gpu_started": False, "gpu_device_isolation": "bwrap_private_dev_no_nvidia_or_drm",
                   "simulation_acceptance": "pending", "family": spec.family,
                   "requested_difficulty": spec.requested_difficulty,
                   "failure_reasons": dict(Counter(a["reason"] for a in attempts if a["reason"]))}
        atomic_write_json(receipt_path, receipt)
        return receipt


def inspect_export(scene_path: Path, solve_path: Path) -> dict:
    """CPU-only OpenUSD parsing; opening a stage does not initialize Isaac/Kit."""
    if not scene_path.is_file() or not solve_path.is_file():
        raise ValueError("EXPORT_FILES_MISSING")
    metadata = read_solve_state(solve_path)
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError("CPU_USD_DEPENDENCY_MISSING: install the standalone usd-core wheel") from exc
    stage = Usd.Stage.Open(str(scene_path.resolve()))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError("USD_STAGE_OR_DEFAULT_PRIM_INVALID")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-8:
        raise ValueError("USD_AXIS_OR_UNITS_UNSUPPORTED")
    names = {}
    meshes, joints = [], []
    for prim in stage.Traverse():
        if prim.IsInstance():
            raise ValueError(f"USD_INSTANCED_GEOMETRY_REQUIRES_EXPLICIT_EXPANSION:{prim.GetPath()}")
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(path)
        if prim.IsA(UsdGeom.Xformable):
            names.setdefault(prim.GetName(), []).append(path)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(str(prim.GetPath()))
    mapping, missing, ambiguous = {}, [], []
    for key, obj in metadata["objects"].items():
        if obj["usd_name"] is None:
            continue
        found = names.get(obj["usd_name"], [])
        if len(found) == 1:
            mapping[key] = found[0]
        elif len(found) > 1:
            ambiguous.append(key)
        else:
            missing.append(key)
    if not meshes:
        raise ValueError("USD_MESH_GEOMETRY_MISSING")
    from pxr import Sdf
    dependencies = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            values = [val] if isinstance(val, Sdf.AssetPath) else (list(val) if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray and val is not None else [])
            for asset in values:
                if asset.path and not asset.resolvedPath:
                    dependencies.append({"attribute": str(attr.GetPath()), "asset": asset.path})
    if dependencies:
        raise ValueError(f"USD_TEXTURE_OR_ASSET_DEPENDENCY_MISSING:{dependencies}")
    return {"status": "export_inspected", "scene_sha256": sha256_path(scene_path),
            "solve_state_sha256": metadata["source_sha256"], "mesh_count": len(meshes),
            "joint_paths": joints, "source_to_usd": mapping, "unmapped_source_objects": missing,
            "ambiguous_source_objects": ambiguous, "all_mesh_paths": meshes,
            "evaluation_ready_episodes": 0, "physical_interactivity_validated": False}
