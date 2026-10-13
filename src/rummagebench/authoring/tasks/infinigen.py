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
           "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/findingbench_mpl",
           "BLENDER_USER_CONFIG": "/tmp/findingbench_blender_config", "PYTHONHASHSEED": "0"}


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
             "infinigen_examples/configs_indoor/singleroom.gin", "infinigen_examples/configs_indoor/fast_solve.gin"]
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
        "print(json.dumps({'python':sys.version,'prefix':sys.prefix,'packages':{n:versions.get(n.lower()) for n in names},'installed_packages':versions}))"
    )
    result = subprocess.run([python, "-I", "-c", script], capture_output=True, text=True,
                            timeout=15., check=True, env=dict(os.environ, **CPU_ENV))
    binary = Path(shutil.which(python) or python).resolve()
    runtime = json.loads(result.stdout)
    extensions = Path(runtime['prefix']) / 'blender_extensions/blender_org'
    runtime['blender_extension_repository'] = str(extensions) if extensions.is_dir() else None
    runtime['blender_extension_files'] = {str(p.resolve()):sha256_path(p) for p in sorted(extensions.rglob('*'))
                                        if p.is_file() and '__pycache__' not in p.parts}
    return {**runtime, "executable": str(binary), "executable_sha256": sha256_path(binary)}

def generation_request(spec: HouseholdTaskSpec) -> dict:
    """Only environment information enters upstream generation, never the hidden target."""
    return {"schema_version": 1, "upstream_commit": spec.upstream_commit, "scene_seed": spec.scene_seed,
            "base_layout_id": spec.base_layout_id, "room_type": spec.room_type,
            "background_policy": spec.background_policy, "generation_constraints": spec.generation_constraints,
            "postchecks": ["kitchen_present", "all_required_candidates_mapped", "source_geometry_complete",
                           "background_support_relations", "real_articulation_for_open", "target_fit",
                           "initial_visibility", "robot_collision_and_reach", "reset_consistency"],
            "target_conditioned_generation": False}


def generation_command(spec: HouseholdTaskSpec, python: str, root: Path, output: Path, solver_profile: str = "default") -> dict:
    if solver_profile not in ("default", "fast"):
        raise ValueError("UNSUPPORTED_UPSTREAM_SOLVER_PROFILE")
    if spec.family != "kitchen_cabinet_search":
        raise ValueError("UNSUPPORTED_TASK_FAMILY")
    if not spec.generation_constraints or spec.generation_constraints.get("candidate_count") != len(spec.locations):
        raise ValueError("GENERATION_REQUIREMENTS_MISSING_OR_CANDIDATE_COUNT_CONFLICT")
    overrides = ["configure_cycles_devices.use_gpu=False", "compose_indoors.terrain_enabled=False",
                 "compose_indoors.animate_cameras_enabled=False", "compose_indoors.floating_objs_enabled=False",
                 "compose_indoors.invisible_room_ceilings_enabled=False", "compose_indoors.hide_other_rooms_enabled=False",
                 "compose_indoors.lights_off_enabled=False", "restrict_solving.restrict_parent_rooms=['Kitchen']",
                 "restrict_solving.solve_max_rooms=1"]
    if solver_profile == "fast":
        # Bound expensive invalid proposal retries on CPU, preserving every
        # constraint and the strict final gate. This can increase rejection.
        overrides.append("SimulatedAnnealingSolver.max_invalid_candidates=1")
    return {"argv": [python, "-m", "rummagebench.authoring.tasks.infinigen", "run-upstream",
                     "--upstream-root", str(root.resolve()), "--requirements", str(Path(output).resolve().parent / "generation_request.json"),
                     "--seed", format(spec.scene_seed, "x"),
                     "--task", "coarse", "--output_folder", str(Path(output).resolve()),
                     "-g", "singleroom.gin", *(["fast_solve.gin"] if solver_profile == "fast" else []), "-p", *overrides],
            "cwd": str(Path(root).resolve()), "environment": dict(CPU_ENV),
            "request": generation_request(spec), "stage": "cpu_geometry",
            "upstream_api": "infinigen_examples.generate_indoors.main", "solver_profile": solver_profile}


def export_command(python: str, root: Path, input_folder: Path, output: Path, resolution: int = 512) -> dict:
    if not 64 <= resolution <= 4096:
        raise ValueError("invalid texture resolution")
    return {"argv": [python, "-m", "infinigen.tools.export", "--input_folder", str(input_folder.resolve()),
                     "--output_folder", str(output.resolve()), "-f", "usdc", "-r", str(resolution), "--omniverse"],
            "cwd": str(root.resolve()), "execution_status": "deferred",
            "reason": "PINNED_EXPORTER_FORCES_CYCLES_GPU", "whole_scene_articulation_preserved": False}


def normalize_solver_tag(tag: str) -> str:
    """Pinned EnumTag.__str__ format; preserve non-enum tags without eval()."""
    if not isinstance(tag,str):
        raise ValueError("INVALID_SOURCE_TAG")
    match = re.fullmatch(r"(-?)(Semantics|Subpart)\(([A-Za-z][A-Za-z0-9-]*)\)",tag)
    if match:
        sign,kind,value = match.groups()
        aliases = {'Semantics':{'ground':'GroundFloor','food-pantry':'FoodPantryItem',
                    'cloth-drape':'ClothDrapeItem','access-stand-near':'AccessStandingNear',
                    'access-with-hand':'AccessHand','FloorMat':'FloorMat'},
                   'Subpart':{'support':'SupportSurface'}}
        member = aliases[kind].get(value,''.join(part.title() for part in value.split('-')))
        return sign+kind+'.'+member
    return tag


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
        # Blender's USD exporter applies USD identifier rules, including a
        # prefix for numeric factory names and slashes in room names.
        usd_name = None if name is None else re.sub(r"[^A-Za-z0-9_]", "_", name)
        if usd_name is not None and (not usd_name or usd_name[0].isdigit()):
            usd_name = "_" + usd_name[1:]
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
                           "tags": sorted(normalize_solver_tag(t) for t in obj.get("tags", [])),
                           "raw_tags": sorted(obj.get("tags", [])), "generator": obj.get("generator"),
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


def generation_failure_reason(error_text: str) -> str:
    """Classify the final exception, not earlier warnings or parameter names."""
    errors = re.findall(r"^[A-Za-z][A-Za-z0-9_.]*(?:Error|Exception):.*$",error_text,re.M)
    cause = errors[-1] if errors else error_text
    if "ModuleNotFoundError" in cause or "ImportError" in cause or re.search(r"addon=.*not found",cause):
        return "UPSTREAM_DEPENDENCY_MISSING"
    if re.search(r"No configurable matching|Unknown configurable|gin.*binding",cause,re.I):
        return "CONFIGURATION_INVALID"
    if re.search(r"unsatisf|infeasible constraint|failed to satisfy constraints",cause,re.I):
        return "CONSTRAINT_UNSATISFIABLE"
    return "GENERATION_FAILED"


def run_cpu_generation(spec: HouseholdTaskSpec, root: Path, python: str, output: Path,
                       *, timeout_s: float = 300., retries: int = 0, solver_profile: str = "default") -> dict:
    if timeout_s <= 0 or not 0 <= retries <= 2:
        raise ValueError("invalid timeout/retry bound")
    audit = source_audit(root)
    output = Path(output).resolve()
    command = generation_command(spec, python, root, output / "geometry", solver_profile)
    inputs = {"command": command, "source": audit, "runtime": runtime_audit(python),
              "adapter_sha256": sha256_path(Path(__file__)),
              "authoring_sources": {name:sha256_path(Path(__file__).with_name(name))
                                    for name in ("household.py","difficulty.py","plan.py")}}
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
        # Keep the exact bridge / cost source for an auditable run even when
        # later development changes the working checkout. Hashes above bind it.
        source_output = output / "input_sources"
        source_output.mkdir(exist_ok=True)
        for name in ("infinigen.py", *inputs["authoring_sources"]):
            destination = source_output / name
            source = Path(__file__).with_name(name)
            if destination.exists() and sha256_path(destination) != sha256_path(source):
                raise ValueError("RESUME_SOURCE_SNAPSHOT_CONFLICT")
            if not destination.exists():
                shutil.copyfile(source, destination)
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
                for rel,digest in attempt.get("artifacts",{}).items():
                    if sha256_path(output / rel) != digest:
                        raise ValueError("RESUME_ARTIFACT_CONFLICT")
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
            cmd = generation_command(spec, python, root, geometry, solver_profile)
            atomic_write_json(attempt_dir / "generation_request.json", cmd["request"])
            own_source = Path(__file__).resolve().parents[3]
            env = dict(os.environ, **CPU_ENV, PYTHONPATH=os.pathsep.join([str(own_source), str(root.resolve())]))
            extension_root = inputs['runtime'].get('blender_extension_repository')
            if extension_root:
                env['FINDINGBENCH_BLENDER_EXTENSIONS'] = extension_root
            else:
                env.pop('FINDINGBENCH_BLENDER_EXTENSIONS',None)
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
                reason = generation_failure_reason(error_text)
            state_path, blend = geometry / "solve_state.json", geometry / "scene.blend"
            geometry_path = geometry / "findingbench_geometry.json"
            measurement = None
            artifacts = {}
            if reason is None:
                try:
                    metadata = read_solve_state(state_path)
                    if not metadata["kitchens"] or not _blend_valid(blend) or not geometry_path.is_file():
                        raise ValueError("incomplete generated kitchen/Blender artifact")
                except (OSError, ValueError) as exc:
                    reason = "GENERATION_OUTPUT_INVALID"
                    error_text += f"\n{type(exc).__name__}: {exc}"
            generated = reason is None
            if generated:
                try:
                    measurement = measure_generated_geometry(spec, state_path, geometry_path)
                except (OSError,ValueError,TypeError,KeyError) as exc:
                    measurement = {'proxy_filter_passed':False,'evaluation_ready_episodes':0,
                                   'rejections':['GENERATED_MEASUREMENT_INVALID'],
                                   'error':f'{type(exc).__name__}: {exc}'}
                atomic_write_json(attempt_dir / "measurement.json", measurement)
                artifacts = {str(p.relative_to(output)): sha256_path(p)
                             for p in (state_path, blend, geometry_path, attempt_dir / "measurement.json")}
                if not measurement["proxy_filter_passed"]:
                    reason = "GENERATED_REQUIREMENTS_REJECTED"
            attempt = {"retry": retry, "input_sha256": identity, "command": cmd, "exit_code": code,
                       "reason": reason, "elapsed_s": round(time.monotonic() - started, 3),
                       "error_excerpt": error_text[-3000:], "artifacts": artifacts,
                       "geometry_generated": generated, "measurement": measurement}
            atomic_write_json(checkpoint, attempt)
            attempts.append(attempt)
            if reason is None:
                break
        success = attempts[-1]["reason"] is None
        generated_attempts = sum(bool(a.get("geometry_generated")) for a in attempts)
        generated_count = int(generated_attempts > 0)  # retries retain the same independent layout/seed
        successful = attempts[-1] if success else None
        deferred_export = None
        if successful:
            actual_folder = output / f"attempt_{successful['retry']}" / "geometry"
            deferred_export = export_command(python, root, actual_folder, output / "export")
        artifacts = {ref: digest for attempt in attempts for ref, digest in attempt.get("artifacts",{}).items()}
        receipt = {"version": 2, "input_sha256": identity, "source": audit, "attempts": attempts,
                   "status": "geometry_generated" if success else "geometry_rejected" if generated_count else "generation_failed",
                   "generated_geometry_scenes": generated_count, "generated_geometry_attempts": generated_attempts,
                   "proxy_accepted_geometry_scenes": int(success),
                   "export": deferred_export, "exported_scene_files": 0,
                   "evaluation_ready_episodes": 0, "artifacts": artifacts,
                   "gpu_started": False, "gpu_device_isolation": "bwrap_private_dev_no_nvidia_or_drm",
                   "simulation_acceptance": "pending", "family": spec.family,
                   "requested_difficulty": spec.requested_difficulty,
                   "failure_reasons": dict(Counter(a["reason"] for a in attempts if a["reason"]))}
        atomic_write_json(receipt_path, receipt)
        return receipt


def inspect_geometry_receipt(spec: HouseholdTaskSpec, receipt_path: Path) -> dict:
    """Revalidate real completed upstream artifacts without rewriting run history.

    This also supports runs rejected by an older parser: an actual zero-exit
    upstream invocation, complete Blender/state/geometry files and matching
    environment request are still mandatory. No export/simulation is implied.
    """
    root = receipt_path.resolve().parent
    receipt = json.loads(receipt_path.read_text())
    original = json.loads((root/'input.json').read_text())
    if original['input_sha256'] != receipt['input_sha256']:
        raise ValueError('GENERATION_RECEIPT_INPUT_MISMATCH')
    completed=[]
    for attempt in receipt['attempts']:
        if attempt.get('exit_code') != 0:
            continue
        if attempt['command']['request'] != generation_request(spec):
            raise ValueError('GEOMETRY_REQUEST_SPEC_MISMATCH')
        folder = (root/f"attempt_{attempt['retry']}"/'geometry').resolve()
        if not folder.is_relative_to(root):raise ValueError('GEOMETRY_PATH_OUTSIDE_RUN')
        state,blend,geometry = (folder/name for name in ('solve_state.json','scene.blend','findingbench_geometry.json'))
        if not _blend_valid(blend) or not geometry.is_file() or not read_solve_state(state)['kitchens']:
            raise ValueError('COMPLETE_ACTUAL_KITCHEN_GEOMETRY_REQUIRED')
        for rel,digest in attempt.get('artifacts',{}).items():
            if sha256_path(root/rel) != digest:raise ValueError('GEOMETRY_RUN_ARTIFACT_CHANGED')
        completed.append({'folder':str(folder), 'measurement':measure_generated_geometry(spec,state,geometry),
                          'artifacts':{str(p):sha256_path(p) for p in (state,blend,geometry)}})
    if not completed:raise ValueError('NO_COMPLETE_UPSTREAM_GEOMETRY')
    return {'version':1,'stage':'inspect_infinigen_geometry','status':
            'geometry_generated' if all(d['measurement']['proxy_filter_passed'] for d in completed) else 'geometry_rejected',
            'original_receipt':str(receipt_path.resolve()),'original_receipt_sha256':sha256_path(receipt_path),
            'inspection_source_sha256':sha256_path(Path(__file__)), 'scene_seed':spec.scene_seed,
            'background_request':generation_request(spec), 'completed':completed,
            'counts':{'real_geometry_scenes':len({d['artifacts'][str(Path(d['folder'])/'scene.blend')] for d in completed}),
                      'proxy_accepted_geometry_scenes':sum(d['measurement']['proxy_filter_passed'] for d in completed),
                      'exported_scenes':0,'compiled_episodes':0,'simulation_verified_episodes':0,'evaluation_ready_episodes':0},
            'gpu_started':False,'physical_acceptance':'pending'}


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
    stage.Reload()
    if stage.GetCompositionErrors():
        raise ValueError(f"USD_COMPOSITION_INVALID:{stage.GetCompositionErrors()}")
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
    dependencies = usd_dependency_ledger(scene_path)
    return {"status": "export_inspected", "scene_sha256": sha256_path(scene_path),
            "solve_state_sha256": metadata["source_sha256"], "mesh_count": len(meshes),
            "joint_paths": joints, "dependency_ledger": dependencies, "source_to_usd": mapping, "unmapped_source_objects": missing,
            "ambiguous_source_objects": ambiguous, "all_mesh_paths": meshes,
            "evaluation_ready_episodes": 0, "physical_interactivity_validated": False}


def usd_dependency_ledger(scene_path: Path) -> dict:
    """Transitive local-file dependencies, including unloaded payloads and textures.

    USD packages, remote resolvers, asset expressions and texture sequences are
    deliberately unsupported. Flatten changes composition, not texture bytes.
    """
    from pxr import Ar, Sdf, UsdUtils

    root = Path(scene_path).resolve()
    layers, resources, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root)))
    if unresolved:
        raise ValueError(f"USD_DEPENDENCY_UNRESOLVED:{list(unresolved)}")
    files, edges = {}, []

    def checked_path(value, kind):
        value = str(value)
        if not value or any(token in value for token in ('://', '[', ']', '<UDIM>', '${', '####')):
            raise ValueError(f"USD_DEPENDENCY_KIND_UNSUPPORTED:{value}")
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"USD_DEPENDENCY_MISSING:{value}")
        path = path.resolve()
        files[str(path)] = {"sha256": sha256_path(path), "kind": kind,
                            "distribution": "local_external; no license grant or upload"}
        return str(path)

    for layer in layers:
        checked_path(layer.realPath, 'usd_layer')
    for resource in resources:
        checked_path(resource, 'resource')

    def edge(layer, authored, kind, spec_path):
        if not authored:  # Internal prim reference has no external file.
            return
        anchored = Sdf.ComputeAssetPathRelativeToLayer(layer, str(authored))
        resolved = Ar.GetResolver().Resolve(anchored)
        path = checked_path(resolved, 'usd_layer' if kind in ('reference','payload','sublayer') else 'resource')
        edges.append({"source_layer": str(Path(layer.realPath).resolve()), "spec_path": str(spec_path),
                      "authored_path": str(authored), "resolved_path": path, "kind": kind})

    for layer in layers:
        for value in layer.subLayerPaths:
            edge(layer, value, 'sublayer', '/')
        def visit(path):
            spec = layer.GetObjectAtPath(path)
            if isinstance(spec, Sdf.PrimSpec):
                for value in spec.referenceList.GetAppliedItems():
                    edge(layer, value.assetPath, 'reference', path)
                for value in spec.payloadList.GetAppliedItems():
                    edge(layer, value.assetPath, 'payload', path)
            elif isinstance(spec, Sdf.AttributeSpec):
                values = [spec.default]
                values.extend(layer.QueryTimeSample(path, t) for t in layer.ListTimeSamplesForPath(path))
                for value in values:
                    assets = ([value] if isinstance(value, Sdf.AssetPath) else
                              list(value) if spec.typeName == Sdf.ValueTypeNames.AssetArray and value is not None else [])
                    for asset in assets:
                        edge(layer, asset.path, 'asset_attribute', path)
        layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if str(root) not in files:
        raise ValueError('USD_ROOT_DEPENDENCY_MISSING')
    return {"version": 1, "root": str(root), "support": "local_files_only", "files": files,
            "references": sorted(edges, key=lambda e: (e['source_layer'],e['spec_path'],e['authored_path'],e['kind']))}


def verify_usd_dependency_ledger(ledger: dict) -> None:
    """Hash first: cached OpenUSD layers must not hide external file changes."""
    from pxr import Ar, Sdf

    if ledger.get('version') != 1 or ledger.get('support') != 'local_files_only' or not ledger.get('files'):
        raise ValueError('USD_DEPENDENCY_LEDGER_UNSUPPORTED')
    for value, record in ledger['files'].items():
        path = Path(value)
        if (not path.is_absolute() or not path.is_file() or str(path.resolve()) != value
                or not record.get('sha256') or sha256_path(path) != record['sha256']):
            raise ValueError(f'USD_DEPENDENCY_CHANGED_OR_MISSING:{value}')
    for ref in ledger['references']:
        if ref['source_layer'] not in ledger['files'] or ref['resolved_path'] not in ledger['files']:
            raise ValueError('USD_DEPENDENCY_REFERENCE_NOT_HASHED')
        layer = Sdf.Layer.FindOrOpen(ref['source_layer'])
        resolved = str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(layer, ref['authored_path'])))
        if not resolved or str(Path(resolved).resolve()) != ref['resolved_path']:
            raise ValueError('USD_DEPENDENCY_REFERENCE_RESOLUTION_CHANGED')


def measure_generated_geometry(spec: HouseholdTaskSpec, solve_path: Path, geometry_path: Path) -> dict:
    """Apply production cost calculation and bins to actual Blender geometry.

    This is a furniture-center Euclidean MOVE proxy. It does not measure robot
    interaction poses, TURN, traversable paths, visibility or real search cost.
    """
    import numpy as np
    from rummagebench.authoring.tasks.difficulty import exact_search_cost, interval_label, euclidean_proxies

    metadata = read_solve_state(solve_path)
    geometry = json.loads(geometry_path.read_text())
    if geometry.get('solve_state_sha256') != metadata['source_sha256']:
        raise ValueError('GEOMETRY_SOLVE_STATE_HASH_MISMATCH')
    req = spec.generation_constraints
    if not req or req['candidate_count'] != len(spec.locations):
        raise ValueError('GENERATION_REQUIREMENTS_MISSING')
    source = metadata['objects']
    def kitchen_ancestor(key):
        seen, queue = set(), [key]
        while queue:
            node = queue.pop()
            if node in seen: continue
            seen.add(node); queue.extend(r['parent'] for r in source[node]['relations'])
        return any('Semantics.Kitchen' in source[n]['tags'] for n in seen)
    candidates = sorted(k for k,o in source.items() if 'Semantics.Storage' in o['tags'] and kitchen_ancestor(k))
    others = sorted(k for k,o in source.items() if 'Semantics.KitchenCounter' in o['tags'] and kitchen_ancestor(k))
    report = {'scope':'cpu_geometry_proxy_only', 'candidate_sources':candidates, 'candidate_count':len(candidates),
              'possible_additional_storage_sources':others,
              'additional_storage_status':'counter/drawer cavities require physical classification',
              'requested_difficulty':spec.requested_difficulty, 'requirements':req, 'proxy_filter_passed':False,
              'real_scene_difficulty_certified':False, 'evaluation_ready_episodes':0, 'rejections':[]}
    constraints = geometry.get('upstream_constraints')
    report['upstream_constraints'] = constraints
    if not constraints or constraints.get('passed') is not True:
        report['rejections'].append('UPSTREAM_FINAL_CONSTRAINTS_NOT_SATISFIED')
    if len(candidates) != req['candidate_count']:
        report['rejections'].append('CANDIDATE_COUNT_MISMATCH'); return report
    report['candidate_mapping'] = {loc.location_id:source_id for loc,source_id in zip(spec.locations,candidates)}
    missing = [k for k in candidates + metadata['kitchens'] if k not in geometry['objects'] or geometry['objects'][k].get('aabb') is None]
    if missing:
        report['rejections'].append('CANDIDATE_OR_ROOM_GEOMETRY_UNRESOLVED');report['unresolved_sources']=missing;return report
    kitchen_boxes = [geometry['objects'][k]['aabb'] for k in metadata['kitchens']]
    if len(kitchen_boxes) != 1:
        report['rejections'].append('SINGLE_KITCHEN_GEOMETRY_REQUIRED');return report
    bounds = np.asarray(kitchen_boxes[0],float)
    centers = [geometry['objects'][k]['center'][:2] for k in candidates]
    # Proposal start chosen using the room envelope and all furniture AABBs,
    # without reading which candidate contains the target. Runtime checks are
    # still required: a rectangular envelope is not a traversability map.
    lo, hi = bounds[0,:2] + .5, bounds[1,:2] - .5
    obstacles = [np.asarray(geometry['objects'][k]['aabb'],float) for k,o in source.items()
                 if 'Semantics.Furniture' in o['tags'] and kitchen_ancestor(k)
                 and geometry['objects'].get(k,{}).get('aabb') is not None]
    points=[]
    for x in np.arange(lo[0],hi[0]+1e-8,.5):
        for y in np.arange(lo[1],hi[1]+1e-8,.5):
            if not any(b[0,0]-.43 <= x <= b[1,0]+.43 and b[0,1]-.43 <= y <= b[1,1]+.43 for b in obstacles):
                points.append([float(x),float(y)])
    if not points:
        report['rejections'].append('CPU_START_ENVELOPE_NO_FREE_PROPOSAL');return report
    midpoint=(lo+hi)/2
    start=min(points,key=lambda p:(float(np.linalg.norm(np.asarray(p)-midpoint)),p))
    model = req['search_model']; proxies=euclidean_proxies(start, centers)
    matrix=np.zeros((len(candidates)+1,len(candidates)+1))
    matrix[:-1,:-1]=proxies['pairwise'];matrix[-1,:-1]=proxies['start_to_candidate'];matrix[:-1,-1]=matrix[-1,:-1]
    movement=np.ceil(matrix/model['movement_proxy_m_per_step'])
    costs=exact_search_cost(movement,[loc.prior for loc in spec.locations],
                            [model['inspection_operations']]*len(candidates),finish_cost=model['finish_operations'])
    bins={k:interval_label(costs['expected_movement' if k=='spatial' else 'expected_operations'],req['difficulty_intervals'][k])
          for k in ('spatial','operations')}
    report.update(geometry_proxy=proxies, costs=costs, proxy_difficulty=bins,
                  start_pose_proposal={'position':[*start,float(bounds[0,2]+.0053)],'orientation':[0.,0.,0.,1.]},
                  start_scope='room AABB/furniture AABB proposal; floor and collision pending')
    for key, label in bins.items():
        if label != spec.requested_difficulty[key]:report['rejections'].append(key.upper()+'_PROXY_OUTSIDE_REQUEST')
    report['proxy_filter_passed']=not report['rejections']
    return report


def apply_search_constraints(graph, count, gap, *, cl, cu, Semantics, shelves):
    """Pinned upstream DSL: initializer bounds plus stage-relevant whole-set count."""
    rooms=cl.scene()[{Semantics.Room,-Semantics.Object}][Semantics.Kitchen].excludes(cu.room_types)
    storage=(cl.scene()[{Semantics.Object,-Semantics.Room}][Semantics.Storage]
             .related_to(rooms,cu.on_floor).related_to(rooms,cu.against_wall))
    graph.constraints['findingbench_storage_count']=rooms.all(lambda r:storage.related_to(r).count().equals(count))
    graph.constraints['findingbench_cabinet_count']=rooms.all(lambda r:storage[shelves.KitchenCabinetFactory].related_to(r).count().equals(count))
    graph.constraints['findingbench_storage_separation']=rooms.all(lambda r:cl.min_distance_internal(storage.related_to(r))>=gap)
    all_storage=cl.scene()[{Semantics.Object,-Semantics.Room}][Semantics.Storage]
    # Cross-stage count cannot be an initializer Bound (the pinned greedy
    # coverage checker rejects overlaps). clamp_min(0) is identical for
    # nonnegative counts, but keeps this an evaluated hard constraint,
    # while the floor/wall bounds above still initialise the factories.
    # The explicit kitchen relation is essential: the pinned evaluator
    # marks a bare global count irrelevant in every greedy stage. Nested
    # or indirect extra storage is still checked by the complete postfilter.
    graph.constraints['findingbench_complete_storage_count']=all_storage.related_to(rooms).count().clamp_min(0).equals(count)
    return graph


def _run_upstream_bridge() -> None:
    """Thin bridge to the pinned upstream Problem/DSL and main(), no generator fork."""
    import argparse
    import sys
    from types import SimpleNamespace

    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['run-upstream'])
    parser.add_argument('--upstream-root',type=Path,required=True)
    parser.add_argument('--requirements',type=Path,required=True)
    parser.add_argument('--seed',required=True)
    parser.add_argument('--output_folder',type=Path,required=True)
    parser.add_argument('--task',nargs='+',default=['coarse'])
    parser.add_argument('-g','--configs',nargs='+',default=['singleroom.gin'])
    parser.add_argument('-p','--overrides',nargs='+',default=[])
    args=parser.parse_args()
    if 'configure_cycles_devices.use_gpu=False' not in args.overrides or args.task != ['coarse']:
        raise ValueError('CPU_ONLY_COARSE_BRIDGE_REQUIRED')
    request=json.loads(args.requirements.read_text());req=request['generation_constraints']
    count=int(req['candidate_count']);gap=float(req['storage_min_distance_m'])
    if not 2<=count<=8 or not 0<=gap<=5:raise ValueError('INVALID_UPSTREAM_CONSTRAINTS')
    sys.path.insert(0,str(args.upstream_root.resolve()))
    extension_root = os.environ.get('FINDINGBENCH_BLENDER_EXTENSIONS')
    if extension_root:
        import bpy
        repo = next(r for r in bpy.context.preferences.extensions.repos if r.module == 'blender_org')
        repo.custom_directory = extension_root
        repo.use_custom_directory = True
        repo.use_remote_url = False
        # Existing local official extension only; the sandbox has no network.
        bpy.ops.preferences.addon_enable(module='bl_ext.blender_org.extra_mesh_objects')
    # The pinned package's __init__.py is empty although generate_indoors
    # imports Terrain there. Expose the real upstream class without editing
    # the checkout or substituting terrain/geometry implementations.
    import infinigen.terrain as terrain
    from infinigen.terrain.core import Terrain
    terrain.Terrain = Terrain
    from infinigen_examples.constraints import home, util as cu
    from infinigen.core.constraints import constraint_language as cl
    from infinigen.core.tags import Semantics
    from infinigen.assets.objects import shelves
    original=home.home_furniture_constraints
    def with_search_constraints():
        graph=original()
        return apply_search_constraints(graph,count,gap,cl=cl,cu=cu,Semantics=Semantics,shelves=shelves)
    home.home_furniture_constraints=with_search_constraints
    # Upstream checks the complete room problem at EACH greedy stage. At the
    # floor/wall stage the ceiling lights and freestanding island do not yet
    # exist. Preserve its default intermediate warnings, then strictly check
    # ALL kitchen constraints after every solving stage, before population.
    from infinigen.core.constraints.example_solver.solve import Solver
    from infinigen.core.constraints.example_solver import greedy
    from infinigen.core.constraints.evaluator.evaluate import evaluate_problem
    from infinigen.core.util.pipeline import RandomStageExecutor
    original_solve = Solver.solve_objects
    original_stage = RandomStageExecutor.run_stage
    live = {}
    def record_solve(self, consgraph, *args, **kwargs):
        result = original_solve(self, consgraph, *args, **kwargs)
        live.update(solver=self, problem=consgraph)
        return result
    def final_gate(self, name, fn, *args, **kwargs):
        if name == 'populate_assets':
            state = live['solver'].state
            kitchens = [key for key,o in state.objs.items() if Semantics.Kitchen in o.tags]
            if len(kitchens) != 1:raise ValueError('SINGLE_KITCHEN_FINAL_CONSTRAINT_SCOPE_REQUIRED')
            active = {key:o.active for key,o in state.objs.items()}
            try:
                greedy.update_active_flags(state,{cu.variable_room:kitchens[0]})
                result = evaluate_problem(live['problem'],state,enable_loss=False)
                violations = {key:float(value) for key,value in result.violations.items()}
                live['constraints'] = {'scope':'all_upstream_and_findingbench_constraints_in_solved_kitchen',
                    'violations':violations, 'passed':all(np.isfinite(v) and v == 0 for v in violations.values()),
                    'physical_acceptance':False}
            finally:
                for key,value in active.items():greedy.set_active(state,key,value)
        return original_stage(self,name,fn,*args,**kwargs)
    import numpy as np
    Solver.solve_objects = record_solve
    RandomStageExecutor.run_stage = final_gate
    from infinigen_examples.generate_indoors import main
    main(SimpleNamespace(seed=args.seed,configs=args.configs,overrides=args.overrides,input_folder=None,
                         output_folder=args.output_folder,task=args.task,task_uniqname=None))
    # Inspect actual evaluated meshes, not fabricated object/category counts.
    import bpy
    from mathutils import Vector
    state_path=args.output_folder/'solve_state.json';raw=json.loads(state_path.read_text())
    graph=bpy.context.evaluated_depsgraph_get();objects={}
    for key,item in raw['objs'].items():
        obj=bpy.data.objects.get(item.get('obj') or '')
        vertices=[]
        if obj is not None:
            for node in [obj,*obj.children_recursive]:
                if node.type=='MESH':
                    evaluated=node.evaluated_get(graph)
                    vertices.extend(list(evaluated.matrix_world@Vector(c)) for c in evaluated.bound_box)
        if vertices:
            import numpy as np
            points=np.asarray(vertices,float);lo=points.min(axis=0);hi=points.max(axis=0)
            objects[key]={'aabb':[lo.tolist(),hi.tolist()],'center':((lo+hi)/2).tolist()}
        else:objects[key]={'aabb':None,'reason':'no evaluated mesh mapped to solver object'}
    atomic_write_json(args.output_folder/'findingbench_geometry.json',{'version':1,'objects':objects,
        'solve_state_sha256':sha256_path(state_path),'source':'actual_Blender_evaluated_mesh_AABBs',
        'upstream_constraints':live.get('constraints',{'passed':False,'reason':'final gate not executed'}),
        'gpu_started':False,'physical_acceptance':'pending'})


if __name__ == '__main__':
    _run_upstream_bridge()
