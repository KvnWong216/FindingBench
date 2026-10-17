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
    taskset = shutil.which('taskset')
    if bwrap is None or taskset is None:
        raise RuntimeError("CPU_DEVICE_SANDBOX_UNAVAILABLE")
    cores=sorted(os.sched_getaffinity(0))[:2]
    if not cores:raise RuntimeError('CPU_AFFINITY_UNAVAILABLE')
    return [bwrap, "--ro-bind", "/", "/", "--bind", str(writable), str(writable),
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--unshare-pid",
            "--unshare-net", "--unshare-user", "--die-with-parent", "--",
            taskset,'--cpu-list',','.join(map(str,cores)),*argv]


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
    if solver_profile not in ("default", "fast", "kitchen", "kitchen-fast"):
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
    if solver_profile in ("kitchen", "kitchen-fast"):
        # Small-room floorplan preset, with the upstream's normal five
        # relation-assignment retries and enough steps for everyday props.
        # These are proposal budgets, never changes to acceptance predicates.
        overrides.extend(["SimulatedAnnealingSolver.max_invalid_candidates=5",
                          "compose_indoors.solve_steps_large=300",
                          f"compose_indoors.solve_steps_medium={40 if solver_profile == 'kitchen-fast' else 80}",
                          f"compose_indoors.solve_steps_small={5 if solver_profile == 'kitchen-fast' else 25}"])
    return {"argv": [python, "-m", "rummagebench.authoring.tasks.infinigen", "run-upstream",
                     "--upstream-root", str(root.resolve()), "--requirements", str(Path(output).resolve().parent / "generation_request.json"),
                     "--seed", format(spec.scene_seed, "x"),
                     *(["--kitchen-stage-budgets"] if solver_profile in ("kitchen", "kitchen-fast") else []),
                     "--task", "coarse", "--output_folder", str(Path(output).resolve()),
                     "-g", "singleroom.gin", *(["fast_solve.gin"] if solver_profile in ("fast","kitchen","kitchen-fast") else []), "-p", *overrides],
            "cwd": str(Path(root).resolve()), "environment": dict(CPU_ENV),
            "cpu_affinity": sorted(os.sched_getaffinity(0))[:2],
            "request": generation_request(spec), "stage": "cpu_geometry",
            "upstream_api": "infinigen_examples.generate_indoors.main", "solver_profile": solver_profile}


def export_command(python: str, root: Path, input_folder: Path, output: Path, resolution: int = 512) -> dict:
    if not 64 <= resolution <= 4096:
        raise ValueError("invalid texture resolution")
    return {"argv": [python, "-m", "rummagebench.authoring.tasks.infinigen", "export-upstream",
                     "--upstream-root", str(root.resolve()), "--input_folder", str(input_folder.resolve()),
                     "--output_folder", str(output.resolve()), "-r", str(resolution), "--gpu", "5"],
            "cwd": str(root.resolve()), "execution_status": "deferred",
            "reason": "PINNED_EXPORTER_FORCES_CYCLES_GPU", "physical_gpu": 5,
            "whole_scene_articulation_preserved": False}


def export_compatibility_source(source: str, *, modern_cycles: bool) -> tuple[str, int]:
    """Remove only obsolete tile hints absent from Blender 4.2's Cycles RNA.

    All geometry, materials, bake calls and upstream export operations stay
    upstream code. Require the pinned four assignments to avoid broad edits.
    """
    if not modern_cycles:
        return source, 0
    import ast
    tree = ast.parse(source)
    removed = 0
    class RemoveOldTiles(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal removed
            if len(node.targets) == 1 and ast.unparse(node.targets[0]) in (
                    "bpy.context.scene.cycles.tile_x", "bpy.context.scene.cycles.tile_y"):
                if ast.unparse(node.value) != "image_res":
                    raise ValueError("UNSUPPORTED_EXPORT_TILE_ASSIGNMENT")
                removed += 1
                return None
            return self.generic_visit(node)
    tree = RemoveOldTiles().visit(tree)
    if removed != 4:
        raise ValueError("PINNED_EXPORT_TILE_COMPATIBILITY_MISMATCH")
    return ast.unparse(ast.fix_missing_locations(tree)), removed


def _run_export_bridge(argv) -> int:
    """Run the actual pinned exporter, confined to one authorized GPU."""
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--input_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("-r", "--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, choices=[4, 5], default=5)
    parser.add_argument("--gpu-isolated", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    audit = source_audit(args.upstream_root)
    if not 64 <= args.resolution <= 4096 or not _blend_valid(args.input_folder / "scene.blend"):
        raise ValueError("EXPORT_RESOLUTION_OR_REAL_BLEND_INVALID")
    if not args.gpu_isolated:
        from rummagebench.sim.gpu_isolation import resolve_gpu, isolated_gpu_command
        command, env = isolated_gpu_command([sys.executable, "-m", __spec__.name,
                                             "export-upstream", *argv, "--gpu-isolated"], resolve_gpu(args.gpu))
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), str(args.upstream_root.resolve())])
        return subprocess.run(command, env=env, timeout=1800, check=False).returncode
    # No Torch or Kit import in this bpy environment. CUDA visibility is
    # enforced by one exposed character device, checked before bpy loads.
    exposed = sorted(str(p) for p in Path("/dev").glob("nvidia[0-9]*"))
    if (exposed != [os.environ.get("FINDINGBENCH_GPU_DEVICE")]
            or os.environ.get('FINDINGBENCH_GPU_INDEX') != str(args.gpu)):
        raise ValueError("BLENDER_SINGLE_GPU_DEVICE_REQUIRED")
    args.output_folder.mkdir(parents=True, exist_ok=False)
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str((args.input_folder / "scene.blend").resolve()))
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    devices = [d for d in preferences.devices if d.type == "CUDA"]
    if len(devices) != 1:
        raise ValueError("BLENDER_SINGLE_CUDA_DEVICE_REQUIRED")
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import infinigen.tools.export as upstream
    source_path = args.upstream_root / "infinigen/tools/export.py"
    source, edits = export_compatibility_source(source_path.read_text(),
                                               modern_cycles=not hasattr(bpy.context.scene.cycles, "tile_x"))
    namespace = {"__name__": "findingbench_pinned_export", "__file__": str(source_path), "__package__": upstream.__package__}
    exec(compile(source, str(source_path), "exec"), namespace)
    import logging
    logging.basicConfig(filename=args.output_folder / "export_logs.log", level=logging.DEBUG)
    folder = namespace["export_scene"](args.input_folder / "scene.blend", args.output_folder,
                                      format="usdc", image_res=args.resolution, vertex_colors=False,
                                      individual_export=False, omniverse_export=True)
    solve = args.input_folder / "solve_state.json"
    shutil.copyfile(solve, args.output_folder / "solve_state.json")
    scenes = list(folder.glob("*.usdc"))
    if len(scenes) != 1:
        raise ValueError("UPSTREAM_EXPORT_SCENE_MISSING_OR_AMBIGUOUS")
    inspection = inspect_export(scenes[0], solve)
    atomic_write_json(args.output_folder / "export.json", {"version":1,"status":"exported_pending_simulation",
        "source":audit,"input_blend":str((args.input_folder / "scene.blend").resolve()),
        "input_blend_sha256":sha256_path(args.input_folder / "scene.blend"),
        "source_solve_state":str(solve.resolve()),"solve_state_sha256":sha256_path(solve),
        "scene":str(scenes[0].resolve()),"gpu_executed":True,
        "physical_gpu":args.gpu,"gpu_uuid":os.environ["FINDINGBENCH_GPU_UUID"],
        "cycles_device":devices[0].id,"compatibility_removed_obsolete_tile_hints":edits,
        "inspection":inspection,"evaluation_ready_episodes":0})
    return 0


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
    if 'POPULATED_STORAGE_REFINEMENT_UNSATISFIED' in cause:
        return 'POPULATED_STORAGE_REFINEMENT_UNSATISFIED'
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
                                    for name in ("household.py","difficulty.py","plan.py","infinigen_components.py")}}
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
            refinement = geometry / 'populated_storage_refinement.json'
            if refinement.is_file():
                artifacts[str(refinement.relative_to(output))] = sha256_path(refinement)
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
        # Blender emits an object Xform and its child Mesh with the same
        # name. A unique ancestral root containing every exact-name match is
        # one object, not an ambiguous binding. Separate roots still reject.
        roots = [p for p in found if all(q == p or q.startswith(p + "/") for q in found)]
        if len(roots) == 1:
            mapping[key] = roots[0]
        elif found:
            ambiguous.append(key)
        else:
            missing.append(key)
    if not meshes:
        raise ValueError("USD_MESH_GEOMETRY_MISSING")
    components = {}
    # Pinned Blender room parts are independent objects, so the USD exporter
    # can emit sibling prims instead of descendants of the solver room mesh.
    # Only exact known component names with actual exported mesh geometry
    # count; arbitrary prefixes never infer source ownership.
    for key,obj in metadata["objects"].items():
        if not {"Semantics.Room","Semantics.Kitchen"} & set(obj["tags"]) or obj["usd_name"] is None:
            continue
        # Pinned populate.py changes the solver room object to <room>.meshed;
        # independently split floor/wall names retain <room>.<component>.
        base = obj["usd_name"]
        if obj["blender_name"].endswith(".meshed"):
            base = base.removesuffix("_meshed")
        parts = {}
        for component in ("floor","wall","ceiling","exterior","meshed"):
            found = [p for p in names.get(base+"_"+component,[])
                     if any(m==p or m.startswith(p+"/") for m in meshes)]
            roots = [p for p in found if all(q == p or q.startswith(p + "/") for q in found)]
            if found and len(roots) != 1:
                ambiguous.append(key+"."+component)
            elif roots:
                parts[component] = roots[0]
        components[key]=parts
    dependencies = usd_dependency_ledger(scene_path)
    return {"status": "export_inspected", "scene_sha256": sha256_path(scene_path),
            "solve_state_sha256": metadata["source_sha256"], "mesh_count": len(meshes),
            "joint_paths": joints, "dependency_ledger": dependencies, "source_to_usd": mapping, "unmapped_source_objects": missing,
            "ambiguous_source_objects": ambiguous, "all_mesh_paths": meshes,
            "source_component_paths":components,
            "evaluation_ready_episodes": 0, "physical_interactivity_validated": False}


def source_mesh_owner(inspection: dict, mesh_path: str) -> tuple[str | None,str | None]:
    matched = [(len(path),key,None) for key,path in inspection["source_to_usd"].items()
               if mesh_path==path or mesh_path.startswith(path+"/")]
    for key,parts in inspection.get("source_component_paths",{}).items():
        matched += [(len(path),key,component) for component,path in parts.items()
                    if mesh_path==path or mesh_path.startswith(path+"/")]
    if not matched:return None,None
    matched.sort(key=lambda v:(-v[0],v[1],v[2] or ''))
    longest = [m for m in matched if m[0]==matched[0][0]]
    if len({m[1] for m in longest})!=1:
        raise ValueError("USD_MESH_SOURCE_COMPONENT_OWNERSHIP_AMBIGUOUS")
    return matched[0][1],matched[0][2]


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
    if req.get('source_kitchen_cabinets_closed_only') is True:
        census=geometry.get('cabinet_component_census') or {}
        if (census.get('status')!='source_complete_pending_binding' or not census.get('sources')
                or any(c['kind']=='none' for s in census['sources'] for c in s['cells'])):
            report['rejections'].append('CONTROLLED_CLOSED_FAMILY_REQUIRES_COMPLETE_CLOSED_SOURCE_CENSUS')
    if 'source_sink_parameter_ranges_m' in req:
        sinks=sorted(k for k,o in source.items() if 'Semantics.Sink' in o['tags'] and kitchen_ancestor(k))
        sampled=geometry.get('source_sink_parameters_m',{})
        valid=bool(sinks) and set(sampled)==set(sinks)
        if valid:
            try:
                valid=all(np.isfinite(sampled[key][field]) and lo<=sampled[key][field]<=hi
                    for key in sinks for field,(lo,hi) in req['source_sink_parameter_ranges_m'].items())
            except (KeyError,TypeError,ValueError):valid=False
        if not valid:report['rejections'].append('SOURCE_SINK_PARAMETER_REQUEST_NOT_SATISFIED')
        report['source_sink_parameters_m']=sampled
    if len(candidates) != req['candidate_count']:
        report['rejections'].append('CANDIDATE_COUNT_MISMATCH'); return report
    invalid_cabinets=[key for key in candidates if 'FromGenerator(KitchenCabinetFactory)' not in source[key]['tags']]
    if invalid_cabinets:
        report['rejections'].append('CANDIDATE_IS_NOT_KITCHEN_CABINET')
        report['invalid_cabinet_sources']=invalid_cabinets
    if 'source_storage_dimensions_m' in req:
        ranges=np.asarray(req['source_storage_dimensions_m'],float)
        sampled=geometry.get('source_storage_factory_dimensions_m',{})
        report['source_storage_factory_dimensions_m']=sampled
        valid=(ranges.shape==(3,2) and np.isfinite(ranges).all() and np.all(ranges[:,0]<=ranges[:,1])
               and set(sampled)==set(candidates))
        if valid:
            values=np.asarray([sampled[key] for key in candidates],float)
            valid=(values.shape==(len(candidates),3) and np.isfinite(values).all()
                   and np.all(values>=ranges[:,0]) and np.all(values<=ranges[:,1]))
        if not valid:report['rejections'].append('SOURCE_STORAGE_FACTORY_DIMENSIONS_NOT_VERIFIED')
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
    """Generate cabinets directly; retain hard whole-set storage validation.

    The generic storage initializer can fill its bound with bookcases before
    cabinets are proposed. Keep its original 1..7 predicate, but make only
    the cabinet-specific bound introduce kitchen storage. clamp_min(0) does
    not change a nonnegative count's value.
    """
    rooms=cl.scene()[{Semantics.Room,-Semantics.Object}][Semantics.Kitchen].excludes(cu.room_types)
    storage=(cl.scene()[{Semantics.Object,-Semantics.Room}][Semantics.Storage]
             .related_to(rooms,cu.on_floor).related_to(rooms,cu.against_wall))
    all_rooms=cl.scene()[{Semantics.Room,-Semantics.Object}]
    original_storage=(cl.scene()[{Semantics.Object,-Semantics.Room}][Semantics.Furniture][Semantics.Storage]
                      .related_to(all_rooms,cu.on_floor).related_to(all_rooms,cu.against_wall))
    graph.constraints['storage']=all_rooms.all(lambda r:original_storage.related_to(r).count().clamp_min(0).in_range(1,7))
    graph.constraints['findingbench_storage_count']=rooms.all(lambda r:storage.related_to(r).count().clamp_min(0).equals(count))
    graph.constraints['findingbench_cabinet_count']=rooms.all(lambda r:storage[shelves.KitchenCabinetFactory].related_to(r).count().equals(count))
    # The pinned evaluator requires a constraint's domain to satisfy the
    # stage's factory filter. A generic Storage distance is IRRELEVANT when
    # solving only KitchenCabinetFactory, so the optimizer would report zero
    # violations while fresh full-room evaluation still rejects spacing.
    # The exact full storage count and cabinet count make these sets equal
    # at final acceptance; explicitly tag the domain for stage relevance.
    cabinets=storage[shelves.KitchenCabinetFactory]
    graph.constraints['findingbench_storage_separation']=rooms.all(lambda r:cl.min_distance_internal(cabinets.related_to(r))>=gap)
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


def apply_source_storage_dimensions(factory, bounds):
    """Sample the real upstream factory's existing Dimensions parameter.

    Preserve its seed context and every mesh/material routine. Each caller
    still needs populated geometry, complete asset binding and physical gates.
    """
    import numpy as np
    ranges=np.asarray(bounds,float)
    original_ranges=np.asarray([[.25,.35],[1.,4.],[.5,1.3]])
    if (ranges.shape!=(3,2) or not np.isfinite(ranges).all() or np.any(ranges[:,0]>ranges[:,1])
            or np.any(ranges[:,0]<original_ranges[:,0]) or np.any(ranges[:,1]>original_ranges[:,1])):
        raise ValueError('SOURCE_STORAGE_DIMENSIONS_OUTSIDE_UPSTREAM_DOMAIN')
    original=factory.sample_params
    def sample(self):
        if self.dimensions is None:
            self.dimensions=tuple(float(np.random.uniform(lo,hi)) for lo,hi in ranges)
        # KitchenSpaceFactory supplies EXPLICIT dimensions for its nested
        # drawers and upper cabinets. Preserve those real upstream calls;
        # only standalone state candidates are sampled here and checked by
        # the complete postfilter. Nested storage remains an acceptance gate.
        return original(self)
    factory.sample_params=sample


def apply_source_sink_parameters(factory, ranges):
    """Condition real upstream scalar draws, preserving every geometry call."""
    import numpy as np
    original_ranges={'Width':(.4,1.),'Margin':(.02,.05)}
    if not ranges or set(ranges)-set(original_ranges):
        raise ValueError('UNSUPPORTED_SOURCE_SINK_PARAMETER')
    for key,(lo,hi) in ranges.items():
        old_lo,old_hi=original_ranges[key]
        if not np.isfinite([lo,hi]).all() or not old_lo<=lo<=hi<=old_hi:
            raise ValueError('SOURCE_SINK_PARAMETER_OUTSIDE_UPSTREAM_DOMAIN')
    original=factory.sample_parameters
    def sample(*args,**kwargs):
        params=original(*args,**kwargs)
        for key,(lo,hi) in ranges.items():
            old_lo,old_hi=original_ranges[key]
            value=float(params[key])
            if not old_lo<=value<=old_hi:raise ValueError('UPSTREAM_SINK_SCALAR_DOMAIN_CHANGED')
            params[key]=float(lo+(value-old_lo)/(old_hi-old_lo)*(hi-lo))
        return params
    factory.sample_parameters=staticmethod(sample)


def apply_closed_kitchen_cabinets(factory):
    """Use the existing constructor's closed-cell option; no geometry fork."""
    import inspect
    from functools import wraps
    original=factory.__init__
    signature=inspect.signature(original)
    @wraps(original)
    def construct(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs)
        bound.arguments['drawer_only']=True
        original(*bound.args,**bound.kwargs)
    factory.__init__=construct


def apply_sink_front_margin(graph, margin, *, cl, cu, Semantics, Subpart, shelves, table_decorations):
    """Same pinned sink predicate, with an explicit fixture inset parameter."""
    import numpy as np
    if not np.isfinite(margin) or not .05<=margin<=.25:
        raise ValueError('SOURCE_SINK_FRONT_MARGIN_OUTSIDE_SUPPORTED_RANGE')
    rooms=cl.scene()[{Semantics.Room,-Semantics.Object}]
    objects=cl.scene()[{Semantics.Object,-Semantics.Room}]
    # Use the exact domain construction in the pinned home constraint.
    counters=objects[Semantics.Furniture].related_to(rooms,cu.on_floor)[Semantics.KitchenCounter]
    wall=counters[shelves.KitchenSpaceFactory].related_to(rooms,cu.against_wall)
    island=counters[shelves.KitchenIslandFactory]
    flush=cl.StableAgainst(cu.bottom,{Subpart.SupportSurface},margin=.001)
    front=cl.CoPlanar(cu.front,cu.front,margin=float(margin),rev_normal=True)
    sinks=objects[Semantics.Sink][table_decorations.SinkFactory].related_to(counters,flush).related_to(counters,front)
    kitchens=rooms[Semantics.Kitchen].excludes(cu.room_types)
    graph.constraints['kitchen_sink']=kitchens.all(lambda room:
        sinks.related_to(wall.related_to(room)).count().in_range(0,1)*
        sinks.related_to(island.related_to(room)).count().in_range(0,1))
    graph.constraints['findingbench_complete_kitchen_sink']=kitchens.all(lambda room:
        sinks.related_to(counters.related_to(room)).count().equals(1))
    return graph


def refine_populated_storage(solver, problem, filter_domain, var_assignments, *, solve_objects, n_steps=200):
    """Repair expanded cabinet meshes before their contents are generated.

    This uses the pinned solver's pose moves only. Population can change a
    placeholder's envelope; its previously cached optimizer result cannot
    certify the expanded mesh. Fresh evaluation and the final full-room gate
    remain mandatory. This is proposal repair, not physical acceptance.
    """
    import bpy
    import numpy as np
    from infinigen.core.constraints import constraint_language as cl
    from infinigen.core.constraints.evaluator.evaluate import evaluate_problem
    from infinigen.core.constraints.example_solver import greedy
    from infinigen.core.tags import FromGenerator
    from infinigen.assets.objects import shelves

    if not isinstance(n_steps, int) or not 1 <= n_steps <= 500:
        raise ValueError('INVALID_POPULATED_STORAGE_REFINEMENT_BUDGET')
    state = solver.state
    bpy.context.view_layer.update()
    tag = FromGenerator(shelves.KitchenCabinetFactory)
    candidates = {key for key, obj in state.objs.items() if tag in obj.tags}
    if not candidates:
        raise ValueError('POPULATED_STORAGE_CANDIDATES_MISSING')
    identities = {key: (obj.obj.name, id(obj.generator)) for key, obj in state.objs.items()}
    stationary = {key: np.array(obj.obj.matrix_world) for key, obj in state.objs.items()
                  if key not in candidates}
    required = {'storage', 'findingbench_storage_count', 'findingbench_cabinet_count',
                'findingbench_storage_separation', 'findingbench_complete_storage_count'}
    if not required <= problem.constraints.keys():
        raise ValueError('POPULATED_STORAGE_CONSTRAINT_SET_INCOMPLETE')
    checks = cl.Problem({key: value for key, value in problem.constraints.items() if key in required}, {})
    active = {key: obj.active for key, obj in state.objs.items()}

    def measure():
        greedy.update_active_flags(state, var_assignments)
        bpy.context.view_layer.update()
        result = evaluate_problem(checks, state, enable_loss=False)
        return {key: float(value) for key, value in result.violations.items()}

    def passed(values):
        return all(np.isfinite(value) and value == 0 for value in values.values())

    try:
        before = measure()
        attempts = 0
        if not passed(before):
            attempts = 1
            solve_objects(solver, problem, filter_domain.with_tags({tag}), var_assignments,
                          n_steps=n_steps, desc='findingbench_populated_storage_repair',
                          restrict_moves=['plane_change', 'reinit_pose', 'translate', 'rotate'],
                          abort_unsatisfied=False)
        after = measure()
        if identities != {key: (obj.obj.name, id(obj.generator)) for key, obj in state.objs.items()}:
            raise ValueError('POPULATED_STORAGE_REPAIR_CHANGED_OBJECT_INVENTORY')
        if any(not np.allclose(pose, np.array(state.objs[key].obj.matrix_world), atol=1e-7, rtol=0)
               for key, pose in stationary.items()):
            raise ValueError('POPULATED_STORAGE_REPAIR_MOVED_NON_CANDIDATE')
        return {'status': 'passed' if passed(after) else 'failed', 'attempts': attempts,
                'steps_per_attempt': n_steps, 'violations_before': before, 'violations_after': after,
                'scope': 'populated_storage_proposal_geometry', 'physical_acceptance': False}
    finally:
        for key, value in active.items():
            if key in state.objs:
                greedy.set_active(state, key, value)


def apply_optional_geometry_guards(graph, *, cl, cu, Semantics, shelves, elements):
    """Correct the declared domains of three pinned optional geometry rules.

    Pair separation has no pair to test for zero/one object. An optional
    island has no wall-counter/island pair when absent. Preserve the original
    predicates verbatim whenever these domains are nonempty; keep original
    expressions for a separately reported raw diagnostic, never erase them.
    """
    rooms=cl.scene()[{Semantics.Room,-Semantics.Object}]
    obj=cl.scene()[{Semantics.Object,-Semantics.Room}]
    rugs=obj[elements.RugFactory].related_to(rooms,cu.on_floor)
    lights=obj[Semantics.Lighting]
    original={name:graph.constraints[name] for name in ('rugs','lighting','kitchen_island_placement')}
    graph.constraints['rugs']=rooms.all(lambda r:(rugs.related_to(r).count()<=1)
        +(cl.min_distance_internal(rugs.related_to(r))>=1))
    graph.constraints['lighting']=rooms.all(lambda r:(lights.related_to(r).count()<=1)
        +(cl.min_distance_internal(lights.related_to(r))>=1))
    kitchens=rooms[Semantics.Kitchen].excludes(cu.room_types)
    island=obj[Semantics.Furniture][Semantics.KitchenCounter].related_to(rooms,cu.on_floor)[shelves.KitchenIslandFactory]
    wallcounter=(obj[Semantics.Furniture][Semantics.KitchenCounter].related_to(rooms,cu.on_floor)
                 [shelves.KitchenSpaceFactory].related_to(rooms,cu.against_wall))
    graph.constraints['kitchen_island_placement']=kitchens.all(lambda r:
        island.related_to(r).count().equals(0)
        +(wallcounter.related_to(r).all(lambda t:t.distance(island.related_to(r)).in_range(.7,3))
          *island.related_to(r).all(lambda t:t.distance(wallcounter.related_to(r)).in_range(.7,3)
                                    *(t.distance(r,cu.walltags)>2))))
    return original


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
    parser.add_argument('--kitchen-stage-budgets',action='store_true')
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
    from infinigen.core.tags import Semantics,FromGenerator,Subpart
    from infinigen.assets.objects import shelves, elements, table_decorations
    from infinigen.assets.objects.shelves.kitchen_cabinet import KitchenCabinetBaseFactory
    from rummagebench.authoring.tasks.infinigen_components import CabinetProvenance, complete_cabinet_census
    provenance = CabinetProvenance(KitchenCabinetBaseFactory).install()
    apply_source_storage_dimensions(shelves.KitchenCabinetFactory,
        req.get('source_storage_dimensions_m',[[.25,.35],[1.,4.],[.5,1.3]]))
    if 'source_sink_parameter_ranges_m' in req:
        apply_source_sink_parameters(table_decorations.SinkFactory,req['source_sink_parameter_ranges_m'])
    if req.get('source_kitchen_cabinets_closed_only') is True:
        apply_closed_kitchen_cabinets(shelves.KitchenCabinetFactory)
    original_optional={}
    original=home.home_furniture_constraints
    def with_search_constraints():
        graph=original()
        original_optional.update(apply_optional_geometry_guards(graph,cl=cl,cu=cu,Semantics=Semantics,
                                                               shelves=shelves,elements=elements))
        if 'source_sink_front_margin_m' in req:
            apply_sink_front_margin(graph,req['source_sink_front_margin_m'],cl=cl,cu=cu,Semantics=Semantics,
                Subpart=Subpart,shelves=shelves,table_decorations=table_decorations)
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
    def record_solve(self, consgraph, filter_domain, var_assignments, *solve_args, **kwargs):
        desc=kwargs.get('desc','')
        if desc.startswith('on_floor_and_wall_'):
            live.update(storage_domain=filter_domain, storage_assignments=dict(var_assignments))
        if args.kitchen_stage_budgets and desc.startswith('on_floor_and_wall_'):
            # Counters/oven establish the kitchen before cabinets consume
            # scarce wall space. Disjoint factory filters use the same
            # pinned solver, collision checks, room and constraint graph.
            cabinet=FromGenerator(shelves.KitchenCabinetFactory)
            first=dict(kwargs,n_steps=200,desc=desc+'_kitchen_fixtures')
            original_solve(self,consgraph,filter_domain.with_tags({-cabinet}),var_assignments,*solve_args,**first)
            second=dict(kwargs,n_steps=150,desc=desc+'_search_cabinets')
            result=original_solve(self,consgraph,filter_domain.with_tags({cabinet}),var_assignments,*solve_args,**second)
        else:
            if args.kitchen_stage_budgets and desc.startswith('on_floor_freestanding_'):
                kwargs=dict(kwargs,n_steps=40)
            result = original_solve(self, consgraph, filter_domain, var_assignments, *solve_args, **kwargs)
        live.update(solver=self, problem=consgraph)
        return result
    def final_gate(self, name, fn, *args, **kwargs):
        if name == 'populate_intermediate_pholders':
            result = original_stage(self, name, fn, *args, **kwargs)
            repair = refine_populated_storage(live['solver'], live['problem'], live['storage_domain'],
                                             live['storage_assignments'], solve_objects=original_solve)
            live['populated_storage_refinement'] = repair
            atomic_write_json(args_output / 'populated_storage_refinement.json', repair)
            if repair['status'] != 'passed':
                raise ValueError('POPULATED_STORAGE_REFINEMENT_UNSATISFIED')
            return result
        if name == 'populate_assets':
            state = live['solver'].state
            kitchens = [key for key,o in state.objs.items() if Semantics.Kitchen in o.tags]
            if len(kitchens) != 1:raise ValueError('SINGLE_KITCHEN_FINAL_CONSTRAINT_SCOPE_REQUIRED')
            active = {key:o.active for key,o in state.objs.items()}
            try:
                greedy.update_active_flags(state,{cu.variable_room:kitchens[0]})
                result = evaluate_problem(live['problem'],state,enable_loss=False)
                violations = {key:float(value) for key,value in result.violations.items()}
                raw_optional=evaluate_problem(cl.Problem(constraints=original_optional,score_terms={}),state,enable_loss=False)
                live['constraints'] = {'scope':'pinned_kitchen_constraints_with_declared_optional_domain_corrections',
                    'violations':violations, 'passed':all(np.isfinite(v) and v == 0 for v in violations.values()),
                    'raw_optional_geometry_violations':{key:float(value) for key,value in raw_optional.violations.items()},
                    'domain_corrections':{'rugs':'pair spacing applies when count >= 2',
                        'lighting':'pair spacing applies when count >= 2',
                        'kitchen_island_placement':'original predicate applies when an island exists'},
                    'physical_acceptance':False}
                live['constraints']['source_fixture_parameters']={k:req[k] for k in
                    ('source_sink_parameter_ranges_m','source_sink_front_margin_m') if k in req}
            finally:
                for key,value in active.items():greedy.set_active(state,key,value)
        return original_stage(self,name,fn,*args,**kwargs)
    import numpy as np
    args_output = args.output_folder
    Solver.solve_objects = record_solve
    RandomStageExecutor.run_stage = final_gate
    from infinigen_examples.generate_indoors import main
    try:
        main(SimpleNamespace(seed=args.seed,configs=args.configs,overrides=args.overrides,input_folder=None,
                             output_folder=args.output_folder,task=args.task,task_uniqname=None))
    finally:
        provenance.restore()
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
        'cabinet_component_census':complete_cabinet_census(raw, provenance),
        'source_sink_parameters_m':{key:{field:float(obj.generator.params[field]) for field in ('Width','Margin')}
            for key,obj in live['solver'].state.objs.items()
            if FromGenerator(table_decorations.SinkFactory) in obj.tags and obj.generator is not None},
        'source_storage_factory_dimensions_m':{key:list(obj.generator.dimensions) for key,obj in live['solver'].state.objs.items()
            if FromGenerator(shelves.KitchenCabinetFactory) in obj.tags and obj.generator is not None},
        'solve_state_sha256':sha256_path(state_path),'source':'actual_Blender_evaluated_mesh_AABBs',
        'upstream_constraints':live.get('constraints',{'passed':False,'reason':'final gate not executed'}),
        'populated_storage_refinement':live.get('populated_storage_refinement'),
        'gpu_started':False,'physical_acceptance':'pending'})


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'export-upstream':
        raise SystemExit(_run_export_bridge(sys.argv[2:]))
    _run_upstream_bridge()
