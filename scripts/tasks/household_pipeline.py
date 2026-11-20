"""One household pipeline entry: CPU audit, or explicit GPU native capture.

Pinned Infinigen proposals and geometry use CPU stages. Native capture and
restore are explicitly GPU-only stages. No stage promotes oracle certificates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import REPO, host_config

from rummagebench.authoring.level1.robot_inventory import discover_installed_robots
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path

MISSING_BASELINE = (
    "src/rummagebench/authoring/tasks/difficulty.py",
    "src/rummagebench/authoring/tasks/families.py",
    "src/rummagebench/authoring/tasks/family_binding.py",
    "src/rummagebench/authoring/tasks/household.py",
    "scripts/tasks/generate_families.py",
    "assets/tasks/household_families_v1.yaml",
    "assets/tasks/household_background_v1.yaml",
    "assets/tasks/household_robots_v1.yaml",
    "docs/household_task_families.md",
    "docs/household_background_pipeline.md",
    "tests/unit/test_household_background.py",
    "tests/unit/test_household_families.py",
)


def write_report(path: Path | None, report: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: reruns may reuse identical evidence, never
        # overwrite a conflicting run or a partial previous write.
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(canonical_json(report).decode())
        except FileExistsError:
            if json.loads(path.read_text()) != report:
                raise ValueError(f"conflicting report: {path}; use a new output")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def preflight_child_result(returncode, output):
    """SDK shutdown may exit zero while unwinding a failed worker."""
    if returncode != 0:
        return returncode
    try:
        report=json.loads((Path(output)/'preflight.json').read_text())
        from rummagebench.sim.omnigibson.observation import head_rgb_metadata_matches
        required={'native_inventory','native_restore_twice','camera_rgb_shape',
                  'loaded_articulation_query','spawn_clearance'}
        if (report.get('status')!='pending_simulation_acceptance' or report.get('failures')
                or report.get('error') or report.get('native_restore_repeats')!=2
                or not required<=set(report.get('completed_checks',[]))
                or not head_rgb_metadata_matches(report.get('camera'))):
            return 2
    except (OSError,ValueError,TypeError):
        return 2
    return 0


def probe_child_result(returncode, output):
    """A clean SDK exit alone cannot attest a complete embodiment probe."""
    if returncode!=0:return returncode
    try:
        from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
        from rummagebench.authoring.tasks.infinigen_probe import verify_overlay_evidence
        overlay=SlotEmbodimentOverlay.load(Path(output)/'overlay.yaml')
        if Path(overlay.source).resolve()!=(Path(output)/'evidence.json').resolve():return 2
        verify_overlay_evidence(overlay)
    except (OSError,ValueError,TypeError,KeyError):
        return 2
    return 0


def cpu_audit(host: dict, robot_assets: Path) -> dict:
    missing = [p for p in MISSING_BASELINE if not (REPO / p).is_file()]
    robots = discover_installed_robots(robot_assets)
    for name, profile in robots.items():
        profile["integration_status"] = ("configured" if name == "r1pro" and
            (REPO / "configs/tasks/robots/r1pro.yaml").is_file() and profile["status"] == "candidate" else
            "integration_pending" if profile["status"] == "candidate" else "unavailable")
    # Count only state bundles actually emitted by this pipeline, never
    # configurations, seeds, historical certificates or .pt files alone.
    measured_native = 0
    for path in (host["build_root"] / "household" / "native").glob("*/inventory.json"):
        try:
            doc = json.loads(path.read_text())
            snapshot = path.parent / "native_state.pt"
            payload = {k: v for k, v in doc.items() if k != "inventory_sha256"}
            measured_native += int(doc.get("status") == "measured_native_occupancy"
                                   and snapshot.is_file()
                                   and sha256_bytes(canonical_json(payload)) == doc.get("inventory_sha256")
                                   and sha256_path(snapshot) == doc.get("snapshot_sha256"))
        except (OSError, ValueError):
            continue
    return {"stage": "cpu_audit", "status": "pending_simulation",
            "missing_baseline_files": missing,
            "failure_reasons": {}, "baseline_note": "historical cloud sources absent; current Infinigen implementation is independent",
            "counts": {"native_state_bundles": measured_native,
                       "generated_family_candidates_this_run": 0,
                       "simulation_validated_families_this_run": 0},
            "robots": robots, "camera_protocol": "current_code_head_rgb; wrist_only_not_certified",
            "stages": {"native_occupancy": "pending_gpu" if not measured_native else "measured",
                       "frozen_background": "pending", "family_binding": "pending",
                       "appearance_application_and_images": "pending_gpu",
                       "robot_scene_skills": "pending_gpu", "search_cost_measurement": "pending_gpu"}}


def native_capture(args) -> dict:
    # Imports capable of loading simulator dependencies stay behind the
    # explicit GPU subcommand. --help and CPU audit never import OG.
    from rummagebench.authoring.tasks.native_occupancy import capture_native_occupancy
    from rummagebench.authoring.tasks.plan import sha256_bytes
    from rummagebench.authoring.tasks.slots import SceneSlots
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.state_io import save_snapshot

    slots = SceneSlots.load(args.slots)
    scenario = load_scenario(args.scenario)
    if scenario.scene.model != slots.scene:
        raise ValueError("scenario and SceneSlots refer to different scenes")
    # Reserve a fresh bundle before loading; incomplete bundles have no
    # inventory commit marker and are never counted or resumed as accepted.
    args.out.mkdir(parents=True, exist_ok=False)
    backend = OmniGibsonBackend(seed=args.seed)
    result = {}

    def capture(b):
        doc, state = capture_native_occupancy(b, slots, source_scene_path=args.source_scene)
        snapshot = save_snapshot(args.out / "native_state.pt", state,
                                 {"source_state_sha256": doc["source_state_sha256"]})
        doc["snapshot_sha256"] = sha256_path(snapshot)
        doc["scenario_sha256"] = sha256_path(args.scenario)
        # Rehash after binding the on-disk restore artifact and invocation.
        doc.pop("inventory_sha256")
        doc["inventory_sha256"] = sha256_bytes(canonical_json(doc))
        result.update(doc)
        return {"inventory_sha256": doc["inventory_sha256"], "counts": doc["counts"]}

    try:
        backend.setup(scenario, native_scene_capture=capture, native_scene_only=True)
        atomic_write_json(args.out / "inventory.json", result)
        return {"stage": "native_capture", "status": "pending_simulation_acceptance",
                "output": str(args.out), "counts": result["counts"],
                "generated_family_candidates_this_run": 0, "validated_families_this_run": 0}
    except Exception as exc:
        atomic_write_json(args.out / "failure.json", {"reason": type(exc).__name__, "detail": str(exc)})
        raise
    finally:
        backend.close()


def verify_native_restore(args) -> dict:
    """GPU check for a locally exported state, without target import or stepping."""
    from rummagebench.authoring.tasks.native_occupancy import require_known_slots, verify_source
    from rummagebench.authoring.tasks.slots import SceneSlots
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.state_io import load_snapshot

    doc = json.loads((args.bundle / "inventory.json").read_text())
    snapshot = args.bundle / "native_state.pt"
    if sha256_path(snapshot) != doc["snapshot_sha256"]:
        raise ValueError("native snapshot file hash mismatch")
    if sha256_path(args.scenario) != doc["scenario_sha256"]:
        raise ValueError("native invocation scenario hash mismatch")
    slots = SceneSlots.load(args.slots)
    # This is a trusted LOCAL snapshot emitted by export-native, never an
    # untrusted downloaded pickle or a redistributed BEHAVIOR asset.
    state, meta = load_snapshot(snapshot)
    if meta.get("source_state_sha256") != doc["source_state_sha256"]:
        raise ValueError("native snapshot metadata mismatch")
    backend = OmniGibsonBackend(seed=args.seed)

    def verify(b):
        for _ in range(2):
            b.restore_observe_state({"sim": state, "commanded_pose": None})
            verify_source(doc, b, slots)
        require_known_slots(doc, args.require_slots)
        return {"source_matches": True, "restore_repeats": 2}

    try:
        backend.setup(load_scenario(args.scenario), native_scene_capture=verify, native_scene_only=True)
        return {"stage": "verify_native_restore", "status": "native_restore_checked",
                "restore_repeats": 2, "validated_families_this_run": 0,
                "physical_acceptance": "pending", "required_slots": args.require_slots}
    finally:
        backend.close()


def load_household_spec(path: Path, policy: Path):
    from rummagebench.authoring.tasks.household import HouseholdTaskSpec, validate_spec_assets
    spec = HouseholdTaskSpec.model_validate_json(path.read_text())
    validate_spec_assets(spec, REPO, policy)
    return spec


def plan_infinigen(args) -> dict:
    import yaml
    from rummagebench.authoring.tasks.household import sample_spec, family_members, grouped_split, validate_spec_assets
    from rummagebench.authoring.tasks.infinigen import generation_command
    if not 1 <= args.count <= 10000 or args.start_index < 0:
        raise ValueError("count must be 1..10000 and start-index nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    all_specs, requests = [], []
    appearance_profile=getattr(args,'appearance_profile','requested')
    appearance_config=None
    if appearance_profile=='lighting-only':
        # Explicit minimal pilot scope. Preserve source textures and colours;
        # never silently drop material requests from an existing task spec.
        policy=yaml.safe_load(args.policy.read_text())
        appearance_config=yaml.safe_load((REPO/policy['appearance_config']).read_text())
        appearance_config['randomize'].update(wall_color_palette=[],support_texture_swap=False,container_texture_swap=False)
    for index in range(args.start_index, args.start_index + args.count):
        base = sample_spec(args.policy, REPO, args.seed, index, split_seed=args.split_seed,
                           test_fraction=args.test_fraction, difficulty_profile=args.difficulty_profile,
                           appearance_config=appearance_config)
        validate_spec_assets(base, REPO, args.policy)
        members = family_members(base)
        for spec in members:
            path = args.out / (spec.task_id + ".json")
            if path.exists():
                if path.read_bytes() != spec.canonical_bytes():
                    raise ValueError(f"SPEC_OUTPUT_CONFLICT:{path}")
            else:
                with path.open("xb") as f:
                    f.write(spec.canonical_bytes())
        all_specs.extend(members)
        requests.append({"family_id": base.family_id, "specs": [s.task_id + ".json" for s in members],
                         "base_layout_id": base.base_layout_id,
                         "generation": generation_command(base, args.python, args.upstream_root,
                                                          args.out / base.base_layout_id / "geometry"),
                         "export": {"execution_status": "pending_successful_generation_receipt",
                                    "reason": "input folder must come from the successful attempt"}})
    report = {"version": 1, "stage": "plan_infinigen", "status": "candidate",
              "seed": args.seed, "start_index": args.start_index,
              "counts": {"task_specs": len(all_specs), "families": args.count,
              "geometry_scenes": 0, "exported_scene_files": 0, "evaluation_ready_episodes": 0},
              "split": grouped_split(all_specs, split_seed=args.split_seed, test_fraction=args.test_fraction),
              "split_seed": args.split_seed, "test_fraction": args.test_fraction,
              "split_strategy_version": "base-layout-sha256-v1",
              "split_scope": "layout_grouped_dev_test; not a claim of asset or family generalization",
              "requests": requests, "failure_reasons": {}, "gpu_started": False,
              "appearance_status": "requested_pending_application_and_images",
              "appearance_profile":appearance_profile,
              "robots": {"r1pro": "configured_pending_scene_skill_validation"},
              "camera_protocol": "current_head_rgb; no wrist-only certification",
              "difficulty_status": "requested_only; actual geometry and search not measured"}
    write_report(args.out / "batch.json", report)
    return report


def cpu_generate(args) -> dict:
    from rummagebench.authoring.tasks.infinigen import run_cpu_generation
    spec = load_household_spec(args.spec, args.policy)
    return run_cpu_generation(spec, args.upstream_root, args.python, args.out,
                              timeout_s=args.timeout, retries=args.retries, solver_profile=args.solver_profile)


def batch_cpu_generate(args) -> dict:
    """Serial, bounded use of the existing runner, once per complete family."""
    import fcntl
    from collections import Counter
    from rummagebench.authoring.tasks.household import family_members
    from rummagebench.authoring.tasks.infinigen import run_cpu_generation
    from rummagebench.authoring.tasks.household_dataset import pipeline_inventory
    batch = json.loads(args.batch.read_text())
    requests = batch["requests"]
    if (batch.get("stage") != "plan_infinigen" or not requests or len(requests) > 10000
            or len({r["family_id"] for r in requests}) != len(requests)):
        raise ValueError("COMPLETE_UNIQUE_PLANNED_BATCH_REQUIRED")
    specifications = []
    for request in requests:
        members = []
        for ref in request["specs"]:
            path = (args.batch.parent / ref).resolve()
            if path.parent != args.batch.parent.resolve() or path.suffix != ".json":
                raise ValueError("BATCH_SPECIFICATION_PATH_INVALID")
            members.append(load_household_spec(path, args.policy))
        if (not members or request["family_id"] != members[0].family_id
                or {s.task_id:s for s in members} != {s.task_id:s for s in family_members(members[0])}
                or len(members) != len(members[0].locations)):
            raise ValueError("BATCH_INCOMPLETE_MATCHED_FAMILY")
        specifications.append(min(members, key=lambda s:s.task_id))
    args.out.mkdir(parents=True, exist_ok=True)
    identity = {"batch": sha256_path(args.batch), "policy": sha256_path(args.policy),
                "upstream_root": str(args.upstream_root.resolve()), "python": args.python,
                "solver_profile": args.solver_profile, "timeout": args.timeout, "retries": args.retries,
                "specifications": {s.task_id:sha256_bytes(s.canonical_bytes()) for s in specifications}}
    with (args.out / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("BATCH_ALREADY_ACTIVE") from exc
        input_path = args.out / "batch_input.json"
        if input_path.exists():
            if json.loads(input_path.read_text()) != identity:
                raise ValueError("BATCH_RESUME_INPUT_CONFLICT")
        else:
            atomic_write_json(input_path, identity)
        failures, results = Counter(), []
        for spec in specifications:
            try:
                receipt = run_cpu_generation(spec, args.upstream_root, args.python,
                    args.out / spec.family_id / "run", timeout_s=args.timeout,
                    retries=args.retries, solver_profile=args.solver_profile)
                failures.update(receipt["failure_reasons"])
                results.append({"family_id":spec.family_id,"status":receipt["status"],
                                "receipt":str((args.out / spec.family_id / "run/generation.json").resolve())})
            except (OSError, ValueError, RuntimeError) as exc:
                reason = f"{type(exc).__name__}:{exc}"
                failures[reason] += 1
                results.append({"family_id":spec.family_id,"status":"failed","reason":reason})
            report = {"version":1,"stage":"batch_infinigen_cpu","status":"pending_simulation",
                      "planned_families":len(specifications),"completed_families":len(results),
                      "failure_reasons":dict(failures),"results":results,
                      "counts":pipeline_inventory(args.out)["counts"],
                      "gpu_started":False,"execution":"serial; two CPU cores per upstream process"}
            atomic_write_json(args.out / "batch_progress.json", report)
        return report


def prepare_infinigen(args) -> dict:
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly, scene_slots_from_assembly
    spec = load_household_spec(args.spec, args.policy)
    try:
        binding_doc = json.loads(args.bindings.read_text())
        if isinstance(binding_doc, dict):
            if binding_doc.get("status") != "candidate" or binding_doc.get("failures"):
                raise ValueError("INTERACTIVE_BINDING_PROPOSAL_REJECTED")
            for path, digest in {**binding_doc["source_hashes"], **binding_doc["asset_hashes"]}.items():
                if sha256_path(path) != digest:
                    raise ValueError("INTERACTIVE_BINDING_INPUT_CHANGED")
            binding_doc = binding_doc["bindings"]
        doc = prepare_assembly(spec, args.scene, args.solve_state, binding_doc,
                               args.assets_root, args.out, generation_receipt=args.generation_receipt)
    except Exception as exc:
        # Keep partial preparation evidence; absence of assembly.json is the
        # commit marker distinguishing a failed bundle from a candidate.
        if not (args.out / "assembly.json").exists():
            args.out.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.out / "failure.json", {"status": "assembly_failed", "family": spec.family,
                "requested_difficulty": spec.requested_difficulty, "reason": f"{type(exc).__name__}: {exc}",
                "evaluation_ready_episodes": 0})
        raise
    slots = scene_slots_from_assembly(args.out / "assembly.json")
    slots.save(args.out / "scene_slots.json")
    return {"stage": "prepare_infinigen", "status": doc["status"],
            "assembly": str(args.out / "assembly.json"), "slots": str(args.out / "scene_slots.json"),
            "counts": {"objects": len(doc["objects"]), "required_candidates": len(doc["candidate_bindings"]),
                       "evaluation_ready_episodes": 0}, "pending_checks": doc["pending_checks"]}


def bind_infinigen(args) -> dict:
    from rummagebench.authoring.tasks.household import family_members
    from rummagebench.authoring.tasks.household_binding import bind_family
    spec = load_household_spec(args.spec, args.policy)
    return bind_family(family_members(spec), args.assembly, args.overlay, args.out)



def gpu_infinigen_preflight(args) -> dict:
    """GPU ONLY, never invoked by CPU commands. Partial checks cannot release data."""
    import yaml
    import numpy as np
    from copy import deepcopy
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.authoring.tasks.native_occupancy import capture_native_occupancy, verify_source
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly, runtime_articulation_checks
    from rummagebench.sim.omnigibson.state_io import save_snapshot

    spec = load_household_spec(args.spec, args.policy)
    manifest = load_assembly_manifest(args.assembly)
    expected_family = spec.family_id
    if getattr(args, 'component_preparation', None):
        from rummagebench.authoring.tasks.infinigen_component_binding import verify_component_preparation
        _, preparation = verify_component_preparation(args.component_preparation)
        if (preparation['family_proposal']['supersedes_family_id'] != spec.family_id
                or manifest.get('component_context',{}).get('preparation_sha256') != sha256_path(args.component_preparation)):
            raise ValueError('COMPONENT_IMPORT_PREFLIGHT_SOURCE_MISMATCH')
        expected_family = preparation['family_proposal']['family_id']
    if spec.base_layout_id != manifest["scene_id"] or expected_family != manifest["family_id"]:
        raise ValueError("ASSEMBLY_SPEC_MISMATCH")
    slots = scene_slots_from_assembly(args.assembly)
    robot_doc = yaml.safe_load((REPO / spec.robot_config_ref).read_text())
    start = json.loads(args.start.read_text())
    scenario = ScenarioSpec.model_validate({"id": "preflight_" + spec.base_layout_id,
        "instruction": "evaluator-only assembly preflight", "scene": {"model": spec.base_layout_id,
            "source": "infinigen", "assembly_manifest": str(args.assembly.resolve())},
        "robot": {**robot_doc["robot"], "init_anchor": "start"}, "feasibility": robot_doc["feasibility"],
        "anchors": {"start": start}, "target": {"entity": spec.target.internal_entity, "category": spec.target.category},
        # The existing ScenarioSpec requires a declared real target. Native
        # setup returns before task import, so it is not placed or captured.
        "objects": [{"name":spec.target.internal_entity,"category":spec.target.category,"model":spec.target.model}],
        "placements": [], "skills": ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"]})
    args.out.mkdir(parents=True, exist_ok=False)
    backend = OmniGibsonBackend(seed=spec.seed)
    evidence = {"stage": "infinigen_gpu_preflight", "status": "pending_simulation_acceptance",
                "assembly_manifest_sha256": sha256_path(args.assembly), "evaluation_ready_episodes": 0,
                "pending_checks": list(manifest["pending_checks"]), "failures": []}
    if manifest.get('component_context'):
        evidence['scope'] = 'real_component_asset_import_only; complete_leaf_binding_pending'
        evidence['interactive_asset_bodies'] = len(manifest['candidate_bindings'])
        evidence['complete_search_locations'] = manifest['component_context']['complete_search_distribution']['candidate_count']
    def progress(phase):
        evidence['native_phase'] = phase
        atomic_write_json(args.out/'progress.json',{'stage':'infinigen_gpu_preflight',
            'status':'running_pending_acceptance','phase':phase,
            'assembly_manifest_sha256':evidence['assembly_manifest_sha256'],
            'evaluation_ready_episodes':0})
        print('Native import preflight: '+phase,flush=True)
    try:
        # Callback intentionally captures no task entities; this is the native
        # occupancy bundle. It is NOT yet a topology-compatible family snapshot.
        def capture(b):
            progress('loaded_before_initial_close')
            b.validate_physics_state()
            if manifest.get('component_context'):
                from rummagebench.authoring.tasks.infinigen_component_binding import inspect_loaded_component_topology, propose_loaded_search_distribution
                topology=inspect_loaded_component_topology(b,manifest)
                topology['state_phase']='initial_before_close; diagnostic_only'
                atomic_write_json(args.out/'initial_component_topology.json',topology)
                atomic_write_json(args.out/'initial_native_search_proposal.json',propose_loaded_search_distribution(manifest,topology))
            if getattr(args,'diagnose_entity',None):
                from rummagebench.authoring.tasks.household_physics import diagnose_nearby_contacts
                atomic_write_json(args.out/'initial_contacts.json',diagnose_nearby_contacts(b,args.diagnose_entity))
            for item in manifest["candidate_bindings"]:
                progress('initial_close:' + item['name'])
                if not b.set_open(item["name"], False):
                    raise ValueError(f"INITIAL_CLOSE_FAILED:{item['name']}")
                b.validate_physics_state()
            progress('settling_closed_background')
            b.settle(30); b.validate_physics_state(); b._validate_spawn_clearance()
            from rummagebench.authoring.tasks.native_occupancy import state_hash
            restore_contract=getattr(args,'restore_contract','serializer-identity')
            source_checkpoint=b.dump_state()
            if restore_contract=='same-checkpoint':
                # A simulator checkpoint is a reconstruction input. Capture
                # the actual reconstructed world B, then require repeated
                # loads of the SAME input A to reproduce B exactly. Never
                # replace A with a freshly dumped B on the next repetition.
                progress('establish_observed_checkpoint_state')
                b.restore_observe_state({'sim':deepcopy(source_checkpoint),'commanded_pose':None})
                b.validate_physics_state();b._validate_spawn_clearance()
            if manifest.get('component_context'):
                topology=inspect_loaded_component_topology(b,manifest)
                topology['state_phase']='closed_background_after_settling'
                atomic_write_json(args.out/'loaded_component_topology.json',topology)
                atomic_write_json(args.out/'loaded_native_search_proposal.json',propose_loaded_search_distribution(manifest,topology))
                evidence['native_region_artifacts']={name:sha256_path(args.out/name) for name in
                    ('loaded_component_topology.json','loaded_native_search_proposal.json')}
            evidence["failures"] += runtime_articulation_checks(b, manifest)
            evidence['interaction_interfaces']=getattr(b,'_findingbench_interaction_interfaces',{})
            progress('native_occupancy')
            doc, state = capture_native_occupancy(b, slots, source_scene_path=args.assembly)
            restore_input=source_checkpoint if restore_contract=='same-checkpoint' else state
            recipe={'version':1,'contract':restore_contract,
                    'checkpoint_input_state_sha256':state_hash(restore_input),
                    'observed_world_state_sha256':doc['source_state_sha256'],
                    'spatial_refresh':'finite_state_and_existing_spawn_collision_check; no physics steps',
                    'comparison':'exact full state, object state and geometry; no numeric tolerance'}
            evidence['native_restore_contract']=recipe
            evidence['native_restore_observed_hashes']=[doc['source_state_sha256']]
            # Retain failed restore evidence too. These diagnostic names are
            # distinct from the final twice-verified native snapshot below.
            atomic_write_json(args.out/'captured_inventory.json',doc)
            save_snapshot(args.out/'captured_native_state.pt',state,
                          {'source_state_sha256':doc['source_state_sha256'],'restore_validation':'pending'})
            save_snapshot(args.out/'restore_input_state.pt',restore_input,recipe)
            for repeat in range(2):
                progress('restore_native_repeat_'+str(repeat+1))
                if state_hash(restore_input)!=recipe['checkpoint_input_state_sha256']:
                    raise ValueError('NATIVE_CHECKPOINT_INPUT_MUTATED')
                b.restore_observe_state({"sim": deepcopy(restore_input), "commanded_pose": None})
                if restore_contract=='same-checkpoint':
                    b.validate_physics_state();b._validate_spawn_clearance()
                try:
                    verify_source(doc, b, slots)
                    evidence['native_restore_observed_hashes'].append(state_hash(b.dump_state()))
                except Exception:
                    from rummagebench.authoring.tasks.native_occupancy import plain, state_hash
                    restored=b.dump_state()
                    save_snapshot(args.out/'failed_restored_native_state.pt',restored,
                                  {'restore_validation':'failed; diagnostic_only'})
                    before=np.asarray(plain(state));after=np.asarray(plain(restored))
                    delta={'repeat':repeat+1,'before_sha256':state_hash(state),
                           'after_sha256':state_hash(b.dump_state()),'before_shape':list(before.shape),
                           'after_shape':list(after.shape),'changed_native_objects':[]}
                    if before.shape==after.shape:
                        indices=np.flatnonzero(before!=after)
                        delta.update(changed_scalar_count=len(indices),
                                     max_absolute_change=float(np.max(np.abs(before-after))))
                        delta['first_differences']=[{'index':int(i),'before':float(before.flat[i]),
                            'after':float(after.flat[i])} for i in indices[:32]]
                        if before.dtype==after.dtype:
                            bits=(before.reshape(-1).view(np.uint8)!=after.reshape(-1).view(np.uint8))
                            bit_indices=np.flatnonzero(bits.reshape(before.size,before.dtype.itemsize).any(1))
                            delta['bitwise_changed_scalar_count']=len(bit_indices)
                            delta['first_bitwise_differences']=[{'index':int(i),'before':float(before.flat[i]),
                                'after':float(after.flat[i])} for i in bit_indices[:32]]
                    for obj in b._env.scene.objects:
                        if obj.name in doc['objects']:
                            previous=doc['objects'][obj.name]['state']
                            actual=plain(obj.dump_state(serialized=False))
                            if previous!=actual:
                                delta['changed_native_objects'].append({'entity':obj.name,
                                    'before':previous,'after':actual})
                    atomic_write_json(args.out/'restore_failure.json',delta)
                    rounds=[]
                    for index in range(getattr(args,'diagnose_restore_roundtrips',0)):
                        previous=b.dump_state()
                        b.restore_observe_state({'sim':previous,'commanded_pose':None})
                        b.validate_physics_state()
                        actual=b.dump_state()
                        left,right=np.asarray(plain(previous)),np.asarray(plain(actual))
                        row={'repeat':index+1,'before_sha256':state_hash(previous),
                             'after_sha256':state_hash(actual)}
                        if left.shape==right.shape:
                            changed=np.flatnonzero(left!=right)
                            row.update(changed_scalar_count=len(changed),
                                max_absolute_change=float(np.max(np.abs(left-right))),
                                first_differences=[{'index':int(i),'before':float(left.flat[i]),
                                                   'after':float(right.flat[i])} for i in changed[:32]])
                        rounds.append(row)
                        atomic_write_json(args.out/'restore_roundtrips.json',{'scope':'diagnostic_only_no_acceptance',
                            'rounds':rounds,'evaluation_ready_episodes':0})
                    raise
            snapshot = args.out / ('native_checkpoint.pt' if restore_contract=='same-checkpoint' else 'native_state.pt')
            save_snapshot(snapshot, restore_input, {"source_state_sha256": doc["source_state_sha256"],
                                                   'native_restore_contract':recipe})
            doc["snapshot_sha256"] = sha256_path(snapshot)
            if restore_contract=='same-checkpoint':
                doc['version']=2
                doc['snapshot_file']=snapshot.name
                doc['native_restore_contract']=recipe
            doc.pop("inventory_sha256")
            doc["inventory_sha256"] = sha256_bytes(canonical_json(doc))
            atomic_write_json(args.out / "inventory.json", doc)
            evidence["native_occupancy_counts"] = doc["counts"]
            evidence["native_restore_repeats"] = 2
            return {"inventory_sha256": doc["inventory_sha256"]}
        progress('creating_environment')
        backend.setup(scenario, native_scene_capture=capture, native_scene_only=True)
        progress('camera_rgb')
        evidence['source_collision_import'] = getattr(backend._env,'_findingbench_source_collider_import',None)
        image = backend.get_observation()
        if (image.shape != (scenario.robot.image_height,scenario.robot.image_width,3)
                or image.dtype != np.uint8):
            raise ValueError("CAMERA_RGB_CONTRACT_FAILED")
        backend.render_snapshot(str(args.out / "initial_rgb.png"))
        from rummagebench.sim.omnigibson.observation import select_head_rgb_sensor
        observations,_=backend._env.get_obs()
        sensor_name,_=select_head_rgb_sensor(observations[0].get(scenario.robot.name,{}))
        camera = backend._camera_prim(sensor_name)
        if camera is None:raise ValueError('SELECTED_RGB_CAMERA_PRIM_UNAVAILABLE')
        evidence["camera"] = {"prim": str(camera.GetPath()), "shape": list(image.shape),
                              'sensor_name':sensor_name,
                              "protocol": "head_rgb_current_code; wrist_only_unverified"}
        from rummagebench.sim.omnigibson.observation import head_rgb_metadata_matches
        if not head_rgb_metadata_matches(evidence['camera']):raise ValueError('HEAD_RGB_CAMERA_IDENTITY_MISMATCH')
        evidence["appearance"] = getattr(backend._env, "_findingbench_appearance_evidence", {"status": "unavailable"})
        evidence["completed_checks"] = ["native_inventory", "native_restore_twice", "camera_rgb_shape",
                                        "loaded_articulation_query", "spawn_clearance"]
        # All checks requiring inserted targets, images, skills or reference
        # search remain pending even if this limited preflight passes.
        if evidence["failures"]:
            evidence["status"] = "rejected_preflight"
        atomic_write_json(args.out / "preflight.json", evidence)
        return evidence
    except Exception as exc:
        evidence.update(status="failed_preflight", error=f"{type(exc).__name__}: {exc}")
        evidence['physics_failure'] = getattr(backend,'_last_physics_failure',None)
        evidence['spawn_failure'] = getattr(backend,'_last_spawn_failure',None)
        atomic_write_json(args.out / "preflight.json", evidence)
        raise
    finally:
        backend.close()


def gpu_freeze_infinigen(args) -> dict:
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.infinigen_adapter import freeze_family_background
    family = json.loads((args.family_root / "family.json").read_text())
    from rummagebench.authoring.tasks.household_binding import validate_family_index
    validate_family_index(family)
    if family.get("status") != "pending_simulation":
        raise ValueError("FAMILY_NOT_PENDING_ASSEMBLY_CHECK")
    scenarios = [load_scenario(args.family_root / "candidates" / f"{tid}.yaml") for tid in family["task_ids"]]
    if sha256_path(args.assembly) != family["assembly_manifest_sha256"]:
        raise ValueError("FAMILY_ASSEMBLY_CHANGED")
    backend = OmniGibsonBackend(seed=0)
    try:
        return freeze_family_background(backend, scenarios, args.assembly, args.out)
    finally:
        backend.close()

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    cpu = sub.add_parser("cpu", help="static audit only; no simulator imports")
    cpu.add_argument("--host-config")
    cpu.add_argument("--robot-assets", type=Path)
    cpu.add_argument("--report", type=Path)
    gpu = sub.add_parser("export-native", help="GPU: capture builder-native state before task insertion")
    gpu.add_argument("--scenario", type=Path, required=True)
    gpu.add_argument("--slots", type=Path, required=True)
    gpu.add_argument("--source-scene", type=Path, required=True)
    gpu.add_argument("--out", type=Path, required=True)
    gpu.add_argument("--seed", type=int, default=0)
    restore = sub.add_parser("verify-native", help="GPU: restore an exported local native state twice")
    restore.add_argument("--scenario", type=Path, required=True)
    restore.add_argument("--slots", type=Path, required=True)
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--require-slots", nargs="*", default=[])
    restore.add_argument("--seed", type=int, default=0)
    default_policy = REPO / "assets/tasks/household_families_v1.yaml"
    plan = sub.add_parser("plan-infinigen", help="CPU: bounded complete proposal families; no generated-data claims")
    plan.add_argument("--policy", type=Path, default=default_policy)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--split-seed", type=int, default=0)
    plan.add_argument("--test-fraction", type=float, default=.2)
    plan.add_argument("--count", type=int, default=1)
    plan.add_argument("--start-index", type=int, default=0)
    plan.add_argument("--difficulty-profile", choices=["low_low","high_low","low_high","high_high"])
    plan.add_argument('--appearance-profile',choices=['requested','lighting-only'],default='requested',
                      help='explicit pilot scope; lighting-only preserves original materials and textures')
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--upstream-root", type=Path, required=True)
    plan.add_argument("--python", required=True)
    run = sub.add_parser("run-infinigen-cpu", help="CPU device sandbox: actual pinned upstream geometry")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--policy", type=Path, default=default_policy)
    run.add_argument("--upstream-root", type=Path, required=True)
    run.add_argument("--python", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=300.)
    run.add_argument("--retries", type=int, default=0)
    run.add_argument("--solver-profile", choices=["default","fast","kitchen","kitchen-fast"], default="default",
                     help="pinned upstream solver preset; identical strict final constraint and cost gates")
    batch = sub.add_parser("batch-infinigen-cpu", help="CPU: serial bounded upstream generation with exact-input resumption")
    batch.add_argument("--batch", type=Path, required=True)
    batch.add_argument("--policy", type=Path, default=default_policy)
    batch.add_argument("--upstream-root", type=Path, required=True)
    batch.add_argument("--python", required=True)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--timeout", type=float, default=1800.)
    batch.add_argument("--retries", type=int, default=0)
    batch.add_argument("--solver-profile", choices=["default","fast","kitchen","kitchen-fast"], default="kitchen")
    assets = sub.add_parser("propose-infinigen-bindings", help="CPU: exact easy_v1 cabinet assets matched to all source candidates")
    assets.add_argument("--spec", type=Path, required=True)
    assets.add_argument("--policy", type=Path, default=default_policy)
    assets.add_argument("--scene", type=Path, required=True)
    assets.add_argument("--solve-state", type=Path, required=True)
    assets.add_argument("--generation-receipt", type=Path, required=True)
    assets.add_argument("--assets-root", type=Path, required=True)
    assets.add_argument("--approved-slots", type=Path, nargs="+", required=True)
    assets.add_argument("--report", type=Path, required=True)
    export = sub.add_parser("export-infinigen", help="GPU 5 ONLY: pinned upstream texture baking and USD export of admitted CPU geometry")
    export.add_argument("--spec", type=Path, required=True)
    export.add_argument("--policy", type=Path, default=default_policy)
    export.add_argument("--receipt", type=Path, required=True)
    export.add_argument("--python", required=True, help="pinned bpy environment; no Kit launch")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--resolution", type=int, default=512)
    export.add_argument("--gpu", type=int, choices=[4, 5], default=5)
    inspect = sub.add_parser("inspect-infinigen", help="CPU standalone OpenUSD: real export inspection")
    inspect.add_argument("--scene", type=Path, required=True)
    inspect.add_argument("--solve-state", type=Path, required=True)
    inspect.add_argument("--report", type=Path)
    geometry_check = sub.add_parser("inspect-infinigen-geometry", help="CPU: verify completed real geometry against current parser and cost gates")
    geometry_check.add_argument("--spec",type=Path,required=True)
    geometry_check.add_argument("--policy",type=Path,default=default_policy)
    geometry_check.add_argument("--receipt",type=Path,required=True)
    geometry_check.add_argument("--report",type=Path,required=True)
    prepare = sub.add_parser("prepare-infinigen", help="CPU: real asset replacements and retained background")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--policy", type=Path, default=default_policy)
    prepare.add_argument("--scene", type=Path, required=True)
    prepare.add_argument("--solve-state", type=Path, required=True)
    prepare.add_argument("--generation-receipt", type=Path, required=True,
                         help="generation.json; solve state must come from an actually proxy-admitted completed attempt")
    prepare.add_argument("--bindings", type=Path, required=True)
    prepare.add_argument("--assets-root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    bind = sub.add_parser("bind-infinigen", help="CPU: compile full family using a separately GPU-probed overlay")
    bind.add_argument("--spec", type=Path, required=True)
    bind.add_argument("--policy", type=Path, default=default_policy)
    bind.add_argument("--assembly", type=Path, required=True)
    bind.add_argument("--overlay", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    preflight = sub.add_parser("gpu-preflight-infinigen", help="GPU ONLY: native, articulation, RGB and restore preflight; no release")
    preflight.add_argument("--spec", type=Path, required=True)
    preflight.add_argument("--policy", type=Path, default=default_policy)
    preflight.add_argument("--assembly", type=Path, required=True)
    preflight.add_argument("--start", type=Path, required=True, help="explicit initial pose JSON; collision checked at runtime")
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--component-preparation", type=Path,
                           help="verify whole-kitchen component import context; no leaf or family acceptance")
    preflight.add_argument('--diagnose-entity',help='private initial FCL contact measurements; does not change loaded scene or distribution')
    preflight.add_argument('--diagnose-restore-roundtrips',type=int,choices=range(6),default=0,
                           help='private bounded measurements after restore failure; never overrides rejection')
    preflight.add_argument('--restore-contract',choices=['same-checkpoint','serializer-identity'],default='same-checkpoint',
                           help='repeat the identical source checkpoint and require exact observed world/state/geometry identity')
    freeze = sub.add_parser("gpu-freeze-infinigen", help="GPU ONLY: freeze background once and compare all target insertions")
    freeze.add_argument("--family-root", type=Path, required=True)
    freeze.add_argument("--assembly", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    probe = sub.add_parser("prepare-infinigen-probe", help="CPU: geometry pose proposals and hash-bound pending probe request")
    probe.add_argument("--spec", type=Path, required=True)
    probe.add_argument("--policy", type=Path, default=default_policy)
    probe.add_argument("--assembly", type=Path, required=True)
    probe.add_argument("--assets-root", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    gpu_probe = sub.add_parser("gpu-probe-infinigen", help="GPU ONLY: existing SceneProber over every required Infinigen candidate")
    gpu_probe.add_argument("--request", type=Path, required=True)
    gpu_probe.add_argument("--out", type=Path, required=True)
    validation = sub.add_parser("prepare-infinigen-validation", help="CPU: verify compiled/frozen family and emit explicit GPU worker commands")
    validation.add_argument("--family-root", type=Path, required=True)
    validation.add_argument("--frozen-root", type=Path, required=True)
    validation.add_argument("--python", required=True, help="existing simulator Python; command generation only")
    validation.add_argument("--out", type=Path, required=True)
    worker = sub.add_parser("gpu-certify-infinigen-member", help="GPU ONLY: existing certifier and target-blind search on one frozen member")
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--episode", required=True)
    worker.add_argument("--out", type=Path, required=True)
    collect = sub.add_parser("verify-infinigen-family", help="CPU: recompute whole-family gates; pending never releases")
    collect.add_argument("--request", type=Path, required=True)
    collect.add_argument("--results", type=Path, required=True)
    collect.add_argument("--report", type=Path, required=True)
    publish = sub.add_parser("publish-infinigen-family", help="CPU: immutable evaluator-local dataset, only after complete family acceptance")
    publish.add_argument("--request", type=Path, required=True)
    publish.add_argument("--results", type=Path, required=True)
    publish.add_argument("--out", type=Path, required=True)
    check_dataset = sub.add_parser("verify-infinigen-dataset", help="CPU: recheck dataset inputs, dependencies and live acceptance")
    check_dataset.add_argument("--dataset", type=Path, required=True)
    inventory = sub.add_parser("infinigen-status", help="CPU: count actual protected pipeline artifacts and failure reasons")
    inventory.add_argument("--root", type=Path, required=True)
    inventory.add_argument("--report", type=Path)
    difficulty = sub.add_parser("infinigen-difficulty", help="CPU: recompute real reference strata and validated quota deficits")
    difficulty.add_argument("--root", type=Path, required=True)
    difficulty.add_argument("--target-per-cell", type=int, default=100)
    difficulty.add_argument("--report", type=Path)
    locations = sub.add_parser("inspect-infinigen-search-locations", help="CPU: enumerate complete source-backed search location audit; unknown remains pending")
    locations.add_argument("--assembly", type=Path, required=True)
    locations.add_argument("--report", type=Path, required=True)
    components = sub.add_parser("prepare-infinigen-components", help="CPU: separate source-backed nested cabinet cells and rebuild the complete drawer prior")
    components.add_argument("--spec", type=Path, required=True)
    components.add_argument("--policy", type=Path, default=default_policy)
    components.add_argument("--receipt", type=Path, required=True)
    components.add_argument("--upstream-root", type=Path, required=True)
    components.add_argument("--python", required=True)
    components.add_argument("--out", type=Path, required=True)
    components.add_argument("--assets-root", type=Path, required=True)
    components.add_argument("--approved-slots", type=Path, nargs='+', required=True)
    components.add_argument('--binding-profile',choices=['source-kind','single-joint-containers'],default='source-kind',
                            help='execution asset proposal; actual imported regions always rebuild the complete prior')
    component_export = sub.add_parser("export-infinigen-components", help="GPU: pinned exporter over hash-checked real split meshes; no task acceptance")
    component_export.add_argument("--preparation", type=Path, required=True)
    component_export.add_argument("--upstream-root", type=Path, required=True)
    component_export.add_argument("--python", required=True)
    component_export.add_argument("--out", type=Path, required=True)
    component_export.add_argument("--resolution", type=int, default=64)
    component_export.add_argument("--gpu", type=int, choices=[4,5], default=5)
    component_assembly = sub.add_parser("prepare-infinigen-component-assembly", help="CPU: existing builder input for every real component asset; full leaf binding remains pending")
    component_assembly.add_argument("--spec", type=Path, required=True)
    component_assembly.add_argument("--policy", type=Path, default=default_policy)
    component_assembly.add_argument("--preparation", type=Path, required=True)
    component_assembly.add_argument("--export-receipt", type=Path, required=True)
    component_assembly.add_argument("--assets-root", type=Path, required=True)
    component_assembly.add_argument("--out", type=Path, required=True)
    native_regions=sub.add_parser('bind-infinigen-native-regions',
        help='CPU: map every actually imported fillable region to the existing SceneSlots; physical probe still required')
    native_regions.add_argument('--assembly',type=Path,required=True)
    native_regions.add_argument('--preflight',type=Path,required=True)
    native_regions.add_argument('--out',type=Path,required=True,help='new manifest JSON beside the source assembly')
    native_family=sub.add_parser('derive-infinigen-native-family',
        help='CPU: derive a complete native-location candidate family with original generation lineage')
    native_family.add_argument('--source-spec',type=Path,required=True)
    native_family.add_argument('--policy',type=Path,default=default_policy)
    native_family.add_argument('--assembly',type=Path,required=True)
    native_family.add_argument('--generation-receipt',type=Path,required=True)
    native_family.add_argument('--out',type=Path,required=True)
    gpu_stages={name for name in sub.choices if name.startswith('gpu-')}|{'export-native','verify-native'}
    for name in gpu_stages:
        sub.choices[name].add_argument('--gpu',type=int,choices=[4, 5],default=5,
                                      help='physical GPU 4 or 5, enforced via PCI/UUID device isolation')
        sub.choices[name].add_argument('--gpu-isolated',action='store_true',help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.stage in gpu_stages:
        from rummagebench.sim.gpu_isolation import resolve_gpu,isolated_gpu_command,verify_isolated_cuda,configure_single_gpu_kit
        if not args.gpu_isolated:
            import subprocess
            device=resolve_gpu(args.gpu)
            command,environment=isolated_gpu_command([sys.executable,str(Path(__file__).resolve()),
                *(sys.argv[1:] if argv is None else argv),'--gpu-isolated'],device)
            print(f"GPU isolation: physical={device['index']} PCI={device['pci']} device={device['device']} UUID={device['uuid']}",file=sys.stderr)
            result=subprocess.run(command,env=environment,timeout=1800,check=False)
            if args.stage=='gpu-preflight-infinigen':
                return preflight_child_result(result.returncode,args.out)
            if args.stage=='gpu-probe-infinigen':
                return probe_child_result(result.returncode,args.out)
            return result.returncode
        import torch
        verify_isolated_cuda(torch)
        import os
        os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
        import isaacsim
        configure_single_gpu_kit(isaacsim)
        # GPU stages verify assembly/restore USD inputs before constructing
        # an Environment. Isaac exposes its own pxr modules after Kit starts.
        from rummagebench.sim.omnigibson.env_factory import apply_runtime_env
        apply_runtime_env()
        import omnigibson as og
        from omnigibson.macros import gm
        gm.ENABLE_TRANSITION_RULES = False
        if og.sim is None:
            og.launch()
    if args.stage == "inspect-infinigen-search-locations":
        from rummagebench.authoring.tasks.household_locations import inspect_search_locations
        report = inspect_search_locations(args.assembly)
        write_report(args.report,report)
        return 2 if report["status"]=="failed" else 0
    if args.stage == "prepare-infinigen-components":
        import os
        import subprocess
        from rummagebench.authoring.tasks.infinigen import CPU_ENV, cpu_sandbox_argv, inspect_geometry_receipt, source_audit
        from rummagebench.authoring.tasks.infinigen_component_binding import expand_search_distribution, propose_component_assets
        args.out = args.out.resolve()
        spec = load_household_spec(args.spec, args.policy)
        source_audit(args.upstream_root)
        verified = inspect_geometry_receipt(spec, args.receipt)
        if len(verified['completed']) != 1:
            raise ValueError('ONE_COMPLETE_SOURCE_SCENE_REQUIRED')
        folder = Path(verified['completed'][0]['folder'])
        geometry = json.loads((folder / 'findingbench_geometry.json').read_text())
        proposal = expand_search_distribution(spec, geometry.get('cabinet_component_census') or {})
        if args.out.exists():
            raise ValueError('COMPONENT_OUTPUT_ALREADY_EXISTS_USE_NEW_OUTPUT')
        args.out.mkdir(parents=True)
        env = dict(os.environ, **CPU_ENV, PYTHONPATH=os.pathsep.join([str(REPO / 'src'), str(args.upstream_root.resolve())]))
        command = [args.python, '-m', 'rummagebench.authoring.tasks.infinigen_component_binding',
                   '--input-folder', str(folder), '--out', str((args.out / 'geometry').resolve())]
        with (args.out / 'split.log').open('w') as log:
            result = subprocess.run(cpu_sandbox_argv(command, args.out), cwd=args.upstream_root,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
        report = {'version': 1, 'stage': 'prepare_infinigen_components',
                  'status': 'source_components_prepared_pending_binding' if result.returncode == 0 else 'failed',
                  'returncode': result.returncode, 'source_receipt_sha256': sha256_path(args.receipt),
                  'code_sha256': sha256_path(REPO / 'src/rummagebench/authoring/tasks/infinigen_component_binding.py'),
                  'family_proposal': proposal, 'compiled_families': 0, 'evaluation_ready_episodes': 0}
        if result.returncode == 0:
            component_doc = json.loads((args.out / 'geometry/components.json').read_text())
            report['asset_proposals'] = propose_component_assets(component_doc, args.approved_slots, args.assets_root,args.binding_profile)
            report['artifacts'] = {str((args.out / 'geometry' / n).resolve()): sha256_path(args.out / 'geometry' / n)
                                   for n in ('scene.blend', 'solve_state.json', 'components.json')}
        write_report(args.out / 'components_preparation.json', report)
        return 0 if result.returncode == 0 else 2
    if args.stage == "export-infinigen-components":
        import os
        import subprocess
        from rummagebench.authoring.tasks.infinigen_component_binding import verify_component_preparation
        folder,_ = verify_component_preparation(args.preparation)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / 'src'), str(args.upstream_root.resolve())]))
        command = [args.python, '-m', 'rummagebench.authoring.tasks.infinigen', 'export-upstream',
                   '--upstream-root', str(args.upstream_root.resolve()), '--input_folder', str(folder),
                   '--output_folder', str(args.out.resolve()), '-r', str(args.resolution), '--gpu', str(args.gpu)]
        return subprocess.run(command, env=env, timeout=1800, check=False).returncode
    if args.stage == "prepare-infinigen-component-assembly":
        from rummagebench.authoring.tasks.infinigen_component_binding import prepare_component_assembly
        manifest = prepare_component_assembly(load_household_spec(args.spec,args.policy),args.preparation,
            args.export_receipt,args.assets_root,args.out)
        write_report(None, {'status':manifest['status'],'assembly':str(args.out / 'assembly.json'),
                           'real_interactive_asset_bodies':len(manifest['candidate_bindings']),
                           'complete_search_locations':manifest['component_context']['complete_search_distribution']['candidate_count'],
                           'evaluation_ready_episodes':0})
        return 0
    if args.stage == 'bind-infinigen-native-regions':
        from rummagebench.authoring.tasks.infinigen_component_binding import bind_native_region_catalog
        doc=bind_native_region_catalog(args.assembly,args.preflight,args.out)
        write_report(None,{'status':'explicit_native_regions_pending_probe','assembly':str(args.out),
            'native_regions':len(doc['native_region_bindings']),'evaluation_ready_episodes':0})
        return 0
    if args.stage == 'derive-infinigen-native-family':
        from rummagebench.authoring.tasks.infinigen_component_binding import derive_native_family
        from rummagebench.authoring.level1.snapshot import write_environment_snapshot
        members=derive_native_family(args.source_spec,args.policy,args.assembly,args.generation_receipt,REPO)
        artifacts={f'specs/{s.task_id}.json':s.model_dump(mode='json') for s in members}
        report={'status':'candidate_pending_complete_physical_probe','family_id':members[0].family_id,
            'candidate_locations':len(members),'compiled_candidates':0,'evaluation_ready_episodes':0,
            'source_layout_group':members[0].split_assignment.source_layout_group,
            'spec_artifact_hashes':{name:sha256_bytes(canonical_json(value)) for name,value in artifacts.items()}}
        write_environment_snapshot(args.out,{**artifacts,'native_family.json':report})
        write_report(None,report)
        return 0
    if args.stage == "propose-infinigen-bindings":
        from rummagebench.sim.omnigibson.infinigen_adapter import propose_interactive_bindings
        report = propose_interactive_bindings(load_household_spec(args.spec,args.policy),args.scene,args.solve_state,
            args.generation_receipt,args.assets_root,args.approved_slots)
        write_report(args.report,report)
        return 2 if report["status"] == "rejected" else 0
    if args.stage == "export-infinigen":
        import subprocess
        import os
        from rummagebench.authoring.tasks.infinigen import inspect_geometry_receipt, export_command
        report = inspect_geometry_receipt(load_household_spec(args.spec,args.policy),args.receipt)
        completed = [r for r in report["completed"] if r["measurement"]["proxy_filter_passed"]]
        if not completed:
            raise ValueError("GPU_EXPORT_REQUIRES_ADMITTED_COMPLETE_CPU_GEOMETRY")
        receipt = json.loads(args.receipt.read_text())
        root = Path(receipt["source"]["root"])
        command = export_command(args.python,root,Path(completed[-1]["folder"]),args.out,args.resolution)
        environment = dict(os.environ,PYTHONPATH=os.pathsep.join([str(REPO / "src"),str(root.resolve())]))
        result = subprocess.run(command["argv"],cwd=command["cwd"],env=environment,timeout=1860,check=False)
        if result.returncode != 0:
            return result.returncode if result.returncode > 0 else 2
        doc = json.loads((args.out / "export.json").read_text())
        write_report(None,{"status":doc["status"],"scene":doc["scene"],"physical_gpu":5,"evaluation_ready_episodes":0})
        return 0
    if args.stage == "batch-infinigen-cpu":
        report = batch_cpu_generate(args)
        write_report(None,report)
        return 2 if report["failure_reasons"] else 0
    if args.stage == "infinigen-difficulty":
        from rummagebench.authoring.tasks.household_dataset import difficulty_inventory
        write_report(args.report,difficulty_inventory(args.root,args.target_per_cell))
        return 0
    if args.stage == "infinigen-status":
        from rummagebench.authoring.tasks.household_dataset import pipeline_inventory
        write_report(args.report,pipeline_inventory(args.root))
        return 0
    if args.stage == "prepare-infinigen-validation":
        from rummagebench.authoring.tasks.household_dataset import prepare_family_validation
        write_report(None, prepare_family_validation(args.family_root,args.frozen_root,args.out,args.python,Path(__file__)))
        return 0
    if args.stage == "gpu-certify-infinigen-member":
        from rummagebench.authoring.tasks.household_certify import certify_member
        report = certify_member(args.request,args.episode,args.out)
        write_report(None, {"episode_id": report["episode_id"], "status": report["status"],
                            "checks": report["checks"], "evaluation_ready_episodes": 0})
        return 0 if report["status"] == "certified" else 2
    if args.stage == "verify-infinigen-family":
        from rummagebench.authoring.tasks.household_dataset import collect_family_validation
        report = collect_family_validation(args.request,args.results)
        write_report(args.report,report)
        return 0 if report["status"] == "validated" else 2
    if args.stage == "publish-infinigen-family":
        from rummagebench.authoring.tasks.household_dataset import publish_family
        write_report(None,publish_family(args.request,args.results,args.out))
        return 0
    if args.stage == "verify-infinigen-dataset":
        from rummagebench.authoring.tasks.household_dataset import load_dataset
        write_report(None,load_dataset(args.dataset))
        return 0
    if args.stage == "inspect-infinigen-geometry":
        from rummagebench.authoring.tasks.infinigen import inspect_geometry_receipt
        report = inspect_geometry_receipt(load_household_spec(args.spec,args.policy),args.receipt)
        write_report(args.report,report)
        return 0 if report["status"] == "geometry_generated" else 2
    if args.stage == "prepare-infinigen-probe":
        from rummagebench.authoring.tasks.infinigen_probe import prepare_probe
        request = prepare_probe(load_household_spec(args.spec,args.policy),args.assembly,REPO,args.assets_root,args.out)
        write_report(None,{"status":request["status"],"request":str(args.out/"request.json"),
                           "evaluation_ready_episodes":0})
        return 0
    if args.stage == "gpu-probe-infinigen":
        from rummagebench.authoring.tasks.infinigen_probe import run_gpu_probe
        write_report(None,run_gpu_probe(args.request,args.out))
        return 0
    if args.stage == "gpu-freeze-infinigen":
        write_report(None, gpu_freeze_infinigen(args))
        return 0
    if args.stage == "gpu-preflight-infinigen":
        write_report(None, gpu_infinigen_preflight(args))
        return 0
    if args.stage == "plan-infinigen":
        plan_infinigen(args)
        return 0
    if args.stage == "run-infinigen-cpu":
        report = cpu_generate(args)
        write_report(None, report)
        return 0 if report["status"] == "geometry_generated" else 2
    if args.stage == "inspect-infinigen":
        from rummagebench.authoring.tasks.infinigen import inspect_export
        write_report(args.report, inspect_export(args.scene, args.solve_state))
        return 0
    if args.stage == "prepare-infinigen":
        write_report(None, prepare_infinigen(args))
        return 0
    if args.stage == "bind-infinigen":
        write_report(None, bind_infinigen(args))
        return 0

    if args.stage == "cpu":
        host = host_config(args.host_config)
        root = args.robot_assets or host["behavior_assets_root"].parent / "omnigibson-robot-assets"
        report = cpu_audit(host, root)
        write_report(args.report, report)
        return 2 if report["status"] == "blocked" else 0
    report = native_capture(args) if args.stage == "export-native" else verify_native_restore(args)
    write_report(None, report)
    return 0


if __name__ == "__main__":
    try:
        result = main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                          "evaluation_ready_episodes": 0}, ensure_ascii=False))
        result = 2
    raise SystemExit(result)
