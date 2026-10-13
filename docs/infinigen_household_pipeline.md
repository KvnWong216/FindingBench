# Infinigen household pipeline — CPU handoff, 2026-10-13

This implementation starts from the actual remote repository. PR #2 (`93dc197`)
was fetched and merged without rewriting history. The earlier cloud commits
`0304c04` / `200f9e6` remain unavailable; their reported 55 tests are not evidence
for this implementation. The user subsequently selected Infinigen Indoors.
Only kitchen closed-cabinet search is implemented; other families are rejected.

## Code map and reuse

| Boundary | Real code / entry |
|---|---|
| Existing easy sampling | `assets/tasks/tiers/easy_v1.yaml`, `tasks/tiers/easy.py::propose`, `planner.py::plan_episode`; approved `PlacementPriors`, `ObjectEligibility`, `SceneSlots` and robot-dependent overlays remain the asset/placement contracts |
| Task / environment schemas | `tasks/plan.py::TaskPlan`, `core/scenario.py::ScenarioSpec`; `SceneSpec` now accepts a hash-checked Infinigen assembly reference, with BEHAVIOR remaining the default |
| Current public protocol | `core/public_types.py::PUBLIC_SKILLS`: MOVE, TURN, OPEN, CLOSE, GRASP, PLACE, OBSERVE, REPORT_DONE; `core/visual_session.py::VisualProtocolSession` implements the visual protocol. Internal NAV/entity IDs and compiler/oracle paths stay private. Wrong grasps are recoverable; success requires reporting while holding the target |
| New CPU specifications | `tasks/household.py::{sample_spec,validate_spec_assets,family_members,grouped_split}`, `assets/tasks/household_families_v1.yaml`; full candidate set, independent asset/candidate/location/background/appearance/difficulty streams, input hashes, one split per base layout |
| Upstream boundary | `tasks/infinigen.py::{source_audit,runtime_audit,generation_command,run_cpu_generation,read_solve_state,inspect_export}`; fixed CLI, no shell execution, CPU device isolation, version/config/runtime hashes, finite retries, logs, kernel-released locks and per-attempt recovery |
| Assembly and compiler | `sim/omnigibson/infinigen_adapter.py::{prepare_assembly,assembly_env_config,scene_slots_from_assembly}`, `tasks/household_binding.py::bind_family`, existing `compile.py::compile_plan`, `env_factory.py::{build_env_config,build_env}` |
| Occupancy / freeze | `native_occupancy.py::{capture_native_occupancy,require_known_slots,verify_source}`, builder hooks in `backend.py::setup`; adapter `freeze_family_background` freezes after the target topology exists but before insertion, restores the same parked-target snapshot for every member, and rejects background changes |
| Difficulty | `tasks/difficulty.py::exact_search_cost`: exact DP only for supplied static travel costs, positive known priors, one perfect reveal per node and mandatory confirmation. Cost includes declared inspection plus GRASP/REPORT; MOVE/TURN belong in the supplied travel cost. `euclidean_proxies` explicitly cannot certify traversable paths |
| Unified entry | `scripts/tasks/household_pipeline.py --help`; CPU stages and explicitly named GPU stages share the above contracts |

`bind_family` requires every candidate's real OPEN/GRASP probe and causal reveal
in an overlay bound to the assembly hash and current URDF. It never prunes a
failed candidate. It compiles existing scenario fields, marks family provenance,
and leaves acceptance pending. The legacy promoter additionally rejects any
Infinigen scenario even if family markers are removed.

## Integration findings and regression checks

| Issue / code | Impact | Fix / CPU regression |
|---|---|---|
| Native `active` flags, `infinigen.py::inspect_export` | Omitting inactive solver objects loses background | Map exported geometry independently of solving activity; `test_real_usd_inspection_retains_inactive_and_nonsearch_geometry` |
| Geometry names, relations and asset paths, `read_solve_state` / `prepare_assembly` | Misbound IDs or missing textures can look usable | Reject unresolved/ambiguous mappings, dependencies and incomplete candidate sets; mapping/texture unit tests |
| Whole-scene cabinet meshes | Configuring a joint field cannot recover articulation | Require real local replacements, then query actual handles/joints/limits; missing-joint unit tests, runtime pending |
| Unbound kitchen storage, `prepare_assembly` | Plausible search nodes could become noninteractive background while the certificate counts fewer nodes | Reject unresolved extra storage, never shrink the candidate set; extra-storage regression |
| Bare-category task with native same-class props | Instruction can identify several plausible targets | Preserve background and reject family binding on native confounds; `test_native_ambiguous_target_is_preserved_and_rejected_before_family_binding` |
| Relation-only matched members | Target resampling may disturb or rebuild background | One topology-compatible parked-target freeze, restore every member, exact background comparison; perturbation fixture tests, runtime pending |
| Legacy compiler termination | Wrong-grasp failure/automatic holding success conflict with current visual protocol | New source compiles recoverable wrong grasps and report-required success; full-family compilation regression |
| Native restore identity | Same configuration/state could be labelled as another scene | Bind scene ID, actual manifest/config/asset/state hashes; `test_native_restore_rejects_changed_scene_identity` |
| Legacy single-episode promoter | Removing family markers could bypass family validation | Also reject `scene.source=infinigen`; promotion-bypass regression |
| Interrupted subprocess runs | A stale lock or partial output could break continuation | Kernel advisory lock, immutable input identity and bounded attempt checkpoints; timeout/interruption/idempotency regressions |

## Pinned upstream evidence

Checkout: `/data1/ygwang/codes/_external/infinigen-indoors-b11700e`,
`indoors-stable` peeled commit **b11700ebac77695d8f53538705cfdc0cd890c54d**,
BSD-3-Clause. The source audit stores exact file hashes. No upstream code or
BEHAVIOR assets were copied into this repository.

- [Generation entry](https://github.com/princeton-vl/infinigen/blob/b11700ebac77695d8f53538705cfdc0cd890c54d/infinigen_examples/generate_indoors.py): Python 3.11 / bpy 4.2, `-m infinigen_examples.generate_indoors --seed ... --task coarse -g singleroom.gin -p ...`. `home_room_constraints` and `home_furniture_constraints` provide a constraint DSL, not a native natural-language API. No paid model API or generated Python is used.
- [Home semantics](https://github.com/princeton-vl/infinigen/blob/b11700ebac77695d8f53538705cfdc0cd890c54d/infinigen_examples/constraints/semantics.py) maps actual factories to room/furniture/handheld categories; the home constraints supply interior/surface placement and co-occurrence. The adapter preserves this whole background without target-conditioned object removal. It adds no new full-house generator. `singleroom.gin` limits furnishing to one room; it does **not** promise that the blueprint has only one room. Completeness of other visible regions remains a postcheck, and this pilot must not be called a fully furnished house.
- [Solve-state serialization](https://github.com/princeton-vl/infinigen/blob/b11700ebac77695d8f53538705cfdc0cd890c54d/infinigen/core/constraints/example_solver/state_def.py): `objs` contains Blender names, tags, factory representation and relations with target IDs and relation types. `active` describes a solving stage; it is not a safe geometry omission flag. These relations are proposals, not measured OG Inside/OnTop states.
- [CPU device selection](https://github.com/princeton-vl/infinigen/blob/b11700ebac77695d8f53538705cfdc0cd890c54d/infinigen/core/init.py): `configure_cycles_devices.use_gpu=False` returns before GPU enumeration. The runner additionally requires bubblewrap: private `/dev` without NVIDIA/DRM nodes, no network, source read-only, only the attempt directory writable, two CPU threads and bounded timeout. No fallback to headless GPU execution.
- [Exporter](https://github.com/princeton-vl/infinigen/blob/b11700ebac77695d8f53538705cfdc0cd890c54d/infinigen/tools/export.py) sets Cycles to GPU inside whole-scene texture baking. Consequently the tool emits a **deferred** export command and never runs it in this CPU round. [The upstream simulator guide](https://github.com/princeton-vl/infinigen/blob/b11700ebac77695d8f53538705cfdc0cd890c54d/docs/ExportingToSimulators.md) explains the whole-scene articulation limitation. The different `articulated-stable` branch is not mixed into this pipeline.

Whole USD cabinets are replaced only through explicit source-ID → exported
prim → existing local BEHAVIOR model bindings. Missing candidates, extra unbound kitchen storage that would enlarge the search
set, duplicate source bindings, invalid scale, missing openable metadata/assets or missing
texture dependencies reject preparation. Native unknown/same-category handheld
items remain in the background; ambiguous bare-category instructions block
family binding. Real joints, limits, handles and door space still need runtime
checks. Metadata cannot prove articulation. Other meshes, including nonsearch
regions, are preserved; explicitly tagged single-mesh handheld items become
movable bodies, grouped dynamic meshes and instanced geometry are rejected until
supported. Collision meshes/convex hulls are proposals and require physical checks.

`install_runtime_lighting` actually applies intensity, temperature, ambient and
light yaw through USD attributes after OG loads. CPU OpenUSD tests prove these
edits leave geometry attributes untouched. Wall palettes and support/container
texture swaps are recorded as unsupported pending material mapping; renderer
settings and image acceptance are also pending. Appearance is not structural
layout diversity. The assembly remains pending regardless of light application.

All five local model YAML/USD assets exist (R1Pro, R1, Fetch, Tiago, Stretch).
Only R1Pro has this repository's task/kinematics configuration; the others are
`integration_pending`, none is `validated`. Current code selects head RGB.
There is no wrist-only or paper-camera equivalence claim. Common solvable and
per-body subsets must be measured separately; no shared acceptance label.

## CPU reproduction and actual evidence

Run from `/data1/ygwang/codes/FindingBench`. The isolated environment is
`/data1/ygwang/codes/_external/findingbench-cpu`; it inherits the existing Python
3.11 CPU packages and installs only standalone `usd-core==26.8` / `gin-config==0.5.0`
locally. Standalone USD parsing does not start Isaac/Kit. The optional project
extra is `indoor-cpu`. Complete Infinigen dependencies are still missing:
installation of its 519.4 MB bpy wheel timed out; do not assume installation
succeeded. The current external checkout is sparse source/configs and needs the
remaining upstream resources/dependencies before geometry generation can finish.

```bash
export PYTHONPATH=src
CPU_PYTHON=/data1/ygwang/codes/_external/findingbench-cpu/bin/python
IG_ROOT=/data1/ygwang/codes/_external/infinigen-indoors-b11700e

# Existing CPU asset/semantic inventory, run successfully on this machine:
$CPU_PYTHON scripts/tasks/mine_priors.py
$CPU_PYTHON scripts/tasks/build_eligibility.py

$CPU_PYTHON scripts/tasks/household_pipeline.py plan-infinigen \
  --seed 216 --count 1 --out build/household/infinigen/pilot216_current/specs \
  --upstream-root "$IG_ROOT" --python "$CPU_PYTHON"

$CPU_PYTHON scripts/tasks/household_pipeline.py run-infinigen-cpu \
  --spec build/household/infinigen/pilot216_current/specs/task_99314009ab96085785e4.json \
  --upstream-root "$IG_ROOT" --python "$CPU_PYTHON" \
  --out build/household/infinigen/pilot216_current/cpu_run --timeout 120 --retries 0

$CPU_PYTHON -m pytest tests/unit -q
$CPU_PYTHON scripts/tasks/household_pipeline.py cpu
```

The real upstream invocation exited **1**, `ModuleNotFoundError: No module named
'bpy'`, classified `UPSTREAM_DEPENDENCY_MISSING`; the unified runner returned 2.
Evidence is in `build/household/infinigen/pilot216_current/cpu_run/{input.json,
generation.json,attempt_0/attempt.json,attempt_0/stderr.log,attempt_0/stdout.log}`.
`input.json` includes interpreter/package versions and executable hash.
Identical reruns reuse the failure receipt; installing dependencies changes the
runtime fingerprint and requires a new output directory, retaining old evidence.
Earlier debug receipts remain under `pilot216/`; they are not accepted data.

Current pilot: **4 specs, 1 complete proposal family, 1 base-layout request,
0 real geometry scenes, 0 exported scene files, 0 simulation-validated families,
0 evaluation-ready episodes**. Across development, three spec samplings wrote 12
JSON specs for the same family (8 obsolete debugging specs, 4 current); this does
not represent independent layouts. Five real upstream calls failed at the bpy import,
including four earlier debug/provenance calls. No GPU, Isaac Sim/Kit, rendering or training
was started. No real export exists to inspect; successful OpenUSD assembly and
light checks used explicitly synthetic unit fixtures and count as **zero** data.

CPU tests cover schema/semantic rejection, deterministic independent streams,
background target independence, split grouping, exact hand/exhaustive search
costs, bin boundaries, real USD APIs, missing maps/joints/textures, complete
family compilation, perturbation rejection, bounded failures/timeouts,
checkpoint recovery, hash conflicts, GPU-free module imports and promotion guards.
Executed command: `PYTHONPATH=src /data1/ygwang/codes/_external/findingbench-cpu/bin/python -m pytest tests/unit -q` — **603 passed, 14 skipped in 75.45 s**. The full suite ran before the final identity/search-set guards; the latest focused regressions include those additions (**84 passed in 1.05 s** across Infinigen/native occupancy/baseline audit tests). Skips are not physical validation. The original cloud household test command cannot be reproduced because its source files remain absent.

## Deferred commands and acceptance boundaries

**Do not execute these GPU stages during this CPU-only round.** After real
geometry succeeds, the pinned export command is recorded in `specs/batch.json`.
It uses `python -m infinigen.tools.export --input_folder ... --output_folder ...
-f usdc -r 512 --omniverse`; run it only in a later GPU-authorized session.

Once a real export, its `solve_state.json`, and explicit real cabinet bindings
are available, the following two stages are CPU-only:

```bash
$CPU_PYTHON scripts/tasks/household_pipeline.py inspect-infinigen \
  --scene "$SCENE_USD" --solve-state "$SOLVE_JSON" --report "$EXPORT_REPORT"
$CPU_PYTHON scripts/tasks/household_pipeline.py prepare-infinigen \
  --spec "$SPEC_JSON" --scene "$SCENE_USD" --solve-state "$SOLVE_JSON" \
  --bindings "$BINDINGS_JSON" --assets-root /data1/ygwang/codes/BEHAVIOR-1K/datasets/behavior-1k-assets \
  --out "$ASSEMBLY_OUT"
```

Each binding contains exactly `location_id,source_id,category,model`, taken from
actual solver/export objects and existing local models. The specification's
complete set must be mapped. `prepare-infinigen` produces `assembly.json`,
normalized per-object USDs, lights and `scene_slots.json`, all pending. It does
not publish or upload raw assets. A partial failed bundle has `failure.json` and
no assembly commit marker.

Smallest later **GPU** preflight (existing BEHAVIOR/OG environment; actual
start JSON is `{position:[x,y,z],orientation:[x,y,z,w]}` in exported world units):

```bash
PYTHONPATH=src /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/household_pipeline.py gpu-preflight-infinigen \
  --spec "$SPEC_JSON" --assembly "$ASSEMBLY_OUT/assembly.json" \
  --start "$START_JSON" --out "$PREFLIGHT_OUT"
```

It measures actual entity inventory/three-way occupancy, closed native state,
real joint limits/handle presence, spawn clearance, RGB shape and two native
restores; it saves evidence and an image before process shutdown. These checks
remain **unexecuted**. This preflight does not produce a probed embodiment overlay.
A real scene-specific OPEN/GRASP/visibility/navigation probe is still needed;
the legacy `probe_embodiment.py` relies on BEHAVIOR room maps and cannot simply
be pointed at an Infinigen layout. Porting that room-map/probe path is unfinished.
Once that probe supplies a complete overlay with the assembly-hash note:

```bash
$CPU_PYTHON scripts/tasks/household_pipeline.py bind-infinigen \
  --spec "$SPEC_JSON" --assembly "$ASSEMBLY_OUT/assembly.json" \
  --overlay "$PROBED_OVERLAY" --out "$FAMILY_OUT"

# Later GPU session only:
PYTHONPATH=src /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/household_pipeline.py gpu-freeze-infinigen \
  --family-root "$FAMILY_OUT" --assembly "$ASSEMBLY_OUT/assembly.json" \
  --out "$FROZEN_CHECK_OUT"
```

Acceptance must also establish natural furnishing outside search locations,
cavity/target fit, collisions/stability, door sweep, temporary placement,
initial invisibility, OPEN reveal, OBSERVE bypass/coupled reveal, reachable
public skill chains, correct completion/recoverable wrong grasps, actual reset
and identical background after every insertion. Unknown occupancy rejects the
whole family. The freeze command checks full per-object states/poses/AABBs and
rejects perturbations; it is not the final family verifier. A verified physical
snapshot publisher, all skill/image gates, target-blind reference search
measurement and final spatial×operation bins remain unfinished. Reference costs
must label measured routes/actions and assumptions; oracle execution is only a
feasibility witness. Layout-grouped dev/test is implemented, not a claim of
unseen assets or task combinations. No release path accepts these pending bundles.
