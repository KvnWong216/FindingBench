# Infinigen household pipeline — CPU handoff, 2026-10-14

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
| Frozen runtime state | `tasks/household_state.py`; hash-check local member tensors, scenario and background records, verify serialized object/link order BEFORE deserialization; `backend.setup/reset` restores and checks the frozen background |
| Family validation / dataset | `tasks/household_dataset.py`; immutable validation requests, generation-admission binding, whole-family ID/split/artifact validation, gate/cost recomputation, local publication and reload verification |
| Real reference search | `tasks/household_search.py`; fixed target-blind order, dynamic production MOVE/TURN routes and point skills, accepted closed-OBSERVE view checks, measured reference bins; no optimality claim |
| GPU worker | `tasks/household_certify.py`; existing TaskCertifier and GeometryRoomMap on a frozen member; pending checks block publication |
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

## CPU reproduction and actual evidence (rectification round)

The scope is one kitchen cabinet-search family and the existing R1Pro/head RGB
protocol. Public skills, recoverable wrong grasps and REPORT_DONE semantics are
unchanged. No GPU, Isaac/Kit, rendering or training has run in this round.

Fixes and regression evidence:

- `household.SplitAssignment` persists seed, fraction, strategy, layout and source
  blueprint group. Difficulty/candidate variants of the same upstream scene seed
  share a split. `household_binding.bind_family` writes a bijection between spec
  `task_...` IDs and compiled `fb_...` IDs, keyed compiled splits, and rejects
  duplicate/missing/extra mappings. Tests use non-default seed 17.
- `infinigen.usd_dependency_ledger` walks transitive layers/references/payloads,
  asset attributes and time samples. Canonical file paths, hashes and reference
  edges are verified before loading. Packages, remote resolvers and unresolved
  sequences are rejected. Assembly v2 is required; regenerate v1 bundles.
  `copy_effective_materials` makes inherited/collection bindings explicit and
  anchors texture paths. Five new regression cases failed before the fix;
  unchanged fixtures now load, modified/deleted textures/layers reject.
- Generation consumes pinned upstream `Problem`/DSL floor-and-wall Storage and
  KitchenCabinet count constraints plus separation. Every actual kitchen Storage
  object is counted in the postfilter; extras reject rather than get removed.
  Counter/drawer alternatives are listed and require runtime classification.
  Geometry receipts call the production DP and interval bins and retain the full
  requested candidate mapping. No hidden target location enters generation.
- Under uniform priors, OPEN+OBSERVE costs 2 per inspected location and
  GRASP+REPORT_DONE costs 2 at completion: n=2/3/4 gives expected operations
  5/6/7. Low [0,6) uses n=2; high [6,infinity) uses n=3/4. Unreachable request
  profiles fail before sampling. Spatial units are 0.5 m Euclidean proxy units;
  the public MOVE cap remains 1 m. These are furniture-center proxies, not
  interaction poses, navigable paths, TURN counts or final search difficulty.
- The pinned `infinigen.terrain.__init__` is empty despite the upstream import.
  The thin bridge exposes its real `terrain.core.Terrain` class. It retains the
  upstream intermediate-stage warnings and checks ALL kitchen constraints after
  all solving stages: lights/islands must not be required before their stage.
  Final violations still reject the entire scene. No constraint is deleted.
- `infinigen_probe.prepare_probe` replaces BEHAVIOR PNG maps/scene_json floor
  lookup with actual exported kitchen floor triangles, geometry pose proposals
  and an explicitly pending request. Live `SceneProber` supplies footprint/route,
  IK, placement, closed hiding and open reveal measurements. The whole candidate
  set is required. Receipt hashes bind assembly, URDF, robot/camera config,
  skills/probe implementations and local target resources. Binding rejects an
  overlay without the matching receipt. No CPU command emits a probed overlay.

Two independent environments now exist. Ordinary tests use a fresh venv without
system site packages, installed with `pip install -e '.[test,indoor-cpu]'`.
SciPy is a declared dependency. Local asset checks require explicit
`FINDINGBENCH_BEHAVIOR_ASSETS`, `FINDINGBENCH_TASK_BUILD` and
`FINDINGBENCH_R1PRO_URDF`; optional collision tests report missing Pinocchio/FCL.
The previous implicit absolute-path test uses a temporary synthetic fixture.
All synthetic USD/robot/probe fixtures count as zero generated data.

```bash
export PYTHONPATH=src
TEST_PYTHON=/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python
CPU_PYTHON=/data1/ygwang/codes/_external/infinigen-cpu-b11700e/bin/python
IG_ROOT=/data1/ygwang/codes/_external/infinigen-indoors-b11700e
$TEST_PYTHON -m pytest tests/unit -q

$TEST_PYTHON scripts/tasks/household_pipeline.py plan-infinigen \
  --seed 216 --split-seed 17 --difficulty-profile low_low --count 1 \
  --out build/household/infinigen/reproduce216/specs \
  --upstream-root "$IG_ROOT" --python "$CPU_PYTHON"
# Select either task_*.json listed in batch.json (both share one background).
$TEST_PYTHON scripts/tasks/household_pipeline.py run-infinigen-cpu \
  --spec build/household/infinigen/reproduce216/specs/<task_id>.json \
  --upstream-root "$IG_ROOT" --python "$CPU_PYTHON" \
  --out build/household/infinigen/reproduce216/run --timeout 900 --retries 0 \
  --solver-profile fast
```

`default` uses the normal upstream solve budget; `fast` explicitly selects the
pinned `fast_solve.gin` preset. Both use the same strict final constraints and
cost filter. The fast CPU budget also limits invalid proposal retries to one. Reduced sampling can increase rejection and never grants acceptance. The kernel sandbox
requires Linux bubblewrap, read-only source, private /dev without NVIDIA/DRM,
no network, private /tmp and two CPU threads. PYTHONHASHSEED=0 fixes upstream
hash iteration. Seeds are encoded in hexadecimal, matching upstream parse_seed.
A successful receipt emits export input from its actual attempt_N/geometry.
Rejected or interrupted runs retain errors and cannot supply an accepted export.
Source snapshots/input hashes and artifacts protect resume/conflict checking.

The isolated upstream environment uses Python 3.11.16, bpy 4.2.0, NumPy 1.26.4,
SciPy 1.10.1, scikit-image 0.19.3, shapely 2.0.5 and standalone usd-core 26.8.
All imported successfully inside the CPU device sandbox; pip check passes.
The official bpy wheel SHA256 is
`a8217c2bd7b3424d1f51ce0698c0bf65439de352f7f6ed7b9942e2a9e5eb121b`.
Scikit-image was built on CPU with Cython 0.29.37 and pythran 0.12.2.
Reproduce the isolated dependency installation (Python 3.11):

```bash
python3.11 -m venv "$CPU_ENV_PATH"
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' "$CPU_ENV_PATH/bin/python" -m pip install \
  --index-url https://pypi.org/simple numpy==1.26.4 scipy==1.10.1 \
  Cython==0.29.37 pythran==0.12.2 setuptools==69.5.1 wheel
NPY_NUM_BUILD_JOBS=2 OMP_NUM_THREADS=2 PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' \
  "$CPU_ENV_PATH/bin/python" -m pip install --index-url https://pypi.org/simple \
  --no-build-isolation -r configs/authoring/infinigen_cpu_b11700e.txt
"$CPU_ENV_PATH/bin/python" -m pip install -e .
"$CPU_ENV_PATH/bin/python" -m pip check
```

Use the [official Extra Mesh Objects 0.3.12 archive](https://extensions.blender.org/download/sha256:c0708c26e4d7a5fd9f3dd6dadc86288f2270c338ba1c984ba0f37e6098f7756d/add-on-extra-mesh-objects-v0.3.12.zip),
SHA256 `c0708c26e4d7a5fd9f3dd6dadc86288f2270c338ba1c984ba0f37e6098f7756d`.
Its roughly 120 KB download is a GPL-2.0-or-later Blender extension, not a new
benchmark asset library. Verify the checksum and unpack it under
`$CPU_ENV_PATH/blender_extensions/blender_org/extra_mesh_objects`.
The runner fingerprints every extension file and enables the existing local
extension before importing factories, with online installation disabled by the
kernel sandbox. Its real rock operator was registered successfully in CPU-only
Blender. Do not substitute a fake factory or remove the natural shelf trinket.

No terrain/CUDA compilation was run. The source remains at
b11700ebac77695d8f53538705cfdc0cd890c54d with pinned OcMesher and GPL submodules;
sparse checkout now includes required Python/Gin/JSON/NumPy/text/script resources.
These are small source resources, not a new asset-library download.

Latest ordinary CPU run: **585 passed, 38 skipped in 16.32 s**; the additional skip is the explicitly configured real-SDK test.
The CPU environment with optional kinematics completed **627 passed, 14 skipped
in 100.19 s**. Neither is physical
acceptance. The explicit real-SDK contract test also passed (**1 passed, 2.98 s**):

```bash
FINDINGBENCH_INFINIGEN_ROOT="$IG_ROOT" \
FINDINGBENCH_INFINIGEN_CPU_PYTHON="$CPU_PYTHON" \
  "$TEST_PYTHON" -m pytest tests/unit/test_infinigen_upstream_cpu.py -q
```

The same test is an explicit skip in an ordinary environment without these paths.
It checks actual pinned initializer bounds for n=2/4 and constraint relevance in
all four primary greedy stages, inside the private-device sandbox. A bare global
Storage count is irrelevant to those stages; the implemented kitchen-related
count participates. `clamp_min(0)` preserves the exact nonnegative count while
avoiding an overlapping initializer Bound. Extra/indirect locations remain
subject to the complete metadata postfilter.

## Actual generation record — 2026-10-14

The final diagnostic command was the CPU runner above with `--seed 217` planned
spec `task_2b44dce1a4d44afdc2c0.json`, scene seed recorded in `input.json`,
`--solver-profile fast --timeout 900 --retries 0`, output:
`build/household/infinigen/rectification217/run_kitchen_count`.
It finished the real upstream main in **307.74 s**; the runner took **311.861 s**.
Actual files are `attempt_0/geometry/{scene.blend,solve_state.json,findingbench_geometry.json}`.
The roughly **181 MiB** Blender scene has two actual Storage candidates; the
full mapping is `storage_0 -> 15823_KitchenCabinetFactory`,
`storage_1 -> 708944_SingleCabinetFactory`.
The production postfilter called the DP and interval bins on these real AABBs:
expected movement proxy **6.5**, operations **5.0**, bins **low/low**. Assumptions
and target-blind negative-result order are stored in `attempt_0/measurement.json`.
This is a declared discrete-model proxy, not an actual robot route or final grade.

The scene is **rejected**: final cabinet-factory count, lighting, island placement
and rug constraints are not satisfied. The countertop is also recorded as a
possible additional search location pending cavity classification. No failed
position or background object was dropped, and no validation was disabled.
The current fast solve preset has not produced an accepted background. More
solver quality work is needed before scaling; do not infer a production success
rate from development runs using different bridge/configuration versions.

Across this rectification work there are **4 planned spec files, 2 source seeds
and planned base-layout IDs, 5 actual geometry outputs, 0 accepted backgrounds,
0 exported benchmark scenes, 0 compiled real episodes, 0 simulation-verified
or evaluation-ready episodes**. The five raw outputs include repeated-seed
implementation variants and are not five independent structures. They are local
provenance/debug artifacts, not runnable benchmark data. No training ran.
There were **11 completed development invocations** (5 completed geometry,
6 earlier failures) and **1 user-interrupted invocation**. Seed 216 completed
1/6 geometries; seed 217 completed 4/5; accepted families were 0 for both. These
cross-version diagnostics are not generator performance estimates. Every run
used a finite timeout and zero automatic retries.

Earlier failures/logs are retained under `rectification216/{run_v1,run_v2,run_v3,
run_fast,run_fast_bounded,run_addon}` and `rectification217/run`.
They show initial greedy-bound coverage error, premature stage rejection,
timeouts, the missing real Blender extension, and a user interruption.
The missing-extension run's old taxonomy incorrectly matched an earlier
"unsatisfied" warning; the current classifier uses the final exception.
Raw solver tags are `Semantics(kitchen)` etc, not `Semantics.Kitchen`.
The real output exposed and fixed this parser mismatch. Canonical tags are now
normalised while raw strings are preserved, without importing Blender or eval().
Real Blender/Tf identifier checks also fixed leading digits and slashes in source
names; all ambiguous sanitisation collisions are rejected.

The first two geometry outputs were wrongly marked invalid by the old parser.
Their original receipts remain unchanged. Current independent CPU inspections
verify the files and reject their complete candidate sets (4 and 3 instead of 2):

```bash
$TEST_PYTHON scripts/tasks/household_pipeline.py inspect-infinigen-geometry \
  --spec build/household/infinigen/rectification216/specs/task_f228ad3548b99f7a2d26.json \
  --receipt build/household/infinigen/rectification216/run_addon_resumed/generation.json \
  --report build/household/infinigen/rectification216/geometry_inspection.json
```

Exit 2 means rejected. Successful upstream execution and real geometry count do
not grant candidate acceptance. Changed artifacts reject inspection. The final
rejected scene emits no accepted export command; a later authorized GPU diagnostic
may inspect it, but cannot release it. Replan legacy specs lacking the source
layout split group before binding. Asset-held-out generalisation and a global
physical-source split audit remain outside the currently accepted scope.


The final CPU acceptance contract is intentionally strict: the two-location
proxy calculation ran on a real scene, but there is still no accepted candidate.
The family publisher and Infinigen public-skill worker are now implemented
(commands below), but have not run on a real simulator. Collision representation for combined
architecture AABBs and decrypted BEHAVIOR dependency graphs need runtime work.
GPU probing, appearance images, native freeze/reset and actual search costs are
implemented only in part or as explicit later entries; none has been executed.
Training and scaling are not part of this CPU round.

## Deferred commands and acceptance boundaries

**Do not execute these GPU stages during this CPU-only round.** After real
geometry succeeds, the pinned export command is emitted only by a successful `run/generation.json` receipt; its input is the actual successful `attempt_N/geometry` folder.
It uses `python -m infinigen.tools.export --input_folder ... --output_folder ...
-f usdc -r 512 --omniverse`; run it only in a later GPU-authorized session.

Once a real export, its `solve_state.json`, and explicit real cabinet bindings
are available, the following two stages are CPU-only:

```bash
$CPU_PYTHON scripts/tasks/household_pipeline.py inspect-infinigen \
  --scene "$SCENE_USD" --solve-state "$SOLVE_JSON" --report "$EXPORT_REPORT"
$CPU_PYTHON scripts/tasks/household_pipeline.py prepare-infinigen \
  --spec "$SPEC_JSON" --scene "$SCENE_USD" --solve-state "$SOLVE_JSON" \
  --generation-receipt "$ACCEPTED_RUN/generation.json" \
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
The Infinigen input adapter is implemented and CPU-tested. Legacy dependencies
were the three BEHAVIOR layout PNGs, room-category naming, scene_json floor lookup
and the z=0 spawn assumption. The new entry uses real exported floor triangles,
geometry start proposals and the same SceneProber/production runtime checks.
No real probe has run. The configured `build/robots/r1pro.urdf` is currently absent
on this host; export it in the later GPU session before preparing the request:

```bash
# Later GPU session ONLY, existing articulation exporter:
PYTHONPATH=src /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/export_robot_kinematics.py --robot r1pro --scene Beechwood_0_int \
  --out build/robots/r1pro.urdf

# CPU preparation after a real export, assembly and URDF exist:
$TEST_PYTHON scripts/tasks/household_pipeline.py prepare-infinigen-probe \
  --spec "$SPEC_JSON" --assembly "$ASSEMBLY_OUT/assembly.json" \
  --assets-root /data1/ygwang/codes/BEHAVIOR-1K/datasets/behavior-1k-assets \
  --out "$PROBE_INPUT_OUT"

# Later GPU session ONLY:
PYTHONPATH=src /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/household_pipeline.py gpu-probe-infinigen \
  --request "$PROBE_INPUT_OUT/request.json" --out "$PROBE_OUT"
```

The request remains pending and supplies no overlay. All candidates must have
OPEN/GRASP feasibility, successful real placement, closed invisibility and
ungraspability, sufficient open visibility and a valid configured head camera.
Any failed candidate rejects the whole family; only a complete live prefilter
writes `overlay.yaml` and its hash-bound `evidence.json`. These prefilter
measurements do not certify actual public skill execution or search difficulty.
The new `gpu-certify-infinigen-member` worker reuses TaskCertifier, this
GeometryRoomMap and overlay. The legacy `certify_tasks.py` CLI still loads
BEHAVIOR maps. Do not route pending
Infinigen candidates through that CLI or the legacy promoter.
Combined architecture mesh AABBs may conservatively block every route; the
runtime must reject that case until its actual collision representation is
adapted, never bypass the footprint gate. Encrypted BEHAVIOR resource graphs
remain pending until decrypted runtime dependency inspection. The entry is
implemented but unexecuted, with these physical/interface risks still pending.

Once that probe supplies a complete overlay and matching receipt:

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
rejects perturbations; it is not the final family verifier. It now saves a
separate state and manifest for every member. The runtime restores these states
instead of sampling placements again. Reference search measurement and measured
bins are implemented but unexecuted; oracle execution is only a feasibility
witness. Layout-grouped dev/test is implemented, not a claim of unseen assets or
task combinations. No release path accepts these pending bundles.

## 2026-10-14 delivery and remaining acceptance work

Scope is still one kitchen/direct-container family and the existing R1Pro head
RGB protocol. No GPU, Kit, rendering, training or large asset downloads were
started. This is a CPU-tested data module, **not an accepted experimental dataset**.

Final ordinary CPU validation: **613 passed, 38 skipped in 20.60 s**, using the
fresh non-system-site-packages test environment and `pytest tests/unit -q`.
Skips retain their optional kinematics/local-asset reasons and are not passes.
Log: `/data1/ygwang/codes/_external/findingbench_delivery_cpu_final.log`.
New snapshot/probe/publication fixtures are explicitly synthetic tests in pytest
temporary directories; their simulated receipts are never counted as real data.

New fixes and regression evidence (`tests/unit/test_household_dataset.py`,
`test_backend_lifecycle.py`, `test_infinigen_pipeline.py`):

| Finding | Implementation | Regression |
|---|---|---|
| Reveal anchors depended on the actual hidden position | Infinigen compiler includes every probed reveal anchor in every member | Distinct reveal poses, identical full anchors across target positions |
| Freeze state was discarded when loading an episode | Persist topology-compatible per-member states; SceneSpec's evaluator-only `frozen_member_manifest` is loaded by builder/reset | No placement resampling; restore after full import; changed state/background and object order reject |
| Compiled files could change after binding | Complete artifact ledger for plans, scenarios, specs and overlay; revalidate asset approvals and inherited split | Changed/missing/extra artifacts and non-default split tests |
| Assembly could bypass a rejected geometry run | CLI requires generation receipt; match admitted solve hash and complete candidate mapping; recheck at validation | Rejected/unmatched/missing geometry cannot qualify; successful retry selected by its actual solve hash |
| Report-required success cannot be reached by the legacy oracle interface | Evaluator-only oracle clone ends at holding; public witness independently executes REPORT_DONE | Original public termination remains unchanged |
| Known-target witness could be mislabeled search difficulty | Separate target-blind nearest-uninspected policy, actual public calls including TURN/OBSERVE/failures, complete prior-weighted family summary | Asymmetric prior/trace, costs, missing members, changed order, OBSERVE bypass and pending physical checks |
| A single certificate could release a family | Recompute gates, full member set, reference order/costs and required checks; immutable local publisher/reloader | Any necessary failure/pending check blocks the whole family, output remains absent |

The new seed **218**, source seed **3274909325** (`c333268d`), split seed **17**,
has two specifications and one requested source layout. One real CPU generation
attempt completed in **279.484 s** (upstream main 265.189 s), with zero retries:

```bash
export PYTHONPATH=src
TEST_PYTHON=/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python
CPU_PYTHON=/data1/ygwang/codes/_external/infinigen-cpu-b11700e/bin/python
IG_ROOT=/data1/ygwang/codes/_external/infinigen-indoors-b11700e
$TEST_PYTHON scripts/tasks/household_pipeline.py plan-infinigen \
  --seed 218 --split-seed 17 --difficulty-profile low_low --count 1 \
  --out build/household/infinigen/delivery218/specs --upstream-root "$IG_ROOT" --python "$CPU_PYTHON"
$TEST_PYTHON scripts/tasks/household_pipeline.py run-infinigen-cpu \
  --spec build/household/infinigen/delivery218/specs/task_1af27e855d2da59fb856.json \
  --upstream-root "$IG_ROOT" --python "$CPU_PYTHON" \
  --out build/household/infinigen/delivery218/run --timeout 900 --retries 0 --solver-profile fast
$TEST_PYTHON scripts/tasks/household_pipeline.py infinigen-status \
  --root build/household/infinigen/delivery218
$TEST_PYTHON -m pytest tests/unit -q
```

Real output: `build/household/infinigen/delivery218/run/attempt_0/geometry/`
contains `scene.blend`, `solve_state.json`, `findingbench_geometry.json`.
`attempt_0/measurement.json` records two SimpleBookcaseFactory sources, E[movement
proxy]=7.0, E[operations]=5.0, low/low proxy bins, but violations of cabinet count
(2), kitchen counters (1) and rugs (1). Receipt is **geometry_rejected**, accepted
backgrounds/exported scenes/compiled real episodes/simulator-verified episodes/
evaluation-ready episodes are **all zero**. No candidate was pruned or constraint
disabled. The pinned upstream `min_distance_internal_impl` returns 0 for sets of
size <=1; the mandatory rugs separation constraint therefore remains a blocker
for those scenes. This delivery does not reinterpret that constraint as passed.

The inventory entry checks actual files and hashes, deduplicates by content,
and reports failures rather than counting configurations. Across the local
development root it finds 18 distinct historical spec IDs, 3 source-seed groups,
4 requested base-layout IDs and **6 real geometry outputs** (including hashed
reinspection of the two older tag-parser failures), with **0 admitted outputs**.
Those repeated implementation variants are not six independent source layouts
or a production success-rate estimate. Historical invalid/incomplete receipts
remain separately reported. For today's run use the narrower `delivery218` root.

### Later commands, not executed in this CPU-only delivery

After a real admitted generation, export, assembly, full probe, binding and
freeze (earlier commands), prepare an immutable validation request:

```bash
$TEST_PYTHON scripts/tasks/household_pipeline.py prepare-infinigen-validation \
  --family-root "$FAMILY_OUT" --frozen-root "$FROZEN_CHECK_OUT" \
  --python /data1/ygwang/miniconda3/envs/behavior/bin/python --out "$VALIDATION_OUT"
```

`commands.json` contains exact argv arrays, one process per complete family
member. Smallest explicit **later GPU** worker invocation:

```bash
PYTHONPATH=src /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/household_pipeline.py gpu-certify-infinigen-member \
  --request "$VALIDATION_OUT/request.json" --episode "$COMPILED_EPISODE_ID" \
  --out "$VALIDATION_OUT/results/$COMPILED_EPISODE_ID"
```

Run every listed member. Worker saves before Kit shutdown. It reuses the
execution witness and oracle, then resets to the same frozen member and runs a
separate reference search with no target-informed ordering/routes/viewpoints.
One submitted public call costs one step: MOVE+TURN are spatial; OPEN, CLOSE,
OBSERVE, GRASP, PLACE and REPORT_DONE are operations. Rejected calls count too.
The reference checks accepted closed-container OBSERVE views for target pixels.
Detection only consults the target after it appears in the current frame; this
is an evaluator-grounded reference policy, not an MLLM result or an optimality
proof. Dynamic routes and coupled reveals are measured rather than forced into
the static CPU model. All outcomes retain the planned positive target prior.

CPU final verification and conditional local publication:

```bash
$TEST_PYTHON scripts/tasks/household_pipeline.py verify-infinigen-family \
  --request "$VALIDATION_OUT/request.json" --results "$VALIDATION_OUT/results" \
  --report "$VALIDATION_OUT/family_acceptance.json"
# Only a fully validated family can execute this successfully:
$TEST_PYTHON scripts/tasks/household_pipeline.py publish-infinigen-family \
  --request "$VALIDATION_OUT/request.json" --results "$VALIDATION_OUT/results" --out "$DATASET_OUT"
$TEST_PYTHON scripts/tasks/household_pipeline.py verify-infinigen-dataset --dataset "$DATASET_OUT"
```

Local publication includes accepted scenarios, public task text, inherited split,
reference difficulty and acceptance receipts. It keeps local assembly/state/asset
references; it does not redistribute BEHAVIOR assets. Changes to implementations,
URDF/camera/skill configuration, approved priors, textures/layers, frozen states or
member evidence invalidate reuse and require the applicable stages to rerun.

**Still not implemented as complete automated acceptance:** replacement-support
proof, furniture-to-background door sweep, usable temporary placement space,
wall/texture appearance application and image/semantic review, and classification
of every additional counter/drawer search cavity. These checks remain explicitly
pending in the live worker; the family publisher requires all of them and cannot
currently release a family from that worker alone. Opaque decrypted BEHAVIOR
resource dependency inspection and combined-room collision representation also
remain runtime risks. No CPU fixture, oracle result or boolean candidate label
substitutes for these checks. GPU-executable probe/freeze/worker code is implemented
but **unexecuted**. Therefore today's strict release acceptance is still pending.
