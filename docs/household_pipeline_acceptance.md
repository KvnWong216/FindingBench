# Household pipeline acceptance — 2026-10-13

This checkout was clean on `main` at `1fda80d`; `git pull --ff-only` reported
already up to date. Neither `0304c04` nor `200f9e6` exists in the fetched Git
object database. No equivalent household modules were found in fetched branches
or the inspected local code directories. The shared conversation URL could not
be fetched. The cloud implementation has **not** been accepted.

## Missing baseline source

The following files must be supplied as patches, a source directory, or a
fetchable branch before continuing family integration. Existing versions of
other listed easy/task/Level-1 modules are present, but any cloud changes to
those modules also need the original diff.

- `src/rummagebench/authoring/tasks/{difficulty,families,family_binding,household}.py`
- `scripts/tasks/generate_families.py`
- `assets/tasks/household_{families,background,robots}_v1.yaml`
- `docs/household_{task_families,background_pipeline}.md`
- `tests/unit/test_household_{background,families}.py`

The requested 55-test command exits with code 4: the first household test file
is missing, and no tests run. That is a source blocker, not a passing test.

## Findings and changes

| Problem / location | Impact | Fix and CPU regression |
|---|---|---|
| `tasks/tiers/easy.py::_distractors` samples the actual target slot first and excludes it from ordinary background selection | Background draws, IDs, densities and role labels can depend on the answer | Reserve the same target budget for every semantically compatible slot and sample containers/surfaces uniformly with respect to the actual hidden location. `test_background_proposals_do_not_depend_on_actual_target_slot` moves a fixed target through all candidate drawers and compares complete proposals. This proves proposal invariance only. |
| `tasks/gates.py::REJECTION_CODES` omits `START_UNREACHABLE` | The planner reports this code, but constructing a validated `Rejection` for it raises | Add it to the shared taxonomy; test supported construction. Existing planner routing test already exercised the unvalidated planning path. |
| `tasks/split.py::leakage_violations` consumes iterators and compares two deduplicated lengths | Duplicate task IDs are missed | Materialize once; compare against original record count; report unknown and duplicate split IDs. Test lists and iterators. |
| `tasks/plan.py::git_state` ignores scripts/configs | Provenance can claim clean inputs after generator/config changes | Include both paths in the dirty check. Bump generator version for the changed easy policy. |
| `level1/snapshot.py::write_environment_snapshot` overwrites existing artifacts one at a time | A frozen environment can silently change or become a mixture of old/new files | Preflight all encoded artifacts, reject conflicts/partial bundles, serialize writers with an exclusive lock, publish a complete staging directory, allow identical reruns. Test conflict leaves every old file unchanged and invalid payloads publish nothing. This helper does not itself prove physical reset equivalence. |
| `scripts/discover_level1_robots.py` imports OG and looks for robot subclasses | Installed OG uses YAML models; old discovery can require simulator dependencies and return no usable model inventory | Read installed model YAML plus USD/USDA/USDC file presence without OG import. Output to `build/level1/robot_candidates.yaml`; never replace the reviewed pool. Synthetic inventory tests are isolated in temporary directories. |
| `level1/robot_registry.py::verified` trusts `status=gpu_verified` and an empty missing list | An unmeasured robot can enter production | Require all nine capability checks in a hash-bound evidence file, a current URDF hash, matching robot/camera identity and the current default public-protocol/skill implementation hash. Report every unverified profile as pending. Historical R1Pro label becomes candidate. Tests omit each check, mutate URDF, and test label-only claims. Evidence must be emitted by real measurement; the registry is not a measurement producer. |
| `level1/grammar.py` did not reserve the target pair before sampling distractors, and target draws could use reserved container/cover assets | Duplicate USD instances or a target equal to a reserved asset | Sample directly over remaining pairs; exclude container/cover pairs for target too; record target usage separately. Reject exhausted pools. Tests assert distinct pairs and excluded targets. |
| `scripts/generate_level1_candidates.py` tests for an undeclared `--full-catalog`; resume restores counts but not RNG/coverage usage; it writes provisional scenarios into repository `scenarios/` | CLI gate cannot be explicitly selected; continuation diverges; CPU/test assets can look like production scenarios | Declare the flag, bind catalog/config/source/seed hashes, reject changed inputs, replay the CPU prefix including failures, compare immutable candidate bundles, serialize generators, retain all outputs in the requested candidate directory. Add resumed-extension vs uninterrupted byte comparison and changed-input conflict tests. Existing test catalogs were enlarged to meet the unchanged distinct-instance constraint, and explicitly select CPU full-catalog proposals. No asset checks were weakened. |
| `scripts/tasks/promote_tasks.py` copies/deletes release artifacts before checking all certificates and treats an episode label as sufficient | Late failure can modify released robot files; a tagged family could inherit single-episode acceptance | Preflight every episode against current structural/simulator gates and candidate hashes before copying; reject family/background-marked records; reject conflicting frozen robots. Legacy manifests explicitly say single-episode execution/visibility, `family_certified=false`, search difficulty unmeasured. Test a tagged synthetic family is rejected before release mutation. This is a fail-closed barrier, not a household family promoter. |

The existing compiler retains the full native house: the simulator template
sets room/category filters to null and `load_task_relevant_only=false`. New
native capture enforces this and compares loaded entity IDs with the actual
scene JSON, including objects outside the search room. Legacy easy sampling
still does not add items to every room; that is the missing scene-wide cloud
pipeline's responsibility. Legacy medium still removes background from its target stack location and
constructs covers; it is not accepted as a matched natural-occlusion family.
No action schema, wrong-grasp termination or public success condition was changed.

## Implemented native occupancy boundary

`tasks/native_occupancy.py` runs on the existing `OmniGibsonBackend.setup`
**before** task imports, placement and initial-state door changes. It captures:

- Full native entities, per-object states, poses/velocities/articulation state
  as available from OG's own state dump, world AABBs, and positive actual
  `Inside` / `OnTop` relations, including nested containers.
- Actual loaded scene-file hash and local USD/encrypted-USD asset-source hashes, serialized source-state tensor hash including
  shape/dtype, SceneSlots hash, builder configuration hash, saved local snapshot
  hash, and a final inventory hash.
- `confirmed_empty`, `confirmed_occupied`, or `unknown` per slot. Errors, missing
  geometry, particle systems and unresolved membership remain explicit.

Inside is object-relative in this installed OG. A parent-level positive Inside
relationship cannot prove which drawer contains the object. Such drawer
occupancy stays unknown. Parent AABBs and a thin live support-plane envelope
are conservative exclusion tests. A confirmed empty support plane does not
prove clearance above it, opening space, collision-free insertion or stability.
AABB overlaps without positive relationships remain possible occupants. IDs
are deduplicated; volumes of nested objects are not summed as if disjoint.
`require_known_slots` rejects an unresolved requested candidate set as a whole.

`verify_source` rejects a changed inventory, state, scene/asset source, SceneSlots,
builder configuration, native entity set, restored AABB or per-object state. `background_mismatches` compares explicitly captured
background records without implicitly excluding any object IDs. It is a
contract helper for later insertion checks; background placement/freezing and
family restore are **not** implemented by that helper. Source snapshots include
the robot, so the current restore command requires the same robot/configuration;
it is not a cross-robot canonical background yet.

All capture and restore outputs remain evaluator-private. They are not added
to `ScenarioSpec`, agent observations, feedback or reference search hints.

## CPU commands and evidence

Use the already installed environment; do not install or download assets:

```bash
cd /data1/ygwang/codes/FindingBench
export PYTHONPATH=src
PY=/data1/ygwang/miniconda3/envs/behavior/bin/python

"$PY" -m pytest tests/unit -q
"$PY" scripts/discover_level1_robots.py \
  --assets-root /data1/ygwang/codes/BEHAVIOR-1K/datasets/omnigibson-robot-assets
"$PY" scripts/tasks/household_pipeline.py cpu
```

`cpu` exits **2** while the baseline files are missing; it reports the exact
blockers and stage statuses. Optional `--report PATH` preserves identical
reruns and rejects conflicting reports. Its native bundle counts require both
inventory integrity and the referenced snapshot file hash. Those counts are
measured-state bundles, not validated tasks.

Real local static inventory: **5 candidates** (R1Pro, R1, Fetch, Tiago, Stretch)
with model definitions, controlled-arm/base definitions and USD files. Static
inventory verifies no camera, IK, collision, holding or public skill chain.
Current code explicitly selects head/ZED RGB, and the pool labels R1Pro head
camera. There is no wrist-only certification or verified paper protocol in this
checkout. Retain the declared head-camera baseline until an actual protocol
conflict is supplied; do not silently relabel it wrist-only.

No household backgrounds/families were generated or physically accepted in
this run: **0 native state bundles, 0 family candidates, 0 validated families**.
Test fixtures and provisional JSON counts are excluded from that statement.

Validation record: the original task-focused CPU suite reproduced **92 passed,
8 skipped**. The first broad CPU run found two stale Level-1 test fixtures
(insufficient distinct models and synthetic categories excluded by the pilot
filter); the corrected fixtures preserve production constraints. Final full CPU validation: **540 passed, 15 skipped in 75.85 seconds**.
The focused native-occupancy/promotion/registry regression suite also passed
**25 tests** after asset binding was added. Skips are not acceptance.
No simulator, rendering, GPU job, asset download or original asset upload ran.

## Minimum later GPU native-state acceptance

These commands are **pending and were not executed**. They use the existing
easy/household scene source and R1Pro, with no new simulator backend. First
produce SceneSlots on CPU:

```bash
"$PY" scripts/tasks/build_scene_slots.py --scenes Beechwood_0_int
```

Then, only when GPU work is authorized:

```bash
OMNIGIBSON_HEADLESS=1 "$PY" scripts/tasks/household_pipeline.py export-native \
  --scenario scenarios/knife_search_001/scenario.yaml \
  --slots build/tasks/scenes/Beechwood_0_int.yaml \
  --source-scene /data1/ygwang/codes/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/Beechwood_0_int/json/Beechwood_0_int_best.json \
  --out build/tasks/household/native/Beechwood_0_int --seed 0

OMNIGIBSON_HEADLESS=1 "$PY" scripts/tasks/household_pipeline.py verify-native \
  --scenario scenarios/knife_search_001/scenario.yaml \
  --slots build/tasks/scenes/Beechwood_0_int.yaml \
  --bundle build/tasks/household/native/Beechwood_0_int --seed 0
```

Use a fresh output directory. A failed capture leaves `failure.json`, never an
inventory commit marker. Existing/partial bundles are refused, not silently
reused. `verify-native` restores the local snapshot twice using the existing
counterfactual-observation restore with **zero physics steps**, and checks exact
source hashes/configuration. Optional `--require-slots ID ...` requires every
intended candidate to have complete known occupancy and never drops one.
Acceptance requires both commands to succeed, no missing/extra native entities,
no invalid or changed source state, and known occupancy for the whole chosen
candidate set. Exact tensor mismatch is a failed restore check, not a reason to
relax comparison silently. `native_restore_checked` certifies only this restore
check, not household playability.

## Remaining work, explicitly pending

Receiving the cloud sources is necessary to review difficulty.py's declared
discrete model, family priors, common background sampling, split grouping,
input hashes and compiler/certifier integration. This work does not replace
those absent modules with a parallel schema.

After that source integration: place a complete life background once using
approved priors and the real asset whitelist; retry failed local placements
within a declared bound; verify collisions, stability, door sweeps and temporary
placement space; save a canonical restore state; insert each target only after
restoring it; reject any changed native/added background entity, pose or state.
Apply appearance and verify images/geometry invariance. Check each robot's
actual camera/control/feasibility/holding/public skills per scene, and retain
both common and robot-specific feasible subsets. Discover natural occlusion
and test OBSERVE bypass and multi-location reveals. All of these need physical
evidence and remain pending.

The legacy TaskCertifier explicitly uses target-known navigation/view selection
and an oracle for **execution/visibility** witnesses. It is not a search-order
reference planner. Candidate counts, clutter counts, and oracle execution depth
must never be published as measured search difficulty. A future search
measurement must use one target-blind policy over the full family, account for
MOVE/TURN/OBSERVE/OPEN/GRASP/PLACE/REPORT_DONE under the public protocol, and
report measured reference cost or justified bounds unless optimality under the
declared model is proved. Main space × operation bins remain unmeasured.

Legacy family-fallback splits share houses and are explicitly flagged, not
house-held-out data. The fixed duplicate-ID check does not establish the
missing cloud implementation's family/appearance/embodiment leakage policy.
All original house/layout/target-location/appearance/robot variants of a family
must later share a split. No symmetric-distance or fixed-reveal model was
applied to real scenes in this run.
