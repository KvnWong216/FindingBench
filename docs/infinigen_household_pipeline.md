# Infinigen household pipeline — 2026-10-15

Current release status: **no validated family, zero evaluation-ready episodes,
no training and no push**. The actual closed kitchen has 21 source cells / 32
source locations. Its explicit single-joint replacement variant has 21 native
regions; the previous mixed 55-region proposal is retained separately. All 96
objects loaded. Repeated loads of one saved checkpoint reproduced the complete
observed state, all native object states and geometry exactly twice, with
separate checkpoint-input and observed-world hashes audited on CPU.

The old full-scene receipt recorded a wrist prim although RGB selection used
the head camera; that receipt is rejected by the corrected camera-identity gate.
Default pose/prim, private modalities and RGB now share one head sensor, and a
sole known wrist camera is refused. A real R1Pro diagnostic confirmed
`robot_0:zed_link:Camera:0`, `/zed_link/Camera`, and matching default camera pose.
No wrist-only paper-protocol acceptance is claimed.

The full-scene initial RGB was visually rejected: 61.529% of pixels were clipped
despite passing coarse numeric QC. The temporary whole-body slots report four
occupied and 17 unknown; these are not verified per-region target placements.
Complete native-region family binding, frozen-member reconstruction inputs,
target insertion invariance, full skills and measured difficulty remain pending.
Limited import / camera diagnostics cannot release benchmark data.

Final software check for the checkpoint/camera changes: **742 passed, 37
optional skips, 13 simulator tests deselected in 70.10 s**. The old full-scene
camera metadata cannot pass the new parent exit/status gate; successful numeric
QC cannot override a rejected image review. `pipeline_inventory` counts these
failures separately from generated geometry and never increases ready episodes.
The actual selected ZED identity and default camera pose were verified on GPU 4
in `build/diagnostics/head_camera_identity_gpu4/result.json`. The protected
checkpoint input/output hashes, `.pt` artifact hash and recipe metadata were
also rechecked on CPU with CUDA disabled.

## Current entry points and difficulty contract

### Whole-kitchen nested components, 2026-10-15

Latest iteration: real full-scene import failed while settling the closed
background. Initial native FCL measurements found up to 78.7 mm penetration
between a replacement cabinet and the original sink, and 29.4 mm with the
adjacent replacement. The same real cabinet, floor and robot stayed finite
both at the proposed anisotropic scale and at native scale in separately
labelled partial-scene diagnostics; those diagnostics are not benchmark data.
The upstream generator cuts its own cabinet mesh around the sink. Generic
replacement assets lose that cutout, so source-level sink parameters now
explicitly request Width 0.4–0.45 m, Margin 0.02–0.03 m and the existing
front-coplanar relation with 0.18 m margin. Original protected policies still
replay without these optional controls. No collision mesh is disabled.

The new source proposal `iteration219_closed` completed real CPU generation
in 863.404 s and contains 32 source locations across 21 cells, including
countertop and upper units. It uses the existing upstream `drawer_only=True`
option (despite its name, this samples both doors and drawers) to request a
closed-container controlled experiment before generation. Previous open cells
remain in their original families. The source seed group and split are shared
with earlier variants. `--binding-profile single-joint-containers` explicitly
proposes only approved native models with one annotated openable link; it does
not claim source drawer/cardinality preservation or live joint validation.
Every actual loaded native region must replace the old prior, with no dropped
body or region. On the previous mixed-asset import, the observed native prior
was 54 annotated regions plus one retained source open cell, superseding 33.

The topology receipt now exports real native fillable-link types and region
visual AABBs. Those AABBs are proposal envelopes. The existing placement path
can carry an exact native fillable-link name in `PlacementSpec.link`; a thin
adapter changes only the native Inside sampler's link selection, retains all
its original collision/stability/settling checks, then checks native selected
volume membership and rolls back on failure. Region OPEN requirements and
reveal coupling remain pending real measurements. No candidate may compile
through the pending component leaf-binding gate.

CPU verification before the latest receipt/region regression additions:
**726 passed, 37 skipped, 13 simulator tests deselected**, 69.36 s.
The additional focused region and Inside tests passed 10 cases; dataset
tests passed 36 cases. These are software checks, not physical acceptance.
Dataset status now reports native-import preflight passes/failures and exact
failure phases separately; neither count contributes evaluation-ready episodes.

Latest complete CPU rerun after region selection, native-region occupancy,
strict SceneProber sampler success, mapping recomputation and receipt counting:
**732 passed, 37 skipped, 13 simulator tests deselected in 70.48 s**.
`verify_component_preparation` now recomputes the deterministic asset mapping
from the exact split geometry, approved slots and hashed model metadata; valid
asset hashes cannot hide a dropped component or changed mapping. The real
21-body preparation was reverified successfully. The existing SceneProber
uses the builder's exact membership predicate for a selected meta volume,
requires sampler success as well as final relation, and preserves the legacy
physical-link path. Native occupancy uses the loaded region envelope and
native membership predicate when such a link is explicit; intersecting
unresolved geometry remains unknown. AABB/volume budgets never certify fit.

Current integration boundary: source census/splitting/export and all-body
builder import are connected. Complete native per-region **family compilation
is still blocked**: HouseholdTaskSpec's original 2–8-location source proposal
and generic one-slot-per-body assembly must not be repurposed as a 32/55-region
family. A verified derived native family, per-region SceneSlots/SceneProber,
upper-fixture mounting evidence and all-member freeze/search checks remain
required. No schema bypass, fabricated overlay or single-task certificate is
used to cross that boundary.

Real GPU diagnostic `build/diagnostics/native_region_gpu4/result.json` passed
all three distinct fillable regions of the actual `qudfwe` asset, using the
real source floor, R1Pro and whitelisted target. Each sample passed the native
Inside checks and selected-region membership; membership in the other two
regions was false. Native RigidPrim names have an object prefix, unlike links
dictionary keys; selection now resolves the actual primitive name correctly.
This partial-scene diagnostic grants no family or task acceptance.

The closed 21-body whole-kitchen import remained finite after all CLOSE calls
and 30 physics steps, then correctly failed spawn clearance. Offline inspection
identified the original, unmapped `/World/skirtingboard_support` mesh: its
whole-house AABB covered the kitchen although its actual geometry was trim.
Unmapped static meshes now use the same real triangle-clipping navigation
representation as walls, in both CPU pose proposals and public MOVE checks.
No collider or triangle is removed. Source mesh coordinates are clipped to the
actual base band; uncertain polygon tessellation still rejects. Triangle
clipping is batched, and runtime results may be cached only while all live link
frames, object scale and link scales remain unchanged. The CPU start remains
the same 1.6 m-clearance proposal; a fresh GPU preflight is required.

The final CPU rerun including skirting, exact clipped-triangle envelopes,
pose/scale cache invalidation and private spawn diagnostics passed
**736 tests; 37 optional skips; 13 simulator tests deselected in 69.84 s**.

| Finding / impact | Fix and regression evidence |
| --- | --- |
| Joined nested cabinets were outside the standalone prior. | `infinigen_components.py` records actual upstream cell faces and each drawer; real pinned KitchenSpace geometry/UV preservation and dropped-drawer rejection tests in `test_infinigen_components.py`. |
| A multi-slot parent could resolve to its first drawer. | `household_binding.resolve_family_slots` requires unique explicit IDs and rejects aliases; unit test covers two drawers of one parent. |
| Native Inside sampled its first fillable region; primitive and dictionary names differ. | `inside_volume.py` selects the resolved native primitive; source AST equivalence, wrong-region rollback tests and the real three-region GPU diagnostic. |
| SceneProber compared physical-link and volume names and ignored sampler failure. | `probe.py` / `probe_support.py` reuse builder membership and require sampler success; rejected-sampler and wrong-region regressions. |
| Source meshes disappeared when their sole render mesh became a guide collider. | `infinigen_adapter.py` keeps exact separate visible and physics meshes; real USD regression. |
| SDK collider approximation and floor/wall/trim envelopes corrupted physics/navigation. | Verified source collider restoration, floor-relative clipped live geometry and pose/scale-bound cache; real USD and common MOVE/authoring footprint regressions. |
| SDK shutdown could exit zero during an exception. | `backend.close` preserves exceptions; CLI requires a durable complete limited-preflight receipt; zero-exit failure regression. |
| Hashing assets did not protect an incomplete component mapping. | Recompute every asset proposal from split geometry and exact approved inputs; dropped-body regression and actual 21-body recheck. |
| Family import failures could vanish from dataset status. | Deduplicated native preflight failure phases counted separately, never as accepted episodes; receipt counting regression. |

Minimal fresh whole-kitchen import check for the current protected artifacts:

```bash
FB_RUN=build/household/infinigen/iteration219_closed
FB_PARTS="$FB_RUN/components_single"
/data1/ygwang/miniconda3/envs/behavior/bin/python scripts/tasks/household_pipeline.py \
  gpu-preflight-infinigen --gpu 5 \
  --spec "$FB_RUN/specs/task_51966f3de38d406e5828.json" --policy "$FB_RUN/policy_run.yaml" \
  --component-preparation "$FB_PARTS/components_preparation.json" \
  --assembly "$FB_PARTS/assembly/assembly.json" --start "$FB_PARTS/assembly/start.json" \
  --out "$FB_PARTS/fresh_gpu5_import"
```

Use a fresh output directory. `progress.json` and console phases expose long
import/close/settle stages. A pass requires all five limited-preflight checks,
two exact native restores, finite physics, legal spawn and configured head RGB.
Closed settled topology and its new entire native prior are separately hashed
in the receipt; pre-close geometry remains a diagnostic. Full family release
still requires every separate pending check listed above.

The fresh run `components_single/gpu5_static_navigation` passed the corrected
spawn gate, all 21 native interaction-interface queries and closed-background
settling. Its settled native topology has **21 annotated regions** (the explicit
single-joint asset variant), so its new prior supersedes the 32-location source
proposal; the original source and previous mixed-asset 55-location proposal are
retained. It then **failed exact native restore** with `native source state
mismatch`. No tolerance or comparison key was relaxed. The current dataset
inventory therefore reports one real geometry output, one verified export,
21 proposed interactive bodies, two rejected limited preflights, zero compiled
candidates and zero evaluation-ready episodes for this run root. Source task
specification files are candidates, not evaluation samples.

Future preflights retain `captured_inventory.json`, `captured_native_state.pt`
and `restore_failure.json` before returning a restore error. These are diagnostic
artifacts, distinct from the final twice-verified native snapshot; the latter
is never written when either restore check fails. The full chain remains
unaccepted until exact restore and all family gates pass. Remote main was
rechecked read-only at `2fe9266e233686a45c0e915f53f9028769354b99`; no push.

The real partial-scene restore diagnostic
`build/diagnostics/native_restore_gpu4/result.json` completed five consecutive
round trips with identical full serialized hashes, zero changed scalars and
zero changed object states, including actual R1Pro. This does not explain or
excuse the whole-scene restore failure. `--diagnose-restore-roundtrips 5` now
optionally records bounded follow-up SDK round trips after a failed whole-scene
restore; the original exception is always re-raised and no record is accepted.
The full-scene diagnostic is `components_single/gpu5_restore_diagnostic`.

That diagnostic completed and rejected the first restore. Its protected
`captured_inventory.json`, `captured_native_state.pt` and `restore_failure.json`
show 25 changed pose scalars in 14 background objects; the maximum absolute
change was 9.5367431640625e-7. Five further SDK round trips still changed 23,
18, 19, 17 and 16 scalars respectively, with the same maximum displacement;
this is not a converged restorable state. Eight changed objects had position
differences and seven had orientation differences, with one in both groups.
These are measured numerical changes, not fabricated instability
or a licensed-asset absence. Source collision fidelity and sampler success do
not excuse the restore mismatch. Full-family compilation and certification
remain blocked, and no numeric tolerance has been introduced.

The real counterpart `build/diagnostics/native_restore_all_cabinets_gpu4/result.json`
contains all 21 replacement cabinets, the original floor and R1Pro, and passed
five byte-identical full-state round trips. Together the two diagnostics narrow
the issue to the additional native background state; they do not certify the
entire scene. Future failed preflights also retain the restored `.pt` and
bitwise scalar differences, including signed zero. A unit regression proves
that even a pure signed-zero hash mismatch is still rejected and cannot create
the final verified `native_state.pt`. The raw hash and object/geometry checks
are unchanged. No GPU job modifies or stops another user's processes.

The follow-up `--restore-contract same-checkpoint` explicitly distinguishes
the SDK reconstruction input A from its actually observed world B. It saves A,
loads it once, refreshes spatial queries without stepping physics, and captures
the complete inventory/state/geometry of B. Both repeated checks must load the
same immutable A and exactly reproduce B. A freshly dumped B is never substituted
for A. This is a deterministic reconstruction contract, not an assertion that
dump/load is numerically idempotent. The old check is still available explicitly
as `--restore-contract serializer-identity`. No tolerance, rounding, omitted
state key or changed state hash is used in either contract.

For this versioned contract the final input artifact is `native_checkpoint.pt`,
with separate input-state and observed-world hashes. The inventory is version 2;
it is not a legacy snapshot that can bypass pending component-family binding.
Mutable deserialization buffers receive a copy of A, and input mutation is
rejected. Unit tests exercise A != B with exact repeatability, rejection when
the next reconstruction drifts, and preservation of the original checkpoint
when a test deserializer mutates its supplied buffer. The real full-scene
experiment is `components_single/gpu5_same_checkpoint`; its receipt determines
whether this contract actually passes on the installed SDK.

The confirmed instruction scope is **all kitchen cabinets and drawers**. The
standalone count in the old sampling policy is a source-generation proposal,
not the final search set. The upstream KitchenSpace factory creates genuine
bottom and upper cabinet cells, then joins their geometry into one mesh.
`infinigen_components.py` instruments the real component API before that join;
it adds face provenance without consuming RNG or changing geometry/materials.
Each drawer layer remains a distinct location. Native open cells are retained
with pending initial visibility rather than silently excluded.

One real rerun completed in 12m54.87s:
`build/household/infinigen/iteration219_complete/runs/family_50bb1ef97bcb29c8692d/run/`.
It contains two standalone units and two KitchenSpace units, **21 actual cells,
32 closed cavities and one open cell: 33 locations**. Its new prior is uniform
over those 33 locations and explicitly supersedes the original two-location
family. The source layout split is unchanged. Original requested difficulty
is not carried forward as certified difficulty.

`prepare-infinigen-components` uses Blender's real mesh separation operator
and verifies the world-space face/UV/material multiset before and after.
It saves a recoverable derived blend and an explicit component relation graph,
leaving original source artifacts immutable. All 20 closed cells fitted exact
approved easy_v1 models within the existing [0.5, 2] scale limits; this checks
dimensions and metadata only. The open cell remains native geometry. GPU 5
then completed the pinned exporter: a 215.25 MiB USD, with all 21 cells mapped
exactly and no ambiguous source mappings. Missing noncandidate source mappings
remain recorded; this does not certify scene-wide semantic completeness.

`prepare-infinigen-component-assembly` passes all those real interactive assets
through the existing assembly/builder input. Its private manifest retains the
complete 33-location distribution. Asset-body slots are **not** drawer cavity
bindings. It has no family generation admission, and `bind_family` explicitly
rejects it with `COMPLETE_KITCHEN_LEAF_BINDING_PENDING`. The separate import
preflight cannot release tasks or count as complete family validation.

One OPEN may reveal several drawers. Under the explicitly hypothetical zero
travel, perfect per-component OPEN+OBSERVE reveal model (OBSERVE for the open
cell), the target-blind operation order gives E[ops]=20.4545, worst=43. The
old independent-location OPEN+OBSERVE model would give E[ops]=36, worst=68,
exceeding the 48-call horizon before movement. Neither calculation measures
real search difficulty or proves a physical bound. Real reveal sets, movement,
turning and every submitted operation must be measured before final partition.
The current low-operation sampling profiles cannot label this complete kitchen.

Actual floor/wall USD parsing also exposed collapsed boolean-wall triangles.
Navigation now retains their clipped conservative envelopes as obstacles;
it never removes that collision evidence. The real kitchen has floor top
z=0.118377 m and 70 wall envelopes in the base band. This is a geometry check,
not a collision, reach or skill certificate. GPU startup also now launches
OmniGibson before reading assembly USD: Isaac's `pxr` is supplied by the running
Kit app. CPU parsing continues using standalone OpenUSD without Kit.

Further real import checks found two SDK-specific issues: imported collision
meshes are hidden as `guide`, so each background asset now has separate exact
visible and collision meshes; fixed source collision approximations are restored
by the SDK post-import callback before first play. Lighting and palette edits
now run inside `og.sim.editing_usd()` for USD/Fabric synchronization. The
reassembled bundle is `components_v2/assembly_visible/`; earlier assemblies
and failed GPU logs remain immutable diagnostics. The native-only preflight
declares the real target to satisfy ScenarioSpec and returns before task import.

Latest complete CPU check: **718 passed, 37 skipped, 13 simulator tests
deselected**, 67.18 seconds. This includes real pinned upstream asset provenance
and geometry-preserving separation tests; their unit assets are not benchmark
scenes. GPU component import, actual RGB quality and physical family acceptance
are reported separately, never inferred from this test count.

Reproduce the new component stages from the protected real generation (use
new output directories; conflicts are rejected):

```bash
FB_CPU=/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python
FB_IG=/data1/ygwang/codes/_external/infinigen-indoors-b11700e
FB_BPY=/data1/ygwang/codes/_external/infinigen-cpu-b11700e/bin/python
FB_RUN=build/household/infinigen/iteration219_complete
FB_SPEC=build/household/infinigen/iteration219_sized/specs/task_c29b629456a3bd55ba46.json
FB_ASSETS=/data1/ygwang/codes/BEHAVIOR-1K/datasets/behavior-1k-assets
$FB_CPU scripts/tasks/household_pipeline.py prepare-infinigen-components \
  --spec "$FB_SPEC" --policy "$FB_RUN/policy_run.yaml" \
  --receipt "$FB_RUN/runs/family_50bb1ef97bcb29c8692d/run/generation.json" \
  --upstream-root "$FB_IG" --python "$FB_BPY" --out "$FB_RUN/recheck_components" \
  --assets-root "$FB_ASSETS" --approved-slots build/tasks/scenes/*.yaml
$FB_CPU scripts/tasks/household_pipeline.py export-infinigen-components \
  --preparation "$FB_RUN/recheck_components/components_preparation.json" \
  --upstream-root "$FB_IG" --python "$FB_BPY" --resolution 64 --gpu 5 \
  --out "$FB_RUN/recheck_components/export64"
$FB_CPU scripts/tasks/household_pipeline.py prepare-infinigen-component-assembly \
  --spec "$FB_SPEC" --policy "$FB_RUN/policy_run.yaml" \
  --preparation "$FB_RUN/recheck_components/components_preparation.json" \
  --export-receipt "$FB_RUN/recheck_components/export64/export.json" \
  --assets-root "$FB_ASSETS" --out "$FB_RUN/recheck_components/assembly"
```

The existing GPU preflight additionally accepts `--component-preparation`.
For the existing actual assembly, its minimal import check is:

```bash
/data1/ygwang/miniconda3/envs/behavior/bin/python scripts/tasks/household_pipeline.py \
  gpu-preflight-infinigen --gpu 4 --spec "$FB_SPEC" --policy "$FB_RUN/policy_run.yaml" \
  --component-preparation "$FB_RUN/components_v2/components_preparation.json" \
  --assembly "$FB_RUN/components_v2/assembly_visible/assembly.json" \
  --start "$FB_RUN/components_v2/assembly/start.json" \
  --out "$FB_RUN/components_v2/recheck_gpu4_visible_import"
```

It must retain finite physics state, check every loaded articulation/handle,
restore the same native state twice, and obtain the configured R1Pro head RGB.
Per-drawer cavity identity, OPEN reveal sets, collision/mounting/door sweeps,
target insertion invariance and complete-family search remain separate required
checks. Current evaluation-ready episodes remain **zero**. No training or push.

### GPU 4/5 iteration, 2026-10-15

Changes remain local; no push until an actual complete family passes. Real
isolated CUDA checks on both GPU 4 and GPU 5 saw exactly one device with the
expected UUID. The whole CPU suite passed 696 tests, with 37 optional skips
and 13 simulator tests deselected (32.98 seconds). In the existing `behavior`
environment, 11 focused replacement support/FCL tests passed; those use
explicit synthetic CPU colliders and do not count as household data.

Replacement placement now converts source bbox centers to BEHAVIOR root
poses using the actual metadata `base_link_offset`, scale and orientation,
matching the installed DatasetObject API. Missing offsets reject proposals;
live articulation acceptance also checks the resulting bbox center. The
new fixed-fixture support check exports every enabled physics collider,
measures floor contact and all foreign penetration, and remains pending when
fidelity or contact information is unknown. It permits at most 2 mm foreign
penetration and requires basal floor contact within 5 mm. It is explicitly
limited to fixed fixtures; it does not certify movable-body stability. Coal 3 /
hpp-fcl 3 contact penetration uses signed distance, now verified against the
installed headers and real FCL queries. Family validation recomputes the
numeric evidence and rejects flags without complete collider-pair coverage.

Native source classification now includes the pinned jar, food-box, book and
nature-trinket factories. Unknown factories and actual target-category
duplicates still reject bare-category tasks. These are factory labels, not
visual semantic acceptance. Existing approved SceneSlots were regenerated
from the 14 scenes in `slot_rules_v1.yaml`, yielding 12 real kitchen cabinet
models available locally. This is an asset inventory, not generated scenes.

For the first runtime pilot, `plan-infinigen --appearance-profile lighting-only`
explicitly preserves source textures and colours while sampling the approved
lighting parameters. The default `requested` profile still records the full
appearance requests, including currently unsupported material swaps. Seeded
configuration/request equality is checked; changing an existing request to
silently omit a swap rejects it. Both profiles have the same background
generation request and split group; appearance is not structural diversity.

GPU 4 legacy diagnostics are retained in
`build/diagnostics/easy_v1_gpu4/` and local logs outside the repository:
`_external/findingbench_gpu4_easy_open_reset.log` had 2 reset passes and 2 OPEN
failures, followed by a Kit teardown exit 139. The durable diagnostic shows
NAV was refused because the old hand-authored anchors were unvalidated, so
OPEN ran from the living room and was unreachable. An experiment enabling
all those anchors caused nonfinite joint positions (4 setup errors), and that
factory change was removed. The public interface and refusal gates are
unchanged. These tests do not validate an Infinigen family or justify bypassing
anchor probing. The generated pilot must use its real SceneProber overlay.

The bounded new generation is
`build/household/infinigen/iteration219/runs/family_850435734a9086a69991/run/`:
seed 219, scene seed `0xaf858eec`, high_low request, two cabinets, `kitchen`
profile, two CPU cores, zero retries and a 2100-second deadline.
`geometry/populated_storage_refinement.json` records the expanded cabinets
passing all five storage checks with zero repair attempts. Final geometry,
binding, probing and complete family acceptance are separate downstream gates.

The current user instruction authorizes **physical GPUs 4 and 5**. Each job
still exposes only its selected card; the default is GPU 5. The historical
CPU-only rounds below describe their original executions. New GPU entry points
resolve PCI bus / UUID to the actual NVIDIA and DRM device nodes, expose only
that device with bubblewrap, and reject other physical GPU indices before
launch. Kit uses isolated local GPU 0 and disables its machine-wide P2P test.
On this host physical GPU 5 is `/dev/nvidia7`, PCI `0000:a1:00.0`; this mapping
is discovered, not hardcoded. A real CUDA allocation on this isolated device
passed with exactly one visible CUDA device. No training is started.

The source integration reuses the existing easy_v1 direct-container TaskPlan,
compile_plan, SceneSlots, SceneProber, builder/reset, TaskCertifier and visual
skills. Infinigen creates geometry and household props. Exact kitchen cabinet
models from existing easy_v1 SceneSlots are proposed as interactive BEHAVIOR
replacements; metadata dimensions rank eligible fits, not the target position.
No new generator, robot skill library, success rule or automatic asset download
is introduced. All candidates must map and subsequently pass live probing.

Difficulty is two dimensional, with one cost for **every submitted public
call**, including rejections. MOVE + TURN form the spatial cost; OBSERVE,
OPEN/CLOSE, GRASP/PLACE and REPORT_DONE form the operation cost. For a complete
family the reported cost is the prior-weighted expectation over all target
positions, accompanied by each outcome and the worst case. The reference
order is computed from the common start and all candidate anchors before
consulting target visibility. Real dynamic routes use the existing skills.
This is a **measured reference cost**, with no claim of real-scene optimality.

| Final reference cell | Expected spatial calls | Expected operation calls |
|---|---:|---:|
| low_low | [0, 8) | [0, 6) |
| high_low | [8, infinity) | [0, 6) |
| low_high | [0, 8) | [6, infinity) |
| high_high | [8, infinity) | [6, infinity) |

These are explicit protocol budget boundaries, **not empirically calibrated
thresholds**. Calibration is pending until representative complete families
have actual traces. The bins cannot prove MLLM performance ordering.
Furniture-center Euclidean CPU costs are labelled proxies; they omit TURN,
valid interaction poses and traversable paths. Requested cells guide proposals.
Final measured cells may differ; only validated complete families fill quotas.

The live reference policy now tries OPEN from the current RGB, uses the returned
OPEN/MOVE image to detect and grasp the target, and invokes OBSERVE only if it
needs another view or must recover a failed OPEN. No closed/open OBSERVE pair
is forced on every cabinet to manufacture a high-cost cell. Its policy version
is `nearest-uninspected-anchor-adaptive-rgb-v2`; older policy evidence is rejected.
The CPU proposal model separately assumes OPEN+OBSERVE per location, then
GRASP+REPORT_DONE. Under its fixed single-reveal assumptions, uniform priors
and no extra operations, `E[operations]=n+3`: n=2/3/4 gives 5/6/7; worst cases
are 6/8/10. **These are assumed proxy costs**, not mandatory live costs. A live
OPEN/GRASP/REPORT_DONE trace with no extra OBSERVE is explicitly regression
tested. Every visited candidate still requires a successful OPEN; missing
OPENs or a target revealed from a different candidate reject the controlled
family. Coupled reveals need a different model.

The household experiment uses an explicit **48-call exploration horizon**;
the existing easy_v1 dataset retains its own 16-call horizon. This leaves room
for real navigation and recovery rather than truncating the last family member.
Requested cells whose declared proxy worst-case lower bound already exceeds
the horizon fail before generation (e.g. n=4, high spatial lower bound 8 and
proxy worst operation cost 10 exceed 16). This check is scoped to that declared
model; every actual member must independently finish within its public horizon.

Generation fixes also address the pinned SDK's storage initializer selecting
bookcases before kitchen cabinets, and optional pair-distance rules reporting
nonzero violations for empty/singleton sets. Only real KitchenCabinetFactory
objects qualify. Pair spacing is still checked for two or more objects; island
clearance is unchanged when an island exists. Raw original optional-rule
violations are saved alongside the declared domain corrections. All remaining
pinned kitchen constraints and FindingBench constraints must pass. The
`kitchen` profile changes stage proposal budgets, not collision predicates or
the final acceptance thresholds. CPU processes are pinned to two cores,
including Blender's internal thread pools.

The bridge now re-evaluates storage constraints immediately after upstream
`populate_intermediate_pholders`, before generating small props. Expanded
cabinet meshes can violate spacing even when their placeholders passed.
One bounded 200-step repair uses the existing solver's pose moves only;
addition, deletion and asset resampling are disabled. Inventory changes or
movement of non-candidates reject the run. Fresh checks must pass before
continuing, and the full final kitchen gate still applies. Failed repair
evidence is saved and hashed even when no geometry output qualifies. Real
Blender/FCL fixture regressions pass; this repair still needs a new complete
household simulation run. It does not certify collision, stability or skills.

Publication is on hold at the user's request: retain changes locally until
one real background, complete task family and R1Pro skill/judgment chain pass
end to end on an isolated physical GPU 4 or 5. CPU tests and a diagnostic export do not meet
that condition. After this repair the full CPU run passed **677 tests**, with
34 skips and 13 simulator tests deselected; a further focused regression
checks failure evidence retention and rejects tampered resume artifacts.

```bash
CPU_PYTHON=/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python
IG_PYTHON=/data1/ygwang/codes/_external/infinigen-cpu-b11700e/bin/python
IG_ROOT=/data1/ygwang/codes/_external/infinigen-indoors-b11700e
PIPELINE=scripts/tasks/household_pipeline.py

# Candidate batch only. Start with count=1; expand after a real family passes.
$CPU_PYTHON $PIPELINE plan-infinigen --seed 217 --split-seed 17 \
  --start-index 0 --count 1 --difficulty-profile low_low \
  --python "$IG_PYTHON" --upstream-root "$IG_ROOT" --out build/household/pilot/specs
$CPU_PYTHON $PIPELINE batch-infinigen-cpu \
  --batch build/household/pilot/specs/batch.json --out build/household/pilot/runs \
  --python "$IG_PYTHON" --upstream-root "$IG_ROOT" \
  --solver-profile kitchen --timeout 1800 --retries 0

# Export only a completed proxy-admitted receipt; actual pinned CUDA texture
# baking and USD export are confined to GPU 5. No Kit or Torch dependency here.
$CPU_PYTHON $PIPELINE export-infinigen --spec "$SPEC" --receipt "$RECEIPT" \
  --python "$IG_PYTHON" --gpu 5 --resolution 512 --out "$EXPORT_ROOT"

# Exact existing cabinet pool, using the original CPU slot extractor.
$CPU_PYTHON scripts/tasks/build_scene_slots.py --scenes Beechwood_0_int
$CPU_PYTHON $PIPELINE propose-infinigen-bindings --spec "$SPEC" \
  --scene "$EXPORTED_SCENE" --solve-state "$SOLVE_STATE" \
  --generation-receipt "$RECEIPT" \
  --assets-root /data1/ygwang/codes/BEHAVIOR-1K/datasets/behavior-1k-assets \
  --approved-slots build/tasks/scenes/Beechwood_0_int.yaml --report "$BINDINGS_JSON"
```

`SPEC` is an actual task JSON named in batch.json; `RECEIPT` is the actual
family run/generation.json; `SOLVE_STATE` is from its successful attempt;
`EXPORTED_SCENE` is export.json's `scene`. A rejected binding report cannot
prepare an assembly. Feed the complete binding report to the existing
prepare-infinigen command below, then prepare-infinigen-probe,
gpu-probe-infinigen, bind-infinigen, gpu-freeze-infinigen and family validation.
All these GPU commands default to `--gpu 5` and also accept `--gpu 4`.
The exporter reuses the pinned upstream export code; only four obsolete
Cycles tile-size hints absent from Blender 4.2 are removed. A real-source CPU
AST check verifies that every geometry, bake and export operation is retained.
An actual GPU 5 diagnostic export of the rejected seed-217 geometry completed
texture baking and wrote a 648 MiB USD with 140 meshes. This tests the exporter
interface only; it is not an admitted scene or a physical certificate.
The real file exposed two mapping bugs now covered by CPU regressions:
same-name Xform/Mesh wrappers, and solver rooms ending in `.meshed` while their
sibling floor/wall parts retain the base room name. Reinspection uniquely maps
70 solver objects, both cabinets, and the kitchen floor/wall/ceiling/exterior.

For expansion, plan the next immutable batch with a new output directory,
increased `--start-index`, and a finite count (maximum 10,000 families per
invocation). Run generation serially; exact-input checkpoints reject changes
to seeds, sources, runtime or outputs. Interrupted attempts consume the finite
retry budget. No command converts requested counts into validated counts.

```bash
$CPU_PYTHON $PIPELINE infinigen-status --root build/household/infinigen
$CPU_PYTHON $PIPELINE infinigen-difficulty --root build/household/infinigen \
  --target-per-cell 100 --report build/household/infinigen/difficulty_inventory.json
$CPU_PYTHON -m pytest -m 'not sim' -q
FINDINGBENCH_INFINIGEN_ROOT="$IG_ROOT" FINDINGBENCH_INFINIGEN_CPU_PYTHON="$IG_PYTHON" \
  $CPU_PYTHON -m pytest tests/unit/test_infinigen_upstream_cpu.py -q
```

Difficulty inventory recomputes family evidence, reports measured and validated
counts separately, and ranks missing cells for subsequent sampling. It does
not silently relabel a request, prune targets or pool incompatible thresholds.
Final CPU suite with the real pinned SDK configured: **676 passed, 34 skipped,
13 simulator tests deselected in 29.54 seconds**. Focused export/asset/family
checks: **129 passed**. Skips are optional
kinematics/assets, not physical acceptance. The initial unfiltered pytest run
had 13 simulator-fixture errors because this CPU environment has no OmniGibson;
its log is retained, and no simulation-test pass is claimed.

Known physical blockers still require explicit work/acceptance: replacement
support/contact now has a live checker and requires actual family execution;
semantic image review remains pending in the live certifier;
support/container texture swaps are requested but unsupported, and ambiguous
native target-category props are preserved and reject bare-category tasks.
An exported USD or a registered robot cannot bypass these checks. Historical
R1Pro export produced 38 links/37 joints, with 50 live FK comparisons passing
(max translation 1e-6 m, rotation 0.0001 degrees); this is kinematic evidence,
not household camera/skill acceptance. Derived robot meshes and BEHAVIOR
assets remain gitignored and local.

Current actual artifacts are in `build/household/infinigen/protocol217/`.
CPU generation used scene seed `0x43f4e503`, `kitchen` profile, zero retries,
1800-second timeout and cores `[0,1]`; it finished in 1433.751 seconds.
`run/generation.json` records rejection: final storage separation violation,
and expected spatial proxy 8.5 outside the requested low interval. This run's
earlier three-call proposal model is preserved in its source/input snapshots;
it is not reused as a receipt for the final adaptive policy. Raw output is
`run/attempt_0/geometry/{scene.blend,solve_state.json,findingbench_geometry.json}`.
Diagnostic export is `diagnostic_export64/export.json` and
`diagnostic_export64/export_scene.blend/export_scene.usdc`; original receipts
are unchanged, with corrected inspection saved separately. Its Cycles device
was `CUDA_NVIDIA RTX 5880 Ada Generation_0000:a1:00` on physical GPU 5.
No GPU job from this iteration remains active.

The inventory across historical rounds and this run reports **7 actual geometry
outputs across 3 upstream source-layout groups, 1 diagnostic exported scene,
0 proxy-admitted scenes, 0 compiled families, and 0 evaluation-ready episodes**.
These are separate stages, not a claim of seven usable households. Three
interrupted historical runs and all failed/rejected receipts remain recorded.
Mass generation is bounded and resumable, but mass **accepted** data is still
blocked by scene constraint yield and the pending physical checks above.

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
  5/6/7 in the declared OPEN+OBSERVE proposal model. Low [0,6) uses n=2; high [6,infinity) uses n=3/4; live adaptive costs are measured separately. Unreachable request
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

At the first delivery, door sweep, temporary placement and wall application were
not yet implemented; the next iteration below implements scoped checks without
claiming simulator acceptance. Replacement-support proof, complete texture
application, semantic image review and opaque decrypted dependency inspection
still require work. The publisher cannot release a family with pending checks.
No CPU fixture, oracle result or boolean candidate label substitutes for them.

## Follow-up iteration: physical checks and room-component integration

The fixed main line remains pinned Infinigen + existing OG builder, one direct
kitchen family and R1Pro head RGB. No GPU, Kit, rendering or training was run.
No new geometry scenes were generated in this iteration; experimental episode
counts remain zero.

CPU result for this iteration: **650 passed, 40 skipped** (ordinary suite,
including the live-body-pose guard). Optional real hpp-fcl subset:
**2 passed**, no GPU or simulator imports. Skips remain optional kinematics and
explicit local-asset checks, not passes. Logs are
`/data1/ygwang/codes/_external/findingbench_iteration_cpu_final.log` and
`/data1/ygwang/codes/_external/findingbench_iteration_real_fcl_final.log`.

| Change / implementation | Acceptance scope and limits | Regression evidence |
|---|---|---|
| `household_physics.py::verify_door_sweeps` | Collects every enabled collision prim, actual fully-open joint endpoints and robot/background bodies; validates rigid one-joint kinematics before certifying continuous clearance | Clear endpoints with mid-path collision fail; incomplete interval cover and exhausted query budget cannot pass; **real CPU hpp-fcl** rotating-door tests |
| `usd_collision.py::collect_entity_collision`, `sim/base.py::WorldCollisionObject` | Disabled collision prims are omitted; enabled prim path and PhysX approximation are recorded, allowing missing collider exports to reject verification | Actual standalone OpenUSD collider schema with disabled/enabled prims; synthetic geometry used only in test |
| `verify_temporary_place` | Executes public MOVE/TURN/OPEN/GRASP/PLACE on a real support, checks production release/support relation, actual OnTop contacts, seven poses over 30 physics steps and final velocities; resets/releases grasp then restores original frozen state | Drift/rotation/velocity/contact and missing public-trace evidence reject; failed route restores unit-test world |
| `household_locations.py` | Enumerates actual source ancestry and every mapped/plausible kitchen storage source; loaded joints/fillable links identify additional candidates; static unannotated counters stay **unknown** | An extra drawer rejects the whole distribution; no reweighting/pruning; static counter absence of joints does not prove no cavity |
| Room sibling parts in `infinigen.py::inspect_export/source_mesh_owner` | Exact pinned `_floor/_wall/_ceiling/_exterior/_meshed` exported mesh names retain solver room ownership; arbitrary prefixes do not. Floor sampling uses the actual floor sibling | Real OpenUSD sibling floor/wall fixture; unrelated prefix ignored; actual floor triangles generate proposals |
| `household_appearance.py::apply_wall_palette` | Changes a private, composed USDPreviewSurface network for explicitly mapped wall meshes, retaining table/object materials, asset resolution and geometry/physics attributes | Shared textured wall/table network: only wall tint changes; geometry digest identical; unsupported shader stays pending |
| `household_appearance.py::image_quality` | Checks real RGB contract, nearly blank/black/clipped images and luminance range; records **numeric QC only** | Blank/dtype/shape reject; passing numeric QC still has semantic review pending |
| Family verifier | Recomputes interval proof, placement trace/contact/stability and complete search-location inventory; `checks: passed` alone cannot qualify these gates | Missing physical proof with all summary flags still set passed rejects publication |

The door certificate is bounded to 1,025 distance queries. Each midpoint distance
must exceed the 5 mm margin plus a conservative rigid-body motion bound for the
entire interval; safe endpoints alone are insufficient. Only tested exact
primitive/convex-hull/triangle collision representations are accepted. Missing
colliders, proxy/fallback geometry, unsupported cooked convex decomposition,
bidirectional OPEN, nonrigid descendants and multi-joint motion remain pending.
The scope is door-to-foreign-background/robot clearance; same-cabinet hinge/frame
contacts and the task target are excluded explicitly. This is neither a general
articulated trajectory planner nor a certificate of unsupported PhysX geometry.
Evaluator sweep checks restore Python/NumPy random state as well as world state.

Temporary PLACE requires actual top-surface support, six positive OnTop contact
samples and a stability window: position drift <=5 mm, rotation <=2 degrees,
final linear speed <=0.01 m/s and angular speed <=0.02 rad/s. Surface area and
AABB proposals never establish success. This trace is a separate feasibility
witness; its trials do not inflate or alter the reference search cost. Failure
or exhaustion cannot yield usable placement space.

CPU inspection after a real admitted assembly:

```bash
export PYTHONPATH=src
$TEST_PYTHON scripts/tasks/household_pipeline.py inspect-infinigen-search-locations \
  --assembly "$ASSEMBLY_OUT/assembly.json" --report "$ASSEMBLY_OUT/search_locations.json"
# Ordinary CPU suite, including standalone OpenUSD:
$TEST_PYTHON -m pytest tests/unit -q
# Optional actual CPU collision queries; no simulator is imported:
PYTHONPATH=src /data1/ygwang/codes/_external/findingbench-cpu/bin/python \
  -m pytest tests/unit/test_household_physics.py -m kinematics -q
```

The existing later `gpu-certify-infinigen-member` command now runs these checks
on the loaded frozen member, records detailed evidence and restores the member
between the reference search and temporary-placement witness. Its command is
unchanged. Do not execute it in this CPU-only session.

Still pending: actual GPU execution; supported replacement contact/support proof;
MDL/unknown wall shaders and texture swaps; semantic/full-scene image acceptance;
unknown static counter/drawer cavities; cooked convex decomposition, multiple
joint or bidirectional door paths; runtime decrypted dependency inspection and
combined-room collision handling. These are explicit limits, not passed checks.
# Complete native-region pilot and analytical search model

The pilot now reads its own `assets/tasks/tiers/household_direct_v1.yaml`.
It reuses execution and certification code without copying the easy contract.
The physical task-class name does not determine difficulty. Its exact native
candidate count, source horizon and complete contract/config hashes enter the
derived proposal and are rechecked by the compiler and family verifier.

The explicit search model and its proven closed-form subcase are in
[household_search_formalization.md](household_search_formalization.md).
The ratio rule applies only to declared separable, static travel with one
perfect reveal per inspection. Real kitchen strata still use complete-family
public-action traces; neither this rule nor known-target oracle costs certify
the scene's search difficulty.

For a component assembly, keep the source manifest immutable and use the
unified entry point to bind every actual imported fillable link:

```bash
python scripts/tasks/household_pipeline.py bind-infinigen-native-regions \
  --assembly RUN/assembly/assembly.json \
  --preflight RUN/gpu_preflight/preflight.json \
  --out RUN/assembly/assembly_native_regions.json
python scripts/tasks/household_pipeline.py derive-infinigen-native-family \
  --source-spec SOURCE_SPEC.json --policy SOURCE_POLICY.yaml \
  --assembly RUN/assembly/assembly_native_regions.json \
  --generation-receipt SOURCE_RUN/generation.json --out RUN/native_family
python scripts/tasks/household_pipeline.py prepare-infinigen-probe \
  --spec SOURCE_SPEC.json --policy SOURCE_POLICY.yaml \
  --assembly RUN/assembly/assembly_native_regions.json \
  --assets-root /LOCAL/behavior-1k-assets --out RUN/probe_request
```

`bind-infinigen-native-regions` requires an actual successful import preflight,
two exact checkpoint reconstructions and matching head-camera metadata. It
recomputes the full region mapping from hash-bound native topology. It retains
all asset bodies, source geometry and source open cells; an unresolved open
cell rejects binding instead of disappearing. A region's AABB remains only a
proposal envelope. Its explicit native fillable link enters the existing
SceneProber, builder and compiler. No first-drawer fallback is permitted.

The derived `household-infinigen-native-3` proposal can represent up to 128
locations. Original source versions retain their eight-location limit and
original policy gates. Native derivatives validate the original specification,
protected policy, generation receipt and original solve state, then recompute
their complete new prior and retain the original source split group. These
are candidates, not compiled tasks or evaluation data.

The minimum next GPU command uses the existing SceneProber over the whole set:

```bash
/LOCAL/behavior/bin/python scripts/tasks/household_pipeline.py gpu-probe-infinigen \
  --request RUN/probe_request/request.json --out RUN/probe_result --gpu 5
```

Acceptance requires every required slot's real sampler success, selected-link
membership, feasible OPEN/GRASP anchors, initial invisibility, causal reveal,
valid head camera and complete hash-bound overlay. Partial progress is saved
per furniture. A failed slot rejects the whole family. A zero SDK shutdown
exit without the complete verified overlay is a failure. Only physical GPUs
4 and 5 are accepted by the isolation wrapper.

Frozen member format v2 stores the immutable checkpoint input separately from
the expected hash of the reconstructed world. It composes the original native
object inputs with the placed target through the SDK's structured state API,
preserving actual object order. Target import changes are recorded and the
native background is explicitly reconstructed before insertion. Disturbance
from target insertion still rejects the member. Every saved member must
reconstruct the same full world twice, including the target and background.
Reset replays that original recipe without extra physics steps or robot
teleports. These new writer paths require live family verification; CPU tests
exercise the integrity contract only.

The pinned exporter's `adjust_wattages` applies a legacy Omniverse point-light
multiplier `20/(4*pi) * 1000/(4*pi*r^2) * 100`. The actual metre-scale,
normalized-light export retained this multiplier: a 59.3847 W source light
became intensity 557278592. Changing exposure alone left most of the initial
image clipped. Runtime import now removes precisely this source multiplier
once, before applying seeded appearance requests. It records each source and
normalized intensity; unknown source contracts and double application reject.
This is an explicit photometry adapter, not per-member image fitting. It does
not change geometry or physics, and does not grant image acceptance. Real RGB
and unchanged physics state must be checked after the conversion.
