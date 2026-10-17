# Native BEHAVIOR generation interface

The current integration path uses BEHAVIOR-1K assets directly in OmniGibson.
The home entry retains a complete native house and calls the pinned InfiniGen
constraint solver to place daily objects on its existing tables and counters.
Model dimensions and root offsets come from installed asset metadata. Runtime
objects are `DatasetObject`s with original joints, collisions and object states.
No Blender export or InfiniGen mesh is required by this path.

For randomized multi-house batches, resumable generation, diversity accounting
and GPU sampling, see [the batch entry](behavior_randomized_batch.md).

`behavior_adapter.py` connects both generation entries to `build_env`.
`SceneSpec(source="behavior_layout", assembly_manifest=...)` loads native placements.
An ordinary BEHAVIOR scene without a manifest retains its existing behavior.
When `base_scene` names a BEHAVIOR house, its complete native scene is loaded,
including other rooms. Generated additions still require live collision and
support checks against the existing contents; CPU rectangles do not prove that.

The supplied `behavior_native_smoke.yaml` is explicitly an integration fixture:
two native cabinets, a table, two daily objects, and one real target on an
OmniGibson ground plane. It is not a complete household or a released benchmark
sample. All object geometry comes from the installed BEHAVIOR asset library.

## Run

生活情境驱动的可选输入与配对试验见 [living_brief_pilot.md](living_brief_pilot.md)。
它复用下列 `prepare-home` 与 `gpu-smoke` 入口，不改变技能和成功判定。

From the repository root, with the existing environments:

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/behavior_pipeline.py prepare-home \
  --config configs/layouts/behavior_home_v1.yaml \
  --seed 31 --out build/behavior_home/example31

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario build/behavior_home/example31/scenario.yaml \
  --out build/behavior_home/example31/gpu5 --gpu 5
```

Output directories must be new. `--assets` overrides the CPU asset root; the
OmniGibson runtime must use the same installed dataset. Only physical GPU 4 or 5
is accepted, with PCI/UUID device isolation. The parent checks the child receipt
as well as its exit code.

`prepare-home` invokes the actual upstream `Solver.solve_objects`, with
`StableAgainst` (`cu.ontop`) support relations, per-support count constraints,
surface-area fullness and separation objectives. The default native home is
`Benevolence_1_int`: all 87 native objects, non-search rooms and 13 kitchen
cabinet bodies are preserved. Seven tables/counters each receive two native
daily objects. All native non-structural occupants enter the solver as fixed
metadata envelopes, so new items cannot ignore known existing furniture.
The solve request contains no target entity or target location. The authored
robot start position comes from the native room map; its heading is a fixed
`start_yaw` (default zero), independent of the hidden target. The GPU diagnostic
separately labels its known-target interaction pose proposals.

This is a staged connection: native room architecture, cabinet rows, appliances
and furniture layout remain fixed. InfiniGen currently varies daily-object
placement; it does not yet resynthesize room architecture or relocate large
furniture. These variants count as **zero new structural layouts**. We reuse
InfiniGen's DSL/solver and its support/fullness formulation, not the entire
procedural-asset-specific `home_furniture_constraints` graph. Future furniture
layout variation must bind that graph to native model geometry explicitly.

Blender's temporary boxes are metadata-derived **planning envelopes**, never
exported visual assets. The final scene uses original native objects plus real
`DatasetObject` additions. The CPU solver runs in the existing sandbox without
GPU devices or network; missing Blender addons cannot trigger downloads.
Physical support is checked again in the live house. Approximate envelope
packing does not certify stability, full collision freedom, or door clearance.

Source scene JSON, native metadata, room maps and solver input/output hashes
are checked before loading. Every required support must have its own satisfied
constraint and exact item count; a correct total cannot hide an empty counter.
The loader binds the original scene JSON and checks that native house objects
were not lost. Existing task success conditions and public skills are unchanged.

The visual-protocol smoke sets the existing `expose_available_skills=False`
option. It does not use the legacy oracle action list. Without this setting,
after GRASP the legacy observation builder enumerates feasible PLACE actions
across the whole house, delaying the result even after grasp realization.
Every submitted pixel action still passes the same original validation gates.

For the earlier empty-floor integration fixture, use `prepare` with
`configs/layouts/behavior_native_smoke.yaml`; it remains a debugging fixture.

CPU preparation produces an assembly manifest and a `ScenarioSpec`. It rejects
missing model IDs, missing native joints for storage, malformed geometry,
insufficient placement space, changed asset files and transform mismatches.
Native +X fronts are rotated to the sampler's -Y access strips; geometry remains
at scale one. Root positions account for native bounding-box offsets.

GPU smoke uses the unchanged builder, target `Inside` check, renderer grounding,
Pinocchio/coal feasibility, `VisualProtocolSession`, and public pixel actions
`OPEN`, `GRASP`, `REPORT_DONE`. It records proposed start poses, feedback and RGB
images in `result.json` and PNG files. The evaluator knows the target and tries
start poses, so this is an execution diagnostic, not search-cost measurement.
Even `execution_smoke_passed` leaves `evaluation_ready_episodes` at zero. Full
household acceptance, complete candidate-set probing and family acceptance remain
separate requirements before dataset release.

## CPU regression tests

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python -m pytest \
  tests/unit/test_behavior_native_adapter.py \
  tests/unit/test_indoor_background.py \
  tests/unit/test_layout_generator.py tests/unit/test_scenario_schema.py -q
```

Synthetic metadata appears only in those unit tests. Production preparation
requires real installed asset metadata and USD files and never substitutes
missing models. Licensed assets and generated simulator states stay local.

## Measured integration result (2026-10-15)

`build/behavior_native/drawer31/gpu4_repeat/result.json` records a successful
real run on physical GPU 4, with normal process exit code 0. The historically
named output directory contains **hinged cabinets**, not sliding drawers:
`bottom_cabinet_no_top/vzzafs` has one native movable joint and one loaded
fillable region per instance. Two such cabinets, a `breakfast_table/bhszwe`,
two `plate/aewthq` background objects and a `tablespoon/pltxjb` target were loaded
at scale one. Background plates passed native `OnTop`; the target passed
native `Inside`. R1Pro used the existing head camera and Pinocchio/coal checks.

The first proposed interaction start produced this public trace:

| Call | Feedback | Episode status |
| --- | --- | --- |
| OPEN, current-image point | EXECUTED | RUNNING |
| GRASP, current-image point | EXECUTED | RUNNING |
| REPORT_DONE | EXECUTED | SUCCESS |

The preceding run at `drawer31/gpu4/result.json` produced the same three-call
success but crashed during interpreter teardown. Explicit existing-backend
shutdown fixed that runner issue; the repeat completed normally. The earlier
`seed31/gpu5_v2` cabinet (`bottom_cabinet/jrhgeu`) failed the bounded 25-pose
smoke: OPEN succeeded at several poses, but the low target was not visible for
GRASP. Its failure is retained; this does not prove that asset is unsolvable.
An optional bowl-target probe was stopped after the tablespoon repeat passed;
its `bowl31/gpu5/cancellation.json` labels it cancelled, not accepted.

CPU validation after the interface change: **792 passed, 37 skipped,
13 deselected**, using `pytest -m 'not sim' -q` with the installed upstream CPU
paths enabled. Focused native/layout/schema/legacy-pipeline checks: **144 passed**.

This demonstrates the native asset → OmniGibson → existing public skill → goal
judgement interface. It does not certify a full house, all target positions,
a shared frozen task family, or search difficulty. Released household episodes
remain **zero**. No training or remote push was performed.

## Complete-house constraint integration (2026-10-15)

The actual InfiniGen CPU solve at `build/behavior_home/pilot31_v3` retained
`Benevolence_1_int` and added 14 native bowls/plates across seven surfaces.
All seven per-surface constraints were satisfied. A second solve at `upper31`
used the same seed with a different target cabinet: its native context and
all 14 background placement records were identical. These are two diagnostic
members of one house, not two new structural layouts or a frozen family.

On physical GPU 5, `pilot31_v3/gpu5/result.json` confirms that all 87 original
house objects were retained and every added object passed actual `OnTop`.
All 13 kitchen cabinet bodies remain in the candidate list. Target `Inside`
passed; the bounded 25-pose smoke failed. OPEN executed at some poses, but
the low-cabinet target was not visible for GRASP. Other poses were rejected
for base collision. The unsuccessful member remains recorded.

The upper-drawer diagnostics `upper31/gpu4` and `upper31/gpu5_threads` were
cancelled while legacy observation construction enumerated PLACE actions.
Their `cancellation.json` files and `execution_smoke_cancelled` receipts preserve
that distinction; neither is an accepted episode or an infeasibility result.
A stack sample localized the work to `BenchmarkSession._finish_step` →
`available_skills` → `_place_approach`, motivating the existing observation
option above rather than any change to skill validation.

With that correction, `upper31/gpu5_public/result.json` and the fresh default
entry `final31/gpu4/result.json` both record **OPEN → GRASP → REPORT_DONE →
SUCCESS**, with normal process exit code 0. Each rejected five colliding base
proposals before the sixth proposal completed all three public actions. The
target was a native `tablespoon/pltxjb` inside
`bottom_cabinet_no_top_qudfwe_0/link_2`. Both runs retained all 87 original
objects and passed `OnTop` for all 14 additions. R1Pro used its head camera.
`final31/gpu4/06_GRASP.png` shows the real household and opened drawer before
grasp; `scene_start.png` records the fixed initial view.

This is **one complete native house**, two tested target placements (one
successful upper drawer and one unsuccessful low cabinet), and a successful
repeat of the upper-drawer execution. It is not multiple structural layouts,
an accepted full task family, or a certified search-difficulty stratum.
Released/evaluation-ready episodes remain **zero**. No training or push was run.

CPU regression after the home integration and target-independent initial-view
fix: **796 passed, 37 skipped, 13 deselected** (`pytest -m 'not sim' -q`, with
the installed upstream and CPU Python paths enabled). Focused home/native
interface tests: **10 passed**. Tests cover existing occupants and non-search
rooms, changed native-state hashes, per-surface rather than aggregate coverage,
and identical initial views when the hidden target moves between cabinets.

The currently covered constraints are support, per-surface coverage and
packing objectives within a complete fixed house. Long-term stability,
door clearance, shared frozen physical state, every nested search region and
target-blind search-cost measurement are still pending. Cabinet-body counts
must not be reported as an exhaustive nested-region catalog or search difficulty.
