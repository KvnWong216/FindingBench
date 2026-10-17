# Native collision geometry in the existing Gen → OmniGibson pipeline

Follow-up: [observed-motion and contact-restore audit](background_observed_motion.md)
records the subsequent stability fix, real moving controls, and successful
known-target public-skill execution. The v5 failures below are historical and
have not been relabeled as passes.

The opt-in `--geometry` path replaces metadata boxes with the loaded native
scene's collision geometry in the existing InfiniGen `Solver.solve_objects`.
It keeps BEHAVIOR DatasetObjects, OmniGibson, the native placement sampler,
and the public skills and success conditions. The legacy envelope path is
still explicitly available without this flag.

## Reproduce one complete native home

Run from the repository root. Geometry and scene snapshots contain local
licensed asset derivatives: keep them under ignored `build/`, never publish
them with source code. Use a new output directory for every attempt.

```bash
CPU=/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python
SIM=/data1/ygwang/miniconda3/envs/behavior/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1

$SIM scripts/tasks/behavior_pipeline.py export-geometry \
  --config configs/layouts/living_brief_pilot/uniform.yaml \
  --seed 127 --gpu 5 --out build/geometry_trial/export

$CPU scripts/tasks/behavior_pipeline.py prepare-home \
  --config configs/layouts/living_brief_pilot/uniform.yaml --seed 127 \
  --geometry build/geometry_trial/export/collision_geometry.json \
  --out build/geometry_trial/home

$SIM scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario build/geometry_trial/home/scenario.yaml \
  --gpu 5 --realize-background --out build/geometry_trial/physical

$CPU -m pytest tests/unit/test_indoor_geometry.py \
  tests/unit/test_indoor_background.py tests/unit/test_behavior_native_adapter.py -q
```

Both simulator stages use the existing device/PCI isolation wrapper and
accept only GPUs 4 or 5. Export does not spawn a search target or read its
hidden placement. Four approved daily-asset prototypes are loaded remotely
from the home only to read their native geometry. No procedural replacement
assets or downloaded asset libraries are used.

## Contract and acceptance

1. `export-geometry` saves the actual loaded native state in the standard OG
   scene JSON format, retaining the original complete native inventory. It
   exports enabled collision prims using **live rigid-link transforms**, not
   stale USD body poses. Mesh `convexHull` uses the solid point hull with
   outward triangles; explicit triangle meshes and cubes are also supported.
   Cooked convex decomposition, SDF and unsupported shapes reject. A source
   `visual_only` object with no enabled collider is retained as explicitly
   noncolliding; missing collision data on other objects rejects. Active
   particle/system states currently reject because per-object restore would
   not restore them completely.
2. Source scene, saved state, native and daily asset metadata/USD hashes bind
   the export. The request/result and assembly bind that export in turn.
   Gen consumes real native support/obstacle meshes, including architecture,
   and mesh factories for the approved daily assets. Metadata remains useful
   for semantic support selection and the existing origin/front convention.
   The legacy occupied-envelope exclusion is not used in this mode. A lower
   support fully covered by another support envelope is still excluded as a
   documented proposal policy; this is not a claim of physical occupancy.
3. Gen's complete rigid transforms are round-tripped through the existing
   BEHAVIOR origin offset and front-axis conversion. The result's `native_poses`
   map carries position and quaternion for every generated object; the legacy
   yaw field is descriptive only in geometry mode. Tilt is preserved for
   subsequent physical validation. Shear, reflection and scale reject. Missing
   poses cannot silently fall back to yaw, and the manifest loader verifies
   the native object configuration against the hash-bound result.
4. The existing environment builder loads the saved native scene, restores
   its per-object state after environment initialization, compares all native
   collider topology and world vertices (maximum error 10 µm), and compares
   daily-asset local shapes with the prototypes. This prevents a different
   scene state or articulation shape from silently replacing the planning input.
5. The existing background realizer then checks native OnTop and the existing
   unchanged pose/velocity stability gates, performs bounded native placement
   retries, and writes the actual checkpoint only on success. Target insertion
   still receives the existing daily-prop disturbance check. Failure remains
   failure even if SDK shutdown exits zero.

`geometry_exported_pending_physics` means that geometry export passed.
`pending_simulation` means that Gen's layout constraints passed.
`support_and_stability_passed_pending_family` means that the existing limited
physical background check passed. `execution_smoke_passed` is a known-target
public-skill diagnostic, not a search certificate. All these stages keep
`evaluation_ready_episodes = 0`.

The exported hull/surface model does not reproduce PhysX cooking tolerances,
contact/rest offsets, mass, friction, controller dynamics or continuous motion.
Mesh layout feasibility cannot replace physical validation. Full native-state
matching after target insertion, door sweeps, matched-family restoration,
search measurements and final difficulty assignment remain separate required
acceptance steps. This change alone does not certify those steps or produce
new house architectures.

## Local integration record (2026-10-15–16)

Artifacts are in `build/behavior_geometry_loop/`. The first export attempt
failed scenario schema validation before simulation. The second captured 86
native collision objects plus four asset prototypes and rejected the native
visual-only carpet. That failure exposed a missing interface case; it did not
justify dropping an unknown collider. Version 3 explicitly verifies the
visual-only case and exports all 87 original entities, 1,129 enabled native
collision components, four asset prototypes and 142 bound input files.

The first mesh-based Gen solve completed all 800 steps in about 235 seconds,
placed all 14 requested daily objects and had zero constraint violations.
The initial yaw-only conversion rejected a transform; a full rigid-pose
adapter was then added, but the repeat also rejected non-unit scale. The
first error alone did **not** establish that tilt was the actual cause.
A separate 60-step diagnostic preserved raw matrices even on a failed solve.
It showed scale drift (for example `0.9999991059, 0.9999990463, 1.0`) and
orthogonality error around `1.94e-6`. Upstream `constraint_language.util.rotate`
repeatedly decomposes and reconstructs float32 matrices, including their scale;
its default move set does not intentionally resize these assets.

The geometry adapter now updates the native asset's rotation component
directly, keeping unit scale and the upstream Euler pose backup/restore
convention. It still synchronizes the upstream collision representation and
uses the same post-move checks. The final rigidity threshold stays unchanged.
The rejected attempts remain under `v3/home`, `v4/home` and `diagnostic/home`;
the 60-step run is diagnostic only. `v5/home` repeats the full 800-step budget.

Version 5 produced one valid CPU candidate: 87 native entities, 14 added props,
13 cabinet bodies, seven fulfilled support-count constraints. The maximum
matrix orthogonality error was `1.95e-7`, below the unchanged `1e-6` gate.
GPU5 then restored all 87 native entities with maximum world-vertex error
`4.768e-7 m`; all 14 added-asset shapes also matched their exported prototypes.
This demonstrates the geometry/state handoff on a real household.

The physical background stage **failed**. After seven bounded placement
repairs, all 14 support checks passed but eight objects exceeded the unchanged
`0.02 rad/s` angular-speed gate. The follow-up 361 samples over 12 simulated
seconds recorded no support loss; maximum positional drift was `5.93e-6 m`,
maximum rotational drift `0.00602°`, and maximum raw angular speed `0.07996 rad/s`.
Getters matched the raw PhysX velocity data. Several objects slept naturally;
the bowl on the console and the salt shaker on the console kept exceeding
the gate in 323/361 and 342/361 samples respectively. Waiting alone has not
resolved this case. Small net motion does not authorize bypassing that gate.

Target insertion, public skill execution, a certified frozen family and search
measurements were **not reached** in this trial. There is **one CPU candidate,
zero accepted physical backgrounds and zero evaluation-ready episodes** from
this integration run. The image is an R1Pro head-camera diagnostic, not a
camera-protocol or complete visual acceptance certificate.

Final CPU regression: **862 passed, 45 skipped, 13 deselected** with
`python -m pytest -m 'not sim' -q`. Separately, the installed CPU bpy environment
passed `tests/unit/test_indoor_geometry_blender.py`: 1,000 actual Blender
rotation updates retain unit scale and pass the rigid-pose check. This test
uses a synthetic triangle, not a production asset. Runtime-version metadata
and export-seed bookkeeping were subsequently hardened with CPU tests; the
historical v3 export is retained unchanged. The generation/physical source
snapshot is under `v5/sources`. No remote push is authorized yet.

| Finding | Code | Effect / regression |
| --- | --- | --- |
| Metadata boxes differ from loaded collision shapes | `indoor_background.solve`, `indoor_geometry.collision_mesh` | Geometry opt-in uses enabled live colliders; unsupported representations and incomplete inventory reject. |
| Original source JSON differs from loaded scene state | `native_scene_snapshot`, `restore_native_state` | Snapshot loaded state; compare restored collider topology/vertices; CPU source binding tests and real GPU restore. |
| A missing collider was confused with an explicitly visual-only carpet | `collision_mesh`, `load_geometry` | Preserve original visual-only object with source evidence; reject an unknown missing collider. |
| Yaw-only conversion cannot carry the solver's rigid pose | `native_layout_pose`, `behavior_adapter.object_config` | Full hash-bound native poses, no missing-pose fallback; vertex round-trip tests include tilted and asymmetric meshes. |
| Repeated matrix decomposition accumulates scale drift | `rigid_rotation_update`, native `iu.rotate` adapter | Assign rotation only before upstream collision validation; float32 and actual bpy repeated-rotation regressions. |
| Low drift still accompanies excessive reported angular speed | Existing `indoor_realization` gate | Remains failed with trace evidence; no physics override, forced sleep or threshold relaxation. |
