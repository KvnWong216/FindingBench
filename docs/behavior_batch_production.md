# Native household environment production

The production boundary is **physical environments only**. InfiniGen proposes
living arrangements; native BEHAVIOR assets are instantiated in OmniGibson.
Baselines supply all policies. Production does not search for successful robot
poses, submit OPEN/GRASP/REPORT_DONE, or filter environments by policy success.
The old fixed-policy loop has been removed from `behavior_pipeline.py`.

## Current entry points

From this repository, reuse the existing 200 candidates without regenerating or
changing their target distribution:

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/produce_behavior.py \
  --out build/behavior_production_environment_v1 \
  --existing-batch build/behavior_production_20261016_v2/batch \
  --acceptance-policy execution-v1 --count 200 --gpus 4 5

/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/report_behavior_production.py \
  --run build/behavior_production_environment_v1
```

Do not launch a second coordinator in an active output directory. A new batch
can use the same producer without `--existing-batch`. `execution-v1` is the
historical name of the **motion acceptance profile**, not a request to execute a
policy: finite peak speeds are diagnostic, while measured drift, support loss
and invalid states still reject. The validation identity also binds
`environment_only: true`, sources, scenario and assembly hashes. Historical
policy runs are excluded from this production's environment counts.

A minimal GPU environment check (no policy):

```bash
/data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario <candidate-scenario.yaml> --out <fresh-output-directory> --gpu 5 \
  --environment-only --realize-background \
  --background-stability-profile observed-motion-v3 \
  --acceptance-policy execution-v1 --verify-background-restore
```

`gpu-smoke` now always checks environments; the flag is explicit documentation
and is enabled by default. `batch-probe` uses this same environment-only path.
Physical GPUs 4 and 5 are isolated by the existing device wrapper. No other
users' processes are controlled. No training is launched.

## Environment acceptance

1. Restore the source household geometry, preserve native objects, and realize
   added daily objects using the existing placement routines and bounded repairs.
2. Observe physical motion and support, serialize the background, perturb a prop,
   restore the disk checkpoint, and verify native/added poses and joints.
3. Insert the target and check its actual Inside relation. Check the resulting
   background's support and motion. In environment-only production, a stable
   supported prop displaced during insertion is recorded as a displacement;
   it does not imply that the resulting environment is unusable. This is **not**
   a matched-family certificate. Family callers retain pose matching by default.
4. Reset the existing public visual session at the scenario's declared spawn.
   Validate finite physics state, spawn clearance and renderer grounding. No
   target-dependent pose search or public action is submitted.
5. Recheck background support/motion after reset. Save and reload the full
   environment checkpoint including target, robot and spatial-predicate caches;
   validate physics, spawn and target relation, and capture the initial image.
6. Require clean process exit and hash-bound evidence. Missing evidence, crashes,
   corrupt artifacts or failed checks require review; they are not silently
   accepted or treated as proof that the environment is intrinsically impossible.

`environment_ready` means these bounded physical initialization checks passed.
It does not prove every door can open, every object is reachable, the task is
solvable, or a search cost is optimal. Those are distinct evaluation questions;
a baseline's performance is never this producer's acceptance predicate.

## Outputs and consumption

Each `batch/home_*/gpu_realized_<validation-id>/` contains:

- `background_realization.json`, `background_restore.json`,
  `background_after_target.json`: measured physical evidence and diagnostics.
- `environment_ready.json`: reset/spawn/target checks, `policy_executed: false`,
  image and full-environment checkpoint hashes.
- `environment_start.pt`: private full state in
  `observe-state-with-spatial-cache-v1` format. After building the **same** bound
  scenario/assets, load with `state_io.load_snapshot` and restore with
  `backend.restore_observe_state`. This is different from the background-only
  `restorable_background.pt`, whose target is parked outside the task.
- `environment_start.png`: initial camera image (R1Pro's existing head-camera
  protocol). No wrist-camera equivalence is claimed.
- `result.json`, `process.json`: source/input/stage bindings and completion.

The baseline harness owns restoration into its session and policy execution;
this change does not modify action semantics or success predicates. Checkpoints
and manifests are evaluator-side privileged artifacts, not model observations.
The full environment snapshot must not be passed to the model as input.

The live reporter gives `environment_ready`, `environment_needs_review`,
`environment_pending`, accepted IDs and per-environment evidence directories.
Legacy `execution_*` fields describe historical evidence only; new runs keep
them zero. `production_status.json` updates between validation chunks; the live
reporter reads completed receipts during the chunk. Old `completed_with_rejections`
statuses from policy runs must not be interpreted as environment yield.

The 200 existing candidates use six original houses. They are different living
arrangements/targets, not 200 newly generated floorplans. All remain development
samples. `evaluation_ready_episodes` and new floorplans stay zero: no matched
search-family, GT-depth, difficulty-bin or split certification is claimed.
BEHAVIOR assets, geometry, images and checkpoints stay local under ignored
`build/`; no raw assets are committed or uploaded.

## Verification and history

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  -m pytest tests/unit/test_environment_only.py \
  tests/unit/test_execution_acceptance.py tests/unit/test_behavior_production.py \
  tests/unit/test_behavior_smoke_receipt.py tests/unit/test_behavior_production_report.py \
  tests/unit/test_indoor_batch.py -q

/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  -m pytest -m 'not sim' -q
```

Unit fixtures are explicitly synthetic and never constitute production samples.
Regression tests forbid action submission during environment validation, verify
that no policy rollout is needed for acceptance, retain physical support checks,
and reject crashes, changed checkpoints/images and incomplete receipts.

The previous `build/behavior_production_execution_v1` coordinator was stopped
at the user's scope correction; its evidence is retained unchanged. It ran a
fixed evaluator policy and its success rate is not environment yield. Original
CPU candidates and previous strict-motion receipts are retained as well. Earlier
implementation details are in local commits `cfef7d9`, `f973c86`, `e4d49cb`.
No remote push is authorized until the user changes that instruction.
