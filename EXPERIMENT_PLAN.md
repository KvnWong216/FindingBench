# Acceptance gates and next experiments

This plan separates executable CPU checks from experiments requiring the
licensed BEHAVIOR dataset, OmniGibson and a supported GPU. Never reuse old
`build/` counts as evidence for a new commit. Record the tested commit, Python
and dependency versions, seeds, commands, exit codes, and passed/failed/skipped
counts for every run. Keep evaluator-private geometry and identities out of
agent requests and public responses.

## 1. CPU regression gate

From the repository root, install the declared test and kinematics extras in
an isolated environment and run:

```bash
python -m pytest tests/unit -q
python -m compileall -q src scripts
python scripts/generate_layout.py \
  --config configs/layouts/kitchen_search_v1.yaml \
  --catalog configs/layouts/asset_pool_v1.yaml --seed 0
```

Required cases include movement-to-point-action frame freshness, public-only
adapter serialization, explicit AGENT versus ORACLE mode selection, terminal
feedback consistency, supported clutter placement, missing requested-object
counts, insertion-order-independent cabinet clearance and missing simulator
checks. A structural candidate is not a physically accepted layout.

## 2. Real-camera and AGENT acceptance gate

Use the existing simulator host/environment; do not reinstall the licensed
dataset or accept new licenses as part of these instructions. After checking
that the environment and assets are already available:

```bash
python scripts/verify_setup.py
python -m pytest tests/integration -m sim -q --tb=short
```

The existing integration suite is necessary but insufficient: much of it uses
ORACLE mode. Add/run AGENT traces through Python, MCP and human-play adapters,
using only returned observations to select the next action. Require:

1. RESET -> MOVE/TURN -> point interaction with the returned frame ID;
   stale IDs reject while the current ID succeeds when otherwise valid.
2. Independently marked image targets at multiple pixel positions, depths and
   camera orientations. Verify optical axes, XYZW quaternion convention,
   range versus Z-depth, same-camera modalities and renderer segmentation.
   Projecting expectations with the same implementation is not independent
   validation. Missing required modalities must not count as a passing test.
3. OPEN -> observe inside -> GRASP -> PLACE -> GRASP target -> REPORT_DONE.
   Wrong-object grasp must remain recoverable under the current AGENT rules.
4. OBSERVE while holding an object and near obstacles; verify robot, joints,
   held-object pose and world state before/after, including injected errors.
5. Unsafe rejection, malformed action, capability rejection and final-budget
   REPORT_DONE; exactly the documented feedback and terminal semantics.

Record per-step RGB, private frame/calibration evidence, private execution
traces and public payloads separately. Investigate simulator errors as invalid
evaluation runs, not agent reasoning failures. Essential tests that skip mean
this gate is incomplete.

AGENT `run_dir/events.jsonl` uses the visual-session trace schema, not the
legacy ORACLE schema consumed by `compute_metrics`. It can contain private
grounding diagnostics and must never be supplied to the agent. An end-to-end
AGENT evaluation runner with its own SR/ST aggregation remains to be added;
the legacy `cli run` command deliberately runs ORACLE acceptance agents.

Strict visual grounding requires synchronized renderer segmentation, depth
and camera geometry internally; the public observation remains RGB-only.
Hosts that previously relied on physics-ray fallback without these modalities
must repair/configure the modalities before running AGENT experiments. Failing
setup in that case is intentional; it is not an agent failure or a passed run.

## 3. Generated-layout acceptance gate (implementation still required)

The current candidate generator is not a scene loader. Before using its seeds
in a benchmark, implement a manifest-to-scene build path with an explicit
policy for retaining architecture and removing/replacing original furniture.
Resolve exact dataset `(category, model)` pairs against installed assets;
never infer model codes by stripping scene-instance names. Calibrate local
origins, support heights, front direction and articulation bounds.

For each accepted layout, instantiate, settle, test support and collision,
opening clearance, connectivity from robot spawn, feasible interactions and
OBSERVE visibility. Save/reload an immutable accepted layout keyed by manifest
hash. Missing checks remain not-run and block simulator acceptance.

Prove that changing layout_seed changes furniture, changing episode_seed
does not change furniture, and a later appearance_seed changes only approved
visual properties. Start with seeds 0–19 as a debugging batch, not a claimed
representative scientific test set. Record rejected layouts and reasons.

## 4. First scientific pilot, after the AGENT gate

Use the existing fixed scene first, without claiming generated-layout
generalization. Freeze the simulator, skill interface, embodiment, budget and
episode set. Compare the same selected VLM under (a) current RGB only,
(b) RGB plus interaction history and (c) history plus OBSERVE. Treat an ORACLE
agent as an engine diagnostic; it has different information and action
semantics and is not a directly comparable public-policy baseline.

Use a paired target-location/occlusion design, independent held-out episodes,
and equal total planning budgets including OBSERVE. Report SR and successful
episode step count together with timeout/false-completion rates, feedback
distributions, observation usage, wall time and model cost. Report simulator
invalid runs and exclusions separately. For stochastic agents repeat seeds;
use paired confidence intervals over episodes rather than cherry-picked runs.
Do not interpret successful-only steps as efficiency in isolation from SR.

Only after the generated-layout gate passes should a second pilot vary layout
and embodiment. Only after layout acceptance should texture, lighting and
render-quality ablations begin, with geometry held fixed.
