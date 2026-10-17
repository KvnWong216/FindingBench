# Randomized native-home development batch

The entry uses the pinned InfiniGen solver for daily-object layouts inside
complete native BEHAVIOR homes. Runtime assets and all action validation remain
native OmniGibson/BEHAVIOR. This batch varies **house templates, approved daily
asset palettes, per-surface counts, object poses, target models and target
locations**. Original walls and large furniture remain fixed within each house;
these samples are not newly synthesized floorplans.

The initial pool contains six native kitchen houses, four daily-object palettes,
three per-surface count settings and six native utensil models. Configured count
settings are proposal parameters, not difficulty labels. The current palette
contains bowls, plates, mugs, apples and salt/pepper shakers; it is a limited
living-background pilot, not a comprehensive learned household distribution.

## Generate and resume

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/behavior_pipeline.py batch-home \
  --config configs/layouts/behavior_batch_v1.yaml \
  --count 200 --max-proposals 300 --seed 20261015 --workers 8 \
  --out build/behavior_batch/randomized200_v1
```

Run the same command to resume. Source/configuration/asset hashes must still
match for unfinished batches. A completed batch permits **read-only validation**
against its exact archived Python sources in `generation_sources`; asset and
configuration changes still fail, and this fallback never permits new generation.
A conflicting request is refused. Each successful background has an
actual upstream solver receipt and its full native assembly plus ScenarioSpec;
configuration files alone never count as generated samples. The same command
validates completed artifacts on resume instead of overwriting them. An
interrupted attempt is retained and retried in a new directory, at most three
times. A completed failed proposal remains failed. Additional prespecified
proposals can fill the requested count, up to the explicit 300-proposal bound.
The final successful scene/slot histogram can consequently differ from the
proposal histogram; it is not treated as a renormalized task-family prior.

Workers run separate CPU-only Blender sandboxes, each pinned to two CPU cores.
There is no GPU access or network inside those solvers. Each proposal has
separate deterministic background and task selection streams: the chosen
hidden target location does not drive background generation or the initial
camera direction. All generated tasks disable unused legacy oracle action-list
enumeration while keeping submitted-action validation unchanged.

All native openable kitchen cabinet bodies are retained. Legacy metadata slot
rules omit some bodies, which are explicitly represented as unresolved native
interiors rather than dropped. Their live placement and full nested-region
census remain pending. Metadata slots are not a physical cavity certificate.

## Inspect diversity and acceptance

- `plan.json`: immutable proposal plan, native house and slot catalogs,
  source hashes and unresolved-region records. Evaluator-private.
- `summary.json`: requested/generated/unique counts, source-house count,
  scene/target/count-setting histograms, failures and selected sample IDs.
- `records.json`: every completed proposal, including failures and duplicates.
- `home_NNNN/attempt_00`: real solver input/output, assembly and task scenario.
- `home_NNNN/receipt.json`: bound proposal, artifact hashes, timing and failure.

Background uniqueness ignores seed values, file paths, target choice and
instance numbering. It hashes original house-state identity, asset models,
supports, positions at 2 cm resolution and orientations at 15 degree resolution.
Tiny numeric jitter and moving only the target cannot inflate background counts.
This fingerprint is a reproducible diversity proxy, not a perceptual-distance
or structural-diversity certificate. Target-location counts are reported
separately. All variants stay in the **development** split with their original
house grouping; no train/test split is manufactured from variants of one house.

## GPU diagnostics

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/behavior_pipeline.py batch-probe \
  --batch build/behavior_batch/randomized200_v1 --limit 12 --gpus 4 5
```

Only physical GPU 4/5 are accepted. One isolated simulator child runs per GPU.
The default now includes native background realization and writes
`gpu_realized_pipeline_v1/process.json` plus `gpu_realized_summary.json`.
Use `--background-mode direct-proposal` only to reproduce the historical raw
proposal diagnostic below; it writes the original `gpu`/`gpu_summary.json`
paths. These modes never overwrite each other's receipts.
Selection is round-robin by house, independent of prior GPU success. Repeat
with `--limit 200` to probe every selected candidate after CPU generation
completes; completed diagnostics are hash-checked and reused. Interrupted GPU
folders are refused for manual inspection instead of silently overwritten.

Each child preserves full-house loading, native OnTop/Inside checks, the R1Pro
head camera, public pixel actions and existing feasibility/success judgement.
It records actual images, action feedback, a bounded 25-pose diagnostic and
normal/failed process exit. `home_NNNN/gpu/process.json` distinguishes execution
success, failure and incomplete/crashed runs. `gpu_summary.json` reports these
counts for the historical direct-proposal mode; the default realization mode
uses the separate paths above. CPU solver success is never GPU success;
GPU known-target execution success is never search-cost or family certification.
Formal evaluation-ready episodes remain **zero** until those later gates pass.

An optional native placement realization runs before target placement:

```bash
/data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario build/behavior_batch/randomized200_v1/home_0010/attempt_00/scenario.yaml \
  --out build/behavior_batch/randomized200_v1/home_0010/gpu_realized_recheck \
  --gpu 4 --realize-background
```

Use a fresh output directory. The target is first parked outside the native
furniture envelope using only background geometry. The realization resets the
background RNG and checks OnTop plus measured stability. Only failing daily
props are retried on their original approved supports, for at most three passes
using the existing native placer (five trials per call). Sampler success alone
does not pass verification. Thresholds stay at 5 mm translation, 2 degrees
rotation, 0.01 m/s linear speed and 0.02 rad/s angular speed. Each placement pass
allows at most three settling windows; velocity-only failures no longer
trigger unnecessary re-placement. Persistent dynamics still reject the case.
Failed checks are retained in `background_realization.json`; successful background states have a
hashed local checkpoint and actual poses. After target insertion,
`background_after_target.json` rejects changes exceeding those same tolerances.

This optional check covers daily props. It is **not** a frozen-family certificate:
native object/joint invariance, restore/replay across every target location,
complete contact/door sweeps and temporary placement-space checks remain pending.
The checkpoint includes a parked target and requires matching scene topology.
Original Gen proposals and baseline GPU failures remain unchanged.

Build a local audit gallery after generation and probing:

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/report_behavior_batch.py --batch build/behavior_batch/randomized200_v1
```

`index.html` links all selected scenarios and shows actual simulator images
with their acceptance status. `audit.json` additionally counts distinct
backgrounds **without using orientation**, target models, daily assets,
target slots and specific solver-constraint failures. Neither is an agent
observation; hidden target configuration stays evaluator-private.

## CPU regression

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python -m pytest \
  tests/unit/test_indoor_batch.py tests/unit/test_indoor_background.py \
  tests/unit/test_indoor_realization.py tests/unit/test_behavior_native_adapter.py -q
```

The clearance-map optimization crops only all-false exterior pixels before
running the existing distance transform; distances and row-major tie breaking
are unchanged. Regression tests compare it against the full-map result, retain
unresolved native cabinets, verify target-independent background sampling,
and reject duplicate-count inflation. Synthetic objects exist only in tests.

## Measured pilot: randomized200_v1 (2026-10-15)

250 proposals completed: 200 selected, 6 reserves and 44 solver failures. Every
failure contains actual unsatisfied surface constraints; the failing proposals
were preserved. There are 200 distinct asset/position combinations even when
orientation is ignored, six target models and 85 metadata target slots. These
slots still require a live cavity census, including explicitly unresolved bodies.

| Original house | Selected backgrounds |
| --- | ---: |
| Beechwood_0_int | 41 |
| Benevolence_1_int | 41 |
| Merom_1_int | 40 |
| Pomaria_1_int | 40 |
| Wainscott_0_int | 38 |

The sixth proposed house, Ihlen_1_int, has no successful sample in this batch.
Most failed constraints concern `countertop_tpuwys_6`; breakfast tables and a
console table also fail in some proposals. See `audit.json` for exact counts.
Five native houses are represented; **newly generated floorplans: 0**.

The first 12 house-stratified GPU diagnostics produced **1 execution success,
11 failures, 188 unprobed cases**. `home_0010` (Pomaria) completed the unchanged
public skill chain. Eight failures were background support checks; the other
three exhausted the bounded pose diagnostic with base collisions, capability,
visibility or unsafe-opening failures. This is evidence that candidate
generation runs, not that the 200 candidates are physically usable episodes.
There are no certified search costs, family-matched episodes or training results.

Two initial physical-realization experiments are retained separately:
`home_0000/gpu_realized` exhausted three repair passes and failed;
`home_0001/gpu_realized` passed background support/stability but failed the
public skill chain. The latter passed without needing support resampling after
target-independent parking; this does not by itself establish the cause of its
earlier support failure. The stricter subsequent runs additionally measure
background changes after target insertion.

Final stricter realization diagnostics (all original failures retained):

| Candidate | Background realization | After target insertion | Public skills / process |
| --- | --- | --- | --- |
| home_0001 | Passed, no resampling | Rejected: angular speed 0.0266 rad/s exceeds 0.02; position almost unchanged | Not attempted after rejection; exit 1 |
| home_0010 | Passed, one local re-placement | Passed daily-prop invariance | Failed bounded skill diagnostic; exit 1 |
| home_0127 | Passed, one local re-placement | Passed daily-prop invariance | Passed; exit 0 |
| home_0217 | Passed, six local re-placements across bounded passes | Passed daily-prop invariance | Passed; exit 0 |

The last two were **targeted diagnostics of a previously reachable cabinet**,
not additional random acceptance samples. Both retain all 87 native objects in
Benevolence, with different seven-object daily backgrounds and different target
utensils. Their images, checkpoints, actual poses and action results are under
`home_NNNN/gpu_realized_v2`. In total 14 distinct candidates received some GPU
diagnostic; 186 received none. The original stratified result remains 1/12.
The two stricter successes do not certify a matched target-location family,
navigation/search cost, all native-state invariance or difficulty tiers.

The completed-resume bug (unnecessary spare jobs after reaching the requested
count) was fixed and regression-tested: a completed run validates all selected
artifacts and starts no worker. The selected 200 IDs did not change. Extra
attempts created before the fix remain visible in the 250-attempt accounting.

## Fixes and evidence

The subsequent [pipeline root-cause audit](behavior_pipeline_root_causes.md)
identifies repeated invalid support constraints, includes controlled CPU
replays, and explains the default native-realization mode added after this batch.

| Issue | Implementation | Regression evidence |
| --- | --- | --- |
| Full-house clearance transform repeated per task was expensive | `indoor_background.native_start` crops only false exterior pixels; clearance and tie breaking unchanged | Full-map versus cropped-map equivalence test |
| Completed resume could schedule more proposals | `indoor_batch.generate_batch` validates completed receipts before starting any executor | Completed-resume test forbids worker creation and rejects changed scenario hashes; actual 200-case read-only resume passed |
| Exit code alone cannot prove an accepted GPU result | `indoor_batch_probe.classify_process` binds completed status to process exit, inputs, runner and GPU | Missing/running results, crashed success claims and changed inputs are rejected |
| Background realization can depend on task import location | `indoor_realization.parking_position` uses only native geometry; background RNG resets before bounded native placement | Target-independent parking and bounded-retry tests; real GPU diagnostics retained separately |
| A placed target can leave daily props outside stability limits | `indoor_realization.verify_after_target` compares actual pre/post poses, velocities and OnTop | Synthetic displacement rejection plus an actual angular-speed rejection; no threshold relaxed |

Final CPU run for this revision:
`python -m pytest -m 'not sim' -q` → **808 passed, 44 skipped, 13 deselected**.
The skips are not physical acceptance. GPU results above are independent of the
unit tests. This batch did not start training. All generated artifacts remain
local under `build/behavior_batch/randomized200_v1`; licensed state checkpoints
are not committed. The optional GPU experiment's local source snapshot is under
`runtime_sources_v2`, including existing working-tree changes for provenance.
