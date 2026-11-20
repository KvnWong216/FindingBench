# vla_difficulty_v2 — L×S task set built from BEHAVIOR-1K tasks and assets

Goal: ~1000 runnable tasks = ~111 base tasks × 3 language levels (L1–L3) × 3
spatial levels (S1–S3), 111 per cell. The level definitions are those of v1
(`../vla_difficulty_v1/levels.yaml`); v1 (36 variants, Beechwood only) is kept
unchanged.

## What is re-used from BEHAVIOR-1K

| Part | Source |
|---|---|
| Task goal | an **official BDDL sub-goal** `(ontop X Y)` / `(inside X Y)` from one of 476 activity definitions (`build/vla_difficulty_v2/official_pairs.json`, 1007 patterns). The base task keeps the object category X and the receptacle category Y, cited in `source.official_activity`. `fetch` tasks keep only the object (goal `holding`, the first half of the sub-goal; not an official terminal predicate) |
| L3 intent | the official activity name ("I'm putting away Halloween decorations …") |
| Scenes | BEHAVIOR-1K scenes, unmodified except the four widened layouts below |
| Objects | BEHAVIOR-1K object models of the official sub-goal's categories (`object_pool_official.json`) |
| Furniture | native scene furniture only (surfaces, cabinets, drawers, desks, shelves …) |
| Robot / sim | R1Pro in OmniGibson, as in v1 |

Scene edits (declared, never to the original assets): the teammate's clearance
manifests (`build/behavior/clearance`, copied to `build/vla_difficulty_v2/clearance`)
widen doorways in Beechwood_0_int, Rs_int and house_double_floor_{lower,upper};
the Rs_int copy also applies the six chair/table removals of the
`store_honey_open_route` derivation. Other scenes are used as shipped.

## Spatial levels in v2

| S | Gate (checked per variant in simulation, `scripts/v2/run_variant.py`) |
|---|---|
| S1 | target in the start room, **RGB-visible** in the start panorama: ≥ 60 px **and** Lab contrast ΔE ≥ 15 against its surroundings (v1 counted segmentation pixels only — a silver knife on a beige counter passed and was invisible in RGB) |
| S2 | target in the start room, 0 px from the start; subtype `container` (closed drawer, revealed after OPEN) or `occluded` (surface hidden behind furniture), recorded as `s2_kind` |
| S3 | target in another room, 0 px from the start (cross-room navigation) |

## Pipeline (`scripts/v2/`)

1. `survey_scene.py` — one scene load: entities, support regions, drawer interiors, footprint obstacles
2. `analyze_survey.py` — CPU: reachable floor, candidate locations, start poses per room
3. `certify_locations.py` — one scene load: every location with a probe object (placement, visibility from every start, GRASP / PLACE pose search)
4. `make_phrases.py` → `phrases.yaml` — unambiguous room / furniture / drawer names (landmarks when a noun is not unique)
5. `extract_official_pairs.py`, `score_object_pool.py --official` — official sub-goals and object models
6. `generate_tasks.py` → `base_tasks/B###.yaml`
7. `compile_variants.py` → `build/vla_difficulty_v2/<task>__<L>_<S>/scenario.yaml` (L variants verified identical except the instruction)
8. `run_batch.py` / `run_variant.py` → `runs/vla_difficulty_v2/` (initialization + S gates, null baseline, scripted oracle ≤ 30 steps); `summary.md`

"Runnable" means all three levels pass for the physical configuration; it is
not VLA success. Human spot checks (`scripts/play_behavior.sh`) are listed in
`runs/vla_difficulty_v2/spot_check.md` once the batch is done.

## Structural difficulty audit

`PYTHONPATH=src python scripts/v2/analyze_difficulty.py` writes the physical
configuration scores and plots to `build/vla_difficulty_v2/difficulty/`.
Distance continues to use the original geometry surveys to isolate this structural
repair; source hashes for both captures are recorded.
The choice audit reads fresh relation surveys from
`runs/vla_difficulty_v2/_survey/relations_refresh/`; produce these with
`survey_scene.py --out runs/vla_difficulty_v2/_survey/relations_refresh` and
the scene's original start / probe receptacle arguments.

Survey schema version 1 stores `relations` separately from room membership:
each record has `parent`, `child`, `relation` (`inside` / `on_top`) and
`source=omnigibson_state`. These are native binary state queries after reset,
not inferred from overlapping boxes or object names. Query errors are recorded;
an incomplete capture cannot produce a valid choice score. The authoring probe
is excluded. Room membership remains metadata for nested objects: only objects
without relational parents get room-to-object edges. Transitive relation edges
are removed so that a shortcut cannot bypass an intermediate container. Multiple
independent parents are retained; cycles invalidate the score and are exported
for inspection. The task target's declared initial placement is then added.

Choice is the mean, over all scene objects plus the target, of the minimum
root-to-object path cost, with edge cost `1 + log2(parent outdegree)`.
This remains a **whole-scene** prior: tasks sharing a background can have very
similar choice, even with correct nesting. It is not target-specific search
entropy. The added target placement is declared configuration, while native
relations are measured in the surveyed scene; neither proves all relations in
every settled task instance. The distance remains an optimistic grid proxy;
labels are provisional and do not replace execution validation.

## Residential expansion and local tasks

The expansion queue (`scripts/v2/expand_dataset.py`) targets 20 residential
**scene layouts**, including the original three. Different floors of the same
house are not independent building sources. The queue records each scene's
stage, failures and counts in `runs/vla_difficulty_v2_expansion/progress.json`.
Use `--parallel 2` for two scene workers; shared dataset writes are locked.
It uses the existing location certifier and public
protocol oracle, with no VLM evaluation.

New `X...` task IDs preserve earlier tasks. `--allow-partial` retains only
available spatial conditions, so a local task does not require a cross-room
counterpart. `spatial[S].task_scope` records `single_room` when the start,
target and destination share a room, otherwise `cross_room`. This describes
task extent, not a cropped world or a constraint on where the agent may move.
It is not exposed as privileged task information. Missing S conditions are
recorded explicitly and must not be counted as full factorial experiments.

`--diverse-objects` uses the full accepted object pool instead of the extra
16 cm / colourfulness 40 restriction. Native BDDL `rigidBody` is required;
category and model reuse are penalized against the retained corpus. The
existing pool's geometry and saliency prefilter still applies. Probe-certified
locations do not certify the new target model: every generated configuration
remains pending until its own rollout passes initialization, visibility,
negative baseline and scripted execution checks. New validation results are
kept in `runs/vla_difficulty_v2_expansion/validation/`.

Resume with:

```bash
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate ygbench
PYTHONPATH=src python -u scripts/v2/expand_dataset.py --parallel 2
```

## Description coverage repair

Source locations and destination descriptions are now separate requirements.
A measured GRASP location need not have a unique furniture name for L2/L3
(the target category remains explicit and unique under the generator's native
object check). L1 is emitted only when its source location can be described
unambiguously; missing L1 cells are recorded in each task, never substituted
with duplicate L2 text. The compiler supports these incomplete language sets.
Therefore variant counts must come from the manifest, not `physical * 3`.
PLACE destinations still require an unambiguous room / furniture description.

New furniture descriptions may use a unique landmark (closest / farthest,
with a 0.3 m gap from all same-noun rivals) or a conspicuously distinct height
or horizontal extent (at least 30% and 0.2 m larger). Supporting measurements
are stored in `description_evidence`, privately. Colors are not guessed from
asset names. Existing reviewed descriptions and compiled scenarios are retained.

## Curated human trial menu

```bash
bash scripts/play_behavior.sh --suite smoke --list
bash scripts/play_behavior.sh --suite smoke
bash scripts/play_behavior.sh --suite smoke --pick 1
```

This fixed selection contains 10 L2 fetch tasks across eight layouts, checked
against the expansion validation summary on every launch. No source locations,
oracle routes or oracle step counts appear in the menu. It does not include
unverified PLACE tasks. Open `http://127.0.0.1:8090` (forward port 8090 when
playing remotely). Press Ctrl+C in the launch terminal before starting another
task. Each launch records a separate unassisted run in `runs/human_oracle/`.
Use REPORT_DONE when the requested object is held; reset starts another episode.

Relation-refresh surveys also record `relation_query_audit`: object count,
attempted / true / false / error counts, and unsupported state types per object.
Completeness means every available native Inside/OnTop predicate was queried
for every ordered pair of native objects; it does not invent relations for
unsupported states or infer a semantic assembly tree. Original geometry surveys
remain the distance input. The native inventories must match between geometry
and relation snapshots (excluding explicit authoring probes).

`difficulty/summary.json` includes `all_counts` and `all_variant_bands` for the
whole generated corpus. `all_easy.csv`, `all_medium.csv`, `all_hard.csv`, and
`all_unclassified.csv` include every physical configuration; the unprefixed
band files retain the accepted-only view. Missing distance remains unclassified
even when the relation graph and choice are complete.
