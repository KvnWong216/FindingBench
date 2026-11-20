# Dense household search: 100 review recipes

Open [review.html](review.html) or [review.csv](review.csv) for the complete task list. These are 100 distinct physical specifications using installed BEHAVIOR scenes and objects, not 100 language paraphrases and not official BEHAVIOR terminal tasks.

The catalogue contains 14 native scene layouts, 13 task families and 30 named target types. Eight episodes require both target fruits. Instructions, skill inputs/outputs and review files are English.

## What is implemented

- Every scenario specifies the source area, target objects, a destination, native models, clutter placements, initial container states, robot start and executable terminal predicates.
- Instructions name a household request. They do not tell the player to CHECK, remove occluders, restore objects or report completion.
- Destinations described as “in front of you” refer to the initial robot orientation. Private authoring records retain the exact receptacle. They must not enter model messages.
- A named medicine/supplement is distinguished by its native label. The inspected textures include Women's Multi (`egondf`), Co-Q10 (`fmfwng`) and Krill Oil (`kasbsy`). Rendered CHECK identifiability is still a release gate.
- Distractors use native assets and dimensions. Single-container tasks include alternative objects; multi-target fruit tasks exclude extra matching target fruits from the generated distractors.
- Success checks every required target at the destination. Holding an object is not delivery. Sink/basket containment uses the actual native Inside predicate; an outer bounding box alone does not establish success.
- Restoration is not a hard goal or an instruction. The validator saves private initial object poses, relations and container states for later continuous restoration scoring. The continuous scoring formula is not implemented by this dataset compiler.

## Validation boundaries

All 100 recipes passed schema/asset/cardinality/physical-specification uniqueness checks and conservative offline route screening. The navigation proposal uses a 0.1 m grid, an all-heading circumscribed base disc and a 0.2 m free endpoint neighborhood. It is based on frozen surveys. It is **not** proof of current scene traversability, arm reach, OPEN access, pose-insensitive manipulation or human solvability.

`physics_status` is updated separately by native initialization checks. A successful initializer verifies declared placements and initial states after settling, object stability, robot spawn clearance and an initially unsatisfied goal. It exports the actual panoramic start observation and private restoration baseline. It does not solve the task. The release gates in each `authoring.private.json` still require visual inspection, live route/interaction checks and human completion.

Generation failures and timeouts are infrastructure/data issues, never model failures. Failed recipes are excluded from the human trial menu. The compiler deliberately does not repair agent decisions or weaken skill execution checks.

## Deliberate substitutions

- Red-car trunk episodes are deferred: car color, trunk containment and a tolerant garage route have not been jointly verified. Yogurt/milk discrimination is represented by breakfast grocery baskets.
- Unspecified medicines use the actual named supplement labels above.
- Available garden/kitchen baskets replace balcony-only arrangements. Cross-room shopping-to-bathroom proposals failed the conservative route screen; bathroom-local shopping baskets are included instead.
- Meat delivery ends at the sink. Defrosting is not evaluated.

These substitutions are prototype choices, not claims that the original examples have been implemented.

## Commands

From the FindingBench directory, activate `ygbench` and use `PYTHONPATH=src` for Python commands.

```bash
# Refresh review files from validation results without changing scenarios.
PYTHONPATH=src python scripts/v2/generate_dense_pilot.py --refresh-review

# Validate one exact recipe in a fresh native process.
PYTHONPATH=src python scripts/v2/validate_dense_pilot.py --task dense_search_035

# Validate the catalogue, with at most three small scenes concurrently.
# Large scenes run alone. This does not start VLM evaluation.
PYTHONPATH=src python scripts/v2/validate_dense_batch.py --workers 3

# List human trials; only current fresh initialization passes can be selected.
bash scripts/play_behavior.sh --suite dense --list
bash scripts/play_behavior.sh --suite dense --pick 35
```

The batch writes `runs/dense_search_pilot/batch_status.json`, per-task `native_validation.log` and `validation.json`. Regenerating recipes with `generate_dense_pilot.py` changes the catalogue and invalidates changed recipe hashes; do not regenerate during evaluation. No human difficulty levels are assigned before human reference runs.
