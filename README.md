# RummageBench

Embodiment-aware interactive object-search benchmark built on **BEHAVIOR-1K v3.9.3 + OmniGibson**.

**Positioning.** Existing embodied benchmarks evaluate agents through end-to-end
successful execution, which entangles high-level decision quality with low-level
execution noise. RummageBench is a *diagnostic* benchmark: the admissible action
space is dynamically grounded from robot capability, object affordance and world
state

```
A_t = Ground(Robot, Object, WorldState_t)
```

and every validated interaction executes as an instant symbolic state
transition. The benchmark therefore measures **embodiment-aware interactive
reasoning, not manipulation control** (the Action Expert is assumed to be a
perfect executor: no trajectory optimization, no grasp controller, no physics
manipulation).

The agent reasons at the semantic-skill level (`NAV` / `OPEN` / `CLOSE` /
`GRASP` / `PLACE`); the backend realizes each validated action symbolically, so
correct high-level reasoning never fails because of navigation, IK,
motion-planning or grasp-controller noise.

## Repository layout

```
configs/       simulator + robot configs (data, not code)
scenarios/     data-driven episode definitions (YAML) — "game levels"
src/rummagebench/
  core/        canonical data model, session, events (one schema everywhere)
  core/skill_grounder.py
               Embodied Action Grounding Engine (EAGE): regenerates the
               available action space every step from robot x object x state
  object_interface/
               Rigid / Articulated / Receptacle adapters; each entity proposes
               its semantically valid skill candidates (available_skills())
  feasibility/ kinematic feasibility: IKSolver protocol (reachability proxy;
               TRAC-IK pluggable for URDFs) + CollisionChecker with an
               allowed-collision matrix (finger<->target allowed, closed
               foreign volumes forbidden)
  robots/      RobotEmbodiment capability parameters (reach radius, height
               band, hand capacity) — the embodiment layer
  skills/      NAV / OPEN / CLOSE / GRASP / PLACE (execution only, never
               episode decisions)
  environment/ JSON facade: env.reset() / env.step(action_json)
  sim/         abstract SimBackend; OmniGibson lives ONLY under sim/omnigibson/
  validation/  semantic / target / safety / feasibility validators
  grounding/   target grounding (oracle entity now, pixel later)
  authoring/   scene inspection, scenario builder
  evaluation/  episode loop, JSONL logs, metrics
  agents/      scripted success / wrong-object / timeout / unsafe (data-driven)
  adapters/    python_api + MCP (transport only, zero benchmark logic)
tests/         unit (no simulator) + integration (marked `sim`)
build/ runs/   generated artifacts (gitignored)
```

## Install (BEHAVIOR-1K v3.9.3 external dependency)

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git --branch v3.9.3 --depth 1
cd BEHAVIOR-1K
./setup.sh --new-env behavior --omnigibson --bddl --dataset \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
conda activate behavior

# RummageBench itself
pip install -e .            # from this repo
pip install mcp             # only needed for serve-mcp
```

Datasets land in `BEHAVIOR-1K/OmniGibson/datasets` by default; set
`OMNIGIBSON_DATA_PATH` before importing to relocate them.

## Authoring workflow (game levels, not scripts)

```bash
python -m rummagebench.cli inspect --list-scenes
python -m rummagebench.cli inspect --scene Beechwood_0_int   # rooms/objects/containers
python -m rummagebench.cli build --scenario scenarios/knife_search_001/scenario.yaml
# -> build/scenarios/knife_search_001/{initial_state.pt,preview.png,resolved_entities.json,build_report.json}
```

## Run the benchmark (direct Python, no MCP needed)

```bash
python -m rummagebench.cli run --scenario knife_search_001 --agent scripted       # SUCCESS
python -m rummagebench.cli run --scenario knife_search_001 --agent wrong_object   # FAIL_WRONG_TARGET
python -m rummagebench.cli run --scenario knife_search_001 --agent timeout        # FAIL_MAX_STEPS
python -m rummagebench.cli run --scenario knife_search_001 --agent unsafe         # FAIL_UNSAFE_ACTION
```

Trajectories: `runs/<run_id>/events.jsonl` + `summary.json` (+ per-step PNGs unless `--no-images`).

## MCP adapter

```bash
python -m rummagebench.cli serve-mcp        # stdio; tools: reset_episode / observe / act / episode_status
```

## Tests

```bash
python -m pytest tests/unit -q              # no simulator needed
python -m pytest tests/integration -m sim -q # needs the built scenario + dataset
```

## Architecture rules enforced by the layout

1. No scenario-specific Python — scenario differences live in YAML.
2. OmniGibson imports exist only under `src/rummagebench/sim/omnigibson/`.
3. Skills execute; BenchmarkSession decides success/failure/horizon.
4. MCP forwards transport; it contains zero benchmark logic.
5. Generated files stay in `build/` and `runs/`.
6. One canonical schema: `Action`, `Observation`, `StepResult`, `EpisodeStatus`, `ScenarioSpec`.

## Verified results (Beechwood_0_int, R1Pro, v3.9.3)

```
MVP ACCEPTANCE OK:
  scripted      SUCCESS            5 steps  (NAV kitchen -> OPEN x3 -> GRASP knife)
  wrong_object  FAIL_WRONG_TARGET  5 steps
  timeout       FAIL_MAX_STEPS     17 steps
  unsafe        FAIL_UNSAFE_ACTION 2 steps
  reset determinism: semantic state + pose (<10 cm) restored
unit tests 35/35 - integration tests 11/11 - MCP acceptance OK
```

## v2: embodiment-aware extension

- **Dynamic skill grounding.** `available_skills` in every observation is
  regenerated each step: entity adapters propose semantically valid candidates,
  the feasibility engine (reach + collision) filters them into the skills that
  actually exist for this robot at this world state. A skill *emerges* from the
  world; it does not pre-exist in a fixed action list.
- **Feasibility as a first-class failure dimension.** Structured, non-terminal
  failures `UNREACHABLE` / `COLLISION` / `INVALID_STATE` come from the IK /
  collision / state validators — failure attribution is mechanical, not labeled.
- **JSON environment API** (`environment/env.py`): `env.reset()` and
  `env.step({"type": "OPEN", "target": "cabinet_01"})` with
  `{status, reason, state_update, observation}` responses.
- **Capability-parameterized robot.** `reach_radius`, `z_min/z_max`,
  `hand_capacity` live in scenario YAML — same world, different embodiments.

### Difficulty ladder (roadmap)

| Level | Axis | Status |
|---|---|---|
| 0 | Known location (interaction execution) | `knife_search_001` |
| 1 | Unknown container (search planning) | `knife_search_001` |
| 2 | Distractors (semantic discrimination) | `knife_search_001` |
| 3 | Embodiment constraint (capability-aware planning) | supported: capability params + `tests/unit/test_embodiment.py` (same task, different embodiment -> different admissible action graph); multi-robot scenario variants pending |
| 4 | Occlusion / rearrangement (interactive perception) | future |

### Key experiment this design enables

Same scene, same instruction, robots with different `reach_radius` / height
band: the grounded action graph differs (`OPEN(top_cabinet)` exists for the
mobile manipulator, not for the compact arm). An agent that understands
*its own capability* vs *the world's capability* is exactly what Level 3
measures.

## Known limitations

- Kit/Isaac Sim teardown segfaults after a fully successful run on this host
  (benign; scripts exit via `os._exit` after writing all artifacts).
- Environment creation (Kit launch + scene import) takes ~5-10 min per process
  and segfaults nondeterministically ~20% of the time on first launch; retry.
- Reset restores the semantic state exactly and the robot base within ~2 cm
  (suspension re-settling is history-dependent); RTX rendering is not
  pixel-deterministic, so the acceptance gate is semantic + pose, not pixels.
- `Inside` relation sampling is probabilistic; the builder retries 5x and
  falls back to direct pose placement, then verifies or fails the build.
- GRASP teleports the robot base next to the target before establishing the
  assisted-grasp joint (a joint created across the room breaks under stretch).
- Pixel grounding is stubbed; oracle entity naming is the MVP grounding mode.
- Safety is rule-based (fixed-base grasp, forbidden categories); it does NOT
  evaluate grasp-region safety (no interaction_region in the action schema).
