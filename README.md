# RummageBench

Embodiment-grounded interactive object-search benchmark built on
**BEHAVIOR-1K v3.9.3 + OmniGibson**.

**Positioning.** Existing embodied benchmarks evaluate agents through
end-to-end successful execution, which entangles high-level decision quality
with low-level execution noise. RummageBench is a *diagnostic* benchmark:

> **FindingBench abstracts away motor execution while preserving
> embodiment-level physical constraints.**

The admissible action space is dynamically grounded from the robot's REAL
kinematic model, the object's interaction interface and the world state:

```
A_t = Ground(RobotGeometry, ObjectInterface, WorldGeometry, WorldState_t)

Robot model (URDF) -> IK -> collision geometry -> interaction feasibility
                   -> grounded action space -> instant symbolic execution
```

**Execution is symbolic; physical feasibility is real.** Every validated
interaction executes as an instant, deterministic symbolic state transition —
no trajectory optimization, no grasp controller, no physics manipulation —
but a skill only EXISTS in the action space if at least one interaction
configuration satisfies semantic preconditions, state preconditions, URDF-IK,
joint limits and configuration-space collision constraints.

**FindingBench evaluates configuration-level physical feasibility, not
trajectory-level executability.** No RRT/OMPL/CuRobo, no motion planning, no
controller simulation: the feasibility question is "does at least one
collision-free interaction configuration exist?", and the answer is derived
from geometry, not from hand-authored capability proxies.

## Repository layout

```
configs/       simulator + robot configs (data, not code)
scenarios/     data-driven episode definitions (YAML) — "game levels"
src/rummagebench/
  core/        canonical data model, session, events (one schema everywhere)
  core/skill_grounder.py
               Embodied Action Grounding Engine (EAGE): regenerates the
               available action space every step from robot geometry x object
               interface x world state
  core/navigation.py
               NavigationTargetProvider protocol (NamedAnchorProvider today;
               NAV stays a perfect executor over build-time-verified anchors)
  object_interface/
               Rigid / Articulated / Receptacle adapters; each entity proposes
               semantically valid skill candidates AND its interaction
               interfaces (interaction_targets): articulated handle links,
               canonical rigid-body candidates, receptacle regions — never
               the entity root pose
  feasibility/ the physical grounding toolbox:
               ik_solver.py        Pose/IKResult contracts; RobotKinematicsBackend
                                   protocol; legacy reach-radius proxy (TESTS ONLY)
               pinocchio_solver.py URDF-backed kinematics: FK, joint limits,
                                   numerical SE(3) IK (damped least squares,
                                   joint-limit projection, deterministic
                                   multi-seed restarts, fine-grained failure
                                   attribution: NO_IK_SOLUTION / JOINT_LIMIT /
                                   NUMERICAL_FAILURE)
               collision.py        AllowedCollisionMatrix, CollisionResult;
                                   legacy point-vs-AABB checker (TESTS ONLY)
               hpp_fcl_checker.py  configuration-space collision: robot
                                   collision links (URDF <collision>) x world
                                   geometry (coal/hpp-fcl), self-collision +
                                   robot-world, ACM-filtered, q-based
               interaction_target.py
                                   interaction interfaces: handle links
                                   (fallback_link_anchor auto-derived from the
                                   articulation), canonical rigid candidates
                                   (AABB center + face centers), receptacle
                                   top/inside regions; gripper standoff applied
  robots/      RobotEmbodiment capability data + model_loader (backend
               selection; missing URDF / missing pinocchio FAILS LOUDLY,
               never silently degrades to the reach-radius proxy)
  state/       benchmark-owned world state (holding truth lives HERE, never
               in the simulator's assisted-grasp internals)
  skills/      NAV / OPEN / CLOSE / GRASP / PLACE (execution only; GRASP does
               NOT move the base — implicit NAV is forbidden)
  environment/ JSON facade: env.reset() / env.step(action_json)
  sim/         abstract SimBackend (articulation info, collision geometries,
               link poses, receptacle regions, joint seeds); OmniGibson lives
               ONLY under sim/omnigibson/ (incl. robot_export.py: USD
               articulation -> URDF + FK cross-validation, and usd_collision.py:
               world colliders -> coal geometries with recorded approximation
               levels: physics mesh BVH / exact primitive / AABB fallback)
  validation/  semantic / target / safety / feasibility validators; the
               feasibility pipeline: state preconditions -> interaction
               candidates -> per-candidate IK -> configuration collision ->
               FEASIBLE(q, target) | structured failure (UNREACHABLE /
               COLLISION / INVALID_STATE, fine-grained IK reasons preserved)
  grounding/   target grounding (oracle entity now, pixel later)
  authoring/   scene inspection, scenario builder
  evaluation/  episode loop, JSONL logs, metrics
  agents/      scripted success / wrong-object / timeout / unsafe (data-driven)
  adapters/    python_api + MCP (transport only, zero benchmark logic)
tests/         unit (no simulator; proxy fixtures + real URDF kinematics) +
               integration (marked `sim`)
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
pip install pin             # real kinematics + coal collision (feasibility)
pip install mcp             # only needed for serve-mcp
```

## Supported kinematic robots

The feasibility engine needs a URDF under `robot.kinematics.urdf_path`
(joint limits, FK/IK and `<collision>` bodies are all URDF-derived — robot
geometry is never hand-authored):

| Robot | Status | URDF source |
|---|---|---|
| `test arms` (short_arm / long_arm, 3-DOF planar) | bundled test fixtures | `tests/unit/fixtures/robots/*.urdf` |
| `r1pro` (OmniGibson mobile manipulator) | supported on the simulator host | exported from the OmniGibson articulation: `python scripts/export_robot_kinematics.py` (also auto-exported to `build/robots/r1pro.urdf` on first run); the exporter FK-cross-validates the result against the simulator and refuses to write a wrong model |

`reach_radius / z_min / z_max` remain in the scenario YAML as coarse-prefilter
data for the proxy backend (unit tests). They can NEVER authorize an
interaction in the pinocchio backend, and `feasibility.backend: pinocchio`
(the production default) raises instead of falling back to them.

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

## Play (human interface)

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/serve_play.py [--scenario scenarios/knife_search_001/scenario.yaml] [--port 8090]
```

Boots the real simulator, then serves a browser UI on `http://0.0.0.0:8090`.
Top-left card: the task instruction and step budget. Bottom: one square button
per available skill type (icon + caption), with the layout density tracking
the size of the grounded action space A_t. Clicking an object-directed skill
(OPEN/CLOSE/GRASP) executes directly when exactly one target is grounded;
otherwise a dialog lists the currently grounded targets to choose from
(NAV/PLACE always ask). RESET restarts the episode. The player sees exactly
what the benchmark exposes — the egocentric view and A_t, nothing else.

## MCP adapter

```bash
python -m rummagebench.cli serve-mcp        # stdio; tools: reset_episode / observe / act / episode_status
```

## Tests

```bash
python -m pytest tests/unit -q              # no simulator needed (proxy + real-URDF kinematics tests)
python -m pytest tests/integration -m sim -q # needs the built scenario + dataset
```

Unit tests include the physical-grounding suite: real IK on URDF fixtures
(reachable / unreachable / joint-limit attribution), the morphology headline
(same world + task, different URDF morphology -> different admissible action
sets), self-collision and world-collision gates, the allowed-collision matrix
(finger<->target allowed, arm<->target forbidden), interaction-interface
grounding (handle unreachable => OPEN unavailable even when the object center
is "reachable"), no-base-teleport during GRASP, and the benchmark-owned
holding-state lifecycle.

## Scientific evaluation layer (certified difficulty)

**Episode difficulty is certified by an oracle semantic planner rather than
manually assigned.** `evaluation/oracle_planner.py` computes the exact
full-information semantic depth d*_E(s) with BFS over the canonical semantic
state (robot anchor, open set, held object, object relations), using the
PRODUCTION feasibility engine through an overlay backend — per-open-set
geometry snapshots make oracle admissibility identical to production
admissibility, and consistency tests lock that equality over every
BFS-reachable state.

- **Certification.** `evaluation/certification.py` + `cli certify` /
  `certify-split`: every benchmark episode gets a certificate
  (`build/certificates/<episode_id>.json`: solvable, oracle depth, optimal
  plan, scenario/URDF hashes). Episodes proven
  UNSOLVABLE_FOR_EMBODIMENT are excluded from the standard split and
  retained in an embodiment_stress split. `oracle_min_steps` is written
  back from the certificate — it is computed, never hand-authored.
- **Metrics (§9).** NSE = success * d*/max(d*, N); ESC = N - d*; IAR =
  (UNREACHABLE + COLLISION) / interaction_attempts; mechanical RER =
  exhausted-location revisits / search interactions (§5,
  `evaluation/search_state.py`: UNSEEN / PARTIALLY_OBSERVED /
  EXHAUSTED_EMPTY / TARGET_FOUND). NSE/ESC/RER return None rather than an
  invented value when certification/ground truth is missing. The old crude
  repetition counter is renamed `repeated_location_actions` and is NOT RER.
- **Failure taxonomy (§4).** INVALID_ACTION (malformed/unknown) /
  INVALID_STATE (wrong current state) / UNREACHABLE (semantic candidate
  exists, no IK under joint limits) / COLLISION (IK exists, every candidate
  collides) / UNSAFE_ACTION (policy) / MAX_STEPS / WRONG_TARGET. In
  candidate mode, a no-IK attempt returns UNREACHABLE (with a debug-only
  not_in_manipulation_space flag).
- **Placement regimes (§6).** `configs/task_priors/knife_search.yaml`
  declares natural vs counterfactual target locations (explicit benchmark
  data, not vague randomness); paired episodes share
  scene/target/distractors/embodiment via target-independent distractor
  assignment and carry `pair_id` for paired statistics. PRG = Perf_natural -
  Perf_counterfactual is computed in evaluation, not at generation.
- **History counterfactual (§7).** `diagnostics/history_counterfactual.py`
  builds paired situations with identical present-time observable state and
  different valid histories (container inspected+restored vs never
  touched); the history-consistent next-search sets differ, and HTA scores
  a deterministic output against them.

Exact FULL-INFORMATION oracle depth is supported; exact partial-observation
optimal cost is future work (no belief-space solver yet).

## FindingBench Core v0.1 — FROZEN

Validated 2026-09 on the simulator host against all ten freeze criteria:
URDF cross-validation pass, production pinocchio knife_search runs, q-based
collision on real BEHAVIOR geometry, no silent proxy fallback, hidden objects
do not leak, admissible/candidate interfaces, benchmark-owned holding state,
GRASP without base motion, generated episode distribution, morphology-different
action graphs. Core mechanisms (skills, grounding pipeline, feasibility
engine, scenario schema) are frozen; new work goes into agents, evaluation
and additional scenarios.

## Architecture rules enforced by the layout

1. No scenario-specific Python — scenario differences live in YAML.
2. OmniGibson imports exist only under `src/rummagebench/sim/omnigibson/`.
3. Skills execute; BenchmarkSession decides success/failure/horizon.
4. MCP forwards transport; it contains zero benchmark logic.
5. Generated files stay in `build/` and `runs/`.
6. One canonical schema: `Action`, `Observation`, `StepResult`, `EpisodeStatus`, `ScenarioSpec`.
7. Holding state is benchmark-owned; the backend's assisted grasp is
   visualization/optional realization only.

## Verified results

### v3 physical grounding acceptance (HISTORICAL numbers below are v2/proxy-era)

Production run on the simulator host (BEHAVIOR-1K v3.9.3 + OmniGibson +
Beechwood_0_int + R1Pro, `feasibility: backend: pinocchio, mode: endpoint`):

```
R1Pro URDF export cross-validation (§2):
  50 random joint configurations, simulator vs Pinocchio FK
  max translation error 2e-06 m   max rotation error 0.0001 deg   PASSED
  (gates: 0.01 m / 3 deg — a failing sample REJECTS the export)

v3 acceptance trajectories (real IK + q-based collision, no proxy):
  scripted      SUCCESS            8 steps
  wrong_object  FAIL_WRONG_TARGET  6 steps
  timeout       FAIL_MAX_STEPS     17 steps
  unsafe        FAIL_UNSAFE_ACTION 2 steps
  grounding_trace.jsonl per run: interaction targets (fallback_link_anchor /
  canonical_candidates / receptacle regions), per-candidate IK errors and
  collision results, final grounded action set

Morphology action-graph comparison (full arm vs restricted arm, same
episode): graphs_differ=True — 9/9 trajectory positions diverge; concrete
differences include GRASP(breakfast_table) / OPEN(fridge) available only to
the full arm. Real URDF morphology (right_arm_joint2 fixed), no reach_radius.

Memory pre-flight (scripts/probe_memory.py, required before benchmark runs
on shared GPUs): peak ~8.9 GB device memory for the full production path;
>= 3 GiB free-VRAM headroom gate enforced.

Generated episode distribution: 30 knife-search episodes (10 x target in
cabinet_A / drawer_A / cabinet_B, distractor permutations, instruction
paraphrases) + benchmark_splits/knife_search_v0.yaml. Certified so far:
knife_search_001 (base) and s000/s001 — oracle depth **2** each (NAV +
GRASP: the no-top cabinet exposes the knife WITHOUT opening; the hand-
authored depth-3 assumption — NAV, OPEN, GRASP — was mechanically WRONG),
plan replays SUCCESS through the production session. The replay gate
revokes certification when a plan fails to execute.

Batch note: single-process episode batches degrade after ~4 episodes from
OmniGibson teleport corruption (NaN BroadPhase cascade); the supported
batch path is the chunked certifier with per-episode resume
(scripts/certify_final.sh: 5-episode chunks, fresh Kit process per chunk,
auto-retry). Remote certification of the remaining episodes is pending on
sim-host GPU availability (Warp CUDA 700 errors from competing workloads
abort scene loads; rerun scripts/certify_final.sh when GPUs free up).
```

### v2 proxy-era results (HISTORICAL — superseded by v3 above)

```
  scripted SUCCESS 8 steps / wrong_object FAIL_WRONG_TARGET / timeout
  FAIL_MAX_STEPS / unsafe FAIL_UNSAFE_ACTION with reach-radius grounding;
  unit tests 35/35, integration 11/11 (proxy engine)
```

## Action interface protocols (§7)

`action_interface.mode` in the scenario YAML selects the agent-facing
protocol; the feasibility oracle is identical underneath (§9 layering:
`semantic_candidates` -> `physical_filter`, no duplicated logic):

- **admissible** (default): `Observation.available_skills` = skills with at
  least one collision-free interaction configuration. For evaluating
  search/planning over physically admissible action spaces.
- **candidate**: `Observation.candidate_skills` = semantic + state-valid
  skills for VISIBLE objects, no IK/collision filtering, no feasibility
  metadata. Attempts are answered by the in-environment oracle with
  structured step feedback (SUCCESS / UNREACHABLE / COLLISION /
  INVALID_STATE / ...). For evaluating whether the agent understands its own
  embodiment limitations. `unreachable_attempt_rate`,
  `collision_attempt_rate`, `infeasible_attempt_rate` (= (UNREACHABLE +
  COLLISION) / attempts) decompose those attempts in the metrics.

**Failure taxonomy (paper-facing, uniform across admissible and candidate
modes):**

| reason | meaning |
|---|---|
| `INVALID_ACTION` | malformed action / unknown skill / invalid target schema |
| `INVALID_STATE` | valid semantic action but wrong current state |
| `UNREACHABLE` | semantic candidate exists, but no valid IK / joint-limit-satisfying interaction configuration |
| `COLLISION` | IK configuration exists, but every interaction candidate violates collision constraints |
| `UNSAFE_ACTION` | task-level safety policy violation |
| `MAX_STEPS` / `WRONG_TARGET` | horizon / non-target grasp termination |

IAR = (UNREACHABLE + COLLISION) / interaction_attempts; INVALID_STATE is
reported separately. The `not_in_manipulation_space` event flag survives as
a debug annotation on UNREACHABLE events only.

**Visibility**: agent-facing skills are grounded over `backend.visible_entities()`
only — contents of closed containers are never exposed (§19-22 tests), while
oracle entity grounding remains available for execution and evaluation.

## v3: physical grounding correctness

Replaced hand-authored capability proxies with geometry- and
kinematics-derived admissible actions, while keeping execution symbolic and
deterministic.

- **Real kinematics.** `PinocchioKinematics` loads a URDF (link frames, joint
  tree, joint limits, `<collision>` bodies), locks non-controlled joints
  (mobile base) and answers IK with damped-least-squares CLIK, joint-limit
  projection and deterministic multi-seed restarts. Failures are attributed
  (NO_IK_SOLUTION / JOINT_LIMIT / NUMERICAL_FAILURE) and preserved in event
  logs; the benchmark-facing reason stays UNREACHABLE.
- **Real collision.** `PinocchioCollisionChecker` checks whole robot
  CONFIGURATIONS: self-collision (adjacency-filtered) and robot-vs-world
  pairwise coal queries over URDF `<collision>` bodies and backend-exported
  world geometry, filtered by the AllowedCollisionMatrix
  (finger<->target allowed, arm<->target forbidden, foreign volumes forbidden,
  held-object semantics for PLACE). Visual meshes are never used.
- **Interaction interfaces, not entity centers.** OPEN/CLOSE ground at the
  articulated handle/moving-link anchor (auto-derived, `fallback_link_anchor`),
  GRASP at canonical rigid-body candidates (AABB center + face centers; no
  grasp annotation, no grasp quality), PLACE at the receptacle top/inside
  region. A tool standoff keeps the gripper frame off the surface.
- **Benchmark-owned state.** `BenchmarkWorldState.held_object` is the holding
  truth; task success reads it. GRASP never moves the robot base (the
  assisted-grasp joint is realization only), PLACE releases the
  benchmark-held entity explicitly (`symbolic_place(entity, receptacle)`).
- **Fail loudly.** `feasibility.backend: pinocchio` (production default)
  requires `robot.kinematics.urdf_path` and the `pin` package; anything
  missing raises `FeasibilityBackendError` instead of silently degrading to
  the reach-radius proxy. `backend: proxy` is an explicit test-mode choice.
- **Feasibility failure attribution** now separates task-level safety
  (`UNSAFE_ACTION`, SafetyValidator) from physical failure
  (`UNREACHABLE`/`COLLISION`, physical_failure: true in events).

### Difficulty ladder (roadmap)

| Level | Axis | Status |
|---|---|---|
| 0 | Known location (interaction execution) | `knife_search_001` |
| 1 | Unknown container (search planning) | `knife_search_001` |
| 2 | Distractors (semantic discrimination) | `knife_search_001` |
| 3 | Embodiment constraint (capability-aware planning) | supported: URDF morphology decides the grounded action graph (`tests/unit/test_morphology.py`); multi-robot scenario variants pending |
| 4 | Occlusion / rearrangement (interactive perception) | future |

### Key experiment this design enables

Same scene, same instruction, robots with different URDF morphologies: the
grounded action graph differs because IK + collision say so — not because a
reach scalar was tuned. An agent that understands *its own body* vs *the
world's affordances* is exactly what Level 3 measures.

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
- Pixel grounding is stubbed; oracle entity naming is the MVP grounding mode.
- Not yet modelled (documented abstractions, not oversights): trajectory- and
  path-level feasibility beyond the endpoint configuration (no
  canonical-corridor sampling yet — `feasibility.mode: canonical_corridor` is
  the planned extension), grasp force/momentum, held-object inertia, deformable
  or articulated held objects, dynamics of any kind.
- World-collision fidelity depends on the exported USD colliders
  (approximation level recorded per body: mesh BVH / exact primitive /
  AABB fallback).
