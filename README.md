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

Proxy-era acceptance (v2, Beechwood_0_int, R1Pro, reach-radius grounding):

```
MVP ACCEPTANCE OK:
  scripted      SUCCESS            8 steps  (NAV kitchen -> NAV cabinet A -> OPEN A
                                             -> NAV drawer A -> OPEN drawer
                                             -> NAV cabinet B -> OPEN B -> GRASP knife)
  wrong_object  FAIL_WRONG_TARGET  5 steps
  timeout       FAIL_MAX_STEPS     17 steps
  unsafe        FAIL_UNSAFE_ACTION 2 steps
  reset determinism: semantic state + pose (<10 cm) restored
unit tests 35/35 - integration tests 11/11 - MCP acceptance OK
```

v3 (physical grounding correctness) re-acceptance on the simulator host:
pending — run the unit suite (`pytest tests/unit`) anywhere, then
`pytest tests/integration -m sim` + `scripts/run_all.py` on the GPU host
after exporting the R1Pro URDF. Do not quote proxy-era numbers as
kinematic-grounding results.

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
