# Observed motion and reported solver velocity

The native home geometry handoff already restores 87 native entities and 14
added BEHAVIOR asset shapes. Its next blocker was a stability predicate:
sub-millimetre pose motion coexisted with reported angular velocities above
0.02 rad/s, even over a twelve-second observation. Increasing velocity solver
iterations from one to four did not fix it.

PhysX documents that its position and velocity constraint iterations solve
different equations: velocities reported to the caller need not equal the
velocities used to integrate poses under contact constraints. See the official
[constraint solver documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html#solver-iterations).
This explains why a reported solver velocity is not itself a measurement of
geometric motion. It does not establish that every discrepant scene is stable.

## Versioned, opt-in observation

`--background-stability-profile observed-motion-v3` selects a separate
background observation contract. The default `legacy` contract and the general
household `placement_stability` function are unchanged. Historical failures
are not retrospectively promoted.

The new observer reads raw PhysX transforms and velocities after **every physics
step**, while the existing normal OmniGibson stepping and controllers continue.
It observes the root and every link with collision meshes, including articulated
props. It requires three seconds, a physics interval no larger than 1/60 s,
the expected exact sample count, valid handles and finite data. SDK read errors,
missing callbacks and changed timesteps reject. No physics parameters or sleep
flags are changed by this observer.

For consecutive positions and sign-aligned unit quaternions it computes

```
v_i = ||p_i - p_(i-1)|| / dt_i
w_i = 4 atan2(||q_i - q_(i-1)||, ||q_i + q_(i-1)||) / dt_i
```

Maximum translation from the initial pose must stay within 5 mm, rotation within
2 degrees, each step's translational speed within 0.01 m/s and angular speed
within 0.02 rad/s. There is no averaging or percentile exemption. These are the
same numeric tolerances as before, applied to the observed motion. Raw solver
velocity checks remain in each link's receipt, with their own pass/fail result.
OnTop must hold initially and at **every ordinary state-update step**; this is
not a claim that support is verified continuously between steps.

Target insertion additionally checks the saved root poses and repeats the
motion observation. This gate covers added daily props. Full native-state
matching, clearance, collision penetration, door sweeps, cross-member restore,
and search certification still require their separate checks. A three-second
trace cannot prove indefinite stability or absence of motion within one physics
step. Neither a passed background nor a known-target skill smoke is a formal
evaluation episode.

## Reproduction

Use a fresh output directory for each run. These commands reuse the actual
mesh-based Gen candidate; they do not regenerate or replace assets.

```bash
SIM=/data1/ygwang/miniconda3/envs/behavior/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1

# Always produces a diagnostic-only outcome, even if the background passes.
$SIM scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario build/behavior_geometry_loop/v5/home/scenario.yaml \
  --out build/motion_trial/controls --gpu 5 --realize-background \
  --background-stability-profile observed-motion-v3 --diagnose-motion-controls

# Separate unperturbed run, with public OPEN / GRASP / REPORT_DONE afterwards.
$SIM scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario build/behavior_geometry_loop/v5/home/scenario.yaml \
  --out build/motion_trial/execution --gpu 5 --realize-background \
  --background-stability-profile observed-motion-v3
```

The private controls restore a saved state before each stationary, sliding and
tipped-drop case for a real bowl and a real shaker. Interventions are explicitly
recorded. They are never emitted as data, and their diagnostic flag prevents
public-skill smoke success. Raw transforms and measured motion allow checking
that moving controls fail and that apparent stationary readings are not stale
getters. Licensed snapshots and traces remain in ignored `build/`.

CPU tests cover real-motion spikes despite zero reported velocity, slow drift,
oscillation, invalid quaternions, missing samples, partial support loss, and a
moving collision link with a stationary root. Fixtures are synthetic unit tests
and are not physical acceptance evidence.

## Remaining restore defect — explicitly not fixed

The controls exposed a separate reset defect. **Follow-up:** the spatial-cache
restore fix and new independent controls are recorded in
[the batch production handoff](behavior_batch_production.md). The failures below
remain historical evidence; their outputs have not been relabeled as passes.
After the old object-only
checkpoint restore, a sleeping shaker lost OnTop for all 91 ordinary-step
observations despite unchanged raw poses. In the 120 Hz controls this affects
`daily_008`; the two selected stationary controls themselves still pass.

A subsequent experiment copied the installed SDK's persistent contact buffers,
verified their row/column/body path ordering, restored them in place, and reset
the pending-step count. **This did not fix the defect**: all six cases still lost
`daily_008` support. Inspection of the saved checkpoint found no current contact
entries for this shaker even though its cached OnTop query had passed. The SDK
also caches spatial predicates while bodies sleep. Contact-buffer restoration
alone is therefore insufficient; the precise interaction between sleeping
contact updates and the predicate cache needs further verification.

That experimental snapshot wrapper was **removed from production code**. Its
source is retained locally under `build/behavior_motion_controls/v3_sources`
with the failed controls under `v3`. No physics parameters, forced sleep or
velocity zeroing were used to turn this failure into a pass. Do not certify a
frozen family using these runs. A reliable restoration check must preserve and
revalidate the same physical background, including these support relations.

## Local evidence

`build/behavior_motion_controls/v1` ran real 30 Hz stationary/slide/tipped-drop
controls. Getter poses matched raw PhysX transforms exactly. Sliding displaced
the selected bowl by 4.67 cm and shaker by 9.78 cm; tipped drops rotated them
by more than 50 degrees. These are diagnostic interventions, not generated data.

`v2` observed every 1/120-second physics step. The two selected stationary
controls passed and all four moving controls failed. The background itself
passed with 14 props and zero placement repairs, though another sleeping shaker
lost support readings after the old snapshot restore. That remaining defect
motivated the contact snapshot change above.

`execution_v1` separately ran the unperturbed scene on GPU5. Its background and
post-target-insertion checks passed. Five evaluator pose proposals rejected on
base collision; the sixth executed the unchanged public pixel `OPEN`, `GRASP`,
and `REPORT_DONE` interfaces and reached `SUCCESS`. Process exit and the bound
receipt both confirm `execution_smoke_passed`. The scene is `Benevolence_1_int`,
robot R1Pro with its actual head camera. `06_GRASP.png` shows the open drawer
and tablespoon. This run used the new observer but predates the contact snapshot
experiment. It demonstrates one known-target execution, **not** a matched family,
search difficulty, 200 accepted tasks or new house architecture.

`execution_v2` also reached public-skill SUCCESS, but used the experimental
contact-buffer wrapper. It is retained as separate experimental evidence and
is not used to claim that the removed wrapper works. The retained production
source matches `execution_v1`'s observer/realizer and standard snapshot code.

Final CPU regression: **880 passed, 45 skipped, 13 deselected**, with the normal
`python -m pytest -m 'not sim' -q` command in the CPU test environment. An
additional six CPU tests for the withdrawn snapshot experiment passed in the
simulator's PyTorch environment; they did not prove the real restoration case
and are not counted as retained production tests.

There is **one distinct generated home/background**, two successful execution
runs (one experimental), and **zero evaluation-ready episodes**. Six repeated
intervention cases per diagnostic run are controls, not six generated tasks.
The camera remains the R1Pro head camera. No new house architectures, certified
search costs, camera-protocol equivalence, or 200 accepted episodes are claimed.
