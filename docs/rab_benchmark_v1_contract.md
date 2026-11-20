# FindingBench RAB Benchmark v1 Protocol & Contract Draft (`findingbench_rab_pinhole_v1`)

- **Version**: `v1-draft-20261120`
- **Status**: Draft established; items requiring pre-freeze empirical verification are explicitly marked `[PENDING FREEZE]`.
- **Authoritative Inventory**: `runs/rab_dense_serial_20261119/combined_manifest.json` (`SHA256: 715ea3f8f5fe7ae5e274968d4c70e1ca1c006e3aa0d529126da812536848667e`)
- **Upstream Assets**: RobotAgentBench (`fa207f52d8da139f591f9dc590eb002c58f72383`), Runtime-room `0.1.7` (HF revision `8c0c1c95f29dd658476e1ad8572ebf960ba98aa2`).
- **Pilot & Reserve Ledger**: `configs/benchmarks/rab_v1_pilot_draft.json`

---

## 1. Scientific Problem & Two-Layer Boundary

FindingBench decouples **high-level interactive search planning** from **low-level embodied execution** via a fixed public skill interface (`findingbench_rab_pinhole_v1`):
1. **High-Level Policy ($\pi_H$)**: Observes only the natural-language household request, the current ego pinhole RGB frame (`frame_id`), the auxiliary 4-view `CHECK` card (when invoked), and public action-feedback history. It outputs one structured skill invocation per step.
2. **Low-Level Executor ($\mathcal{E}_L$)**: Grounds normalized 2D image coordinates (`[0, 1]`, top-left origin) into 3D scene rays, verifies reachability, collision/clearance, and state preconditions, and drives native PhysX actuation (mobile base A* + velocity ramp, articulated drawer/refrigerator joint drives, bimanual Jacobian arm + gripper drives).

Both layers emit separate audit logs so evaluation can distinguish **"the policy did not know what to do next"** from **"the policy issued a sensible intent that the current robot/executor could not realize."**

---

## 2. Dataset Sets & Pre-Registered Pilot Contract

### 2.1 Three Canonical Sets
1. **RAB 1K Full Inventory (`N = 1000`)**:
   - Verified in `runs/rab_dense_serial_20261119/combined_manifest.json` and `serial_repair_verification.json` (`1000/1000` accepted initial stability + native occlusion ray or articulated drawer/fridge sweep; `0` pending).
   - **Arrangement strata**: `872` articulated dual-drawer (`articulated_drawer`), `123` open-surface movable occlusion (`occluded_open`), `5` articulated refrigerator (`articulated_refrigerator`).
   - **Regime strata**: `400` `short_range`, `600` `cross_zone_or_composite`.
   - **Target count**: `824` (1 target), `147` (2 targets), `26` (3 targets), `3` (4 targets).
   - **Layout & asset diversity**: `50` layout IDs (`20` tasks each), all belonging to the `LIVING_STRAIGHT` architectural family with **6 unique architectural geometry signatures** (`909510c455e6`: 16 layouts; `efc8f17104a8`, `c5c94501c5ef`, `559f2c98017f`, `0be0a38c42a6`: 8 layouts each; `68d395c91811`: 2 layouts), `28` world-transform architectural groups, and `50` furniture/material signatures (`runs/rab_migration_20261118/diversity_audit_v2/diversity.json`). `37` unique placed object model IDs.
   - **Rebinding ledgers**: `292` shelf-stable packaged snack tasks rebound to living-room snack drawers (`living_room_snack_storage`); `5` refrigerator tasks rebound (`rab_refrigerator`).
   - **Retired legacy data**: BEHAVIOR object libraries and 1,310 spectator caches are retired and never mixed with RAB.
2. **Pre-Selected Stratified Pilot (`N = 30` primary)** (`configs/benchmarks/rab_v1_pilot_draft.json`):
   - **Refrigerator stratum (`5` tasks, oversampled 100% of available 1K fridge tasks)**: `F014`, `F054`, `F199`, `F205`, `F262`. Reported as an explicit separate stratum; never pooled unweighted to claim overall 1K SR.
   - **Open-occlusion stratum (`10` tasks)**:
     - Short-range, 1 target (`3`): `F251`, `F164`, `F030`
     - Short-range, multi-target (`2`): `F275`, `F064`
     - Cross-zone, 1 target (`3`): `F016`, `F017`, `F021`
     - Cross-zone, multi-target (`2`): `F087`, `F039`
   - **Articulated-drawer stratum (`15` tasks)**:
     - Short-range, 1 target, standard (`4`): `F022`, `F032`, `F063`, `F065`
     - Short-range, 1 target, snack rebinding (`2`): `F168`, `F304`
     - Short-range, multi-target (`2`): `F067`, `F069`
     - Cross-zone, 1 target, standard (`3`): `F001`, `F047`, `F048`
     - Cross-zone, 1 target, snack rebinding (`2`): `F055`, `F081`
     - Cross-zone, multi-target (`2`): `P136`, `F066`
   - **Disjointness guarantees**: All 30 primary tasks + 18 reserve tasks span **48 distinct layout IDs** across all 6 architectural geometry signature groups (`909510c455e6`: 10 primary, `559f2c98017f`: 6, `0be0a38c42a6`: 5, `c5c94501c5ef`: 4, `efc8f17104a8`: 4, `68d395c91811`: 1) and unique semantic signatures.
3. **Exclusion & Reserve Ledger (`18` pre-ranked reserves)**:
   - Pre-registered in `configs/benchmarks/rab_v1_pilot_draft.json` by stratum (`reserve_rank = 1, 2, ...`).
   - **Replacement Rule**: Every primary candidate must undergo public point-selection playability verification (find $\to$ access/unocclude $\to$ grasp $\to$ deliver $\to$ `REPORT_DONE`). If a task is physically or semantically blocked under the frozen executor (e.g., target unreachable from any collision-free base pose), the exact physical/semantic evidence and trace path are logged in `exclusion_ledger`, and the lowest-rank reserve from the **same stratum** replaces it. Silent target substitution or unlogged task swapping is strictly prohibited.

---

## 3. Public Interface & MCP Contract (`findingbench_rab_pinhole_v1`)

### 3.1 Nine Public Skills
| Skill | Parameters | Public Semantics |
|---|---|---|
| `MOVE_TO` | `point: {frame_id, x, y}` | Navigate mobile base to clicked visible floor point (preserving heading) or to a collision-free standoff pose facing a clicked furniture/object (`0.75..1.05 m`). Resets `yaw=pitch=-12°`, `zoom=1.0`. |
| `TURN` | `direction: "left" \| "right" \| "back"` | Rotate mobile base in place (`+90°`, `-90°`, `180°`) after 360° base radial ray clearance check (`0.42 m`). Resets `yaw=pitch=-12°`, `zoom=1.0`. |
| `OPEN` | `point: {frame_id, x, y}` | Actuate clicked closed articulated drawer or refrigerator door (`<= 0.94 m` XY distance from base). |
| `CLOSE` | `point: {frame_id, x, y}` | Actuate clicked open articulated drawer or refrigerator door (`<= 0.94 m` XY distance from base). |
| `GRASP` | `point: {frame_id, x, y}` | Pick up clicked movable object (`<= 0.86 m` XY distance, min AABB span `<= 0.09 m`, empty hand, clear arm ray path). |
| `PLACE` | `point: {frame_id, x, y}` | Release currently held object onto clicked horizontal support surface (`normal_z >= 0.8`, `<= 0.86 m` XY distance) and settle 80 PhysX steps. |
| `VIEW` | `yaw: [-180,180], pitch: [-90,90], zoom: [1,4]` | Adjust virtual head pinhole gaze without moving the robot body. |
| `CHECK` | `point: {frame_id, x, y}` | Isolate the clicked visible movable object and return a $720\times 450$ 4-view orthographic/perspective inspection sheet (`30°, 120°, 210°, 300°`). **Cannot be used as a point-action coordinate frame.** |
| `REPORT_DONE` | *(none)* | Terminate episode and trigger goal & restoration evaluation. |

### 3.2 Observation, Action, and Termination Accounting
- **Step Budget**: Default `max_planning_steps = 120` (`planning_step` starts at `0`).
- **Frame Freshness**: Every processed `act` call increments `planning_step` by `1` and mints a fresh `frame_id` (except post-termination calls, which are rejected without charging steps or mutating state). Point actions (`MOVE_TO`, `OPEN`, `CLOSE`, `GRASP`, `PLACE`, `CHECK`) must reference the current scene `frame_id`; stale or `CHECK` frame IDs return `INVALID_ACTION` and consume `1` step.
- **Four Public Feedback Codes**: `EXECUTED`, `INVALID_ACTION`, `OUT_OF_CAPABILITY`, `UNSAFE`. Fine failure reasons (`private_error`, `measurements`, `physx_steps`) are written exclusively to evaluator-private logs.
- **Terminal Statuses (`episode_status`)**:
  - `RUNNING`: `planning_step < max_planning_steps` and `REPORT_DONE` not yet submitted.
  - `SUCCESS`: `REPORT_DONE` submitted (including on step `120`) and goal predicate holds.
  - `FAIL_FALSE_COMPLETION`: `REPORT_DONE` submitted and goal predicate does not hold. (Takes precedence over `FAIL_MAX_STEPS` if `REPORT_DONE` is submitted on the final budget step).
  - `FAIL_MAX_STEPS`: `planning_step == max_planning_steps` reached via a non-`REPORT_DONE` action.
- **MCP Transport (`serve-mcp --backend rab`)**:
  - Stdio process never imports Kit/OmniGibson; it manages an isolated RAB skill worker subprocess (with dedicated `TMPDIR`, clean `build/rab_mcp_empty_edits` overlay isolation, and stdout/stderr redirected to `worker.log` so stdio remains pure JSON-RPC).
  - `reset_episode(scenario_id)` spawns a fresh native world (guaranteed clean state for same-ID or cross-ID resets) and returns `[TextContent(metadata_json), ImageContent(image/png)]`.
  - `observe()` returns `[TextContent(metadata_json), ImageContent(scene_png)]` (plus a second `ImageContent(check_png)` when `inspection_view` is active on the current step) without advancing physics or consuming budget.
  - `act(action)` forwards verbatim to the authoritative RAB skill worker and returns `[TextContent(result_and_obs_json), ImageContent(scene_png), ...]` (plus `ImageContent(check_png)` if `CHECK` succeeded).

---

## 4. Metrics & Evaluation Accounting

### 4.1 Success Rate (`SR`), Step Cost, and Completion Time (`ST`)
- **Goal Predicate (`SR = 1`)**: Upon `REPORT_DONE`:
  1. All required target objects (`r['target'] == True`, matching requested count $k \in \{1,2,3,4\}$) are currently supported (`support(r)`) by an authorized delivery surface in `episode['delivery']['placements']`.
  2. The robot is not holding any object (`held is None`).
  3. Across a 30-step native PhysX settling window, every target object remains on its delivery support and moves $< 0.012\text{ m}$ (`1.2 cm`).
- **Restoration Exclusion from Primary `SR`**: Restoration score is reported alongside `SR` in the main table and never merged into primary `SR` unless an explicit restoration-constraint diagnostic condition defines a joint threshold.
- **Disambiguated Time & Step Columns (`ST`)**:
  - `ST_wall_s` (v1 primary `ST`): Total episode wall-clock completion time (seconds) for **successful** episodes (`t_episode_end - t_reset_ready`), excluding simulator startup/USD load time.
  - `ST_steps`: Planning steps consumed (`planning_step` at termination) for **successful** episodes.
  - **Timing Breakdown (all valid episodes)**: Every episode log records `load_wall_s` (worker boot + USD/PhysX init), `model_wall_s` (VLM inference + client transport), `sim_tool_wall_s` (MCP `act`/`observe` execution including PhysX stepping and rendering), and `total_episode_wall_s`.
  - **Accounting Denominators**: Main tables report `N_assigned`, `N_valid`, `N_success`, `N_false_completion`, `N_timeout`, `N_infra_invalid`, and `N_retried`. `SR = N_success / N_valid`. Efficiency is never extrapolated from winners-only `ST` without pairing against `SR`. Uncertainty on pilot subsets is reported via layout-cluster bootstrap 95% confidence intervals (clustering by `layout_id` / `arch_group`).

### 4.2 Continuous Restoration Score (`Restoration` $\in [0, 1]$)
Reuses `scripts/rab_restore.py` (`score_restoration`) over all non-target objects (`target == False`) and task articulations:
- **Non-target bodies**:
  - Must match initial observed `support` path; otherwise body restoration score is `0.0`.
  - Position error $d$ (in world frame, or drawer-relative frame `drawer_relative_xyz_m` for drawer contents) and quaternion orientation difference $\Delta\theta$ are scored via Gaussian kernel:
    $$s_{\text{body}} = \exp\left(-\frac{1}{2}\left[\left(\frac{d}{\sigma_p}\right)^2 + \left(\frac{\Delta\theta}{\sigma_\theta}\right)^2\right]\right), \quad \sigma_p = 0.05\text{ m}, \quad \sigma_\theta = 0.35\text{ rad}$$
- **Articulated joints (drawers & fridge doors)**:
  - Measured directly from native rigid-body pose difference relative to `hinge_start` (never from drive target command):
    - Prismatic drawers (`articulations_m`): $s_{\text{joint}} = \exp\left(-\frac{1}{2}(\Delta x / \sigma_s)^2\right)$ with $\sigma_s = 0.04\text{ m}$.
    - Revolute fridge doors (`articulations_deg`): $s_{\text{joint}} = \exp\left(-\frac{1}{2}(\Delta\phi / 12^\circ)^2\right)$.
- **Aggregate**: Arithmetic mean over all non-target bodies and tracked articulations.
- **Four Pre-Freeze Calibration Controls**: Verified on (1) untouched initial state (`score == 1.0`), (2) minor perturbation (`2 cm` shift / `5°` tilt), (3) wrong support / dropped object (`body score == 0.0`), and (4) drawer/door left open (`joint score` decay).

### 4.3 Difficulty Stratification & Structural Covariates
- **Primary Difficulty Axis**: Human reference step cost ($S_{\text{human}} + O_{\text{human}}$) and completion status under the identical public `findingbench_rab_pinhole_v1` interface.
  - Records: operator ID, trial count, step budget, success count, bin boundaries, and required open/close or occluder moves.
  - Until full human trials across all 30 pilot tasks are completed, difficulty bins are explicitly labeled `provisional_structural_tier` and never misrepresented as human oracle data.
- **Separately Logged Structural Covariates**: Candidate search zones, lookalike distractors, occlusion layers, and spawn-to-source/delivery Euclidean distance.

---

## 5. Figures & Two-Level Failure Taxonomy

### 5.1 Figure 1: Search Uncertainty Decay / Candidate Elimination (`DER-n`)
- **Formalism** (`docs/household_search_formalization.md`): Under $n$ equally likely candidate locations ($p_i = 1/n$), single-candidate perfect inspection, and zero inter-candidate travel difference, expected inspections are $(n+1)/2$ and worst-case $n$; under separable costs $q_i$, optimal ordering sorts by $q_i / p_i$.
- **Empirical Curve**: Only computed when a task has a pre-defined discrete candidate set $\mathcal{L}$, prior $p_0$, and auditable visual elimination rule ($H_t = -\sum_{i \in \mathcal{L}} p_t(i) \log_2 p_t(i)$, $\text{DER-}n = H_t - H_{t+n}$, base-2 bits). If dynamic partial observability prevents exact posterior tracking without GT leakage, report verifiable **Remaining Uninspected Plausible Candidate Containers/Zones vs. Planning Step**, or mark `DER: NOT_RUN`. Never use ground-truth target coordinates or uncalibrated model self-confidence as $p_t$.

### 5.2 Figure 2a: Episode-Level Outcome Taxonomy (Denominator: `N_valid` episodes; `N_infra_invalid` listed separately)
Every valid episode is assigned to exactly one mutually exclusive terminal category:
1. `SUCCESS`: Goal predicate satisfied on `REPORT_DONE`.
2. `FAIL_UNFOUND`: Target object was never visible in any primary RGB or `CHECK` view and never grasped.
3. `FAIL_MISIDENTIFIED`: Target was visible (or a distractor was inspected/grasped), and the agent grasped/delivered a non-target distractor instead of the target.
4. `FAIL_WRONG_DELIVERY_OR_FALSE_DONE`: Agent grasped the true target (or submitted `REPORT_DONE` early) but placed it on the wrong support, failed to release/settle all required target copies, or called `REPORT_DONE` before completing delivery.
5. `FAIL_EXECUTION_BLOCKED`: Agent visibly located and repeatedly targeted the relevant container/occluder/target, but low-level execution rejections (`OUT_OF_CAPABILITY`, `UNSAFE`, or post-actuation slippage) prevented progress until termination.
6. `FAIL_BUDGET_EXHAUSTED`: Episode reached `max_planning_steps` (`120`) during search/navigation without triggering categories 2–5.
*(Note: Low restoration score `< 0.8` on otherwise finished episodes is tracked as an orthogonal diagnostic rate, not conflated with task failure).*

### 5.3 Figure 2b: Action-Level Execution & Rejection Taxonomy (Denominator: Total submitted actions $M$)
Every submitted action is classified by its public feedback AND evaluator-private branch (`private_error` / `measurements`):
1. `EXECUTED`: Action passed all preconditions and postconditions.
2. `REJECTED_ACTION_OR_STATE_INAPPLICABLE` (`INVALID_ACTION`):
   - `STALE_OR_INVALID_FRAME`: `frame_id` does not match current scene `frame_id` (or used `CHECK` `frame_id`).
   - `SCHEMA_OR_PARAM_ERROR`: Malformed JSON, out-of-range coordinate/angle/zoom, or unknown skill.
   - `NO_SURFACE_OR_ENTITY_HIT`: Ray missed scene geometry, clicked non-movable geometry for `GRASP`/`CHECK`, clicked non-articulated geometry for `OPEN`/`CLOSE`, or clicked non-horizontal surface (`normal_z < 0.8`) for `PLACE`.
   - `STATE_PRECONDITION_VIOLATED`: `GRASP` while hand occupied, `PLACE` while hand empty, `OPEN` already-open joint, `CLOSE` already-closed joint, or action after termination.
3. `REJECTED_UNREACHABLE_OR_IK` (`OUT_OF_CAPABILITY`):
   - `STANDOFF_DISTANCE_EXCEEDED`: Target XY distance $> 0.86\text{ m}$ (`GRASP`/`PLACE`) or $> 0.94\text{ m}$ (`OPEN`/`CLOSE`).
   - `ARM_IK_CONVERGENCE_FAILED`: 60-step Jacobian position drive did not converge within tolerance (`0.022 m` / `0.014 m`).
4. `REJECTED_RISK_OR_COLLISION_PRECHECK` (`UNSAFE` — pre-actuation refusal, physics not driven or cleanly aborted):
   - `NAV_NO_CLEAR_ROUTE`: `MOVE_TO` lazy A* (`stride=0.36 m`, 850 nodes) found no collision-free base path (`clear_base` height bands `0.13..1.35 m` or floor normal/step check).
   - `TURN_RADIAL_CLEARANCE_BLOCKED`: `TURN` 16-ray radial check (`0.42 m` radius at heights `0.17..1.1 m`) hit an obstacle.
   - `ARM_PATH_BLOCKED_OR_OVERSIZED`: `arm_path_clear` ray hit an obstacle or object minimum AABB dimension $> 0.09\text{ m}$.
5. `FAILED_POST_ACTUATION_PHYSICS` (`UNSAFE` or `INVALID_ACTION` after driving physics):
   - `BASE_MOTION_SHORTFALL`: Mobile base velocity ramp finished with progress error $> 0.075\text{ m}$ (drive) or $> 0.10\text{ rad}$ (turn).
   - `GRASP_LIFT_OR_SLIP_FAILED`: Gripper closed and lifted, but object vertical lift $< 0.055\text{ m}$ or slipped $> 0.16\text{ m}$ from hand during transport.
   - `PLACE_SETTLE_MISMATCH`: Object released and settled 80 steps, but final support $\neq$ intended support or XY placement error $> 0.12\text{ m}$.
   - `MECHANISM_TRAVEL_OR_CONTENT_DRIFT`: Drawer/door driven, but travel $< 0.10\text{ m}$ / $< 45^\circ$ or internal items shifted $> 0.05\text{ m}$ (automatically swept back).
6. `OTHER_OR_UNKNOWN`: Unclassified simulator exception.

---

## 6. Controlled Diagnostic Design (Single-Factor Ablations)

All diagnostic comparisons hold the pilot task subset, starting states, executor version, and step budget (`120`) constant, changing **only one factor at a time**:
1. **Prompt Constraint / Restoration Reminder (Primary Week-1 Diagnostic)**:
   - *Baseline*: Natural household request only (no mention of `CHECK`, restoration, or avoiding collisions).
   - *Controlled Variant*: Appends explicit safety/restoration reminder (`"Please return any moved non-target items and close any opened drawers or refrigerator doors before finishing."`).
2. **Interaction History Horizon**:
   - *Full History*: Complete multi-turn action + feedback text history (current image attached to latest turn).
   - *Short History*: Sliding window of last $K = 5$ action-feedback turns.

---

## 7. Items Explicitly Marked `[PENDING FREEZE]` (To Resolve Before Nov 24 Freeze)

1. `[PENDING FREEZE - F164/F275 Navigation Diagnostic]`: Classify the two `UNSAFE` navigation smoke outcomes in `runs/rab_spectator_transfer_20261119/native_navigation_smoke.json` (distinguishing test-script ray selection vs. `clear_base` / `motion` shortfall) and verify whether a general executor fix is required before locking the executor hash.
2. `[PENDING FREEZE - Pilot End-to-End Playability Gate]`: Run public point-selection verification across the 30 primary pilot candidates; log any physical block in `exclusion_ledger` and promote the pre-ranked reserve from `configs/benchmarks/rab_v1_pilot_draft.json`.
3. `[PENDING FREEZE - Restoration 4-Control Calibration]`: Execute the 4 synthetic/controlled restoration probes (untouched, minor shift, wrong support, open door/drawer) and attach the calibration JSON receipt.
4. `[PENDING FREEZE - Human Reference Stratification]`: Record at least one complete human closed-loop trajectory under `findingbench_rab_pinhole_v1`; mark remaining unmeasured pilot strata as `provisional_structural_tier`.
