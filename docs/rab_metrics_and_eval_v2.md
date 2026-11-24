# RummageBench — Metrics, Proprioceptive Feedback & Scalable Memory Evaluation (v2)

> [!IMPORTANT]
> **Core Research Positioning ("Complete Cerebellum, Bottlenecked Brain")**
> RummageBench encapsulates low-level motor execution (**Cerebellum**) into physically grounded skill primitives backed by Isaac Sim / PhysX 60 Hz dynamics, NVLabs cuRoboV2-Fast GPU IK, and multi-band ray clearance validation. Any skill refusal or episode failure directly reflects a high-level **Brain (VLM / Agent)** bottleneck — spatial reach misjudgment, collision/occlusion oversight, affordance misgrounding, visual misidentification, or working-memory decay.
>
> Every metric and failure label is **100% deterministic** from simulator traces (`zero LLM-as-a-judge`).

---

## 1. Core Orthogonal Metrics (`scripts/rab_eval_metrics.py`)

| Dimension | Metric | Definition | Direction | Diagnostic Focus |
|---|---|---|---|---|
| **01 Outcome** | **SR** (Success Rate) | $\text{SR} = N_{\text{success}} / N_{\text{episode}} \times 100\%$ (+ Wilson 95% CI) | Higher is better | Task completion within budget (`REPORT_DONE` with target stably settled on destination support and empty grippers). |
| **02 Trajectory Cost** | **ST** (Steps) | $\text{ST}_{\text{succ}} = \frac{1}{N_{\text{success}}}\sum_{e \in \text{succ}} T_e$ (plus $\text{ST}_{\text{all}}$) | Lower is better | Semantic search prior and multi-step planning efficiency. |
| **03 Execution Validity** | **EFF** ($\text{EFF}_{\text{phys}}$ / $\text{EFF}_{\text{all}}$) | $\text{EFF}_{\text{phys}} = N_{\text{EXECUTED, phys}} / N_{\text{called, phys}} \times 100\%$ over `{MOVE_TO, TURN, OPEN, CLOSE, GRASP, PLACE}` | Higher is better | Single-step embodied physical feasibility hit rate (anti-gaming guard excludes pure head `VIEW` from $\text{EFF}_{\text{phys}}$). |
| **04 Inference Latency** | **LAT** (Inter-Step Agent Latency) | $\text{LAT} = \frac{1}{N_{\text{act}}}\sum_{k=1}^{N_{\text{act}}} \max(0,\; (t_k - w_{\text{sim},k}) - t_{k-1} - w_{\text{obs},k})$ (`s/step`, plus p50/p95) | Lower is better | Pure agent thinking wall-time per planning step after subtracting physics/rendering execution time. |
| *Hygiene* | **RES** (Restoration Score) | Mean scene restoration score (`0–100%`) | Higher is better | Closing opened drawers/fridge doors and keeping non-target items undisturbed. |
| *Error Recovery* | **RSER** (Repeat-Same-Error Rate) | $\text{RSER} = N_{\text{consecutive same detail\_code}} / N_{\text{refused}} \times 100\%$ | Lower is better | Zero-hint error reflection vs. repeating identical physical refusals back-to-back. |

---

## 2. Deterministic Two-Layer Failure Taxonomy

### Layer A — Skill-Level Refusal Attribution (Decomposes $1 - \text{EFF}$, sums to 100%)
1. **Spatial & Workspace Awareness**: `GRASP_TOO_FAR`, `PLACE_TOO_FAR`, `ARTICULATION_TOO_FAR`, `ARM_IK_TARGET_FAILED`, `ARM_IK_PLACE_FAILED`, `ARM_REACH_FAILED`
2. **Collision & Safety Clearance**: `NAV_NO_CLEAR_ROUTE`, `NAV_SEGMENT_CLEARANCE_BLOCKED`, `TURN_RADIAL_CLEARANCE_BLOCKED`, `BASE_MOTION_SHORTFALL`, `HAND_PATH_BLOCKED`, `ARTICULATION_BLOCKED`
3. **Physical Affordance & Stability**: `UNSAFE_OVERSIZED_GRASP`, `UNSAFE_PLACEMENT_UNSTABLE`, `CLICK_NON_HORIZONTAL_SURFACE`
4. **Visual Grounding & Affordance**: `CLICK_NON_INTERACTABLE`, `CLICK_NON_ARTICULATED`
5. **State Tracking & Precondition Memory**: `PRECONDITION_HAND_OCCUPIED`, `PRECONDITION_HAND_EMPTY`, `PRECONDITION_ALREADY_OPEN`, `PRECONDITION_ALREADY_CLOSED`, `PLACE_ON_SELF`, `STALE_FRAME_ID`, `SCHEMA_INVALID`, `UNSUPPORTED_SKILL`, `EPISODE_ALREADY_TERMINATED`

*(Harness/runtime `INFRA_EXCEPTION` events are excluded from model scores and reported separately.)*

### Layer B — Episode-Level Terminal Attribution (Decomposes $1 - \text{SR}$, sums to 100%)
- **Under `FAIL_FALSE_COMPLETION`**: `TARGET_NOT_RELEASED` → `TARGET_UNSTABLE` → `PARTIAL_DELIVERY` → `WRONG_TARGET_DELIVERED` → `WRONG_DESTINATION` → `WRONG_TARGET_MANIPULATED` → `PREMATURE_DONE`
- **Under `FAIL_MAX_STEPS`**: `TARGET_ENGAGED_NOT_DELIVERED` → `CONTAINER_OPENED_TARGET_NOT_ENGAGED` → `TARGET_NEVER_ENGAGED`

---

## 3. Zero-Hint Proprioceptive Feedback (`proprioceptive_v2`, `scripts/rab_feedback.py`)

Controlled via `FINDINGBENCH_RAB_FEEDBACK_MODE=proprioceptive_v2|four_class_v1`. Preserves top-level 4-class `code` (`EXECUTED`, `INVALID_ACTION`, `OUT_OF_CAPABILITY`, `UNSAFE`) and adds factual proprioceptive `detail_code` (`NAV_BLOCKED`, `TURN_BLOCKED`, `NO_ARTICULATION_HIT`, `OUT_OF_REACH`, `JOINT_AT_LIMIT`, `NO_TARGET_AT_POINT`, `GRIPPER_OCCUPIED`, `IK_INFEASIBLE`, `ARM_PATH_OBSTRUCTED`, `OBJECT_EXCEEDS_GRIPPER`, `GRIPPER_EMPTY`, `SURFACE_NOT_LEVEL`, `INVALID_SUPPORT`, `PLACEMENT_UNSTABLE`, `STALE_FRAME`, `PROTOCOL_ERROR`) under two strict scientific invariants:
1. **Zero Directive Hinting**: Reports physical/state facts only — never suggests next actions.
2. **Zero Privileged Information Gain**: Never leaks object identities, semantic classes, scene graph paths, or private metric thresholds.

---

## 4. 3-Tier Self-Maintained Agentic Memory, Prefix Caching & Hot-Worker Reload

Implemented in [`scripts/rab_agent_memory.py`](../scripts/rab_agent_memory.py), [`scripts/run_rab_mcp_eval.py`](../scripts/run_rab_mcp_eval.py), and `POST /reset` in [`scripts/rab_spectator.py`](../scripts/rab_spectator.py) / [`scripts/rab_skills.py`](../scripts/rab_skills.py):

1. **Tier 1 — Self-Authored Working Scratchpad (`memory`)**: The evaluated VLM outputs `{"memory": "...", "action": {"skill": ...}}` (backward-compatible with bare `{"skill": ...}`). Only the model's *own* previous `"memory"` string (`<= 600` chars) is echoed back on step $t+1$ — zero simulator ground-truth injection.
2. **Tier 2 — Lossless Compact Action & Proprioceptive Trace**: Compresses steps `1..t` into ~14 tokens/step (`#03 OPEN(0.490,0.550) -> OUT_OF_CAPABILITY[OUT_OF_REACH]`), stripping stale historical `frame_id`s.
3. **Tier 3 — $O(1)$ Short-Term Visual Memory Strip**: Resizes primary pinhole RGB to `960x540` (preserving normalized `[0,1]` coordinates) and attaches a single `772x238` side-by-side thumbnail strip of frames `t-2` and `t-1` (explicitly marked non-selectable).
4. **100% Prefix-Cache-Aligned System Prompt**: Keeps `9 skills + coordinate rules + static embodiment_spec + episode instruction` byte-identical across steps `0..T`, cutting effective billed tokens on a 50-step episode to **~79.7k tokens** (`~73x` cheaper than multi-turn coding-agent swarms and `2.68x` cheaper than unsplit legacy JSON while adding full multimodal memory).
5. **Persistent Hot-Worker Stage Reload (`POST /reset`) & `RummageBench-Core-200`**: Reuses the live `SimulationApp` + compiled `UnitreeCuRoboSolver` CUDA graphs across up to 50 consecutive episodes, reducing `reset_episode` from `~105s` cold boot to **`2.6s–3.1s` (`34x–40x` speedup)**. [`configs/benchmarks/rab_v1_core200.json`](../configs/benchmarks/rab_v1_core200.json) provides an exact 20% proportional stratified 200-episode split (`4 tasks x 50 layouts`).
