# Embodied Agent & Interactive Search Benchmarks: Literature Review & Table 1 Comparison

- **alphaXiv Library Folder**: `agent bench` (`folder_id: 01a1f6bc-d6ad-729c-a0a2-024afb9a65b4`, 18 core benchmark & baseline papers indexed and saved).
- **Purpose**: Provide a rigorous, generalizable **Table 1 comparison** for a benchmark paper and verify that RummageBench's core scientific thesis—**decoupling high-level interactive search planning from low-level embodied execution across dense household clutter & articulated containers with continuous environment restoration**—occupies a clear, non-overlapping niche in the literature.

---

## 1. Generalizable Benchmark Evaluation Dimensions (Table 1 Axes)

To make our Table 1 broadly informative for reviewers across robotics (RSS/ICRA/CoRL) and AI/vision (NeurIPS/ICLR/CVPR), we define **10 generalizable benchmark dimensions** grouped into four structural pillars:

### A. Environment & Spatial-Physical Realism
1. **3D Cont. (`Continuous 3D Metric Space & Physics`)**: Whether the agent operates in continuous 3D Euclidean space with rigid/articulated body physics (PhysX / MuJoCo / Bullet), rather than discrete 1D/symbolic state transitions or pre-baked topological navigation graphs.
2. **House Nav. (`Multi-Zone Household Mobile Manipulation`)**: Whether tasks span room-scale or multi-zone household environments requiring mobile base navigation combined with arm manipulation (vs. tabletop-only fixed-base arms).
3. **Dense Occ. (`Dense Local Clutter & Multi-Candidate Containers`)**: Whether target search zones systematically contain dense local distractors (e.g., $\sim 10.7\text{ cm}$ median nearest-neighbor distance inside drawers, $4+$ movable occluders on open surfaces) and articulated containers (drawers, refrigerators) requiring physical interaction to reveal hidden targets.

### B. Perception & Active Information Gathering
4. **Partial Obs. (`Strict Egocentric Visual Partial Observability`)**: Whether the evaluated policy receives strictly egocentric RGB images without privileged scene graphs, ground-truth 3D bounding boxes, object IDs, or dynamic admissible action lists.
5. **Active View (`Dedicated Non-Prehensile Active Perception & Close-Up Inspection`)**: Whether the action space provides explicit active visual exploration primitives—both continuous virtual gaze (`VIEW` yaw/pitch/zoom) and close-up multi-angle object inspection (`CHECK`) to resolve fine-grained visual ambiguity among lookalike items.

### C. Architecture & Diagnostic Decoupling (Our Core Methodological Contribution)
6. **H/L Decoupled (`Explicit High-Level Planning vs. Low-Level Execution Boundary`)**: Whether the benchmark defines a fixed, model-agnostic 2D pixel-grounded skill contract (`findingbench_rab_pinhole_v1`) that separates high-level semantic/spatial decision-making from low-level geometric/kinematic feasibility and physical actuation.
7. **2-Tier Attr. (`Two-Level Planning vs. Execution Failure Attribution`)**: Whether the evaluator logs paired public feedback (`EXECUTED`, `INVALID_ACTION`, `OUT_OF_CAPABILITY`, `UNSAFE`) and private geometric/physics traces to disentangle cognitive/search errors (`FAIL_UNFOUND`, `FAIL_MISIDENTIFIED`, `FAIL_FALSE_COMPLETION`) from low-level embodiment bottlenecks (`STANDOFF_EXCEEDED`, `ARM_IK_FAILED`, `NAV_BLOCKED`, `POST_ACTUATION_SLIP`).
8. **Math Theory (`Analytical Search-Cost & Information-Decay Formalization`)**: Whether the benchmark provides a formal decision-theoretic search cost baseline (e.g., separable inspection cost ordering $q_i / p_i$, expected inspections $(n+1)/2$, and step-wise uncertainty/candidate reduction) decoupled from travel distance.

### D. Household Collaboration & Protocol Standardization
9. **Restore (`Continuous Non-Target & Articulation Restoration Metric`)**: Whether the benchmark quantitatively measures how well the robot restores non-target clutter (moved occluders, support relations, position/orientation drift) and closes opened drawers/doors after retrieving the target.
10. **MCP Native (`Standardized Model Context Protocol Visual Tool Interface`)**: Whether the benchmark provides a native, simulator-isolated stdio MCP server (`reset_episode`, `observe`, `act`, `episode_status` with native PNG `ImageContent`) for zero-boilerplate evaluation of frontier and local VLMs.

---

## 2. Table 1: Systematic Comparison of Embodied & Agent Benchmarks

Legend:
- **✓**: Fully supported as a core benchmark contract.
- **◐**: Partially supported (e.g., symbolic/discrete subset, tabletop-only, or single-tier heuristic attribution without execution decoupling).
- **✗**: Not supported or out of scope.

| Benchmark / Paper | Venue / Year | Primary Task Paradigm | 1. 3D Cont. | 2. House Nav. | 3. Dense Occ. | 4. Partial Obs. | 5. Active View | 6. H/L Decoupled | 7. 2-Tier Attr. | 8. Math Theory | 9. Restore | 10. MCP Native |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mechanical Search** ([1903.01588](https://www.alphaxiv.org/abs/1903.01588)) | ICRA 2019 | Tabletop bin target extraction | ✓ | ✗ | ✓ | ✓ | ✗ | ◐ | ✗ | ◐ | ✗ | ✗ |
| **ALFRED** ([1912.01734](https://www.alphaxiv.org/abs/1912.01734)) | CVPR 2020 | Egocentric household instruction following | ◐ | ✓ | ✗ | ✓ | ◐ | ◐ | ✗ | ✗ | ✗ | ✗ |
| **Habitat 2.0 / HAB** ([2106.14405](https://www.alphaxiv.org/abs/2106.14405)) | NeurIPS 2021 | Household object rearrangement | ✓ | ✓ | ◐ | ✓ | ✗ | ◐ | ✗ | ✗ | ◐ | ✗ |
| **BEHAVIOR-1K** ([2403.09227](https://www.alphaxiv.org/abs/2403.09227)) | CoRL 2022/24 | 1,000 everyday household activities | ✓ | ✓ | ◐ | ✓ | ✗ | ◐ | ✗ | ✗ | ◐ | ✗ |
| **VisualAgentBench** ([2408.06327](https://www.alphaxiv.org/abs/2408.06327)) | arXiv 2024 | GUI, web & symbolic embodied foundation agents | ◐ | ◐ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **ManiSkill3** ([2410.00425](https://www.alphaxiv.org/abs/2410.00425)) | arXiv 2024 | GPU-parallelized low-level manipulation | ✓ | ◐ | ◐ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **VLABench** ([2412.18194](https://www.alphaxiv.org/abs/2412.18194)) | arXiv 2024 | Tabletop language-conditioned VLA reasoning | ✓ | ✗ | ◐ | ✓ | ✗ | ◐ | ✗ | ✗ | ✗ | ✗ |
| **EmbodiedBench** ([2502.09560](https://www.alphaxiv.org/abs/2502.09560)) | arXiv 2025 | Multi-tier MLLM benchmark (ALFRED/Habitat/ManiSkill) | ✓ | ✓ | ◐ | ✓ | ✗ | ◐ | ◐ | ✗ | ✗ | ✗ |
| **CubeBench** ([2512.23328](https://www.alphaxiv.org/abs/2512.23328)) | arXiv 2025 | Symbolic/3D Rubik's cube mental state tracking | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **AsgardBench** ([2603.15888](https://www.alphaxiv.org/abs/2603.15888)) | arXiv 2026 | Visually grounded interactive planning (AI2-THOR) | ◐ | ✓ | ✗ | ✓ | ◐ | ✓ | ◐ | ✗ | ✗ | ✗ |
| **HumanCLAW** ([2607.27180](https://www.alphaxiv.org/abs/2607.27180)) | arXiv 2026 | Full-body humanoid navigation & interaction (HSSD) | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **ActiveArena** ([2609.24124](https://www.alphaxiv.org/abs/2609.24124)) | arXiv 2026 | Tabletop active viewpoint exploration & memory VLA | ✓ | ✗ | ✓ | ✓ | ✓ | ◐ | ◐ | ✗ | ✗ | ✗ |
| **LIVIN** ([2610.12069](https://www.alphaxiv.org/abs/2610.12069)) | arXiv 2026 | Digital twins of 30 lived-in homes (nav & loco-manip) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **RobotAgentBench (RAB)** ([2610.34649](https://www.alphaxiv.org/abs/2610.34649)) | arXiv 2026 | 20 Hz continuous VLA/agent household tasks & discovery | ✓ | ✓ | ◐ | ✓ | ✗ | ◐ | ◐ | ✗ | ◐ | ✗ |
| **RummageBench (Ours)** | **2026** | **Collaborative dense household search, delivery & restoration** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## 3. Detailed Review of Key Related Papers & Why RummageBench Does Not Collide

### 3.1 RobotAgentBench (RAB, Mao et al., [2610.34649](https://www.alphaxiv.org/abs/2610.34649))
- **What RAB does**: Provides 50 real-to-sim `LIVING_STRAIGHT` room layouts on Isaac Sim 5.1 with the AgiBot G1 bimanual mobile manipulator (`1280x800` head camera + dual `848x480` wrist cameras), evaluating 18 short manipulation tasks, 8 long tasks, Task Discovery ($N+2$ physical finger-contact budget), Context Reasoning, Failure Recovery, Robustness, and 4 OpenEnd scenarios via a **20 Hz continuous joint-target + base-velocity control interface** (or optional code-execution broker).
- **How RummageBench differs & complements RAB**:
  1. **Explicit Asset Attribution**: We explicitly credit RAB (`0.1.7`) as our reusable upstream physical asset & robot foundation—never claiming the 50 room layouts or 37 object assets as our own creation.
  2. **Task Paradigm**: RAB's *Task Discovery* measures whether a 20 Hz policy physically touches $N$ target objects within an $N+2$ contact budget; RAB's *Context Reasoning* tests single-item selection. RummageBench formulates **1,000 dense collaborative household search-retrieve-deliver-restore episodes** (872 dual-drawer, 123 open-occluded, 5 refrigerator) where targets are systematically concealed inside closed drawers or behind movable foreground clutter (`median 10.7–15.3 cm` local distractor spacing).
  3. **Action Abstraction & Diagnostic Goal**: RAB primarily evaluates end-to-end 20 Hz continuous joint trajectories (`20 DoF + 2 base velocities`), which entangles VLM high-level search reasoning with low-level trajectory generation. RummageBench introduces the **9-skill pixel-grounded public contract (`findingbench_rab_pinhole_v1`)** over native PhysX execution, plus `CHECK` 4-view close-up inspection and a continuous non-target & articulation **restoration metric (`score_restoration`)**, specifically to disentangle high-level search/planning failures from low-level execution constraints.

### 3.2 HumanCLAW (Siyao et al., [2607.27180](https://www.alphaxiv.org/abs/2607.27180))
- **What HumanCLAW does**: Evaluates VLMs controlling a simulated humanoid body in HSSD scenes across find-navigate-sit/touch tasks, using a 4-stage pipeline (structured perception $\to$ 6-step mid-level goal $\to$ skill selection $\to$ VLM verifier) and deterministic offline failure attribution (Find / Navigate / Interact).
- **How RummageBench differs**:
  1. **Task & Embodiment**: HumanCLAW focuses on humanoid whole-body locomotion and sitting/touching in relatively sparse HSSD rooms without dense container rummaging, movable occlusion clearing, multi-object delivery, or environment restoration.
  2. **No Intervening Verifier**: HumanCLAW uses an internal VLM verifier (`v3`) and retry loop that intercepts and modifies proposed actions before execution. RummageBench enforces a strict **1-inference = 1-submitted-action** contract via stdio MCP so that raw high-level policy quality and error recovery from 4-class public feedback are measured without harness intervention.

### 3.3 CubeBench (Gao et al., [2512.23328](https://www.alphaxiv.org/abs/2512.23328))
- **What CubeBench does**: Diagnoses long-horizon 3D spatial reasoning, mental state tracking, and active exploration under partial observation using a symbolic/visual **Rubik's Cube** testbed.
- **How RummageBench differs**: CubeBench deliberately removes physical embodiment, continuous 3D scene geometry, navigation, and kinematic reachability by reducing the world to a deterministic finite-state Rubik's Cube. RummageBench asks the complementary embodied question: **when an agent must search in a continuous 3D household with real PhysX collisions, bimanual arm kinematics, and movable occluders, how do high-level belief/planning bottlenecks interact with low-level execution limits?**

### 3.4 AsgardBench (Tupini et al., [2603.15888](https://www.alphaxiv.org/abs/2603.15888))
- **What AsgardBench does**: Evaluates visually grounded interactive planning in AI2-THOR under minimal feedback, demonstrating that many prior benchmarks (like ALFRED) allow text-only agents to succeed via linguistic priors or rich corrective error messages. It introduces automatic stand-pose positioning when clicking furniture.
- **How RummageBench differs**: AsgardBench uses AI2-THOR's symbolic magic-snap object state transitions (clicking an object teleports it into the agent's abstract inventory without arm kinematics, gripper geometry, or physical settling). In contrast, while RummageBench adopts pixel-grounded skill invocation (`findingbench_rab_pinhole_v1`), **every skill in RummageBench drives native PhysX actuation** (A* base velocity ramp, Jacobian arm trajectory, physical gripper closure, 80-step gravity settling on delivery supports) and records the exact gap between high-level pixel intent and physical execution.

### 3.5 ActiveArena (Li et al., [2609.24124](https://www.alphaxiv.org/abs/2609.24124)) & Knowing Where to Look / SSF (Agarwal et al., [2611.00499](https://www.alphaxiv.org/abs/2611.00499))
- **What they do**: ActiveArena benchmarks tabletop active viewpoint manipulation and memory writing in 35 manipulation tasks. SSF (`2611.00499`) proposes a spatial-semantic Bayesian voxel belief field for interactive object search on a mobile humanoid.
- **How RummageBench differs**: ActiveArena is restricted to fixed-base tabletop workspaces without room navigation or environment restoration. SSF is an *algorithmic method paper* (evaluated on custom office/snack setups) rather than a standardized public diagnostic benchmark with a 1,000-episode dense physical inventory, continuous restoration scoring, and two-level planning-vs-execution failure taxonomy.

### 3.6 LIVIN (Xu et al., [2610.12069](https://www.alphaxiv.org/abs/2610.12069)) & BEHAVIOR-1K $\pi_{0.5}$ Paired Evaluation (Hammele et al., [2610.37917](https://www.alphaxiv.org/abs/2610.37917))
- **What they do**: LIVIN constructs 30 lived-in home digital twins showing that dense clutter and narrow free space challenge navigation and loco-manipulation. Hammele et al. (`2610.37917`) conduct a paired evaluation of inference settings for fine-tuned $\pi_{0.5}$ on BEHAVIOR-1K (`Q-score = 0.39`), showing that end-to-end VLA inference tricks do not transfer uniformly across diverse household tasks.
- **How RummageBench complements them**: Both works highlight that dense household layouts severely degrade end-to-end policies, yet neither provides a **decoupled high-level/low-level diagnostic boundary** to tell whether a failure in dense clutter stems from **search/identification/memory** or **kinematic/collision execution**, nor do they evaluate **continuous post-search restoration**.
