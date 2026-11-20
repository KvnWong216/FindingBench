**RAB 1K 已成为默认数据。** [RAB 巡检、编辑和技能说明](docs/rab_spectator_skills.md)。`python scripts/spectator.py --task F251` 巡检，`--edit --task P136` 编辑，`python scripts/play_data.py F251` 打开机器人技能界面。以下 BEHAVIOR 命令是历史记录。

RAB 1K 已完成 1,000/1,000 个 USD 场景及 PhysX 初始化验收。最终清单及独立核验在 `runs/rab_migration_20261118/search_build_v3/release/`，参见 [交接记录](HANDOFF.md)。完整技能闭环待评估。

RAB 迁移（2026-11-18）：已按要求删除旧 BEHAVIOR 物体库、下载和 1310 个依赖旧资产的渲染缓存，释放约 220.1 GiB；1000 条原配方、306 条增强配方及人类记录保留。旧数据目录已标为退役，不能继续回放。当前 1000 个 RAB 迁移请求见 `runs/rab_migration_20261118/migration_queue.jsonl`，请求不等于完成的物理数据。资源下载／完整 SHA 校验／官方安装及原生抽屉开关诊断后台运行，实时状态见同目录 `status.json`，启动记录 `launch_http_v3.json`。多样性将分别统计架构几何、家具布局、材质和实际物体模型；50 个布局不直接当作 50 种房型，缺失资产及房间须明确重写。旧家具增强队列保持停止。详细交接见 [HANDOFF.md](HANDOFF.md)。以下 BEHAVIOR 使用说明为历史记录，部分依赖已按用户要求删除。

稀疏家具增强（2026-11-17）：按最新要求恢复全量处理。所有 1000 个父任务已进入 `runs/furniture_1k_sparse_20261117/queue_manifest.json`；后台双路处理，实时进度见同目录 `status.json`，入队不等于完成。新发布目录是 `build/search_1k_furnished_20261117`，原始任务及原缓存保留。按房间用途为每个空桌面、每个原生柜内收纳格／抽屉默认补 1 件已有资产；已有物品的区域不继续堆叠，任务核心原有布局保留。物品保持原生尺寸、重力和碰撞，按最新简化要求，经过实际碰撞表面探测、原生关系、连续 3 秒运动观测及快照／缓存检查后逐条发布；新 sparse_authoring 路径省去整屋对照与逐条独立重载，结果注明验证范围，旧 paired_native 记录保留。无法安全放置的格子记录为未覆盖，不强行摆放，也不算全部家具完成。新添加物品不得扩大 MOVE 底盘高度带内的障碍区域；没有宣称全任务技能成功率已重测。新增逻辑见 `scripts/furniture_planner.py`、`scripts/furnish_native.py`。

查看已验收增强条目：`python scripts/spectator.py --asset-run build/search_1k_furnished_20261117 --task F002`。部分发布期间使用父任务 ID，不使用原始 1–1000 序号。`status.json` 同时统计条目通过数、家具新增数、已有物品数和未覆盖数。

人工编辑数据：`conda activate ygbench` 后运行 `python scripts/spectator.py --edit --task F001`，浏览器打开 `http://127.0.0.1:8091`。左侧可用常用中文名（酱油、纸巾盒等）、英文类别或型号搜索原生资产，将结果拖入台面／已打开柜子的内表面；也可先点击结果再点击表面。点击已有物体选中，拖动改变位置，Delete 删除。选中柜门／抽屉后使用 OPEN、CLOSE。点击“切换到飞行”沿用 WASD／鼠标／穿模相机，Esc 回到编辑。任务目标、终点和房屋结构不能删除；仍有物品依赖的容器须先清空。放置时采用原生资产记录的稳定姿态（缺省为原生默认朝向），沿重力竖直下方寻找第一个平面并等待停稳；点击侧壁会先按物品宽度留出间隙。移动已有物品保留水平朝向、重置倾斜，可在左侧指定水平角度。物品仍可能自然侧躺；不是所有型号都能直立。失败提示以中文说明原因和操作建议，“最近失败的原因”保留技术详情与错误编号，日志位于 `runs/spectator_editor_errors/`。

“保存配置并更新缓存”（Ctrl+S）会先创建新快照和渲染缓存，经过独立原生重载后覆盖**当前 `--asset-run` 目录内的对应条目**，并更新发布清单与文件校验值；旧条目及元数据备份位于该目录的 `editor_revisions/`。保存失败不覆盖原条目。保存会包含当前柜门开关状态；验证期间可以移动相机。编辑模式需要原生物理环境，首次加载和保存重载都需等待；未加 `--edit` 的普通观察仍使用快速缓存。批量自动增强按 2026-11-17 的新要求另开发布版本，详见页首。

巡检加速：**1000/1000 缓存已于 2026-11-16 完成，失败 0；全量文件完整性复查通过**，见 `runs/spectator_cache_1k_20261115/completion_verification.json`。缓存保存在 `build/spectator_cache/`，再次查看同一条目会跳过物理环境构建。P007 实测首帧从 149.8 秒降至 13～14 秒；其他条目耗时尚未逐一测量。`--no-cache` 使用原始加载，`--rebuild-cache` 重建缓存；可用 `PYTHONPATH=src:scripts python scripts/build_spectator_cache.py --workers 2` 补建缺失或失效缓存。缓存仅供自由相机巡检，不是物理环境或评估输入；文件验收不等于全任务技能完成验收。

数据巡检（2026-11-15）：`conda activate ygbench` 后执行 `python scripts/spectator.py --task P007`，浏览器打开 `http://127.0.0.1:8091`。也可输入 `--task 1`，编号对应 `build/search_1k_20261115/prompts.csv`；`--list` 查看全部条目。不带 `--task` 时在终端输入编号或任务 ID。点击画面捕获鼠标，WASD 飞行、鼠标转视角、Space/E 上升、Shift/Q 下降、Ctrl 加速、滚轮调速、R 回到初始视角、F 全屏、Esc 释放鼠标。相机可以穿墙、穿柜门；物体和机器人冻结。该入口恢复每条数据自己的已验证快照及固定运行版本，不执行技能、不计入人类评估。切换数据时 Ctrl+C 退出，再用另一个编号启动；首次加载需要等待场景初始化。远程使用时转发端口：`ssh -L 8091:127.0.0.1:8091 ubuntu@<server>`，在本地浏览器打开上述地址。`--width 1920 --height 1080` 可提高细节分辨率。

生活化增强（2026-11-16）：按用户要求已停止，后台进程均已退出；保留 2 条已验收增强版（F001/F002）及其日志，不再继续批量增强。全部 1000 个父任务的队列记录保留。实时数量见 `runs/background_1k_20261116/status.json`，全量入队不代表全部完成。按房间用途从 14 类、160 个原生模型中默认选择 1 件小物（已验收的双物体试点保留），排除任务及比较样本所属语义族，避开任务家具和至少 1 米操作区；只使用现有高台面，原生支撑、稳定性、未增强对照、独立重载及缓存完整性全部通过后才进入 `build/search_1k_lived_in_20261116`。使用 `python scripts/spectator.py --asset-run build/search_1k_lived_in_20261116 --task F001` 巡检；部分发布期间请用父任务 ID 选择，原始 1～1000 编号继续对应原版。原版数据与缓存保留，尚未做动作失败率对照。断点续跑入口为 `PYTHONPATH=src:scripts python scripts/enrich_background_queue.py --workers 2`，重复运行由锁拒绝，代码修复需指定新的 `--tool-revision`，不会改写运行中的固定源码。F001 原始试点证据仍在 `runs/background_pilot_20261116`：新增杯子和苹果占台面约 0.46%，距任务操作区至少 1.72 米。

> **2026-10-19：本机多 VLM 接入**：36 个 L/S 变体已通过脚本化验证；新增 Qwen3-VL / InternVL3.5 / Llama 3.2 Vision / Molmo2 的独立本机推理与评估入口，详见 [运行说明与验证边界](docs/local_vlm_evaluation.md)。

> **2026-10-17：原生 BEHAVIOR 试跑路线**：技能库已改为 MOVE / TURN / OPEN / CLOSE / GRASP / PLACE / VIEW / REPORT_DONE，移除公共 OBSERVE。首条任务为 `carrying_in_groceries`，选定 `house_double_floor_lower`。全景 / VIEW 真实仿真验收已通过，接口、准备命令和验证边界见 [试跑交接](docs/behavior_pilot.md)。下文旧自造场景流程保留作历史参考。

人工操作：`./scripts/play_behavior.sh`，浏览器打开 `http://127.0.0.1:8090`。脚本自动激活 `ygbench`；安装与验收说明见 [环境文档](docs/ygbench_environment.md)。

# RummageBench

> Household generation (2026-10-15): `behavior_pipeline.py prepare-home` retains a complete BEHAVIOR-1K house and uses InfiniGen's constraint solver for daily-object placement, then loads native assets through OmniGibson's existing skill executor. House architecture and large furniture remain fixed; **0 evaluation-ready household episodes**. See [commands and measured scope](docs/behavior_native_pipeline.md). The [earlier procedural-mesh implementation](docs/infinigen_household_pipeline.md) remains available for historical comparison.


> **Validation status (2026-09-30):** the `build/` reports and `HANDOFF.md`
> contain historical results, not acceptance evidence for the current code.
> CPU regressions and real OmniGibson acceptance are separate gates. Offline
> layout generation produces candidate manifests; the native BEHAVIOR adapter
> loads these into the simulator, but household acceptance remains separate. See
> `EXPERIMENT_PLAN.md` for the remaining acceptance gates and experiment order.

Embodiment-grounded interactive object-search benchmark built on
**BEHAVIOR-1K v3.9.3 + OmniGibson**.

> **NEW (2026-10-03): Level-1 data factory** — local interactive search
> (400 environments x 5 real mobile manipulators = 2000 episodes),
> occupancy-guided physical scene compilation, center-pixel visual picking
> (grounding_version=center_pixel_v2). See [README_level1.md](README_level1.md).

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
pip install -e '.[kinematics]' # use the NumPy-compatible declared constraint
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

## Final Agent Protocol

The evaluated agent (AGENT mode — `InteractiveSearchEnv`, MCP, human-play UI)
receives **only**: the current RGB frame (+ opaque `frame_id`), the task
instruction, a fixed skill library, and four-class feedback for its previous
action.

1. **8 skills**: `MOVE` `TURN` `OPEN` `CLOSE` `GRASP` `PLACE` `OBSERVE`
   `REPORT_DONE`.
2. **Signed conventions**: `MOVE.distance_cm` > 0 forward / < 0 backward
   (5..100 cm); `TURN.angle_deg` > 0 left CCW / < 0 right CW (5..180 deg).
   No clamping; out-of-range = `INVALID_ACTION`.
3. **Point convention**: object skills address the object by a normalized 2D
   point on the CURRENT frame (`origin top-left`, x right, y down, [0,1]) plus
   the mandatory `frame_id`; stale frames are `INVALID_ACTION`.
4. **Bridge**: point -> synchronized private frame -> robust instance
   selection (11x11 patch, >=20 px, >=50% dominance) -> owning-entity
   canonicalization -> median in-instance depth -> depth unprojection
   (convention VERIFIED, not assumed) -> private 3D surface point.
5. The click selects the OBJECT; the `ObjectInterface` — never the clicked
   point — selects the physical interaction target.
6. `OUT_OF_CAPABILITY` = visual target exists and is semantically valid but no
   joint-limit-valid IK exists. `UNSAFE` = IK exists but every configuration
   collides, or a swept MOVE/TURN is blocked, or no safe OBSERVE viewpoint
   exists. Everything else invalid maps to `INVALID_ACTION`.
7. `OBSERVE(point)` = object-centric active perception: 8 deterministic
   viewpoints around the object AABB, accepted only when base-collision-free
   and the object is visibly present (>=100 private pixels); the robot pose is
   exactly restored; returned views are RGB-only reasoning evidence and are
   NOT valid point-reference frames.
8. Wrong-object grasps are RECOVERABLE (no `WRONG_TARGET`); the agent infers
   mistakes visually.
9. `REPORT_DONE` is the only normal success trigger; a false completion is
   terminal `FAIL_FALSE_COMPLETION`.
10. Only the current RGB frame is a valid point-reference frame.
11. The agent never receives entity names, object states, robot pose, held
    entity, admissible/candidate action lists, segmentation, depth, camera
    extrinsics, or oracle state. Episode IDs are opaque.
12. **SR / ST**: success rate and mean planning steps of successful episodes.
13. **Layout generation**: `scripts/generate_layout.py` builds deterministic
    candidate-layout manifests. Structural validation is separate from
    simulator acceptance. Manifest-to-scene construction and real dataset
    model verification remain required before layout generalization studies.
14. **Render profiles**: deferred until simulator layout acceptance. No
    completed render-profile implementation is claimed by this repository.
15. **Reproduce**:

    ```bash
    python -m pytest tests/unit -q                    # no simulator
    python -m pytest tests/integration -m sim -q      # real simulator
    python scripts/generate_layout.py \
        --config configs/layouts/kitchen_search_v1.yaml \
        --catalog configs/layouts/asset_pool_v1.yaml --seed 0
    ```

## Play (human interface)

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/serve_play.py [--scenario scenarios/knife_search_001/scenario.yaml] [--port 8090]
```

Boots the real simulator, then serves a browser UI on `http://0.0.0.0:8090`.
The public interface uses the fixed eight-skill library and current RGB image.
Object-directed actions select a normalized image point with the current
opaque frame ID. RESET restarts the episode. Entity names and evaluator-private
grounded candidate lists are not part of the AGENT interface.

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

> **Results below are INVALIDATED by the collision fix.** They were produced
> while the collision export was broken (world colliders lost their scale;
> the exported R1Pro URDF had no collision bodies).

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
