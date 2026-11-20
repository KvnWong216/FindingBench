# Phase A · 3.2 FindingBench 之外 / 新 BEHAVIOR 工程调查（2026-10-18 实查）

## 0. “新的 BEHAVIOR 工程”具体是什么

| 位置 | 是什么 | 与本研究的关系 |
|---|---|---|
| GitHub `KvnWong216/FindingBench` main 最新提交 `cce72ad`（2026-10-17）“Integrate official BEHAVIOR pilot, panoramic VIEW skills and ygbench human runner”；本地工作副本 `/home/ubuntu/ygwang/FindingBench-publish`（`src/`、`scenarios/`、`assets/` 与 `FindingBench` 逐文件一致） | **就是新的 BEHAVIOR 工程**：官方 BehaviorTask 加载、全景 VIEW、ygbench 人工试玩、官方 REPORT_DONE 判定 | 主要运行基础 |
| `/home/ubuntu/ygwang/BEHAVIOR-1K-6cbf70b…`（官方源码，固定提交） | OmniGibson 3.9.3 + bddl3（editable 安装到 `ygbench`） | 底层仿真、BehaviorTask、BDDL；不修改 |
| `/home/ubuntu/ygwang/behavior-datasets` | `behavior-1k-assets`（51 个场景、1829 个物体类别目录）、`2026-challenge-task-instances`（7 个场景、199 个任务模板）、`omnigibson-robot-assets` | 场景 / 物体 / 官方实例资产 |
| `/home/ubuntu/ygwang/Interactive-Perception`（GitHub `KvnWong216/Interactive-Perception`，分支 `feat/geometry-history-action-jepa`） | MolmoAct2-Think 空间微调 + LIBERO 主动信息获取诊断 | **不含 BEHAVIOR/OmniGibson 代码**；其 README 把 FindingBench 列为“后续扩展，不作当前主环境”。将来接 VLA 模型时可参考它的 MolmoAct2 推理加载（`load_finetuned_vlm`） |

本次 10-17 提交修改的 BEHAVIOR 相关文件（按修改时间核对）：`sim/omnigibson/{env_factory,backend,panorama,usd_collision}.py`、`datasets/behavior.py`、`core/{scenario,session,visual_session,public_types,events}.py`、`perception/{panorama,visual_bridge}.py`、`skills/report_done.py`、`adapters/python_api.py`、`scripts/{prepare_behavior_pilot,behavior_catalog,serve_play}.py`、`scripts/play_behavior.sh`、`scripts/environment/{verify_ygbench,smoke_behavior_pilot}.py`、`configs/protocol/*_v2.schema.json`、`configs/benchmarks/behavior_pilot.yaml`。

## 1. 环境（第二节验证项）

| 项 | 结果 |
|---|---|
| `ygbench` 是否存在 | 是，`/home/ubuntu/miniforge3/envs/ygbench` |
| Python | 3.11.16 |
| omnigibson / bddl | 可导入，指向 `BEHAVIOR-1K-6cbf…/OmniGibson`、`…/bddl3`（editable） |
| pinocchio | 可导入（cmeel）。`hpp.fcl` 名称导入失败，但文档称 HPP-FCL 2.4.4 以 `coal`/cmeel 形式提供，单测 1000 passed 包含碰撞测试，**本次未复跑单测** |
| 环境变量 | 激活时自动设 `OMNIGIBSON_DATA_PATH=/home/ubuntu/ygwang/behavior-datasets`、`OMNI_KIT_ACCEPT_EULA=YES`（`docs/ygbench_environment.md`） |
| 已知问题 | 上游依赖 4 项版本钉死冲突（`runs/ygbench_setup/pip-check.txt`），运行验证已通过；`git` 命令未安装 |
| 可直接运行的入口 | `./scripts/play_behavior.sh`（人工网页，8090）、`scripts/environment/smoke_behavior_pilot.py`（GPU 冒烟）、`InteractiveSearchEnv`（Python）、`python -m rummagebench.cli evaluate-visual / run / build / inspect / serve-mcp` |

## 2. 模块逐项

| 模块 | 实际实现 | 状态 | 复用方式 |
|---|---|---|---|
| 环境构建与初始化 | `sim/omnigibson/env_factory.py` `build_env_config/build_env`；`OmniGibsonBackend.setup()` | 已实现，四种来源：① 普通 BEHAVIOR 场景 + YAML `objects/placements`；② `behavior_layout`（批量整屋）；③ `infinigen`；④ `behavior_task` 官方预采样实例 | **直接调用** |
| Task Loader | `core/scenario.load_scenario` → `ScenarioSpec`（pydantic，含 `goal/temporal/panorama/behavior_task`）；官方实例由 `scripts/prepare_behavior_pilot.py` 绑定模板 | 已实现 | 直接调用 |
| Scene Loader | OmniGibson `InteractiveTraversableScene`（`scene.model`）；官方路径由 BehaviorTask 加载缓存场景 | 已实现 | 直接调用 |
| Object Registry | 运行时：`backend.entity_names / describe_entity / resolve_entity / entity_aabb / visible_entities`；离线：`sim/omnigibson/scene_inspector.inspect_scene`（需启动仿真）、`cli inspect`；场景 JSON `scenes/<scene>/json/<scene>_best.json` 的 `objects_info.init_info`（含 `in_rooms`）可离线读取 | 已实现 | 直接调用；本次设计用离线 JSON 读取家具位置/房间 |
| BDDL | `datasets/behavior.py`：1016 条定义解析 + 刚体审计（191 条刚体，其中 56 条有本机官方实例）；运行时 `backend.behavior_goal_satisfied()` 调用 `compiled_task.check_goal` | 已实现；`skill_coverage: unverified` | 直接复用；逐条可执行性未验证 |
| Skills / Action Primitives | `skills/{move,open,close,grasp,place,report_done}.py`（MOVE/TURN 在 move.py），VIEW 在 `core/visual_session.py`；**符号执行**：IK + 碰撞可行性判定通过后瞬时改变状态（`set_open`、`symbolic_grasp`、`symbolic_place`） | 已实现；OBSERVE 已废弃 | 直接调用；不重写 |
| 上游 `omnigibson/action_primitives`（starter / symbolic semantic / curobo） | 存在于 BEHAVIOR-1K | **FindingBench 未使用** | 不需要；与 FindingBench skills 功能重叠 |
| Agent 控制接口 | `environment/env.InteractiveSearchEnv`（reset/step JSON）、`adapters/python_api.create_session`、`adapters/mcp_server`、`scripts/serve_play.py`（HTTP）；schema `configs/protocol/*_v2.schema.json` | 已实现 | 直接调用 |
| Task Success Checker | `skills/report_done.goal_satisfied`：官方 BDDL → 数据驱动 `goal`（holding/on_top/inside/open/closed）+ `temporal` → 默认 holding(target)；`evaluation/goal_checker.py` 与 PLACE 后验共用同一放置判定 | 已实现；数据驱动 goal 有单测，**未在仿真里跑过放置目标** | 直接复用 |
| Rollout / Evaluation | `cli evaluate-visual` → `episode_log.run_visual_episode` + `metrics.compute_metrics`（AGENT 协议，但 agent 只有 `ScriptedPixelAgent`，单回合）；`cli run`（ORACLE 旧路径） | 部分实现：**无模型 agent、无变体批量评测** | 需新增轻量 runner / 模型适配器 |
| Demo / 人工 | `play_behavior.sh` + `serve_play.py`（支持 VIEW）、`smoke_behavior_pilot.py` | 已实现，groceries 已实测 | 直接复用做人工验收 |
| 日志 / 轨迹 | `events.jsonl`（版本化，含私有 grounding 诊断）、per-step PNG、`summary.json` | 已实现 | 直接复用 |
| 可见性 / 难度门禁 | `sim/omnigibson/visibility.py`、`backend.entity_visible_pixels`；任务工厂 gates（`start_visibility_max`、`revealed_visibility_min`、`open_required`）在 `authoring/tasks/` | 已实现（用于 easy_v1） | **适配**后用于 S1/S2/S3 判定 |

## 3. 要求明确列出的五项

1. **可直接调用**：env_factory/backend、load_scenario、全部技能、VisualProtocolSession/InteractiveSearchEnv、report_done/goal_checker、events 日志、serve_play、官方 BehaviorTask 加载与判定。
2. **需简单适配**
   - 官方任务路径（`behavior_task`）在 `backend.setup()` 中**提前返回**：① 用 BDDL 自然语言目标**覆盖** YAML 的 `instruction`，② 忽略 `objects/placements`，③ 机器人用官方预采样位姿。要在官方任务上做 L1/L2/L3 或 S 变体，需加“instruction 覆盖”和“初始位姿/物体位置覆盖”的小适配。
   - 旧场景缺 `robot.panorama` → VIEW 返回 OUT_OF_CAPABILITY；补 YAML 即可。
   - `ScenarioSpec._cross_check` 要求 `goal` 的 `receptacle` 必须是 YAML `objects` 中生成的物体，**场景原生家具（餐桌、冰箱）作为放置目标会在加载时被拒绝**；而 `placements` 与运行时 `goal_checker`（`backend.receptacle_region`）都支持场景家具。放置类任务需放宽这一条校验（2026-10-18 用新设计的 T02/T03 变体离线校验时发现）。
   - 可见性门禁需从任务工厂抽成 S 条件检查。
3. **尚未实现**：VLA/VLM 模型适配器（全仓无模型调用）；按变体集合批量运行与 SR/步数汇总；官方任务的完整抓放执行轨迹；放置类 `goal` 的仿真验证。
4. **可能重复的功能**：上游 action_primitives vs FindingBench skills；`core/session.py`（ORACLE）vs `core/visual_session.py`（AGENT）；`environment/env.py` vs `adapters/python_api.py`；`scripts/run_episode.py` vs `cli run`；多套布局生成（`authoring/layout`、`authoring/level1`、`infinigen_adapter`、`scripts/tasks/behavior_pipeline.py`）；URDF 两份（`build/robots/`、`assets/robots/r1pro/`）；OBSERVE（废弃）vs VIEW。
5. **支持 FindingBench 任务迁移的功能**：普通 BEHAVIOR 场景路径完整支持 FindingBench 的 `objects/placements/anchors/initial_states/goal`；Beechwood_0_int 等 FindingBench 场景在本机资产中齐全；只需补 panorama 并在本机重新冒烟。

## 4. 物体与场景资产

| 问题 | 结论（来源） |
|---|---|
| 现有场景可交互物体 | 以 Beechwood_0_int 为例共 188 个场景物体（离线 JSON）。厨房：柜体 `bottom_cabinet_no_top_{pluwfl,qohxjq×2,qudfwe,spojpj×3,ufhpbn}`、`bottom_cabinet_rvpunw_0`、`top_cabinet_dmwxyl_0..3`、`fridge_dszchb_0`、`dishwasher_znfvmj_0`、`microwave_bfbeeb_0`、`oven_wuinhm_0`、台面 `countertop_tpuwys_{0,2,3,4}`/`countertop_jveutp_{0,1,2}`、`breakfast_table_uhrsex_0`；客厅、餐厅、办公室、杂物间各有桌/柜 |
| 哪些物体可抓 | 由可行性引擎动态判定（刚体 + IK + 碰撞）；FindingBench 已在 GPU 上用过的可抓模型：`table_knife/eebpap`、`bowl/adciys`、`teacup/cpozxi`、`tablespoon/huudhe`、`tablefork/flexrc`、`vegetable_peeler/njgqao`、`apple/agveuv`、`cutting_board/aibvew`、`mug/dhnxww`、`plate/aewthq`；easy_v1 另有 39 类 |
| 哪些容器可开 | `backend.articulation_info` 判定；knife 系列 GPU 验证过 `bottom_cabinet_no_top_qohxjq_0/spojpj_0`、`bottom_cabinet_rvpunw_0`；冰箱、洗碗机未在 FindingBench 中验证 |
| 哪些任务涉及导航 | knife（客厅→厨房）、groceries（车库→厨房）；easy_v1 为单房间 |
| 物体初始位置可改 | 普通场景路径：可（`placements` inside/on_top + link）；官方任务路径：不可（需适配） |
| Agent 初始位置/视角 | 普通路径：`anchors` + `robot.init_anchor`（位置 + 四元数朝向）；官方路径：预采样位姿；视角：VIEW(yaw,pitch) 选择全景方向 |
| 自动检查物体到达目标区域 | 可：`goal` 的 `on_top/inside`（`receptacle_region` 支撑面/内部体积），官方任务用 BDDL |
