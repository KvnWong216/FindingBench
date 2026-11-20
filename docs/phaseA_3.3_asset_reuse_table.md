# Phase A · 3.3 项目资产总表（2026-10-18 实查）

“当前状态”只记录本机可查证据；“历史”指迁移前在旧主机上的结果，本机未复跑。详细依据见 [3.1](phaseA_3.1_findingbench_task_inventory.md) 与 [3.2](phaseA_3.2_behavior_engineering.md)。

| 资产或模块 | 来源 | 作用 | 当前状态 | 可复用方式 |
|---|---|---|---|---|
| knife_search_001 + batch_kitchen_v1（8 条） | FindingBench `scenarios/` | 厨房容器搜索 + 抓取 | 定义完整，可解析；历史 GPU 运行，本机未跑；无 panorama | 适配（补 panorama、改起点/预算）；新分层设计的场景/物体来源 |
| easy_v1（220 条） | FindingBench `assets/benchmark/easy_v1/` | 单房间搜索发布集 | 历史认证；证书缺失；与新分层定义有出入 | 暂不使用（用户确认）；保留原样 |
| household 批次（205 户） | FindingBench `build/behavior_production_*` | 整屋厨房橱柜搜索 | 84 户物理通过（历史），0 可评测；路径指向 `/data1` | 暂不使用 |
| behavior_groceries_pilot | 新 BEHAVIOR 工程 | 官方长程任务 | 本机初始化 + 官方判定通过；抓放未跑 | 直接加载；做长程对照 |
| 官方 BDDL 目录 | BEHAVIOR-1K + `datasets/behavior.py` | 任务定义 | 1016 条；刚体 191；本机有实例 56 | 直接复用定义；变体需适配 |
| 场景 Beechwood_0_int 等 51 个 | `behavior-datasets/behavior-1k-assets/scenes` | 仿真场景 | 已安装；Beechwood_0_int 历史 GPU 加载 | 直接加载；家具位置离线可读 |
| 官方实例场景 7 个（house_single_floor 等） | `2026-challenge-task-instances` | 官方任务实例 | 199 个模板；house_double_floor_lower 已实测 | 直接加载（仅官方路径） |
| 物体模型（刀/碗/杯/勺/叉/苹果/马克杯/盘/砧板/削皮刀） | BEHAVIOR 资产，经 knife 系列使用 | 交互对象 | 历史 GPU 实例化成功 | 直接复用 |
| R1Pro URDF + 碰撞网格 | `build/robots/`、`assets/robots/r1pro/` | IK / 碰撞可行性 | 存在，含 FK 交叉验证文件 | 直接复用 |
| 技能 MOVE/TURN/VIEW/OPEN/CLOSE/GRASP/PLACE/REPORT_DONE | 新 BEHAVIOR 工程 `skills/`、`visual_session.py` | 执行动作 | 实现；groceries 实测 VIEW/MOVE/TURN；OPEN/GRASP 历史实测；PLACE 仅单测 | 直接调用 |
| 环境构建 / Task Loader / Scene Loader | `env_factory.py`、`scenario.py`、`backend.py` | 初始化 | 实现；官方路径会覆盖 instruction | 直接调用；官方路径需小适配 |
| 观测接口（全景 + VIEW、点击 grounding） | `visual_session.py`、`perception/` | 观测 | 本机实测 | 直接复用 |
| 评测逻辑 goal_satisfied / goal_checker | `report_done.py`、`evaluation/goal_checker.py` | 成功判断 | 实现；放置谓词仅单测 | 直接复用 / 扩展 |
| 可见性门禁 | `visibility.py`、任务工厂 gates | S 条件判定 | easy_v1 历史使用 | 适配为 S1/S2/S3 检查 |
| Rollout / 指标 | `cli evaluate-visual`、`episode_log.py`、`metrics.py` | 执行与统计 | 仅脚本化 agent、单回合 | 扩展：模型适配器 + 变体批量 runner |
| 人工试玩 | `serve_play.py`、`play_behavior.sh` | 人工验证 | 本机实测 | 直接复用 |
| 日志 | `events.jsonl`、`summary.json` | 轨迹记录 | 实现 | 直接复用 |
| 新分层任务设计 | `assets/tasks/vla_difficulty_v1/`（本次新增） | L×S 任务变体 | 设计 + schema 校验，**未仿真** | 下一阶段编译为 scenario 并冒烟 |
