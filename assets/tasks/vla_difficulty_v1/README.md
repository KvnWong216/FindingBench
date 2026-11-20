# VLA difficulty v1 — 任务分层设计

用于评测 VLA 模型在 **语言抽象程度（L）× 空间可发现性（S）** 两个维度上的表现。
每个变体 = 基础任务 × L × S。同一基础任务的所有变体共享场景、背景物体、起点、技能库、步数预算与成功条件，**只改指令文字和目标初始位置**。
定义见 [levels.yaml](levels.yaml)，公共配置见 [common.yaml](common.yaml)，变体索引见 [variants.yaml](variants.yaml)。
运行状态见 `runs/vla_difficulty_v1/summary.md`，过程与结论见 `docs/vla_difficulty_v1_report.md`。

## 目录与工具

```
vla_difficulty_v1/
  README.md          本文件
  levels.yaml        L1–L3、S1–S3 定义与仿真门禁
  common.yaml        场景 / 机器人 / 全景相机 / 起点 / 预算（所有变体共享）
  base_tasks/        每个基础任务一份：目标、成功条件、背景物体、各 S 的目标位置、各 L 的指令
    T01_knife_fetch.yaml       holding(刀)
    T02_mug_to_counter.yaml    on_top(杯子, 走廊门旁台面 countertop_tpuwys_0)
    T03_fork_to_drawer.yaml    inside(叉子, 走廊门旁柜子 qohxjq_0 的顶层抽屉)；必须先开抽屉再拿叉子
    T04_apple_to_counter.yaml  on_top(苹果, countertop_tpuwys_0)
    _superseded/               废弃稿（餐桌、冰箱、客厅、岛台目标）及原因
  variants.yaml      36 个变体 ID（9 格 × T01–T04）；active_tasks = T01–T04
```

| 步骤 | 命令（`conda activate ygbench`，工作目录为仓库根） |
|---|---|
| 编译变体 → `build/vla_difficulty_v1/<id>/scenario.yaml` | `python scripts/compile_vla_variants.py` |
| 单个变体仿真（初始化 + 空报告基线 + 脚本化 oracle） | `python scripts/run_vla_variant.py build/vla_difficulty_v1/<id>/scenario.yaml` |
| 接入模型策略 | 同上加 `--agent my_pkg.my_mod:make_policy`（`agents/policy_adapter.py`） |
| 批量 + 汇总 | `python scripts/run_vla_batch.py --variants ...` / `--summarize-only` |
| 候选位置认证（一次加载测多个位置） | `python scripts/certify_vla_locations.py <scenario> --target-locations rec:rel[:link] ... --goal-receptacles ...` |
| 场景探针 / 可达性图 | `scripts/probe_vla_scene.py`、`scripts/plot_vla_reachability.py` |
| 某个技能为什么不可行（打印碰撞对，带 / 不带手持物） | `python scripts/diagnose_vla_feasibility.py <scenario> --skill OPEN --entity <家具>` |

## 场景与机器人的硬约束（仿真实测，2026-10-18）

1. **R1Pro 底盘 0.80 m（+0.03 m 余量）过不了 Beechwood_0_int 的任何门（门宽约 0.75 m）**。机器人只能在 `kitchen_0` 与开放式相连的 `living_room_0` 中活动（可达面积 9.3 m²，见 `docs/evidence/vla_difficulty_v1/beechwood_reachability.png`）。因此 S3 = 客厅（与厨房开放相连、起点看不到的位置），不是隔着门的房间。
2. **PLACE**：表面放置只在两块厨房台面可行（餐桌、茶几、客厅柜都碰撞）；2026-10-18 扩展了固定家具**抽屉内部**放置（T03）。冰箱前没有无碰撞底盘位姿，冰箱目标仍不可用。
3. **很多容器抓不到**：微波炉、洗碗机、吊柜、冰箱、`bottom_cabinet_rvpunw_0` 的两个落地抽屉（z<0.31）均 GRASP 不可行（碰撞或够不到）。
4. **全景分辨率**：VIEW 是对缓存全景的重采样，不重新渲染；1024×512 时 2.4 m 外的餐刀只有 6 px → 改为 2048×1024。
5. **起点**：knife_search_001 的 `kitchen` 锚点在 AGENT 协议的底盘碰撞检查下与椅子重叠 → 移到 (0.85, −7.70)。
6. **手里拿着东西不能开抽屉**（2026-10-19）：模型化的是右手，持物会和抽屉面板碰撞 → 放进抽屉的任务要先开抽屉再拿物体（T03 的 `plan_constraint`）。
7. **抓取手 = 可行性模型的手**（2026-10-19 修复）：之前抓取时物体被挂到较近的手、偏移量按左手记录，而 IK / 碰撞检查用右手，导致“能否放下取决于怎么抓的”。修复前的 T03/T04 结果存于 `runs/vla_difficulty_v1/_history/*_before_arm_fix`。
8. **台面下的抽屉要斜着放**（2026-10-19）：`qohxjq_0` 顶层抽屉被 `countertop_tpuwys_0` 盖住，竖直向下放会撞台面 → 放进打开的抽屉时工具从拉出一侧斜向进去（45°/30°/60°/0°）。
9. **长物体挂在指尖前方**（2026-10-19）：叉子挂在 EEF 原点会和夹爪 / 相机连杆重叠，把底盘顶起 → 最大半尺寸 > 8 cm 的物体挂在指尖前方。

## 空间条件（目标初始位置，已认证）

| S | 位置 | 认证结果 | 相对 S1 的额外操作 |
|---|---|---|---|
| S1 直接可见 | 厨房岛台 `countertop_tpuwys_4` | 起点全景 37 px（刀）/ 517 px（杯）；T01 脚本化 8 步成功 | — |
| S2 遮挡 / 容器内 | 走廊门旁柜子顶层抽屉 `bottom_cabinet_no_top_qohxjq_0/link_2`（= knife_search_001 原目标抽屉，上方有台面 `countertop_tpuwys_0`） | 起点 0 px；交互位姿处关闭时 0 px 且 GRASP 不可行；打开后 316–496 px；T01 脚本化 7 步成功 | OPEN |
| S3 跨房间 | 客厅电视下的矮书架 `bookcase_owvfik_1` 顶面 | 起点 0 px，`living_room_0`；GRASP 可行，路线 7 步 | 跨房间导航（开放式） |

## 难度网格

| | S1 | S2 | S3 |
|---|---|---|---|
| **L1** | T01–T04 | T01–T04 | T01–T04 |
| **L2** | T01–T04 | T01–T04 | T01–T04 |
| **L3** | T01–T04 | T01–T04 | T01–T04 |

每格 4 个任务。T03 的 S2 例外：目标抽屉就是共用 S2 抽屉，所以叉子放在相邻柜子 `bottom_cabinet_no_top_pluwfl_0`（同样隐藏、需 OPEN）。

L2、L3 指令不随 S 变化；L1 写出当前 S 的位置和步骤，因此 **L1 下空间维度只衡量“已知位置时的执行/导航”，L2/L3 下才包含“寻找”**（levels.yaml 已注明）。

## L3 歧义控制

每个任务的 `l3_ambiguity_check` 列出竞争解释及处理（移除砧板、削皮刀、茶杯等）。移除对该任务的所有变体同样生效。

## 三个层级不要混淆

| 层级 | 含义 | 在哪里 |
|---|---|---|
| Task Initialization | 场景加载、目标摆放成立、目标初始未满足、S 门禁 | `report.json: task_initialization` |
| Scripted Execution | 特权 oracle（`evaluation/route_oracle.py`）只通过公共动作完成任务 → 任务可解、预算够 | `report.json: scripted_execution` |
| VLA Task Success | 模型策略的成功率 | `--agent` 后的 `policy_episode`；**尚未有模型** |
