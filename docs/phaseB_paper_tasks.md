# Phase B · 论文 / 官方 Benchmark 任务调查（2026-10-18）

## 0. 目标论文：BEHAVIOR-1K（用户 2026-10-18 确认）

**Li et al., *BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation*.** 会议版 CoRL 2022（PMLR 205, `proceedings.mlr.press/v205/li23a`），扩展版 arXiv:2403.09227（2024）；项目页 behavior.stanford.edu；官方代码 StanfordVL/BEHAVIOR-1K（本机固定提交 `BEHAVIOR-1K-6cbf70b…`，OmniGibson 3.9.3 + bddl3）。

论文中与本研究相关的位置（arXiv HTML 版核对；章节号来自该版本）：

| 内容 | 论文位置 | 本机对应 |
|---|---|---|
| 1,000 个活动来自“希望机器人帮你做什么”的问卷（909 条问卷排序 + 91 条 BEHAVIOR-100） | Sec. 2，Fig. 1 | — |
| 活动用 BDDL 定义：初始条件 + 目标条件（谓词逻辑） | Sec. 4；App. B（定义标注）；App. C（BDDL 扩展）；Listings 1–5 | `assets/behavior_official/bddl/activity_definitions/*/problem*.bddl`（1016 个文件） |
| OmniGibson 由初始条件采样场景、用目标条件判定成功；谓词兼作检查与采样函数 | Sec. 5；App. E.1.4–E.1.5 | 官方 BehaviorTask；FindingBench `backend.behavior_goal_satisfied()` |
| 成功率（稀疏成功奖励）、效率指标（导航距离、仿真时间、运动学扰动）、成功分 Q | Sec. 6.1，Table 2–4；App. F.5 Table A.13 | 我们用 REPORT_DONE + goal 判定成功；步数代替效率 |
| 实验任务：CollectTrash、StoreDecoration、CleanTable（按技能分：刚体 / 铰接 / 柔性与流体） | Sec. 6.1，Table 2、Table 4 | 论文用驼峰别名，本机 BDDL 无同名文件。可能对应 `picking_up_trash`、`putting_away_Halloween_decorations`、`clean_a_kitchen_table` 等，**对应关系未核实** |
| Sim-to-real：CollectTrash 在真实样板间（仿真约 40%，真实最优策略约 22%，训练策略 0%） | Sec. 6.2，Fig. 5 | — |
| 活动分类：问卷偏好分、约 200 个清洁类和 200 多个烹饪类、按所需物体 synset 分组 | Sec. 2；Fig. 4；App. A.1（可行性过滤） | `scripts/analyze_official_tasks.py` 另按 L/S 轴分类 |

另外，BEHAVIOR Challenge 2025 从 1,000 个定义中选了 50 个任务、R1Pro 机器人、按 BDDL 目标判定（behavior.stanford.edu/challenge）；本机的 `2026-challenge-task-instances`（7 个场景，199 个模板）属于同一系列官方实例。

**结论**：论文定义的是 **1,000 个活动定义（BDDL）+ 仿真器 + 评测方式**，不是我们需要的“语言 / 空间分级”。论文本身只在 3 个任务上报告实验。我们的做法是：复用它的目标形式（ontop / inside / open）、物体与场景资产和判定机制，在其上构造 L/S 变体。

## 1. 官方任务的总体构成（本机实查）

脚本：`scripts/analyze_official_tasks.py`（纯 CPU，读 BDDL）；输出 `build/behavior/official_task_axes.json`。

| 统计项 | 数量 | 说明 |
|---|---|---|
| 官方 BDDL 定义 | 1016 | `activity_definitions/*/problem*.bddl` |
| 刚体审计通过（`datasets/behavior.py`） | 191 | 不含液体、布料、切割、加热等 |
| 其中本机有官方预采样实例 | 56 | 7 个场景（house_single_floor、house_double_floor_lower/upper、Rs_int、restaurant_diner、hotel_suite_large、office_cubicles_right） |
| 目标只涉及 **一个** 被移动物体（单物体取放） | 6 | 全部无本机实例（store_honey 有，但需 inside 柜子） |
| 多物体 / 量词目标（forall / exists） | 180 | 官方任务的主体 |
| 只有状态目标（open） | 5 | opening_doors 等 |
| 目标物体初始在**关闭的可开容器内**（≈ 我们的 S2） | 32 | 例：freeze_fruit、preparing_lunch_box、bringing_water |
| `inroom` 涉及多个房间（≈ 我们的 S3） | 43 | 例：picking_up_trash、bringing_water、carrying_in_groceries |
| 目标要求 `inside(物体, 可开的场景家具)`（冰箱 / 柜子） | 39 | **当前 PLACE 无法满足**，见第 3 节 |

结论：**官方 BEHAVIOR 没有“单物体 × 语言分级 × 空间分级”的任务**。官方任务是长程、多物体、量词目标，指令只有一种（BDDL 自动生成的自然语言目标）。它们定义了**目标形式**（ontop / inside / not open）和**场景、物体**，但没有我们的 L / S 维度。

## 2. 论文任务表（与本研究最相关的条目）

| Paper Task（BDDL activity） | 任务目标（BDDL :goal） | 所需物体 | 所需技能 | 成功条件 | 原始实现位置 | 本机实例 | 类似我们的 |
|---|---|---|---|---|---|---|---|
| carrying_in_groceries | 番茄、牛奶放进同一冰箱；关车门、冰箱门 | 纸袋、番茄、牛奶、汽车、冰箱 | MOVE/TURN、OPEN、GRASP、PLACE(inside)、CLOSE | 官方 `check_goal` | `bddl/activity_definitions/carrying_in_groceries/problem0.bddl` | house_double_floor_lower#0（**本机已实测初始化+判定**） | S3 + 放入家具 |
| clearing_food_from_table_into_fridge | 食物装进保鲜盒，盒子全部进冰箱，冰箱关闭 | 半只鸡、半个苹果派、盘子、保鲜盒、冰箱 | 同上 + 多物体 | 官方 | `…/clearing_food_from_table_into_fridge/problem0.bddl` | house_double_floor_lower#0 | T03 原设计的官方类比 |
| freeze_fruit | 苹果、草莓进保鲜盒，盒子进冰箱，冰箱关闭 | 苹果、草莓、保鲜盒（初始在关着的柜子里）、冰箱 | 同上 | 官方 | `…/freeze_fruit/problem0.bddl` | house_single_floor#0 | S2（容器内）+ 放入冰箱 |
| preparing_lunch_box | 食物放进饭盒，饭盒关闭 | 茶瓶（初始在冰箱里）、饼干、三明治、半个苹果、饭盒 | OPEN、GRASP、PLACE(inside) | 官方 | `…/preparing_lunch_box/problem0.bddl` | house_single_floor#0 | S2 |
| bringing_water | 两瓶水从冰箱拿到客厅桌上，冰箱关闭 | 瓶子（初始在冰箱里）、冰箱、桌子 | OPEN、GRASP、跨房间 MOVE、PLACE(ontop)、CLOSE | 官方 | `…/bringing_water/problem0.bddl` | house_single_floor#0 | **S2 + S3 + ontop**，最接近 T02 |
| picking_up_trash | 汽水罐扔进垃圾桶（跨厨房/客厅） | 汽水罐、垃圾桶 | 跨房间、GRASP、PLACE(inside 非家具) | 官方 | `…/picking_up_trash/problem0.bddl` | house_double_floor_lower#0 | S3 |
| set_up_a_coffee_station_in_your_kitchen | 咖啡机、水壶、杯子等摆在台面上 / 彼此相邻 | 6 个物体 | GRASP、PLACE(ontop)、nextto | 官方（nextto 不在我们的 goal 谓词里） | `…/set_up_a_coffee_station_in_your_kitchen/problem0.bddl` | house_single_floor#0 | T02 目标形式 |
| moving_boxes_to_storage | 两个箱子从客厅搬到车库 | 储物箱 | 跨房间、GRASP、PLACE(ontop) | 官方 | `…/moving_boxes_to_storage/problem0.bddl` | house_double_floor_lower#0 | S3 + ontop |
| store_honey | 蜂蜜罐放进柜子 | 蜂蜜罐、柜子 | 跨房间、OPEN、PLACE(inside 家具) | 官方 | `…/store_honey/problem0.bddl` | Rs_int#0 | 单物体，但需 inside 家具 |

（完整 191 条见 `build/behavior/official_task_axes.json`，字段：goal_kinds、goal_movables、hidden_at_init、rooms、open_required、goal_needs_inside_furniture、local_instances。）

“论文只给类别、没有实例”的情况：上表全部有完整 BDDL 定义；“本机实例”一栏为空的任务只有定义，没有预采样场景，无法直接运行。

## 3. 复用分类（A 直接 / B 少量修改 / C 需适配 / D 暂不适合）

| 类别 | 任务 | 理由 |
|---|---|---|
| **A 直接运行** | carrying_in_groceries | 已接入 FindingBench（`scenarios/behavior_groceries_pilot`），本机实测初始化 + 官方判定。但只能运行官方的一种指令和一种初始布局，**不能直接产生 L/S 变体**；完整抓放未跑。 |
| **B 少量修改** | bringing_water、moving_boxes_to_storage、set_up_a_coffee_station（只保留 ontop 部分） | 目标是 ontop，当前 PLACE 支持；需要 ① 官方任务路径支持 instruction 覆盖（`backend.setup` 目前用 BDDL 文本覆盖 YAML 指令），② 或把目标改写为 FindingBench 的数据驱动 goal。 |
| **C 需要适配** | freeze_fruit、preparing_lunch_box、clearing_food_from_table_into_fridge、store_honey、carrying_in_groceries 的完整执行 | 目标含 `inside(·, 冰箱/柜子)`。**本次核实：`skills/place.py` → `backend.symbolic_place` 只把物体放在接收物的顶面；`backend.receptacle_region` 对 fixed-base 家具（冰箱、柜子）永远返回 `top_surface`。** 因此“放进冰箱/柜子”目前在 FindingBench 技能层不可执行，需要扩展 PLACE 的 inside 放置（内部隔板采样 + inside_volume 区域）。 |
| **D 暂不适合** | 多物体量词任务的大多数（180 条中的多数）、nextto/under/touching 目标、无本机实例的任务 | 步数长、需要多次抓放；谓词不在 goal_checker 支持范围；或无官方预采样实例。 |

## 4. 对本研究的意义

1. 官方任务提供**目标形式和动机**（ontop 取放、放入冰箱、容器内取物、跨房间搬运），我们的 T01/T02 采用相同目标形式，但在 FindingBench 已验证的 Beechwood_0_int 场景中**单物体化**，以便控制 L/S 两个维度。这是“适配”而不是重新发明：每个基础任务在 YAML 的 `source.official_behavior_analogue` 中记录了对应官方任务。
2. 若要在官方场景上直接做 L/S 变体，最小改动是给官方任务路径加“instruction 覆盖 + 目标物体初始位置覆盖”。这会改动官方初始化语义，留给用户决定。
3. “放进冰箱 / 柜子”这一类（39 条官方任务）当前不可执行，是后续最值得补的技能缺口。

## 5. 用户决定（2026-10-18）

- 目标论文 = BEHAVIOR-1K（见第 0 节）。
- 允许扩展 PLACE 以支持 `inside(物体, 可开家具)`；实现与验证见 `docs/vla_difficulty_v1_report.md`。

参考链接：arxiv.org/abs/2403.09227；proceedings.mlr.press/v205/li23a.html；behavior.stanford.edu/challenge/call_for_participation.html
