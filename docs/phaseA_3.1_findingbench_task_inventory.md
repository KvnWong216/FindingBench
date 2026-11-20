# Phase A · 3.1 FindingBench 任务清单（2026-10-18 实查）

调查范围：`/home/ubuntu/ygwang/FindingBench`（符号链接 → `FindingBench_20261016_200444/workspace`，Python 包名 `rummagebench`）。
`/home/ubuntu/ygwang/FindingBench-publish` 与本目录的 `scenarios/`、`assets/tasks/`、`assets/benchmark/`、`src/` 逐文件比较无差异（仅缺 egg-info），不单列。

## 0. 调查方法与证据等级

- 读取全部 `scenarios/**`、`assets/benchmark/easy_v1/**`、`assets/tasks/**`、`benchmark_splits/**`、`configs/{benchmarks,layouts,task_priors,level1}/**`、`build/behavior_production_*`、`runs/ygbench_setup/**`，以及成功判定代码 `core/session.py`、`core/visual_session.py`、`skills/report_done.py`、`evaluation/goal_checker.py`。
- 在 `ygbench`（Python 3.11.16；omnigibson / bddl / pinocchio 均可导入）中用 `rummagebench.core.scenario.load_scenario` 解析全部 232 个场景 YAML：**232 成功 / 0 失败**。这只证明 schema 合法，**不代表能在仿真中初始化**。
- **本次未启动任何 GPU 仿真。** 下表“运行验证”一列只引用仓库中已有的证据文件，并注明其日期和主机。

状态五级（对应文档要求）：
`D` 仅有任务定义 · `I` 已实现环境初始化 · `S` 已实现技能执行 · `J` 已有成功判定 · `V` 已有实际运行验证

## 1. 任务总表

| Task ID | Task Name | Task Goal | Instruction（原文） | Scene | Objects | Required Skills | Success Condition | Source Files | Implementation Status | Reusability |
|---|---|---|---|---|---|---|---|---|---|---|
| `knife_search_001` | 找刀（基础手写任务） | 找到并拿起目标刀 `table_knife`（不包括切水果） | "Help me cut some fruit. My knife may be in the kitchen." | `Beechwood_0_int`；起点 `living_room` 锚点，目标在厨房 | 目标 `table_knife/eebpap` 放在 `bottom_cabinet_no_top_qohxjq_0` 的 link_2 内；干扰物碗/茶杯（cabinet_A）、勺/叉/削皮刀（drawer_A）；苹果、砧板、马克杯放台面，盘子放早餐桌 | YAML：NAV/OPEN/CLOSE/GRASP/PLACE；当前公共协议：MOVE/TURN/OPEN/GRASP/REPORT_DONE | ORACLE：抓到目标即 SUCCESS，抓错即 FAIL_WRONG_TARGET。AGENT：REPORT_DONE 时 `held_object == target_knife`，抓错可恢复 | `scenarios/knife_search_001/{scenario.yaml,README.md}`、`tests/integration/test_knife_search_episode.py`、`configs/task_priors/knife_search.yaml` | **D I S J V(历史)**：9/27–9/30 在旧主机上 4 条 ORACLE 轨迹通过；README 注明这批数字已被碰撞修复判为无效（INVALIDATED）；本机未运行 | 高：容器搜索+抓取的标准模板，场景和物体可直接复用 |
| `knife_search_v0`（s000–s029） | 找刀·自然/反事实放置配对 | 同上；目标位置按 natural（3 个柜/抽屉）或 counterfactual（早餐桌台面）放置，用 `pair_id` 配对 | 指令改写（paraphrase），文件缺失，无法核对原文 | 同上 | 同上，干扰物排列不同 | 同上 | 同上 | `benchmark_splits/knife_search_v0{,_natural,_solvable}.yaml`、`configs/task_priors/knife_search.yaml`、`src/rummagebench/authoring/episode_generator.py` | **D（仅清单）**：split 引用的 `build/generated_scenarios/*.yaml` 和 `build/certificates/*.json` **在本机不存在**，需用生成器重新生成 | 中：其中的“自然 vs 反事实位置配对”机制可直接用于 S1/S2 空间变体 |
| `ketchup_search_001` | 找番茄酱 | 拿起 `tomato_sauce_jar` | "Season the fries for me. My ketchup may be in the kitchen." | Beechwood_0_int | 目标放 qohxjq_0/link_2；干扰物：碗、茶杯、勺、叉 | 同 knife | holding(target) | `scenarios/ketchup_search_001/scenario.yaml`（由 `scenarios/batch_kitchen_v1.yaml` + `scripts/generate_scenarios.py` 生成） | **D J**：沿用 knife 已在 GPU 上跑通的容器；无运行证据 | 高：换目标物体的同构变体 |
| `barbecue_sauce_search_001` | 找烧烤酱 | 拿起 `barbecue_sauce_bottle` | "The grill is almost ready — grab my barbecue sauce. It may be in the kitchen." | 同上 | 目标放 qohxjq_0/link_2；干扰物 5 件 | 同上 | 同上 | 同上 | **D J** | 高 |
| `basil_jar_search_001` | 找罗勒罐 | 拿起 `basil_jar` | "The pasta needs flavor. Find my basil jar — it may be in the kitchen." | 同上 | 目标放 qohxjq_0/link_2 | 同上 | 同上 | 同上 | **D J** | 高 |
| `bowl_search_001` | 找碗 | 拿起 `bowl` | "Serve the salad in my big bowl. It may be in the kitchen." | 同上 | 目标放 qohxjq_0/link_2 | 同上 | 同上 | 同上 | **D J** | 中：碗较大，可视性和抓取与小物体不同 |
| `peeler_search_001` | 找削皮刀 | 拿起 `vegetable_peeler` | "Prep the carrots for me. My peeler may be in the kitchen." | 同上 | 目标放 qohxjq_0/link_2 | 同上 | 同上 | 同上 | **D J** | 高 |
| `basil_jar_search_002` | 找罗勒罐（换位置） | 拿起 `basil_jar` | "We are out of herbs. There is another basil jar somewhere in the kitchen." | 同上 | 目标放 **spojpj_0**/link_2 | 同上 | 同上 | 同上 | **D J**，标记 `gpu_smoke_needed: true`（新容器未经 GPU 验证） | 中：可作为同一任务的不同空间条件 |
| `ketchup_search_002` | 找番茄酱（抽屉） | 拿起 `tomato_sauce_jar` | "One more bottle: find the spare ketchup in the kitchen drawers." | 同上 | 目标放 **rvpunw_0**（drawer_A） | 同上 | 同上 | 同上 | **D J**，`gpu_smoke_needed: true` | 中 |
| `level1_container_rummage_000001` | 桌上容器翻找 | 在桌上的容器中找到并拿起茶杯 | "Rummage the container on the table and find the teacup. Pick it up." | Beechwood_0_int，早餐桌（起点锚点 `table_side`） | 茶杯 + 番茄酱/勺/马克杯/叉/苹果 + 罗勒罐/盘 + 容器（碗） | MOVE/TURN/GRASP（可能还需 PLACE 移开物体） | holding(target) | `scenarios/level1/level1_container_rummage_000001/scenario.yaml`、`README_level1.md`、`configs/level1/*.yaml`、`src/rummagebench/authoring/level1/` | **D**：只有 3 条样例 YAML，且 YAML 中所有物体都只写了 `on_top breakfast_table`，**“放在容器里”这一关系没有写进 YAML**；README 中 2000 回合的数据集在本机无产物 | 低–中：任务形式合适（局部遮挡），但需要 Level-1 编译管线才能重现遮挡 |
| `level1_tabletop_clutter_000002` | 桌面杂乱找刀 | 在杂乱桌面上拿起刀 | "Find the table knife on the table and pick it up." | 同上 | 刀 + 约 10 件桌面物体 | 同上 | 同上 | 同上 | **D** | 中：可作为 S1（可见）或 S2（被遮挡）条件 |
| `level1_deliberate_cover_000003` | 刻意遮挡找勺 | 掀开遮挡物后拿起勺子 | "Something on the table may be covering the tablespoon. Uncover it and pick it up." | 同上 | 勺 + 碗（遮挡物）+ 苹果/砧板等 | GRASP + PLACE（移开碗）+ GRASP | 同上 | 同上 | **D**（遮挡关系同样未写进 YAML） | 中 |
| `easy_v1`（220 回合，`fb_*`） | Easy 单房间物体搜索（已发布） | 在指定房间中找到并拿起目标物 | 模板，如 "Find the {object}. It should be somewhere in this {room}." / "Could you get the {object} for me? It is in the {room}."；另有 16 条 landmark 区域模板 | 7 个场景：Beechwood_0 62、Pomaria_1 43、Wainscott_0 41、Merom_0 38、Ihlen_1 28、Ihlen_0 5、Pomaria_2 3；房间：厨房 119、私人办公室 34、游戏室 38、客厅 12、卧室 9、储藏室 5、浴室 3 | 39 类目标（盘/托盘/笔记本/碗/勺/眼镜/铲…）；每回合 2–4 个候选位置；in_container 140 / on_surface 80 | MOVE/TURN/OPEN/GRASP/REPORT_DONE（ORACLE 序列使用 NAV） | REPORT_DONE 时持有目标；抓错即失败；16 步预算；UNSAFE 即失败 | `assets/benchmark/easy_v1/{README.md,manifest.json,split.json,scenarios/*.yaml}`、`assets/tasks/tiers/easy_v1.yaml`、`scripts/tasks/*`、`src/rummagebench/authoring/tasks/` | **D I S J V(历史认证)**：manifest 为每条记录了 oracle 深度 d*（2/3/4 → 100/118/2）和 agent 协议见证步数（5–14 步，均 ≤16）；证书文件在 `build/`，**本机缺失**；认证用的是旧主机和旧协议（含 OBSERVE、无 VIEW 全景） | **最高**：唯一成规模、带难度元数据、dev/test 已划分的任务集；`mode`（in_container / on_surface）可直接作为 S2 / S1 条件 |
| `household_direct_v1` / `kitchen_cabinet_search`（生产批次） | 整屋厨房橱柜搜索 | 在保留完整 BEHAVIOR 房屋的条件下，在厨房橱柜中找到并拿起餐具 | "Find and pick up the {tablespoon\|tablefork\|teaspoon} in the kitchen cabinets."（勺 71 / 叉 67 / 茶匙 67） | 6 套原生房屋：Beechwood_0、Benevolence_1、Ihlen_1、Merom_1、Pomaria_1、Wainscott_0（各约 34 户），共 205 个 home 目录 | 目标餐具 + InfiniGen 求解器摆放的日常物品（daily_*） | MOVE/TURN/OPEN/GRASP/REPORT_DONE（历史版本含 OBSERVE） | REPORT_DONE 时持有目标；48 步；抓错不终止 | `assets/tasks/{household_families_v1.yaml,tiers/household_direct_v1.yaml}`、`docs/household_search_formalization.md`、`build/behavior_production_20261016_v2/batch/home_*/`、`build/behavior_production_environment_v1/` | **D I(部分)**：GPU 结果为 environment_ready 84 / failed 93 / incomplete 7；`evaluation_ready_episodes: 0`；无交互或搜索认证。场景 YAML 中 `assembly_manifest` 写的是 `/data1/...` 绝对路径，**本机不存在**，需重新绑定路径 | 中：84 个物理通过的背景可用于 S2 / S3，但须先重绑路径、重做恢复检查 |
| living-brief pilot（uniform / cooking / meal） | 生活情境背景生成试验 | 用结构化情境描述控制背景物品分布（固定任务：找勺） | 情境描述为中文 brief，任务指令沿用固定任务 | Benevolence_1_int 厨房 | 碗、盘、盐瓶、胡椒瓶 14 件 + 目标勺 | 同上 | 同上 | `configs/layouts/living_brief_pilot/*.yaml`、`docs/living_brief_pilot.md`、`scripts/tasks/pilot_living_briefs.py` | **D**：GPU 物理验收 0/3（`BACKGROUND_PERSISTENT_DYNAMICS`） | 低：属于场景生成实验，不是可用任务 |
| `behavior_groceries_pilot` | 官方任务 `carrying_in_groceries` | 把车内纸袋中的番茄和盒装牛奶放进同一台冰箱，并关闭汽车和冰箱 | "Bring the groceries from the car into the refrigerator and close the car and refrigerator." | `house_double_floor_lower`（263 个物体；车库、厨房、客厅、走廊、浴室、花园） | 官方 BDDL 物体：纸袋、番茄、盒装牛奶、汽车、冰箱 | MOVE/TURN/VIEW/OPEN/CLOSE/GRASP/PLACE/REPORT_DONE；跨房间导航 | REPORT_DONE 调用**官方 BehaviorTask 目标判定**（`backend.behavior_goal_satisfied()`）；200 步 | `scenarios/behavior_groceries_pilot/{scenario.yaml,readiness.json}`、`configs/benchmarks/behavior_pilot.yaml`、`assets/behavior_official/bddl/activity_definitions/carrying_in_groceries/problem0.bddl`、`docs/behavior_pilot.md`、`scripts/prepare_behavior_pilot.py` | **D I J V(初始化与判定)**：2026-10-17 在本机 GPU 上验证了场景加载、全景、6 个 VIEW 方向、旧帧拒绝、复位、官方判定（初始未满足）；人工网页 MOVE/TURN 验证通过（`runs/ygbench_setup/pilot_smoke/report.json`、`human_play/report.json`）。**S 未验证**：完整抓放轨迹未运行 | 高：当前唯一在本机通过初始化验收的任务；已覆盖跨房间、容器开合和多步骤，但难度偏长程 |
| `behavior_collect_dishes_pilot` | 官方任务 `collecting_dishes_from_around_house` | 收集全屋餐具 | 无 | 预选 Wainscott_0_int | — | — | — | `scenarios/behavior_collect_dishes_pilot/readiness.json` | **未实现**：官方实例模板未安装（`unavailable_in_installed_official_releases`），已被 groceries 替代 | 暂不可用 |
| BEHAVIOR 官方目录（资产，非任务实例） | 1016 条官方活动定义的刚体筛选 | — | — | 2026-challenge 实例包覆盖 7 个场景 | — | `skill_coverage: unverified` | 官方 BDDL | `build/behavior/official_catalog.json`、`assets/behavior_official/bddl/` | **D**：1016 条定义中刚体任务 191 条，其中 **56 条有官方 2026 实例模板**；未逐条仿真 | 留给 3.2 详细分析 |

## 2. 按五级状态汇总

| 任务组 | D 定义 | I 初始化 | S 技能执行 | J 成功判定 | V 运行验证 |
|---|---|---|---|---|---|
| knife_search_001 | ✓ | ✓（历史 BUILD OK） | ✓（ORACLE，NAV 抽象） | ✓ | 历史，已被标为无效；本机未跑 |
| knife_search_v0 | 仅 split 清单 | ✗（文件缺失） | — | — | 仅 s000/s001 有历史认证记录 |
| batch_kitchen_v1（7 条） | ✓ | 5 条复用已验证容器 / 2 条待 GPU 冒烟 | 同 knife | ✓ | ✗ |
| Level-1 样例（3 条） | ✓（但遮挡/容器关系未编码） | ✗ | — | ✓ | ✗ |
| easy_v1（220） | ✓ | ✓（历史认证） | ✓（agent 协议见证） | ✓ | 历史认证；证书和本机复跑均缺 |
| household 生产批次 | ✓ | 84/184 物理通过（历史） | ✗（0 交互认证） | ✓ | ✗；路径需重绑 |
| living-brief | ✓ | 0/3 | ✗ | — | ✗ |
| behavior_groceries_pilot | ✓（官方） | **✓（本机 10-17）** | 部分（VIEW/MOVE/TURN/复位已验；抓放未验） | **✓（官方判定）** | 仅初始化与判定 |
| collect_dishes | 仅 BDDL | ✗ | ✗ | — | ✗ |

## 3. 按能力维度的可复用资产

| 能力 | 现有可用资产 | 说明 / 风险 |
|---|---|---|
| 物体获取（GRASP） | 所有搜索任务；`skills/grasp.py`；`BenchmarkWorldState.held_object` 作为持有状态的真值 | 抓取是符号执行（IK + 碰撞可行性判定后瞬时生效），不是物理抓取控制 |
| 物体寻找 | knife 系列、batch_kitchen_v1、easy_v1（220）、household | easy_v1 manifest 带有 `candidate_slot_count`、`open_depth`、`oracle_depth`、`witness_steps` 等难度字段 |
| **物体放置（PLACE）** | 技能 `skills/place.py` 已实现；`evaluation/goal_checker.py` 支持数据驱动的 `goal:` 谓词（`holding/open/closed/on_top/inside`）和 `temporal:` 时序条件，有单元测试 | **FindingBench 自有场景中没有任何一个使用 `goal:` 放置目标**（全部场景 `goal: []`）。“把 X 放到 Y 上”类任务的判定机制已有，但没有在仿真中跑过。只有官方 groceries 任务要求放置 |
| 容器打开及内部搜索 | knife 系列（3 个厨房柜/抽屉）；easy_v1 in_container 140 条（open_depth=1）；household | 认证发现 knife 所在的 `bottom_cabinet_no_top` **无顶板，不开门也能看到并抓到刀**（oracle 深度为 2 而不是 3）。做 S2“容器内”条件时必须按 `open_required` 门禁重新核验 |
| 跨房间导航 | knife 起点在客厅、目标在厨房（同屋两个房间）；groceries 从车库到厨房 | easy_v1 是 `room_count: 1`。knife 起点到三个柜子的直线距离为 5.0 / 6.7 / 8.3 m，MOVE 每步最多 1 m，按当前 MOVE/TURN 协议 **16 步预算很可能不够**，需要重新评估 |
| 多步骤任务 | groceries（抓×2、放×2、关×2，并跨房间）；Level-1 遮挡（移开遮挡物再抓） | groceries 的完整成功轨迹尚未跑通 |
| 指令/难度变体机制 | `assets/tasks/tiers/*.yaml` 中的指令模板（房间级 / landmark 区域级）；`configs/task_priors/knife_search.yaml` 的自然/反事实位置；`household_families_v1.yaml` 的 spatial × operations 2×2 profile | 现有指令基本属于 L2（目标导向）；knife 系列的情境化指令接近 L3；**没有 L1（逐步操作）指令** |

## 4. 关键发现（影响后续阶段）

1. **协议代差。** 除 groceries 之外，所有 FindingBench 场景都写于旧协议时期：YAML 列的是 NAV 技能，旧协议还有 OBSERVE。当前公共协议 `agent-visual-v2` 删除了 OBSERVE，新增 VIEW，而 VIEW 依赖 `robot.panorama`。这些旧场景没有 panorama 配置，因此 VIEW 会返回 `OUT_OF_CAPABILITY`（`core/visual_session.py:358`）。旧场景在 AGENT 模式下仍可加载，但如需与 groceries 相同的观测设置，要给它们补 `panorama` 配置块。这是一处 YAML 适配，不需要改代码。
2. **成功判定是统一的，可以直接复用。** `skills/report_done.goal_satisfied()` 有三条分支：官方 BEHAVIOR 判定 → 数据驱动的 `goal`/`temporal` → 默认的 holding(target)。L1/L2/L3 三种指令共享同一个成功条件，在这个框架内只需改 `instruction` 字段，判定逻辑不动。
3. **历史证据不能当作本机证据。** knife 和 easy_v1 的验证、认证都在旧主机（`/data1`，`behavior` 环境）上完成，证书文件也没有迁移过来。迁移包 README 明确要求在新机上重新做恢复检查。本机只有 groceries 有实测证据。
4. **场景资产齐备。** 本机 `behavior-datasets/behavior-1k-assets/scenes` 包含 easy_v1 和 household 用到的全部 7 个场景，以及 groceries 用的 `house_double_floor_lower`；R1Pro URDF 在 `build/robots/` 和 `assets/robots/r1pro/` 各有一份，后者附带验证文件。

## 5. 推荐优先作为新 Benchmark 候选的 FindingBench 任务（供第三阶段交叉对照）

| 优先级 | 候选 | 理由 | 缺口 |
|---|---|---|---|
| 1 | `knife_search_001` 及 batch_kitchen_v1 同构任务 | 我们自己设计的简单任务；场景、物体、容器锚点都已有；目标位置只需改 YAML 的 `placements` 就能构造 S1（台面，counterfactual）/ S2（柜内）/ S3（另一房间） | 补 panorama 配置；在本机重新冒烟测试；核验无顶板柜的可见性；重估步数预算 |
| 2 | `easy_v1` 子集（同场景内 on_surface 与 in_container 配对） | 已认证，带难度元数据，规模足够 | 证书缺失，需在本机复认证；当前协议下见证轨迹需要复核 |
| 3 | `behavior_groceries_pilot` | 唯一在本机通过初始化与官方判定验证的任务，覆盖放置、开合和跨房间 | 抓放轨迹未验证；长程任务，不适合作为 2×2 的第一个原型 |
| 4（如需放置任务） | 在 knife 场景上加 `goal: [on_top(target, breakfast_table_uhrsex_0)]` | 判定代码和单元测试都已存在 | 仿真中从未跑过，属于新配置，不是已有任务 |
