# vla_difficulty_v1：交叉复用分析、任务设计与可运行原型（2026-10-19 更新）

承接 `docs/phaseA_3.*`（现有资产调查）与 `docs/phaseB_paper_tasks.md`（论文任务调查，目标论文 BEHAVIOR-1K）。本文件覆盖 Phase C–F：交叉映射、任务设计、实现、仿真验证和运行说明。所有数字来自本机 `ygbench` 的实际运行，原始记录在 `runs/vla_difficulty_v1/`。

## 1. 结论速览

- **当前网格：4 个基础任务 × L1–L3 × S1–S3 = 36 个变体，每格 4 个任务**（T01 取刀、T02 杯子放到台面、T03 叉子收进抽屉、T04 苹果放到台面）。全部变体用最终代码跑过，结果见第 6 节。
- **“能否放下取决于怎么抓起来的”已查明并修复（第 4.1 节）**：抓取时物体挂在较近的那只手上，偏移量按左手记录，而 IK / 碰撞检查用的是右手。修复后，此前失败的 T04 L2_S1、L2_S3 都通过了，单个取放变体的耗时也从 5–28 分钟降到 6–10 分钟（PLACE 检查不再在不可行位姿上遍历整个网格）。
- **T03（叉子收进抽屉）已打通（第 4.2 节）**：先后修了 6 个问题。其中“往抽屉里放东西允许手斜着进去”由用户批准（2026-10-19）。T03 还有顺序约束：必须先开目标抽屉再拿叉子，L2/L3 要求模型自己推断这一点。
- **VLA Task Success 尚未测**：按用户决定，由队友接模型；接口为 `--agent module:factory`（`src/rummagebench/agents/policy_adapter.py`）。

## 2. 交叉映射（Phase C）

| 候选任务 | FindingBench 支持 | BEHAVIOR 支持 | 论文 / 官方来源 | 缺少的部分 | 状态 |
|---|---|---|---|---|---|
| **T01 取刀**（holding） | knife_search_001：同场景、同刀模型、同 S2 抽屉 | 普通场景路径 + AGENT 技能 + `goal_checker.holding` | 无官方“只拿起”任务（官方目标都是放置） | 无 | **网格内** |
| **T02 杯子 → 走廊门旁台面**（on_top） | 同场景、杯子模型 dhnxww | `goal_checker.on_top` + PLACE（表面） | set_up_a_coffee_station（ontop）、bringing_water（ontop，跨房间） | 无 | **网格内** |
| **T04 苹果 → 走廊门旁台面**（on_top） | 同场景、苹果模型 agveuv | 同上 | bringing_water、moving_boxes_to_storage（ontop） | 无 | **网格内** |
| **T03 叉子收进抽屉**（inside 家具抽屉） | 同场景、叉子 flexrc、目标抽屉 = knife_search_001 的 qohxjq_0 | `goal_checker.inside` + 抽屉内部 PLACE（斜向接近，2026-10-19） | putting_dishes_away_after_cleaning、storing_food 等 39 条 inside(·, 家具) | 无 | **网格内** |
| 苹果放进冰箱 / 放到餐桌、茶几 | — | — | freeze_fruit、bringing_water 等 | 冰箱前无无碰撞底盘位姿；餐桌、茶几 PLACE 全碰撞 | 废弃（`base_tasks/_superseded/`） |
| carrying_in_groceries 官方任务 | `behavior_groceries_pilot`（本机实测初始化 + 官方判定） | 官方 BehaviorTask | BEHAVIOR-1K | 指令 / 初始位置覆盖适配；inside 冰箱 | 参考，不进网格 |

“表面相似、实际不同”的情况：T01 与 knife_search_001 同物体同抽屉，但旧任务用 NAV 锚点协议、拿到即成功；新任务用 AGENT 协议（MOVE/TURN/VIEW/点选）+ REPORT_DONE 判定。T02 与 bringing_water 同为 ontop，但官方任务是从关着的冰箱取两瓶水，多物体、量词目标。T03 与官方 inside(·, cabinet) 任务目标形式相同，但官方任务没有“手里拿着东西不能开抽屉”的顺序约束——我们的单手模型有。

## 3. 任务设计（Deliverable 4）

定义文件：`assets/tasks/vla_difficulty_v1/`（README、levels.yaml、common.yaml、base_tasks/、variants.yaml）。公共条件：Beechwood_0_int，R1Pro，起点 kitchen (0.85, −7.70)，2048×1024 全景 + VIEW，技能 MOVE/TURN/VIEW/OPEN/CLOSE/GRASP/PLACE/REPORT_DONE，60 步，成功只由 REPORT_DONE 时的 goal 判定。

| 任务 | 目标（成功条件） | L1（S1 例） | L2 | L3 |
|---|---|---|---|---|
| T01 取刀 | `holding(target_knife)` | Go to the kitchen island, pick up the table knife lying on it, and then report that you are done. | Get the table knife for me. | I want to slice an apple for a snack. Could you get me something to cut it with? |
| T02 杯子 → 走廊门旁台面 | `on_top(target_mug, countertop_tpuwys_0)` | Go to the kitchen island and pick up the mug on it. Then carry it to the counter next to the hallway door, place it on the counter, and report that you are done. | Put the mug on the counter next to the hallway door. | I'll make my coffee at the counter by the hallway door in a minute. Could you have my mug waiting for me there? |
| T03 叉子 → 走廊门旁柜子顶层抽屉 | `inside(target_fork, bottom_cabinet_no_top_qohxjq_0)`（抽屉之后重新关上也算） | First go to the top drawer of the cabinet next to the hallway door and open it. Then go to the kitchen island, pick up the fork on it, bring it back, put it inside the open drawer, and report that you are done. | Put the fork in the top drawer of the cabinet next to the hallway door. | We're done eating. Could you put the fork away in the cutlery drawer by the hallway door? |
| T04 苹果 → 走廊门旁台面 | `on_top(target_apple, countertop_tpuwys_0)` | Go to the kitchen island and pick up the apple on it. Then carry it to the counter next to the hallway door, place it on the counter, and report that you are done. | Put the apple on the counter next to the hallway door. | I'm going to cut up some fruit at the counter by the hallway door. Could you bring the apple over there for me? |

| S | 目标初始位置（四个任务相同，T03 的 S2 例外） | 额外操作 |
|---|---|---|
| S1 | 厨房岛台 `countertop_tpuwys_4` | — |
| S2 | 走廊门旁柜子顶层抽屉 `bottom_cabinet_no_top_qohxjq_0/link_2`（上有台面，关闭时看不到） | OPEN |
| S3 | 客厅电视下矮书架 `bookcase_owvfik_1` 顶面，固定在 `position_xy: [4.45, -5.45]`（书架远端，从起点看不到） | 跨房间（开放式）导航 |

T03 的 S2：目标抽屉就是共用的 S2 抽屉，所以叉子放在相邻柜子 `bottom_cabinet_no_top_pluwfl_0`（同样关着看不到，需要 OPEN）。T03 的顺序约束：拿着叉子不能开抽屉（持物会撞抽屉面板），必须先开目标抽屉；L1 写明了顺序，L2/L3 要模型自己推断。

可比性控制：同一任务的 9 个变体只差指令文字和目标物体初始位置；背景物体、起点、相机、预算、成功条件相同。S2 的 OPEN、S3 的额外导航记录在 `extra_ops_vs_S1` 中，不应把步数增加全部归因于空间推理。T02/T04 的 S2 起点抽屉正好在目标台面下方，搬运距离比 S1/S3 短（`confound_note`）。L3 的竞争解释已从场景中移除（各任务 `l3_ambiguity_check`）。L2/L3 指令不随 S 变化；L1 写出位置和步骤，所以 L1 下空间维度只衡量执行 / 导航。

## 4. 仿真中证伪并已修正的设计点

### 4.1 抓取手与可行性模型的手不一致（2026-10-19，影响所有取放任务）

| 现象 | 原因 | 修正 | 证据 |
|---|---|---|---|
| 同一个台面，苹果从有的位置抓起后能放下，从别的位置抓起后所有位姿都放不下；T03 拿着叉子时开抽屉“有时可行” | R1Pro `arm_names = [left, right]`；`backend.eef_pose()` 返回 `arm_names[0]`（左手），`symbolic_grasp` 把物体挂到离得近的手上，而可行性引擎（IK + 持物碰撞体）用的是 `right_eef_link`。于是 `held_offset` 描述的是另一只手，PLACE 检查时持物被放在离右手几十厘米的地方 | `backend._modelled_arm()`：抓取和 `eef_pose()` 都用场景 kinematics 指定的那只手 | `tests/unit/test_backend_modelled_arm.py`；修复前结果 `runs/vla_difficulty_v1/_history/*_before_arm_fix`；修复后 T04 L2_S1（13 步）、L2_S3（15 步）由失败变为成功 |

### 4.2 T03 叉子收进抽屉：逐个排除后打通

| 步骤 | 现象 | 处理 | 证据 |
|---|---|---|---|
| 1 | 修复 4.1 后，拿着叉子对目标抽屉 OPEN，所有位姿都碰撞 | 碰撞对全部是 `held:target_fork ~ qohxjq_0:link_0/1/2`；不算持物时 4/6 位姿可行 → 这是真实约束：单手拿着东西不能拉抽屉。oracle 改为先开目标抽屉再取物（`derive_subgoals`），L1 指令改为先开抽屉 | `runs/vla_difficulty_v1/_diagnose/T03_fork_to_drawer__L2_S1_OPEN_*.json`；`plan_constraint` |
| 2 | PLACE 判定可行，但执行后叉子掉出抽屉 | 候选点离抽屉侧壁 6 cm，18 cm 的叉子插进侧壁被弹出 → 候选点按持物半尺寸 + 2 cm 内缩（`INSIDE_WALL_CLEARANCE_M`）；`symbolic_place` 改用物体原生包围盒（放下时的姿态）而不是手中旋转后的 AABB | `_diagnose/T03_fork_to_drawer__L2_S1_PLACE_*.json` |
| 3 | 叉子穿过抽屉底板掉下去 | 抽屉内部区域原来用抽屉链接 AABB，其底面是面板底边，低于真实底板 → 改用原生 fillable 体积（`visual_aabb`） | `_diagnose/after_footprint_fix/` |
| 4 | 叉子被放进最下层抽屉 | OPEN 会打开柜子的所有抽屉；原规则选“拉出最多”的 → 改为选最上层（下层被上层盖住，从上方够不到） | `_diagnose/after_fillable_region/` |
| 5 | 选对顶层抽屉后，10 个位姿 PLACE 全部碰撞 `countertop_tpuwys_0` | 顶层抽屉被台面盖住，PLACE 原来只建模自上而下接近。**用户批准斜向接近**：放进打开的抽屉时，工具从拉出的一侧、与竖直方向成 45°/30°/60°/0° 斜向进去（`INSIDE_APPROACH_TILTS_DEG`），放下时保持接近姿态，只要求物体中心到达放置点；只改 inside 路径，表面放置不变 | `_diagnose/after_top_drawer/`、`after_open_strip/`（失败），`after_angled/`（成功：第一个位姿可行，叉子在顶层抽屉内，inside 成立） |
| 6 | 在岛台某个抓取位姿抓起叉子后，下一步底盘 z 从 0.005 升到 0.20，随后 TURN 被拒（L1_S1、L2_S1 可复现） | 叉子原点被放在 EEF 上，和右手夹爪 / 相机连杆重叠，接触把底盘顶起（`scripts/probe_grasp_lift.py`）。先试了运行时过滤“持物 ↔ 机器人”碰撞，结果 PhysX 张量视图失效、抓取失败，已撤回。最终做法：**长物体（最大半尺寸 > 8 cm）挂在指尖前方**（`LONG_OBJECT_HALF_M`），杯子、苹果不变 | `_diagnose/grasp_lift.json`；同一抓取位姿重跑 T03 L3_S1，底盘 z 全程 0.005 |

### 4.3 更早的修正（2026-10-18）

| 原设计 | 仿真结果 | 修正 |
|---|---|---|
| 起点 = knife_search_001 的 kitchen 锚点 (0.75, −8.15) | 底盘与椅子重叠，reset 失败 | 移到 (0.85, −7.70) |
| 全景 1024×512 | 2.4 m 外的刀只有 6 px | 2048×1024 |
| S1 门禁 ≥100 px | 扁平餐刀达不到 | 起点全景 ≥20 px |
| S2 = `bottom_cabinet_rvpunw_0/link_2` | 链接名不存在；落地抽屉 GRASP 不可行 | 换成 `qohxjq_0/link_2` |
| S3 = 办公室书桌 / 餐厅矮桌 | 门宽约 0.75 m < 底盘 0.86 m | 客厅 `bookcase_owvfik_1` |
| T02 目标餐桌、T04 目标客厅茶几 / 岛台 | PLACE 碰撞，或依赖抓取方式（实为 4.1 的 bug） | 统一放到 `countertop_tpuwys_0` |

## 5. 放置目标认证

| 表面 | 结果 |
|---|---|
| 走廊门旁台面 `countertop_tpuwys_0` | ✅ T02、T04 共 18 个变体的目标 |
| 岛台 `countertop_tpuwys_4` | ✅（修复 4.1 前测得，杯子） |
| 厨房餐桌、客厅茶几、客厅餐桌 | ❌ 所有位姿碰撞 |
| 客厅矮书架 `bookcase_owvfik_1` | ❌ 太矮 |
| `qohxjq_0` 顶层抽屉内部 | ✅ 斜向接近（4.2 第 5 步）→ T03 目标 |

成功过的 PLACE 位姿写成评测器私有锚点 `<entity>__PLACE_<i>`（`common.yaml`），oracle 先试锚点再做环形搜索；AGENT 模式不暴露锚点。

## 6. 运行结果（Deliverable 6）

完整表格：`runs/vla_difficulty_v1/summary.md`（`python scripts/run_vla_batch.py --summarize-only` 重新生成）。三个层级分开报告：Task Initialization / Scripted Execution（特权 oracle 只用公共动作完成）/ VLA Task Success（未测）。

**36/36 个变体三级验证全部通过**（2026-10-19 04:25，最终代码；T03 S2 打开后可见 230 px）：

| 任务 | 初始化 + S 门禁 | “未完成就报告”被拒 | 脚本化执行 | 最少步数 S1 / S2 / S3（预算 60） | 打开抽屉后目标像素（S2） |
|---|---|---|---|---|---|
| T01 取刀 | 9/9 | 9/9 | 9/9 SUCCESS | 8 / 7 / 9 | 316 |
| T02 杯子 → 台面 | 9/9 | 9/9 | 9/9 SUCCESS | 17 / 8 / 15 | 603 |
| T03 叉子 → 抽屉 | 9/9 | 9/9 | 9/9 SUCCESS | 22 / 14 / 18 | 230 |
| T04 苹果 → 台面 | 9/9 | 9/9 | 9/9 SUCCESS | 13 / 8 / 15 | 388 |

起点全景中目标像素：S1 为 37（刀）/ 100（杯）/ 29（叉）/ 80（苹果）；S2、S3 均为 0（门禁要求）。所有运行中底盘 z ≤ 0.011 m。全部 36 个变体都在 02:59:38（最后一次修改代码）之后启动，最早的一个是 03:02:39，所以全部结果都出自最终代码。T02、T04 在更早的版本上也是 18/18，步数相同。旧结果在 `runs/vla_difficulty_v1/_history/`。

同一 S 下三个 L 的脚本化结果基本相同是预期的：oracle 不读指令，L 维度的差异只会在模型上出现。S2 的最短步数可能比 S1 还短——空间等级是实验条件，不是预设的难度排序；分析模型结果时应对照这个“最少步数”基线。

每个变体目录：`report.json`（三级结果、门禁、oracle 日志）、`start_panorama.png`、`null_baseline/events.jsonl`、`route_oracle/{events.jsonl, step_*.png, summary.json}`。日志：`runs/vla_difficulty_v1/logs/`。

## 7. 新增 / 修改的代码

| 文件 | 内容 |
|---|---|
| `src/rummagebench/core/scenario.py` | goal 的 receptacle 允许是场景原生家具；`PlacementSpec.position_xy` |
| `src/rummagebench/sim/omnigibson/backend.py` | `_place_at_xy`；抽屉内部区域（`receptacle_interior_regions` 用 fillable 体积、`_with_open_strip` 记录拉出条带和方向）；**`_modelled_arm`（4.1）**；`place_half_extents`、`held_entity_name`；`symbolic_place` 按原生包围盒放置；长物体挂在指尖前方（`_bring_to_hand`） |
| `src/rummagebench/feasibility/interaction_target.py` | inside 区域放置点（抽屉底面、持物尺寸内缩、斜向接近 `_angled_inside_targets`）；`select_open_interior` 选最上层 |
| `src/rummagebench/validation/feasibility.py` | PLACE 接近段：inside 区域放下时保持接近姿态 |
| `src/rummagebench/evaluation/goal_checker.py` | inside 目标在抽屉重新关上后仍按抽屉内部判定 |
| `src/rummagebench/evaluation/route_oracle.py` | 特权脚本 oracle；目标容器先于抓取打开；评测器私有锚点 |
| `src/rummagebench/agents/policy_adapter.py` | 模型无关策略适配器 |
| `scripts/compile_vla_variants.py`、`run_vla_variant.py`、`run_vla_batch.py` | 编译、单变体三级验证、批量与汇总（blocked 任务单独标注） |
| `scripts/certify_vla_locations.py`、`probe_vla_scene.py`、`plot_vla_reachability.py` | 位置认证、场景探针、可达性图 |
| **`scripts/diagnose_vla_feasibility.py`** | 某技能为何不可行：逐位姿打印碰撞对（带 / 不带持物）；`--execute-place` 实际执行放置并输出关系判定 |
| **`scripts/probe_grasp_lift.py`** | 在指定位姿抓取，检查底盘是否被抬起、持物与哪些连杆重叠 |
| `tests/unit/test_backend_modelled_arm.py`、`test_inside_drawer_place.py`、`test_route_oracle.py` 等 | 全量单测 1020 通过 / 24 跳过（2026-10-19） |

未修改：FindingBench 原有场景与任务、easy_v1、成功判定器的判定逻辑。

## 8. 已知限制与待决定事项

- 斜向接近只用于“放进打开的抽屉”；表面放置仍只建模自上而下。
- 长物体挂在指尖前方是符号化抓取的实现细节，不影响成功判定；杯子、苹果仍挂在 EEF 原点（T02/T04 的放置认证基于此）。
- 脚本化 oracle 是特权的（读位姿、可行性、分割），只证明“可解 + 预算够”，不是模型基线。
- 跨房间只覆盖开放式相连的客厅；真正隔门的多房间需要换场景或缩小底盘。
- L3 的“歧义已移除”只针对我们生成的物体，场景原生小物件未审计。
