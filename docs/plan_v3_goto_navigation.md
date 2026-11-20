# v3 规划：指点导航（GOTO）+ 复用官方资产造任务

日期：2026-10-26　状态：**规划稿，等待确认，尚未改代码**

---

## 0. 一句话

把“直线 MOVE + 每步碰撞门禁”的导航换成 **GOTO**：模型在当前画面上点一个点，系统在 2D 栅格上跑 A* 检查能否到达，能到就把底盘传送过去。手臂和躯干的关节不动，**GRASP / PLACE / OPEN 的可行性门禁一律不改**。目的是让大多数 BEHAVIOR 场景都能做跨房间（S3）任务，再用现有 v2 流水线批量造任务。

---

## 1. 现在的问题到底是什么

| 现象 | 真正原因 | 证据 |
|---|---|---|
| 19 个场景里只有 3 个能出 S3 | MOVE 只能走直线，每一步都要过 0.8 m 方形底盘 + 3 cm 余量的碰撞检查，家具按外包框算 | `runs/vla_difficulty_v2/_survey/*.candidates.json` |
| 很多场景“只能到一个房间” | 没挂 manifest 时，MOVE 门禁用**整面墙实体的外包框**，带门洞的墙把门洞整个封住（`backend.navigation_obstacle_aabbs`，约 1100 行） | 2026-10-26 Wainscott_1 挂上 manifest 后：可达面积 5.0 → 33.9 m²，有可达家具的房间 3 → 7 |
| “原作者验证过能通行”不成立 | 官方 `floor_trav_*.png` 只标空地，**不含机器人尺寸**（默认腐蚀半径 0），而且老场景里**门洞在图上也是封住的**（Rs_int 开门版地图里，就算把机器人当成一个点，最大连通区域也只有 13 m²） | 见第 3.2 节的表 |
| 已验证配置里的失败 | 和导航有关的只有 2 个（路线 UNSAFE）。其余是：S1 RGB 对比度不够 9、物体太大在 S2/S3 露出 9、初始化放置 4、抓不到 4、放不下 3、仿真错误/超时 3 | `runs/vla_difficulty_v2/summary.md` |

结论：资产本身够用，卡住的是**我们自己的导航技能**和**门洞的障碍表示**。抓放门禁筛掉的是真实的物理约束，按用户要求保留。

---

## 2. 不变的部分（明确不动）

- **GRASP / PLACE / OPEN / CLOSE 的可行性门禁**：底盘距离、IK、手臂和持物的碰撞检查，代码和阈值都不改。
- L1–L3、S1–S3 的定义（`assets/tasks/vla_difficulty_v1/levels.yaml`），S 门禁（S1 RGB 可见且有对比度，S2/S3 起点看不到）。
- VIEW（全景视口）、REPORT_DONE、`PointAction` 的坐标约定（当前帧上的归一化点，带 `frame_id`）。
- 现有的 v1（36 个变体）和 v2（918 个变体）文件，都不删、不改名。MOVE / TURN 技能保留，旧场景配置照常能跑。
- 队友的文件（`scripts/prepare_clearance.py`、`scripts/environment/*`、`sim/omnigibson/clearance.py`）不改。

---

## 3. 新技能 GOTO

### 3.1 接口

```json
{"skill": "GOTO", "point": {"frame_id": "<当前帧>", "x": 0.42, "y": 0.81}}
```

和 GRASP 用同一个 `PointAction.Point`。点可以落在地面上，也可以落在家具上（表示“走到它旁边”）。

### 3.2 执行流程（每次调用算 1 步）

1. **反投影**：用当前帧的深度和相机位姿，把像素点换算成世界坐标 (x, y, z)。点在天空或无效深度上 → `INVALID_TARGET`。
2. **确定终点候选**：
   - 点在地面上（z 接近地面）→ 以该点为中心、0.5 m 半径内找底盘位姿；
   - 点在家具或物体上 → 用现有的环形位姿采样（`route_oracle._ring_poses`，离家具外包框 0.45–1.1 m，面朝它）。
   - 每个候选终点都必须通过**现有的完整底盘门禁**（`FootprintChecker.free`，0.8 m 方形 + 3 cm 余量）。这一步不放宽。
3. **A\* 检查**：在 2D 占据栅格上，从当前底盘位置到候选终点做 A*（8 邻域）。候选按到点击点的距离排序，取第一个有路的。
4. **传送**：底盘设到终点位置。朝向 = 面向点击点（家具则面向家具）。**手臂、躯干、头部关节、手里的物体保持不变**（持物跟着手走，和 MOVE 一致）。然后物理 settle，渲染新观测。
5. **反馈**：`EXECUTED`；没有可达终点 → `UNREACHABLE`（新反馈码，和 `MOVE_BLOCKED` 区分）。

### 3.3 A* 用的占据栅格（不用官方 trav map）

官方开门版地图半径 0 时的最大连通区域（m²），说明它不能直接拿来判断跨房间：

| Rs_int | Ihlen_0 | Wainscott_1 | house_double_floor_upper | hotel_suite_large |
|---|---|---|---|---|
| 13 | 33 | 46 | 25 | 37 |

所以栅格从**场景里真实的碰撞体**生成：
- 墙、家具：按**每个碰撞体**算外包框（`usd_collision.collider_navigation_aabbs`，和 MOVE 门禁在挂 manifest 时的算法相同），只取 0.05–0.5 m 高度带（`navigation.Z_LOW/Z_HIGH`）；
- 室内门板：在场景配置里声明移除（manifest 只用 `removed_objects`，**不再拓宽墙**），等价于“门开着”；
- 0.05 m 分辨率，在 reset 后算一次并缓存；GRASP/PLACE 改变的只是小物体，不影响栅格。

### 3.4 路径半径（需要你拍板）

A* 用的“机器人半径”决定窄门能不能过。终点始终用完整 0.8 m 底盘检查，这里只决定**沿途**的宽松程度：

| 选项 | 含义 | 影响 |
|---|---|---|
| A. 0.40 m（底盘半宽） | 和真实底盘一致，侧身也不行 | 0.75–0.9 m 的门大多过不去，S3 场景仍然少 |
| **B. 0.30 m（推荐）** | 能过 ≥ 0.6 m 的通道，几乎所有门洞都能过 | 和真实尺寸不完全一致，在文档和论文里写明“导航按可通行拓扑判断，不评测底盘运动” |
| C. 0（点机器人） | 只要连通就行 | 会穿过家具之间的窄缝，太宽松 |

### 3.5 对难度维度的影响（要写进 benchmark 说明）

- S3 测的是：**知道该去哪里找**（房间语义、探索、记住看过什么），而不是开车技巧。
- GOTO 只能点**当前画面里看得见的点**（含 VIEW 视口）。所以跨房间时仍然要先看到门口、走过去、再看下一处，一步跨不过整个房子。S3 比 S1 多出的步数来自“搜索 + 走到门口”，这正是想测的。
- MOVE / TURN 仍然开放（用于微调朝向和距离），模型可以不用。

---

## 4. Oracle 和验证流程的改动

- `RouteOracleAgent` 增加一个 GOTO 规划器：对目标位姿，从当前位置**找一个能在当前帧看到、且 A* 可达的地面点**，逐段推进（每段 1 步）。到了之后再走原来的 GRASP/PLACE 位姿搜索（门禁不变）。
- 三级验证保持：初始化 + S 门禁 / 空报告被拒 / 特权脚本 oracle 只用公共动作完成。新增：oracle 的每一次 GOTO 都必须是合法的公共动作（点在当前帧里）。
- 步数预算：仍是 60。oracle 步数 ≤ 30 的筛选条件保留。

---

## 5. 场景和任务怎么扩

1. **场景筛选（CPU）**：对 51 个场景生成第 3.3 节的栅格（墙和门用 CPU 读 USD，`scripts/v2/propose_clearance.py` 已经能做到；家具用勘察结果），按“房间连通数 ≥ 3、每间可达 ≥ 4 m²”排序。先看有官方 BEHAVIOR 任务实例的住宅类场景：Wainscott、Beechwood、Ihlen、Merom、Pomaria、Benevolence、Rs、house_*、hotel_*。
2. **GPU 勘察**：`survey_scene.py`（挂只含 `removed_objects` 的 manifest）→ `analyze_survey.py` 改用 GOTO 的可达性（同一个栅格 + 终点门禁）。
3. **其余流水线不变**：`certify_locations.py` → `make_phrases.py` → `generate_tasks.py` → `compile_variants.py` → `run_batch.py` → `refine_loop.py`。物体和子目标继续从官方 476 条活动定义里取。
4. **规模估计**：目前 3 个场景、102 个基础任务，约一半配置能通过。如果能用的场景达到 8–10 个，每个场景出 15–25 个基础任务，就有希望达到约 111 个三个 S 全过的基础任务（≈ 1000 个变体）。这个数字要等第 8 节的里程碑 M2 实测后再确认。

---

## 6. 交付物

| # | 内容 | 位置（计划） |
|---|---|---|
| D1 | GOTO 技能：反投影、终点采样、A*、传送、反馈码 | `src/rummagebench/skills/goto.py`，注册进 `skills/registry.py`，`core/public_types.py` 加 `GOTO` |
| D2 | 2D 占据栅格 + A*（可复用 OmniGibson 的 `motion_planning_utils.astar`） | `src/rummagebench/sim/omnigibson/nav_grid.py` |
| D3 | 单元测试：栅格、A*、反投影、终点必须过完整底盘门禁、持物随手走、不可达反馈 | `tests/unit/test_goto_*.py` |
| D4 | Oracle 的 GOTO 规划器 | `src/rummagebench/evaluation/route_oracle.py`（新增函数，不改原有 MOVE 路径） |
| D5 | v3 任务公共配置：技能列表 `[GOTO, TURN, MOVE, VIEW, OPEN, CLOSE, GRASP, PLACE, REPORT_DONE]`、路径半径、提示词里的动作说明 | `assets/tasks/vla_difficulty_v3/common.yaml` |
| D6 | 场景筛选报告（51 个场景的连通房间数和面积） | `runs/vla_difficulty_v3/_scenes/summary.md` |
| D7 | 新场景勘察 / 认证结果，生成的基础任务和变体 | `runs/vla_difficulty_v3/`、`assets/tasks/vla_difficulty_v3/base_tasks/`、`build/vla_difficulty_v3/` |
| D8 | 汇总：每格（L×S）可用变体数、失败原因分布 | `runs/vla_difficulty_v3/summary.md` |
| D9 | 人工抽查清单（每格约 5 条）+ 试玩命令 | `runs/vla_difficulty_v3/spot_check.md` |
| D10 | 说明文档：GOTO 的定义和局限、与 v1/v2 的差别 | `assets/tasks/vla_difficulty_v3/README.md`，并更新研究文档的交接节 |

v2 的 102 个基础任务可以**直接改成 v3 配置重跑**（物理布局不变，只换技能列表和 oracle），不用重新生成。

---

## 7. 验收标准

1. **GOTO 正确性**：单测全过；全量单测不回退（目前 1023 passed / 24 skipped 是 10-19 的数字，开工前先重跑一次作基线）。
2. **门禁未放宽**：同一批 v2 配置在 v3 下，GRASP/PLACE 的可行性判定结果和 v2 一致（抽 20 个对比）。
3. **跨房间真的可达**：每个入选场景至少 3 个房间互相可达，并且有 oracle 实跑轨迹为证（不是只看 CPU 估算）。
4. **任务数量**：如实报告每格数量。目标是每格约 111 个，达不到时写明差距和原因。
5. **可复现**：所有变体由配置编译而来；每个通过的变体都有 `report.json`（初始化、空报告被拒、oracle 成功、步数）。
6. **人工抽查**：每格至少 5 条，记录在 D9。
7. 严格区分 Task Initialization / Scripted (oracle) Success / VLA Success，最后一项由队友接模型测。

---

## 8. 里程碑和顺序

| 里程碑 | 内容 | GPU |
|---|---|---|
| M0 | 暂停拓宽门洞队列（**已完成**），v2 批量继续 2 路；跑全量单测作基线 | 否 |
| M1 | D1–D3：GOTO 技能 + 栅格 + 单测；在 Beechwood_0 上手动试几步 | 1 路，短 |
| M2 | D4–D5：oracle 改造；拿 v2 已通过的 11 个基础任务（99 个变体）在 v3 下重跑，对比步数和结果 | 1 路 |
| M3 | D6：51 个场景筛选（CPU），选 8–10 个；对前 3 个做勘察 + 认证 | 1 路 |
| M4 | D7–D8：生成任务、批量验证、refine，直到数量够或场景用尽 | v2 批量结束后 3 路 |
| M5 | D9–D10：人工抽查、文档、交接 | 否 |

---

## 9. 需要你决定的事

1. **A\* 路径半径**：推荐 0.30 m（第 3.4 节）。
2. **MOVE / TURN 是否继续开放给模型**：推荐保留，GOTO 为主。
3. **v2 批量**：继续用 2 路跑完（推荐，S1/S2 的可见性和抓放结论可以直接用于 v3），还是现在就停，把 GPU 全给 v3。
4. GOTO 点到家具上时，是“走到它旁边”（推荐），还是只允许点地面。

---

## 10. 当前进程和文件状态（2026-10-26 13:30）

- **在跑**：`runs/vla_difficulty_v2/queue_v2_jobs2.sh`（PID 文件 `queue_v2_jobs2.pid`）：v2 批量 2 路，跑完后自动跑 `refine_loop.py --rounds 4 --jobs 2`。
- **已停**：`_clearance/queue_certify.sh`。门洞实测已跑完（`_clearance/<scene>_replay/`）：拓宽到 1.2–1.4 m 的 34 个门洞里，从两侧各用直线 MOVE 试一次，能穿过的只有 12 个（Wainscott_1 3/9、Beechwood_1 3/9、Ihlen_0 3/5、Wainscott_0 2/6、Pomaria_1 1/5）。失败的 22 个都至少有一侧离门 0.65 m 的位置被家具占住，另一侧的 MOVE 被判 UNSAFE（18 个），或两侧都被占（4 个）。门口附近的家具让“拓宽门洞 + 直线 MOVE”很难走通，这也是改用 GOTO 的直接依据。Wainscott_1 的位置认证在中途被停掉。`_clearance/queue_scenes.sh` 此前已经正常跑完。
- **已经生效、需要注意**：`build/vla_difficulty_v2/clearance/` 下新增了 Wainscott_1、Beechwood_1、Ihlen_0、Wainscott_0、Pomaria_1 五个**带墙体拓宽**的 manifest，`_survey/` 下这五个场景的勘察结果已按拓宽后的几何更新（旧结果在 `_survey/pre_clearance/`）。v2 现有任务都不在这五个场景，不受影响。v3 不需要拓宽墙，开工时会把这五个 manifest 换成只含 `removed_objects` 的版本。
- 新增工具（可继续用）：`scripts/v2/propose_clearance.py`（CPU 读墙和门的碰撞体、估算连通性）、`scripts/v2/replay_doorways.py`（逐门洞实测）。
