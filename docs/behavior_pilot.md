# BEHAVIOR 原生任务试跑与模型接口（2026-10-17）

当前代码协议是 `agent-visual-v2`。公共技能为 MOVE、TURN、OPEN、CLOSE、GRASP、PLACE、VIEW、REPORT_DONE。OBSERVE 已移除，其请求返回 INVALID_ACTION；历史 authoring 证据仍可按 v1 读取。新路线不调用自造布局或任务生成器。

## 已选数据

- 官方任务：`carrying_in_groceries`，definition 0。
- 选定场景：`house_double_floor_lower`，保留完整场景，不启用仅任务物体加载。
- 官方初始条件：车库内打开的汽车里有纸袋，袋内有番茄和盒装牛奶。目标：两件食品放入同一冰箱，并关闭汽车和指定冰箱。保留整屋场景，从车库到厨房完成实际收纳任务；袋和盒装牛奶均为官方刚体模型，不涉及布料或液体模拟。
- 原始定义：`assets/behavior_official/bddl/activity_definitions/carrying_in_groceries/problem0.bddl`。官方来源提交、许可证见 `assets/behavior_official/`。
- 实例来源：官方 `2026-challenge-task-instances`，definition 0 / instance 0；包含官方 R1Pro 初始位姿。此前收集餐具任务只有任务定义和场景匹配信息，当前官方包没有其模板，故更换为实际可绑定的数据。
- 选择配置：`configs/benchmarks/behavior_pilot.yaml`；运行准备状态：`scenarios/behavior_groceries_pilot/readiness.json`。

原始 BDDL 定义和 rigidBody 分类已保存；所有任务对象均通过刚体分类检查。资产的原始摆放、机器人出生位姿、BDDL 量词和目标由官方 BehaviorTask 加载/评估。REPORT_DONE 调用官方目标判断，不再使用“拿到一个目标即成功”。无重采样，无自造替代场景。

## VIEW 与图像

相机挂在机器人实际 `zed_link` 下，以 `torso_link4` 的正前方/上方定义固定安装方向，并将 `[0, 0, 0.15]` 米偏移转换到倾斜的 ZED 局部坐标；因此光心在 ZED 上方 15 cm。已在真实仿真中核对光心位置与朝向；全景下方会看到机器人自身，未隐藏本体。仿真用六个**同一光心、固定外参**的立方体渲染面实现一台逻辑全景相机，不增加实体运动自由度。

初始观测是 1024×512 全景 RGB。VIEW 对当前全景缓存作透视取景，返回 640×480 RGB，固定水平视场角 90°。此动作不调用渲染、物理推进、底盘、机械臂或头部关节控制；只改变视觉方向。其他动作改变世界后重新采集全景，并沿用选定方向。

```json
{"skill":"VIEW","yaw":90,"pitch":0}
```

`yaw` 必填，范围 [-180,180]，正数向左；`pitch` 可省略，默认 0，范围 [-90,90]，正数向上。单位均为度，都是相对相机固定正前方的**绝对角度**，不是累加转角。VIEW(0,0) 回到正前方。方向随头顶安装基准而非世界坐标变化。每次动作计一个规划步，包括 VIEW 和无效动作；本条试跑预算 200 步。

切换视角产生新 frame_id，旧图上的像素引用立即失效。RGB、私有深度、实例 ID 和射线使用相同采样位置，保持点击定位的一致性；深度和实例信息不提供给模型。

## 大模型接口

Python 入口为 `rummagebench.environment.env.InteractiveSearchEnv`：

```python
from rummagebench.environment.env import InteractiveSearchEnv

env = InteractiveSearchEnv("scenarios/behavior_groceries_pilot/scenario.yaml", mode="agent")
initial = env.reset()
obs = initial["observation"]
# 将 obs 的 instruction、image_png_b64、skill_library、feedback 等提供给模型。
result = env.step({"skill": "VIEW", "yaw": 90})
obs = result["observation"]
# 由你的模型继续生成动作；终态时停止调用。
```

完整机器可读 JSON Schema：`configs/protocol/action_v2.schema.json`、`observation_v2.schema.json`、`step_result_v2.schema.json`。不要把评估器私有轨迹、实体名字、深度或任务初始条件直接发给模型。

| 技能 | 参数 |
| --- | --- |
| MOVE | `distance_cm`：正数前进，负数后退；绝对值 5–100 cm |
| TURN | `angle_deg`：正数底盘左转，负数右转；绝对值 5–180° |
| VIEW | `yaw`，可选 `pitch`；只选视觉方向 |
| OPEN / CLOSE / GRASP / PLACE | `point: {frame_id, x, y}`，当前图的归一化像素，x/y ∈ [0,1] |
| REPORT_DONE | 无参数；官方目标满足则 SUCCESS，否则 FAIL_FALSE_COMPLETION |

PLACE 的点选中接收物体/容器，沿用原执行器语义。正常反馈仍为 EXECUTED、INVALID_ACTION、OUT_OF_CAPABILITY、UNSAFE。观测包含 instruction、frame_id、image_png_b64、planning_step、max_planning_steps、skill_library、feedback。MCP 的 `observe` 读取接口只是读取当前观测，不是被删除的 OBSERVE 动作；Python 模型循环直接使用 reset/step 返回的 observation。

## 绑定官方实例并启动

在已安装 OmniGibson、Isaac Sim、BEHAVIOR 原始资产以及原有 Pinocchio 运动学依赖的 Python 3.11 环境执行：

```bash
cd /home/ubuntu/ygwang/FindingBench
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate ygbench
python scripts/prepare_behavior_pilot.py --assets-root "$OMNIGIBSON_DATA_PATH"
```

脚本只从选定场景中发现官方 `*_task_carrying_in_groceries_0_*_template.json`，生成 `scenarios/behavior_groceries_pilot/scenario.yaml`。找不到就报错，不会换场景、重新采样或伪造实例。运行加载时再次检查模板存在及已安装 BDDL 与审计原文一致。实例 ID 来自实际文件名，不提前编造。

之后可用上面的 Python API 接模型；人工调试可执行 `python scripts/serve_play.py --scenario scenarios/behavior_groceries_pilot/scenario.yaml`，界面已支持 VIEW。

## 地面通行

官方场景中固定的 `floors` 和 `paver` 是可通行支撑面，不因碰撞体厚度高于 z=0 就禁止 MOVE / TURN。保留其物理碰撞体；墙、家具、台阶及未固定的散放砖块继续参与导航障碍检查。

## 验证边界

`ygbench` 环境及完整官方资产已安装，官方采购物品实例已成功加载。CPU 全量回归为 1000 passed、17 skipped；地面通行修改后的后端定向回归为 16 passed。跳过项为旧生成资产 / 外部工具检查，不算通过。

真实 GPU 验收已通过：完整官方场景加载、头顶安装位置/朝向、1024×512 全景、六个 VIEW 方向（机器人位姿/关节/物理时间不变）、旧帧拒绝、OBSERVE 拒绝、复位及官方 REPORT_DONE 判定。证据和图像见 `runs/ygbench_setup/pilot_smoke/`。人工网页 HTTP 实测已通过：取图、VIEW、MOVE(5 cm)、TURN(5°)、复位；记录见 `runs/ygbench_setup/human_play/`。尚未完成整条采购任务的抓放轨迹，也未接入大模型。

## Official task switch — 2026-10-22

The instrumented evaluator now accepts `--scenario PATH` and defaults to
`scenarios/behavior_groceries_pilot/scenario.yaml`; it no longer hardcodes the
custom knife-fetch scenario. This choice propagates through serial model launches
and the simulator worker. Public task instructions come from the loaded official
BehaviorTask goal descriptions. Camera resolution is now 2048x1024, consistent
with the current harness; this does not change official task initial states.

Prepared definition-0/instance-0 bindings: store_honey in Rs_int,
boxing_books_up_for_storage in house_double_floor_upper, and carrying_in_groceries
in house_double_floor_lower. Selection and original template/definition provenance
are recorded in `runs/official_behavior_task_switch_20261022/selection.json`.
Initialization checks run serially, without a model or an oracle trajectory;
`status.json` and each task's `report.json` record results. A successful check means
scene loading and official false-completion rejection worked, not that the full
task is solvable by this robot within the 30-step evaluation budget.

Example evaluation entry point after initialization validation:
```bash
bash scripts/run_instrumented_demo.sh --scenario scenarios/behavior_store_honey/scenario.yaml --model internvl3.5-4b --max-steps 30 --out runs/official_store_honey_internvl_001
```

## Human oracle session — 2026-10-24

From the repository root:

```bash
bash scripts/play_behavior.sh --list
bash scripts/play_behavior.sh
# Or choose directly:
bash scripts/play_behavior.sh --task store_honey
```

The launcher activates `ygbench`, presents three initialization-checked official
pilots, and starts the browser interface at http://127.0.0.1:8090. Forward port
8090 with VS Code, or on your own computer run:

```bash
ssh -N -L 8090:127.0.0.1:8090 YOUR_SSH_HOST
```

Open the URL on that computer. Wait for initialization. Select a skill, enter its
parameters, or click the panorama/current VIEW image for object skills. Positive
TURN means left, negative means right. VIEW changes gaze only, never body heading.
REPORT_DONE uses the existing task success judge. Reset starts a new recorded
attempt. Ctrl+C in the server terminal exits. Only one simulation should run at a
time. The default scenario's existing action budget is unchanged.

Other prepared tasks can be opened with `--scenario PATH`. `--port`, `--seed`,
and `--output NEW_DIRECTORY` are available. No click assistance is enabled by the
human launcher. Do not mix debug-assisted traces with unassisted human results.

Each session has its own `runs/human_oracle/TIMESTAMP_ID/` directory containing:

- `manifest.json`: seed, scenario/source fingerprints, assistance flag and status.
- `scenario.yaml`: exact scenario copy.
- `episode_NNN/events.jsonl`: initial observation, each submitted action and its
  result, or execution error; reset creates a separate episode.
- `episode_NNN/*.png`: observed panorama and any supplementary VIEW images.
- `private_trace.jsonl`: evaluator-side trace; not shown in the UI.

Logs are flushed after each action. The files preserve unsuccessful attempts as
well as successful ones. A recorded success is evidence for that task instance,
not an optimality claim. Action/frame records support later replay implementation;
automatic cross-process replay is NOT yet validated (`replay_verified=false`).
The three pilots have passed initialization tests, not complete human solutions.

## High-level task inventory contract

Target: approximately 1,000 DISTINCT high-level household task meanings, not
1,000 episodes obtained by changing rooms, objects, or seeds. Prefer BEHAVIOR
semantic seeds and assets. A scene/initial state need not reproduce the official
activity unchanged; adapted instances must have separate provenance and a checked
success predicate, and must not be called native official instances.

`build/behavior/high_level_task_review.csv` is an unapproved review queue of all
1,016 local official definitions. 191 pass the coarse rigid audit; the others
require adaptation or replacement. No row is approved merely by being generated.
`source_title` is a source label, not a final public instruction.

Each accepted task needs a natural high-level English instruction, explicit
machine-checkable completion conditions, valid initial states, verified coverage
by the rigid two-finger skill interface, and human-oracle evidence. Wording must
make clear whether any qualifying object/destination is accepted or a particular
one is required, without revealing hidden locations or prescribing action steps.
Multiple valid solutions are allowed; success ambiguity and search difficulty
are separate. Repetition alone is not a reason to exclude or label a task hard.
Review semantic duplicates explicitly; simple noun/room substitutions do not
establish distinct high-level task meanings. Keep discovery and verification
examples separate from the final evaluation split.

## Human observation revision — 2026-10-24

The three official human pilots now use a rigid camera mount offset of
[0.30, 0.00, 0.15] m relative to the configured mount basis, rather than
[0.00, 0.00, 0.15]. This is a forward-offset trial to reduce self-occlusion.
Panorama stays 2048x1024. VIEW now directly renders a co-located virtual pinhole
camera at 1024x768 with 90-degree horizontal FOV, rather than upsampling a crop
of the panorama. The renderer camera has no physical body. RGB, range and
instance IDs are rendered together for point grounding. Physics is not advanced.
The panorama stays the primary input; detail appears only after VIEW and is
removed by the next non-VIEW action. Legacy/fake backends without native VIEW
capture retain explicit panorama-reprojection fallback.

In the human UI, drag the panorama with no point skill selected. During dragging,
a low-resolution browser preview reprojects only the already supplied panorama;
it receives no new simulator images. Release submits exactly one counted VIEW
and obtains a new directly rendered pinhole image. A Look down button
submits VIEW(yaw=0,pitch=-70); Look forward submits VIEW(0,0). Numeric controls
remain available. Positive pitch looks up. VIEW does not turn the body.

The body reference is now generated from the loaded robot's visual-mesh vertices
and measured camera pose, in the robot base frame. It contains top/side
orthographic projections, executor footprint and turn sweep. It excludes scene
objects and routes. Each mesh projection uses its convex hull; this is not a
photorealistic rendering or exact collision silhouette. Skipped meshes are
reported in body_reference.json. The card represents initial joint posture,
not a continuously updated body shape or a feasibility oracle.

No scene furniture, collision rules, task goals or VLM evaluation runs are
changed by this revision. Scene navigability remains unverified; changing
observation alone does not establish that the narrow kitchen route is feasible.

## Derived honey scene: wider living-room / kitchen passage

Run `bash scripts/play_behavior.sh --task store_honey_open_route` (first menu entry).
Both honey launch aliases now use the clearance edit. The upstream source file remains unchanged. This is a declared derived
BEHAVIOR instance, not an unmodified official instance. Five background chairs
and one breakfast table are removed before scene loading; task-bound honey,
countertop, cabinet, floor, agent spawn, predicates and success conditions are
preserved. R1Pro, MOVE/TURN collision geometry and manipulation skills are unchanged.
This isolates scene clearance from a robot/platform change.

Rebuild with `PYTHONPATH=src python scripts/prepare_open_route.py` in `ygbench`.
The scenario points to `build/behavior/store_honey_open_route/manifest.json`;
loading checks original/derived SHA-256 and verifies that the only JSON changes
are declared removals of objects not bound to the task. Human recordings archive
this provenance manifest and its hash. Do not combine original and derived scene
scores without labeling the variant.

Private authoring validation: `PYTHONPATH=src python scripts/environment/check_open_route.py`.
Its actual collision-geometry route is replayed through public MOVE/TURN skills;
`runs/store_honey_open_route_validation/report.json` states the outcome.
The witness and scene metadata are not supplied in player/model observations.
A successful kitchen route certifies that route only, not complete honey-task
solvability or comfortable navigation for all users. Human play remains the next
check before any VLM evaluation. Future scene selection should reject or edit
incidental furniture bottlenecks when these dominate the intended understanding
task; keep meaningful room/layout structure and task objects.


### Local clearance revision (2026-10-24)

`scene.clearance_manifest` is an evaluator-side authoring patch. It deforms only
wall mesh vertices, with identical world-space transforms for visible meshes and
native colliders, inside fixed outer wall bounds. Furniture and robot scales are
unchanged. Declared translations and removal of non-task objects are recorded;
task-bound objects cannot be removed. A complete state snapshot is restored
across the USD/PhysX update so the robot spawn and non-edited object states do
not revert to asset defaults. Player resets retain the authored layout.

The honey kitchen opening is about 1.4 m wide. Its intact, initially empty target
cabinet is relocated by (-0.17, -2.9, 0) m into the cleared living-room area.
This changes the cabinet location; it is NOT an unchanged official task instance.
The native goal binding and predicates remain. Eight public MOVE/TURN calls
successfully reached the kitchen, and reset was checked in
`runs/clearance_honey_validation/report.json`. This is a navigation witness,
not proof of full task completion or of all rooms being accessible.

Other layouts are tracked in `build/behavior/clearance_audit.json`. Proposal
files alone are NOT evidence of passable scenes. Do not reuse original layout
distance/difficulty measurements or mix original and edited scores without
recomputation and variant labels. Recordings archive the clearance manifest and
its hash. No private portal coordinates or witness actions are supplied to the
player or evaluated policy.


Upper-floor layout: four internal openings widened to approximately 1.4 m.
Four non-task door assemblies and one obstructing outlet are removed (doorless
openings); an intact empty bookcase is shifted 1.5 m along its wall. The stairs
remain intact. All four local crossings and 90-degree turns were replayed through
public skills, followed by reset; see `runs/clearance_upper_validation/report.json`.
These local witnesses start from evaluator-selected approach poses, so they do
not certify a full route from the task spawn or full manipulation success.


Lower-floor layout: the two narrow internal openings were locally widened to
approximately 1.4 m, including their lintels, and their non-task door assemblies
removed. The washstand cabinet, sink, mirror and light move together by (-1,0,0)
m. Both local crossings / 90-degree turns and reset passed in
`runs/clearance_lower_validation/report.json`; the initial official goal remains
unsatisfied. Wide existing exterior doors retain their normal behavior.

Clearance-enabled legacy scenarios also use the same per-native-collider
navigation bounds as the official pilots. A whole-wall object AABB can span an
empty doorway; it must not replace the edited wall's actual collider pieces.
The 0.8 m square executor footprint, 0.03 m obstacle margin and swept sampling
remain unchanged. This does not omit the wall or ignore its collisions.
Legacy scene patches run after task-object import and before placement sampling;
private interaction anchors and explicit receptacle-relative authoring positions
follow declared receptacle translations.


Beechwood main-task area: the kitchen/hall opening is 1.2 m wide, constrained by
adjacent partitions. The two native cabinets and their shared worktop are moved
as an intact assembly by (+0.5,-1.4,0) m, replacing the old unbound central island
(three cabinets, worktop and burner removed). Task-bound cabinets and worktop
remain, with their original dimensions and contents/placement rules. The knife
scenario initialized and passed the local crossing, turn and reset checks:
`runs/clearance_beechwood_validation/report.json`. The same layout patch is
configured in all 11 Beechwood scenarios; their individual task placement / boot
paths have NOT all been replayed. Three Level-1 starts move 0.19 m left to center
in the opening. All 11 native-layout start footprints and full-turn discs pass an
offline check before task-specific spawned-object placement.

All 15 prepared scenario configurations now reference a clearance manifest.
This is NOT a claim that the entire BEHAVIOR-1K dataset, every room, or every task
is accessible. Unrelated bedroom/bathroom/closet openings, especially in
Beechwood, remain native and need further authoring if future tasks use them.
The eight tested openings are main-task-area witnesses, not whole-house proofs.
The original source assets remain untouched. No VLM evaluation was started.


### Asset cleanup (2026-10-25)

Superseded clearance proposal JSONs, candidate YAMLs and the old honey wall-primitive
export were deleted. Use the four formal layout manifests referenced by current
scenario files. Original BEHAVIOR model assets and source templates remain required
runtime dependencies: these are patch-based derived scenes, not standalone baked
asset bundles. Validation and human-play records are preserved; historical candidate
paths in those records are no longer launch targets.

Cleanup was subsequently cancelled. Seven proposal/candidate configuration files
were reconstructed from the retained formal versions; they are not byte-identical
recoveries of historical drafts. The deleted `build/behavior/honey_wall_prims.json`
export has not been recovered. No further deletion is authorized.

## MOVE_TO human pilot — 2026-10-26

`MOVE_TO` accepts `point: {frame_id, x, y}` on the current panorama or current
VIEW supplement. Select a visible fixed floor/paver position, not an object.
The executor uses private depth/segmentation to ground the point, then searches
an 8-connected 10 cm BEV grid. Obstacles come from the same live collision
geometry as MOVE; the fixed-heading .8 m square footprint and .03 m margin
remain. Edges are checked at <=5 cm and diagonal corner cutting is rejected.
The exact goal is connected by a checked segment, never snapped to another goal.

After a route is found the robot is transferred directly to that endpoint,
without a trajectory controller or simulated travel. Heading is preserved;
attached objects translate with the robot. This intentionally changes navigation
semantics from incremental MOVE. Each request costs one decision step; planned
route and distance are saved under evaluator-private `navigation_private`.
Planned distance is NOT measured physical travel. The map/path are not public.
VIEW and manipulation actions remain available. Public MOVE is removed. TURN accepts only direction=left/right/back (90/-90/180 degrees); angle_deg is rejected. Rotation uses the existing swept footprint check, costs one step, and returns a fresh panorama on success. VIEW does not rotate the body.

Scope: same-level base-footprint feasibility, not whole-body/held-object swept
collision certification or stair navigation. Floor support currently uses the
union of same-level fixed floor AABB envelopes; irregular floors/holes need
separate scene validation. Grid-search failure means no route found at this
resolution, not proof of continuous-space impossibility. Search-budget exhaustion
has distinct feedback. These limitations must be disclosed before batch use.

Validation: CPU tests cover obstacle detours, disconnected maps, blocked/exterior
endpoints, search limits, exact endpoint/heading, zero movement on rejection,
and frame/floor validation. Saved honey collider boxes admit a spawn-to-kitchen
route (36 points, about .47 s), using a synthetic bounded support envelope for
that offline check. Live floor grounding/arrival has NOT yet been GPU-validated.
Existing human session was not restarted; restart the human launcher to load
this code. Do not combine previous MOVE-only trials with this navigation pilot.

### Floor height tolerance — 2026-10-27

MOVE_TO accepts a selected fixed floor surface when its height differs from the
current robot base height by strictly less than 0.30 m. Floor support selection
uses the same threshold for both ground and object navigation. Exactly 0.30 m
is rejected. This replaces the old 0.15 m gate; furniture/wall collision tests
are unchanged. Oracle arrival retains the current base z: this is navigation
eligibility, not validated physical step climbing.
