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
