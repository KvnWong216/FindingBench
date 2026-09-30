# RummageBench 交接文档

> 历史记录提示（2026-09-30）：下文的“全部验收通过”、接口和测试计数仅对应当时版本，
> 不代表当前 AGENT 视觉协议或生成布局已经通过验收。当前待验证项与实验顺序见
> `EXPERIMENT_PLAN.md`；不能以旧 ORACLE 轨迹替代当前 AGENT 的端到端验证。

> 交接日期：2026-09-27（v2 更新）
> 状态：MVP + v2 具身感知扩展全部验收通过（单测 35/35，集成 11/11）
> 仓库：GitHub `KvnWong216/FindingBench`（本目录即其工作副本）
> 主机：10.20.37.124（8x RTX 5880 Ada 49G，/data1 约 8.8T 可用）

## 0. v2 具身感知扩展（2026-09-27）

按设计对话定稿的 prompt 实现，核心不变量仍是：**只评估具身感知的交互推理，
不评估操作控制**（Action Expert 完美执行器）。

新增模块：

| 模块 | 内容 |
|---|---|
| `object_interface/` | Rigid/Articulated/Receptacle 适配器，实体自己提议语义合法技能（`available_skills(robot, state)`） |
| `feasibility/` | `ik_solver.py`（IKSolver 协议 + reach_radius/z带 可达性代理，TRAC-IK 可插拔）、`collision.py`（交互点 vs AABB + 允许碰撞矩阵：指<->目标允许，关闭异物体积禁止） |
| `core/skill_grounder.py` | **EAGE（Embodied Action Grounding Engine）**：A_t = Ground(Robot, Object, WorldState_t)，每步重生成可用动作空间 |
| `environment/env.py` | JSON 门面：`env.reset()` / `env.step({"type":"OPEN","target":...})` |
| `robots/robot.py` | RobotEmbodiment 能力参数（reach_radius、z_min/z_max、hand_capacity），进 scenario YAML |
| skills | 新增 PLACE / CLOSE（5 技能集：NAV/OPEN/CLOSE/GRASP/PLACE） |

行为变化：

- Observation 增 `available_skills`（动态动作空间）
- FailureReason 增 `UNREACHABLE` / `COLLISION` / `INVALID_STATE`（结构化失败、非终态、消耗一步）
- metrics 增 `interaction_efficiency` / `feasibility_failures` / `reasoning_failures`
- scenario.yaml：5 技能、5 锚点、robot 能力参数；scripted_success 每柜先 NAV 再 OPEN（8 步）

### 设计对话结论（论文定位，来自 ChatGPT 设计讨论定稿）

- **三支柱创新**：① 动态具身条件化动作空间（动作从世界状态"涌现"，非固定动作表）；② 物理可行性作为一等评估维度（UNREACHABLE/COLLISION/INVALID_STATE 来自 IK+碰撞+状态校验器，失败归因是机械可信的）；③ 推理与执行解耦的因子化评估（回应 ForageBench：现有基准把高层决策质量和底层执行噪声搅在一起）。
- **定位一句话**：Existing benchmarks evaluate embodied agents through successful execution, making it difficult to distinguish whether failures originate from reasoning or embodiment limitations. We introduce an embodiment-grounded diagnostic environment that explicitly derives executable interaction spaces from robot capabilities and object states, enabling controlled evaluation of interactive reasoning.
- **不要打的点**：不要声称"首次支持 open/pick/place"（RoboCasa/BEHAVIOR-1K 都有）；不要比资产规模。技能本身不是创新，**grounding 机制**才是。
- **核心实验**：同一场景同一指令，只换机器人能力参数（reach/高度带）→ 同一观测下可用动作图不同。智能体必须理解"我的能力 ≠ 世界的能力"。已落地为 `tests/unit/test_embodiment.py`（同任务不同本体 → OPEN(cabinet_B)/OPEN(drawer_A) 可用性翻转）。
- **难度阶梯**：L0 已知位置 / L1 未知容器 / L2 干扰物 / L3 本体约束 / L4 遮挡重排。当前场景覆盖 L0-L2；L3 已有机制与单测支撑（多本体场景变体待做）；L4 未来。

## 1. 这是什么

RummageBench：基于 BEHAVIOR-1K + OmniGibson 的长时程具身物体搜索基准。
智能体只做语义决策（NAV / OPEN / CLOSE / GRASP / PLACE），后端用符号/特权机制
保证语义后果可靠实现；基准以"语义规划步数"计时，不考核导航、IK、抓取控制等
底层执行。

**验收结论（MVP，全部通过）**：

| 项 | 结果 |
|---|---|
| 场景编译（knife_search_001） | BUILD OK，谓词全部满足 |
| scripted（oracle 基线） | SUCCESS，5 步 |
| wrong_object | FAIL_WRONG_TARGET，5 步 |
| timeout | FAIL_MAX_STEPS，17 步 |
| unsafe | FAIL_UNSAFE_ACTION，2 步 |
| 复位确定性 | 语义严格复原，位姿 <2cm |
| 单元测试（无模拟器） | 24/24（v2 后 35/35） |
| 集成测试（真实模拟器） | 11/11 |
| MCP 验收 | 与 Python API 结果一致 |

## 2. 目录与环境位置

| 内容 | 路径 |
|---|---|
| 本项目（GitHub 工作副本） | `/data1/ygwang/codes/FindingBench` |
| 旧开发目录（v2 之前的开发副本，内容已并入新仓库） | `/data1/ygwang/codes/Find-Bench` |
| BEHAVIOR-1K v3.9.3（外部依赖，勿改源码） | `/data1/ygwang/codes/BEHAVIOR-1K`（OmniGibson/bddl3 以 editable 方式装入环境） |
| BEHAVIOR 数据集（31G，含密钥） | `/data1/ygwang/codes/BEHAVIOR-1K/datasets` |
| conda 环境 `behavior`（Python 3.11 + Isaac Sim 5.1 + OmniGibson 3.9.3 + torch 2.7.0 cu128） | `/data1/ygwang/miniconda3/envs/behavior` |

激活环境（任意用户）：

```bash
source /data1/ygwang/miniconda3/etc/profile.d/conda.sh
conda activate /data1/ygwang/miniconda3/envs/behavior   # 按路径激活
```

目录权限已对全体用户开放（a+rwX）；数据集与密钥仅限 OmniGibson 内部使用
（BEHAVIOR 数据许可：仅限非商业学术研究，不得提取/转发）。

## 3. 快速上手

```bash
cd /data1/ygwang/codes/FindingBench
export CUDA_VISIBLE_DEVICES=5        # 挑一块空闲 GPU（nvidia-smi 查看）
/data1/ygwang/miniconda3/envs/behavior/bin/pip install -e . --no-deps   # 首次
export PYTHONPATH=/data1/ygwang/codes/FindingBench/src

# 一键全量验收（单进程：构建场景 + 4 条轨迹 + 复位确定性，约 15-25 分钟）
python scripts/run_all.py            # 预期输出 ACCEPTANCE OK

# 单条轨迹
python -m rummagebench.cli run --scenario knife_search_001 --agent scripted
#   --agent 可选: scripted | wrong_object | timeout | unsafe

# 场景检查 / 编译 / MCP
python -m rummagebench.cli inspect --scene Beechwood_0_int
python -m rummagebench.cli build  --scenario scenarios/knife_search_001/scenario.yaml
python -m rummagebench.cli serve-mcp                          # stdio 服务
python scripts/mcp_acceptance.py                              # MCP 验收（GPU 写死为 4，可改）

# 测试
python -m pytest tests/unit -q                                # 不需要模拟器
python -m pytest tests/integration -m sim -q --tb=short       # 需要 GPU + 数据集
```

## 4. 架构速览（详细规则见 README.md）

```
scenarios/*.yaml          场景=数据（"游戏关卡"），无场景专属 Python
src/rummagebench/
  core/                   Action/Observation/StepResult/EpisodeStatus/ScenarioSpec
                          BenchmarkSession：解析→语义→目标→安全→执行→后验→观测
  sim/base.py             抽象 SimBackend（mock/其他模拟器可替换）
  sim/omnigibson/         唯一允许 import omnigibson 的地方
  skills/                 NAV/OPEN/GRASP（只执行，不判定成败）
  validation/             semantic / target / safety
  grounding/              oracle 实体定位（pixel 定位是预留桩）
  authoring/              场景检查/编译器/构建器
  evaluation/             episode 循环、JSONL 轨迹、指标
  agents/                 scripted/wrong_object/timeout/unsafe（序列来自 YAML）
  adapters/               python_api、mcp_server（MCP 零基准逻辑）
tests/unit                无模拟器（FakeBackend 替身）
tests/integration         真实模拟器（-m sim）
```

核心数据流：`Agent.act(obs) -> Action -> 校验管线 -> Skill.execute(backend) ->
StepResult(+JSONL 事件) -> Observation`。成败/步数/安全由 Session 判定，
技能与 MCP 一律不判（Rule 4/5）。

## 5. 运行前必读的坑（都实际踩过）

1. **GPU 选择**：用 `CUDA_VISIBLE_DEVICES=N` 隔离整卡。只设
   `OMNIGIBSON_GPU_ID` 没用——torch 张量操作会全挤到 GPU 0，多实例并发时
   互相顶爆（CUDA launch timeout → Abort）。
2. **warp 版本锁死 1.12.0**：1.17 缺 `wp.types.array`，1.7 段错误。环境里已装对，别动。
3. **Kit 启动慢且偶发段错误**：首次启动（着色器编译）约 5-10 分钟，约 20%
   概率在启动期 Segfault/Abort。重试即可，不是代码问题。脚本全部
   `os._exit()` 干净退出以绕过 Kit 退出清理段错误（良性）。
4. **pkill 自匹配**：远程 `pkill -f xxx` 会把自己所在的 shell 杀掉（命令行
   含关键字），用 `pkill -f "x""y"` 断词写法。
5. **机器人注册名小写**：`model: r1pro`（大写 R1Pro 会 assert 失败）。
6. **Inside 放置是概率采样**：builder 已做 5 次重试 + 直接位姿兜底 + 谓词
   验证，构建失败说明真放不进去。
7. **复位语义**：保证语义状态严格一致 + 底盘位姿 ~2cm 内；RTX 渲染不是
   像素级确定的，不要用像素对比做门禁。
8. **抓取前置**：隔着关闭的柜门抓取会失败（关节拉断）；后端 GRASP 会先
   把底盘传送到目标旁再建关节。测试脚本要先 OPEN 容器。
9. **单 ssh 只带一个后台任务**：`setsid nohup ... &` 一条 ssh 连接里只放
   一个，第二个会随连接退出被杀。
10. **huggingface 被墙**：下载用 `HF_ENDPOINT=https://hf-mirror.com`；
    pip 慢时用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；数据集解密
    key 在 `datasets/omnigibson.key`（storage.googleapis.com 直连可用）。
11. **NAV 传送物理发散（v2 修）**：timeout 轨迹反复传送后 PhysX 报
    `Illegal BroadPhaseUpdateData`（全套件约 240 次），机器人链接四元数
    变 NaN。这是 OmniGibson/Isaac Sim 5.1 对运行中动态关节体反复
    `set_position_orientation` 的固有不稳定，**不要在传送前后碰物理**：
    - `keep_still()` 会清零关节 effort 目标 → 位置控制的悬挂直接塌地，
      scripted/reset 全挂（实测两次）；
    - 传送前清零根部线/角速度同样引发非确定性塌落（实测）。
    最终修复（保持基线物理路径不动）：`robot_pose()` 读取包 try/except +
    NaN 检查，异常/非有限时回退到最后命令的锚点位姿（完美执行器假设，
    物理发散是执行噪声，不污染语义状态、不终止 episode）；
    `teleport_robot()` 在 Robot 级 setter 因内部读 EEF 断言崩溃时，降级
    到 `EntityPrim.set_position_orientation`（只写根位姿、不读链接）。
12. **GitHub 推送**：服务器无凭据，从本机推；`~/.gitconfig` 里给 GitHub
    配的代理 127.0.0.1:8080 已失效，用
    `git -c http.https://github.com/.proxy="" push`（SSH key 认证为
    KvnWong216）或删掉该段配置。

## 6. 关键产物位置

```
FindingBench/runs/setup_report.json                       Phase 1 环境验证
FindingBench/runs/acceptance_report.json                  总验收报告
FindingBench/runs/acceptance_knife_search_001_*/          四条轨迹（events.jsonl+summary+逐步截图）
FindingBench/runs/anchors_gpu2.json                       Phase 2 锚点实测数据
FindingBench/build/scenarios/knife_search_001/            编译产物（snapshot/preview/报告）
FindingBench/logs/                                        安装/运行日志
```

## 7. 已知限制与下一步建议

- pixel grounding 未实现（`grounding/pixel.py` 桩），当前是 oracle 实体名定位。
- 安全校验是规则级（固定家具/禁抓类别），动作 schema 无 interaction_region，
  不评估抓取部位安全（prompt §14 的预留扩展点）。
- Kit 启动成本高：批量评测建议每进程跑完整套（参考 run_all.py 的单进程模式）。
- 下一步方向：接入 VLM 智能体（走 MCP 或 python_api）、扩充第二场景
  （只需写新 YAML + pick_anchors.py 采锚点）、像素定位（需要 GT 分割输出，
  在 sim/omnigibson/observation.py 扩展）。

## 8. 环境复现（万一要重装）

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git --branch v3.9.3 --depth 1
cd BEHAVIOR-1K
HF_ENDPOINT=https://hf-mirror.com ./setup.sh --new-env behavior --omnigibson --bddl --dataset \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
# setup.sh 结束后补三件事：
pip install warp-lang==1.12.0 pillow==11.3.0      # warp 必须手动装/锁版本
pip install -e /data1/ygwang/codes/FindingBench --no-deps
pip install "mcp<2" pytest                         # mcp 必须 <2（2.x 改了 API）
```
