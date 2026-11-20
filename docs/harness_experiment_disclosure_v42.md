> Current visualization protocol (same harness version label): every completed step is exported to media/step_NNN with full text, ordinary/attention video clips, and separate maps for each generated skill/parameter value token. media/live.html updates after each export; full episode videos are assembled at termination. Removed first-five-steps-only publishing. Attention instrumentation retains the individual head with maximum summed probability on current image tokens across all language layers for each predictive query; no head/layer/token averaging. Primary overlay uses the last skill-value token; parameter-value tokens have separate PNG/NPY files. Layer/head/token identity, raw image probability mass, and display scale are exported. One scale per query shared across images; no p99 rescaling or Gaussian smoothing. This is exploratory attention inspection, not causal importance or validated grounding. Inspired by VL-InterpreT (https://github.com/meng-du/VL-InterpreT/) individual token/head inspection and LocalizationHeads (https://github.com/seilk/LocalizationHeads) visual-mass criterion. NOT a replication of LocalizationHeads: no elbow/entropy/bottom-row filtering or RefCOCO calibration; no averaging of selected heads. Queries predict the named output token (one-token shift). Only actual generation is instrumented; no extra inference or action correction; attention/exo remain evaluator-private. GQA extraction avoids repeating full-length keys. Explicit-QK/SDPA invariance and separate-token export checks passed; 32 related tests passed.

> 反馈全面审计整改（版本标签仍v4.7，当前保持停止）：已将成功MOVE/TURN/VIEW/GRASP/PLACE/OPEN/CLOSE/REPORT_DONE反馈具体化为执行后的效果；失败区分对象不支持技能/已有开闭状态/持物容量满/未持物/放置目标不支持。仅对已经选中的失败对象读取类别和状态，不暴露实例ID/坐标/目标身份。校验失败从Pydantic实际错误提取缺失字段/类型/范围/多余参数，不补参数；深度无效/像素无对象/无法绑定对象各自解释；OPEN打开后侵入底盘与操作姿态碰撞区分；任务安全规则拒绝不冒充物理碰撞；无交互姿态定义不误称距离够不到；错误报告完成明确episode已结束。无预检查、自动纠错、重试或下一步建议，游戏判定不变。83项相关回归通过（随后补充VIEW缺失和无交互姿态解释并重测）。

> 抓取反馈整改（保持v4.7版本标签）：按用户要求区分UNREACHABLE=当前姿态无法够到有效抓取姿态、COLLISION=候选抓取姿态碰撞/不安全、SKILL_NOT_APPLICABLE=选中对象不能抓或当前状态不允许。仅在动作已拒绝且对象已绑定后读取该对象category/fixed_base/graspable；固定或不可抓才说cannot be picked up，其余保留状态相关表述，避免把持物冲突误说物体永远不能抓。只反馈类别、不输出实例ID/位置/目标答案、不推荐后退或新点、不改判定和动作。57项回归通过。当前仍停止，未自动开跑。

> 2026-10-20 解码整改：用户要求Qwen官方推荐设置。serve_local_vlm对qwen3-vl-4b启用官方Instruct评测采样：temperature=.7,top_p=.8,top_k=20,repetition_penalty=1,presence_penalty=1.5；默认解码种子3407（与场景seed0分开），每次请求重置，可显式decoding_seed覆盖。presence penalty由HF LogitsProcessor实现，仅对本次生成中已出现token扣分一次，不作用于历史文本，不检查/修改技能。instrumented runner输出预算改为32768；其他调用者仍按各自请求的预算执行并记录，不能误称全部一致。health和每次响应披露实际decoding，源码指纹含qwen_decoding.py。官方来源https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction。仍为HF+NF4，不是官方vLLM运行环境的逐位复现。任务提示保持v4.7，评估保持停止，未自动重启。

> 2026-10-20 最新 v4.5-multiturn：按用户要求使用明确的 user/assistant 多轮消息，完整保存所有公开观测文字和模型原始输出，不摘要、不按步数/字符裁剪。规则在首条user消息（兼容不同模型模板）；每轮user公开观测包含上一实际动作与反馈，assistant原样保留返回文本。历史图片不回灌，当前图片只附最后user消息，VIEW仍按次补充。两套服务均优先消费messages；prompt字段仅为旧报告兼容的审计视图，不能当作实际模型输入。MiniCPM取消max_inp_length=32768的隐式截断，容量错误显式失败。所有模型共享消息协议，但未完成各模型GPU生成实测。42项测试通过；不启动评估，旧v4.3保持人工中断。

> 2026-10-20 最新：按用户要求停止 v4.3 Qwen 实验及视频/HTML发布进程，run_state=interrupted，不自动重启。当前 HumanCLAWPolicy 升为 v4.4-full-history：取消10步窗口和所有字符截断；完整保留模型原始返回（含格式错误、显式返回的推理）、结构化字段、实际动作和公开反馈。历史只来自模型返回与公开协议，不读取私有事件、oracle、exo或attention。图像仍是当前全景+当次VIEW补图，不回灌历史图片；这是既有图像协议，不是全多模态聊天回放。无静默上下文裁剪；超出实际模型容量的推理错误应作为基础设施限制披露，不能算任务失败。旧 strict 策略仍为60步动作反馈窗口，不能冒充本次完整历史配置。此次只整改，不启动新实验。

# FindingBench：HumanCLAW 对照、实验协议与失败归因披露

本文对应 `T01_knife_fetch__L1_S1` 单任务试验。历史基线为 v4.2；
整改版为 `humanclaw-panorama-reason-feedback-v4.3`。两者必须分开计分。
原八模型队列按用户要求停止；后续明确授权的单模型记录实验见 §12。

## 1. 原始证据与阅读入口

- [逐模型逐步报告](../runs/harness_v42_review/index.html)：HTML 可直接在浏览器打开。
- [已完成 Qwen 参考轨迹](../runs/harness_v42_review/qwen3-vl-4b-reference.html)。
- [当前八模型队列原始记录](../runs/harness_v42_all_models_single_task/suite.json)。
- [先前独立 Qwen 试跑](../runs/harness_v42_ondemand_pilot/suite.json)。
- [v4.2 关键源码、模型配置和场景快照](../runs/harness_v42_review/protocol_snapshot_v42/files_sha256.json)。
- 每个模型都有同名 `.html`、`.md`、`.json`。未运行的模型标为待运行，不编造轨迹。
- HTML 展示每一步全部已记录模型输出、结构化理由、输入提示、技能参数、原始执行事件、
  动作前后全景、可用时的补充针孔图。原始输出保留实际返回的 `<think>` 文本。
  **没有被模型输出或被服务端解码保留的内部计算过程无法恢复，不称作“完整内部 CoT”。**
- 原始日志 `model_calls.jsonl` 的 `planning_step=0` 对应执行日志第 1 步；
  第 k 次动作输入是 `step_{k-1}.png`，输出是 `step_k.png`。报告据此对齐，避免错位。

## 2. HumanCLAW 对照：哪些已借鉴，哪些刻意不同

参考[论文 v1](https://arxiv.org/html/2607.27180v1)的 §2.2、§4.3、附录 A/B，
以及[官方实现固定提交](https://github.com/Human-CLAW/HumanCLAW/tree/6323d938795afd27d66ab926216bf4ba7b003ee6)。
论文描述的框架采用结构化感知、短期规划、技能选择和 verifier；本文实现仅借鉴前面三个阶段。
它的 HSSD 人体找寻—导航—坐下任务与我们的机器人拿刀任务不同，分数不能横向等同。
论文首版与后续网站榜单也不能混成同一实验。

以下实现对照以固定版本源码为依据，具体文件：
[planner.py](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/src/humanclaw_bench/agent/planner.py)、
[prompt v4](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/src/humanclaw_bench/agent/prompts/v4.py)、
[verifier v3](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/src/humanclaw_bench/agent/verifiers/v3.py)。

| 项目 | HumanCLAW 固定版本 | FindingBench |
|---|---|---|
| 决策组织 | 感知描述、进度、mid-level goal、当前技能理由 | 采用同类字段，要求一个最终 JSON 动作 |
| 规划尺度 | 六动作 horizon | 六动作 horizon；不是一次强制执行六步 |
| 记忆 | 近期结构化文本 | 最近十步描述，并绑定真实动作及公开反馈 |
| 本体提示 | 蓝色人体，图中身体与脚部约定 | r1pro、自身局部坐标与相机标定、底座近似轮廓；不假定脚部可见 |
| 视觉 | ego 图片 | 常驻全景；成功 VIEW 后一次性额外针孔图 |
| verifier | 可检查、拒绝或更新技能提议 | **不采用**，用户要求不干预试错 |
| 失败重试与默认动作 | planner 存在多次尝试、解析及恢复路径 | 一次推理对应一次提交；不自动修补参数或选择替代动作 |
| 环境反馈 | 所核对 `_plan_skill_prompt` 明确丢弃 `env_feedback` | v4.2 四类通用反馈；v4.3 追加事后自然语言原因 |
| 运动生成 | 人体连续运动与半物理执行 | 基座扫掠检查后位姿更新；操作走 IK/碰撞可行性与技能实现 |
| 复现声明 | 原论文框架 | **不是原论文完整复现**，尤其不包含其 verifier |

“不采用 verifier”是实验定义的有意差异，不能把两种 harness 的结果直接用来衡量模型强弱。
本地模型规模、量化、任务、动作空间也不同。

归因方面，[HumanCLAW 附录 B](https://arxiv.org/html/2607.27180v1#A2)
采用确定性的离线规则，并依赖导航距离、真实 mesh 接触、目标语义像素和动作/自述日志。
这不是让另一个模型自由猜测根因。我们借鉴“可复现规则 + 分层证据”，但当前没有同等的
mesh 接触与每步目标可见性证据，不能照搬其完整 Find/Nav/Interact 根因标签或阈值。
报告中的“连续受阻”只描述日志事实，不自动解释成“模型没有意识到受阻”。

## 3. 实际任务、数据与成功定义

- 实际场景配置：[编译后的 scenario.yaml](../build/vla_difficulty_v1/T01_knife_fetch__L1_S1/scenario.yaml)。
  IDE 中 `scenarios/knife_search_001` 或 `barbecue_sauce_search_001` 并不是本轮运行入口。
- 场景：BEHAVIOR 资产 `Beechwood_0_int`。
- 指令原文：`Go to the kitchen island, pick up the table knife lying on it, and then report that you are done.`
- 目标：`target_knife`，类别 `table_knife`，资产型号 `eebpap`，置于 `countertop_tpuwys_4`。
  这些实例 ID 属于评分端信息，不出现在原始模型输入中。
- 初始位置 `[0.85, -7.7, 0.0053]` m；四元数 xyzw `[0, 0, 1, 0]`。
  这些世界位姿仅在评估与诊断端使用。
- 模型需要实际持有目标刀，再提交 `REPORT_DONE`。只看见、靠近或拿错物体不能成功。
- `REPORT_DONE` 是终止动作，目标不满足时会失败，不是暂停或“请求下一步”。
- 当前场景本身的预算是 60；运行参数覆盖实际执行会话为 **100**，日志 `effective_max_steps`
  记录真正生效的预算。
- 不因一次碰撞拒绝立即终止；无效动作、VIEW、报告动作都消耗一步。
- 这条任务是我们基于官方资产编译的指令、摆放和目标，不是未经修改的官方 BEHAVIOR-1K 全量任务。
  当前任务矩阵是四个基础任务的语言/空间变体；本轮只取其中一个。不能宣称已评估全部 BEHAVIOR-1K。

## 4. 图像与模型信息边界

### 4.1 相机

- r1pro，`head_link=zed_link`，`mount_frame_link=torso_link4`，配置偏移 `[0,0,0.15]` m。
- 相机跟随安装坐标系刚性变换，VIEW 不改变机器人、安装位置或本体朝向。
- 六个 512×512 立方体面合成 2048×1024 的 360°×180°等距柱状全景。
- VIEW 生成 640×480、水平 FOV 90°的针孔投影。这里是同一全景的投影，不是第二台可移动实体相机。
- 全景 x=.5 为参考前方、.25 左、.75 右、0/1 是后方接缝；y=0 天顶，.5 地平线，1 正下方。
- `yaw=180-360*x`，`pitch=90-180*y` 只适用于全景，不适用于针孔图。
- 全景坐标 x/y 总是 `[0,1]`；负 yaw 不意味着负图像 x。
- 相机参考朝向不必等于身体前方。提示要求依据本体—相机标定做区分。
- 每次输入都有全景。成功 VIEW 的返回观察额外有针孔图；下一次任何非 VIEW 动作后，
  即便动作失败，针孔图也消失。连续 VIEW 只更新当前补充图。
- 两图有各自 frame_id；图像点技能允许引用当时仍有效的任何一张图。旧补充图失效。
- 所有适配器接受一或两张图；由各模型原生 processor 决定视觉 patch/crop 数量。
  相同输入像素并不保证相同视觉 token 数。

### 4.2 模型可见

任务指令、当前帧号与图片、已用/最大步数、固定技能库、前一步公开反馈、十步文本记忆、
以下本体信息：

- robot_model、末端 link；
- 执行器使用的底座正方形半边长 0.4 m，不冒充精确机器人网格；
- 单手持物容量及执行器持物布尔值，不包含所持物体身份；
- 相机相对机体位置、身体前向与上向在相机坐标中的表达；
- 图像尺寸/投影说明、补充图角度及 FOV。

### 4.3 模型不可见

实例 ID、世界坐标、目标真值位置、深度、分割、场景图、完整物体状态、
动态可行动作列表、IK 候选、碰撞 AABB、oracle 路线、评分真值与 exo 摄像机。
深度和分割用于执行器把像素绑定到对象，不作为模型输入。

**v4.3 唯一新增信息边界**：失败后可在自然语言里披露执行器检查到的障碍物类别；
例如 cabinet，仍不暴露实例 ID、位置、完整障碍列表或绕行路线。这是额外传感反馈，
有可能帮助模型，因此必须明确作为独立协议比较。

## 5. 技能契约与执行

| 技能 | 参数 | 语义 |
|---|---|---|
| MOVE | `distance_cm`，绝对值 5–100 | 正前进、负后退，沿身体当前朝向 |
| TURN | `angle_deg`，绝对值 5–180 | 正左转、负右转，原地旋转身体 |
| VIEW | `yaw∈[-180,180]`；可省略 pitch，默认 0，范围 `[-90,90]` | 绝对取景角，正 yaw 左、正 pitch 上 |
| OPEN/CLOSE/GRASP/PLACE | `point:{frame_id,x,y}` | 当前图上的归一化像素点 |
| REPORT_DONE | 无 | 检查完整任务目标并终止 |

- 不存在 OBSERVE。VIEW 不会转动 MOVE 的运动方向。
- 交互点先通过私有分割/深度绑定选中物体，再调用实体技能。
  当前实现随后生成该物体的规范交互候选；并不是要求机械臂精确到达所点击表面点。
- 先进行 schema、帧有效性、点绑定和技能适用性判定，再执行安全/可行性流程。
- MOVE/TURN：0.8×0.8 m 旋转正方形轮廓，3 cm margin，平动采样间距不大于 5 cm，
  转动间距不大于 5°；与导航障碍 AABB 做分离轴测试。
- 单个采样点被拒绝则整次运动不执行，不作部分前进。不应把这种拒绝称为已经发生的物理撞击。
- 基座障碍检查涉及相应高度带；本任务的传统几何路径使用实体 AABB。
  官方任务专用的地面/建筑分解分支不能自动视为已经覆盖这个编译任务。
- 操作可行性配置为 `pinocchio` / `endpoint`：检查离散交互候选的 IK 与碰撞；
  不是仅用“距离小于一米”的半径代理。完整连续空间不可达性也没有被证明。
- 终点模式不是对机械臂全程运动的通用规划保证。技能提交、持物/放置和状态变化另有执行实现。
- 柜门开启会改变几何；手持物体排除在基座外部障碍集合之外。诊断必须复用相同几何接口。

## 6. Harness 输入、输出、记忆与解析

原文可查 `protocol_snapshot_v42/src/rummagebench/agents/humanclaw_vlm.py`；
新版本源码位于 `src/rummagebench/agents/humanclaw_vlm.py`。
每一步完整实际提示保留在模型日志，而不是只保存一个模板名称。

模型输出一个 JSON：

```json
{
  "visible_state": "当前图像证据",
  "mid_level_progress_analysis": "进度与上一步结果",
  "mid_level_goal": "未来约六步的目标",
  "low_level_action_reasoning": "当前动作理由",
  "action": {"skill": "MOVE", "distance_cm": 25}
}
```

- 只提交 `action`；模型自己决定是否遵循短期计划，不额外执行规划文本。
- 提示包含通用导航、先靠近再交互、失败后重新判断等指导；没有目标坐标、任务路径、解法样例。
- 最近十步记忆分别将四段文本截为 220/220/160/180 **字符**，不是 token；
  换行替为空格，并配对实际提交动作与真实公开反馈。
- 全部原始输出在日志中保留，记忆截断不影响人类审计报告。
- 解析可去掉完整 `<think>...</think>` 和一层完整代码围栏；不保留这些包装作为执行命令。
- 不接受多个动作、数组、重复 JSON key 或未闭合 think；不修复动作参数。
- 解析失败提交空动作，正常消耗一步并收到 INVALID_ACTION。
- 每一步只有一次模型生成；没有第二个模型检查，没有自动重试、夹紧数值、代填帧号或脱困动作。
- 推理服务异常/模型错配不是模型动作错误，回合标为基础设施无效并排除成绩。

## 7. 模型、服务、采样与可比性

配置精确 revision 见 [local_vlm.json](../configs/models/local_vlm.json) 及快照。
运行时实际依赖和量化信息以各模型 `report.server` / `suite.models.*.server` 为准。

| profile | 权重 | 环境 / 后端 |
|---|---|---|
| qwen3-vl-4b | Qwen/Qwen3-VL-4B-Instruct | ygbench-vlm / 原生 HF |
| internvl3.5-4b | OpenGVLab/InternVL3_5-4B-HF | ygbench-vlm / 原生 HF |
| molmo2-4b | allenai/Molmo2-4B | ygbench-vlm / 模型自定义代码 |
| smolvlm2-2.2b | HuggingFaceTB/SmolVLM2-2.2B-Instruct | ygbench-vlm-extended |
| phi4-multimodal | microsoft/Phi-4-multimodal-instruct | ygbench-vlm-phi4 / 自定义适配 |
| gemma3-4b | google/gemma-3-4b-it | ygbench-vlm-extended |
| minicpm-v4.5-8b | openbmb/MiniCPM-V-4_5 | ygbench-vlm-extended / model.chat |
| llava-onevision-7b | llava-hf/llava-onevision-qwen2-7b-ov-hf | ygbench-vlm-extended |

Llama-3.2-11B-Vision 在配置中但沿用用户之前暂缓的安排，没有列入本轮八模型。

- 单 RTX 3090 24 GB，串行一个模型服务和一个仿真 worker。
- NF4，double quantization，BF16 计算；模型自定义层存在不同 skip_modules。
- Molmo2 保留 patch_embedding；MiniCPM 保留 vpm/resampler/lm_head；
  Phi-4 保留视觉/音频/扩展 embedding、lm_head 与 LoRA 层。精确值见服务元数据。
- greedy / `do_sample=False`，seed=0；每步最多 4096 输出 token；每回合最多 100 动作。
- MiniCPM 明确 `enable_thinking=False`，但仍按统一 prompt 输出结构化理由。
  因此不能把本试验描述成所有模型均启用原生深度 thinking 模式。
- 图像数量、文字协议与任务相同；不同模型 tokenization、视觉 patching、原生模板和量化覆盖不同。
  统一 token 上限不代表完全相等的推理 FLOPs 或可输出文字长度。
- 历史默认加载超时 1800 s；单回合进程上限 7200 s；单次策略 HTTP 调用默认 300 s。
- 仿真准备、推理与总耗时应分别读取，不能把推理总时长当完整启动时长。
- 只有一个任务、一个 seed，不适合估计稳定成功率或推断模型家族总体优劣。

## 8. 结果与归因分层

请以逐步报告实时快照为准，不把旧版本、不完整回合和新协议混算。
v4.2 独立 Qwen 参考回合：100 步，14 次执行（MOVE 12、TURN 2），86 次 UNSAFE，
没有 VIEW 或操作技能，FAIL_MAX_STEPS，回合耗时 1622.85 s。
这是一个重复受阻案例，不证明“模型纯视觉能力差”或“任务完全公平”。

| 层级 | 证据 | 可以说什么 |
|---|---|---|
| 运行状态 | server/worker 错误、有效性字段 | 基础设施错误与有效任务失败分开 |
| 输出协议 | 原始文本、解析结果、schema 原因 | 缺参数、帧过期、坐标超界等 |
| 点绑定/语义 | selected_entity、bridge、适用性 | 点到了谁、技能是否适用；只给人看实例 ID |
| 基座几何 | MOVE_BLOCKED/TURN_BLOCKED + 对应扫掠 | 哪个采样位姿、哪类几何阻挡 |
| 操作可行性 | legacy_failure_reason、候选 IK/碰撞事件 | 未找到候选解、候选碰撞；有证据才再分子类 |
| 行为模式 | 多步动作和反馈 | 重复碰撞、纯看图、过早结束等统计描述 |
| 认知根因 | 图像、理由、动作与人工核验 | 只能作有证据的解释，不能从一句自述自动判定幻觉 |

`fine_reason` 历史上把 TURN_BLOCKED 合并为 MOVE_BLOCKED；
报告同时保留 private_reason 和原动作，避免误以为都是平动。
模型说“已到达”“刀可见”是模型自述，不自动当成真值。

历史日志没有完整机器人四元数；旧任务的离线重建以固定初始朝向加成功 TURN 累加。
历史探针来自 L2_S1 且坐标取整，不能给它贴“精确物理接触”的标签。
新协议失败后的 live geometry 诊断与执行器用同一接口，仍只是该执行器几何近似下的证据。

## 9. 整改：简单自然语言反馈

新协议名称 `reason-feedback-v1`；harness 为 v4.3。保留四类 code 方便机器统计，
`message` 用一到两句自然语言描述结果。英文与当前英文任务/提示一致。
报告中的中文界面用于帮助人阅读，不替换原模型输出。

示例：

```text
Movement failed: the robot's base clearance check intersects the cabinet. The robot did not move.
Turn failed: the robot's base clearance check intersects an obstacle. The robot did not move.
Grasp failed: the arm could not reach a valid interaction pose for the selected object.
Action failed: the selected image is no longer current.
Action failed: this skill does not apply to the selected object in its current state.
```

含义分别是“底座检查与橱柜相交”“转动被障碍物阻挡”“未找到可达交互姿态”
“引用的图像已过期”“技能不适用于选中物体当前状态”。

- 不提供“后退 25 cm”“绕左侧”“换一个目标”等下一步建议。
- 不做动作提交前的模型提议检查，不改变动作、参数、判分或步数。
- 先由原执行器给出拒绝，再读取未改变的场景几何解释这次失败。
- 诊断异常时退回通用原因；不更改已产生的执行结论。
- 障碍名称仅来自已验证的 live geometry 类别；历史近似诊断不向模型命名障碍。
- 失败的操作候选证据新加入 evaluator-private 日志；自然语言不输出候选列表或可行解。
- 当前未运行 v4.3 模型比较，不声称自然语言反馈已经提高成功率。

## 10. Ego / Exo demo 与复现

逐步 HTML 已有真实 ego 全景，上下排列的离线 exo 几何示意；
后者明确不是相机拍摄，不冒充真实仿真图。

`scripts/analysis/render_exo_replay.py` 从已完成 Qwen 轨迹独立重放前五步，
用外部透视相机采集 RGB，并生成上 ego / 下 exo 的配对图。
这属于诊断重放，不是模型实际收到的双视角，也不是原评估同时录制的视频。
重放逐步核对动作前位置误差（2 cm 阈值）和反馈 code；不一致即标错，不能冒充对齐成功。
实际是否完成以 `runs/harness_v42_review/exo_replay/manifest.json` 为准。

```bash
conda activate ygbench
cd /home/ubuntu/ygwang/FindingBench
python scripts/analysis/build_harness_review.py
# 独立重放需 GPU 空闲；输出目录必须是新目录：
python scripts/analysis/render_exo_replay.py --out runs/exo_demo_new
```

此整改阶段未启动旧队列的自动续跑；后续单模型实验及发布进程见 §12。
人类报告可以重复离线生成；不会调用 VLM、不修改现有成绩。

## 11. 必须继续承认的限制

1. 当前不是官方 BEHAVIOR-1K 全量任务评估，也不是 HumanCLAW 完整复现。
2. 正方形底座 + AABB 的保守性需要和真实几何或同观察条件的人类操作进一步校验。
3. Oracle 成功只能证明可执行，不足以证明模型当前视觉信息足够或障碍公平。
4. 没有完成“通用反馈 vs 原因反馈”的同配置对照；v4.3 只是已实现的新协议。
5. 八模型部分未运行；不能提供不存在的真实输出或以旧模型日志冒充新协议。
6. 本轮没有让模型接收 exo、障碍位置、路径提示或运行时替代动作。
7. v4.2 基础 source_fingerprint 未涵盖所有扩展服务文件，本披露额外保存快照及 SHA。
   v4.3 已把扩展服务、适配器、服务启动脚本和模型配置纳入统一 source_fingerprint。
8. 评分只检查配置定义的目标；没有自行追加 HumanCLAW 的 FindSR/NavSR/坐下指标。
   若新增“看到/到达”指标，必须先定义独立可复现的真值判据。

## 12. 后续授权：带真实注意力与同步 Exo 的实验

用户在停跑整改后授权进行新实验。仅启动一轮 Qwen：
`runs/harness_v43_instrumented_qwen_verified`，100 步 / 4096 token / seed 0。
原八模型队列仍停止。一次最初启动因未激活 conda 的编译器 PATH 失败，
保存在 `runs/harness_v43_instrumented_qwen`；它没有形成有效模型成绩。
正式入口改为 `bash scripts/run_instrumented_demo.sh --out <新目录> --max-steps 100`。

- 新提示将 MOVE、TURN、VIEW 的相对/绝对含义和正负号用通用数值例子集中说明。
- 交互技能、原始技能参数、碰撞/IK/目标判定没有增加规划器、动作修复或脱困策略。
- 游戏仍运行原配置；优化的是信息说明和事后反馈。增加的反馈信息本身可能影响表现。
- 摄影机为 RGB-only VisionSensor，没有可见相机实体或碰撞体；跟随机器人平移，
  保持世界参考方位，并朝向身体。它不交给 VLM，也不影响可行动作检查。
- 每次额外渲染核对机器人前后位姿不变；不调用物理步进。
- 最初静态 Exo 机位被遮挡，旧重放 manifest 已标记 `camera_usable=false`，
  不能以“产生了 PNG”代替人工画面验收。
- 参数提示和自然语言反馈同时变化，本轮不是单变量反馈消融。

注意力实现参考 [Transformers AttentionInterface](https://huggingface.co/docs/transformers/attention_interface)，
并核对当前安装的 transformers 4.57.1 Qwen3-VL 与 SDPA 实现：

1. 模型仍调用原 SDPA 计算并返回原结果，不将导出的热图反馈给生成过程。
2. 读取最后一个语言层本次前向计算的 post-RoPE Q/K 和实际 mask，另算最后 query 的
   FP32 softmax；按 attention head 平均，提取 image token 部分。
3. 每个生成 token 均保存一行真实权重，并与生成 token ID 对齐；包括生成第一 token 的 prefill。
4. `.npz` 保留原始权重、生成 token IDs、图像 token positions 和 image_grid_thw；
   `.json` 保留方法、层号、token 文本及图像注意力质量。
5. 视频默认对当前整个输出的生成 token 求平均，再按图像 grid 映射回全景；
   仅显示时按各图 p99 归一化。它不是全层 rollout，不是因果 saliency，也不是“理解程度”。
6. 数值测试核对显式矩阵计算的一致性，以及记录前后 SDPA 输出逐位一致。
   没有声称额外记录完全没有耗时/显存开销。

视频由 `scripts/analysis/compose_instrumented_video.py` 生成：
上方全景、下方左侧同步 exo、下方右侧上一步反馈 → 当前模型实际输出 → 新动作参数 →
本次执行结果；过长文本分页，不静默截断。绿色点仅表示模型实际提交的图像点；
MOVE/TURN/VIEW 没有像素点时不会伪造一个点。额外 VIEW 图存在时单独显示。
同时输出普通版 `media/episode.mp4` 与热图版 `media/episode_attention.mp4`，
每步热图 PNG、原概率 NPY、完整 attention NPZ 与视频清单。

相机诊断另见 `runs/harness_v42_review/camera_projection_audit.json`：
记录光心距底座约 1.768 m；执行器 0.8 m 方形轮廓在基座高度平面的投影位于全景高度
89.3%–94.0%。该投影只供人工审查，不是可见性真值，不提供给模型。


## 2026-10-20 protocol and local-server correction (v4.7 label retained)

The original InternVL run in `runs/harness_v47_other_vlms_serial_001` submitted 54 empty actions after strict parsing failures, then failed with CUDA OOM on request 55. Its first raw output contained prose and a standalone skill JSON, rather than the required outer planner object containing `action`. This is not evidence of 54 unsuccessful navigation attempts. All raw outputs and exports are preserved.

Every model now receives the same response-format schema on every public user turn. It specifies text fields and a nested action object but does not choose a skill, supply parameter defaults, inspect action safety, retry requests, or repair output. The original strict parser and full alternating user/assistant text history remain intact. InternVL's five-step integration check produced five parse-valid actions (MOVE, VIEW, MOVE, TURN, TURN). This check is not a successful task evaluation.

The local server checks loaded model context capacity before generation: InternVL 40960 tokens; Molmo2 36864 tokens. Input plus requested maximum output must fit. Requests exceeding this capacity fail explicitly as infrastructure errors; history and requested output budgets are not silently shortened. These limits come from the pinned checkpoints' text configurations, with no configured RoPE scaling. The CUDA allocator uses `expandable_segments:True` by default. A one-token replay of the old 47764-token input with 5 GiB allocated to simulate scene memory passed after that allocator change (peak model allocation 14477669376 bytes). This is a memory diagnostic beyond declared model context, not a valid scored inference or proof that all future requests fit.

The serial launcher permits socket reuse after server shutdown and records launcher failures even when no child state file was created. Every model still fully exits before the next model starts. Exo images remain observer-only; unavailable attention is explicitly marked and no attention video is produced for these models.

Molmo2's five-step integration check (`runs/molmo_protocol_check_001/molmo2-4b`) completed with five parse-valid MOVE actions; the first 20 cm move executed, subsequent forward moves were rejected by cabinet clearance. All five steps were exported. This establishes protocol/execution integration, not task completion or successful recovery. The full serial evaluation resumes in `runs/harness_v47_other_vlms_serial_003` with the corrected common prompt; old runs must not be pooled as the same prompt condition. 33 relevant unit tests passed.


## 2026-10-21 debugging correction (supersedes the previous context reservation rule)

The seven-model 30-step suite completed with infrastructure errors and no task success. It is not a clean capability ranking. Debugging retains original observations, messages and outputs as evidence. The paired replay summaries are in `runs/vlm_debug_20261021/debug_summary.json`; these probes do not execute actions and are not scored episodes.

Phi-4 previously forced eager attention and failed in an attention-matrix softmax allocation. The adapter now uses the checkpoint's supported SDPA implementation. The original request completed with an additional 8 GiB GPU reservation. The extended server now uses expandable CUDA allocation. MiniCPM's official pinned `chat(..., enable_thinking=False, sampling=True)` defaults are restored: temperature 0.7, top-p 0.8, top-k 100, repetition penalty 1.03. These are the pinned chat implementation defaults, not the different generic generation_config.json values. Actual decoding is recorded for every request. Its original long-history request completed with 6 GiB reserved, but returned malformed JSON; that output is preserved and not repaired.

Molmo2's original second request repeats under NF4 but completes under BF16 in the paired replay. Its deployment default is now BF16. Separately, replacing the verbose response schema with a common concise prose format instruction yields complete action JSON for Molmo2 NF4 and a recognized skill for LLaVA in the paired probes. This instruction applies to every model, supplies no chosen action and does not validate or correct a decision. Invalid planner output is described in the next public observation after the failed attempt; missing versus unrecognized skill feedback is now distinct.

Native model context capacity is checked using the loaded configuration. Input history is never truncated. A request with input below capacity may generate up to the lesser of the requested output budget and remaining native capacity; both limits are logged. A generation cut by native capacity is retained in the raw log and ends the run as an infrastructure error, without submitting a partial action. This replaces the earlier overly conservative rule that rejected an otherwise usable input if the entire requested output budget could not fit. SmolVLM2 declares only 8192 tokens, so full-history 30-step compatibility cannot be assumed.

Real closed-loop debug validation keeps a model-visible 30-step budget but stops after three steps via a separate diagnostic flag. Such reports are explicitly unscored (`run_valid=false`, `validation_only=true`). This avoids changing the visible task horizon merely to run a short test. Source and prompt hashes change while the v4.7 label is retained, as requested. Old and new results are separate experimental conditions.


Debug validation results: Phi-4 and Molmo2 each completed three real closed-loop steps with model-visible budget 30, no OOM, and three parse-valid actions. Phi-4 moved 25 cm once, then repeated two blocked moves. Molmo2 BF16 executed a 90-degree left turn, had a 50 cm move rejected, and executed a 45-degree left turn. Every step was exported. This establishes short integration stability only. SmolVLM2 BF16 was also checked with the full requested 4096-token output budget: outputs ended at 676 and 664 tokens, without reaching the budget or context boundary. The former mixed forbidden observation fields into its action, and the latter supplied no action; this format failure remains unresolved. Offline action-schema audits are post-hoc only and are never sent back into a scored run or used to select/repair an action. The replay helper is retained as a run artifact at `runs/vlm_debug_20261021/replay_debug.py`.

Final format regression probes: InternVL and Gemma produce valid MOVE25 action objects under both the original schema text and the concise prose variant. These use their existing greedy configuration; this debugging pass is not a claim of universal official-default decoding reproduction. All diagnostic processes have terminated. No new scored evaluation has been started. Remaining issues include Smol action-schema failure and limited native context, MiniCPM malformed output on the old long-history request, and untested full 30-step reliability after these corrections.

## Same-model context compaction (2026-10-21; v4.7 label retained)

The instrumented runner now defaults to a separate compaction condition. Public
reference: https://developers.openai.com/api/docs/guides/compaction . We align the
threshold / compact / continue lifecycle, not OpenAI's proprietary encrypted
compaction representation. Local models generate ordinary English memory text.

Before each planning generation, `/count_tokens` applies the same multimodal
processor and chat template as inference. MiniCPM counts at its underlying LLM
entry point after vision embeddings; it does not generate tokens during counting.
Trigger input tokens: `min(12288, floor(0.70 * native_context_limit))`, with 16384
as the fallback window if the server does not report one. The 12288 cap is an
explicit operational setting for the shared 24 GB GPU, not a claim about native
model capacity or a guarantee against OOM. All models use the same formula.

On crossing the threshold with history present, the same loaded model summarizes
older public messages, including any earlier summary. It is asked for at most
400 English words, with a 768-token generation cap, using its existing decoding
settings. One recent complete user/assistant pair remains verbatim when at least
three pairs exist; otherwise all completed pairs enter the summary. The next
request contains the unchanged skill/body/calibration protocol, the model's memory,
retained recent messages, and the latest public observation and current images.
Summary requests carry only the same current public images for compatibility with
all VLM adapters; the prompt explicitly asks not to reinterpret history using them.
No exo image, private trace, oracle state, action repair, or recommended action is
provided. Summary content is not semantically corrected or checked for truth.

Original model outputs, returned reasoning, actions, and feedback remain fully
archived. `context_compactions.jsonl` records summary inputs/outputs, image hashes,
token counts before/after, server token usage, timing and errors. These calls do
not consume robot action steps and do not enter planner/video step logs. Empty,
wrong-model, truncated, or non-reducing summaries cause an infrastructure stop;
there is no silent history deletion or fallback action. Compaction can lose
information and is not evidence of improved planning ability.

`--no-context-compaction` selects the prior complete-text replay baseline. Record
and compare these conditions separately. The standalone policy class remains
opt-in via `compact_threshold_tokens`; the instrumented runner enables it by
default. Existing result directories are unchanged. No new full scored suite has
been launched as part of implementing this change.

Validation: 54 relevant unit tests pass, including archival preservation, repeated
compaction, atomic rejection of incomplete/non-reducing summaries, and unchanged
invalid-action pass-through. Offline replay artifacts are in
`runs/context_compaction_validation/`. InternVL's boundary-delimited summary test
reduced 17061 input tokens to 6510; the summary is model-authored and not verified
for factual accuracy. An earlier Smol test repeated text until its 768-token cap
and stopped without replacing history. These are unscored checks, not task-success
results. `context_budget.jsonl` separately records token-count request latency,
including failed counts. No claim is made that other adapters have all passed a
real-model compaction replay or that compaction eliminates every GPU OOM.

## Model-owned memory contents (2026-10-21; supersedes constrained compaction above)

The user rejected harness constraints on model memory. The earlier compaction
suite was interrupted before changing sources; its original outputs remain intact.
The following changes supersede the 400-word / 768-token / recent-pair policy:

- All active public text history, including previous memory, is supplied to the
  same model. It decides memory content, organization, detail, and length. No
  fixed recent-turn retention window is imposed. The unchanged protocol and
  current public observation are provided separately on the next action turn.
- Memory generation has no separate fixed output-token cap. Both server families
  allocate the remaining native context window after multimodal tokenization,
  stopping on the model's EOS or the actual context boundary. Normal planning
  generation budgets and decoding settings remain unchanged.
- Empty, repetitive, non-reducing, output-limited or context-limited memory text
  is carried forward verbatim. These are no longer rejection/termination criteria.
  Model identity and presence of a string response remain transport checks.
- Memory is not executed as an action. No semantic validation, factual correction,
  retry, substitute summary, or oracle information is introduced. All original
  histories and generated memories remain archived.

The existing token threshold is a resource-management trigger, not a limit on
memory size or a memory-quality test. Native context capacity, available VRAM,
ordinary transport timeouts (300 s), and the existing worker runtime limit still
apply. This is not an assertion of unlimited resources or guaranteed successful
continuation: a non-reducing memory can still exhaust the actual context window,
and long generation can exhaust VRAM or time out. These must be reported as
infrastructure limitations rather than benchmark task failures. The 30-step
robot action budget is unchanged. 55 relevant unit tests pass, including verbatim
acceptance of empty, non-reducing, and boundary-limited memories without an action.
