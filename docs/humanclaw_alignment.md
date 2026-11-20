> English-only runtime presentation: updated existing planner prompts, video labels, HTML/Markdown review templates, batch audit UI, play CLI messages, and layout brief inputs in place. Model output is explicitly requested in English; raw historical model outputs are not translated or altered. Existing review pages and all three retained run videos (ordinary and attention overlays) were regenerated at their original paths; English frame labels were visually checked. No alternate English-version files or harness version bump. 56 related tests passed. Evaluation remains stopped.

> 反馈全面审计整改（版本标签仍v4.7，当前保持停止）：已将成功MOVE/TURN/VIEW/GRASP/PLACE/OPEN/CLOSE/REPORT_DONE反馈具体化为执行后的效果；失败区分对象不支持技能/已有开闭状态/持物容量满/未持物/放置目标不支持。仅对已经选中的失败对象读取类别和状态，不暴露实例ID/坐标/目标身份。校验失败从Pydantic实际错误提取缺失字段/类型/范围/多余参数，不补参数；深度无效/像素无对象/无法绑定对象各自解释；OPEN打开后侵入底盘与操作姿态碰撞区分；任务安全规则拒绝不冒充物理碰撞；无交互姿态定义不误称距离够不到；错误报告完成明确episode已结束。无预检查、自动纠错、重试或下一步建议，游戏判定不变。83项相关回归通过（随后补充VIEW缺失和无交互姿态解释并重测）。

> 抓取反馈整改（保持v4.7版本标签）：按用户要求区分UNREACHABLE=当前姿态无法够到有效抓取姿态、COLLISION=候选抓取姿态碰撞/不安全、SKILL_NOT_APPLICABLE=选中对象不能抓或当前状态不允许。仅在动作已拒绝且对象已绑定后读取该对象category/fixed_base/graspable；固定或不可抓才说cannot be picked up，其余保留状态相关表述，避免把持物冲突误说物体永远不能抓。只反馈类别、不输出实例ID/位置/目标答案、不推荐后退或新点、不改判定和动作。57项回归通过。当前仍停止，未自动开跑。

> 当前运行：用户要求修正帧号harness、不增版本号、直接重跑。保留v4.7标签，已移除CURRENT_FRAME抓取示例，明确从最新主图/补图各自frame_id逐字填写；错误反馈区分字面占位符，不再一律称过期图，不自动纠正参数。源码/提示hash仍如实变化。旧official_decode_001已人工中断；新runs/harness_v47_qwen_official_decode_002，主进程564356，视频发布564357。官方采样/100步/完整多轮不变。56项回归通过。运行时勿改源码。

> 2026-10-20 解码整改：用户要求Qwen官方推荐设置。serve_local_vlm对qwen3-vl-4b启用官方Instruct评测采样：temperature=.7,top_p=.8,top_k=20,repetition_penalty=1,presence_penalty=1.5；默认解码种子3407（与场景seed0分开），每次请求重置，可显式decoding_seed覆盖。presence penalty由HF LogitsProcessor实现，仅对本次生成中已出现token扣分一次，不作用于历史文本，不检查/修改技能。instrumented runner输出预算改为32768；其他调用者仍按各自请求的预算执行并记录，不能误称全部一致。health和每次响应披露实际decoding，源码指纹含qwen_decoding.py。官方来源https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction。仍为HF+NF4，不是官方vLLM运行环境的逐位复现。任务提示保持v4.7，评估保持停止，未自动重启。

> 2026-10-20 v4.7试跑：按用户要求将公开半边长0.4m明确换算为80×80cm、0.64平方米、中心前后左右各40cm，并说明方形随身体旋转及障碍边界3cm余量，非外壳测量尺寸，无后退或路径建议。旧v4.6人工中断，新目录runs/harness_v47_explicit_footprint_qwen，主进程552574，视频发布552575。其余配置不变，运行时勿改源码。

> 2026-10-20 v4.6：用户指出VIEW被误当成身体转向。已补充转头/转体比喻及方向A示例：VIEW(90)只让补图看左，随后MOVE仍沿身体原方向；仅执行成功的TURN改变身体朝向，不加入后退脱困提示。核查v4.5第6次动作VIEW(90)，第7次模型误称body已朝左并MOVE20，证据成立。v4.5在完成13步后发生CUDA OOM，run_state=error/report=infrastructure_error，不能算模型任务失败；未重启。

> 2026-10-20 最新 v4.5-multiturn：按用户要求使用明确的 user/assistant 多轮消息，完整保存所有公开观测文字和模型原始输出，不摘要、不按步数/字符裁剪。规则在首条user消息（兼容不同模型模板）；每轮user公开观测包含上一实际动作与反馈，assistant原样保留返回文本。历史图片不回灌，当前图片只附最后user消息，VIEW仍按次补充。两套服务均优先消费messages；prompt字段仅为旧报告兼容的审计视图，不能当作实际模型输入。MiniCPM取消max_inp_length=32768的隐式截断，容量错误显式失败。所有模型共享消息协议，但未完成各模型GPU生成实测。42项测试通过；不启动评估，旧v4.3保持人工中断。

> 2026-10-20 最新：按用户要求停止 v4.3 Qwen 实验及视频/HTML发布进程，run_state=interrupted，不自动重启。当前 HumanCLAWPolicy 升为 v4.4-full-history：取消10步窗口和所有字符截断；完整保留模型原始返回（含格式错误、显式返回的推理）、结构化字段、实际动作和公开反馈。历史只来自模型返回与公开协议，不读取私有事件、oracle、exo或attention。图像仍是当前全景+当次VIEW补图，不回灌历史图片；这是既有图像协议，不是全多模态聊天回放。无静默上下文裁剪；超出实际模型容量的推理错误应作为基础设施限制披露，不能算任务失败。旧 strict 策略仍为60步动作反馈窗口，不能冒充本次完整历史配置。此次只整改，不启动新实验。

> 最新授权实验：用户在停跑整改后明确要求闭环反馈 + 无实体跟随 exo + 真实注意力 + 点选标记视频。当前单 Qwen 100 步实验为 `runs/harness_v43_instrumented_qwen_verified`（主进程 PID533767）；v4.3 参数契约已补充，原任务技能/判分保持。真实注意力为最后语言层 live QK softmax、head mean、逐生成 token 保存，不参与生成输出；数值检查 SDPA 输出逐位一致，127项回归通过。跟随 exo 图已人工验收；其图像不送模型。视频发布进程 PID536608，逐步HTML发布 PID536039；先生成前6步36秒预览，实验结束自动更新完整视频并清理本队列服务。勿运行时改执行源码。完整路径/说明在 `docs/harness_experiment_disclosure_v42.md` §12。旧八模型队列保持 interrupted，不自动续跑。

> 最新整改：用户已停止八模型评估，队列 `runs/harness_v42_all_models_single_task/suite.json` 为 interrupted；InternVL 100 步完成失败、Molmo2 7 步人工中断，其余未运行。不要自动重启。已实现 `humanclaw-panorama-reason-feedback-v4.3` / `reason-feedback-v1`：仅在动作结果产生后给简短自然语言原因，可命名 live 几何检查到的障碍类别；不推荐下一步、不改动作、不修参数。失败操作的 IK/碰撞候选证据保留在私有事件。127 项测试通过。完整披露见 `docs/harness_experiment_disclosure_v42.md`，人类逐步报告见 `runs/harness_v42_review/index.html`。旧 v4.2 轨迹不混入新协议。真实 exo 为独立诊断重放，以 `runs/harness_v42_review/exo_replay/manifest.json` 为准。

> 最新 v4.2：全景是每步原始主输入和主要场景信息来源。仅成功VIEW的返回观测附带一张针孔补充图；下一次任意非VIEW动作（包括失败、非法动作）后不再返回补充图。它可以用于紧随其后的点选，先执行绑定再移除；重复observe不消耗补充图，重复VIEW生成新补充图。新主张明确写入共享标定与planner提示；旧v4/v4.1持久细节约定被替代。120项相关单元测试通过。旧v4.1队列停止且保留数据，新Qwen同任务试跑位于runs/harness_v42_ondemand_pilot，100步/4096输出token。

> v4.1 更新：统一 PANORAMA_CALIBRATION 提示明确360×180投影、左右接缝相接、顶部向上/底部向下、x/y必须在[0,1]，以及全景 yaw=180−360x、pitch=90−180y 的换算（不适用于透视细节）。标定内容纳入提示指纹。Qwen同任务完整试跑位于runs/harness_v4_calibrated_pilot，100步/4096输出token；36项相关测试通过。相对v3同时改变双图观测和标定，不作单变量因果解释。

# 当前观测：全景常驻 + VIEW 细节图 v4

用户明确要求：模型每步始终看到完整全景，VIEW 只添加细节，不再替换全景。策略版本 `humanclaw-panorama-plus-detail-v4`。

- 主图 `image_png_b64/frame_id` 始终对应全景；`detail_view` 在 VIEW 后包含透视图、自己的 frame_id、yaw/pitch/hfov。仅保留最近一个细节方向。
- VIEW 不移动相机/机器人，不改变主图；成功移动或操作后两图同步刷新，细节保持所选方向。两张当前图均可点选，必须使用各自帧号及各自坐标；旧细节和移动前两图均失效。
- 本体字段中的相机坐标轴描述主图参考光学坐标；view_yaw/pitch 描述额外细节。主图投影不再在 VIEW 后改变。
- LocalVLMPolicy / HumanCLAWPolicy 均按主图、细节图顺序发送带标签的 images；服务器采用模型原生多图输入，不拼贴、不只保留最后一张。HF、MiniCPM、Phi适配器已修改；本轮只实测了Qwen，其他模型的双图能力尚需逐个实测。
- 日志保存每张输入图的hash，episode保存 step_NNN.png（全景）与 step_NNN_detail.png；人玩UI增加可点选的细节窗。
- 新 schema: configs/protocol/observation_panorama_detail_v4.schema.json、step_result_panorama_detail_v4.schema.json。

## 静态全景理解诊断

脚本 `scripts/probe_panorama_understanding.py`，结果 `runs/panorama_understanding_v1/probe/`。无动作执行，不修改原试跑。Qwen 4B NF4，单场景三次独立调用：

1. 无标定说明：识别为全景，但误判左右接缝不相邻、底部为前方；沙发横坐标为-0.8（非法）。
2. 给出投影标定说明：接缝/顶部/底部回答正确，但沙发横坐标仍为-0.4（非法）。投影回答可来自复述提示，不代表视觉空间能力已解决。
3. 全景与同一时刻左向透视双图：正确回答细节方向为左、无需移动机器人，并识别出两图共有的棕色沙发。

结论仅为小样本诊断：能识别全景类别且能对应部分双图内容，空间关系及坐标输出仍不稳定，不能声称整体支持或整体不支持全景。

真实仿真双图接口诊断 `scripts/smoke_panorama_detail.py`，输出 `runs/panorama_understanding_v1/live_smoke/`；诊断脚本调用一次VIEW并向模型发送两图，模型提议动作只记录不执行，不属于任务成绩。实测已通过：主图frame_000000保留、细节frame_000001新增，机器人姿态不变；模型收到两图并返回TURN<30>提议（未执行）。115项相关单元测试通过。临时模型服务和仿真已关闭，未启动v4任务评估。

下文为历史v3及更早版本，已被v4观测约定替代。

---

# 当前实现：HumanCLAW 对齐版 v3

`humanclaw-aligned-embodied-v3`，参考上游 commit `6323d938795afd27d66ab926216bf4ba7b003ee6`。以下是当前协议；后文 v1/v2 为历史记录。不是相同任务/身体/模型上的论文复现，不宣称性能不低于上游。

## 逐项对照

| 项目 | HumanCLAW | 本实现及理由 |
|---|---|---|
| 身体参照 | 蓝色身体/脚位于头视图底部中央 | 提供机器人型号、执行底盘半宽、操作末端、容量、执行器持物布尔值；相机在底盘坐标中的位置与身体前/上轴在光学坐标中的方向。不声称全景底部中央是脚 |
| 图像 | 当前 448×448 头视图 | 保留用户指定刚性全景与 VIEW；明确全景方位和透视视线/身体运动解耦；不无声裁掉全景 |
| 感知/计划/动作 | 三阶段字段、六动作时域 | 对齐字段和时域；恢复持续扫描后移动、清晰地标、接近与局部动作指导 |
| 历史 | 最近10步，语义字段截断220/220/160/180字符 | 相同长度；将每步描述与实际动作和反馈配对。反馈是执行事实，不生成纠错意见 |
| 动作接口 | action_id/action_name，8种全身技能 | 原生8技能 action 对象；保留用户任务的点选操作和 VIEW，避免双字段冲突 |
| JSON解析 | 宽松提取首个对象 | 支持完整 think 前缀和单个 JSON 代码块；拒绝多对象、数组、重复键；绝不挑选动作 |
| verifier | 可替换动作 | 按用户明确要求禁用 |
| 重试/回退 | 最多5次，失败自动走；默认/裁剪参数 | 按用户要求禁用；非法输出消耗环境步，服务异常中止并排除计分 |
| 环境反馈 | planner丢弃环境反馈 | 保留原四类公开执行反馈，不增加错误原因变量；未来独立对照 |
| 步数 | 100 | 对齐试跑显式设置100；8步仅接入冒烟，不用于成功率结论 |
| token | Qwen4096、Intern2048、其他依配置 | evaluator humanclaw 的既有对应默认保留；Qwen试跑显式4096，不再套用队列512默认 |
| 解码 | 后端相关 JSON mode | 本地 HF 贪心自由解码；尚无语法约束采样，不声称完全一致 |
| 低层执行 | 动作生成＋half-physics | 原有 FindingBench 运动/IK/碰撞与点选执行不变；不能视为等价物理系统 |

## 额外修复：步数覆盖必须作用于执行会话

审计发现 evaluator 原先覆盖 env._scenario，但 VisualProtocolSession 构造时深拷贝了 scenario，导致非默认步数覆盖不生效，而 report.effective_max_steps 可能误报。现改为 session.set_planning_budget，并从实际会话读取有效预算；新增2步终止回归测试。此前60步批跑与场景默认一致，不受此处覆盖错误影响。首次 v3 联调在决策前中止并记录原因，重新使用新目录。

## 本体信息的来源与边界

`embodiment-v1` 只在 humanclaw evaluator 路径启用，旧 strict 不开启。新增可选字段有类型约束，协议 schema 位于 `configs/protocol/observation_embodied_v3.schema.json` 与 `step_result_embodied_v3.schema.json`。

相机位置和轴方向由当前相机标定与当前机器人底盘姿态换算到机器人局部坐标，再输出；不传世界位姿、对象ID/名字、图、深度、分割、目标点或可行动作列表。数值表示仿真的理想本体标定，不假称真实有噪声的传感器测量。executor_holding 是已执行技能维护的持物状态，不是夹爪传感器读数，也不含物体身份。

VIEW 的透视旋转显式合成到参考相机旋转；初始全景没有单一视线，yaw/pitch 为 null。执行器的0.40m半宽方形范围不等于机器人精确网格；不把旧 reach_radius=1m 宣称为真实抓取工作空间。

## 验证与运行

83 项单元测试通过：包括世界坐标变换不变性、VIEW 对身体方向的变换、输出包装和重复键处理、动作透传、历史结果绑定、公开协议、全景、执行事件与能力边界。

真实8步接入诊断目录 `runs/harness_v3_smoke_verified`。是否实际完成，以 report.json 为准；该目录不用于正式成功率。旧 `runs/vlm_no_intervention_v2` 队列已停止，保留全部旧日志，不能与 v3 混合。

对齐预算的完整单条命令（模型服务需已启动）：

```bash
./scripts/evaluate_humanclaw.sh --model qwen3-vl-4b \
  --variants T01_knife_fetch__L1_S1 --max-steps 100 --max-new-tokens 4096 \
  --out-root runs/harness_v3_pilot
```

真实8步联调已结束：run_valid=true，8次模型调用/8环境步，EXECUTED=2（VIEW一次、MOVE30cm一次），UNSAFE=6；FAIL_MAX_STEPS，未成功操作目标。没有解析/基础设施故障或输出预算耗尽。联调后将原提示的 signed left/right、forward/backward 改为显式 positive/negative 对应方向；动作语义不变，完整试跑使用新源码指纹。模型的可达性/目标可见性描述仍可能不可靠，不能当作任务进展。

已启动 Qwen 单条100步完整试跑：`runs/harness_v3_pilot`，4096输出token上限。性能是否改善仍须以完整回合报告为准。没有恢复8模型全量队列。

---

# HumanCLAW 提示借鉴：不干预模型试错

当前版本 `humanclaw-inspired-no-intervention-v2`。根据用户要求，只借鉴任务/技能描述、结构化的简短推理与最近 10 步语义历史，不复现 HumanCLAW 的 verifier 和恢复策略。入口仍为 `scripts/evaluate_humanclaw.sh`；strict 策略保留。

## 当前执行约定

- 每个环境步只调用模型一次，模型输出感知、进展、短期目标、简短动作理由及一个原生 `action` 对象。
- 模型决定搜索、移动、点选、操作、恢复以及何时 REPORT_DONE。没有第二次核验，不替换停止或移动动作。
- action 原样提交：不推断技能、不补缺失参数、不裁剪数值、不绑定/修复 frame_id、不代选目标。公开技能协议本身已有的可选参数默认值不变。
- 非法 JSON、多对象/数组、缺失 action 输出转换为空动作，由环境按 INVALID_ACTION 计一步。没有免费重试或自动前进。
- 推理服务故障立即抛出基础设施异常并记录，不合成动作；不得作为模型能力失败计入有效成绩。
- 仅向模型提供当前公开观测、最近 10 步原始动作及公开反馈、按固定长度截断的模型自写语义历史。不会根据私有状态生成提示或恢复建议。
- 环境保留执行所必需的 schema/物理约束及原有四类公开反馈。任务判定只用于计分，不替模型决定下一步；失败的 REPORT_DONE 不被拦截改写。
- 默认沿用场景步数上限，不再自动改成 100 步。输出 token 预算仍可显式设置，公平配对需统一预算并记录实际调用成本。

## 运行和证据边界

```bash
cd /home/ubuntu/ygwang/FindingBench
./scripts/serve_vlm.sh --model qwen3-vl-4b
# 另一个终端；使用新目录，显式统一预算
./scripts/evaluate_humanclaw.sh --model qwen3-vl-4b \
  --variants T01_knife_fetch__L2_S1 --max-steps 60 --max-new-tokens 512 \
  --out-root runs/humanclaw_no_intervention_v2
```

本次未启动模型服务或完整评估。24 项相关单元测试通过，覆盖动作原样透传、停止不核验、错误输出不重试、服务异常不生成动作、历史边界和结果目录隔离。旧 `runs/humanclaw_alignment_pilot/` 是 v1 干预版的诊断，不能作为 v2 的结果。版本及源码指纹会防止向旧目录混入新结果。

以下为历史 v1 实现记录，已被上述 v2 约定替代，保留用于解释旧日志。

---

# HumanCLAW agent 流程对齐

实现版本 `humanclaw-findingbench-psv-v1`，参考 Human-CLAW/HumanCLAW commit `6323d938795afd27d66ab926216bf4ba7b003ee6`。这是对 agent 流程的适配，不是 HumanCLAW 仿真、任务集、原提示或论文成绩的复现。新策略位于 `src/rummagebench/agents/humanclaw_vlm.py`；原 strict 策略仍可用。

## 已对齐的行为

- 同一模型以当前 RGB 进行感知描述、短期计划（六动作时域）、动作选择，输出一个结构化 JSON 对象。
- 保存最近 10 步语义记忆：可见状态、进展分析、目标、动作理由和核验替换摘要，按字段限长。不把私有对象信息、oracle、真实坐标或私有碰撞原因传给模型。和上游一样，planner 不消费环境反馈；只使用自身语义历史及当前图像。
- MOVE（对应 Walk）和 REPORT_DONE（对应 Stop）额外调用同一模型核验，可接受或替换动作。其他 FindingBench 技能不自动添加额外核验，避免不透明地改变调用次数。
- 每阶段传输/JSON 失败最多尝试 5 次，退避 1/2/4/8 秒，不推进环境。宽松提取第一个 JSON 对象；每次尝试的原文、提示、token 和时间全部写入 model_calls.jsonl。
- planner 连续失败时执行 MOVE<20>；verifier 连续失败时保留提议动作。这两种回退沿用上游语义。任何传输异常会另外标记 infrastructure_degraded，整回合从任务成功率中排除；这是我们对上游故障计分的显式区别。
- 根据 action_id/action_name 解析技能，缺失移动/转向参数取默认值，数值裁剪到合法范围。原始提议、规范化过程、核验结果和最终执行动作写入 agent_decisions.jsonl。
- HumanCLAW 模式默认 100 环境步；Qwen 配置 4096、InternVL 2048、其他本地模型 1200 输出 token/调用。参数量和型号与论文不同，这只是对应调用预算的适配。

## 保留的 FindingBench 环境与显式差异

- 保留 8 技能、头顶刚性全景、独立 VIEW、原碰撞和任务判定，不增加侧步、坐下或楼梯技能。
- 物体技能输入 OPEN/CLOSE/GRASP/PLACE<x><y>，由适配器绑定当前公开 frame_id；模型不再抄写帧号。缺失/非有限坐标不自动补中心点，仍作为无效动作送交环境；坐标超出 [0,1] 会裁剪并记录。HumanCLAW 没有这些点选技能，这部分属于任务适配而非上游原样行为。
- 原始全景/透视分辨率和模型原生预处理不变，不声称等同 HumanCLAW 的 448×448 第一人称输入。
- 本机 Transformers 服务仍是贪心自由解码，JSON 约束通过提示和重试实现，**没有**实现 vLLM JSON constrained decoding；不把它描述为 JSON 语法约束采样。
- 100 步版本和旧 60 步版本不能直接解释为“仅提示改进”；配对消融可显式指定 `--max-steps 60 --max-new-tokens 512`。还应报告新增 verifier/retry 的调用与 token 成本。

## 运行

```bash
cd /home/ubuntu/ygwang/FindingBench
# 终端 1：已有模型服务时无需再启动
./scripts/serve_vlm.sh --model qwen3-vl-4b
# 终端 2：仅单条联调，不自动恢复之前批跑
./scripts/evaluate_humanclaw.sh --model qwen3-vl-4b \
  --variants T01_knife_fetch__L2_S1 --out-root runs/humanclaw_new_pilot
```

扩展模型仍使用对应的 serve_extended_vlm.sh 和独立环境。完整网格需显式 `--all`；本次未恢复任何完整网格队列。

输出根下的每模型/seed 新增 run_manifest.json，拒绝混合不同 agent、代码、服务配置或预算的结果。旧目录没有 manifest 时也拒绝追加；旧结果仍能 summarize-only。每条 report 记录 agent_policy、提示代码 hash、有效环境步数、所有阶段总调用/token/耗时。原始 scenario hash 与 max_steps_override 分别记录。

## 验证

16 项接入相关测试通过，涵盖：REPORT_DONE 经核验改为继续动作、同图两阶段、私有信息隔离、五次重试和回退日志、服务故障标记、参数裁剪、当前帧绑定、缺失目标不代选、语义记忆清空/截断及不同 agent 结果禁止混合。

真实 Qwen 单条联调输出在 `runs/humanclaw_alignment_pilot/`。是否运行有效、是否完成任务，以该目录 report.json 为准；实现对齐不意味着任务成功。

### 本次实测结果（2026-10-20）

- 18 项接入/队列相关测试通过。
- Qwen 在真实 T01 L2 S1 场景完成 6 次 planner 决策，输出感知、计划、动作；表现为重复 VIEW。诊断按计划人工结束，`diagnostic_stop.json` 标记 run_valid=false，不作为任务失败或成功率样本。没有 report.json 表示没有完成完整回合，不能称为 100 步完整验收。
- 同一初始公开图像上，单独提交假设的 REPORT_DONE 给 verifier：返回 goal_completed=false、verdict=replace、VIEW<0><0>。这是反事实核验测试，不是模型自然提出停止后被纠正的完整轨迹，证据 `runs/humanclaw_alignment_pilot/stop_probe/`。
- 本次临时推理/仿真进程已关闭，完整队列保持停止。接入通过不意味着目标定位、搜索或抓取能力已改善。
