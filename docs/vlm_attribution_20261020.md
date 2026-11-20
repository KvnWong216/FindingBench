# VLM 停跑、归因与 HumanCLAW 对照（2026-10-20）

## 当前状态与可复核证据

正式队列已按用户要求停止。原三模型批次完成；扩展批次 interrupted。SmolVLM 完成 27 条，第 28 条 T04_apple_to_counter__L1_S1 被人工中断，保留轨迹，不计模型失败。总计 135 条有效完成、0 成功；这不是 288 条完整排行榜。临时 Qwen 提示对照服务也已关闭。未经新的实验决定，不恢复批跑。

离线归因：`runs/vlm_audit_20261020/attribution.json`，复现脚本 `scripts/audit_vlm_runs.py`。只统计有 report 且 run_valid 的完整回合，未完成回合另列，不改写历史报告。

| 模型 | 回合/动作数 | 已确认的主要现象 |
|---|---|---|
| Qwen | 36/149 | 36 次 GOAL_UNSATISFIED；12 次物体技能不适用；无 schema 拒绝 |
| InternVL | 36/2160 | 1399 次 VIEW 缺 yaw；5 次 OPEN 缺 point；1 次 TURN 缺 angle；387 次 MOVE_BLOCKED |
| Molmo2 | 36/2160 | 1517 次 VIEW；467 次 MOVE_BLOCKED、98 次 TURN_BLOCKED；33 次照抄 CURRENT_FRAME；无 GRASP |
| SmolVLM | 27/1620 | 1066 次 MOVE 附带禁止的 direction 字段；527 次 MOVE_BLOCKED；27 次 MOVE 执行 |

四模型均没有成功执行物体定向技能（OPEN/CLOSE/GRASP/PLACE）。因此当前总成功率主要混合了协议、搜索导航和终止判断，尚不足以评估抓放能力。上述错误分布是描述性证据，不能转换为某个因素的因果贡献率。

### 已做的 debug

- 56 项已有回归通过：全景/VIEW 方向与点选几何、协议适配、事件、视觉回归、grounder、碰撞；新增审计及候选提示测试 2 项通过。日志 `runs/vlm_audit_20261020/tests.log`。
- 人工查看 T01 L2 S1 的 VIEW 后图像：有正常渲染的门、桌椅，未见黑屏。一次查看不能证明所有图像有效，也不能证明模型正确利用图像。
- 现有 36 条 oracle 报告均 scripted_execution SUCCESS；route_oracle 依靠私有状态找路和点选，最后经过公共技能执行。它证明存在可执行路径，不能证明纯 RGB 智能体能观察和推断同样的路径。
- 检查 MOVE/TURN 实现：沿机体朝向做扫掠碰撞检查，任一点碰撞则整段拒绝，没有部分进展。高拒绝率可能来自错误方向/大步长/保守碰撞几何，当前日志不含阻挡物明细，尚不能排除几何误杀或局部困住；不得声称地砖问题已通过本次真实场景复测。
- 6 组配对静态观测提示对照（3 个任务 × 初始及下一帧、Qwen NF4、同图同历史同 seed/预算），共 12 调用。原提示 6/6 schema 有效；候选提示 0/6，有数组/多动作输出，1 次输出截断。证据 `runs/vlm_audit_20261020/qwen_prompt_ab_ready/`。候选模板列出数组式技能示例，输出也呈数组；但候选还同时改了其他文字，所以不能把变化单独归因于数组格式。这不是完整 rollout、不能估计成功率提升。候选**否决，不替换正式提示**。初次服务未就绪的探测未产生模型调用，空目录保留，不计入配对数据。

## 结论分级

已确认：当前策略在协议遵循/导航/停止阶段退化；没有基础设施异常并不等于实验设计无缺陷。不能把严格协议惩罚全部称为视觉推理失败。

尚未确认：纯图像可解性、碰撞规则是否过于保守、初始全景与模型训练分布的影响、观察方向和机体方向分离的影响、NF4 相对 BF16 的影响。现有结果不支持“只有 VLM 不行，实验没问题”。

## HumanCLAW 源码对照

核对官方仓库 commit `6323d938795afd27d66ab926216bf4ba7b003ee6`，本次只读审查，未安装或执行其仿真。官方报告最高成功率 16.8%，并非全面解决具身任务。

| 维度 | FindingBench 当前基线 | HumanCLAW 发布实现 |
|---|---|---|
| agent | 单次调用，只返回动作 | 感知摘要、短期目标、动作理由、动作；部分动作追加 verifier |
| 停止 | REPORT_DONE 立即终止并检查 goal | Stop 候选先核验完成条件，可以被替换 |
| 历史 | 最多 60 条动作/四类反馈，无语义计划 | 最近 10 条压缩感知/计划/理由和 verifier 拒绝摘要；仍使用当前图，不是历史视频 |
| 输出处理 | 严格 action schema；缺参/多键/过期帧消耗一步 | JSON 失败最多 5 次尝试；动作名称推断、缺参默认值、数值裁剪；重试耗尽可默认向前慢走 |
| 观察 | 初始全景 2048×1024，VIEW 640×480；视角与机体运动解耦 | 448×448 第一人称、身体/脚作为视觉参照 |
| 任务 | 取刀/拿杯放置/叉子入抽屉/苹果放置，需像素点选 | 找目标、导航及坐下/上下楼等身体交互，不是同一组抓放任务 |
| 配置 | 60 环境步，单次 512 输出 token；已测约 2.2B–4B NF4 | 100 环境步，6 步计划时域；模型专属 1200–4096 token；含较大开源模型和闭源模型 |

HumanCLAW 的 planner 组装函数明确丢弃 env_feedback，不能误读为它把真值碰撞/位置反馈给模型；verifier 同样以当前图像及模型输出为依据。我们不能通过向 VLM 泄露私有 collision/goal/entity 信息仿制它。

源码证据：
- [Planner、重试、默认值、范围裁剪与历史](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/src/humanclaw_bench/agent/planner.py)
- [Planner prompt v4](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/src/humanclaw_bench/agent/prompts/v4.py)
- [Verifier v3](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/src/humanclaw_bench/agent/verifiers/v3.py)
- [发布配置](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/configs/paper_fullval_v1.json)
- [模型调用配置](https://github.com/Human-CLAW/HumanCLAW/blob/6323d938795afd27d66ab926216bf4ba7b003ee6/docs/MODELS.md)

## 公平比较与下一轮准入

1. 旧结果保留为 strict-single-call-v2，仅比较相同任务/seed 的配对结果；SmolVLM 只有 27 条，不能与其他模型的 36 条总率直接排名。36 变体来自 4 基础任务、单场景；不是 36 个独立任务或 BEHAVIOR-1K 全集。
2. 先做公共界面的人工完成记录，以及可见近距离物体的 GRASP/PLACE 单技能任务。前者尚未有人完成本次盲测，后者尚未运行，不以 oracle 替代。对 MOVE/TURN 扫描记录评估器私有的阻挡几何，核验地板不阻挡、家具合理阻挡和可行脱困路线。
3. 另设通用 agent 条件：A 原严格单次；B 完整技能说明/当前帧示例；C 在 B 上增加可审计语义记忆；D 在 C 上增加同模型 verifier，尤其是 REPORT_DONE。每次只引入一个因素，在开发任务确定后冻结，避免按测试答案调提示。
4. JSON 约束/语法重试与“补参数、裁剪数值、选择替代动作”分开评估。后者改变行为语义，不能偷偷添加到解析器。模型原输出、最终动作、修改原因和额外调用全部保存。
5. 环境步、模型调用次数、输入/输出 token、时间、量化与保留浮点层均报告。主表固定环境条件，另做计算预算相同的对照；同样温度、同一提示不等于相同计算预算。未运行的 Phi/LLaVA/MiniCPM/Gemma 不赋零分。
6. 本轮正式回合输出未触及 512 上限（离线报告可复核），所以现阶段不能声称增大输出预算能解决失败；多阶段 agent 的预算应重新定义。NF4/BF16 和不同视觉预处理分别做对照，不从目前零成功推断量化无影响。

下一次完整网格只应在至少一个公共输入条件下验证任务链可执行、阶段成功指标有区分度、接口处理规则冻结后启动。当前保持停跑。
