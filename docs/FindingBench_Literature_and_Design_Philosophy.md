# FindingBench 深度文献调研与设计哲学定位报告（面向 Proposal 立论）

> **文档定位**：本报告通过 OpenResearch 连接 alphaXiv 文献库，对用户已收藏及新检索到的 **29 篇高度相关具身评测与主动感知文献（2019–2026）** 进行了原文级精读与分类整理（已在您的 alphaXiv `agent bench` 文件夹下建立 5 个二级专题子目录完成归档）。报告重点围绕 **“核心 Idea 立意”**、**“与最相近 Benchmark 的相似点与核心机制差异”** 以及 **“背后深层设计体哲学（Design Philosophy）的根本分野”** 展开，直接为后续撰写项目 Proposal 提供核心论据与学术坐标。

---

## 一、 核心 Idea 再凝练：我们在回答什么别人没回答清的科学问题？

当前具身智能社区在面对“家庭服务机器人在复杂任务上成功率极低”这一现实时，主流归因往往指向底层控制：*“因为缺少海量真机灵巧操作轨迹、双足平衡难控制、端到端 VLA 动作块（Action Chunking）泛化不够。”*

**FindingBench 提出了一个直击本质、可严格证伪的反共识科学假设：**

> **核心假设（Complete Cerebellum, Bottlenecked Brain）**：
> 哪怕我们已经给一台真实规格的双足人形机器人（宇树 Unitree G1 + 二指平行夹爪 Dex1-1）配齐了近乎完备、零控制噪声的 **“标准小脑”**（能以 100% 成功率避障寻路穿行窄道、用 10 自由度腰臂联动精准解算无碰撞抓取姿态、平稳拉开抽屉与 88° 冰箱门、并在重力沉降不稳时自动保护回滚），现有的前沿多模态大模型在充当机器人的 **“具身大脑”** 时，面对真实生活化杂乱家庭中的物品翻找与递送任务，**依然会因主动探索盲区、三维身体感缺失、细粒度指代失准、长程工作记忆衰减与环境复原遗忘而系统性失败。**

换言之：**底层的“手脚”即便彻底修好，上层的“脑子”今天依然远远没准备好进入真实家庭。** 而要证明这句话，唯一的科学方法就是造出一座既**彻底剥离底层电机抖动噪声**、又**100% 保留真实人形三维几何与物理交互约束**的“隔离诊断台”——这正是 FindingBench 存在的根本意义。

---

## 二、 文献全景地图与最相近 Benchmark 深度对比

我们在 alphaXiv 上梳理并归档了 **5 大相关文献流派（共 29 篇代表作）**。下表选取与我们关系最紧密的 **10 个代表性基准** 在 7 个核心维度上做正交对照：

| 基准名称（年份） | 核心评测对象 | 场景规模与杂乱度 | 机器人本体与物理接触 | 动作空间接口（大小脑边界） | 环境反馈设计 | 判分与失败诊断机制 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ALFRED** ([2020](https://www.alphaxiv.org/abs/1912.01734)) | 早期指令跟随规划与导航 | AI2-THOR 房间，稀疏摆放 | 无实体躯干，魔术吸附交互 | 离散导航步 + 像素掩码语义动作 | 简化状态转移 | 终局二值 SR（无具身几何失败分解） |
| **BEHAVIOR-1K** ([2024](https://www.alphaxiv.org/abs/2403.09227)) | 全栈长程日常家务活动 | 50 场景 / 1000 活动，高真实度 | 轮式双臂等，全动力学控制 | 底层连续关节控制为主（大小脑强耦合） | 物理状态直出 | 逻辑谓词 BDDL 成败（控制失败与规划失败混杂） |
| **VLABench** ([2024](https://www.alphaxiv.org/abs/2412.18194)) | 语言条件端到端机械臂操作 | 固定桌面场景，100 类任务 | 单臂/双臂桌面操作，连续控制 | 端到端 6-DoF 末端轨迹（大小脑强耦合） | 视觉闭环无显式反馈 | 进度分 + 终局 SR（聚焦桌面抓取技能泛化） |
| **EmbodiedBench** ([2025](https://www.alphaxiv.org/abs/2502.09560)) | MLLM 多层级具身代理 | 跨 ALFRED/Habitat/桌面 1128 例 | 高层无物理躯干碰撞；低层单臂桌面 | **高低层割裂**：高层点物体名字（`pick apple_1`），低层输出 7 维离散量 | 高层返回文本错误描述 | 按任务子集统计 SR，高层缺真实三维身体约束 |
| **AsgardBench** ([2026.03](https://www.alphaxiv.org/abs/2603.15888)) | 极简反馈下的视觉条件重规划 | AI2-THOR，12 类任务共 108 例 | 抽象掉导航与机械臂几何 | **纯符号物体名动作**（`FIND Mug`, `PUT SinkBasin`） | 极简二值成败信号 | 测条件分支重规划，但完全剥离了三维空间与身体可达性 |
| **HumanCLAW** ([2026.07](https://www.alphaxiv.org/abs/2607.27180)) | 虚拟人体闭环“行动智能” | 41 室内场景，1218 集“找椅子坐下” | SMPL 虚拟数字人全身动画 + 碰撞 | 亚秒级人体动作基元（走、转、坐下），**无精细手部抓放与开柜** | 视觉闭环物理碰撞结果 | 终局 SR（最高仅 16.8%），聚焦身体位置自知，无容器翻找操作 |
| **ActiveArena** ([2026.09](https://www.alphaxiv.org/abs/2609.24124)) | 主动视角选择与跨视角记忆 | **固定 180° 半圆桌面**，35 类任务 | 固定底座 + 可转头躯干双臂 | 训练专用 `ActiveArena-VLA` 连续控制（最多 9 轮感知-动作） | 记忆标注监督 + 本体关节角 | ID/OOD 成功率对比，限于固定工位桌面，无全屋跨区寻路 |
| **LIVIN** ([2026.10](https://www.alphaxiv.org/abs/2610.12069)) | 真实居住家庭数字孪生空间智能 | 30 套扫描重建居住房屋 | 3D 检测/重建/导航与基础移动操作 | 分立任务各自接口（非统一大小脑隔离协议） | 视具体子任务而定 | 四类独立子任务指标，非交互式翻找认知诊断台 |
| **MultINav-Bench** ([2026.10](https://www.alphaxiv.org/abs/2610.17089)) | 超越可达性的交互式目标导航 | MolmoSpaces 3K 导航集 | 移动底盘 + 简化开门/开柜动作 | 导航 + 触发式开门/开容器，**止步于“看见目标”** | 交互前后视觉观测 | 导航 SR 与交互必要性指标（无抓取、运送、放置与复原） |
| **RobotAgentBench** ([2026.10](https://www.alphaxiv.org/abs/2610.34649)) | 开放世界机器人自治系统评测 | 50 家庭布局，90 原子 + 30 场景题 | 轮式双臂 + VLA/WAM 端到端执行器 | 开放式规划—VLA执行—验证框架（大小脑耦合） | 多模块自由设计 | 测端到端系统与自改进架构，小脑 VLA 失败与大脑推理混杂 |
| **FindingBench（本项目）** | **完备人形小脑下的具身大脑认知瓶颈与失败范式诊断** | **50 房型 / 1,000 题生活化高密杂物**（每屋 30–40 件，993 条零歧义多样指令） | **Unitree G1 双足人形 + Dex1-1 二指夹爪**（cuRobo 10-DoF IK + 60Hz PhysX） | **严格 MCP 盲态隔离**：2D 像素点选 9 大高层技能 ↔ 零抖动真实物理小脑（含因果等价连续动画回放） | **零提示本体感觉 `proprioceptive_v2`**（4 主类 + 16 物理事实码，零暗示零泄密） | **100% 物理真值确定性判分（零 LLM 裁判）**：正交六指标 + Layer A/B 映射 **五大失败范式** |

---

### 与最核心 6 个相近工作的深度逐一对标（相似点 vs. 本质区别）

#### 1. 对标 [HumanCLAW](https://www.alphaxiv.org/abs/2607.27180)（2026.07）—— *问题意识最相近的“同盟军”，但任务层级与操作深度截然不同*
- **核心共识**：HumanCLAW 同样敏锐指出，传统符号智能体“没有身体后果”，而端到端 VLA 又把决策失败和“摔倒/失去平衡”混在一起（原文：*"a failed episode reads as a wrong decision, not a lost balance"*，[HumanCLAW, p. 1](https://www.alphaxiv.org/pdf/2607.27180?page=1)），并测出最强前沿 VLM 在闭环身体决策中成功率仅 `16.8%`。
- **与我们的三大本质区别**：
  1. **只有“走过去坐下”，没有“翻找与灵巧操作”**：HumanCLAW-Bench 的全部 1,218 集任务形态是 *“找到目标家具、走过去并坐在上面（find a target object, bring the body to it, and sit on it）”*（[HumanCLAW, p. 2](https://www.alphaxiv.org/pdf/2607.27180?page=2)），使用的是虚拟数字人全身运动生成模型；它**完全不涉及拉开双层抽屉/双开门冰箱、不涉及移开遮挡杂物、不涉及二指夹爪与 10 轴腰臂逆运动学抓取细小餐具、不涉及四视图包装细粒度核验（`CHECK`），更不涉及跨房间递送与家务关门复原（`RES`）**。
  2. **控制时间尺度不同**：HumanCLAW 让大模型每半秒决定一次“往前迈半步还是左转一点”，因此模型大量死在“低头看不见脚下的路、不知道自己贴没贴上沙发”的亚秒级里程计自知上；而真实机器人早已具备成熟的局部 SLAM 与 A\* 避障小脑！**FindingBench 把局部走路和全身 IK 真正交还给机器人小脑**，从而向上追问更本质的一层：*当机器人连走路和伸手都不会出错时，大脑在高层主动搜索、容器假设、遮挡顺序和长程记忆上还会怎么错？*

#### 2. 对标 [AsgardBench](https://www.alphaxiv.org/abs/2603.15888)（Microsoft Research, 2026.03）—— *“极简反馈防作弊”理念一致，但具身空间接地完全相反*
- **核心共识**：AsgardBench 一针见血地指出，过去很多基准给模型返回过于详尽的文本报错提示，导致纯文本模型不看图也能靠提示猜出答案（原文：*"Unlike prior embodied AI benchmarks that conflate reasoning with navigation or provide rich corrective feedback that substitutes for perception, AsgardBench restricts agent input to images, action history, and lightweight success/failure signals"*，[AsgardBench, p. 1](https://www.alphaxiv.org/pdf/2603.15888?page=1)）。
- **与我们的三大本质区别**：
  1. **符号物体名指令 vs. 连续图像像素到三维物理射线的真实接地**：AsgardBench 的动作是纯符号字符串 `FIND Mug`、`PICKUP Mug`、`PUT CounterTop`——只要写出类名就能自动瞬移过去拿起杯子，**完全抹去了“目标在画面哪个像素、离身体够不够得着、抽屉拉没拉开挡不挡手臂、桌面平不平放不放得稳”这一整套具身三维空间推理**；
  2. **规模与生活化复杂度**：AsgardBench 仅 108 个条件分支实例；FindingBench 拥有 **1,000 个满屋日常杂物（每屋 30～40 件）的生活化家庭任务**；
  3. **从“纯成败哑谜”升级为“零提示身体本体感觉（`proprioceptive_v2`）”**：真机器人在现实中伸手被挡住会有力矩/触觉感知，因此我们提供纯客观身体状态码（如 `ARM_PATH_OBSTRUCTED`、`OUT_OF_REACH`），既严格守住 AsgardBench 倡导的“零提示暗示”红线，又保留了真实机器人的物理本体感觉通道。

#### 3. 对标 [RobotAgentBench](https://www.alphaxiv.org/abs/2610.34649)（2026.10）—— *共享上游房间资产血统，但科学定位从“全栈工程竞赛”转向“受控因果诊断”*
- **渊源与联系**：FindingBench 复用了 RobotAgentBench（RAB）生态中的 50 套高精度家庭室内建筑布局底座，这是我们必须在 Proposal 中坦诚交代并重点讲透差异的一篇基准。
- **与我们的四大本质区别**：
  1. **系统耦合 vs. 大小脑正交解耦**：RobotAgentBench（共 90 个原子任务 + 30 个综合任务）旨在评测 `InternVLA-A1.5` 等端到端 VLA/WAM 模型及外挂 Agent 框架的开放世界全栈能力（[RobotAgentBench, p. 1](https://www.alphaxiv.org/pdf/2610.34649?page=1)），其执行成败高度依赖底层 VLA 操作策略本身的成功率；而 FindingBench 专为**隔离测试高层具身大脑**而生，用确定性 cuRobo 全身 IK + 防回滚 PhysX 小脑彻底消除了底层策略方差。
  2. **面向“翻找与遮挡”的 1,000 题生活化重构**：我们针对全屋 10 大生活区域注入了高密度日常杂物富化（每屋 30～40 件不冲突家庭物品），专门构建并逐题物理验证了 **872 道双抽屉隐藏搜索题、123 道可移遮挡物翻找题与 5 道双开门冰箱题（共 1,000 题）**，并配套了 993 条经 USD 材质/几何消歧的多样化自然语言指令。
  3. **本体迁移与双模式因果等价引擎**：我们将默认执行本体完整迁移并标定至 **宇树 Unitree G1 双足人形 + Dex1-1 二指平行夹爪**，并独创了 `discrete`（0.7s/步快测）与 `replay`（双腿屈膝踩实、腰臂联动连续视频）100% 因果等价的“状态三明治”架构。
  4. **零裁判双层确定性归因体系**：不依赖主观打分，直接从仿真底层状态导出正交六指标与 Layer A（24 原型）+ Layer B（10 终局归因）五大失败范式。

#### 4. 对标 [ActiveArena](https://www.alphaxiv.org/abs/2609.24124)（2026.09）与 [SpaceAct](https://www.alphaxiv.org/abs/2610.38405)（2026.10）—— *从“局部桌面/几何玩具场的主动感知”走向“全屋家庭语义—物理协同翻找”*
- **核心共识**：ActiveArena 证明了被动固定视角无法解决遮挡问题，必须靠可控头部/躯干视角与工作记忆（[ActiveArena, p. 1](https://www.alphaxiv.org/pdf/2609.24124?page=1)）；SpaceAct 更用精妙实验证明：即便直接塞给前沿大模型一份正确的参考高层计划（`+plan`），模型在三维视觉接地与防碰撞执行上依然频繁撞墙，只有同时给全三维坐标真值（`+state`）才能解决（[SpaceAct, p. 1](https://www.alphaxiv.org/pdf/2610.38405?page=1)）——这与我们的发现高度呼应！
- **核心区别**：ActiveArena 固定在 180° 半圆操作台前（35 类桌面任务，最多 9 轮交互），没有全屋多房间导航、家庭功能分区常识搜索（如客厅茶几 vs 厨房冰箱 vs 储物架）与容器开合复原；SpaceAct 则聚焦穿墙拼图等抽象几何推理。FindingBench 将主动转头观察（`VIEW`）、四面细看（`CHECK`）与物理翻找（`OPEN`/移障）统一放进了 **100 步长程真实家庭服务闭环**。

#### 5. 对标 [MultINav-Bench](https://www.alphaxiv.org/abs/2610.17089)（2026.10）与 [Mechanical Search](https://www.alphaxiv.org/abs/1903.01588)（2019）—— *打通“走过去打开看见”之后的下半场*
- MultINav-Bench（[MultINav, p. 1](https://www.alphaxiv.org/pdf/2610.17089?page=1)）迈出了从静态导航走向“开门、开柜以看见隐藏目标”的重要一步，但它的任务终点依然是 **ObjectGoal Navigation（走到看见即算赢）**；而真实家庭发现问题绝不止于“看见”——看见抽屉深处的餐刀后，还面临着抽屉拉出角度够不够伸手、挡路杂物往哪挪、抓起来怎么走到厨房白色台面稳稳放下、以及最后记不记得回去把抽屉关好。FindingBench 完整覆盖了从 **搜索 → 揭露 → 核验 → 抓取 → 跨区递送 → 现场复原** 的全链条认知闭环。

#### 6. 来自 [FailBench](https://www.alphaxiv.org/abs/2609.03611)（2026.09）的强力外部背书 —— *为什么我们坚决不用 VLM 当裁判？*
- 近期不少开放基准采用 GPT/Gemini 看录像给机器人成败和失败原因打分。而 FailBench 在 14 个真实与仿真数据集、2,197 条操作轨迹上系统评测了 13 种 VLM 失败检测器，结果发现：**最强模型的平均平衡准确率仅为 `0.77`，且一旦任务成败取决于精细物理接触与停稳状态，所有 VLM 裁判的准确率直接跌至随机猜测水平！**（[FailBench, p. 1](https://www.alphaxiv.org/pdf/2609.03611?page=1)）这为我们坚持 **“100% 物理引擎真值确定性判分 + 零大模型裁判（Zero LLM-as-a-Judge）”** 提供了极具说服力的文献依据。

---

## 三、 深层“设计体哲学（Design Philosophy）”的四大根本分野

在写 Proposal 的 Introduction 与 Design Philosophy 章节时，我们可以把 FindingBench 与现有工作的差异上升到 **四个层面的设计哲学抉择**——这正是让审稿人眼前一亮的理论高度：

### 哲学分野一：全栈耦合论 vs. 离体符号论 vs. 我们的“具身因果隔离论（Embodied Causal Isolation）”
- **传统路线 A（全栈耦合论，如 BEHAVIOR-1K / VLABench / RobotAgentBench）**：认为“要测真实具身智能，就必须连底层连续电机控制一起测”。代价是**因果不可归因**——底层夹爪滑脱 5 毫米就会掩盖上层长达 20 步的完美搜索推理。
- **传统路线 B（离体符号论，如 ALFWorld / AsgardBench / EmbodiedBench 高层 / Plan-Bench）**：认为“要隔离测高层规划，就该把身体、走路和三维几何统统抽象成 `PICKUP(apple_1)` 文字指令”。代价是**把具身问题退化成了纯符号看图说话**——模型根本不需要理解自己的身高臂展、不需要理解“没拉开抽屉手伸不进去”、也不需要在画面上精确定位把手与落点。
- **FindingBench 的“具身因果隔离论”**：
  我们主张——**真正应该被隔离掉的是“控制器的执行方差与动力学抖动”，而绝对不能被抽象掉的是“机器人真实的三维身体几何、运动学可达空间与物理接触先决条件”！**
  因此我们的小脑不会手滑、不会走歪，但它严格执行宇树 G1 的 22cm 底盘包络、86cm 臂展、8.8cm 夹爪跨度、10 轴无碰撞腰臂逆解和 65 步重力稳定判据，并用“状态三明治”保证离散评测与连续全身动画严格等价。**这在“离体符号游戏”与“含噪端到端控制”之间开辟了精准测量具身大脑的黄金中间地带。**

### 哲学分野二：保姆式文本纠错 vs. 黑盒哑巴信号 vs. 我们的“零提示本体感觉实在论（Zero-Hint Proprioceptive Realism）”
- 许多 Agent 基准在动作失败时返回诸如此类的保姆式提示：*`"Cannot grasp Item_03 because CoffeeTable_Drawer_1 is closed"`*——这实际上违反了部分可观测马尔可夫决策过程（POMDP）的物理边界，把环境变成了泄题教练。
- 反过来，完全只给 `True / False` 又剥夺了生物体最基础的触觉与本体感觉（人伸手撞到玻璃门，即使看不见玻璃，手也能感觉到“路径被硬物挡住”而非“手不够长”）。
- **FindingBench 的“零提示本体感觉实在论”**：严格模拟生物小脑向大脑回传的**纯物理本体感觉信号**（区分“够不着 `OUT_OF_REACH`”、“路径被挡 `ARM_PATH_OBSTRUCTED`”、“手里已满 `GRIPPER_OCCUPIED`”、“没放稳 `PLACEMENT_UNSTABLE`”），同时严守**零动作建议、零语义名称泄露**的信息论红线。

### 哲学分野三：死板模板指令 vs. 含糊欠定义指令 vs. 我们的“物理真值锚定的自然语言多样性（Verified Referential Diversity）”
- 真实家庭里往往同时存在多个同类台面或近似物品（如白色厨房台面 vs 黑色水槽台；绿皮笔记本 vs 黄色便签本；眼镜盒 vs 眼镜）。
- 我们的设计哲学是：**裁判端保持唯一真值刚性（确保评估可复现、落位区物理预验证），语言端通过 100% USD 几何材质核验实现“指代绝对无歧义 + 句式词汇高多样性（993/1000 唯一自然指令）”**——既拒绝不公平的“猜谜冤案”，又彻底堵死模型靠背诵固定句式模板投机取巧的可能。

### 哲学分野四：单一终局成败 / 主观裁判 vs. 我们的“全确定性正交诊断学（Orthogonal Deterministic Diagnostics）”
- 单一的成功率 `SR` 是一把钝尺子：`0%` 和 `0%` 之间可能隔着“压根没离开原地”与“已经拉开抽屉抓出目标精准放到边桌上只差最后交卷（如 `rab_F002/F003`）”的天壤之别。
- 我们的评测哲学是 **“结果与过程正交、能力与卫生解耦、全程零主观裁判”**：
  - 用 `SR`（成败）、`ST`（步数效率）、`EFF_phys`（剔除转头刷分后的单步具身物理准度）、`LAT`（纯大脑思考延迟）构成四维正交主轴；
  - 把“家务复原度 `RES`”和“死循环重复报错率 `RSER`”作为独立行为体检指标，不搅浑主成功率；
  - 将 `1 - EFF`（单步五维能力）与 `1 - SR`（十类终局归因）严密闭合为 **五大具身大脑失败范式**，让每一分差距都能直接指导下一代具身大模型的架构改进。

---

## 四、 您的 alphaXiv 文献库分类整理清单（已完成同步标记）

我们已在您的 alphaXiv 帐号 `agent bench` 主目录下创建了 **5 个结构化专题子文件夹**，并将原有 18 篇与本次新检索补充的 11 篇高相关前沿文献（共 **29 篇**）全部按对标关系归类到位，团队成员登录 alphaXiv 即可按模块直接精读：

1. **`01_大小脑解耦与VLM具身规划 (Brain-Cerebellum & Planning)`（7 篇）**
   - [HumanCLAW (2607.27180)](https://www.alphaxiv.org/abs/2607.27180) · [AsgardBench (2603.15888)](https://www.alphaxiv.org/abs/2603.15888) · [EmbodiedBench (2502.09560)](https://www.alphaxiv.org/abs/2502.09560) · [Embodied Agent Arena (2610.00854)](https://www.alphaxiv.org/abs/2610.00854) · [Plan-Bench (2610.18836)](https://www.alphaxiv.org/abs/2610.18836) *(新增)* · [Embodied Playground (2610.27039)](https://www.alphaxiv.org/abs/2610.27039) *(新增)* · [VisualAgentBench (2408.06327)](https://www.alphaxiv.org/abs/2408.06327)
2. **`02_交互式翻找与容器遮挡搜索 (Interactive Search & Rummaging)`（6 篇）**
   - [Mechanical Search (1903.01588)](https://www.alphaxiv.org/abs/1903.01588) · [MultINav-Bench (2610.17089)](https://www.alphaxiv.org/abs/2610.17089) *(新增)* · [Knowing Where to Look (2611.00499)](https://www.alphaxiv.org/abs/2611.00499) · [BW-PO (2611.19179)](https://www.alphaxiv.org/abs/2611.19179) *(新增)* · [CMAR-search (2610.30606)](https://www.alphaxiv.org/abs/2610.30606) *(新增)* · [Finder (2609.18058)](https://www.alphaxiv.org/abs/2609.18058) *(新增)*
3. **`03_主动感知与闭环空间推理 (Active Perception & Spatial)`（5 篇）**
   - [ActiveArena (2609.24124)](https://www.alphaxiv.org/abs/2609.24124) · [SpaceAct (2610.38405)](https://www.alphaxiv.org/abs/2610.38405) · [CubeBench (2512.23328)](https://www.alphaxiv.org/abs/2512.23328) · [JRDB-AVR (2609.35032)](https://www.alphaxiv.org/abs/2609.35032) *(新增)* · [CLOSER (2610.28476)](https://www.alphaxiv.org/abs/2610.28476) *(新增)*
4. **`04_生活化家庭场景与移动操作基座 (Lived-In Homes & Suites)`（6 篇）**
   - [RobotAgentBench (2610.34649)](https://www.alphaxiv.org/abs/2610.34649) · [LIVIN (2610.12069)](https://www.alphaxiv.org/abs/2610.12069) · [BEHAVIOR-1K (2403.09227)](https://www.alphaxiv.org/abs/2403.09227) · [Habitat 2.0 (2106.14405)](https://www.alphaxiv.org/abs/2106.14405) · [ALFRED (1912.01734)](https://www.alphaxiv.org/abs/1912.01734) · [RoboTidy (2511.14161)](https://www.alphaxiv.org/abs/2511.14161) *(新增)*
5. **`05_端到端VLA与失败归因判别 (E2E VLA & Failure Diagnosis)`（5 篇）**
   - [VLABench (2412.18194)](https://www.alphaxiv.org/abs/2412.18194) · [ManiSkill3 (2410.00425)](https://www.alphaxiv.org/abs/2410.00425) · [What Actually Matters on BEHAVIOR-1K (2610.37917)](https://www.alphaxiv.org/abs/2610.37917) · [LIBERO-RECOVER (2609.05178)](https://www.alphaxiv.org/abs/2609.05178) *(新增)* · [FailBench (2609.03611)](https://www.alphaxiv.org/abs/2609.03611) *(新增)*
