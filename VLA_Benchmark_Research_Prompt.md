> **新窗口请先读文末「十九、进度交接（2026-10-26 下午，最新）」**，再读「十八」。十四–十七是更早的交接（v1：36 个变体），保留作历史。

# Role：VLA Benchmark 研究工程师与现有项目整合助手

你是一名熟悉 Embodied AI、Vision-Language-Action (VLA)、BEHAVIOR Benchmark、OmniGibson、机器人任务规划、仿真环境及 Benchmark 评测设计的高级研究工程师。

我正在现有代码仓库的基础上，继续开发一个用于评测 VLA 模型的任务难度 Benchmark。

**请注意：这不是一个从零开始的项目。我们已经实现了部分任务、环境配置、技能及测试代码。你的主要职责是阅读并理解现有工程，分析论文和已有 Benchmark 中能够复用的任务与资产，尽可能减少重复开发，最后完成 2–3 个可运行的任务原型。**

请先调查，再设计，最后实现，不要一开始就新建一套系统。

---

# 一、现有项目情况

目前仓库中主要有两部分内容。

## 1. FindingBench

仓库中存在一个名为 `FindingBench` 的文件夹。

这是我之前构建的 Benchmark，其中已经定义和实现了一部分任务。

请重点检查：

- FindingBench 的整体目录结构。
- 已经定义了哪些任务。
- 任务对应哪些环境和场景。
- 任务如何初始化。
- 任务指令如何生成。
- 任务是否已经有不同难度设置。
- 是否实现了物体搜索、导航、抓取或放置。
- 是否存在可复用的场景配置、物体位置及任务生成逻辑。
- 任务成功条件如何判断。
- 是否存在可以直接调用的任务运行脚本。

尤其需要弄清楚：

**FindingBench 中哪些内容能够直接复用到我们现在的 BEHAVIOR 任务评测体系中？**

不要简单复制整个 FindingBench，而是分析哪些功能需要复用、哪些功能需要适配，以及哪些部分已经不再需要。

## 2. FindingBench 之外的新 BEHAVIOR 工程

在 FindingBench 文件夹之外，仓库中还有我们最近基于 BEHAVIOR 开发的新代码和文件夹。

这些文件夹不一定具有统一的命名方式，请根据代码、配置及依赖关系识别相关模块。

这部分工程可能已经包含：

- BEHAVIOR 环境初始化。
- 原生任务加载。
- 场景及资产加载。
- 机器人控制接口。
- Agent 执行逻辑。
- Skills / Actions。
- 人工交互或试玩脚本。
- 任务运行与成功判定。
- 执行日志或轨迹记录。

请检查真实代码，判断上述功能哪些已经实现、哪些尚未完成。

我们希望把这部分 BEHAVIOR 工程作为主要运行基础，并尽可能复用 FindingBench 中已经开发的任务设计及相关代码。

**不要在没有检查现有实现的情况下，重新编写环境初始化、任务系统、技能系统或模型接口。**

---

# 二、运行环境要求

本项目使用的 Conda 环境为：

`ygbench`

请优先使用以下方式运行命令：

```bash
conda run -n ygbench python ...
```

需要交互式运行时，可以使用：

```bash
conda activate ygbench
```

请首先验证：

1. `ygbench` 环境是否存在。
2. 当前 Python 版本。
3. BEHAVIOR / OmniGibson 等相关依赖是否可用。
4. 当前工程有哪些可以直接运行的入口。
5. 是否需要设置额外环境变量或资产路径。

所有后续运行、测试和验证均优先使用 `ygbench`。

不要擅自创建新的 Conda 环境。

不要随意升级、降级或重装已有依赖。

如果发现环境缺失依赖或存在版本问题，请先解释原因、影响及最小修复方案。

---

# 三、第一阶段：现有代码和资产调查

这是目前最重要的阶段。

请暂时不要直接编写新任务，而是先对项目进行系统梳理。

## 3.1 分析 FindingBench 中的任务

请递归检查 FindingBench 的相关代码和配置，提取所有已实现或部分实现的任务。

为每个任务记录：

| 字段 | 含义 |
| --- | --- |
| Task ID | 任务唯一标识 |
| Task Name | 任务名称 |
| Task Goal | 最终目标 |
| Instruction | 已有自然语言指令 |
| Scene | 使用的场景 |
| Objects | 涉及的物体 |
| Required Skills | 所需要的技能 |
| Success Condition | 成功判断 |
| Source Files | 具体代码和配置文件 |
| Implementation Status | 已实现 / 部分实现 / 未实现 |
| Reusability | 是否可以复用 |

请根据实际文件给出任务清单，不能根据文件名猜测任务已经完整实现。

同时区分：

- 仅有任务定义。
- 已实现环境初始化。
- 已实现技能执行。
- 已有成功判定。
- 已有实际运行验证。

重点识别能够用于以下任务的现有资产：

- 物体获取。
- 物体寻找。
- 物体放置。
- 容器打开及内部搜索。
- 跨房间导航。
- 多步骤任务。

如果 FindingBench 中已有我之前设计的简单任务，请优先将这些任务作为新 Benchmark 的候选。

## 3.2 分析GitHub上面最新提交的新的 BEHAVIOR 工程

检查 FindingBench 之外的 BEHAVIOR 相关代码。

重点寻找：

- 环境构建和初始化模块。
- Task Loader。
- Scene Loader。
- Object Registry。
- BDDL 相关任务定义（如果当前版本使用）。
- Skills / Action Primitives。
- Agent 控制接口。
- Task Success Checker。
- Rollout / Evaluation Script。
- 可运行的 Demo。

请明确列出：

1. 哪些模块可以直接调用。
2. 哪些模块需要简单适配。
3. 哪些功能尚未实现。
4. 哪些文件可能存在重复功能。
5. 哪些功能能够支持 FindingBench 任务迁移。

需要特别关注物体和场景资产。

例如：

- 现有场景里有哪些可交互物体。
- 哪些物体可以抓取。
- 哪些容器可以打开。
- 哪些任务涉及空间导航。
- 物体初始位置是否可以修改。
- 是否能够设置 Agent 初始位置和初始视角。
- 是否能够自动检查物体是否到达目标区域。

请尽可能复用已有资产，而不是为新任务制作全新的 3D 物体或者场景。

## 3.3 输出项目资产总表

最终请整理一张现有项目的复用清单，至少包含：

| 资产或模块 | 来源 | 作用 | 当前状态 | 可复用方式 |
| --- | --- | --- | --- | --- |
| 任务定义 | FindingBench / BEHAVIOR | 任务目标 | 实际检查 | 直接复用 / 适配 |
| 场景 | 现有环境 | 仿真场景 | 实际检查 | 直接加载 / 修改 |
| 物体 | 现有资产 | 交互对象 | 实际检查 | 直接复用 |
| 技能 | 新 BEHAVIOR 工程 | 执行动作 | 实际检查 | 直接调用 |
| 评测逻辑 | 现有工程 | 成功判断 | 实际检查 | 直接复用 / 扩展 |

表格中的结果必须来自真实仓库，不要虚构文件、API 或实现状态。

---

# 四、第二阶段：论文任务定义调查

除了现有项目，我们还希望研究相关论文或 Benchmark 原始实现中已经定义的任务。

**这一部分非常重要：我们不希望重新发明已经存在的任务，而是希望优先复用论文及官方 Benchmark 中成熟的任务定义。**

请先在仓库中查找：

- README 中引用的论文。
- 已存在的论文 PDF 或文档。
- BEHAVIOR 相关参考资料。
- CubeBench 等已引用 Benchmark 的参考资料。
- 官方代码仓库链接。
- 已下载的原生任务定义及配置。

如果用户提供了特定论文，以该论文为主要研究对象。

如果无法确认具体是哪篇论文，请先列出可能的候选论文及证据，并请求用户确认，而不是自行认定某篇论文就是目标。

## 4.1 论文到底定义了哪些任务？

对目标论文，请弄清楚：

1. 论文提出的是新任务、新场景还是新的评测方法？
2. 是否明确列出了任务集合？
3. 每个任务的具体目标是什么？
4. 是否存在任务分类？
5. 是否区分短程任务和长程任务？
6. 是否涉及物体搜索、抓取、放置、导航或复杂操作？
7. 是否定义了任务成功条件？
8. 是否提供官方任务配置或实现代码？
9. 任务是否依赖特定场景、物体或技能？
10. 哪些任务可以在我们现有环境中复现？

请将任务整理成表格。

| Paper Task | 任务目标 | 所需物体 | 所需技能 | 成功条件 | 原始实现位置 |
| --- | --- | --- | --- | --- | --- |

如果论文只给出了任务类别而没有完整任务实例，请明确标记。

如果论文描述了任务但没有公开对应实现，也要与可直接复用的任务区分开。

## 4.2 对照论文和官方代码

请不要仅凭论文中的文字就认定任务能够运行。

尽可能将以下内容对应起来：

- 论文中的任务名称。
- 论文中的任务描述。
- 官方代码中的任务 ID。
- 具体配置文件。
- 场景和物体要求。
- 初始化条件。
- 任务成功条件。

对于论文中的每个重要任务，尽可能给出可追溯的来源：

- 论文名称。
- Section / Table / Figure。
- 官方仓库。
- 实际任务定义文件路径。

如果论文和官方代码存在不一致，请明确指出。

如果无法获取官方实现，请标注为未验证，不要虚构文件路径或任务 ID。

## 4.3 分析哪些论文任务适合复用

对每个论文任务，重点判断：

### A. 是否能够直接复用

当前 BEHAVIOR 工程已经存在对应任务定义、场景、物体和技能，只需要加载并运行。

### B. 是否只需少量修改

已有场景和物体可以满足要求，只需要修改自然语言指令、物体位置或少量任务配置。

### C. 是否需要适配

任务已有相关实现，但需要适配 FindingBench 和当前 BEHAVIOR 的接口。

### D. 是否暂时不适合

任务需要当前工程尚未实现的复杂技能、特殊物体或者大规模场景修改。

请重点推荐 A、B 两类任务。

我们现阶段的目标是尽快形成可以运行的任务评测原型，而不是重新实现复杂的机器人技能。

---

# 五、第三阶段：FindingBench、BEHAVIOR 与论文任务的交叉对照

现在需要回答一个关键问题：

**论文中有哪些任务，我们现有工程实际上已经能够支持？**

请建立交叉映射表。

| 候选任务 | FindingBench 支持 | BEHAVIOR 支持 | 论文来源 | 缺少的部分 | 复用优先级 |
| --- | --- | --- | --- | --- | --- |

特别关注以下情况：

1. FindingBench 已实现，但新 BEHAVIOR 工程尚未接入。
2. 原生 BEHAVIOR 已经实现，不需要重新设计任务。
3. 论文已有任务定义，现有工程恰好具备所需物体与技能。
4. FindingBench 与 BEHAVIOR 有相似任务，可以共享成功条件或任务配置。
5. 任务目标相同，但需要适配物体名称、状态谓词或动作接口。

请不要只比较任务名称。

需要比较任务实际执行过程、所需技能、场景条件以及最终成功状态。

如果两个任务表面相似，但一个需要打开容器、另一个只需要拿起桌面物体，应明确说明差异。

请尽可能以最小改动的方式建立复用关系。

---

# 六、第四阶段：设计新的 VLA 难度评测任务

完成以上调查后，我们希望通过现有资产构建自己的难度评测体系。

核心研究问题是：

当一个 VLA 模型无法完成任务时，这种失败与语言指令的抽象程度、空间目标的可发现性有什么关系？

我们暂时定义两个维度。

## Dimension 1：Language / Semantic Difficulty

### L1：明确操作指令

直接给出目标和必要的动作步骤。

例如：

"拿起桌上的杯子，然后放到餐桌上。"

### L2：目标导向指令

明确最终目标，但不提供全部步骤。

例如：

"把杯子放到餐桌上。"

### L3：抽象意图指令

使用更自然、抽象的方式描述需求。

例如：

"我待会儿需要在餐桌上用那个杯子，先帮我准备好。"

这一维度主要用于观察：

- 语言理解。
- 意图推断。
- 任务规划。
- 子任务分解。

要尽可能保证三个级别对应相同的最终成功条件。

## Dimension 2：Spatial / Visual Difficulty

### S1：Directly Visible

目标物体可以在初始观察中直接看到。

### S2：Occluded / Local Search

目标物体位于当前房间，但是被遮挡或放在容器中，需要局部搜索。

### S3：Multi-Room Search

目标物体位于另一个房间，需要导航和探索。

这一维度主要用于观察：

- 视觉感知。
- 目标发现。
- 局部搜索。
- 空间导航。
- 历史观测和记忆利用。

这里的难度级别是实验条件，不应在收集数据之前认定其一定存在严格性能排序。

---

# 七、任务设计必须尽可能保持可比性

假设已有一个任务：

"将指定杯子放到餐桌上。"

希望通过修改指令和目标初始位置生成不同任务变体。

例如：

| Variant | Language | Spatial | 说明 |
| --- | --- | --- | --- |
| V1 | L1 | S1 | 明确操作，物体可见 |
| V2 | L2 | S1 | 目标指令，物体可见 |
| V3 | L1 | S2 | 明确操作，局部遮挡 |
| V4 | L2 | S2 | 目标指令，局部遮挡 |

请尽量控制：

- 相同的基础任务。
- 相同的最终目标。
- 相同的成功条件。
- 相同的动作接口。
- 相同的环境版本。
- 可复现的场景初始化条件。

需要注意：如果改变空间位置后，必须增加导航、开柜门等操作，应把这些额外操作和环境变化记录下来。

不能简单把执行步数增加全部归因于空间推理能力不足。

对于 L3 抽象指令，需要检查是否存在多种合理解释，避免因为指令本身模糊而破坏实验可比性。

---

# 八、第五阶段：选择最值得实现的 2–3 个任务

根据前面的调查结果，推荐 2–3 个任务。

选择优先级为：

1. FindingBench 已有实现。
2. 新 BEHAVIOR 工程已经具备相应技能。
3. 原生 BEHAVIOR 或目标论文已有对应任务定义。
4. 不需要新建 3D 资产。
5. 场景和目标物体位置容易控制。
6. 成功条件能够自动判断。
7. 能够构造至少两种语言和两种空间条件。

对于每个推荐任务，请给出：

- 任务来源。
- 对应现有代码文件。
- 使用的场景。
- 需要的物体。
- 需要的技能。
- 当前已有的实现。
- 需要修改的部分。
- 可以生成的任务变体。
- 预计实现成本。
- 潜在风险。

不要先预设必须完成某个杯子、刀具或烹饪任务。

**最终任务必须根据实际 BEHAVIOR 和 FindingBench 已经支持的能力来选择。**

如果现有工程中已经有成熟的任务，请优先使用它。

---

# 九、第六阶段：任务实现与整合

在完成分析和任务选择之后，开始修改工程。

请遵守以下工程原则。

## 9.1 复用现有系统

不要重新搭建 BEHAVIOR 环境。

不要重新实现已存在的 Skills。

不要重复实现已有成功判定器。

优先复用已有的：

- Scene Loader。
- Task Loader。
- Agent Controller。
- Skill Executor。
- Observation Interface。
- Success Checker。
- Logging System。

## 9.2 尽量通过配置生成任务变体

不要为每个难度级别创建完全独立的任务执行代码。

希望能够通过统一的任务配置来切换：

- 基础任务。
- 自然语言指令。
- 场景。
- 物体初始位置。
- Agent 初始位置。
- 观察方式。
- 成功条件。

任务变体应尽可能共享同一个底层任务定义。

## 9.3 不要破坏已有工程

请保留 FindingBench 中的原有任务。

不要随意重命名、删除或移动已有文件。

如需适配旧模块，优先编写轻量级 Adapter。

尽可能保持原有 BEHAVIOR 代码可运行。

不要进行与本研究无关的大规模重构。

---

# 十、第七阶段：运行与验证

所有测试优先在 `ygbench` 环境下进行。

先完成以下验证：

1. 环境能够正常初始化。
2. 任务配置能够正常加载。
3. 场景及物体位置正确。
4. Agent 能够获得预期观测。
5. Skills 能够正常调用。
6. 成功条件能够正常判断。
7. 可以保存任务执行结果。

如果真实 VLA 模型尚未接入，可以先使用现有的脚本、控制器或人工试玩接口验证。

但请严格区分：

- Task Initialization Success。
- Scripted Execution Success。
- VLA Task Success。

任务能够运行，并不意味着 VLA 模型能够完成任务。

请保存运行命令、日志和错误信息。

---

# 十一、最终希望得到的结果

## Deliverable 1：项目现状报告

完整说明 FindingBench 和新 BEHAVIOR 工程已经实现了哪些内容。

重点是任务、物体、场景、技能和评测逻辑。

## Deliverable 2：论文任务分析报告

这是本次新增的重要交付内容。

需要明确回答：

**论文究竟定义了哪些任务？**

以及：

**哪些论文任务能直接在我们的 BEHAVIOR 环境中运行？**

请尽可能给出具体任务名称、定义来源、实现文件和复用条件。

## Deliverable 3：任务复用映射

建立 FindingBench、原生 BEHAVIOR、论文任务之间的对应关系。

列出推荐优先复用的任务，以及需要适配的代码。

## Deliverable 4：2–3 个任务设计

每个任务应包含：

- 标准任务目标。
- 自然语言指令变体。
- 空间条件变体。
- 场景与物体要求。
- 所需技能。
- 成功条件。
- 配置方式。

## Deliverable 5：可运行原型

优先实现第一个任务的 2×2 条件组合，验证后尽可能扩展到 2–3 个基础任务。

## Deliverable 6：运行说明与测试记录

提供真实的执行命令、任务配置、运行结果及已知限制。

---

# 十二、工作顺序与优先级

请严格按照以下顺序推进：

### Phase A：现有资产调查

先看 FindingBench，再看其他 BEHAVIOR 相关代码。

列出已有任务、场景、物体、技能及可以运行的入口。

### Phase B：论文任务调查

找出目标论文和对应实现，分析论文定义的任务，整理任务清单与复用条件。

如果论文身份不能确认，请明确提出这一阻塞，不要自行编造论文内容。

### Phase C：交叉复用分析

对比 FindingBench、BEHAVIOR 和论文任务，找出改动最小、最适合作为 Benchmark 原型的任务。

### Phase D：任务设计

为 2–3 个基础任务设计语言和空间难度变体。

### Phase E：实际实现

先实现一个任务的最小对照实验，并在 `ygbench` 环境中运行验证。

### Phase F：扩展与总结

如果第一个任务运行成功，再扩展其他任务，整理实验记录和后续 Benchmark 开发计划。

---

# 十三、特别强调

当前工作的最高优先级是：

**弄清楚我们已经拥有什么、论文定义了什么、这些内容能够复用多少，然后以最少的代码修改实现可运行的评测任务。**

不是重新设计一个全新的 Benchmark 框架。

不是把 FindingBench 推倒重写。

不是重新实现 BEHAVIOR 的原生功能。

也不是立即构建大型任务数据集。

请始终遵循：

**Existing Assets First → Paper Task Analysis → Reuse Mapping → Minimal Task Adaptation → Runnable Benchmark Prototype**

现在请从检查项目目录和 `ygbench` 环境开始，先完成 FindingBench、BEHAVIOR 和论文任务的调查，输出任务及资产清单。

在没有完成调查之前，不要直接创建大量新文件或者重复实现已有功能。

---

# 十四、进度交接（2026-10-18）

## 1. 路径与环境（已核实）

- 工作目录：`/home/ubuntu/ygwang/FindingBench`（符号链接 → `FindingBench_20261016_200444/workspace`，Python 包名 `rummagebench`，editable 安装进 `ygbench`）。**本目录即 FindingBench，也是“新 BEHAVIOR 工程”所在仓库**。
- “GitHub 最新提交的 BEHAVIOR 工程” = `KvnWong216/FindingBench` main `cce72ad`（2026-10-17，官方 BEHAVIOR pilot + 全景 VIEW + ygbench 人工试玩）。本地 git 副本 `/home/ubuntu/ygwang/FindingBench-publish`，代码与工作目录逐文件一致。本机未安装 `git` 命令。
- `/home/ubuntu/ygwang/Interactive-Perception`：MolmoAct2 + LIBERO 研究仓库，**无 BEHAVIOR 代码**，与本任务无关（将来接 VLA 模型可参考）。
- 上游：`/home/ubuntu/ygwang/BEHAVIOR-1K-6cbf70b…`（OmniGibson 3.9.3、bddl3）；资产：`/home/ubuntu/ygwang/behavior-datasets`（51 个场景、7 个官方 2026 实例场景 / 199 个模板）。
- `ygbench`：Python 3.11.16，omnigibson / bddl / pinocchio 可导入；激活时自动设置 `OMNIGIBSON_DATA_PATH`。**本对话未启动任何 GPU 仿真**。`/data1` 在本机不存在（旧主机路径）。

## 2. 已完成的交付物

| 内容 | 文件 |
|---|---|
| 3.1 FindingBench 任务清单（11 字段 + 五级状态） | `docs/phaseA_3.1_findingbench_task_inventory.md` |
| 3.2 新 BEHAVIOR 工程模块调查 | `docs/phaseA_3.2_behavior_engineering.md` |
| 3.3 项目资产总表 | `docs/phaseA_3.3_asset_reuse_table.md` |
| 任务分层设计（Phase D 初稿） | `assets/tasks/vla_difficulty_v1/`（README、levels、common、base_tasks/T01–T03、variants） |

## 3. 关键结论（不必重新调查）

1. **FindingBench 任务**：knife_search_001 + batch_kitchen_v1（共 8 条，Beechwood_0_int，历史 GPU 运行、本机未跑）；knife_search_v0（30 条，生成文件缺失）；Level-1 样例 3 条（遮挡关系未写进 YAML）；easy_v1（220 条，历史认证）；household 批次（84/184 物理通过，0 可评测，路径指向 `/data1`）；`behavior_groceries_pilot`（官方 `carrying_in_groceries`，**本机唯一实测通过初始化 + 官方判定**，抓放未跑）。
2. **用户决定**：easy_v1 先不用（与新的任务定义有出入），原样保留，不移动不改名。
3. **官方任务**：已接入 FindingBench 的只有 groceries 一条；官方 BDDL 1016 条，刚体 191 条，本机有官方实例的刚体任务 56 条（house_single_floor / house_double_floor_* 等新场景，与 easy_v1 的经典场景不同）。
4. **可直接调用**：`sim/omnigibson/env_factory.py` + `backend.py`、`core/scenario.load_scenario`、技能 MOVE/TURN/VIEW/OPEN/CLOSE/GRASP/PLACE/REPORT_DONE（符号执行 + IK/碰撞可行性）、`environment/env.InteractiveSearchEnv`、`skills/report_done.goal_satisfied` + `evaluation/goal_checker.py`（holding/on_top/inside/open/closed）、`events.jsonl` 日志、`scripts/serve_play.py` / `play_behavior.sh`。
5. **尚未实现**：VLA/VLM 模型适配器（全仓无模型调用）；变体批量 runner + SR/步数汇总（`cli evaluate-visual` 只有 ScriptedPixelAgent 单回合）；放置类 goal 的仿真验证。
6. **需要适配的点**：
   - 官方任务路径（`behavior_task`）在 `backend.setup()` 提前返回：用 BDDL 自然语言目标覆盖 `instruction`，忽略 `objects/placements`，机器人用预采样位姿 → 在官方任务上做 L/S 变体需加覆盖适配。
   - 旧场景缺 `robot.panorama` → VIEW 返回 OUT_OF_CAPABILITY（补 YAML 即可）。
   - **`core/scenario.py` 的 `_cross_check` 要求 goal 的 receptacle 是 YAML 生成物体，场景原生家具（餐桌、冰箱）被拒绝**；运行时 `goal_checker` 支持家具。需放宽这一条校验（尚未修改，等用户确认）。
7. **坑**：knife 系列的 `bottom_cabinet_no_top_*` 无顶板，柜内物体不开门也可见 → 不能当 S2；knife 起点到柜子 5–8 m，MOVE 每步 ≤1 m，原 16 步预算在当前协议下不够。

## 4. 任务分层设计现状（`assets/tasks/vla_difficulty_v1/`）

- 3 个基础任务 × L1–L3 × S1–S3 = 27 个变体，每格 3 个任务；变体 ID `<task_id>__<L>_<S>`。场景均为 `Beechwood_0_int`，起点 `start_kitchen` = knife_search_001 的 `kitchen` 锚点 `(0.75, -8.15)`，统一 60 步、全景 + VIEW。
- T01_knife_fetch：`holding(target_knife)`；T02_mug_to_table：`on_top(target_mug, breakfast_table_uhrsex_0)`；T03_apple_to_fridge：`inside(target_apple, fridge_dszchb_0)`。
- S1 = 厨房岛台 `countertop_tpuwys_4`；S2 = 微波炉下抽屉 `bottom_cabinet_rvpunw_0/link_2`；S3 = 办公室 `desk_rzyfxk_0/1` 或餐厅 `coffee_table_dnsjnv_0`。L1 指令含位置和步骤（因此 L1 下空间维度只测执行/导航，已在 `levels.yaml` 注明）；L3 歧义物体已从场景移除。
- 离线校验（ygbench，无仿真）：27 个变体组装成 `ScenarioSpec` 通过，家具名、物体模型、房间归属均核对过；**T02/T03 的 18 个变体被上述 goal-receptacle 校验挡住**。校验脚本在旧会话 scratchpad，未入库；逻辑：`common.yaml` + `base_tasks/<T>.yaml` → instruction 取 `language[L][S]`（L1）或 `language[L]`，placements = `spatial[S].placement` + background，objects = target + background。
- 所有新位置（岛台、书桌、餐厅矮桌、冰箱开门+内部放置）从未在 FindingBench 中构建；`rvpunw_0/link_2` 标记 gpu_smoke_needed。

## 5. 下一步（按顺序）

1. 向用户确认是否放宽 `core/scenario.py` goal-receptacle 校验（一行级改动），放宽后补单测。
2. 写轻量编译脚本（如 `scripts/compile_vla_variants.py`）把变体生成到 `build/vla_difficulty_v1/<variant_id>.yaml`，不在 `scenarios/` 手写 27 个文件。
3. 本机 GPU 冒烟 T01 的 L1/L2 × S1/S2（`variants.yaml: first_prototype`）：先确认空闲 GPU（`nvidia-smi`，用 `CUDA_VISIBLE_DEVICES` 隔离），用 `InteractiveSearchEnv(path, mode="agent")` reset，检查 S 门禁（起点全景可见像素、开抽屉后可见率），再用人工网页或脚本化动作验证 OPEN/GRASP/REPORT_DONE；区分 Task Initialization / Scripted Execution / VLA Task Success，保存命令和日志到 `runs/`。
4. 再做 T02/T03（PLACE 与 fridge 首次验证），按脚本化轨迹校准步数预算。
5. 尚未开始：第二阶段论文任务调查（Phase B，需向用户确认目标论文）、第三阶段交叉映射（Phase C）、模型适配器与批量 runner。


---

# 十五、进度交接（2026-10-18 夜间自主运行）

总报告：`docs/vla_difficulty_v1_report.md`；运行汇总：`runs/vla_difficulty_v1/summary.md`；设计：`assets/tasks/vla_difficulty_v1/README.md`；论文调查：`docs/phaseB_paper_tasks.md`。

## 1. 已完成

- 放宽 `core/scenario.py`：goal 的 receptacle 可以是场景原生家具（交接 §5.1）。全量单测 1006 passed / 24 skipped。
- 编译器、单变体三级验证（初始化 / 空报告基线 / 特权脚本 oracle）、批量汇总、位置认证、场景探针、可达性图、模型无关策略适配器（`--agent module:factory`）。
- **T01 取刀 9 个变体全部通过**（脚本化 S1 8 步、S2 7 步、S3 10 步 / 60）。
- Phase B：目标论文未确认，以 BEHAVIOR-1K 官方 BDDL 为对象完成分析（191 条刚体任务按 L/S 轴分类）。

## 2. 仿真中发现并已修正（详见报告第 4 节）

起点与椅子碰撞 → 移到 (0.85,−7.70)；全景 1024→2048；S1 门禁改为全景 ≥20 px；S2 抽屉换成 `qohxjq_0/link_2`（rvpunw_0 链接名错且是落地抽屉）；**门宽 0.75 m < 底盘，只能在厨房 + 客厅活动** → S3 = 客厅矮书架 `bookcase_owvfik_1`；**PLACE 只在两块厨房台面可行**（餐桌、茶几、客厅柜都碰撞）→ T02 改为放到 `countertop_tpuwys_0`，T04 改为把苹果放到岛台；T03（放进冰箱）blocked。

## 3. 需要用户决定

1. 目标论文是否为 BEHAVIOR-1K（若是 CubeBench 等请提供）。
2. 是否扩展 PLACE 支持 inside（冰箱 / 柜子），解锁 T03 及 39 条官方任务。
3. 真正隔门的多房间 S3 需要换场景（官方 R1Pro 实例场景）或换小底盘机器人。
4. 接入哪个 VLA / VLM 模型（适配器已就绪）。

## 4. 下一步

T02/T04 其余变体的运行状态见 summary.md；未跑完的用 `python scripts/run_vla_batch.py --all --skip-done` 续跑。


---

# 十六、进度交接（2026-10-18 23:40，最新）

## 0. 先看这些文件

| 内容 | 文件 |
|---|---|
| 总报告（交叉映射、任务设计、证伪记录、运行结果、代码清单） | `docs/vla_difficulty_v1_report.md`（第 5–6 节 T03/T04 部分**尚未按本节更新**） |
| 论文调查（BEHAVIOR-1K，已确认） | `docs/phaseB_paper_tasks.md` |
| 任务设计 | `assets/tasks/vla_difficulty_v1/`（README、levels、common、base_tasks/、variants.yaml；废弃稿在 `base_tasks/_superseded/`） |
| 每个变体的运行结果 | `runs/vla_difficulty_v1/summary.md`（`python scripts/run_vla_batch.py --summarize-only` 重新生成） |
| 位置 / 放置认证记录 | `runs/vla_difficulty_v1/_locations/round1–5_*.json` |
| 可达性图 | `docs/evidence/vla_difficulty_v1/beechwood_reachability.png` |
| 仿真约束备忘 | 记忆文件 `beechwood-r1pro-embodiment-limits`（Claude 自动记忆） |

## 1. 用户已做的决定（2026-10-18）

1. 参考论文 = **BEHAVIOR-1K**（arXiv 2403.09227，CoRL 2022）。已写入 Phase B 文档。
2. **允许扩展 PLACE 支持 inside**。已实现（见第 3 节）。
3. **先不接模型**，由队友接（接口 `--agent module:factory`，见 `src/rummagebench/agents/policy_adapter.py`）。
4. T04 换一个对抓取方式不敏感的放置目标（已换，但见第 2 节：仍不稳定）。
5. 每个难度格 2–3 个任务即可，先搭框架，不大规模造任务。

## 2. 当前网格状态（36 个变体 = 4 个任务 × L1–L3 × S1–S3）

三个任务共用同一套空间位置：S1 厨房岛台 `countertop_tpuwys_4`；S2 走廊门旁柜子顶层抽屉 `bottom_cabinet_no_top_qohxjq_0/link_2`（T03 例外，见下）；S3 客厅矮书架 `bookcase_owvfik_1`，固定 `position_xy: [4.45, -5.45]`。起点 kitchen (0.85, −7.70)，60 步。

| 任务 | 目标 | 状态 |
|---|---|---|
| **T01_knife_fetch** | `holding(knife)` | **9/9 通过**（脚本化 S1 8 步、S2 7 步、S3 9 步） |
| **T02_mug_to_counter** | `on_top(mug, countertop_tpuwys_0)` | **9/9 通过**（S1 17–19、S2 8、S3 15 步） |
| **T03_fork_to_drawer** | `inside(fork, bottom_cabinet_no_top_qohxjq_0)`；S2 改用相邻柜 `bottom_cabinet_no_top_pluwfl_0` | 认证通过（pluwfl 隐藏且必须 OPEN；抽屉内 PLACE 成功 5 步）。**但 L2 变体：S1 失败（拿着叉子时对目标抽屉 OPEN 不可行），S2 失败（PLACE 不可行），S3 运行中**。L1/L3 未跑 |
| **T04_apple_to_counter** | `on_top(apple, countertop_tpuwys_0)` | **L2：只有 S2 成功（12 步）；S1、S3 PLACE 不可行**。L1/L3 未跑 |

截至 23:40：36 个里 23 个有结果，其中 19 个脚本化成功（T01 9 个、T02 9 个、T04 L2_S2），4 个失败（T03 L2_S1/S2、T04 L2_S1/S3）；T03 L2_S3 运行中，其余 12 个（T03/T04 的 L1、L3）未跑。以 summary.md 为准。

## 3. 本轮新增代码（全部有单测；全量 1010 passed / 24 skipped）

- `core/scenario.py`：`PlacementSpec.position_xy`（可选，仅 on_top，用于把物体固定在书架看不到的一端）。
- `sim/omnigibson/backend.py`：`_place_at_xy`；`receptacle_interior_regions(entity)`（固定家具里带原生 fillable 体积的抽屉链接）；`receptacle_region` 对**打开的**固定家具返回被拉出抽屉的 inside_volume。
- `feasibility/interaction_target.py`：inside 区域的放置点在抽屉底面、自上而下接近；`select_open_interior`。
- `evaluation/goal_checker.py`：`inside` 目标在抽屉重新关上后仍按抽屉内部判定。
- `evaluation/route_oracle.py`：先试评测器私有锚点 `<entity>__<SKILL>*`（`common.yaml` 里 4 个 PLACE 锚点），PLACE 反事实搜索只查第一批 8 个点、预算 600 s。
- 测试：`tests/unit/test_inside_drawer_place.py`、`test_route_oracle.py`、`test_scenario_schema.py`。

## 4. 主要未解决问题（下一窗口从这里开始）

1. **“能否放下”依赖于物体是怎么被抓起来的**（苹果尤其明显；T03 拿着叉子时 OPEN 不可行）。原因未查清：怀疑是抓取偏移（held offset）让持物与家具碰撞，或可行性检查把持物也算进碰撞。建议先在一个失败变体上打印 PLACE / OPEN 的可行性详情（`FeasibilityVerdict.details` 里的碰撞对），再决定：
   - 改抓取姿态 / 持物碰撞模型（动 FindingBench 核心，先问用户）；或
   - 换物体（T04 用杯子类物体已证明稳）；或
   - T03 让 oracle 先开目标抽屉再取叉子（子目标顺序问题，最小改动）。
2. PLACE 可行性检查很慢（每次约 100–180 s），一个取放变体要 5–28 分钟。
3. 报告 `docs/vla_difficulty_v1_report.md` 的 T03/T04 部分需按本节更新。

## 5. 运行方式与注意事项

- 单个变体：`conda activate ygbench; python scripts/run_vla_variant.py build/vla_difficulty_v1/<id>/scenario.yaml`（改设计后先 `python scripts/compile_vla_variants.py`）。
- 批量：`python scripts/run_vla_batch.py --timeout 3000 --variants ...`；汇总：`--summarize-only`。
- 位置认证（一次加载测多个位置）：`python scripts/certify_vla_locations.py <scenario> --target-locations rec:rel[:link] ... --goal-receptacles ...`。
- **长任务务必用 `nohup … &` 启动**：会话结束会杀掉普通后台任务（2026-10-18 已发生两次）。
- **不要用 `pgrep -f <命令行片段>` 做等待条件**：启动队列的 shell 的命令行里也包含同样的片段，会永远等下去（2026-10-18 浪费了 8 小时）。改用 PID 文件或 `wait <pid>`。
- 23:40 时 GPU 上仍在跑 `T03_fork_to_drawer__L2_S3`（独立进程，跑完自动退出）；后续 L1/L3 队列已停止，没有其他排队任务。


---

# 十七、进度交接（2026-10-19 早晨，最新）

## 0. 先看这些文件

| 内容 | 文件 |
|---|---|
| 总报告（交叉映射、任务设计、证伪与修复记录、运行结果、代码清单） | `docs/vla_difficulty_v1_report.md`（已按本节更新） |
| 每个变体的运行结果 | `runs/vla_difficulty_v1/summary.md`（`python scripts/run_vla_batch.py --summarize-only` 重新生成） |
| 任务设计 | `assets/tasks/vla_difficulty_v1/`（README、levels、common、base_tasks/T01–T04、variants.yaml） |
| 诊断记录 | `runs/vla_difficulty_v1/_diagnose/`（OPEN/PLACE 碰撞对、实际放置结果、抓取抬升探针） |
| 论文调查 | `docs/phaseB_paper_tasks.md`（BEHAVIOR-1K） |

## 1. 状态：网格完整，36/36 通过

4 个基础任务 × L1–L3 × S1–S3 = 36 个变体，每格 4 个任务。三级验证（初始化 + S 门禁 / 空报告被拒 / 特权脚本 oracle 只用公共动作完成）全部通过。最少步数（S1 / S2 / S3，预算 60）：

| 任务 | 目标 | 步数 |
|---|---|---|
| T01 取刀 | `holding(knife)` | 8 / 7 / 9 |
| T02 杯子 → 走廊门旁台面 | `on_top(mug, countertop_tpuwys_0)` | 17 / 8 / 15 |
| T03 叉子 → 走廊门旁柜子顶层抽屉 | `inside(fork, bottom_cabinet_no_top_qohxjq_0)` | 22 / 14 / 18 |
| T04 苹果 → 走廊门旁台面 | `on_top(apple, countertop_tpuwys_0)` | 13 / 8 / 15 |

36 个结果全部出自最终代码（最早启动 03:02:39，晚于代码最后修改 02:59:38）。VLA Task Success 未测（按用户决定由队友接模型，接口 `--agent module:factory`）。

## 2. 本轮修复（全部有单测；全量 1023 passed / 24 skipped）

1. **抓取手 ≠ 可行性模型的手**（“能否放下取决于怎么抓的”的根因）：R1Pro `arm_names=[left,right]`，`eef_pose()` 用左手，`symbolic_grasp` 用近的手，IK / 碰撞用右手 → `backend._modelled_arm()` 统一为 kinematics 的手。
2. **拿着东西不能开抽屉**（真实约束，持物撞抽屉面板）→ oracle 先开目标容器再抓取；T03 的 L1 指令写明顺序，L2/L3 要模型推断。
3. **抽屉内部放置**：区域改用原生 fillable 体积（链接 AABB 底面低于抽屉底板，叉子穿底）；OPEN 打开全部抽屉时选最上层；候选点按持物尺寸内缩；`symbolic_place` 按原生包围盒放置。
4. **斜向放进抽屉**（用户批准）：放进打开的抽屉时，工具从拉出一侧斜向进去（45/30/60/0°），放下时保持接近姿态。只改 inside 路径。
5. **长物体挂在指尖前方**：叉子挂在 EEF 原点会和夹爪 / 相机连杆重叠，把底盘顶起 0.2 m → 最大半尺寸 > 8 cm 的物体挂在指尖前方；杯子、苹果不变。
6. 新工具：`scripts/diagnose_vla_feasibility.py`（逐位姿碰撞对、`--execute-place`）、`scripts/probe_grasp_lift.py`。

## 3. 注意事项

- **不要在仿真运行中改 USD 碰撞过滤**（`add_filtered_collision_pair`）：会让 PhysX 张量视图失效，抓取直接报错（已试过、已撤回）。
- 每个变体是独立子进程，启动时加载代码；**改代码时正在跑的变体用的是旧代码**，改完要核对 `report.json: started` 并重跑。
- 长任务用 `nohup … &`，用 PID 文件判断结束；不要用 `pkill -f <命令行片段>`（会杀掉自己的 shell）。
- GPU 24 GB 可以同时跑 2–3 个仿真（每个约 3–9 GB）。

## 4. 下一步建议

1. 队友接入 VLA / VLM 模型，跑 36 个变体的 VLA Task Success。
2. 斜向接近目前只用于抽屉内部；如果以后的表面放置也被遮挡（吊柜下的台面等），可以同样扩展。
3. 真正隔门的多房间 S3 需要换场景或缩小底盘（本场景门宽 0.75 m < 底盘）。


---

# 十八、进度交接（2026-10-26 中午，最新）

## 0. 当前目标与用户决定（2026-10-25/26）

- **目标：约 1000 个能跑通的任务 = 约 111 个基础任务 × L1–L3 × S1–S3（每格约 111 个）**，L/S 定义沿用 v1（`assets/tasks/vla_difficulty_v1/levels.yaml`）。v1 的 36 个变体保留不动。
- 用户决定：
  1. “1000 个”按 **每个 L×S 变体算一个**；
  2. “能跑通” = **自动验证 + 人工抽查**（抽查尚未做）；
  3. S3 的门太窄 → **用队友的拓宽门洞**（`build/behavior/clearance`）；
  4. **用多个 BEHAVIOR 场景**；
  5. **优先复用 BEHAVIOR 官方任务和资产，只有在实在贫瘠时才自己造任务**。
- 背景（10-25 已查清，不必重查）：
  - 队友 10-19～10-24 的工作记在 `HANDOFF.md` 顶部（按时间倒序）：接本地 VLM（Qwen3-VL / InternVL / Molmo2 等，2–4B NF4），v1 的 36 变体上 135 回合 0 成功；随后改 agent 协议 v2→v4.7；10-22 转向官方任务；10-24 做官方任务 D/C 难度分级、人工试玩、拓宽门洞。
  - VLM 0 成功的原因诊断见 `runs/vlm_diag_20261025/`：模型照抄提示里的示例坐标 (0.5, 0.5)、提前 REPORT_DONE、刀在 RGB 里几乎不可见（v1 的 S1 门禁只数分割像素，是设计缺陷）。
  - 10-25 用当前代码重跑 v1 的 T01 L2_S1、T03 L2_S2，脚本化仍成功，见 `runs/vla_recheck_20261025/`。

## 1. 现状（2026-10-26 11:53）

| 项 | 数字 |
|---|---|
| 基础任务 | **102 个**：上层 `U001–U026`、Beechwood `W001–W026`、酒店 `H001–H050` |
| 变体 | 918 个（`build/vla_difficulty_v2/`，manifest 里 306 个物理配置） |
| 已验证 | 116 个物理配置：**通过 82 个，失败 34 个**，190 个待跑 |
| S1+S2+S3 全部通过的基础任务 | **11 个**（= 99 个可用变体） |

以 `python scripts/v2/run_batch.py --summarize-only` → `runs/vla_difficulty_v2/summary.md` 为准。
失败原因：S1 RGB 对比度不够 9 个、物体太大在 S2/S3 中露出 9 个、初始化时放置判定不成立 4 个、真实物体抓不到 4 个、放不下 3 个、路线 UNSAFE 2 个、仿真错误/超时 3 个。

**后台正在运行（都用 nohup 启动，PID 文件在 `runs/vla_difficulty_v2/`）**：
- `run_batch.py --jobs 3`（PID 1301811，`batch.pid`）：第一轮，剩约 190 个配置，按每个 3–9 分钟、3 路并行估算，大约 8 小时。它的待办列表里有已删除的旧 `U027–U045`，这些会很快报错，可以忽略（汇总只按当前 manifest 统计）。
- `queue_refine.sh`（PID 1311332，`queue_refine.pid`）：等第一轮结束后自动运行 `refine_loop.py --rounds 4 --jobs 3`。物体层面失败的任务保留布局和官方子目标、换一个物体，三个 S 全部重跑；失败过的模型写进 `assets/tasks/vla_difficulty_v2/failed_objects.txt`；布局层面失败的写进 `dropped_tasks.json`。日志：`runs/vla_difficulty_v2/refine.log`。
- GPU 24 GB，**每个仿真约 7.7–10 GB，最多同时 3 个**。不要再加第 4 个。

## 2. 设计（详见 `assets/tasks/vla_difficulty_v2/README.md`）

- **复用官方任务**：`scripts/v2/extract_official_pairs.py` 从 476 条官方活动定义中提取 1007 种 `(ontop|inside 物体 家具)` 子目标（`build/vla_difficulty_v2/official_pairs.json`）。每个基础任务对应一条官方子目标，物体类别、目标家具类别都来自官方定义，任务 YAML 的 `source` 写明活动名和子目标。L3 = 官方活动名作为意图（“I'm unpacking hobby equipment. The hardback book should end up on …”）。
- 任务分三类：`place_on`（ontop）、`place_in`（inside，目标容器要机器人自己 OPEN）、`fetch`（只取官方子目标的物体，goal = `holding`，**不是官方终态**，已在 YAML 中注明）。
- **资产**：场景、物体、家具全部来自 BEHAVIOR-1K。物体池 `build/vla_difficulty_v2/object_pool_official.json`（官方子目标里的 289 类、658 个模型，按尺寸和贴图颜色显著度筛过）；生成器再收紧为最长边 ≤ 16 cm、高 ≤ 16 cm、最短边 ≥ 5 cm、colourfulness ≥ 40，剩 212 个模型、109 类。
- **S 门禁**（`scripts/v2/run_variant.py`）：
  - S1：同房间，起点全景 **≥ 60 px 且 Lab 对比度 ΔE ≥ 15**（`scripts/v2/visibility.py`）；
  - S2：同房间，起点看不到；子类型 `container`（关着的抽屉，OPEN 后可见）或 `occluded`（被家具挡住的台面），记在 `s2_kind`；
  - S3：别的房间，起点看不到。
  - 另外 oracle 步数 ≤ 30（预算 60）。L1/L2/L3 只差指令，编译器会校验三者其余部分完全相同，所以每个 (任务, S) 只验证一个物理配置。
- **场景改动**（都有声明，没动原始资产）：4 个布局用队友的拓宽门洞，副本在 `build/vla_difficulty_v2/clearance/`；Rs_int 副本额外删了 honey 场景里的 5 把椅子和 1 张餐桌（最终 Rs_int 没出任务）。其他场景原样使用。

## 3. 流水线（`scripts/v2/`）

1. `survey_scene.py --scene S --start x y yaw --probe-on F`：一次加载，导出实体、台面、抽屉内部、MOVE 障碍物 → `runs/vla_difficulty_v2/_survey/S.json`
2. `analyze_survey.py S ...`：CPU 算可达地面、候选位置、各房间起点 → `S.candidates.json`。非拓宽场景会自动找**室内**最大连通区域。
3. `certify_locations.py --scene S [--only-goals]`：用探针杯子 `mug/dhnxww` 认证每个位置（摆放、各起点可见性、GRASP / PLACE 位姿搜索） → `runs/vla_difficulty_v2/_locations/S.json`（`--only-goals` 的结果写到 `S.goals.json`，生成器会自动合并）。
4. `make_phrases.py` → `assets/tasks/vla_difficulty_v2/phrases.yaml`：房间和家具的唯一叫法。不唯一时用地标（“next to the TV”）；无法自动区分的在脚本里 `ROOM_OVERRIDES` / `FURNITURE_OVERRIDES` 手写，例如上层卧室按“有地图 / 有墙面置物架 / 有屏风”区分。
5. `score_object_pool.py --official`，`extract_official_pairs.py`
6. `generate_tasks.py --scenes S --count N --prefix X` → `base_tasks/X###.yaml`（只替换同前缀的文件）
7. `compile_variants.py`（**不要加 `--only`**：它会让 manifest 里只剩那几个任务）
8. `run_batch.py --jobs 3` / `run_variant.py`；`refine_loop.py`

## 4. 场景勘察结论（重要，已实测）

- 19 个场景勘察下来，**原生 BEHAVIOR 场景对 R1Pro 大多太挤**：0.8 m 方形底盘加 3 cm 余量，在大多数户型里只能在一个房间活动，S3 做不出来。
- 实际能出任务的只有 3 个：
  - **house_double_floor_upper**：18 种布局。浴室柜 `jhymlr_0` 机器人打不开，所以上层没有 place_in。
  - **hotel_suite_large**：32 种布局，没有能抓的抽屉，S2 都是 occluded。
  - **Beechwood_0_int**：8 种布局。拓宽门洞后旧岛台被删、柜子平移，厨房深处进不去。
- 没出任务的原因：
  - Rs_int：太小，拓宽后厨房台面从客厅也看得见，没有 S3；
  - house_double_floor_lower、Pomaria_1/2：没有同房间又看不到的位置，或者家具无法唯一命名；
  - Benevolence / Ihlen / Merom / Wainscott / Beechwood_1：可达区域只有 1–19 m²；
  - house_single_floor：勘察 20 分钟超时；hotel_suite_small：仿真崩溃。
- 大多数低于 0.45 m 的抽屉都抓不到，认证时已跳过。

## 5. 待用户决定（用户 10-26 早上尚未回复）

102 个基础任务按约 50% 的配置通过率，**111 个三个 S 都通过的基础任务大概率达不到**。可选的办法：
1. 给更多场景拓宽门洞（队友的方法，需要人工审核墙体）；
2. 缩小碰撞底盘（场景能用很多，但和真实机器人尺寸不一致）；
3. 接受较小的规模（例如 60–80 个基础任务 × 9 格），如实报告每格数量。

其他可以做的：
- 每个房间多选几个起点：现在每间最多 2 个，增加起点能提高布局多样性，但要重新跑一遍可见性认证（GPU）。
- 每格人工抽查约 5 条（`bash scripts/play_behavior.sh --scenario build/vla_difficulty_v2/<id>/scenario.yaml`），抽查清单尚未生成。

## 6. 本轮改动的现有文件

- `src/rummagebench/evaluation/route_oracle.py`：位姿搜索日志多记一项 `feasible_poses`，只改日志，不影响行为。
- 其余全是新文件：`scripts/v2/*`、`assets/tasks/vla_difficulty_v2/*`、`build/vla_difficulty_v2/*`、`runs/vla_difficulty_v2/*`、`runs/vlm_diag_20261025/`、`runs/vla_recheck_20261025/`。
- **没动**队友的文件，也没动 v1；单测没有重跑。

## 7. 注意事项（新增）

- 探针杯子是灰白色的，**它的对比度不能代表真实物体**。S1 预筛只看像素数，对比度由 `run_variant.py` 用真实物体实测。
- 放置目标认证用 `set_open` 打开容器，**没测机器人能不能自己 OPEN**。place_in 的目标容器要以实际跑出来的结果为准：生成器里有 `OPEN_INFEASIBLE` 名单。
- 生成器的 S3 路线用栅格估算；精确路线以 oracle 为准。原来的 `plan_route` 太慢，跑了 20 分钟还没出结果。
- 不要用 `sleep` 轮询；后台等待用 PID 文件判断。


---

# 十九、进度交接（2026-10-26 下午，最新）

用户决定（10-26）：**给更多场景拓宽门洞**（第十八节第 5 条的选项 1）。

## 1. 新发现（重要）

- 没挂 `clearance_manifest` 的场景里，MOVE 门禁用的是**整面墙实体的 AABB**（`backend.navigation_obstacle_aabbs`，backend.py 约 1100 行），带门洞的墙会把门洞整个封死。之前 Ihlen / Wainscott / Beechwood_1 “只有 1–19 m² 可达”一部分是这个原因，一部分是家具挡路。挂上 manifest 后改为按每个碰撞体计算。
- 墙体碰撞体是若干凸包（8 顶点的盒子），贴地还有 0–0.1 m 的踢脚条，它在门洞处是断开的，不会堵门。

## 2. 新工具：`scripts/v2/propose_clearance.py`（CPU，不占 GPU）

- 从官方场景 json + 解密的墙 / 门 USD（`usd-core`，缓存在 `build/vla_difficulty_v2/_decrypted/`，仅本地用）读取原生碰撞体，产出与队友 manifest 同格式的 `wall_warps` + `removed_objects`（门板）。
- 相比队友的 `scripts/prepare_clearance.py`（未改动）：只处理室内门洞（两侧都在官方房间图里、缺口处没有别的墙）；挨着墙角时向较长的一侧偏移拓宽；1.4 m 不行退到 1.2 m（`clearance_audit.json` 的下限）；同一墙段相邻门洞的形变区间不重叠；形变会波及同实体另一条墙线时转人工审核。
- 同时用勘察的家具框 + 原生墙碰撞体做 0.8 m 底盘 flood fill，估算拓宽前后的可达房间。输出 `runs/vla_difficulty_v2/_clearance/<scene>.proposal.json` 和 `.png`。

## 3. 估算结果（CPU，GPU 未验证）

| 场景 | 拓宽 | 去门板 | 可达房间 前→后 |
|---|---|---|---|
| Wainscott_1 | 9 | 12 | 2 → 6（35.8 m²） |
| Beechwood_1 | 9 | 13 | 1 → 4（20.8 m²） |
| Ihlen_0 | 5 | 5 | 1 → 2（37.4 m²；左下客厅被 0.67 m 的结构通道挡住，非门洞） |
| Wainscott_0 | 6 | 6 | 4 → 6（每间 2–7 m²） |
| Pomaria_1 | 5 | 2 | 2 → 3（厨房 2.7 m²） |
| Ihlen_1 / Merom_1 / Pomaria_0/2 / Benevolence_2 | | | 基本无变化（家具挡路） |

Benevolence_0/1、Merom_0、house_single_floor、hotel_suite_small 没有勘察数据，无法估算。

## 4. 待做（需要 GPU）

- 队列脚本已写好**但未启动**：`runs/vla_difficulty_v2/_clearance/queue_scenes.sh`（启用 manifest → 勘察 → 队友 `check_clearance.py` 逐门洞实测 → `analyze_survey.py`；旧勘察备份到 `_survey/pre_clearance/`）。它在 GPU 上仿真数 < `MAX_SIMS` 时才开始每一步。
- 未启动的原因：GPU 被 v2 批量（3 路）占满，refine 循环会在批量结束后立即用 3 路接着跑；把批量降到 2 路的操作被权限拦下，需要用户决定。
- 之后：`certify_locations.py` → `make_phrases.py` → `generate_tasks.py --scenes ... --prefix <新前缀>` → `compile_variants.py` → `run_batch.py`。

## 5. 更新（2026-10-26 13:30）：方向改为指点导航 GOTO

- 用户决定：导航改为“在画面上点目标 → 2D A* 判断可达 → 传送底盘（关节不动）”；**GRASP/PLACE/OPEN 的可行性门禁保持不变**。
- 拓宽门洞队列已停（`queue_scenes.sh` 已跑完，`queue_certify.sh` 中止）；v2 批量 2 路继续跑，之后自动跑 refine。
- 规划文档：`docs/plan_v3_goto_navigation.md`（等待用户确认第 9 节的 4 个决定后再改代码）。

## 6. 更新（2026-10-26 21:30）：改用队友的 MOVE_TO-only 协议，重新验证 v2

- 用户决定：停 v2 批量；**只保留 MOVE_TO**（队友 18:17–19:13 改的协议：点画面里的地面 → A* → 直接到达，**朝向全程不变**，无 MOVE/TURN）；先交约 300 条可人工试玩的任务。
- 19:13 之后旧批量跑的 16 个结果是协议误判（oracle 的 TURN 被判无效），已移到 `runs/vla_difficulty_v2/_invalid_protocol_20261026/`。
- `evaluation/route_oracle.py`：新增 MOVE_TO 模式（`PUBLIC_SKILLS` 里没有 MOVE 时自动启用，旧 MOVE/TURN 路径保留）。固定朝向下搜索交互位姿 → 每步点当前全景里看得见的地面点：先试目标位姿 25 cm 内（在实际落点重新检查门禁），再沿路径找最远可见点，再按路程距离场绕行，离目标太近看不到脚下时先后退。每个子目标最多 8 跳。门禁未改。
- `scripts/v2/moveto_start_yaw.py`：因为朝向不能变，按旧验证记录里各子目标的可行朝向重选出生朝向（只改朝向），输出到 `build/vla_difficulty_v2_moveto/`（原 build 不动）。147 个旧通过配置 → 143 个可选；95 个所有子目标共享一个朝向。
- `run_batch.py` / `playable_list.py` 新增 `--build-dir` / `--runs-dir`。
- 正在跑：`runs/vla_difficulty_v2_moveto/queue.sh`（PID 文件 `queue.pid`，3 路），结束后自动生成 `runs/vla_difficulty_v2_moveto/playable.md`。前 7 个里通过 5 个，失败的 2 个都是 S1 的 RGB 对比度门禁：换朝向后全景里的可见性会变（机器人身体遮挡、光照）。

## 7. 更新（2026-10-26 23:00）：MOVE_TO 支持点家具 / 物体（AsgardBench 式自动站位）

- 用户决定：MOVE_TO 除了点地面，也可以点**看得见的家具或物体**，系统自动把机器人摆到它面前、面向它（点在哪一侧就站哪一侧）。墙和天花板不是有效目标。抓放门禁不变。
- 代码：`skills/move.py` 新增 `select_stand_pose` / `execute_move_to_entity`（环形候选位姿，要求底盘不碰撞、A* 可达 + 原地转身不碰撞、机器人正前方到目标之间除目标及其支撑家具外没有障碍）；`backend.navigate_oracle_pose`（转向时持物跟着刚体转）；`core/visual_session.py` 的 MOVE_TO 分支；反馈文字、两份模型提示词、MCP 描述、试玩界面说明同步更新。新单测 `tests/unit/test_move_to_entity.py`；`test_task_navigation.py` 里一条按新语义改了断言。
- oracle（`evaluation/route_oracle.py`）：目标可见就点它（GRASP 时也可点它所在的家具），用会话同一个站位函数预判站位并检查门禁；不可见就沿地面走过去（路径跳点 / 绕行 / 后退）。
- 因为点物体时会转向，出生朝向不再影响抓放，所以改回**原 build**（`build/vla_difficulty_v2`），S1 可见性的测量条件和原来一致。`build/vla_difficulty_v2_moveto` 和 `runs/vla_difficulty_v2_moveto`（固定朝向版，跑了 13 个）作废但保留。
- 正在跑：`runs/vla_difficulty_v2_moveto2/queue.sh`（3 路，147 个旧通过配置），跑完自动生成 `runs/vla_difficulty_v2_moveto2/playable.md`。
- 全量单测：1094 passed / 17 failed / 24 skipped。17 个失败都是队友删 MOVE/TURN 后过时的测试（还在发 MOVE/TURN、检查旧技能表、mock 缺 `clearance_manifest`），和本次改动无关，尚未修。
- 已知缺口：拿着东西后要去的家具在当前朝向下地面走不到时（H002 S2），oracle 还不会先点别的可见家具来调转朝向。

## 8. 更新（2026-10-27 02:30）

- `moveto2` 全量结果：147 个只通过 54 个（162 条变体）。主要原因：oracle 的地面接近点选在了目标**隔壁房间的墙另一侧**，于是来回打转超过 8 跳（61 个）；点小物体时站位贴得太近（0.45 m），抓不到（28 个）。
- 修复：点**可移动物体**时站在约 0.9 m 处（旧 MOVE/TURN 下可行抓取位姿距物体边缘的中位数是 0.96 m）；点家具时仍站在被点那一侧；“正前方无遮挡”检查里，墙和门永远不算支撑物；oracle 的地面接近点也必须满足这一检查；目标看得见但站位不可行时继续走近再试；物体前面没有可站位置时改走向它所在的家具。新单测：隔墙的架子不能产生站位。
- 4 个失败样例冒烟：3 个通过（H001 S3、H002 S1、H007 S2）。
- `run_batch.py`：每次启动仿真间隔 20 秒（同时启动 3 个仿真偶发 core dump）。
- 正在跑：`runs/vla_difficulty_v2_moveto3/queue.sh`（147 个配置，3 路），结束后自动生成 `runs/vla_difficulty_v2_moveto3/playable.md`。

## 9. 更新（2026-10-27 05:10）

- `moveto3` 第 1 波结束：147 个旧通过配置里 **117 个通过 = 351 条可试玩变体**，覆盖 66 个基础任务（hotel 222 条，house 129 条），全部只用 MOVE_TO 验证。清单：`runs/vla_difficulty_v2_moveto3/playable.md` / `playable.csv`。
- 02:56 加了“点别的家具来调转朝向”和更宽的地面接近点之后，通过率从 26/38 升到 91/109（83%）。
- 剩下的主要失败（18 个，全是 hotel S2/S3）：目标只能从浴室角落**斜着**抓到，当前朝向下找不到接近点，点别的家具也调不过来。修复（只改 oracle）：调转朝向失败后，退而走向场景里记录的交互锚点位置（`<entity>__GRASP_*`，评估器私有），再不行就走向不要求正前方无遮挡的环形点；最终站位仍由会话自己的点物体逻辑决定，门禁不变。
- 正在跑：第 2 波（`queue2.sh`，131 个 v2 从未通过、且不是门禁失败的配置），之后第 3 波（`queue3.sh`）用最新 oracle 重跑所有非门禁失败。两波结束后自动重新生成清单。
