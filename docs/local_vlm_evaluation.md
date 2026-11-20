# 本机多 VLM 评估（2026-10-19）

## 当前基线

实查 `runs/vla_difficulty_v1/*/report.json`：36 个变体，初始化/空间门禁 36/36、脚本化 SUCCESS 36/36、model_used 0/36。这证明任务可执行，不是模型成功率。当前实验是基于 BEHAVIOR 资产构造的 T01–T04 × L1–L3 × S1–S3 网格；不等同于原始官方 191 条刚体定义。官方采购任务暂不进入本轮网格。

最近执行器修复包括：抓取手和可行性模型统一、长物体指尖挂载、抽屉原生 fillable 体积、顶层抽屉选择、内部斜向放置。详见 `vla_difficulty_v1_report.md`。本次模型接入不修改这些任务与技能。

## 固定模型

`configs/models/local_vlm.json` 固定官方仓库及完整 revision SHA；模型调用在独立 `ygbench-vlm` 环境，模拟器仍用 `ygbench`，经本机 HTTP 通信。

| profile | 官方模型 | 说明 |
|---|---|---|
| qwen3-vl-4b | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 首个推理验证对象 |
| internvl3.5-4b | [OpenGVLab/InternVL3_5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF) | 官方 Transformers 格式 |
| llama3.2-11b | [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | gated，需账号获得权重访问权限；参数量不同 |
| molmo2-4b | [allenai/Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B) | 用户确认是通用 Molmo2，非 MolmoAct2-Think |

单张 RTX 3090 24 GB 与模拟器共享显存。默认 NF4（double quant、BF16 compute），逐个模型运行；记录实际显存，不预先保证所有模型都能共存。`--quantization none` 可做 BF16 对照，但需足够显存。NF4 结果应独立标注，不作为官方 BF16 成绩。4B 与 11B 的差异也不能全部归因于架构。

## 启动

```bash
cd /home/ubuntu/ygwang/FindingBench
./scripts/environment/setup_local_vlm.sh
./scripts/serve_vlm.sh --model qwen3-vl-4b
```

服务仅监听 `127.0.0.1:8091`，权重缺失时下载固定 revision。首次加载需等待日志 `Local VLM ready`。Llama 需先在 HF 模型页获批，再在推理环境用 `hf auth login` 登录；不要在代码或聊天里填写 token。停止当前服务后，更换 `--model` 启动其他 profile。Molmo2 启用固定 revision 的官方 remote code。

另一个终端（`evaluate_vlm.sh` 自动激活 `ygbench`，包括编译器和资产路径；不要只调用环境里的 Python 而省略激活）：

```bash
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate ygbench
cd /home/ubuntu/ygwang/FindingBench
# 仅在 build 变体尚未生成时运行：
# python scripts/compile_vla_variants.py
./scripts/evaluate_vlm.sh --model qwen3-vl-4b \
  --variants T01_knife_fetch__L2_S1 --out-root runs/vlm_pilot
```

单条联调通过后，另选新输出目录跑完整网格：

```bash
./scripts/evaluate_vlm.sh --model qwen3-vl-4b --all \
  --seed 0 --out-root runs/vlm_grid
```

同一输出目录的既有回合不会覆盖。每个变体单独启动模拟器进程，默认最多 7200 秒；服务不随回合重启。`--summarize-only` 重新汇总已有结果。其他模型换 profile，建议正式报告使用多个 seed，并报告各 seed；当前未自动跨 seed 合并。

## 比较约定

- 原始公共 RGB 不额外缩小/裁剪；当前网格初始全景 2048×1024，VIEW 为场景规定的透视取景。各模型使用自己的官方视觉预处理，记录输入 token 数；不能宣称视觉 token 预算完全相等。
- 所有模型收到同一提示词、当前帧、公共任务文字、当前 frame_id、步数、固定技能列表，以及最多 60 条历史动作/反馈。当前不传历史图像，也不追加模型推理文本作为记忆。
- 贪心解码，每次最多 512 个输出 token。终止 token 未出现而用满预算会记 `output_budget_reached`。不把此设置当作专门的 Thinking 模型比较。
- 只接受模型选择的动作，不自动纠正点坐标、frame_id，不调用 oracle 或泄露锚点/语义 ID。格式错误交给 session 返回 INVALID_ACTION，照常消耗步数。多个候选动作不会代选。
- 推理服务错误、超时、OOM 和仿真异常标记无效运行；统计 valid/invalid 数，无效运行不进任务成功率分母。合法但错误的决策、提前报告、耗尽 60 步属于有效失败。
- 成功仍由 session 的 REPORT_DONE + goal 判定。总成功率、L/S 分组、L×S 单元格分别汇总；每格仅 4 个任务，不宜据此做强统计结论。空间 S2 有时比 S1 路径短，需对照已有 oracle 步数。

## 产物与验证边界

`runs/<输出根>/<model>/seed_<seed>/<variant>/` 包含：

- `report.json`：模型 SHA、量化/包版本、任务 SHA、提示词 SHA、有效性、成功状态、反馈计数与推理耗时/token 总量。
- `model_calls.jsonl`：发送的文本提示、帧 ID、图片摘要、模型原始输出及推理统计。
- `episode/events.jsonl`：评估器私有完整轨迹，不应发送给模型。
- `episode/step_*.png`：初始图与每一步公共图像。
- 上层 `summary.json`：总体与 L/S 聚合，明确有效/无效分母。

四个模型的配置已准备，不代表四个模型都已实际加载或完成闭环评估。真实验证状态以各运行目录的 report 为准，不能把模拟服务的单元测试当作模型结果。

## 本机实测记录（2026-10-19）

- 独立 `ygbench-vlm` 已安装，`pip check` 通过；依赖快照见 `configs/environment/ygbench-vlm-freeze.txt`。
- 接入层及相关回归：29 passed。
- Qwen3-VL-4B-Instruct / NF4：权重已下载并实际加载。单图推理 2.55 秒、峰值 3.57 GB（十进制）；证据 `runs/vlm_preflight/qwen3-vl-4b/`。
- 首个真实闭环 `T01_knife_fetch__L2_S1` / seed 0：`run_valid=true`，5 步后 `FAIL_FALSE_COMPLETION`。动作为 VIEW、三次带多余 action/feedback 包装的无效输出、最后直接 REPORT_DONE。没有修正模型的包装或代替其选择动作。模型调用合计 6.90 秒，回合执行 15.56 秒（不含场景初始化）。这是单条联调失败，不能当成 36 条网格的整体成功率。证据 `runs/vlm_pilot_activated/qwen3-vl-4b/seed_0/`。
- 初次未激活 conda 的尝试因缺少 C++ 编译器失败，保留在 `runs/vlm_pilot/` 并标记无效；后续使用 `evaluate_vlm.sh` 自动激活。
- InternVL3.5-4B-HF 已完成真实单条闭环：60 步 `FAIL_MAX_STEPS`，`run_valid=true`；反馈 EXECUTED 46、UNSAFE 8、INVALID_ACTION 6。证据 `runs/vlm_additional_pilots_v2/internvl3.5-4b/seed_0/`。
- Molmo2-4B 首次生成遇到量化类型错误，保留为无效运行，不计入成功率分母。修复后单图推理通过，输出合法 VIEW，见 `runs/vlm_preflight/molmo2-fixed/`；真实闭环复测有效：60 步 `FAIL_MAX_STEPS`，EXECUTED 57、UNSAFE 3，60 次推理合计 85.34 秒；证据 `runs/vlm_molmo_fixed_v2/`。
- Llama 3.2 Vision 实测返回 gated repo / 401；需要账号获批并在本机登录，尚未加载。

## 三模型串行批跑（public-skills-v2）

用户确认暂不配置 Llama，先运行 Qwen、InternVL、Molmo2。新版公共提示明确：返回动作本身，顶层为 `skill`，不能复述历史中的 `action` / `feedback` 包装。解析器没有代替模型拆包、修坐标或生成动作。提示版本和哈希写入每条报告；首轮旧提示结果保留，不混合统计。

Qwen 的新版单条联调：3 步 `FAIL_FALSE_COMPLETION`，依次 VIEW、MOVE（UNSAFE）、REPORT_DONE。均无动作包装错误；该结果不代表网格整体成功率，见 `runs/vlm_pilot_v2/`。

```bash
cd /home/ubuntu/ygwang/FindingBench
./scripts/evaluate_vlm_suite.sh --all --out-root runs/vlm_grid_v3
```

默认模型顺序：qwen3-vl-4b、internvl3.5-4b、molmo2-4b。共 3 × 36 = 108 回合、seed 0；每个回合保留原始 60 步预算。该脚本自动管理推理服务和模拟器进程，模型按顺序运行，结束后释放它启动的进程。已有服务占用 8091 时直接报错，不接管或终止它；先关闭自己启动的旧服务，或使用 `--port 8092`。

`--variants T01_knife_fetch__L2_S1` 可只对三个模型跑同一条联调。`suite.json` 记录整体状态、模型加载/评估/完成情况与进程 PID；各模型的 `seed_0/summary.json` 随回合更新。Ctrl+C / SIGTERM 会停止该批次的自有子进程。代码发生变化会中止后续运行，避免混合版本。

### Molmo2 量化兼容

固定版本的 Molmo2 从 `patch_embedding.weight.dtype` 推导视觉激活类型。该层若被 NF4 打包，图像会转为 uint8 并触发 LayerNorm 错误。因此仅该模型的 `patch_embedding` 保留 BF16，其他可量化线性层使用 NF4；服务和回合报告记录 `quantization_skip_modules`。没有修改模型缓存代码或输入图像。首次错误报告保留在 `runs/vlm_additional_pilots_v2/molmo2-4b/`。量化配置变动后的正式批跑使用新目录，旧联调仅用于检查接入。

### 正式批跑启动记录

2026-10-19 启动 `runs/vlm_grid_v3/`：Qwen → InternVL → Molmo2，36 变体/模型，seed 0，共 108 回合。控制进程 PID 343673，后台日志 `runs/vlm_grid_v3.launcher.log`。启动并不表示完成，实时状态以 `runs/vlm_grid_v3/suite.json` 和各模型 `seed_0/summary.json` 为准。当前网格为 4 个任务 × 3 语言级别 × 3 空间级别的编译变体，使用 BEHAVIOR 资产；不是 BEHAVIOR-1K 全部任务成绩。

批跑前的一次启动保留在 `runs/vlm_grid_v2/`：Qwen 加载异常退出后已中止该批次。当前 `vlm_grid_v3` 恢复 Qwen/InternVL 的默认量化排除规则；`quantization_skip_modules` 表示显式配置（空列表表示沿用库默认规则）。Molmo2 单独排除 patch_embedding。当前已确认 Qwen 服务就绪并开始首个仿真回合。
