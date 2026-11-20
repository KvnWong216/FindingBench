# 第二批本机 VLM 评估

用户确认将五个候选全部纳入。新增 Phi-4-multimodal、SmolVLM2-2.2B、Gemma-3-4B、MiniCPM-V-4.5-8B、LLaVA-OneVision-Qwen2-7B；Llama 仍按之前指示暂缓。所有权重 revision 固定在 `configs/models/local_vlm.json`。

## 运行顺序和验证边界

`runs/vlm_expanded_v1/suite.json` 是扩展队列状态，`runs/vlm_expansion_setup/downloads.json` 是下载状态。队列先等待 `runs/vlm_grid_v3/suite.json` 的原三模型批次结束，再确认 8091 可用。不会终止既有服务。

每个模型依次执行：权重加载 → 单图推理 → T01_knife_fetch__L2_S1 联调 → 36 条正式网格。联调任务失败但运行有效可以进入网格；服务、加载或仿真异常阻止该模型进入网格，其他模型继续。没有用 oracle 代替模型。

当前新增模型处于部署/排队阶段，不代表已通过 GPU 推理或闭环验证。Gemma 已实际遇到 GatedRepoError，需要用户在 Google 模型页接受许可，再在本机 `hf auth login`；不得把 token 写进代码或聊天。授权前保留为 blocked，不计入失败率。授权后重新下载并用新输出目录单独运行 Gemma。

每个模型 36 回合，seed 0、60 步、贪心解码、512 输出 token、public-skills-v2 公共提示词。全部五个可运行时新增 180 个正式回合；当前无授权的 Gemma 不运行。联调单独保存在 pilots，不纳入正式网格成功率。此网格为 4 任务 × 3 语言级别 × 3 空间级别，使用 BEHAVIOR 资产，不是 BEHAVIOR-1K 全集。

## 依赖与模型差异

- `ygbench-vlm-extended`：Transformers 4.57.1，SmolVLM2、Gemma、MiniCPM、LLaVA。
- `ygbench-vlm-phi4`：Transformers 4.48.2 / PEFT 0.13.2，采用 Phi 官方示例的模型接口及图像提示模板，attention=eager。
- 不修改当前三模型使用的 `ygbench-vlm` 环境或冻结脚本。扩展脚本位于 scripts/serve_extended_vlm.py、extended_vlm_adapters.py、run_extended_vlm_suite.py。
- MiniCPM 使用官方 chat 接口，enable_thinking=False、num_beams=1、repetition_penalty=1；覆盖其非采样模式默认 beam search。输入和输出 token 数在底层 generate 获取。
- 使用 NF4，Phi 的多模态输入和 LoRA 层、MiniCPM 的视觉编码器及 resampler 保留浮点。显式排除规则、模型版本、适配代码摘要和实际包版本记录在服务元数据与每条报告中。各模型实际视觉 token 数和浮点保留层不同，不能声称计算预算完全相同。

## 重现命令

```bash
cd /home/ubuntu/ygwang/FindingBench
./scripts/environment/setup_extended_vlm.sh
/home/ubuntu/miniforge3/envs/ygbench-vlm-extended/bin/python scripts/prepare_extended_vlm.py
./scripts/evaluate_extended_vlm_suite.sh --all --out-root runs/vlm_expanded_new
```

已有输出目录不会覆盖。可用 `--models gemma3-4b` 选择单一模型；`--after-suite` 指定需要等待的前序批次状态。人工停止扩展队列可向 suite.json 中的 PID 发送 SIGTERM，仅清理该队列启动的子进程。

模型接口依据：[Phi 官方模型卡](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)、[MiniCPM 官方模型卡](https://huggingface.co/openbmb/MiniCPM-V-4_5)、[SmolVLM2 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)、[LLaVA HF 模型卡](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)、[Gemma 模型卡](https://huggingface.co/google/gemma-3-4b-it)。

## 本次启动记录

扩展队列 PID 349345，下载进程 PID 348257。两套新增环境 `pip check` 通过；Phi、SmolVLM2、LLaVA 完成 2048×1024 图像预处理验证，MiniCPM 完成处理器加载。接入及队列测试 9 passed。当前状态是等待原三模型批次，尚未使用 GPU 验证新增模型的生成和任务闭环；实时进度以运行目录为准。

### Gemma 许可更新

用户完成许可填写后，本机已验证固定 revision 的配置和权重文件访问成功，见 `runs/vlm_expansion_setup/gemma-access.json`。原下载进程仍在运行，Gemma 排在 MiniCPM 之后，将直接使用当前已激活的 HF 凭据下载；无需另建批次或修改已冻结配置。此前待授权说明为历史状态。
