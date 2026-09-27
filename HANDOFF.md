# RummageBench 交接文档

> 交接日期：2026-09-26
> 状态：MVP 已完成并通过全部验收（场景 knife_search_001 端到端跑通）
> 主机：10.20.37.124（8x RTX 5880 Ada 49G，/data1 约 8.8T 可用）

## 1. 这是什么

RummageBench：基于 BEHAVIOR-1K + OmniGibson 的长时程具身物体搜索基准。
智能体只做语义决策（NAV / OPEN / GRASP），后端用符号/特权机制保证语义后果
可靠实现；基准以"语义规划步数"计时，不考核导航、IK、抓取控制等底层执行。

**验收结论（全部通过）**：

| 项 | 结果 |
|---|---|
| 场景编译（knife_search_001） | BUILD OK，谓词全部满足 |
| scripted（oracle 基线） | SUCCESS，5 步 |
| wrong_object | FAIL_WRONG_TARGET，5 步 |
| timeout | FAIL_MAX_STEPS，17 步 |
| unsafe | FAIL_UNSAFE_ACTION，2 步 |
| 复位确定性 | 语义严格复原，位姿 <2cm |
| 单元测试（无模拟器） | 24/24 |
| 集成测试（真实模拟器） | 11/11 |
| MCP 验收 | 与 Python API 结果一致 |

## 2. 目录与环境位置

| 内容 | 路径 |
|---|---|
| 本项目 | `/data1/ygwang/codes/Find-Bench` |
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
cd /data1/ygwang/codes/Find-Bench
export CUDA_VISIBLE_DEVICES=0        # 挑一块空闲 GPU（nvidia-smi 查看）
export PYTHONPATH=/data1/ygwang/codes/Find-Bench/src

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

## 6. 关键产物位置

```
Find-Bench/runs/setup_report.json                       Phase 1 环境验证
Find-Bench/runs/acceptance_report.json                  总验收报告
Find-Bench/runs/acceptance_knife_search_001_*/          四条轨迹（events.jsonl+summary+逐步截图）
Find-Bench/runs/anchors_gpu2.json                       Phase 2 锚点实测数据
Find-Bench/build/scenarios/knife_search_001/            编译产物（snapshot/preview/报告）
Find-Bench/logs/                                        安装/运行日志
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
pip install -e /data1/ygwang/codes/Find-Bench --no-deps
pip install "mcp<2" pytest                         # mcp 必须 <2（2.x 改了 API）
```
