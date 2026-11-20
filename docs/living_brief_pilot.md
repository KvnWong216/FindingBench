# 生活情境提案的小规模对照

后续连续动力学诊断与验收采样修复见 [background_dynamics_audit.md](background_dynamics_audit.md)。
下文保留首轮试验的原始结果，不用后续代码重新解释成通过。

本入口在原有 `prepare-home` 上增加可选 `living_brief`，随后仍使用真实
InfiniGen 求解器、BEHAVIOR 原生资产、OmniGibson 与既有技能判定。
此次为一套真实房屋、一个种子的机制试验，不估计大规模通过率。

## 输入与边界

`configs/layouts/living_brief_pilot/` 保存均匀、备餐、餐后三种配置。
两份中文描述与结构化伴随约束由 Codex 助手编写；没有调用外部语言模型
服务，也没有实现任意自然语言解析器。后续语言专家应输出同一严格 schema。
描述和 rationale 供审阅，只有结构化字段参与求解，不能把文字里的承诺
当作已经落地的关系。本次仅表达各表面的补充数量、偏好密度与 on-top 关系；
物品类别与活动的精细搭配、物品成组摆放仍未实现。

固定条件：`Benevolence_1_int`、四种原生碗碟/调味用品资产、14 件新增物品、
种子 127、800 次 Gen 迭代；完整原生房屋保留。三个方案共用同一个固定任务，
用于执行诊断。语言编译器只接收背景上下文与 brief，不接收任务、目标位置、
参考轨迹；strict schema 拒绝额外隐藏字段。自由文本本身不构成防泄漏证明，
今后的自动语言采样端也必须隔离隐藏任务信息。

所有可补充表面都必须在 brief 中明确出现。允许 minimum=0，禁止因为
难以实现而静默删掉表面。已有占用导致的保守几何排除仍有原先的记录，
并不意味着删掉原生实体或搜索位置。非目标房间不会被裁掉。

## 数量分配与几何求解

初版把全局总数和各表面的数量区间一起交给 Gen，种子 127 下均匀方案成功，
两个语言方案失败：备餐违反总数和餐桌约束，餐后违反餐桌约束。
原始失败保存在 `build/behavior_language_pilot/v1/`，没有覆盖或放宽后通过。
一次失败不证明区间约束不可解，只说明该有限预算的尝试未解出。

修正版先精确求解一个有限整数分配问题：

\[
\min_{n_1,\ldots,n_k}\sum_i |n_i-p_i|,
\quad l_i\le n_i\le u_i,\quad\sum_i n_i=N.
\]

动态规划保存最小代价与最优后缀方案数，按种子在最优分配中均匀采样。
随后复用已有每表面精确计数约束，让 Gen 求几何位置和 soft fullness。
不再奖励物品互相远离，因此允许摆得下的杂乱，但没有声称已实现聚簇目标。
数量分配的最优性仅适用于上面的整数问题，不证明几何可行或搜索成本最优。
若所选数量的几何求解失败，保留失败，不自动放宽或换分配掩盖失败。

结果验证独立检查实际物品总数、每表面数量、范围、完整约束证据、资产
白名单和实例名冲突，并重新计算已绑定的数量分配。描述、分配、原生状态、
资产记录进入请求哈希，来源代码与每次尝试保存在输出目录。

## 已运行 CPU 对照

修正版输出：`build/behavior_language_pilot/v2/cpu_summary.json`。

| 方案 | 厨房新增 | 餐区新增 | 走廊新增 | CPU 候选 | 本次耗时（秒） |
|---|---:|---:|---:|---|---:|
| 均匀 | 8 | 4 | 2 | 成功 | 29.263 |
| 备餐 | 10 | 4 | 0 | 成功 | 18.196 |
| 餐后 | 6 | 8 | 0 | 成功 | 24.654 |

同时运行的墙钟时间不是稳定的性能估计。v1/v2 均匀组的请求与求解结果
哈希完全一致，所以没有把重复运行算成额外场景。最终有 3 个不同候选背景，
1 套原生房屋，新增结构房型为 0。并非 3 条已认证评测数据。

## 已运行 GPU 对照

2026-10-15，仅使用物理 GPU 4、5，三组均完成真实加载和背景检查。
汇总：`build/behavior_language_pilot/pilot_summary.json`，其结果与阶段文件
已核对 `process.json` 中的 SHA256。运行时源码快照保存在同目录的
`runtime_sources/`；许可资产仍仅由原安装路径读取。

| 方案 | 背景检查目录 | 支撑关系 | 最后窗口角速度超限实体数 | 结果 |
|---|---|---|---:|---|
| 均匀 | `v1/uniform/gpu4` | 14/14 成立 | 2 | 拒绝 |
| 备餐 | `v2/cooking/gpu5` | 14/14 成立 | 4 | 拒绝 |
| 餐后 | `v2/meal/gpu4` | 14/14 成立 | 2 | 拒绝 |

三组均为 `BACKGROUND_PERSISTENT_DYNAMICS`。最后一个测量窗口里，
失败实体的角速度为 0.02036–0.06234 rad/s，超过现有 0.02 rad/s 门槛；
线速度合格，位置漂移约 0.7–13.9 微米。三个有界测量窗口均未获得全体通过，
没有因速度单独超限而重摆，也没有放宽阈值。这不能直接解释为碰撞严重或
物品倾倒；需要进一步用速度时间序列区分持续接触微动、数值噪声与取样问题。
子进程在 SDK 清理时返回 0，但父进程根据失败证据正确返回 1，没有误报通过。

三组均在背景阶段停止，目标插入后检查和公开技能链没有执行，也没有取得
可用于自然程度比较的场景截图。物理通过数 0/3，正式评测数据 0。
因此本次只证明结构化情境能接入真实 Gen 并改变空间物品分布；没有证明
语言方案自然程度更高、物理通过率更高或生成成本更低。

CPU 回归（包含旧的精确数量验证）：827 passed、44 skipped、13 deselected。
新增测试覆盖未知/缺失表面、额外隐藏字段、数量类型与边界、非法总数、
NaN、允许空表面时仍需完整证据、资产白名单、分配绑定和离散最优性。

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  -m pytest -m 'not sim' -q
```

## 复现

输出目录必须不存在，失败尝试也必须保留：

```bash
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/pilot_living_briefs.py \
  --out build/behavior_language_pilot/reproduce --seed 127

# 单独使用自定义的严格结构化伴随描述：
/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  scripts/tasks/behavior_pipeline.py prepare-home \
  --config configs/layouts/living_brief_pilot/cooking.yaml \
  --seed 127 --out build/behavior_language_pilot/cooking_reproduce

# 仅 GPU 4 或 5；真实物理验收和公开技能执行诊断。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/behavior_pipeline.py gpu-smoke --realize-background --gpu 5 \
  --scenario build/behavior_language_pilot/cooking_reproduce/scenario.yaml \
  --out build/behavior_language_pilot/cooking_reproduce/gpu5
```

GPU 阶段必须分别查看 `background_realization.json`、
`background_after_target.json`、`result.json` 和 `process.json`。
短时支撑与稳定性、目标插入后背景不扰动，以及公开 OPEN/GRASP/REPORT_DONE
通过，只构成已有 known-target 执行诊断；它使用评测端已知目标的视角候选，
不能作为搜索导航、完整任务族、门扫掠、全部原生状态不变或难度认证。
机器人仍为 R1Pro head camera，正式 evaluation-ready 计数保持 0。
