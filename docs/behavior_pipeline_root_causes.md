# 批量管线根因检查（2026-10-15）

本次检查复用 250 次 CPU 提案和原有 GPU 记录，按失败阶段归因；没有继续扩大逐条 GPU 抽查。
原有 200 条候选及其失败记录保持原样。修复后另外保存同输入对照，正式验收数仍为 0。

## 1. 支撑提案约束与原生布置冲突：已定位、修复并做真实 Gen 对照

250 次提案中 206 次生成候选、44 次失败。Ihlen 占 42 次，全部包含
`daily_items_countertop_tpuwys_6` 约束失败。原来的 `native_context` 仅按类别、尺寸、朝向
判定支撑面；`solve` 对每一个这样的表面强制新增 `per_surface` 件用品。

实际输入里 `countertop_tpuwys_6` 上方约 1.9 cm 处，已有 `countertop_tpuwys_5`
覆盖整个支撑包围盒。下层和上层同时被当作暴露台面，造成重复、冲突的摆放要求。
两个 `breakfast_table_skczfi` 小桌上还已有原生台灯；当前资产池的用品在保守包围盒模型中
没有剩余摆放空间。它们不是缺少生活物品的空桌。

修复位于 `indoor_background.py`：

- `exclude_covered_support_envelopes` 排除被另一原生支撑包围盒完全覆盖的下层提案面。
  部分覆盖不排除，所有物体及障碍包围盒仍保留。
- `qualify_supports_for_pool` 检查当前日用品资产池在原生障碍包围盒间的剩余空间。
  使用支撑局部坐标、真实 metadata 尺寸、四个声明的朝向及 5 mm 提案余量；
  只有存在原生障碍、且没有该模型下的候选位置时才不新增用品。
- `skipped_supports` 记录覆盖体、阻挡体、测试资产、朝向和理由。
  这表示保守提案模型不支持新增物品，**不表示已经证明真实表面全满或发生物理碰撞**。
  没有删除原生物体、搜索柜体或目标位置，也没有关闭物理验收。

同输入对照保留原 seed、资产池、800 求解步数、目标配置及原生状态：

| 输入 | 只修复重复支撑 | 加入原生占用检查 | 原生物体 / 柜体候选 |
| --- | --- | --- | --- |
| home_0002，per_surface=3 | 仍失败：两个已有台灯的小桌 | Gen 成功，7 处补充 21 件用品 | 175 / 11，保留 |
| home_0020，per_surface=1 | 仍失败：两个已有台灯的小桌 | Gen 成功，7 处补充 7 件用品 | 175 / 11，保留 |

静态归因发现 44 次失败都包含上述两类规则会排除的支撑约束，但仅这两组完成了真实重放；
**没有宣称另外 42 组已经通过重放或物理验收**。原失败尝试累计 worker 时长 5142.9 秒，
反复换 seed 已消耗约 85.7 worker 分钟，不能解决这种输入约束冲突。

## 2. 批量入口绕过物理实现阶段：已修复接线，CPU 验证

原始 12 次 GPU 抽查全部没有 `background_realizer_sha256`：`batch-probe` 没有传入
`--realize-background`，直接加载 Gen 包围盒位置后检查。8 次停在背景 OnTop 检查，
3 次耗尽技能诊断，1 次完成技能链。包围盒求解成功无法保证真实碰撞、支撑和稳定性。

`indoor_batch_probe.py` 现在默认启用原生物理实现，并写到独立的
`gpu_realized_pipeline_v1` 和 `gpu_realized_summary.json`。
`--background-mode direct-proposal` 保留旧诊断模式。已有失败不会被覆盖或冒充修复后的通过记录。
回归测试实际检查子命令参数和输出目录；本次没有为这项接线重新跑 GPU 批次。

## 3. 重摆策略把动态未稳定当成位置错误：已修复，CPU 验证

原 `indoor_realization.py` 在一次短窗口测量失败后立即重新摆放，即使支撑成立、位姿没有
显著漂移，仅瞬时角速度超过阈值。这会改动原本合理的位置，并可能扰动邻近物体。
原 home_0001 的插入后失败仅为角速度 0.0266 rad/s，位置变化约 0.0000037 m；
该证据不能支持“目标把背景移走了”的因果结论。

现在每轮先有限等待至多三个测量窗口；速度单独超限不触发重摆，持续超限仍拒绝。
所有原有阈值不变。插入后的报告分别记录支撑丢失、位姿变化、线速度和角速度失败；
纯速度失败使用 `BACKGROUND_UNSTABLE_AFTER_TARGET`，不再声称已经测得位姿扰动。
新策略已通过有界等待、不无故重摆、持续失败拒绝的 CPU 测试，GPU 效果尚未测量。

## 4. 目标位置与技能支持尚未完成批量校准：尚未解决

生成器仍从 metadata 槽位采样，未预先建立“本体 × 柜体/腔体 × 目标资产”的真实可用集合。
`behavior_pipeline.smoke` 只检查柜前 25 个启发式姿态，失败不能证明该位置无解。
原 3 个耗尽诊断的样本累计出现 51 次底座碰撞、21 次开柜能力失败、2 次目标不可见、
1 次开柜不安全；这里需要区分姿态参考器不足、真实技能不支持及目标遮挡。
本次没有删除失败槽位、修改动作接口、放宽抓取/开柜判定或把参考器成本当搜索难度。

吞吐量也未解决：原 12 次诊断的进程时长中位数为 236.5 秒，每个任务都冷启动完整仿真。
目前批量路径没有完成背景冻结恢复、同背景家族验证及发布验收的接通。
当前 Gen 接入是固定原生房屋中的包围盒日用品约束求解，不是完整的 Gen 房屋规则生成器；
新增结构房型数仍为 0。这些是明确的剩余接入边界，增加 seed 数量不能替代。

## 复现检查

```bash
PY=/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python
$PY scripts/tasks/audit_behavior_pipeline.py \
  --batch build/behavior_batch/randomized200_v1 \
  --replays build/behavior_pipeline_audit/occupied_support_v2
$PY -m pytest -m 'not sim' -q
```

汇总写入原批次的 `pipeline_audit.json`，包含阶段计数、日志哈希和受控对照输入检查。
本轮 CPU 回归：817 passed、44 skipped、13 deselected。跳过项不算验收通过。
对照产物在 `build/behavior_pipeline_audit/occupied_support_v2`，第一步仍失败的对照在
`covered_support_v1`。这些目录均为本机产物，不上传 BEHAVIOR 资产或状态。
