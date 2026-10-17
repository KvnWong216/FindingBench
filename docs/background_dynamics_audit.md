# 背景稳定性诊断

承接 `living_brief_pilot.md` 中三组背景被角速度门槛拒绝的结果。
只使用 GPU 4、5；资产、布置、质量、摩擦、阻尼、睡眠设置及验收阈值保持原值。
诊断过程中只推进正常仿真并读取状态，不清零速度或强制休眠。

## 连续测量

新增 `gpu-smoke --realize-background --diagnose-background`：正常门槛失败后，
保留失败结论，继续推进 360 个正常模拟步，保存初始点加后续 360 点。
本机每步 1/30 秒，所以记录约 12 秒；实际仿真时间也逐点保存并检查单调。

记录：根 link 位姿、线/角速度、物理句柄有效性、PhysX 原始速度、对象休眠状态、
原有 OnTop 判定、根 link 与实体求解参数。差分速度仅为诊断指标，不替换验收速度。
多 link 物体的完整关节/各 link 稳定性仍未由此获得认证。

文件 `background_dynamics.jsonl` 是原始数据；`background_dynamics.json` 为汇总，
绑定原始数据和截图哈希，且被父进程 `process.json` 绑定。
异常诊断不得掩盖原门槛异常，任何诊断完成状态都不表示正式数据通过。

## 实际基线

`build/behavior_dynamics_audit/v1/{uniform,meal}` 使用先前同一真实房屋、种子 127、
每组 14 件新增物品。两组均取得 361 点、约 12 秒完整轨迹：

- 物理句柄全部有效，公开 getter 与 PhysX 原始速度逐点完全一致。
- 支撑始终成立，仍有反复角速度超限，因此不能归因于一次采样偶然或等候不足。
- 均匀组 `daily_012`（碗）超限 360/361 点，最大角速度约 0.08064 rad/s，
  位置漂移最大约 2.33 微米，姿态漂移约 0.00197°。
- 餐后组 `daily_008`（碗）超限 75/361 点，最大角速度约 0.03679 rad/s，
  位置漂移最大约 0.1114 mm，姿态漂移约 0.08965°。

这排除了这两次运行中的 getter 回退路径问题；不能把极小位姿变化直接等同于
角速度合格。曲线保存在 `v1/baseline_dynamics.{png,pdf}`。截图是在测量结束后
采集，以免渲染影响测量过程，不是所有房间的自然程度验收。

## 迭代参数实验

`--background-velocity-iterations 4` 为显式实验选项，仅作用于新增动态背景物品，
并记录每个对象修改前后参数；未写成生成器默认值或正式数据物理配置。
实验使用 OmniGibson 的**实体级**属性：有 articulation 时设置其求解器，
普通物体则设置其刚体 link。根 link 属性不能一概代表 articulation 的有效设置。

第一次实现错误地只允许非 articulation 日用品，在两组真实加载后被检查拒绝，
未执行参数干预；保留于 `iterations4/`。检查实际资产和 SDK 后改用实体级接口，
重新运行保留于 `entity_iterations4/`。调味瓶本身具有多个 link，不能为方便实验删掉。

该次重跑又被 SDK 的禁止嵌套 USD 编辑事务检查拦住：属性 setter 已自行管理事务。
去掉外层重复事务后，`entity_iterations4_v2/uniform` 实际完成了 1→4 的参数修改，
14 件物品均有修改前后记录；此前两轮适配异常不能计为有效参数对照。

有效对照仍被原速度门槛拒绝。延长 12 秒观察中，同一只碗 `daily_012` 仍超限
360/361 点，最大角速度约 0.09601 rad/s，P95 约 0.08863 rad/s，未改善基线。
因此没有把 4 次迭代设为生成默认值。此对照沿用原采样器，与已归档源码哈希一致。

## 验收计算修复

1. `indoor_realization._measurement` 原先推进 24 步，却只读 3 个位姿和最后一个速度。
   现在保持相同总步数，每个普通仿真步均读取位姿和线/角速度，按窗口最大速度
   应用原阈值；保存全部 24 个样本。窗口中途超限而末尾安静的回归用例必须拒绝。
   普通仿真步在本机为 1/30 秒，这不是每个 1/120 秒物理子步的连续证明。
2. `household_physics.placement_stability` 接受有限四元数范数误差，却未在求角度前
   归一化。两个完全相同、范数 .9999 的四元数会得到虚假的 2.2918° 旋转而被拒绝。
   修正后先归一化，并用数值稳定的角度公式处理符号等价和微小转动；原范数容差
   不变。回归验证静止为 0°、真实 .005° 不被抹掉、真实 2.1° 仍被拒绝。

位置 5 mm、旋转 2°、线速度 .01 m/s、角速度 .02 rad/s 四个阈值均未修改。
旧的失败记录保持原样，不按新公式倒推成成功。

当前采样器在 `strict_v2/{uniform,meal}` 实际 GPU 重跑完成，保持原物理参数：
两组 14/14 支撑均成立，但最后窗口分别有 7 件、8 件角速度超限，仍然拒绝。
保存的每件物品窗口都有 24 个真实样本。两组再次取得 12 秒完整后续轨迹。
这些超限实体数与旧版“只看窗口末尾”的计数不可作为物理改善或恶化对比，
差异首先来自检查覆盖范围修正。未进行目标插入和公开技能链。

汇总文件：`build/behavior_dynamics_audit/audit_summary.json`。本轮总计 9 次 GPU
启动，其中 4 次为上述适配异常，5 次完成物理测量与完整诊断轨迹；所有失败保留。
没有新增已验证评测数据。GPU 4、5 的作业均已退出，没有操作其他用户进程。

CPU 完整回归：838 passed、44 skipped、13 deselected（`pytest -m 'not sim' -q`）。
此外，诊断异常不能掩盖原失败、不能升级验收、未知时间序列不能报告静止、
四元数符号等价、瞬时速度峰值保留和参数干预范围均有单元回归。

这些运行和其本地源码快照在 `build/behavior_dynamics_audit/` 下独立保存；
没有覆盖此前候选、失败或 GPU 结果，也没有上传资产。

## 命令

新目录才可运行，下面先复现原参数诊断；添加 `--background-velocity-iterations 4`
可单独运行参数实验，必须使用另一输出目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  /data1/ygwang/miniconda3/envs/behavior/bin/python \
  scripts/tasks/behavior_pipeline.py gpu-smoke \
  --scenario build/behavior_language_pilot/v2/meal/scenario.yaml \
  --out build/behavior_dynamics_audit/reproduce_meal --gpu 5 \
  --realize-background --diagnose-background

/data1/ygwang/codes/_external/findingbench-rectification-tests/bin/python \
  -m pytest tests/unit/test_indoor_dynamics.py tests/unit/test_indoor_realization.py \
  tests/unit/test_household_physics.py -q
```

正式 evaluation-ready 始终为 0。即使参数实验改善短时稳定性，仍需要绑定
实际加载/恢复的物理配置、检查目标插入、技能、完整家族和实际搜索成本，
才能讨论正式数据；该实验选项本身没有完成这些工作。
