# FindingBench Level-1 数据工厂（阶段总览）

本节记录 Level-1（本地交互搜索）数据工厂的契约与管线。公开评测协议冻结
不变：八技能、RGB-only、点交互、当前帧 frame_id、四类反馈、16 步预算。

## Level-1 是什么

局部工作台搜索：机器人 + 一张验证过的支撑面（可选一个开顶容器）+
约 10 个普通 BEHAVIOR 物体 + 目标物。难点是局部物理杂乱、遮挡与翻找；
不含跨房间导航（Level-2 才有抽屉/柜门开合）。

规模契约：**400 个物理环境 × 5 个真实机器人本体 = 2000 个主评测回合**
（140 容器翻找 + 140 桌面杂乱 + 120 故意遮挡；dev 50 环境/test 350 环境，
同一环境的五个机器人永远同 split）。

## 管线（占据引导的物理场景编译）

    任务语法（ScenePlan，三范式）
      → BEHAVIOR 资产覆盖感知采样（类别→模型两级，1/√(1+usage)）
      → 占据打包提案（2cm 体素，薄物体膨胀；仅提案）
      → 连续位姿提升（种子抖动）
      → 真实网格实例化（GPU）
      → PhysX 松弛（小批量投放，仿真秒稳定性窗）
      → 精确物理+可视认证（Stage B/C，含隔离参考可视比）
      → 五机器人认证（Stage D）+ 协议可解 witness（rev §10）
      → 不可变环境快照 → 机器人专属回合

体素有效性永远只是提案；最终认证只认真实碰撞几何与模拟器测量。

## 使用

```bash
# CPU（无 GPU 依赖）
PYTHONPATH=src python scripts/build_level1_object_catalog.py
PYTHONPATH=src python scripts/precompute_level1_voxels.py
PYTHONPATH=src python scripts/generate_level1_candidates.py --resume

# GPU（先确认卡空闲与显存余量）
PYTHONPATH=src python scripts/discover_level1_robots.py
PYTHONPATH=src python scripts/certify_level1_environments.py --resume
PYTHONPATH=src python scripts/build_level1_split.py
PYTHONPATH=src python scripts/build_level1_episodes.py
PYTHONPATH=src python scripts/report_level1_dataset.py
```

配置全部在 `configs/level1/*.yaml`（数据分布、物理稳定性窗、外观范围、
支撑/容器/机器人池）。生成器输出按候选原子写入并支持 `--resume`。

## 点选契约（grounding_version=center_pixel_v2）

点选权威性在中心像素：中心像素的 renderer 实例决定被选对象，邻域不再
投票；薄物体（如刀）横穿任意背景仍可被选中；背景/未知中心一律
INVALID_ACTION；表面深度只取被选对象自身的采样。版本绑定进每个点选
记录与证书。
