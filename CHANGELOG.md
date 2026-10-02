# FindingBench Level-1 Data Factory — 实施日志

## 2026-10-03 — 代码阶段完成（CPU 全绿），GPU 阶段待卡

基线：main `ab0c8ce` → 本轮提交基于其上。CPU 回归：**311 passed, 13 sim
deselected**（含 Level-1 新增 30 项），compileall / git diff --check 通过。

### 已交付（代码）

**可视点选修复（修订 §4，grounding_version=center_pixel_v2）**
- `perception/visual_bridge.py`：权威选择改为"中心像素 → renderer 实例 →
  所属对象"；11×11 邻域不再选择对象，仅作为被选对象自身表面采样窗口；
  背景/未知/机器人本体中心 → INVALID_ACTION；链接实例仅经已验证的
  owning-object 映射合并。`core/visual_session.py` 点选记录绑定
  `grounding_version`。
- 测试重写：薄刀横穿桌面仍选中刀；刀旁背景中心不选中刀；倒扣碗中心不
  选中隐藏刀；把手/门像素映射到同一交互实体；表面点只用被选对象自身
  深度样本；grounding 版本绑定。

**Level-1 数据工厂（spec §39 布局）**
- `authoring/level1/`：config / rng（SeedSequence 流：geometry/object/
  packing/appearance/physics）/ catalog（清单阶段，CPU）/ coverage_sampler
  （类别→模型两级覆盖感知采样，1/sqrt(1+usage)+种子抖动）/ grammar（三
  范式 ScenePlan：container_rummage / tabletop_clutter / deliberate_cover，
  数量分布来自 dataset_v1）/ workspace / appearance / compiler（Stage A
  结构编译）/ certifier（§43 拒绝码 + Stage 校验纯函数 + witness 记录）/
  snapshot（不可变快照、原子写、canonical hash）/ registry（登记+拒绝直
  方图+INVALID_RUN 分离）/ split（按环境分层 50/350）/ robot_registry
  （五个真实移动操作臂契约，未验证不得进生产）。
- `authoring/occupancy/`：grid（占用/支撑高度/容器内部掩码）、voxelize
  （保守 AABB 代理 + 薄物体膨胀）、proxy（缓存键绑定 asset hash/source/
  scale/朝向 bin/分辨率/版本；crc 校验）、packing（支撑堆放/容器装箱/
  cover 提案）、lift（体素→连续位姿+种子抖动）、support/container/cover。
- `configs/level1/`：dataset_v1（400 环境/140+140+120/分布原样）、
  physics_v1（稳定性以仿真秒计，dt=120Hz）、appearance_v1、support_pool
  （breakfast_table 等 5 族，真实数据集类别）、container_pool（tray/
  outline_tray/recycling_bin/compost_bin，开顶）、robot_pool（r1pro
  gpu_verified + 发现脚本待补，production_ready=false 如实标注）。
- `scripts/`：discover_level1_robots / build_level1_object_catalog /
  precompute_level1_voxels / generate_level1_candidates（原子+resume）/
  certify_level1_environments（子进程隔离+超时+INVALID_RUN 重试）/
  build_level1_episodes / build_level1_split / report_level1_dataset。

### GPU 阶段（阻塞：等 gcchen 释放，起跑前复测显存余量）

按修订优先级依次执行：
1. `discover_level1_robots.py` → 机器人池实名化 + GPU 验证五个实体。
2. `build_level1_object_catalog.py` → Kit 内逐模型几何/物理验证（AABB、
   碰撞、沉降），回填 catalog。
3. `precompute_level1_voxels.py` 升级 collision 源代理。
4. 30 场景试点（3 范式 ×10，认证脚本常量按试点校准）→ `pilot_30_summary`。
5. 试点门禁全过 → 400 场景生产（候选上限 800）→ 5×400=2000 回合登记。
6. witness 重放（协议内可解性）按修订 §10 每对环境-机器人留存记录。

### 诚实边界

- 目录几何字段与一切 GPU 测量为 pending，未标 pass。
- certify 脚本的 OG 实现常量按修订 §49 由试点校准；当前若直接跑会在
  GPU_PILOT_REQUIRED 处显式报错而非给出假结果。
- 主线评测协议未动（八技能/RGB-only/当前帧/四类反馈/预算 16）。
