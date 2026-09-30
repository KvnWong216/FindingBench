# 调试发现 — 2026-10-01（接手 handoff/2026-09-30-agent-debug-public）

基线：检出 `09a7531f23a28a5e73467f8078e4d7cce35be3cb`（核实一致）。
实现提交：`5f85f8f`（raw renderer instance grounding 默认集成 + 立柱回归测试）。
CPU：`pytest -m "not sim"` **274 passed, 13 deselected**；compileall / git diff --check 通过。

## 1. 立柱 MOVE50 拒绝：非误判（回归测试已固化）

collision_audit_12（raw_agent_resume_1）静态复现：`walls_tjfjwe_0`
AABB [2.0389,2.2485]×[-7.0614,-6.8518]×[0,2.4]，机器人在 0.3636m 采样处
底盘前角 (2.2472,-6.9155) 已进入 AABB 约 1.3mm；零余量同样拒绝（真实几何，
非 3cm 余量伪影）。立柱在公开 frame_000007 RGB 左前景可见。
横移 +11cm 后 0.5m 通道畅通 → 合法路线存在。
测试：tests/unit/test_relative_motion.py 三个用例（15 passed 全文件）。

## 2. renderer instance-ID 已集成默认 Python/MCP/UI

raw_instance.py 增加 GROUNDING_MODALITIES / enable_renderer_grounding /
install_physics_only_settle；env_factory 不再把 seg_instance 传入 OG 构建；
python_api（含 MCP worker）与 InteractiveSearchEnv（UI）共用
prepare/install 钩子。严格 AGENT reset 门禁不变；无 raycast、无伪造模态。
新增 tests/unit/test_renderer_grounding_integration.py（6 用例，纯 CPU）。

## 3. 运行证据（runs/server_acceptance/）

- agent_debug_20261001_a（GPU0，timeout 3600）：确定性 13 步路线全部
  EXECUTED 到达柜前 [0.72,-6.35]；OPEN 点选 (0.40,0.95) 精确命中
  bottom_cabinet_no_top_qohxjq_0（实例 152）→ EXECUTED，门真实打开；
  后退 0.5m 后开门舱体可见。1 小时 timeout 到期 SIGINT 退出（driver
  未写 exit_code，进程/跟随关闭干净），非代码缺陷。
- agent_debug_20261001_b（GPU0，timeout 7200）：重放 13 步再次全
  EXECUTED；GRASP 点选两次均解析正确但命中非刀实体；control:exit
  正常退出（driver_completed=true，exit 0）。
- 教训记录：a/b 两次运行中操作员三次 TURN 航向算术错误（多耗步骤），
  均由私有 trace 复盘定位；不影响基准语义。

## 4. 关键事实：刀在柜前地板上（确定性）

target_knife 世界 AABB [0.4709,-5.7512,0]×[0.7325,-5.6927,0.0056]
（audit_37，reset 后步 0 采样 = setup 放置的确定状态）：刀躺在柜门
前地板上，半贴柜体底沿，z≈0。语义 `Inside` 仍成立，物理上不在架内。

## 5. 阻塞点（需用户决策）：薄物体点选被 §8 主导率规则几何性排除

诊断（audit knife_pixels，agent_debug_20261001_b）：在 [0.72,-6.74]
朝北站位（刀 1.02m、可达包络 p75=1.02m/p95=1.41m 内），renderer 实例
365 = target_knife 可见 58 像素（rows 191-197 × cols 78-106，约
30×2.5px）。VisualBridge §8 要求 11×11 patch 内主导实例 ≥50%：刀
占 ~20-30%，地板 floors_sktjer_0 占 ~80% → 点选恒解析为地板
（SKILL_NOT_APPLICABLE，两次实证）。几何上：
- 站距 <0.9m：刀在 FOV 下缘之下（相机 ~1.2m 高、下俯 ~6°，54° 极限）；
- 站距 >0.97m：刀入画但厚度 ≤3.5px < 11×11 主导所需 ≥5.5px。
即任何合法站位都无法点选该刀。OPEN 后舱内亦无刀（刀本就在门外）。

可选最小修复（改的是本基准代码，非外部环境）：
a) 细长物体例外：patch 内次优实例若为基准实体且自身像素 ≥20，允许
   选择（记录 approximation 层级）；
b) 自适应 patch：以目标实例连通域为中心收缩 patch 至 5×5/7×7；
c) 提高渲染分辨率（224→448）由宿主配置承担。
任一方案需 CPU+GPU 回归与验收后才能放开短闭环 GRASP。

## 6. 验收状态

- 短闭环：接近/点选机制/OPEN 已验证；GRASP 被 §5 阻塞 → REPORT_DONE
  未达成（not_run/blocked，非 agent 错误、非基础设施错误，属规则缺口）。
- 30帧×3启动、1cm 标定复测、持物 OBSERVE×10、wrong-grasp/PLACE/预算、
  三接口验收：blocked（待 §5 决策后随短闭环补齐）。
- 固定场景实验：不具备开始条件。
