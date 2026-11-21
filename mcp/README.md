# FindingBench RAB v1 — Stdio MCP Server Package (`findingbench_rab_pinhole_v1`)

本目录将 FindingBench RAB v1 评测基准封装为标准的 **Model Context Protocol (MCP)** 本地 Stdio 服务，供任意支持多模态 `ImageContent` 的智能体客户端（Antigravity、Claude Code、Cursor、Codex、OpenResearch `orx` 评测器等）直接接入试玩或评测。

## 1. 零闲置开销与防卡顿设计（Lazy GPU Boot & Auto-Release）

- **启动零显存占用**：MCP 主进程（`rummagebench.adapters.rab_mcp_server`）绝不导入 Isaac Sim / OmniGibson / `pxr`。IDE 启动或枚举 `tools/list`、`episode_status` 仅耗时 `<50 ms`，显存占用 `0 MiB`，绝不会拖慢桌面。
- **按需拉起与终局自动释放**：仅当智能体显式调用 `reset_episode(scenario_id)` 时才在 `GPU 0` 上启动独立的原生 PhysX 子进程；当开启 `FINDINGBENCH_RAB_STOP_ON_TERMINAL=1`（默认开启）时，一旦回合终止（`REPORT_DONE` 或步数耗尽），底层 Isaac Sim 子进程立即自动退出释放显存，同时保留终局状态与最后一帧观测供查询。
- **严格公私边界隔离**：公开返回值仅包含自然语言任务指令、当前 `1280x720` 针孔 RGB 图像（`ImageContent`）、可选 `720x450` `CHECK` 四视图（`ImageContent`）、`frame_id`、步数与公开反馈码；所有私有评分信号（`success`、`restoration`、精确坐标误差、USD 路径）仅写入评测器私有目录 `runs/rab_mcp_sessions/<episode>/mcp_private_events.jsonl`。

## 2. 暴露的 4 个标准 MCP 工具

1. `reset_episode(scenario_id: str)`
   - 示例：`{"scenario_id": "rab_F065"}`（或 `"F065"`、`"rab_F251"`、`"rab_F262"`）。
   - 锁定 `runs/rab_lived_in_20261120/combined_manifest.json` 校验 USD SHA256，使用空修订层启动场景，返回结构化 JSON 元数据 + 首帧 `1280x720` PNG `ImageContent`。
2. `observe()`
   - 幂等读取当前公开观测（不推进物理步、不扣减 `planning_step`）。
3. `act(action: dict)`
   - 提交单步公共技能（9 选 1），推进低层检查与原生 PhysX 结算，返回反馈码（`EXECUTED` / `INVALID_ACTION` / `UNSAFE` / `IK_UNREACHABLE` / `COLLISION_BLOCKED` 等）及最新观测图像：
     - `MOVE_TO`: `{"skill": "MOVE_TO", "point": {"frame_id": "<current>", "x": 0.5, "y": 0.72}}`
     - `TURN`: `{"skill": "TURN", "direction": "left" | "right" | "back"}`
     - `VIEW`: `{"skill": "VIEW", "yaw": 0.0, "pitch": -15.0, "zoom": 1.0}`
     - `OPEN` / `CLOSE` / `GRASP` / `PLACE` / `CHECK`: `{"skill": "<SKILL>", "point": {"frame_id": "<current>", "x": 0.5, "y": 0.6}}`
     - `REPORT_DONE`: `{"skill": "REPORT_DONE"}`
   - **点选铁律**：归一化图像坐标 `x, y ∈ [0, 1]` 以左上角为 `(0, 0)`，且 `point.frame_id` **必须等于最新主场景的 `frame_id`**；`CHECK` 返回的四视图仅供视觉辨认，不可作为点选 `frame_id`（违例将被拒绝并计 1 步 `INVALID_ACTION`）。
4. `episode_status()`
   - 查询当前回合状态：`NO_EPISODE`、`RUNNING`、`SUCCESS`、`FAIL_FALSE_COMPLETION`、`FAIL_MAX_STEPS`。

## 3. 快速接入方式

- **Antigravity 插件**：已安装至 `~/.gemini/config/plugins/findingbench-rab/`。
- **通用 MCP 客户端配置**：直接复制 [mcp_config.example.json](file:///home/ubuntu/ygwang/FindingBench/mcp/mcp_config.example.json) 中的 `findingbench-rab` 配置项即可。

## Continuous-motion demo replay (optional)

Set `FINDINGBENCH_RAB_MOTION=replay` in the MCP server environment (default `discrete`).
Game mechanics are unchanged: every skill is validated and realized exactly as in discrete mode,
refused skills are never animated and return the same failure. After an `EXECUTED` physical skill,
the already-computed trajectory (A* waypoints for `MOVE_TO`/`TURN`, DLS-IK iterates for `GRASP`/`PLACE`,
drive sweep + animation-only handle IK for `OPEN`/`CLOSE`) is replayed kinematically from the
pre-action state, recorded, and the authoritative post-action state is restored.

Evaluator-private outputs in `<session>/worker_trace/demo/`: `episode_chase.mp4`, `episode_head.mp4`,
`trajectory.jsonl` (per-frame base pose, full DOF vector, attached object, drive targets),
`segments.jsonl` (decision -> frame range), `meta.json`.
Knobs: `FINDINGBENCH_RAB_DEMO_FPS` (20), `FINDINGBENCH_RAB_DEMO_CAMERAS` (`chase,head`),
`FINDINGBENCH_RAB_DEMO_HOLD_S` (0.3). Equivalence check: `scripts/rab_motion_equivalence.py`.
