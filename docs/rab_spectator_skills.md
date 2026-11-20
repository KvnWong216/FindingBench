RAB 1K 的默认入口使用 [已验收的稠密发布](../runs/rab_dense_serial_20261119/combined_manifest.json)。命令从 FindingBench 根目录执行：

```sh
conda activate ygbench
python scripts/spectator.py --list
python scripts/spectator.py --task F251
python scripts/spectator.py --edit --task P136
python scripts/play_data.py F251
```

`--task` 可填 1–1000、原任务 ID（`F251`）或 RAB ID（`rab_F251`）。巡检和编辑在 `127.0.0.1:8091`，`play_data.py` 默认在 `127.0.0.1:8090`；可用 SSH 转发端口。多个实例通过不同端口启动。也可运行 `python scripts/spectator.py --skills --task F251 --run-output runs/my_rab_trial`。

| 模式 | 操作与边界 |
|---|---|
| Spectator | 点击捕获鼠标，WASD 飞行，Space/E 上升，Shift/Q 下降，Ctrl 加速，R 复位。允许穿墙，物理冻结。1000 个缓存完整性均已通过；共享房间和资产，安装目录仍需保留。 |
| 编辑器 | 搜索可放置的原生 RAB 模型（支持常用中文词），拖放或点选落位，Delete 删除。选中该任务已绑定的原生抽屉或冰箱门可开合。保护任务目标、对照样品、遮挡关键物及依赖支撑；没有原生关节的家具不提供开合。Ctrl+S 用独立 PhysX 进程重载后，原子发布本条修订并重建缓存。 |
| 技能 | 从 RAB 机器人头部观察。VIEW、CHECK、TURN、MOVE_TO、OPEN、CLOSE、GRASP、PLACE、REPORT_DONE 共九个动作。点选作用于当前 RGB，CHECK 的四视图只供辨识。OPEN/CLOSE 使用原生关节驱动，带距离与物品位姿检查；GRASP/PLACE 使用原生机械臂与夹爪驱动。 |

独立保存的版本在 `build/rab_spectator/<release-sha>/<rab-id>/revisions/`，该条目的 `current.json` 指向已审核版本。默认巡检和技能读取最新修订；使用单独的 `--rab-edits DIR` 可隔离人工修改。原始 1K 发布、验收记录和原版缓存保留。编辑后的独立重载检查只验证该修订初始状态和保存的关节位姿，不能替代完整的密集任务验收。

程序接口：`GET /observation` 取当前 RGB、`frame_id`、任务及反馈；`POST /action` 提交现有动作 schema，例如：

```json
{"skill":"CHECK","point":{"frame_id":"<current frame>","x":0.42,"y":0.48}}
```

两个端口角色不能混用。技能服务不出位置、对象 ID、任务真值或自由相机；CHECK 不提供可用于动作的场景像素。跟踪文件可供本地评估隔离读取。RAB 当前协议是 `findingbench_rab_pinhole_v1`，不声明与旧 BEHAVIOR 全景观测或历史指标逐项等价。双抽屉／冰箱开合和一条真实罐子抓放已有物理凭证；全 1K 尚未跑完整任务，也未对所有物体与站位承诺可抓取。
