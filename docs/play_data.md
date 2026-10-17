# 试玩生成的数据

## 在自己的电脑上一键启动（推荐）

只需把仓库里的 `scripts/play_remote.py` 这一个文件放到自己的电脑，然后运行：

```bash
python play_remote.py 1
```

第一次按提示填写平时 SSH 登录的 `用户@服务器地址` 或 SSH 配置别名。
脚本记住地址，之后仍然只需传数据编号。SSH 密钥、密码和跳板机沿用本机
OpenSSH 的配置；不需要安装其他 Python 库。

脚本会自动启动或复用服务器上的同一条数据、建立 SSH 端口转发、选择空闲的
本地端口并打开浏览器。无需手动运行 `ssh -L`，也不用在服务器再开一个启动终端。
如果默认 8090 被占用，会换空闲端口并打开正确链接。

保持本地终端开启。Ctrl+C 会关闭隧道；若远端服务由本次启动，还会关闭该服务。
复用的既有服务会保留。网络断开导致清理失败时，下次启动会尝试复用它。
更换服务器可运行 `python play_remote.py 1 --configure`。
远端默认仓库为 `/data1/ygwang/codes/FindingBench`，服务器上需要已有
`scripts/play_data.py`（本次已接好）。本地配置保存于
`~/.config/findingbench/play.json`，只保存地址与仓库路径。

## 在服务器直接启动

在 FindingBench 根目录运行，只需要数据编号：

```bash
python scripts/play_data.py 1
```

`1` 和 `home_0001` 是同一条数据。打开终端打印的
**http://127.0.0.1:8090**，等待真实房屋加载完成即可操作。
如果浏览器在自己的电脑上，需要将服务器的 8090 端口转发到本地
（例如 VS Code 的端口转发，或 `ssh -L 8090:127.0.0.1:8090 用户@服务器`）。

- `MOVE`：输入前进/后退距离，单位厘米。
- `TURN`：输入转向角度，单位度。
- `OPEN / CLOSE / GRASP / PLACE / OBSERVE`：先选技能，再点当前画面中的位置。
- `REPORT_DONE`：报告任务完成；判定沿用现有接口。
- `reset`：恢复该条数据的保存状态，重新开始。
- 在终端按 `Ctrl+C` 退出并释放试玩进程。

使用真实相机画面和原有技能库。页面显示技能反馈；动作失败不会换场景、
自动找站位或代替玩家执行策略。OBSERVE 的辅助画面只供观察，不可作为点击目标。

默认读取 `build/behavior_production_environment_v1`，只允许打开已经完成
物理环境验收、且保存文件校验一致的数据。尚未完成或失败的数据会直接提示，
不会重新随机生成一个环境冒充原数据。程序恢复 `environment_start.pt`，
reset 不重新采样物体摆放。物理验收通过不代表所有交互或任务可解性已认证。

默认用 **物理 GPU 5、2 个逻辑 CPU 核**，通过现有设备隔离器运行；网页默认只监听
本机回环地址。试玩不修改生产数据，日志写到 `build/play/<编号>_<时间>/`：
`simulator.log` 保存加载和错误信息，`public_trace.jsonl` 保存玩家操作，
`replay_check.json` 记录加载后的背景位姿与目标关系核验。

如需换端口、GPU 或数据批次，可用环境变量，数据参数仍只有一个：

```bash
FINDINGBENCH_PLAY_PORT=8091 FINDINGBENCH_PLAY_GPU=4 python scripts/play_data.py 3
FINDINGBENCH_PLAY_RUN=/绝对路径/另一批次 python scripts/play_data.py 1
```

GPU 仅允许 4 或 5。保存状态和装配文件属于本地评测侧资产，不会作为网页的
隐藏位置、对象 ID 或参考路径提示发送给玩家。

本机验证（2026-10-16）：36 项相关 CPU 测试通过；`home_0001` 已在 GPU 5
跨进程恢复，14 个背景物品位姿与目标关系检查通过，网页返回真实相机画面。
网页接口提交 TURN 15° 返回 EXECUTED；reset 后步数归零，背景与目标关系
再次通过检查。验证记录在 `build/play_script_validation/`。这不是全数据或
全部八个技能的交互认证。

远程启动器的 CPU 验证使用明确标注的假 HTTP 服务，覆盖启动、重复启动复用、
端口选择、身份校验后清理、配置复用和隧道失败；不作为仿真验收证据。
本机无法代替用户电脑验证实际 SSH 登录和浏览器环境；这部分仍需在用户电脑
运行确认，脚本会显示实际连接错误。

如果使用同一内网/VPN直接访问，可在服务器明确指定监听地址：

```bash
FINDINGBENCH_PLAY_HOST=10.20.37.124 python scripts/play_data.py 1
```

此时浏览器打开 `http://10.20.37.124:8090`，不需要 SSH 转发。
远程一键脚本仍固定使用回环地址和 SSH 隧道。
