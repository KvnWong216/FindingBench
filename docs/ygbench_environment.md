# ygbench 环境

```bash
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate ygbench
cd /home/ubuntu/ygwang/FindingBench
```

环境路径：`/home/ubuntu/miniforge3/envs/ygbench`。Python 3.11，PyTorch 2.7.0 / CUDA 12.8，Isaac Sim 5.1.0，OmniGibson 3.9.3，BDDL 3.7.0，Pinocchio 2.7 / HPP-FCL 2.4.4，Warp 1.12。机器人资产 3.8.2，BEHAVIOR 资产 3.9.0，官方 2026 任务模板。

激活环境会设置 `OMNIGIBSON_DATA_PATH=/home/ubuntu/ygwang/behavior-datasets` 和 `OMNI_KIT_ACCEPT_EULA=YES`。编译工具链和 FFmpeg 均装在环境中。首次渲染需要编译缓存；缓存后本机完整任务启动约 2–3 分钟。

源码安装路径：`/home/ubuntu/ygwang/BEHAVIOR-1K-6cbf70b075816096e9be53958780769f3264d25d`；固定官方提交。FindingBench 以 editable 模式安装，修改代码后不必重装。

## 自检

```bash
python scripts/environment/verify_ygbench.py
FINDINGBENCH_R1PRO_URDF="$PWD/build/robots/r1pro.urdf" python -m pytest tests/unit -q
python scripts/prepare_behavior_pilot.py --assets-root "$OMNIGIBSON_DATA_PATH"
python scripts/environment/smoke_behavior_pilot.py
```

模型接入见 [原生任务接口](behavior_pilot.md)。用户负责接模型；验收脚本不调用任何大模型。

CPU 全量验证：1000 passed、17 skipped。跳过项为未配置的旧生成数据 / 外部资产检查；不算作通过。真实仿真全景 / VIEW 验收通过，具体结果以 `runs/ygbench_setup/pilot_smoke/report.json` 为准。

启动时先加载环境内的 C++ runtime 与 Pinocchio，再启动 Kit，避免 Boost.Python 容器注册冲突。普通 Python API 和人工脚本都自动执行此顺序，无需手动设置 LD_PRELOAD。

## 版本约束

`configs/environment/ygbench-constraints.txt` 固定已验证的 NumPy 1 兼容依赖。PyArrow 26 要求 NumPy 2，因此使用 PyArrow 21；TorchCodec 使用 CPU wheel，视频解码不依赖额外 CUDA NPP 库。仿真和 PyTorch 运算仍使用 GPU。

Isaac Sim 和 OmniGibson 的上游依赖声明存在相互冲突的版本钉死：Pillow、websockets、packaging；MCP 与 Isaac Sim 的 uvicorn 声明也不一致。环境保留实际运行兼容组合，不修改第三方包元数据伪造 `pip check` 通过。具体检查结果保存到 `runs/ygbench_setup/pip-check.txt`。

## 人工操作

```bash
cd /home/ubuntu/ygwang/FindingBench
./scripts/play_behavior.sh
```

脚本自动激活 `ygbench`，默认加载官方采购物品任务。浏览器打开 **http://127.0.0.1:8090**，等画面出现即可操作；远程 IDE 将服务器的 8090 端口转发到本机。换端口用 `./scripts/play_behavior.sh --port 8091`。

- VIEW：填写绝对 yaw / pitch（度）；正 yaw 向左，正 pitch 向上，0 / 0 看正前方。
- MOVE / TURN：填写带正负号的距离（厘米）或角度（度）。
- OPEN / CLOSE / GRASP / PLACE：先选技能，再点击当前画面中的目标。
- REPORT_DONE：提交官方任务完成判断；未完成时会结束本回合并记失败。
- reset：恢复官方初始状态，重新开始。

页面显示当前图像、任务、步数和反馈。默认不显示物体 ID 或答案。Ctrl+C 停止服务。人工操作使用和模型相同的公共动作接口。
