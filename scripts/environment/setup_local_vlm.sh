#!/usr/bin/env bash
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_root="${YGBENCH_CONDA_ROOT:-/home/ubuntu/miniforge3}"
source "$conda_root/etc/profile.d/conda.sh"
if [[ ! -x "$conda_root/envs/ygbench-vlm/bin/python" ]]; then
  conda create -n ygbench-vlm python=3.11 pip -y
fi
conda activate ygbench-vlm
python -m pip install -r "$bench_root/configs/environment/ygbench-vlm-requirements.txt"
