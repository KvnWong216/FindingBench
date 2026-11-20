#!/usr/bin/env bash
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_root="${YGBENCH_CONDA_ROOT:-/home/ubuntu/miniforge3}"
source "$conda_root/etc/profile.d/conda.sh"
for suffix in extended phi4; do
  bench_env="ygbench-vlm-${suffix}"
  if [[ ! -x "$conda_root/envs/$bench_env/bin/python" ]]; then
    conda create -n "$bench_env" python=3.11 pip -y
  fi
  conda run -n "$bench_env" python -m pip install -r "$bench_root/configs/environment/$bench_env-requirements.txt"
  conda run -n "$bench_env" python -m pip check
done
