#!/usr/bin/env bash
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${YGBENCH_CONDA_ROOT:-/home/ubuntu/miniforge3}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${YGBENCH_VLM_ENV:-ygbench-vlm-extended}"
cd "$bench_root"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec python -u scripts/serve_extended_vlm.py "$@"
