#!/usr/bin/env bash
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${YGBENCH_CONDA_ROOT:-/home/ubuntu/miniforge3}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate ygbench
cd "$bench_root"
exec python -u scripts/evaluate_local_vlm.py "$@"
