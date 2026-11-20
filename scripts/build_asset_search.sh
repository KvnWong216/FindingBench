#!/usr/bin/env bash
# Full authored-task production pipeline. Resume by running the same command.
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${YGBENCH_CONDA_ROOT:-/home/ubuntu/miniforge3}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate ygbench
cd "$bench_root"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$bench_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMNIGIBSON_HEADLESS=True OMNIGIBSON_NO_OMNI_LOGS=True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u scripts/v2/run_asset_pipeline.py "$@"
