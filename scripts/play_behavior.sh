#!/usr/bin/env bash
# Human play of the installed official BEHAVIOR pilot.
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${YGBENCH_CONDA_ROOT:-/home/ubuntu/miniforge3}"
if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
  echo "Conda not found: $conda_root (set YGBENCH_CONDA_ROOT)" >&2
  exit 1
fi
source "$conda_root/etc/profile.d/conda.sh"
conda activate ygbench
cd "$bench_root"
exec python -u scripts/serve_play.py --scenario scenarios/behavior_groceries_pilot/scenario.yaml "$@"
