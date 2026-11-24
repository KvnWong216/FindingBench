#!/usr/bin/env bash
# Portable fixed-contract evaluation entrypoint for OpenResearch (`orx`).
# Automatically resolves the `ygbench` Python environment across collaborator machines.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [[ -n "${FINDINGBENCH_PYTHON:-}" && -x "${FINDINGBENCH_PYTHON}" ]]; then
  PY_BIN="${FINDINGBENCH_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniforge3/envs/ygbench/bin/python" ]]; then
  PY_BIN="${HOME}/miniforge3/envs/ygbench/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/ygbench/bin/python" ]]; then
  PY_BIN="${HOME}/miniconda3/envs/ygbench/bin/python"
elif [[ -x "/home/ubuntu/miniforge3/envs/ygbench/bin/python" ]]; then
  PY_BIN="/home/ubuntu/miniforge3/envs/ygbench/bin/python"
else
  PY_BIN="$(command -v python3 || command -v python)"
fi

exec "${PY_BIN}" -u "${REPO_DIR}/scripts/run_rab_mcp_eval.py" \
  --config "${REPO_DIR}/configs/benchmarks/eval_config.json" \
  --resume
