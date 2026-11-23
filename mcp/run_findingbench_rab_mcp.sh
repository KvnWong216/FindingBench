#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUMMAGEBENCH_ROOT="${RUMMAGEBENCH_ROOT:-${REPO_DIR}}"
export FINDINGBENCH_RAB_RELEASE="${FINDINGBENCH_RAB_RELEASE:-${RUMMAGEBENCH_ROOT}/runs/rab_lived_in_20261120}"
export FINDINGBENCH_RAB_MCP_OUT="${FINDINGBENCH_RAB_MCP_OUT:-${RUMMAGEBENCH_ROOT}/runs/rab_mcp_sessions}"
export RUMMAGEBENCH_MCP_GPU="${RUMMAGEBENCH_MCP_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FINDINGBENCH_RAB_ROBOT="${FINDINGBENCH_RAB_ROBOT:-unitree_g1}"
export FINDINGBENCH_RAB_EMBODIMENT_PRIOR="${FINDINGBENCH_RAB_EMBODIMENT_PRIOR:-text}"
export FINDINGBENCH_RAB_IK_BACKEND="${FINDINGBENCH_RAB_IK_BACKEND:-auto}"
export FINDINGBENCH_RAB_FEEDBACK_MODE="${FINDINGBENCH_RAB_FEEDBACK_MODE:-proprioceptive_v2}"
export FINDINGBENCH_RAB_HOT_RELOAD="${FINDINGBENCH_RAB_HOT_RELOAD:-1}"
if [[ "${FINDINGBENCH_RAB_HOT_RELOAD}" == "0" || "${FINDINGBENCH_RAB_HOT_RELOAD}" == "false" ]]; then
  export FINDINGBENCH_RAB_STOP_ON_TERMINAL="${FINDINGBENCH_RAB_STOP_ON_TERMINAL:-1}"
else
  export FINDINGBENCH_RAB_STOP_ON_TERMINAL="${FINDINGBENCH_RAB_STOP_ON_TERMINAL:-0}"
fi
export PYTHONPATH="${RUMMAGEBENCH_ROOT}/src:${RUMMAGEBENCH_ROOT}/scripts"

if [[ -n "${FINDINGBENCH_PYTHON:-}" ]]; then
  PY_BIN="${FINDINGBENCH_PYTHON}"
elif [[ -x "/home/ubuntu/miniforge3/envs/ygbench/bin/python" ]]; then
  PY_BIN="/home/ubuntu/miniforge3/envs/ygbench/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  PY_BIN="$(command -v python)"
fi

exec "${PY_BIN}" -u -m rummagebench.adapters.rab_mcp_server "$@"
