#!/usr/bin/env bash
set -euo pipefail
bench_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$bench_root/scripts/evaluate_vlm.sh" --agent-policy humanclaw "$@"
