#!/usr/bin/env bash
# Smoke test: unit tests (no simulator) + the four acceptance episodes.
# Run inside the behavior conda env from the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== unit tests (no simulator) =="
python -m pytest tests/unit -q

echo "== building knife_search_001 =="
python -m rummagebench.cli build --scenario scenarios/knife_search_001/scenario.yaml

echo "== scripted (expect SUCCESS) =="
python -m rummagebench.cli run --scenario knife_search_001 --agent scripted --no-images

echo "== wrong_object (expect FAIL_WRONG_TARGET) =="
python -m rummagebench.cli run --scenario knife_search_001 --agent wrong_object --no-images

echo "== timeout (expect FAIL_MAX_STEPS) =="
python -m rummagebench.cli run --scenario knife_search_001 --agent timeout --no-images

echo "== unsafe (expect FAIL_UNSAFE_ACTION) =="
python -m rummagebench.cli run --scenario knife_search_001 --agent unsafe --no-images

echo "smoke test complete"
