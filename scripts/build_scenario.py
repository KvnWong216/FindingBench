#!/usr/bin/env python3
"""Build a scenario: python scripts/build_scenario.py scenarios/knife_search_001/scenario.yaml"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rummagebench.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0].endswith(".yaml"):
        argv = ["build", "--scenario"] + argv
    else:
        argv = ["build"] + argv
    raise SystemExit(main(argv))
