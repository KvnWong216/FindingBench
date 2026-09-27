#!/usr/bin/env python3
"""Run one episode: python scripts/run_episode.py --scenario knife_search_001 --agent scripted"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rummagebench.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run"] + sys.argv[1:]))
