"""§36: assign and write the dev/test split (thin CLI over split.assign_splits).

    PYTHONPATH=src python scripts/build_level1_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_level1_episodes import split  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(split())
