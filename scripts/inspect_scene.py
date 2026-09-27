#!/usr/bin/env python3
"""Inspect a BEHAVIOR scene: python scripts/inspect_scene.py --scene Beechwood_0_int"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rummagebench.authoring.inspect import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
