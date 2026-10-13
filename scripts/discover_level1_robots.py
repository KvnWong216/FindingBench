"""Inventory installed YAML robot definitions without importing OmniGibson.

Writes candidates to build/, preserving the reviewed robot_pool_v1.yaml.
    PYTHONPATH=src python scripts/discover_level1_robots.py --assets-root PATH
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rummagebench.authoring.level1.robot_inventory import discover_installed_robots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("build/level1/robot_candidates.yaml"))
    args = ap.parse_args(argv)
    candidates = discover_installed_robots(args.assets_root)
    doc = {"robot_count": 5, "production_ready": False, "robots": candidates}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if yaml.safe_load(args.out.read_text()) != doc:
            raise SystemExit(f"conflicting robot inventory at {args.out}; choose a new output")
    else:
        with args.out.open("x", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
    print(f"{sum(v['status'] == 'candidate' for v in candidates.values())} real model candidates; "
          f"0 verified by static discovery -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
