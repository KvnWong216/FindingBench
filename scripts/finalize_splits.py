#!/usr/bin/env python
"""Finalize certified splits (§10): read all episode certificates from
build/certificates/, regenerate the solvable / natural / counterfactual /
embodiment_stress splits with depth statistics. No simulator needed.

    python scripts/finalize_splits.py --split benchmark_splits/knife_search_v0.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--certificates", default="build/certificates")
    args = ap.parse_args()

    import yaml

    from rummagebench.evaluation.certification import (
        depth_statistics,
        load_certificate,
    )
    from rummagebench.authoring.episode_generator import write_split

    split = yaml.safe_load((REPO_ROOT / args.split).read_text(encoding="utf-8"))
    split_base = split.get("name", "knife_search_v0")
    cert_dir = REPO_ROOT / args.certificates

    entries = []  # (episode_path, certificate)
    for ep_rel in split.get("episodes", []):
        ep_path = REPO_ROOT / ep_rel
        episode_id = ep_path.stem
        cert_path = cert_dir / f"{episode_id}.json"
        if not cert_path.is_file():
            print(f"[finalize] MISSING certificate for {episode_id}; skipped")
            continue
        entries.append((ep_path, load_certificate(cert_path)))

    solvable = [(p, c) for p, c in entries if c.solvable]
    unsolvable = [(p, c) for p, c in entries if not c.solvable]
    natural = [(p, c) for p, c in solvable
               if (load_scenario(p).placement_regime in (None, "natural"))]
    counterfactual = [(p, c) for p, c in solvable
                      if load_scenario(p).placement_regime == "counterfactual"]

    robots = split.get("robots", ["default"])
    stats = depth_statistics([c for _, c in solvable])
    cert_rel = args.certificates

    write_split(
        f"{split_base}_solvable",
        [p for p, _ in solvable],
        robots,
        out_path=REPO_ROOT / "benchmark_splits",
        certificates=[f"{cert_rel}/{c.episode_id}.json" for _, c in solvable],
        depth_stats=stats,
    )
    write_split(
        f"{split_base}_natural",
        [p for p, _ in natural],
        robots,
        out_path=REPO_ROOT / "benchmark_splits",
        certificates=[f"{cert_rel}/{c.episode_id}.json" for _, c in natural],
        depth_stats=depth_statistics([c for _, c in natural]),
        placement_regime="natural",
    )
    if counterfactual:
        write_split(
            f"{split_base}_counterfactual",
            [p for p, _ in counterfactual],
            robots,
            out_path=REPO_ROOT / "benchmark_splits",
            certificates=[f"{cert_rel}/{c.episode_id}.json" for _, c in counterfactual],
            depth_stats=depth_statistics([c for _, c in counterfactual]),
            placement_regime="counterfactual",
        )
    if unsolvable:
        write_split(
            f"{split_base}_embodiment_stress",
            [p for p, _ in unsolvable],
            robots,
            out_path=REPO_ROOT / "benchmark_splits",
            certificates=[f"{cert_rel}/{c.episode_id}.json" for _, c in unsolvable],
        )

    print(json.dumps({
        "certified_episodes": len(entries),
        "solvable": len(solvable),
        "unsolvable": len(unsolvable),
        "natural": len(natural),
        "counterfactual": len(counterfactual),
        "depth_stats": stats,
    }, indent=2))
    return 0


if __name__ == "__main__":
    main()
