"""Build the object eligibility whitelist -> <build_root>/priors/eligibility_v1.yaml.

    PYTHONPATH=src python scripts/tasks/build_eligibility.py

approved priors ∩ Level-1 catalog ∩ size ∩ object-interface capability.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import ASSETS, host_config, load_yaml

from rummagebench.authoring.level1.catalog import eligible_inventory, scan_inventory
from rummagebench.authoring.tasks.eligibility import build_eligibility
from rummagebench.authoring.tasks.priors import PlacementPriors
from rummagebench.authoring.tasks.tiers.base import file_sha256


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", default=str(ASSETS / "priors" / "eligibility_rules_v1.yaml"))
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)

    host = host_config(args.host_config)
    args.priors = host["build_root"] / "priors" / "placement_v1.yaml"
    args.out = host["build_root"] / "priors" / "eligibility_v1.yaml"
    assets_root = host["behavior_assets_root"]
    objects_root = assets_root / "objects"
    priors = PlacementPriors.load(Path(args.priors))
    rules = load_yaml(Path(args.rules))
    inventory: dict[str, list[str]] = defaultdict(list)
    for e in eligible_inventory(scan_inventory(objects_root)):
        inventory[e.category].append(e.model_id)
    avg = json.loads((assets_root / "metadata" / "avg_category_specs.json").read_text())

    from bddl.object_taxonomy import ObjectTaxonomy

    source = {"priors_sha256": file_sha256(args.priors),
              "rules_sha256": file_sha256(args.rules),
              "objects_root": str(objects_root),
              "inventory_categories": len(inventory)}
    elig = build_eligibility(sorted(priors.approved_categories()), ObjectTaxonomy(),
                             inventory, objects_root, avg, rules, source)
    elig.save(Path(args.out))
    n_models = sum(len(c.models) for c in elig.categories.values())
    print(f"eligible: {len(elig.categories)} categories / {n_models} models; "
          f"rejected {len(elig.rejected)} -> {args.out}")
    for cat, why in sorted(elig.rejected.items()):
        print(f"  - {cat:28s} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
