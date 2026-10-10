"""Mine BDDL :init placements into PlacementPriors evidence and apply the
review table -> <build_root>/priors/placement_v1.yaml.

    PYTHONPATH=src python scripts/tasks/mine_priors.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _common import ASSETS, host_config, load_yaml

from rummagebench.authoring.tasks.priors import apply_manual, apply_review, mine_priors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", default=str(ASSETS / "slot_rules_v1.yaml"))
    ap.add_argument("--review", default=str(ASSETS / "priors" / "review_v1.yaml"))
    ap.add_argument("--manual", default=str(ASSETS / "priors" / "manual_v1.yaml"),
                    help="hand-written commonsense priors (plan E); merged after the review")
    ap.add_argument("--host-config")
    ap.add_argument("--show", type=str, default="",
                    help="comma list of room types to print evidence for")
    args = ap.parse_args(argv)

    host = host_config(args.host_config)
    rules = load_yaml(args.rules)
    from bddl.object_taxonomy import ObjectTaxonomy
    import importlib.metadata as md

    root = host["bddl_activity_root"]
    files = sorted(root.glob("*/problem0.bddl"))
    digest = hashlib.sha256()
    problems = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        digest.update(f.parent.name.encode() + b"\0" + text.encode())
        problems.append((f.parent.name, text))
    source = {"type": "bddl", "bddl_version": md.version("bddl"),
              "activity_root": str(root), "problem_files": len(files),
              "corpus_sha256": digest.hexdigest(), "rules_version": rules["version"]}
    mined = mine_priors(problems, ObjectTaxonomy(), rules, source)
    build = host["build_root"] / "priors" / "mined_placement_v1.yaml"
    mined.save(build, "# Raw BDDL placement evidence.\n")
    print(f"mined {len(mined.entries)} (synset, slot_class, room_type) entries "
          f"from {len(files)} problems -> {build}")
    if args.show:
        rooms = set(args.show.split(","))
        for e in sorted(mined.entries, key=lambda e: -e.evidence_count):
            if e.room_type in rooms:
                print(f"  {e.evidence_count:4d} {e.activity_count:3d}  {e.object_synset:28s}"
                      f" {e.slot_class:17s} {e.room_type}")
    out = host["build_root"] / "priors" / "placement_v1.yaml"
    reviewed = apply_review(mined, load_yaml(Path(args.review)))
    manual_path = Path(args.manual)
    if manual_path.exists():
        reviewed = apply_manual(reviewed, load_yaml(manual_path), ObjectTaxonomy())
    reviewed.save(out, "# Reviewed placement priors (BDDL evidence + review table).\n")
    n = len(reviewed.approved_entries())
    print(f"reviewed -> {out}: {n} approved entries, "
          f"{len({e.object_synset for e in reviewed.approved_entries()})} synsets / "
          f"{len(reviewed.approved_categories())} categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
