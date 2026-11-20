"""Split certified tasks into dev / test with leakage checks
-> <build_root>/<tier>/split.json.

    PYTHONPATH=src python scripts/tasks/split_tasks.py --tier easy_v1 \
        --test-scenes Wainscott_0_int [--test-families <region ids>]
"""
from __future__ import annotations

import argparse
import json

from _common import host_config

from rummagebench.authoring.tasks.plan import TaskPlan
from rummagebench.authoring.tasks.slots import SceneSlots
from rummagebench.authoring.tasks.split import (
    EpisodeRecord, generalization_stats, leakage_violations, make_split)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="easy_v1")
    ap.add_argument("--test-scenes", nargs="*", default=[])
    ap.add_argument("--test-families", nargs="*", default=[])
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)

    build_root = host_config(args.host_config)["build_root"]
    out = build_root / args.tier
    records, skipped = [], 0
    slots_cache: dict[str, SceneSlots] = {}
    for cpath in sorted((out / "search_certificates").glob("*.json")):
        cert = json.loads(cpath.read_text())
        if cert["status"] != "certified":
            skipped += 1
            continue
        plan = TaskPlan.load(out / "plans" / cpath.name)
        ss = slots_cache.setdefault(plan.scene, SceneSlots.load(
            build_root / "scenes" / f"{plan.scene}.yaml"))
        records.append(EpisodeRecord.from_plan(
            plan, ss.by_id()[plan.target.slot_id].slot_type))
    if not records:
        print(f"no eligible episodes ({skipped} not certified)")
        return 1
    split = make_split(records, args.test_scenes, test_families=args.test_families)
    problems = leakage_violations(split, records)
    if problems:
        raise SystemExit(f"LEAKAGE: {problems}")
    doc = split.to_dict()
    doc["generalization"] = {"dev": generalization_stats(records, split.dev),
                             "test": generalization_stats(records, split.test)}
    (out / "split.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"{split.policy}: dev {len(split.dev)} / test {len(split.test)} "
          f"({skipped} uncertified skipped) -> {out/'split.json'}")
    for n in split.notes:
        print("NOTE:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
