"""Summarise a tier's tasks -> <build_root>/<tier>/report.json.

    PYTHONPATH=src python scripts/tasks/report_tasks.py --tier easy_v1

Certificate statistics, search-structure distributions, rejection histogram,
target / room / slot / mode / region distributions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from _common import host_config

from rummagebench.authoring.tasks.gates import histogram


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="easy_v1")
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)
    out = host_config(args.host_config)["build_root"] / args.tier

    certs, plans = [], []
    for cpath in sorted((out / "search_certificates").glob("*.json")):
        certs.append(json.loads(cpath.read_text()))
        plans.append(json.loads((out / "plans" / cpath.name).read_text()))
    rejections = []
    rej = out / "rejections.jsonl"
    if rej.exists():
        rejections = [json.loads(l) for l in rej.read_text().splitlines() if l.strip()]

    gpu_rej = []
    grej = out / "gpu_rejections.jsonl"
    if grej.exists():
        gpu_rej = [json.loads(l) for l in grej.read_text().splitlines() if l.strip()]
    records = [json.loads(p.read_text())
               for p in sorted((out / "certificates").glob("fb_*.json"))]

    def bucket(v, w):
        return None if v is None else round(int(v / w) * w, 2)

    def evidence_dist(name, w=None):
        vals = [(r["search_certificate"].get("evidence") or {}).get(name) for r in records]
        return dict(sorted(Counter(str(v if w is None else bucket(v, w))
                                   for v in vals).items()))

    def dist(key, rows):
        return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))
    report = {
        "tier": args.tier,
        "episodes": len(certs),
        "status": dist("status", certs),
        "rejected_episodes": len(rejections),
        "rejection_histogram": histogram(rejections),
        "search_structure": {k: dist(k, certs) for k in (
            "room_count", "candidate_slot_count", "containment_depth", "open_depth",
            "rearrangement_depth", "lookalike_count", "target_slot_type",
            "navigation_lower_bound", "minimum_required_interactions")},
        "mode": dist("mode", plans),
        "scene": dist("scene", plans),
        "room_type": dist("room_type", plans),
        "search_scope": dict(Counter("region" if p.get("search_region") else "room"
                                     for p in plans)),
        "search_region": dist("search_region", plans),
        "target_category": dict(Counter(p["target"]["category"] for p in plans)),
        "distractors_per_episode": dict(Counter(len(p["distractors"]) for p in plans)),
        "overlay_status": dict(Counter(p["provenance"]["robot"].get("overlay_status")
                                       for p in plans)),
        "gpu": {
            "certified_records": len(records),
            "status": dict(Counter(r["status"] for r in records)),
            "rejection_histogram": histogram(gpu_rej),
            "certified_by_mode": dict(Counter(
                r["structural"]["mode"] for r in records if r["status"] == "certified")),
            "certified_execution_steps": evidence_dist("certified_execution_steps"),
            "oracle_depth": evidence_dist("oracle_depth"),
            "revealed_visibility_0.05": evidence_dist("revealed_visibility", 0.05),
            "start_visibility": evidence_dist("start_visibility", 0.05),
            "errors": dict(Counter((r.get("error") or "")[:80] for r in records
                                   if r.get("error"))),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("episodes", "status", "rejection_histogram",
                                             "mode", "search_scope", "overlay_status", "gpu")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
