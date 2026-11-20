"""Compile TaskPlans into ScenarioSpec candidates -> <build_root>/<tier>/candidates/.

    PYTHONPATH=src python scripts/tasks/compile_tasks.py --tier easy_v1 \
        --scenes Beechwood_0_int
"""
from __future__ import annotations

import argparse
import json

import yaml
from _common import ASSETS, host_config

from rummagebench.authoring.tasks.compile import (
    compile_plan, scenario_hash, view_anchors_of, with_view_anchors)
from rummagebench.authoring.tasks.plan import TaskPlan
from rummagebench.authoring.tasks.planner import load_context


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="easy_v1")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--robot", default="r1pro")
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)

    host = host_config(args.host_config)
    ctx = load_context(args.tier, args.scenes, args.robot, ASSETS, host["build_root"])
    out = host["build_root"] / args.tier
    cand = out / "candidates"
    cand.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted((out / "plans").glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue  # sidecar written below on an earlier compile, not a plan
        plan = TaskPlan.load(path)
        if plan.scene not in ctx.scenes:
            continue
        doc = compile_plan(plan, ctx.scenes[plan.scene], ctx.overlays[plan.scene], ctx.tier)
        prev = cand / f"{plan.task_id}.yaml"
        if prev.exists():  # keep the certifier's view anchors (certify.py v3)
            views = view_anchors_of(yaml.safe_load(prev.read_text(encoding="utf-8")) or {})
            if views:
                doc = with_view_anchors(doc, views)
        h = scenario_hash(doc)
        (cand / f"{plan.task_id}.yaml").write_text(
            "# Compiled task candidate (build artefact, not a release asset).\n"
            + yaml.safe_dump(doc, sort_keys=False, width=100), encoding="utf-8")
        plan.provenance.setdefault("episode", {})["scenario_hash"] = h
        (out / "plans" / f"{plan.task_id}.meta.json").write_text(
            json.dumps({"task_id": plan.task_id, "scenario_hash": h}, indent=1))
        n += 1
    print(f"compiled {n} candidates -> {cand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
