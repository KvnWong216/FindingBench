"""Plan tasks: deterministic TaskPlans + CPU gates -> <build_root>/<tier>/plans/.

    PYTHONPATH=src python scripts/tasks/plan_tasks.py --tier easy_v1 \
        --scenes Beechwood_0_int --seed 0 --count 20

Each plan is re-planned once and compared byte for byte (determinism check).
Failed episodes go to rejections.jsonl with every attempt's codes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from _common import ASSETS, host_config

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.planner import load_context, plan_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="easy_v1")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--robot", default="r1pro")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)

    host = host_config(args.host_config)
    ctx = load_context(args.tier, args.scenes, args.robot, ASSETS, host["build_root"],
                       behavior_assets_root=host["behavior_assets_root"])
    out = host["build_root"] / args.tier
    (out / "plans").mkdir(parents=True, exist_ok=True)
    (out / "search_certificates").mkdir(parents=True, exist_ok=True)
    rej_path = out / "rejections.jsonl"
    codes: Counter[str] = Counter()
    final_codes: Counter[str] = Counter()
    n_ok = 0
    with rej_path.open("w", encoding="utf-8") as rej:
        for i in range(args.count):
            o = plan_episode(ctx, args.seed, i, args.max_attempts)
            for a in o.attempts:
                codes.update(a["codes"])
            if o.plan is None:
                final_codes.update(o.attempts[-1]["codes"])
                rej.write(json.dumps({"stage": "cpu", "seed": args.seed, "index": i,
                                      "codes": sorted({c for a in o.attempts
                                                       for c in a["codes"]}),
                                      "attempts": o.attempts}) + "\n")
                continue
            again = plan_episode(ctx, args.seed, i, args.max_attempts)
            if again.plan is None or again.plan.canonical_bytes() != o.plan.canonical_bytes():
                raise SystemExit(f"DETERMINISM FAILURE at index {i}")
            o.plan.save(out / "plans" / f"{o.plan.task_id}.json")
            (out / "search_certificates" / f"{o.plan.task_id}.json").write_text(
                json.dumps(o.certificate.to_dict(), indent=1, sort_keys=True))
            n_ok += 1
    print(f"{args.tier}: {n_ok}/{args.count} plans -> {out/'plans'}")
    print(f"attempt rejection histogram: {dict(sorted(codes.items()))}")
    print(f"final-attempt codes of failed episodes: {dict(sorted(final_codes.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
