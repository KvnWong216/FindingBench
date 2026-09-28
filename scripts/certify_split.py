#!/usr/bin/env python
"""Split certification + generation (§10): certify every episode in a
benchmark split with the oracle planner, write per-episode certificates, and
regenerate the solvable / natural / counterfactual splits referencing the
certificates with depth statistics.

Run on the simulator host (needs a live session for the production
feasibility engine):

    python scripts/certify_split.py --split benchmark_splits/knife_search_v0.yaml

Unsolvable episodes are excluded from the standard split and recorded in
knife_search_v0_embodiment_stress.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="split YAML with episode paths")
    ap.add_argument("--robots", nargs="+", default=["default"])
    ap.add_argument("--out", default="build/certificates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0,
                    help="first episode index (chunked certification)")
    ap.add_argument("--count", type=int, default=0,
                    help="episodes this process certifies (0 = all)")
    ap.add_argument("--skip-splits", action="store_true",
                    help="only certify + write certificates (chunk mode)")
    args = ap.parse_args()

    import yaml

    from rummagebench.adapters.python_api import create_session
    from rummagebench.core.scenario import load_scenario
    from rummagebench.evaluation.certification import (
        depth_statistics,
        save_certificate,
    )
    from rummagebench.authoring.episode_generator import generate_episode, write_split

    split = yaml.safe_load((REPO_ROOT / args.split).read_text(encoding="utf-8"))
    episode_paths = [REPO_ROOT / p for p in split["episodes"]]
    if args.count > 0:
        episode_paths = episode_paths[args.start:args.start + args.count]

    session = create_session(episode_paths[0], run_dir=None, seed=args.seed)
    out_dir = REPO_ROOT / args.out

    from rummagebench.core.types import Action
    from rummagebench.evaluation.certification import certify_episode

    solvable, unsolvable = [], []
    for path in episode_paths:
        scenario = load_scenario(path)
        # resume support: an episode with a REPLAYED certificate is skipped
        existing = out_dir / f"{scenario.id}.json"
        if existing.is_file():
            from rummagebench.evaluation.certification import load_certificate

            prior = load_certificate(existing)
            if prior.plan_replay_status is not None:
                print(f"[certify] skip {scenario.id} (already certified, "
                      f"replay={prior.plan_replay_status})", flush=True)
                (solvable if prior.solvable else unsolvable).append((path, prior))
                continue
        # re-apply THIS episode's placements on the built scene
        session._backend.reapply_scenario(scenario)
        certificate = certify_episode(scenario, session)
        # replay gate: the certified plan must EXECUTE to the task goal
        # through the production BenchmarkSession — certification without a
        # successful replay is not accepted
        if certificate.solvable and certificate.oracle_plan:
            session.reset()
            for plan_action in certificate.oracle_plan:
                session.act(Action.from_dict(plan_action))
            certificate.plan_replay_status = session.status().value
            certificate.plan_replay_steps = session._planning_step
            print(
                f"[certify] replay {certificate.episode_id}: "
                f"{certificate.plan_replay_status} "
                f"({certificate.plan_replay_steps} steps)",
                flush=True,
            )
            if certificate.plan_replay_status != "SUCCESS":
                certificate.solvable = False
                certificate.reason = "PLAN_REPLAY_FAILED"
        cert_path = save_certificate(certificate, out_dir)
        if certificate.solvable:
            solvable.append((path, certificate))
        else:
            unsolvable.append((path, certificate))
        print(
            f"[certify] {certificate.episode_id}: solvable={certificate.solvable} "
            f"depth={certificate.oracle_depth} ({cert_path.name})",
            flush=True,
        )

    # write back the certified oracle depth into the episode YAMLs
    for path, certificate in solvable + unsolvable:
        scenario = load_scenario(path)
        if certificate.solvable and certificate.oracle_depth is not None:
            scenario.oracle_min_steps = certificate.oracle_depth
            path.write_text(
                yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )

    print(json.dumps({
        "chunk_certified": len(solvable) + len(unsolvable),
        "chunk_solvable": len(solvable),
        "chunk_unsolvable": len(unsolvable),
    }, indent=2))
    if args.skip_splits:
        os._exit(0)

    stats = depth_statistics([c for _, c in solvable + unsolvable])
    cert_dir_rel = str(Path(args.out))
    split_base = split.get("name", "knife_search_v0")

    write_split(
        f"{split_base}_solvable",
        [p for p, _ in solvable],
        split.get("robots", args.robots),
        out_path=REPO_ROOT / "benchmark_splits",
        certificates=[f"{cert_dir_rel}/{c.episode_id}.json" for _, c in solvable],
        depth_stats=stats,
        placement_regime="natural",
    )
    if unsolvable:
        write_split(
            f"{split_base}_embodiment_stress",
            [p for p, _ in unsolvable],
            split.get("robots", args.robots),
            out_path=REPO_ROOT / "benchmark_splits",
            certificates=[f"{cert_dir_rel}/{c.episode_id}.json" for _, c in unsolvable],
        )

    # natural/counterfactual splits by regime metadata
    natural = [(p, c) for p, c in solvable
               if load_scenario(p).placement_regime in (None, "natural")]
    counterfactual = [(p, c) for p, c in solvable
                      if load_scenario(p).placement_regime == "counterfactual"]
    if natural:
        write_split(
            f"{split_base}_natural",
            [p for p, _ in natural],
            split.get("robots", args.robots),
            out_path=REPO_ROOT / "benchmark_splits",
            certificates=[f"{cert_dir_rel}/{c.episode_id}.json" for _, c in natural],
            depth_stats=depth_statistics([c for _, c in natural]),
            placement_regime="natural",
        )
    if counterfactual:
        write_split(
            f"{split_base}_counterfactual",
            [p for p, _ in counterfactual],
            split.get("robots", args.robots),
            out_path=REPO_ROOT / "benchmark_splits",
            certificates=[f"{cert_dir_rel}/{c.episode_id}.json" for _, c in counterfactual],
            depth_stats=depth_statistics([c for _, c in counterfactual]),
            placement_regime="counterfactual",
        )

    print(json.dumps({
        "certified": len(solvable) + len(unsolvable),
        "solvable": len(solvable),
        "unsolvable": len(unsolvable),
        "depth_stats": stats,
    }, indent=2))
    os._exit(0)  # Kit teardown segfaults; artifacts are written


if __name__ == "__main__":
    main()
