"""RummageBench CLI.

    python -m rummagebench.cli run       --scenario knife_search_001 --agent scripted
    python -m rummagebench.cli evaluate  --scenario knife_search_001 --agent scripted
    python -m rummagebench.cli build     --scenario scenarios/knife_search_001/scenario.yaml
    python -m rummagebench.cli inspect   --scene Beechwood_0_int
    python -m rummagebench.cli serve-mcp
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


def _scenario_file(scenario: str) -> Path:
    p = Path(scenario)
    if p.is_file():
        return p
    root = os.environ.get("RUMMAGEBENCH_ROOT")
    base = Path(root) if root else Path(__file__).resolve().parents[2]
    candidate = base / "scenarios" / scenario / "scenario.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"scenario {scenario!r} not found at {candidate}")
    return candidate


def cmd_run(args: argparse.Namespace) -> int:
    from rummagebench.adapters.python_api import create_session
    from rummagebench.core.types import EpisodeStatus
    from rummagebench.evaluation.episode_log import run_episode
    from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence
    from rummagebench.core.scenario import load_scenario

    scenario = load_scenario(_scenario_file(args.scenario))
    validate_agent_sequence(args.agent, scenario)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{scenario.id}_{args.agent}"
    run_dir = Path(args.run_dir) / run_id

    session = create_session(_scenario_file(args.scenario), run_dir=run_dir, seed=args.seed)
    agent = make_agent(args.agent, scenario)
    summary = run_episode(session, agent, run_dir, save_images=not args.no_images)

    from rummagebench.core.events import load_events
    from rummagebench.evaluation.metrics import compute_metrics

    summary["metrics"] = compute_metrics(load_events(run_dir / "events.jsonl"))

    print("=" * 60)
    print(f"episode: {summary['episode_id']}  agent: {summary['agent']}")
    print(f"instruction: {summary['instruction']}")
    for event in load_events(run_dir / "events.jsonl"):
        a = event["action"]
        print(
            f"  step {event['step']:>2}: {a['skill']}({a['target'].get('value', '')}) "
            f"-> {event['status']}"
        )
    print(f"status: {summary['status']}")
    print(f"steps used: {summary['planning_steps_used']}/{summary['max_planning_steps']}")
    print(f"trajectory: {run_dir / 'events.jsonl'}")
    print("=" * 60)

    expected = {
        "scripted": EpisodeStatus.SUCCESS,
        "wrong_object": EpisodeStatus.FAIL_WRONG_TARGET,
        "timeout": EpisodeStatus.FAIL_MAX_STEPS,
        "unsafe": EpisodeStatus.FAIL_UNSAFE_ACTION,
    }.get(args.agent)
    if expected is not None and summary["status"] != expected.value:
        print(f"UNEXPECTED: agent {args.agent} should end with {expected.value}", file=sys.stderr)
        return 1
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from rummagebench.authoring.builder import build_scenario

    result = build_scenario(_scenario_file(args.scenario), out_root=args.out_root, seed=args.seed)
    print(f"built {result['scenario_id']} -> {result['out_dir']}")
    print(f"  preview:   {result['preview']}")
    print(f"  snapshot:  {result['snapshot']}")
    print(f"  report:    {result['build_report']}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from rummagebench.authoring.inspect import main as inspect_main

    argv = []
    if args.scene:
        argv += ["--scene", args.scene]
    if args.list_scenes:
        argv += ["--list-scenes"]
    if args.json:
        argv += ["--json"]
    return inspect_main(argv)


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    from rummagebench.adapters.mcp_server import main as mcp_main

    return mcp_main()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="rummagebench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one episode with a built-in agent")
    run_p.add_argument("--scenario", required=True)
    run_p.add_argument(
        "--agent", required=True, choices=["scripted", "wrong_object", "timeout", "unsafe"]
    )
    run_p.add_argument("--run-dir", default="runs")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--no-images", action="store_true")
    run_p.set_defaults(func=cmd_run)

    eval_p = sub.add_parser("evaluate", help="alias of run (batch evaluation entry)")
    eval_p.add_argument("--scenario", required=True)
    eval_p.add_argument(
        "--agent", required=True, choices=["scripted", "wrong_object", "timeout", "unsafe"]
    )
    eval_p.add_argument("--run-dir", default="runs")
    eval_p.add_argument("--seed", type=int, default=0)
    eval_p.add_argument("--no-images", action="store_true")
    eval_p.set_defaults(func=cmd_run)

    build_p = sub.add_parser("build", help="compile a scenario into a reproducible initial state")
    build_p.add_argument("--scenario", required=True)
    build_p.add_argument("--out-root", default="build/scenarios")
    build_p.add_argument("--seed", type=int, default=0)
    build_p.set_defaults(func=cmd_build)

    ins_p = sub.add_parser("inspect", help="inspect a BEHAVIOR scene")
    ins_p.add_argument("--scene", default=None)
    ins_p.add_argument("--list-scenes", action="store_true")
    ins_p.add_argument("--json", action="store_true")
    ins_p.set_defaults(func=cmd_inspect)

    mcp_p = sub.add_parser("serve-mcp", help="start the MCP stdio server")
    mcp_p.set_defaults(func=cmd_serve_mcp)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
