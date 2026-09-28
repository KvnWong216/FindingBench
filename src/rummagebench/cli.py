"""RummageBench CLI.

    python -m rummagebench.cli run       --scenario knife_search_001 --agent scripted
    python -m rummagebench.cli evaluate  --scenario knife_search_001 --agent scripted
    python -m rummagebench.cli build     --scenario scenarios/knife_search_001/scenario.yaml
    python -m rummagebench.cli inspect   --scene Beechwood_0_int
    python -m rummagebench.cli trace-grounding --scenario knife_search_001
    python -m rummagebench.cli inspect-actions --scenario knife_search_001 --anchor kitchen
    python -m rummagebench.cli generate-episodes --scenario knife_search_001 --count 30
    python -m rummagebench.cli export-action-graph --scenario knife_search_001
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

    trace = None
    if not args.no_trace:
        trace = run_dir / "grounding_trace.jsonl"
    session = create_session(
        _scenario_file(args.scenario), run_dir=run_dir, seed=args.seed, trace_path=trace
    )
    agent = make_agent(args.agent, scenario)
    summary = run_episode(session, agent, run_dir, save_images=not args.no_images)

    from rummagebench.core.events import load_events
    from rummagebench.evaluation.metrics import compute_metrics

    summary["metrics"] = compute_metrics(
        load_events(run_dir / "events.jsonl"), scenario.oracle_min_steps
    )
    if trace is not None:
        summary["grounding_trace"] = str(trace)

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


def cmd_trace_grounding(args: argparse.Namespace) -> int:
    """§4: run the scripted episode with the §4 grounding trace enabled and
    pretty-print every grounding pass."""
    from rummagebench.adapters.python_api import create_session
    from rummagebench.core.grounding_trace import summarize_verdict
    from rummagebench.core.scenario import load_scenario
    from rummagebench.evaluation.episode_log import run_episode
    from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence

    scenario_path = _scenario_file(args.scenario)
    scenario = load_scenario(scenario_path)
    validate_agent_sequence("scripted", scenario)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{scenario.id}_trace"
    run_dir = Path(args.run_dir) / run_id
    trace_path = run_dir / "grounding_trace.jsonl"

    session = create_session(scenario_path, run_dir=run_dir, seed=args.seed, trace_path=trace_path)
    agent = make_agent("scripted", scenario)
    summary = run_episode(session, agent, run_dir, save_images=False)
    print(f"episode: {summary['episode_id']}  status: {summary['status']}")
    print(f"trace: {trace_path}\n")
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        import json

        record = json.loads(line)
        print(f"=== step {record['step']} (mode {record['mode']}) ===")
        print(f"robot_base_pose: {record['robot_base_pose']}")
        print(f"world_state: {record['world_state']}")
        print(f"grounded: {record['grounded']}")
        if args.all_candidates:
            for verdict in record["verdicts"]:
                print(summarize_verdict(verdict))
        else:
            for verdict in record["verdicts"]:
                if verdict["result"] != "AVAILABLE":
                    print(summarize_verdict(verdict))
        print()
    return 0


def cmd_inspect_actions(args: argparse.Namespace) -> int:
    """§5: human-readable grounding sanity check at one anchor."""
    from rummagebench.adapters.python_api import create_session
    from rummagebench.core.scenario import AnchorSpec, load_scenario

    scenario_path = _scenario_file(args.scenario)
    scenario = load_scenario(scenario_path)
    session = create_session(scenario_path, run_dir=None, seed=args.seed)
    session.reset()
    if args.anchor:
        if args.anchor not in scenario.anchors:
            print(f"unknown anchor {args.anchor!r}; have {sorted(scenario.anchors)}")
            return 1
        session._backend.teleport_robot(scenario.anchors[args.anchor])
        session._backend.settle()

    candidates, kept, verdicts = session._grounder.ground_with_verdicts(
        session.robot, session.world_state
    )
    kept_labels = {c.label() for c in kept}
    print(f"anchor: {args.anchor or 'initial'}  robot: {scenario.robot.model}")
    print(f"held: {session.world_state.held_object}\n")
    print(f"{'skill':40s} STATUS")
    for cand in candidates:
        label = cand.label()
        status = "AVAILABLE" if label in kept_labels else "?"
        print(f"{label:40s} {status}")
    for v in verdicts:
        label = f"{v.candidate.skill}({v.candidate.target})"
        if v.feasible:
            continue
        print(f"{label:40s} {v.reason}")
    # hidden entities (closed-container contents): evaluator knowledge
    backend = session._backend
    hidden = sorted(set(backend.entity_names()) - set(backend.visible_entities()))
    for entity in hidden:
        info = backend.describe_entity(entity)
        from rummagebench.object_interface.base import build_object_interface

        obj = build_object_interface(info, backend)
        for cand in obj.available_skills(session.robot, session.world_state):
            print(f"{cand.label():40s} HIDDEN (closed container)")
    return 0


def cmd_generate_episodes(args: argparse.Namespace) -> int:
    """§11-16: generate the knife-search benchmark distribution + split."""
    from rummagebench.authoring.episode_generator import (
        generate_episodes,
        write_split,
    )

    seeds = list(range(args.first_seed, args.first_seed + args.count))
    paths = generate_episodes(
        _scenario_file(args.scenario),
        seeds,
        out_dir=args.out,
        robot_variant=args.robot_variant,
    )
    split = write_split(args.split, paths, robots=[args.robot_variant or "default"])
    print(f"generated {len(paths)} episodes -> {args.out}")
    for p in paths:
        print(f"  {p}")
    print(f"split: {split}")
    return 0


def cmd_export_action_graph(args: argparse.Namespace) -> int:
    """§17: export the grounded action graph of the scripted trajectory."""
    import json

    from rummagebench.adapters.python_api import create_session
    from rummagebench.evaluation.action_graph import (
        compare_graphs,
        export_action_graph,
        write_graph,
    )
    from rummagebench.evaluation.evaluator import make_agent

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # one simulator launch; morphology variants swap the URDF-derived
    # feasibility stack in-process (the sim world is untouched)
    from rummagebench.authoring.episode_generator import ROBOT_VARIANTS

    scenario = load_scenario(_scenario_file(args.scenario))
    first = args.robots[0]
    apply_variant(scenario, first)
    session = create_session_variant(scenario, seed=args.seed)
    agent = make_agent("scripted", scenario)
    graphs = {}
    base_graph = export_action_graph(session, agent)
    graphs[first] = base_graph
    write_graph(base_graph, out_path.parent / f"{first}_action_graph.json")
    print(
        f"{first}: {len(base_graph['nodes'])} nodes, {len(base_graph['edges'])} edges "
        f"-> {out_path.parent / (first + '_action_graph.json')}"
    )

    for variant in args.robots[1:]:
        overrides = ROBOT_VARIANTS.get(variant)
        urdf = (overrides or {}).get("kinematics.urdf_path")
        if not urdf:
            print(f"variant {variant!r} has no kinematics override; skipped")
            continue
        session.swap_morphology(urdf)
        agent_v = make_agent("scripted", scenario)
        graph_v = export_action_graph(session, agent_v)
        graphs[variant] = graph_v
        write_graph(graph_v, out_path.parent / f"{variant}_action_graph.json")
        print(
            f"{variant}: {len(graph_v['nodes'])} nodes, {len(graph_v['edges'])} edges "
            f"-> {out_path.parent / (variant + '_action_graph.json')}"
        )
        comparison = compare_graphs(base_graph, graph_v)
        (out_path.parent / "morphology_comparison.json").write_text(
            json.dumps(comparison, indent=2)
        )
        print(
            f"morphology comparison vs {first}: "
            f"{comparison['common_nodes']} common nodes, differing action sets: "
            f"{len(comparison['differing_action_sets'])} "
            f"(graphs_differ={comparison['graphs_differ']})"
        )
    return 0


def apply_variant(scenario, variant: str) -> None:
    """Apply a registered robot variant override to a scenario in place."""
    from rummagebench.authoring.episode_generator import ROBOT_VARIANTS

    for dotted, value in ROBOT_VARIANTS.get(variant, {}).items():
        section, field = dotted.split(".")
        kin = scenario.robot.kinematics
        if kin is None:
            raise ValueError("robot variant requires a kinematics block")
        if section != "kinematics":
            raise ValueError(f"unsupported variant override {dotted!r}")
        setattr(kin, field, value)


def load_scenario_variant(scenario_path, variant: str):
    """§15/§16: apply a registered robot variant to a scenario."""
    from rummagebench.authoring.episode_generator import ROBOT_VARIANTS
    from rummagebench.core.scenario import load_scenario

    scenario = load_scenario(scenario_path)
    for dotted, value in ROBOT_VARIANTS.get(variant, {}).items():
        section, field = dotted.split(".")
        kin = scenario.robot.kinematics
        if kin is None:
            raise ValueError("robot variant requires a kinematics block")
        setattr(kin, field, value)
    return scenario


def create_session_variant(scenario, seed: int):
    from rummagebench.adapters.python_api import create_session
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "variant_scenario.yaml"
    import yaml as _yaml

    tmp.write_text(_yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False))
    return create_session(tmp, run_dir=None, seed=seed)


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
    run_p.add_argument("--no-trace", action="store_true",
                       help="skip the grounding trace")
    run_p.set_defaults(func=cmd_run)

    eval_p = sub.add_parser("evaluate", help="alias of run (batch evaluation entry)")
    eval_p.add_argument("--scenario", required=True)
    eval_p.add_argument(
        "--agent", required=True, choices=["scripted", "wrong_object", "timeout", "unsafe"]
    )
    eval_p.add_argument("--run-dir", default="runs")
    eval_p.add_argument("--seed", type=int, default=0)
    eval_p.add_argument("--no-images", action="store_true")
    eval_p.add_argument("--no-trace", action="store_true")
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

    tg_p = sub.add_parser("trace-grounding",
                          help="run the scripted episode with the grounding trace")
    tg_p.add_argument("--scenario", required=True)
    tg_p.add_argument("--run-dir", default="runs")
    tg_p.add_argument("--seed", type=int, default=0)
    tg_p.add_argument("--all-candidates", action="store_true",
                      help="print AVAILABLE verdicts too (default: failures only)")
    tg_p.set_defaults(func=cmd_trace_grounding)

    ia_p = sub.add_parser("inspect-actions",
                          help="human-readable grounding sanity check at an anchor")
    ia_p.add_argument("--scenario", required=True)
    ia_p.add_argument("--anchor", default=None)
    ia_p.add_argument("--seed", type=int, default=0)
    ia_p.set_defaults(func=cmd_inspect_actions)

    ge_p = sub.add_parser("generate-episodes",
                          help="generate the knife-search benchmark distribution")
    ge_p.add_argument("--scenario", required=True)
    ge_p.add_argument("--count", type=int, default=30)
    ge_p.add_argument("--first-seed", type=int, default=0)
    ge_p.add_argument("--out", default="build/generated_scenarios")
    ge_p.add_argument("--split", default="knife_search_v0")
    ge_p.add_argument("--robot-variant", default="default")
    ge_p.set_defaults(func=cmd_generate_episodes)

    ag_p = sub.add_parser("export-action-graph",
                          help="export the grounded action graph (scripted trajectory)")
    ag_p.add_argument("--scenario", required=True)
    ag_p.add_argument("--robots", nargs="+", default=["default"],
                      help="robot variants (use --robots default r1pro_restricted)")
    ag_p.add_argument("--out", default="runs/action_graph/index.json")
    ag_p.add_argument("--seed", type=int, default=0)
    ag_p.set_defaults(func=cmd_export_action_graph)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
