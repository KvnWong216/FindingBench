"""Action-graph export (§17): the grounded action graph actually observed on
a trajectory, for same-task / different-morphology comparison.

Nodes: world-state hashes (semantic state + robot base pose bucket).
Edges (two kinds):
    feasibility edges — for EVERY semantic candidate at the node, the oracle
        feasibility verdict (skill, target, feasible, reason); no execution;
    executed edges    — the actions the agent actually took and the state
        transition they caused.

The graph is NOT an exhaustive state-space expansion: it is the scripted /
explored trajectory plus the candidate sets the grounder derived at each
visited node — exactly what the morphology comparison needs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import EpisodeStatus


def world_state_hash(session: BenchmarkSession) -> str:
    """Semantic node id: open states, benchmark-held object, robot pose
    (rounded to 5 cm so suspension noise does not split nodes)."""
    backend = session._backend
    opens = sorted(e for e in backend.entity_names() if backend.is_open(e))
    pos, _ = backend.robot_pose()
    bucket = [round(v / 0.05) for v in pos]
    payload = json.dumps(
        {"opens": opens, "held": session.world_state.held_object, "base": bucket},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def export_action_graph(session: BenchmarkSession, agent=None) -> dict[str, Any]:
    """Walk one trajectory (scripted agent or the current reset state) and
    export nodes + feasibility/executed edges."""
    scenario = session.scenario
    graph: dict[str, Any] = {
        "episode_id": scenario.id,
        "robot": scenario.robot.model,
        "kinematics_urdf": (
            scenario.robot.kinematics.urdf_path if scenario.robot.kinematics else None
        ),
        "nodes": {},
        "node_order": [],
        "edges": [],
    }

    def record_node() -> str:
        node_id = world_state_hash(session)
        if node_id in graph["nodes"]:
            return node_id
        candidates, kept, verdicts = session._grounder.ground_with_verdicts(
            session.robot, session.world_state
        )
        pos, quat = session._backend.robot_pose()
        opens = sorted(
            e for e in session._backend.entity_names() if session._backend.is_open(e)
        )
        graph["node_order"].append(node_id)
        graph["nodes"][node_id] = {
            "base_pose": [round(float(v), 3) for v in [*pos, *quat]],
            "world_state": {"held_object": session.world_state.held_object, "opens": opens},
            "candidates": sorted(c.label() for c in candidates),
            "available": sorted(c.label() for c in kept),
            "feasibility": [
                {
                    "skill": v.candidate.skill,
                    "target": v.candidate.target,
                    "feasible": bool(v.feasible),
                    "reason": v.reason,
                }
                for v in verdicts
            ],
        }
        for v in verdicts:
            graph["edges"].append({
                "kind": "feasibility",
                "from": node_id,
                "skill": v.candidate.skill,
                "target": v.candidate.target,
                "feasible": bool(v.feasible),
                "failure_reason": v.reason if not v.feasible else None,
                "to": None,
            })
        return node_id

    observation = session.reset()
    if agent is not None:
        agent.reset(observation.instruction)

    current = record_node()
    while session.status() == EpisodeStatus.RUNNING and agent is not None:
        try:
            action = agent.act(observation)
        except RuntimeError:
            break  # scripted sequence exhausted
        result = session.act(action)
        next_node = record_node()
        graph["edges"].append({
            "kind": "executed",
            "from": current,
            "skill": action.skill,
            "target": action.target.value,
            "executed": result.executed,
            "failure_reason": result.failure_reason.value,
            "to": next_node,
        })
        observation = result.observation
        current = next_node

    # feasibility edges out of the FINAL node too (terminal states ground
    # differently, e.g. hand full)
    record_node()
    return graph


def compare_graphs(graph_a: dict, graph_b: dict) -> dict[str, Any]:
    """Same episode, two morphologies: where do the grounded action graphs
    differ?

    Nodes are matched positionally along the walked trajectory (same scripted
    action sequence -> step i of graph A corresponds to step i of graph B
    even when the state hashes diverge because an action executed differently
    under the other morphology — that divergence IS the signal).
    """
    nodes_a, nodes_b = graph_a["nodes"], graph_b["nodes"]
    order_a = graph_a.get("node_order") or list(nodes_a.keys())
    order_b = graph_b.get("node_order") or list(nodes_b.keys())
    pairs = list(zip(order_a, order_b))
    common = sorted(set(nodes_a) & set(nodes_b))
    diffs = []
    for node_a, node_b in pairs:
        if node_a not in nodes_a or node_b not in nodes_b:
            continue
        avail_a = set(nodes_a[node_a]["available"])
        avail_b = set(nodes_b[node_b]["available"])
        if avail_a != avail_b:
            diffs.append({
                "node_a": node_a,
                "node_b": node_b,
                "same_state": node_a == node_b,
                "only_a": sorted(avail_a - avail_b),
                "only_b": sorted(avail_b - avail_a),
            })
    # hash-level divergence: same trajectory position landing in different
    # states under the two morphologies
    diverged_states = sum(1 for a, b in pairs if a != b)
    return {
        "robot_a": graph_a.get("robot"),
        "robot_b": graph_b.get("robot"),
        "nodes_a": len(nodes_a),
        "nodes_b": len(nodes_b),
        "common_nodes": len(common),
        "trajectory_positions": len(pairs),
        "diverged_states": diverged_states,
        "differing_action_sets": diffs,
        "graphs_differ": bool(diffs or diverged_states),
    }


def write_graph(graph: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return path
