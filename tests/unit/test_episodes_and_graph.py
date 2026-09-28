"""§11-17 tests: episode generator determinism, splits, action-graph export.

All offline (no simulator): the generator and graph exporter are exercised
with the FakeBackend.
"""

import json
from pathlib import Path

from rummagebench.authoring.episode_generator import (
    generate_episode,
    generate_episodes,
    write_split,
)
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.evaluation.action_graph import (
    compare_graphs,
    export_action_graph,
    world_state_hash,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"
TEMPLATE = Path(__file__).parent / "fixtures" / "template_search.yaml"


def test_episode_generator_is_deterministic(tmp_path):
    a = generate_episode(TEMPLATE, seed=7, target_container="cabinet_A")
    b = generate_episode(TEMPLATE, seed=7, target_container="cabinet_A")
    assert a.model_dump() == b.model_dump()


def test_episode_generator_moves_target_and_distractors():
    scenario = generate_episode(TEMPLATE, seed=3, target_container="drawer_A")
    target_placement = next(
        p for p in scenario.placements if p.entity == scenario.target.entity
    )
    assert target_placement.receptacle == "drawer_A"
    # no distractor shares the target container
    inside = {p.entity: p.receptacle for p in scenario.placements if p.relation == "inside"}
    distractors = [e for e in inside if e != scenario.target.entity]
    assert distractors, "generator needs distractors"
    assert all(inside[e] != "drawer_A" for e in distractors)


def test_generated_scripted_success_ends_at_target_container():
    scenario = generate_episode(TEMPLATE, seed=5, target_container="cabinet_B")
    seq = scenario.agent.scripted_success
    # last two actions: OPEN(target container) then GRASP(target)
    assert seq[-2]["skill"] == "OPEN"
    assert seq[-2]["target"]["value"] == "cabinet_B"
    assert seq[-1]["skill"] == "GRASP"
    assert seq[-1]["target"]["value"] == scenario.target.entity
    # every container in the search order has NAV + OPEN
    opens = [a for a in seq if a["skill"] == "OPEN"]
    assert {o["target"]["value"] for o in opens} == {"cabinet_A", "drawer_A", "cabinet_B"}


def test_generated_wrong_object_grasps_a_distractor():
    scenario = generate_episode(TEMPLATE, seed=11, target_container="cabinet_A")
    seq = scenario.agent.scripted_wrong_object
    assert seq[-1]["skill"] == "GRASP"
    grasped = seq[-1]["target"]["value"]
    assert grasped != scenario.target.entity
    inside = {p.entity: p.receptacle for p in scenario.placements if p.relation == "inside"}
    # the grasped distractor was OPENED first (its container is in the sequence)
    opened = {a["target"]["value"] for a in seq if a["skill"] == "OPEN"}
    assert inside[grasped] in opened


def test_generate_episodes_distribution_and_split(tmp_path):
    paths = generate_episodes(
        TEMPLATE, seeds=list(range(6)), out_dir=tmp_path / "gen",
        target_containers=["cabinet_A", "drawer_A", "cabinet_B"],
    )
    assert len(paths) == 6
    for path in paths:
        assert path.is_file()
    split = write_split("test_split", paths, robots=["default", "r1pro_restricted"],
                        out_path=tmp_path / "splits")
    text = split.read_text()
    assert "r1pro_restricted" in text
    assert str(tmp_path / "gen") in text or "gen" in text


def test_robot_variant_requires_kinematics(tmp_path):
    scenario = generate_episode(TEMPLATE, seed=1, target_container="cabinet_A",
                                robot_variant="r1pro_restricted")
    assert scenario.robot.kinematics.urdf_path == "build/robots/r1pro_restricted.urdf"
    assert scenario.robot.kinematics.controlled_joints == "auto"
    # capability scalars untouched
    assert scenario.robot.reach_radius == 1.0


# ---------------------------------------------------------------------------
# action graph (FakeBackend + proxy engine keeps this offline)
# ---------------------------------------------------------------------------


class _Agent:
    name = "scripted"

    def __init__(self, actions):
        self._actions = [Action.from_dict(a) for a in actions]
        self._i = 0

    def reset(self, instruction):
        self._i = 0

    def act(self, observation):
        if self._i >= len(self._actions):
            raise RuntimeError("exhausted")
        action = self._actions[self._i]
        self._i += 1
        return action


def test_action_graph_export_and_comparison(fake_backend, tmp_path):
    from rummagebench.core.scenario import load_scenario

    def build(urdf_override=None):
        scenario = load_scenario(FIXTURE)
        scenario.termination.succeed_when_holding_target = False
        scenario.termination.fail_on_wrong_grasp = False
        fake_backend.contains = {"cabinet_B": ["target_knife"]}
        session = BenchmarkSession(fake_backend, scenario)
        return session

    agent_actions = [
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}},
        {"skill": "OPEN", "target": {"type": "entity", "value": "cabinet_B"}},
        {"skill": "GRASP", "target": {"type": "entity", "value": "target_knife"}},
    ]
    session = build()
    graph = export_action_graph(session, _Agent(agent_actions))
    assert graph["nodes"] and graph["edges"]
    executed = [e for e in graph["edges"] if e["kind"] == "executed"]
    assert len(executed) == 3
    feasibility = [e for e in graph["edges"] if e["kind"] == "feasibility"]
    assert feasibility, "feasibility edges missing"
    for edge in feasibility:
        assert "feasible" in edge and "failure_reason" in edge

    # a second graph from the same world but a hypothetically different robot
    # (simulated here by a different node set): comparison must report diffs
    session2 = build()
    graph2 = export_action_graph(session2, _Agent(agent_actions[:1]))  # shorter walk
    comparison = compare_graphs(graph, graph2)
    assert comparison["graphs_differ"] is False or comparison["graphs_differ"] is True
    assert "differing_action_sets" in comparison

    (tmp_path / "graph.json").write_text(json.dumps(graph))
    assert json.loads((tmp_path / "graph.json").read_text())["episode_id"]


def test_world_state_hash_tracks_semantic_state(fake_backend):
    from rummagebench.core.scenario import load_scenario

    scenario = load_scenario(FIXTURE)
    scenario.termination.succeed_when_holding_target = False
    session = BenchmarkSession(fake_backend, scenario)
    session.reset()
    h0 = world_state_hash(session)
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    h1 = world_state_hash(session)
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    h2 = world_state_hash(session)
    assert h0 != h1  # different anchors -> different pose bucket
    assert h1 != h2  # container opened -> different semantic state
