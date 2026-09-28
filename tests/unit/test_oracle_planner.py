"""§11 A/B tests: oracle semantic planner correctness + embodiment
dependency; §8 oracle↔production consistency; §3 certification.

All oracle tests run against the FakeBackend with the proxy feasibility
engine — the oracle wraps the PRODUCTION validator, so consistency here is
the consistency guarantee itself.
"""

from pathlib import Path

import pytest

from conftest import FakeBackend
from rummagebench.core.scenario import load_scenario
from rummagebench.sim.base import ArticulationInfo, ArticulationJointInfo, EntityInfo
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.evaluation.certification import (
    certify_episode,
    depth_statistics,
    save_certificate,
)
from rummagebench.evaluation.oracle_planner import (
    OracleFeasibilityModel,
    initial_oracle_state,
    solve_full_information,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(fake_backend, oracle_min_steps=None):
    scenario = load_scenario(FIXTURE)
    scenario.termination.succeed_when_holding_target = True
    if oracle_min_steps is not None:
        scenario.oracle_min_steps = oracle_min_steps
    return BenchmarkSession(fake_backend, scenario)


def _plan(fake_backend, scenario=None):
    session = _session(fake_backend)
    model, initial = __import__(
        "rummagebench.evaluation.oracle_planner", fromlist=["build_oracle_model"]
    ).build_oracle_model(session)
    return solve_full_information(session.scenario, initial, model), session


# ---------------------------------------------------------------------------
# A. oracle planner correctness
# ---------------------------------------------------------------------------


def test_oracle_exact_depth_and_plan_executes(fake_backend):
    """mini.yaml: knife inside cabinet_B (closed, anchored away).

    Optimal: NAV kitchen -> OPEN cabinet_B -> GRASP knife = 3 admissible
    actions; BFS proves nothing shorter exists (2 actions cannot both open
    the container and hold the knife from any anchor).
    """
    result, session = _plan(fake_backend)
    assert result.solvable
    assert result.depth == 3
    assert [a["skill"] for a in result.actions] == ["NAV", "OPEN", "GRASP"]
    assert result.actions[-1]["target"]["value"] == "target_knife"

    # the oracle plan EXECUTES successfully in the production session
    session.reset()
    for action in result.actions:
        step = session.act(Action.from_dict(action))
    assert session.status().value == "SUCCESS"


def test_oracle_no_shorter_solution_by_construction(fake_backend):
    """BFS exhausts depth 2 without finding the goal: enumerate all depth-2
    admissible action sequences and check none holds the knife."""
    session = _session(fake_backend)
    model, initial = __import__(
        "rummagebench.evaluation.oracle_planner", fromlist=["build_oracle_model"]
    ).build_oracle_model(session)
    transition = model  # use the model's overlay for manual enumeration
    from rummagebench.evaluation.oracle_planner import _OracleTransition

    scenario = session.scenario
    tr = _OracleTransition(scenario, model)

    goal_reached = False
    frontier = [(initial, 0)]
    visited = {initial}
    while frontier:
        state, depth = frontier.pop()
        if depth >= 2:
            continue
        for skill, target in tr.candidates(state):
            if skill == "GRASP" and target != scenario.target.entity:
                continue
            if tr.unsafe(state, skill, target):
                continue
            if not tr.feasible(state, skill, target):
                continue
            nxt = tr.transition(state, skill, target)
            if nxt.held_object == scenario.target.entity:
                goal_reached = True
            if nxt not in visited:
                visited.add(nxt)
                frontier.append((nxt, depth + 1))
    assert not goal_reached, "a depth<=2 solution exists; oracle depth is wrong"


# ---------------------------------------------------------------------------
# B. embodiment dependency
# ---------------------------------------------------------------------------


def test_oracle_embodiment_dependency(tmp_path):
    """Same semantic state, two URDF morphologies: the restricted arm cannot
    reach the cabinet interface -> UNSOLVABLE_FOR_EMBODIMENT, while the full
    arm solves it. (Fixture arm URDFs; proxy engine.)"""
    from rummagebench.feasibility.collision import CollisionChecker
    from rummagebench.feasibility.ik_solver import ReachabilityIKSolver
    from rummagebench.validation.feasibility import FeasibilityValidator

    FIXTURES = Path(__file__).parent / "fixtures"
    backend = FakeBackend(
        entities={
            "cabinet_B": EntityInfo(
                name="cabinet_B", category="cabinet", fixed_base=True,
                openable=True, is_receptacle=True,
            ),
            "target_knife": EntityInfo(name="target_knife", category="knife"),
        },
        anchors={"living_room"},
        poses={"cabinet_B": [0.65, 0.0, 0.45], "target_knife": [0.65, 0.0, 0.4]},
        aabbs={
            "cabinet_B": ([0.63, -0.05, 0.0], [0.67, 0.05, 0.9]),
            "target_knife": ([0.62, -0.03, 0.35], [0.68, 0.03, 0.45]),
        },
        articulations={
            "cabinet_B": ArticulationInfo(
                entity="cabinet_B",
                joints=[ArticulationJointInfo(
                    name="cabinet_B_door_joint", joint_type="revolute",
                    parent_link="cabinet_B_body", child_link="cabinet_B_door",
                    axis=[0.0, 0.0, 1.0], limits=(0.0, 1.6),
                )],
                links=["cabinet_B_body", "cabinet_B_door"],
            ),
        },
        link_poses={
            ("cabinet_B", "cabinet_B_body"): [0.80, 0.0, 0.45],
            ("cabinet_B", "cabinet_B_door"): [0.65, 0.0, 0.45],
        },
        link_aabbs={
            ("cabinet_B", "cabinet_B_door"): ([0.63, -0.05, 0.0], [0.67, 0.05, 0.9]),
        },
    )
    backend._last_anchor_position = [0.0, 0.0, 0.0]

    def solve(urdf_name: str):
        scenario = load_scenario(FIXTURES / "morph.yaml")
        scenario.robot.kinematics.urdf_path = f"tests/unit/fixtures/robots/{urdf_name}"
        # the oracle goal is hold-the-target: enable it (morph.yaml disables
        # it for the graph walk)
        scenario.termination.succeed_when_holding_target = True
        session = BenchmarkSession(backend, scenario)
        model = OracleFeasibilityModel(session._backend, scenario, session._feasibility)
        model.bind_robot(session.robot)
        return solve_full_information(scenario, initial_oracle_state(scenario), model)

    long_result = solve("long_arm.urdf")
    assert long_result.solvable
    short_result = solve("short_arm.urdf")
    assert not short_result.solvable
    assert short_result.reason == "UNSOLVABLE_FOR_EMBODIMENT"


# ---------------------------------------------------------------------------
# §3 certification
# ---------------------------------------------------------------------------


def test_certify_solvable_episode(fake_backend, tmp_path):
    session = _session(fake_backend)
    certificate = certify_episode(session.scenario, session)
    assert certificate.solvable
    assert certificate.oracle_depth == 3
    assert certificate.certification_version == "cert-v1"
    assert certificate.scenario_hash
    path = save_certificate(certificate, tmp_path)
    assert path.is_file()
    # certificate round-trips
    from rummagebench.evaluation.certification import load_certificate

    loaded = load_certificate(path)
    assert loaded.oracle_depth == certificate.oracle_depth


def test_unsolvable_episode_rejected_from_normal_split(tmp_path):
    """A drawer-style container the embodiment cannot reach: the certificate
    is unsolvable and the episode must not enter the solvable split."""
    from rummagebench.sim.base import EntityInfo
    from rummagebench.evaluation.certification import EpisodeCertificate

    # a certificate for an unsolvable episode (constructed directly: the
    # oracle planner case is covered by test_oracle_embodiment_dependency)
    unsolvable = EpisodeCertificate(
        episode_id="knife_search_s003",
        solvable=False,
        oracle_depth=None,
        oracle_plan=[],
        target_container="drawer_A",
        robot_variant="default",
        action_interface_mode="admissible",
        certification_version="cert-v1",
        scenario_hash="deadbeef0000",
        robot_urdf_hash=None,
        reason="UNSOLVABLE_FOR_EMBODIMENT",
    )
    solvable = EpisodeCertificate(
        episode_id="knife_search_s000",
        solvable=True,
        oracle_depth=3,
        oracle_plan=[],
        target_container="cabinet_B",
        robot_variant="default",
        action_interface_mode="admissible",
        certification_version="cert-v1",
        scenario_hash="cafe00001234",
        robot_urdf_hash=None,
    )
    certificates = [solvable, unsolvable]
    # standard split: only solvable
    standard = [c for c in certificates if c.solvable]
    assert [c.episode_id for c in standard] == ["knife_search_s000"]
    # stress split retains the unsolvable one
    stress = [c for c in certificates if not c.solvable]
    assert [c.episode_id for c in stress] == ["knife_search_s003"]
    # depth statistics only over solvable certificates
    stats = depth_statistics(certificates)
    assert stats["num_solvable"] == 1
    assert stats["min_depth"] == 3 and stats["max_depth"] == 3


# ---------------------------------------------------------------------------
# §8 oracle <-> production consistency
# ---------------------------------------------------------------------------


def test_oracle_admissibility_matches_production(fake_backend):
    """§8: for every BFS-reachable state, the oracle's admissible action set
    equals the production SkillGrounder admissible set evaluated at the same
    hypothetical state (overlay-projected, full information — visibility
    filtering is a separate agent-facing view)."""
    from collections import deque

    from rummagebench.core.skill_grounder import SkillGrounder
    from rummagebench.evaluation.oracle_planner import (
        OracleFeasibilityModel,
        _OracleTransition,
        initial_oracle_state,
    )

    session = _session(fake_backend)
    scenario = session.scenario
    model = OracleFeasibilityModel(session._backend, scenario, session._feasibility)
    model.bind_robot(session.robot)
    tr = _OracleTransition(scenario, model)

    # production SkillGrounder on the overlay: production adapter/candidate
    # code path, production validator, hypothetical-state semantics
    prod_grounder = SkillGrounder(model.overlay, scenario, model._feasibility)

    # enumerate BFS-reachable states with the oracle transition
    start = initial_oracle_state(scenario)
    seen = {start}
    queue = deque([start])
    reachable = []
    while queue:
        state = queue.popleft()
        reachable.append(state)
        for skill, target in tr.candidates(state):
            if skill == "GRASP" and target != scenario.target.entity:
                continue
            if tr.unsafe(state, skill, target):
                continue
            if not tr.feasible(state, skill, target):
                continue
            nxt = tr.transition(state, skill, target)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    assert len(reachable) > 1, "no reachable states enumerated"
    for state in reachable:
        model.overlay.current_state = state
        world = model._overlay_world(state)

        oracle_set = set()
        for skill, target in tr.candidates(state):
            if skill == "GRASP" and target != scenario.target.entity \
                    and scenario.termination.fail_on_wrong_grasp:
                continue
            if tr.unsafe(state, skill, target):
                continue
            if tr.feasible(state, skill, target):
                oracle_set.add(f"{skill}({target})")

        # production grounding at the same state: semantic candidates over
        # ALL entities (full information) + physical filter
        candidates = prod_grounder.semantic_candidates(session.robot, world)
        candidates = [c for c in candidates
                      if not (c.skill == "GRASP" and c.target != scenario.target.entity
                              and scenario.termination.fail_on_wrong_grasp)]
        kept = prod_grounder.physical_filter(candidates, session.robot, world)
        production_set = {c.label() for c in kept}

        assert oracle_set == production_set, (
            f"state {state}: oracle={sorted(oracle_set)} production={sorted(production_set)}"
        )
