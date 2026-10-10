"""R11: oracle search-limit semantics.

A depth-limited BFS that truncated unexplored states must report
SEARCH_LIMIT_REACHED — never UNSOLVABLE_FOR_EMBODIMENT. Only a fully
explored finite abstract graph may claim unsolvability.
"""

from pathlib import Path

from conftest import FakeBackend
from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.evaluation.oracle_planner import (
    OracleFeasibilityModel,
    initial_oracle_state,
    solve_full_information,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _model_and_initial(fake_backend):
    scenario = load_scenario(FIXTURE)
    scenario.termination.succeed_when_holding_target = True
    session = BenchmarkSession(fake_backend, scenario)
    model = OracleFeasibilityModel(session._backend, scenario, session._feasibility)
    model.bind_robot(session.robot)
    return scenario, model, initial_oracle_state(scenario)


def test_depth_truncation_reports_search_limit(fake_backend):
    """mini.yaml needs 3 actions; a max_depth of 2 truncates the frontier —
    the run must say the search hit its limit, not that the task is
    unsolvable for the embodiment."""
    scenario, model, initial = _model_and_initial(fake_backend)
    result = solve_full_information(scenario, initial, model, max_depth=2)
    assert not result.solvable
    assert result.reason == "SEARCH_LIMIT_REACHED"
    assert result.depth is None


def test_sufficient_depth_finds_the_plan(fake_backend):
    scenario, model, initial = _model_and_initial(fake_backend)
    result = solve_full_information(scenario, initial, model, max_depth=24)
    assert result.solvable
    assert result.depth == 3
    assert result.reason != "SEARCH_LIMIT_REACHED"


def test_exhausted_graph_is_the_unsolvability_proof(fake_backend):
    """With a depth bound well above any plan, exhaustion of the finite
    abstract graph is the actual no-solution evidence. mini.yaml is solvable,
    so simulate exhaustion by disabling the goal rule: the goal can never be
    satisfied, every state is explored, and the run reports the exhausted
    graph (not a search limit)."""
    scenario, model, initial = _model_and_initial(fake_backend)
    # no goal rule can ever fire (succeed_when_holding_target False -> goal()
    # always False) -> the whole finite graph is explored
    scenario.termination.succeed_when_holding_target = False
    result = solve_full_information(scenario, initial, model, max_depth=24)
    assert not result.solvable
    assert result.reason == "EXHAUSTED_ABSTRACT_GRAPH"
