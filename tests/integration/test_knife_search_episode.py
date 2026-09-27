"""The complete knife_search_001 episode with all four terminal outcomes."""

import pytest

pytestmark = pytest.mark.sim

from rummagebench.core.scenario import load_scenario
from rummagebench.core.types import EpisodeStatus
from rummagebench.evaluation.episode_log import run_episode
from rummagebench.evaluation.evaluator import make_agent, validate_agent_sequence

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = REPO_ROOT / "scenarios" / "knife_search_001" / "scenario.yaml"


def test_scripted_success_episode(sim_session, tmp_path):
    scenario = sim_session.scenario
    agent = make_agent("scripted", scenario)
    validate_agent_sequence("scripted", scenario)
    summary = run_episode(sim_session, agent, tmp_path / "success", save_images=True)
    assert summary["status"] == "SUCCESS"
    assert (tmp_path / "success" / "summary.json").exists()
    # events.jsonl lives at the session's log path (conftest run dir)
    assert sim_session._log is not None and sim_session._log.count > 0


def test_wrong_object_episode(sim_session, tmp_path):
    scenario = sim_session.scenario
    agent = make_agent("wrong_object", scenario)
    summary = run_episode(sim_session, agent, tmp_path / "wrong")
    assert summary["status"] == "FAIL_WRONG_TARGET"


def test_timeout_episode(sim_session, tmp_path):
    scenario = sim_session.scenario
    agent = make_agent("timeout", scenario)
    summary = run_episode(sim_session, agent, tmp_path / "timeout")
    assert summary["status"] == "FAIL_MAX_STEPS"


def test_unsafe_episode(sim_session, tmp_path):
    scenario = sim_session.scenario
    agent = make_agent("unsafe", scenario)
    validate_agent_sequence("unsafe", scenario)
    summary = run_episode(sim_session, agent, tmp_path / "unsafe")
    assert summary["status"] == "FAIL_UNSAFE_ACTION"
