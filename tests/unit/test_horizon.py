"""Session-level tests: horizon, success, wrong target, unsafe branch.

These run the full benchmark core against the in-memory fake backend, which
proves the three MVP failure categories plus success are backend-independent.
"""

from pathlib import Path

from rummagebench.core.events import load_events
from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, EpisodeStatus, TargetKind, TargetRef

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(fake_backend, tmp_path, log=True):
    scenario = load_scenario(FIXTURE)
    log_path = tmp_path / "events.jsonl" if log else None
    return BenchmarkSession(fake_backend, scenario, log_path=log_path)


def test_scripted_success(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    for raw in load_scenario(FIXTURE).agent.scripted_success:
        result = session.act(Action.from_dict(raw))
        assert result.episode_status == EpisodeStatus.RUNNING or raw["skill"] == "GRASP"
    assert session.status() == EpisodeStatus.SUCCESS


def test_wrong_target_fails_immediately(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    result = session.act(
        Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value="distractor_spoon"))
    )
    assert session.status() == EpisodeStatus.FAIL_WRONG_TARGET
    assert result.failure_reason.value == "WRONG_TARGET"


def test_unsafe_action_not_executed_and_fails(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    result = session.act(
        Action(skill="GRASP", target=TargetRef(type=TargetKind.ENTITY, value="countertop"))
    )
    assert session.status() == EpisodeStatus.FAIL_UNSAFE_ACTION
    assert result.executed is False
    assert fake_backend.holding_entity is None  # never executed


def test_horizon_exceeded(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    status = None
    for _ in range(20):
        result = session.act(
            Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen"))
        )
        status = result.episode_status
        if status != EpisodeStatus.RUNNING:
            break
    assert status == EpisodeStatus.FAIL_MAX_STEPS


def test_invalid_actions_keep_episode_running(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    result = session.act(
        Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="nowhere"))
    )
    assert session.status() == EpisodeStatus.RUNNING
    assert result.executed is False
    assert result.failure_reason.value == "INVALID_ACTION"
    # CLOSE is now a legal skill; closing an already-closed container is a
    # state-feasibility failure (INVALID_STATE), still non-terminal
    result = session.act(
        Action(skill="CLOSE", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B"))
    )
    assert result.failure_reason.value == "INVALID_STATE"
    assert session.status() == EpisodeStatus.RUNNING


def test_reset_restores_semantic_state(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    assert fake_backend.is_open("cabinet_B")
    session.reset()
    assert not fake_backend.is_open("cabinet_B")
    assert fake_backend.holding_entity is None
    assert session.status() == EpisodeStatus.RUNNING


def test_events_written_as_jsonl(fake_backend, tmp_path):
    session = _session(fake_backend, tmp_path)
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    events = load_events(tmp_path / "events.jsonl")
    assert len(events) == 2
    e = events[-1]
    assert e["episode_id"] == "mini_search_000"
    assert e["action"]["skill"] == "OPEN"
    assert "validation" in e and "execution" in e
