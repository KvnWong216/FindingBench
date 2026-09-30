"""Public/ORACLE adapter migration regressions; no simulator required."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rummagebench.adapters import mcp_server, python_api
from rummagebench.cli import _scenario_file
from rummagebench.core.public_types import AgentProtocolConfig, PublicStepResult
from rummagebench.evaluation.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]


def test_repository_relative_scenario_paths(monkeypatch):
    monkeypatch.delenv("RUMMAGEBENCH_ROOT", raising=False)
    expected = ROOT / "scenarios/knife_search_001/scenario.yaml"
    assert _scenario_file("knife_search_001") == expected
    assert mcp_server._scenario_path("knife_search_001") == expected


def test_legacy_factory_call_sites_explicitly_select_oracle():
    for relative in ("src/rummagebench/cli.py", "scripts/certify_split.py",
                     "scripts/probe_memory.py", "scripts/run_acceptance_v3.py"):
        tree = ast.parse((ROOT / relative).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "create_session"]
        assert calls, relative
        for call in calls:
            mode = next((k.value for k in call.keywords if k.arg == "mode"), None)
            assert isinstance(mode, ast.Constant) and mode.value == "oracle", relative


def test_unknown_factory_mode_fails_before_simulator_launch(monkeypatch):
    monkeypatch.setattr(python_api, "OmniGibsonBackend",
                        lambda **kw: pytest.fail("simulator must not launch"))
    with pytest.raises(ValueError):
        python_api.create_session("unused.yaml", mode="agnet")


def test_mcp_worker_public_observe_and_json_reset(monkeypatch, visual_session):
    monkeypatch.setattr(python_api, "create_session", lambda *a, **kw: visual_session)

    class Connection:
        def __init__(self):
            self.sent = []
            self.commands = iter([("observe", None), ("act", {"skill": "REPORT_DONE"}),
                                  ("status", None), ("exit", None)])
        def send(self, data):
            json.dumps(data)  # no models/arrays cross the transport
            self.sent.append(data)
        def recv(self):
            return next(self.commands)

    conn = Connection()
    mcp_server._worker_main(conn, "unused.yaml", 0)
    assert all(r["ok"] for r in conn.sent)
    reset = conn.sent[0]["reset"]
    observed = conn.sent[1]["observation"]
    assert observed == reset  # observe neither consumes a step nor captures a frame
    assert set(reset) == {"instruction", "frame_id", "image_png_b64", "planning_step",
                          "max_planning_steps", "skill_library", "feedback", "observe_views"}
    assert conn.sent[2]["episode_status"] == "FAIL_FALSE_COMPLETION"
    assert conn.sent[3]["status"] == "FAIL_FALSE_COMPLETION"


def test_safety_metrics_do_not_count_unrun_validators():
    reasons = ["INVALID_ACTION", "MAX_STEPS", "UNSAFE_ACTION", "COLLISION"]
    events = [{"action": {"skill": "GRASP", "target": {"type": "entity", "value": "x"}},
               "execution": {"executed": False, "failure_reason": reason},
               "validation": {"safe": False, "semantic_valid": False},
               "status": "RUNNING"} for reason in reasons]
    result = compute_metrics(events)
    assert result["safety_failures"] == 1
    assert result["unsafe_action_rate"] == .25


def test_public_status_and_config_are_validated(visual_session):
    payload = visual_session.step({"skill": "REPORT_DONE"}).model_dump()
    payload["episode_status"] = "FAIL_WRONG_TARGET"
    with pytest.raises(ValidationError):
        PublicStepResult.model_validate(payload)
    for kwargs in ({"min_move_cm": 0}, {"max_turn_deg": float("inf")},
                   {"min_move_cm": 20, "max_move_cm": 10}):
        with pytest.raises(ValidationError):
            AgentProtocolConfig(**kwargs)


def test_scenario_accepts_public_protocol_bounds():
    from rummagebench.core.scenario import load_scenario, ScenarioSpec
    spec = load_scenario(ROOT / "tests/unit/fixtures/mini.yaml")
    raw = spec.model_dump()
    raw["agent_protocol"] = {"min_move_cm": 10, "max_move_cm": 50}
    assert ScenarioSpec.model_validate(raw).agent_protocol.max_move_cm == 50


def test_play_terminal_snapshot_and_enqueue(monkeypatch):
    spec = importlib.util.spec_from_file_location("serve_play_contract", ROOT / "scripts/serve_play.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = module.PlayState("unused.yaml", 0)
    state.ready = True
    state.snapshot = {"episode_status": "RUNNING"}
    class Env:
        def step(self, action):
            return {"episode_status": "FAIL_FALSE_COMPLETION", "observation": {"planning_step": 1}}
    state.env = Env()
    state.enqueue({"skill": "REPORT_DONE"})
    def stop_after_iteration(_seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr("time.sleep", stop_after_iteration)
    with pytest.raises(KeyboardInterrupt):
        state.run_worker()
    assert state.state_payload()["episode_status"] == "FAIL_FALSE_COMPLETION"
    assert "error" in state.enqueue({"skill": "MOVE", "distance_cm": 10})


def test_public_factory_run_dir_writes_trace(monkeypatch, visual_backend, tmp_path):
    monkeypatch.setattr(python_api, "OmniGibsonBackend", lambda **kw: visual_backend)
    session = python_api.create_session(ROOT / "tests/unit/fixtures/mini.yaml", run_dir=tmp_path)
    session.reset()
    session.step({"skill": "REPORT_DONE"})
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert records[0]["event"] == "reset"
    assert records[-1]["episode_status"] == "FAIL_FALSE_COMPLETION"
    assert records[-1]["planning_step"] == 1


def test_public_factory_explicit_trace_takes_precedence(monkeypatch, visual_backend, tmp_path):
    monkeypatch.setattr(python_api, "OmniGibsonBackend", lambda **kw: visual_backend)
    trace = tmp_path / "public.jsonl"
    session = python_api.create_session(ROOT / "tests/unit/fixtures/mini.yaml",
                                        run_dir=tmp_path, trace_path=trace)
    session.reset()
    assert trace.exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_environment_bootstraps_robot_urdf(monkeypatch, visual_backend):
    from rummagebench.environment.env import InteractiveSearchEnv
    from rummagebench.sim.omnigibson import backend
    monkeypatch.setattr(backend, "OmniGibsonBackend", lambda **kw: visual_backend)
    calls = []
    monkeypatch.setattr(python_api, "_ensure_kinematics_urdf",
                        lambda b, s, p: calls.append((b, s, p)))
    path = ROOT / "tests/unit/fixtures/mini.yaml"
    env = InteractiveSearchEnv(path)
    assert len(calls) == 1
    assert calls[0][0] is visual_backend and calls[0][2] == path
    assert env.reset()["observation"]["planning_step"] == 0
