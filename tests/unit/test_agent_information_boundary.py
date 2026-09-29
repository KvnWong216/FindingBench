"""§39 information boundary: AGENT mode leaks nothing; ORACLE keeps legacy."""

import json


def test_agent_observation_carries_no_entity_names(visual_session):
    obs = visual_session.step({"skill": "MOVE", "distance_cm": 10}).observation
    payload = json.dumps(obs.model_dump())
    for entity in ("cabinet_B", "distractor_spoon", "target_knife", "drawer_A",
                   "cabinet_A", "countertop", "hot_pot"):
        assert entity not in payload
    assert "available_skills" not in payload
    assert "candidate_skills" not in payload
    assert "state_update" not in payload
    assert "robot" not in payload


def test_agent_observation_carries_no_geometry(visual_session):
    obs = visual_session.step({"skill": "TURN", "angle_deg": 30}).observation
    payload = json.dumps(obs.model_dump())
    for key in ("segmentation", "depth", "extrinsics", "intrinsics",
                "instance", "interaction"):
        assert key not in payload


def test_agent_mode_rejects_entity_id_actions(visual_session):
    r = visual_session.step(
        {"skill": "OPEN", "target": "cabinet_B"}  # legacy entity form
    )
    assert r.feedback.code == "INVALID_ACTION"
    assert r.episode_status == "RUNNING"


def test_agent_observation_fixed_skill_library(visual_session):
    obs = visual_session.step({"skill": "REPORT_DONE"}).observation
    assert obs.skill_library == [
        "MOVE", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE", "REPORT_DONE",
    ]


def test_oracle_mode_still_accepts_entity_actions(visual_backend):
    from rummagebench.core.session import BenchmarkSession
    from rummagebench.core.scenario import load_scenario
    from rummagebench.core.types import Action, TargetKind, TargetRef

    scenario = load_scenario(
        __import__("pathlib").Path(__file__).parent / "fixtures" / "mini.yaml"
    )
    session = BenchmarkSession(visual_backend, scenario)
    session.reset()
    session.act(Action(skill="NAV",
                       target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    r = session.act(Action(skill="OPEN",
                           target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    assert r.executed
