"""§7-9 dual action-interface protocol + §19-22 visibility tests.

candidate mode exposes semantic+state-valid skills for VISIBLE objects only,
with no IK/collision metadata; hidden objects (closed-container contents)
never leak into any agent-facing protocol. Admissible mode keeps the full
physical filter.
"""

from pathlib import Path

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.evaluation.metrics import compute_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(fake_backend, mode: str = "admissible") -> BenchmarkSession:
    scenario = load_scenario(FIXTURE)
    scenario.action_interface.mode = mode
    scenario.termination.succeed_when_holding_target = False
    scenario.termination.fail_on_wrong_grasp = False
    # visibility model: the knife is inside cabinet_B (mirrors the placement)
    fake_backend.contains = {"cabinet_B": ["target_knife"]}
    return BenchmarkSession(fake_backend, scenario)


# ---------------------------------------------------------------------------
# §22 visibility tests
# ---------------------------------------------------------------------------


def test_hidden_target_not_exposed_in_candidate_mode(fake_backend):
    """Test 1: knife in CLOSED cabinet -> GRASP(knife) never appears."""
    session = _session(fake_backend, mode="candidate")
    session.reset()
    obs = session.observe()
    assert obs.candidate_skills, "candidate mode exposes nothing at all"
    assert not any("target_knife" in s for s in obs.candidate_skills)
    assert "GRASP(target_knife)" not in obs.candidate_skills


def test_open_container_reveals_target(fake_backend):
    """Test 2: after OPEN(cabinet) the target becomes observable."""
    session = _session(fake_backend, mode="candidate")
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    obs = session.observe()
    assert "GRASP(target_knife)" in obs.candidate_skills


def test_other_closed_container_contents_stay_hidden(fake_backend):
    """Test 3: distractor in another closed container is not exposed."""
    session = _session(fake_backend, mode="candidate")
    fake_backend.contains = {
        "cabinet_B": ["target_knife"],
        "cabinet_A": ["distractor_spoon"],
    }
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    session.act(Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="cabinet_B")))
    obs = session.observe()
    assert "GRASP(target_knife)" in obs.candidate_skills
    assert not any("distractor_spoon" in s for s in obs.candidate_skills)


def test_visibility_also_gates_admissible_mode(fake_backend):
    session = _session(fake_backend, mode="admissible")
    session.reset()
    session.act(Action(skill="NAV", target=TargetRef(type=TargetKind.PLACE, value="kitchen")))
    assert "OPEN(cabinet_B)" in session.available_skills()
    assert not any("target_knife" in s for s in session.available_skills())


# ---------------------------------------------------------------------------
# §7-9 dual protocol
# ---------------------------------------------------------------------------


def test_candidate_mode_exposes_physically_infeasible_actions(fake_backend):
    """candidate mode lists state-valid skills without the physical filter;
    attempting them still runs the real feasibility oracle."""
    session = _session(fake_backend, mode="candidate")
    session.reset()
    obs = session.observe()
    # drawer_A is far from the living_room anchor: admissible mode filters it,
    # candidate mode lists it
    assert "OPEN(drawer_A)" in obs.candidate_skills
    assert "OPEN(drawer_A)" not in obs.available_skills
    # no feasibility metadata anywhere in the observation
    assert obs.previous_action_result is None


def test_candidate_mode_attempt_runs_real_feasibility_oracle(fake_backend):
    session = _session(fake_backend, mode="candidate")
    session.reset()
    result = session.act(
        Action(skill="OPEN", target=TargetRef(type=TargetKind.ENTITY, value="drawer_A"))
    )
    assert result.executed is False
    # candidate mode exposes it, but the attempt lands outside the
    # manipulation space -> INVALID_ACTION with the attribution flag
    assert result.failure_reason.value == "INVALID_ACTION"
    assert any(ev.get("not_in_manipulation_space") for ev in result.events)
    assert session.status().value == "RUNNING"


def test_admissible_mode_unchanged_by_candidate_fields(fake_backend):
    session = _session(fake_backend, mode="admissible")
    session.reset()
    obs = session.observe()
    assert obs.candidate_skills == []
    assert "NAV(kitchen)" in obs.available_skills


# ---------------------------------------------------------------------------
# §10/§23 metrics
# ---------------------------------------------------------------------------


def _events_for_rates():
    return [
        {"action": {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}},
         "execution": {"executed": True, "failure_reason": "NONE", "postcondition_satisfied": True},
         "validation": {"semantic_valid": True, "target_valid": True, "safe": True},
         "events": [],
         "status": "RUNNING"},
        {"action": {"skill": "OPEN", "target": {"type": "entity", "value": "cabinet_B"}},
         "execution": {"executed": True, "failure_reason": "NONE", "postcondition_satisfied": True},
         "validation": {"semantic_valid": True, "target_valid": True, "safe": True},
         "events": [],
         "status": "RUNNING"},
        {"action": {"skill": "OPEN", "target": {"type": "entity", "value": "drawer_A"}},
         "execution": {"executed": False, "failure_reason": "INVALID_ACTION"},
         "events": [{"not_in_manipulation_space": True}],
         "validation": {"semantic_valid": True, "target_valid": True, "safe": True},
         "status": "RUNNING"},
        {"action": {"skill": "GRASP", "target": {"type": "entity", "value": "cup"}},
         "execution": {"executed": False, "failure_reason": "COLLISION"},
         "events": [],
         "validation": {"semantic_valid": True, "target_valid": True, "safe": True},
         "status": "RUNNING"},
        {"action": {"skill": "GRASP", "target": {"type": "entity", "value": "countertop"}},
         "execution": {"executed": False, "failure_reason": "UNSAFE_ACTION"},
         "events": [],
         "validation": {"semantic_valid": True, "target_valid": True, "safe": False},
         "status": "FAIL_UNSAFE_ACTION"},
    ]


def test_embodiment_awareness_rates():
    metrics = compute_metrics(_events_for_rates())
    # 4 interaction attempts: 1 OK, 1 UNREACHABLE, 1 COLLISION, 1 UNSAFE
    assert metrics["interaction_attempts"] == 4
    assert metrics["not_in_space_attempt_rate"] == 0.25
    assert metrics["collision_attempt_rate"] == 0.25
    assert metrics["infeasible_attempt_rate"] == 0.5
    assert metrics["wrong_target_rate"] == 0.0
    assert metrics["unsafe_action_rate"] == 0.2  # 1 unsafe / 5 actions
    assert metrics["task_success"] is False
    assert metrics["search_efficiency"] is None  # no oracle minimum provided


def test_search_efficiency_with_oracle_minimum():
    events = _events_for_rates()
    events[-1]["status"] = "SUCCESS"
    events[-1]["execution"] = {"executed": True, "failure_reason": "NONE",
                               "postcondition_satisfied": True}
    metrics = compute_metrics(events, oracle_min_steps=3)
    assert metrics["task_success"] is True
    assert metrics["search_efficiency"] == round(3 / 5, 3)
