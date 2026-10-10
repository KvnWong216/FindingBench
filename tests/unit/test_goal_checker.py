"""R04: shared goal/temporal condition checking.

Data-driven goal predicates and temporal constraints are evaluated by ONE
implementation (evaluation/goal_checker.py) consumed by REPORT_DONE; the
legacy holding(target) default stays for goal-less scenarios. Missing
required events never satisfy a temporal constraint, and the same terminal
state with a different committed-event order can differ in outcome.
"""

from pathlib import Path

from conftest import FakeBackend

from rummagebench.core.scenario import (
    GoalPredicateSpec,
    TemporalConstraintSpec,
    load_scenario,
)
from rummagebench.evaluation.goal_checker import check_goal, check_temporal
from rummagebench.skills.report_done import goal_satisfied
from rummagebench.state.benchmark_state import BenchmarkWorldState

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _scenario():
    return load_scenario(FIXTURE)


def _backend(on_top: bool = False):
    # on_top variant: knife rests on the countertop OUTSIDE cabinet_B's AABB
    knife_lo_hi = (
        ([3.5, 0.5, 0.7], [3.56, 0.56, 0.76]) if on_top
        else ([3.02, 1.02, 0.32], [3.08, 1.08, 0.38])  # inside cabinet_B
    )
    knife_pose = [3.53, 0.53, 0.73] if on_top else [3.05, 1.05, 0.35]
    return FakeBackend(
        entities={
            "target_knife": _entity("target_knife"),
            "cabinet_B": _entity("cabinet_B", openable=True),
            "countertop": _entity("countertop"),
        },
        anchors={"kitchen"},
        poses={
            "target_knife": knife_pose,
            "cabinet_B": [3.0, 1.0, 0.45],
            "countertop": [3.0, 1.0, 0.6],
        },
        aabbs={
            "target_knife": knife_lo_hi,
            "cabinet_B": ([2.7, 0.7, 0.0], [3.3, 1.3, 0.9]),
            "countertop": ([2.4, 0.4, 0.5], [3.6, 1.6, 0.7]),
        },
    )


def _entity(name: str, **kwargs):
    from rummagebench.sim.base import EntityInfo

    return EntityInfo(name=name, category="knife" if "knife" in name else "furniture",
                      **kwargs)


def test_legacy_default_is_holding_target():
    scenario = _scenario()
    state = BenchmarkWorldState()
    assert not goal_satisfied(state, scenario)
    state.grasp("target_knife", None)
    assert goal_satisfied(state, scenario)


def test_goal_predicates_holding_open_closed():
    scenario = _scenario()
    scenario.goal = [
        GoalPredicateSpec(kind="holding", entity="target_knife"),
        GoalPredicateSpec(kind="closed", entity="cabinet_B"),
    ]
    backend = _backend()
    state = BenchmarkWorldState()
    state.grasp("target_knife", None)
    result = check_goal(state, scenario, backend)
    assert result.satisfied

    backend.set_open("cabinet_B", True)
    result = check_goal(state, scenario, backend)
    assert not result.satisfied
    assert result.predicates[1]["satisfied"] is False

    scenario.goal = [GoalPredicateSpec(kind="open", entity="cabinet_B")]
    assert check_goal(state, scenario, backend).satisfied


def test_goal_predicates_inside_and_on_top():
    scenario = _scenario()
    scenario.goal = [GoalPredicateSpec(kind="inside", entity="target_knife",
                                       receptacle="cabinet_B")]
    assert check_goal(BenchmarkWorldState(), scenario, _backend()).satisfied
    assert not check_goal(BenchmarkWorldState(), scenario,
                          _backend(on_top=True)).satisfied

    scenario.goal = [GoalPredicateSpec(kind="on_top", entity="target_knife",
                                       receptacle="countertop")]
    assert check_goal(BenchmarkWorldState(), scenario,
                      _backend(on_top=True)).satisfied


def test_temporal_missing_event_never_satisfies():
    scenario = _scenario()
    scenario.temporal = [
        TemporalConstraintSpec(kind="required_event", event="grasp_executed",
                               entity="target_knife"),
    ]
    result = check_temporal([], scenario)  # nothing committed yet
    assert not result.satisfied
    # absence is not truth: an empty log cannot satisfy a required event
    assert result.constraints[0]["event_index"] is None


def test_temporal_before_ordering():
    scenario = _scenario()
    scenario.temporal = [
        TemporalConstraintSpec(kind="before", event="grasp_executed",
                               entity="target_knife",
                               before_event="open_executed",
                               before_entity="cabinet_B"),
    ]
    grasp_first = [
        {"event": "grasp_executed", "entity": "target_knife"},
        {"event": "open_executed", "entity": "cabinet_B"},
    ]
    open_first = [
        {"event": "open_executed", "entity": "cabinet_B"},
        {"event": "grasp_executed", "entity": "target_knife"},
    ]
    assert check_temporal(grasp_first, scenario).satisfied
    assert not check_temporal(open_first, scenario).satisfied
    # the reference event missing entirely cannot order anything
    assert not check_temporal(
        [{"event": "grasp_executed", "entity": "target_knife"}], scenario
    ).satisfied


def _visual_session(visual_backend, goal=None, temporal=None):
    from rummagebench.core.visual_session import VisualProtocolSession

    scenario = load_scenario(FIXTURE)
    scenario.goal = list(goal or [])
    scenario.temporal = list(temporal or [])
    session = VisualProtocolSession(visual_backend, scenario)
    session.reset()
    return session


def test_report_done_with_goal_spec_success(visual_session, visual_backend):
    """Data-driven goal flows through the same checker as metrics: holding
    the target reports SUCCESS (same outcome as the legacy default)."""
    from rummagebench.core.scenario import AnchorSpec

    session = _visual_session(
        visual_backend, goal=[GoalPredicateSpec(kind="holding",
                                                entity="target_knife")])
    visual_backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                             orientation=[0.0, 0.0, 0.0, 1.0]))
    visual_backend.set_open("cabinet_B", True)
    visual_backend.frame_entities["target_knife"] = (24, 28, 10, 10)
    session._capture_current()
    r = session.step({"skill": "GRASP",
                      "point": {"frame_id": session.observe().frame_id,
                                "x": 0.46, "y": 0.52}})
    assert r.feedback.code.value == "EXECUTED"
    r = session.step({"skill": "REPORT_DONE"})
    assert r.episode_status == "SUCCESS"


def test_report_done_temporal_order_decides(visual_session, visual_backend):
    """Same terminal state (holding the target, container open), different
    committed-event order: grasp-before-open satisfies the 'before'
    constraint, open-before-grasp fails it."""
    from rummagebench.core.scenario import AnchorSpec

    temporal = [TemporalConstraintSpec(
        kind="before", event="grasp_executed", entity="target_knife",
        before_event="open_executed", before_entity="cabinet_B")]

    def run(open_first: bool):
        session = _visual_session(visual_backend, temporal=temporal)
        visual_backend.reset()
        visual_backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                                 orientation=[0.0, 0.0, 0.0, 1.0]))
        visual_backend.frame_entities["target_knife"] = (24, 28, 10, 10)
        if not open_first:
            # move the knife out of the closed cabinet so the FIRST event can
            # be the grasp (the open happens afterwards, while holding)
            visual_backend.poses["target_knife"] = [3.45, 1.45, 0.35]
            visual_backend.aabbs["target_knife"] = ([3.4, 1.4, 0.32],
                                                    [3.5, 1.5, 0.38])
        session._capture_current()

        def grasp():
            r = session.step({"skill": "GRASP", "point": {
                "frame_id": session.observe().frame_id, "x": 0.46, "y": 0.52}})
            assert r.feedback.code.value == "EXECUTED", r.feedback

        def open_cabinet():
            r = session.step({"skill": "OPEN", "point": {
                "frame_id": session.observe().frame_id, "x": 0.1, "y": 0.5}})
            assert r.feedback.code.value == "EXECUTED", r.feedback

        if open_first:
            open_cabinet()
            grasp()
        else:
            grasp()
            open_cabinet()
        return session.step({"skill": "REPORT_DONE"})

    assert run(open_first=False).episode_status == "SUCCESS"
    assert run(open_first=True).episode_status == "FAIL_FALSE_COMPLETION"
