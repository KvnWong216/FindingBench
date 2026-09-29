"""§39 REPORT_DONE: only success trigger; false completion is terminal."""

from rummagebench.core.scenario import AnchorSpec


def _teleport_kitchen(visual_backend):
    visual_backend.teleport_robot(AnchorSpec(position=[3.0, 1.0, 0.0],
                                             orientation=[0.0, 0.0, 0.0, 1.0]))


def _grasp(visual_session, x, y):  # noqa
    return visual_session.step(
        {"skill": "GRASP", "point": {"frame_id": visual_session._store.current_id,
                                     "x": x, "y": y}}
    )


def test_wrong_object_grasp_is_recoverable(visual_session, visual_backend):
    _teleport_kitchen(visual_backend)
    visual_backend.set_open("cabinet_B", True)  # the spoon lies in its volume
    r = _grasp(visual_session, 0.75, 0.5)  # distractor_spoon region of the frame
    assert r.feedback.code == "EXECUTED", r.feedback
    assert r.episode_status == "RUNNING"  # recoverable, not terminal


def test_correct_target_grasp_does_not_auto_succeed(visual_session, visual_backend):
    _teleport_kitchen(visual_backend)
    visual_backend.set_open("cabinet_B", True)  # expose the knife
    visual_backend.frame_entities["target_knife"] = (24, 28, 10, 10)
    visual_session._capture_current()
    r = _grasp(visual_session, 0.46, 0.52)  # knife square center
    assert r.feedback.code == "EXECUTED", r.feedback
    assert r.episode_status == "RUNNING"  # NOT auto-success (§15)
    assert visual_backend.is_holding("target_knife")


def test_report_done_on_satisfied_goal_success(visual_session, visual_backend):
    _teleport_kitchen(visual_backend)
    visual_backend.set_open("cabinet_B", True)
    visual_backend.frame_entities["target_knife"] = (24, 28, 10, 10)
    visual_session._capture_current()
    r = _grasp(visual_session, 0.46, 0.52)
    assert r.feedback.code == "EXECUTED", r.feedback
    r = visual_session.step({"skill": "REPORT_DONE"})
    assert r.episode_status == "SUCCESS"
    assert r.feedback.code == "EXECUTED"


def test_report_done_on_unsatisfied_goal_terminal(visual_session):
    r = visual_session.step({"skill": "REPORT_DONE"})
    assert r.episode_status == "FAIL_FALSE_COMPLETION"
    assert visual_session.step({"skill": "MOVE", "distance_cm": 10}).episode_status \
        == "FAIL_FALSE_COMPLETION"  # terminal: no further actions


def test_horizon_exhaustion_terminal(visual_session):
    status = None
    for _ in range(visual_session._scenario.termination.max_planning_steps + 2):
        r = visual_session.step({"skill": "MOVE", "distance_cm": 10})
        status = r.episode_status
    assert status == "FAIL_MAX_STEPS"
    # action H was the last allowed one: no hidden H+1
    assert visual_session._public_step <= \
        visual_session._scenario.termination.max_planning_steps
