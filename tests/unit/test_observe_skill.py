"""OBSERVE was removed in agent-visual-v2; old requests must not execute."""

def test_removed_observe_is_rejected_without_moving(visual_session, visual_backend):
    before = visual_backend.robot_pose()
    frame = visual_session.observe().frame_id
    result = visual_session.step({"skill": "OBSERVE", "point": {"frame_id": frame, "x": .5, "y": .5}})
    assert result.feedback.code == "INVALID_ACTION"
    assert result.planning_step == 1
    assert visual_backend.robot_pose() == before
    assert "observe_views" not in result.observation.model_dump()
