import numpy as np
from scipy.spatial.transform import Rotation


def test_embodiment_opt_in_and_no_scene_identity(visual_session):
    assert visual_session.observe().embodiment is None
    visual_session.expose_embodiment=True
    body=visual_session.observe().embodiment
    assert body.executor_footprint_half_extent_m==.4
    assert not body.executor_holding
    assert body.view_yaw_deg is None
    assert set(body.model_dump())=={
        'version','robot_model','executor_footprint_half_extent_m','hand_capacity',
        'end_effector_link','executor_holding','projection','image_size',
        'view_yaw_deg','view_pitch_deg','camera_origin_in_base_m',
        'body_forward_in_camera','body_up_in_camera','perspective_hfov_deg'}


def test_calibration_is_invariant_to_world_transform(visual_session, monkeypatch):
    visual_session.expose_embodiment=True
    frame=visual_session._store.get(visual_session._store.current_id)
    position,quat=visual_session._backend.robot_pose()
    before=visual_session._embodiment_observation(frame)
    turn=Rotation.from_euler('z',73,degrees=True)
    shift=np.array([8.,-9.,2.])
    original=frame.camera_extrinsics.copy()
    transformed=original.copy()
    transformed[:3,:3]=turn.as_matrix() @ original[:3,:3]
    transformed[:3,3]=turn.apply(original[:3,3])+shift
    frame.camera_extrinsics=transformed
    monkeypatch.setattr(visual_session._backend,'robot_pose',lambda:(
        turn.apply(position)+shift,(turn*Rotation.from_quat(quat)).as_quat()))
    after=visual_session._embodiment_observation(frame)
    for field in ['camera_origin_in_base_m','body_forward_in_camera','body_up_in_camera']:
        np.testing.assert_allclose(getattr(before,field),getattr(after,field),atol=1e-8)


def test_view_calibration_rotates_axes_not_camera_origin(visual_session):
    visual_session.expose_embodiment=True
    frame=visual_session._store.get(visual_session._store.current_id)
    frame.meta["projection"]="perspective"
    before=visual_session._embodiment_observation(frame)
    visual_session._view_angles=(90,0)
    after=visual_session._embodiment_observation(frame)
    expected=Rotation.from_euler('y',-90,degrees=True).inv().apply(before.body_forward_in_camera)
    np.testing.assert_allclose(after.body_forward_in_camera,expected,atol=1e-8)
    assert before.camera_origin_in_base_m==after.camera_origin_in_base_m
    assert after.view_yaw_deg==90


def test_budget_applies_to_actual_session_and_termination(visual_session):
    import pytest
    visual_session.set_planning_budget(2)
    assert visual_session.observe().max_planning_steps==2
    first=visual_session.step({'skill':'TURN','angle_deg':10})
    assert first.episode_status=='RUNNING'
    with pytest.raises(RuntimeError,match='during an episode'):
        visual_session.set_planning_budget(3)
    second=visual_session.step({'skill':'TURN','angle_deg':10})
    assert second.episode_status=='FAIL_MAX_STEPS'
    assert second.planning_step==2


def test_configured_footprint_is_shared_by_navigation_and_open(visual_session):
    from rummagebench.core.visual_session import VisualProtocolSession
    scenario=visual_session._scenario.model_copy(deep=True)
    scenario.robot.base_half_extent=.25
    session=VisualProtocolSession(visual_session._backend,scenario)
    assert session._base_half_extent==.25
    assert session._session._feasibility.base_half_extent==.25
