"""Regression tests for the audited RGB/grounding boundary.

CPU fakes test contracts, not real-renderer calibration. Real-host acceptance
must compare known image pixels independently of the transform under test.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from rummagebench.perception.camera_geometry import world_from_usd_camera
from rummagebench.perception.visual_bridge import VisualBridge, VisualGroundingError
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend


@pytest.mark.parametrize("action", [{"skill": "MOVE", "distance_cm": 10},
                                     {"skill": "TURN", "angle_deg": 30}])
def test_motion_returns_current_actionable_frame(visual_session, visual_backend, action):
    old_id = visual_session.observe().frame_id
    old_rgb = visual_session.observe().image_png_b64
    capture = visual_backend.capture_visual_frame

    def changed_capture():
        frame = capture()
        frame.rgb[:] = 37
        return frame

    visual_backend.capture_visual_frame = changed_capture
    result = visual_session.step(action)
    assert result.feedback.code == "EXECUTED"
    assert result.observation.frame_id != old_id
    assert result.observation.image_png_b64 != old_rgb
    assert result.observation.frame_id == visual_session._store.current_id
    assert result.observation.model_dump() == visual_session.observe().model_dump()
    point = {"x": .1, "y": .5, "frame_id": result.observation.frame_id}
    visual_session.step({"skill": "OPEN", "point": point})
    assert visual_session._trace[-1].get("private_reason") != "STALE_FRAME"
    visual_session.step({"skill": "OPEN", "point": {**point, "frame_id": old_id}})
    assert visual_session._trace[-1]["private_reason"] == "STALE_FRAME"


def test_observe_snapshot_does_not_advance(visual_session):
    before = visual_session.observe().model_dump()
    assert visual_session.observe().model_dump() == before
    assert visual_session._public_step == 0


def test_terminal_feedback_matches_snapshot(visual_session):
    visual_session.step({"skill": "REPORT_DONE"})
    result = visual_session.step({"skill": "MOVE", "distance_cm": 10})
    assert result.feedback == result.observation.feedback
    assert result.planning_step == 1


def test_no_segmentation_never_falls_back_to_physics_ray(visual_session, visual_backend):
    visual_session._store.get(visual_session._store.current_id).instance_segmentation[:] = 0
    visual_backend.pixel_ray_hit = lambda *args: pytest.fail("unsafe ray fallback")
    result = visual_session.step({"skill": "GRASP", "point": {
        "frame_id": visual_session.observe().frame_id, "x": .1, "y": .5}})
    assert result.feedback.code == "INVALID_ACTION"
    assert visual_session._trace[-1]["private_reason"] == "VISIBILITY_UNAVAILABLE"


def test_reset_explicitly_rejects_missing_visual_capability(visual_session, visual_backend):
    capture = visual_backend.capture_visual_frame

    def unsupported():
        frame = capture()
        frame.meta.update(visual_grounding_supported=False, unsupported_reason="segmentation")
        return frame

    visual_backend.capture_visual_frame = unsupported
    with pytest.raises(RuntimeError, match="RGB-only raycasts are diagnostic-only"):
        visual_session.reset()


def test_camera_identity_xyzw_has_usd_optical_axes():
    # Independent expected directions from USD: camera -Z forward, +Y up.
    T = world_from_usd_camera([1., 2., 3.], [0., 0., 0., 1.])
    np.testing.assert_allclose(T @ [0., 0., 2., 1.], [1., 2., 1., 1.])
    np.testing.assert_allclose(T @ [0., 1., 0., 0.], [0., -1., 0., 0.])
    np.testing.assert_allclose(T @ [1., 0., 0., 0.], [1., 0., 0., 0.])


def test_camera_rotated_xyzw_does_not_reorder_components():
    # +90 degrees about world Y sends camera's forward -Z toward world -X.
    T = world_from_usd_camera([0., 0., 0.], [0., 2**-.5, 0., 2**-.5])
    np.testing.assert_allclose(T[:3, :3] @ [0., 0., 1.], [-1., 0., 0.], atol=1e-12)


def test_camera_rejects_invalid_pose():
    with pytest.raises(ValueError):
        world_from_usd_camera([0., 0., 0.], [0., 0., 0., 0.])


def capture_backend(data, second=None, labels=None):
    K = np.array([[30., 0., 15.5], [0., 30., 7.5], [0., 0., 1.]])
    sensor = SimpleNamespace(intrinsic_matrix=K,
                             get_position_orientation=lambda: ([1., 2., 3.], [0., 0., 0., 1.]))
    backend = OmniGibsonBackend()
    backend._robot = SimpleNamespace(name="robot", sensors={"head": sensor})
    obs = {"head": data}
    if second is not None:
        obs["wrist"] = second
    infos = [{"robot": {"head": {"seg_instance": labels or {"2": "cabinet"}},
                         "wrist": {"seg_instance": {"9": "wrong"}}}}]
    backend._env = SimpleNamespace(get_obs=lambda: ([{"robot": obs}], infos))
    backend._entity_infos = {"cabinet": object(), "other": object()}
    backend.entity_names = lambda: ["cabinet", "other"]
    return backend


def data_frame(depth_key="depth"):
    return {"rgb": np.zeros((16, 32, 3), dtype=np.uint8),
            depth_key: np.ones((16, 32)),
            "seg_instance": np.full((16, 32), 2)}


@pytest.mark.parametrize("depth_key, convention", [("depth", "euclidean_range"),
                                                    ("depth_linear", "z_depth")])
def test_capture_uses_actual_depth_modality(depth_key, convention):
    backend = capture_backend(data_frame(depth_key))
    frame = backend.capture_visual_frame()
    assert frame.depth_convention == convention
    assert frame.meta["visual_grounding_supported"]
    assert frame.meta["calibration_verified"] is False
    assert frame.camera_intrinsics[0, 0] == frame.camera_intrinsics[1, 1]
    assert backend.instance_to_entity(2, frame) == "cabinet"
    np.testing.assert_allclose(frame.camera_extrinsics[:3, :3], np.diag([1., -1., -1.]))


def test_capture_does_not_mix_camera_modalities():
    backend = capture_backend({"rgb": np.zeros((16, 32, 3), dtype=np.uint8)}, data_frame())
    frame = backend.capture_visual_frame()
    assert not frame.meta["visual_grounding_supported"]
    assert np.isnan(frame.depth).all()
    assert not frame.instance_segmentation.any()


def test_frame_mapping_and_arrays_are_immutable_across_capture():
    data = data_frame()
    backend = capture_backend(data)
    frame = backend.capture_visual_frame()
    backend._instance_labels = {"2": "other"}
    data["seg_instance"][:] = 9
    data["depth"][:] = 99
    assert backend.instance_to_entity(2, frame) == "cabinet"
    assert backend.entity_visible_pixels(frame, "cabinet") == 16*32
    assert (frame.depth == 1).all()


def test_visible_count_sums_multiple_instance_ids():
    backend = capture_backend(data_frame(), labels={"2": "cabinet", "3": "cabinet"})
    frame = backend.capture_visual_frame()
    frame.instance_segmentation[:8] = 3
    assert backend.entity_visible_pixels(frame, "cabinet") == 16*32


def test_surface_is_local_to_clicked_patch():
    from rummagebench.perception.frame_store import VisualFramePrivate
    from rummagebench.perception.camera_geometry import default_intrinsics
    seg = np.full((64, 64), 2)
    depth = np.full((64, 64), 9.)
    depth[10:21, 10:21] = 1.
    frame = VisualFramePrivate("f", np.zeros((64, 64, 3), np.uint8), depth, seg,
                               default_intrinsics(64, 64), np.eye(4), 64, 64, "z_depth")
    bridge = VisualBridge()
    _, near, _ = bridge.resolve(frame, 15/63, 15/63, lambda _: "object")
    _, far, _ = bridge.resolve(frame, 45/63, 45/63, lambda _: "object")
    assert near[2] == 1 and far[2] == 9
    depth[30:] = 1000
    _, still_near, _ = bridge.resolve(frame, 15/63, 15/63, lambda _: "object")
    np.testing.assert_allclose(near, still_near)
    frame.depth_convention = "unknown"
    with pytest.raises(VisualGroundingError, match="INVALID_DEPTH_GEOMETRY"):
        bridge.resolve(frame, 15/63, 15/63, lambda _: "object")


def test_observe_restores_world_after_render_failure(visual_backend):
    from rummagebench.skills.observe import ObserveConfig, run_observe
    visual_backend.opens.add("cabinet_A")
    visual_backend.holding_entity = "distractor_spoon"
    before = visual_backend.dump_state()
    pose = visual_backend.robot_pose()

    def broken_render():
        visual_backend.opens.clear()
        visual_backend.holding_entity = None
        raise RuntimeError("render fault")

    with pytest.raises(RuntimeError, match="OBSERVE rendering failed"):
        run_observe(visual_backend, "hot_pot", ObserveConfig(), .3, .03,
                    broken_render, lambda *args: 100, settle=lambda _: None)
    assert visual_backend.dump_state() == before
    assert visual_backend.robot_pose()[0] == pose[0]


def test_observe_restore_even_if_log_raises(visual_backend):
    from rummagebench.skills.observe import ObserveConfig, run_observe
    before = visual_backend.robot_pose()

    def fail_log(_):
        visual_backend.holding_entity = "hot_pot"
        raise RuntimeError("log fault")

    with pytest.raises(RuntimeError, match="log fault"):
        run_observe(visual_backend, "hot_pot", ObserveConfig(), .3, .03,
                    visual_backend.capture_visual_frame, lambda *args: 100,
                    settle=lambda _: None, log=fail_log)
    assert visual_backend.holding_entity is None
    assert visual_backend.robot_pose()[0] == before[0]


def test_observe_cannot_disable_restoration():
    from rummagebench.skills.observe import ObserveConfig
    with pytest.raises(ValueError, match="must restore"):
        ObserveConfig(restore_robot_pose=False)


def test_engine_failure_invalidates_run_without_mis_scoring(visual_session, visual_backend):
    def broken_settle(_):
        raise RuntimeError("private simulator details")

    visual_backend.settle = broken_settle
    with pytest.raises(RuntimeError, match="reset required") as error:
        visual_session.step({"skill": "MOVE", "distance_cm": 10})
    assert "private simulator details" not in str(error.value)
    assert visual_session._public_step == 0
    assert visual_session._trace[-1]["run_invalidated"]
    with pytest.raises(RuntimeError, match="reset required"):
        visual_session.step({"skill": "REPORT_DONE"})


def test_pose_failure_before_dispatch_invalidates_run(visual_session, visual_backend):
    def broken_pose():
        raise RuntimeError("private pose fault")

    visual_backend.robot_pose = broken_pose
    with pytest.raises(RuntimeError, match="reset required"):
        visual_session.step({"skill": "MOVE", "distance_cm": 10})
    with pytest.raises(RuntimeError, match="reset required"):
        visual_session.observe()
