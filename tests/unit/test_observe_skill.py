"""§39 OBSERVE: deterministic views, RGB-only, pose restore, gating."""

import numpy as np

SPAWN = ([3.65, -8.25, 0.0], [0.0, 0.0, 0.0, 1.0])  # living_room spawn


def _add_hot_pot(visual_session, visual_backend):
    """A clutter object in open space: every viewpoint is collision-free and
    the synthetic frame shows >=100 of its pixels."""
    visual_backend.frame_entities["hot_pot"] = (50, 4, 12, 12)
    visual_session._capture_current()
    return 56 / 63, 10 / 63  # normalized center of the hot_pot square


def test_observe_returns_rgb_only_views(visual_session, visual_backend):
    x, y = _add_hot_pot(visual_session, visual_backend)
    r = visual_session.step(
        {"skill": "OBSERVE", "point": {"frame_id": visual_session._store.current_id,
                                       "x": x, "y": y}}
    )
    assert r.feedback.code == "EXECUTED", r.feedback
    views = r.observation.observe_views
    assert views, "expected at least one accepted viewpoint"
    assert all(set(v.model_dump().keys()) == {"view_index", "image_png_b64"}
               for v in views)
    payload = r.observation.model_dump_json()
    for key in ("azimuth", "viewpoint", "camera_pose", "visible_pixel",
                "base_pose", "segmentation", "depth", "hot_pot"):
        assert key not in payload


def test_observe_view_indices_sequential(visual_session, visual_backend):
    x, y = _add_hot_pot(visual_session, visual_backend)
    r = visual_session.step(
        {"skill": "OBSERVE", "point": {"frame_id": visual_session._store.current_id,
                                       "x": x, "y": y}}
    )
    indices = [v.view_index for v in r.observation.observe_views]
    assert indices == list(range(len(indices)))


def test_observe_restores_exact_robot_pose(visual_session, visual_backend):
    x, y = _add_hot_pot(visual_session, visual_backend)
    before = tuple(visual_backend.robot_pose()[0])
    visual_session.step(
        {"skill": "OBSERVE", "point": {"frame_id": visual_session._store.current_id,
                                       "x": x, "y": y}}
    )
    after = tuple(visual_backend.robot_pose()[0])
    assert np.allclose(before, after, atol=1e-9)


def test_observe_with_no_valid_viewpoint_unsafe(visual_session, visual_backend):
    x, y = _add_hot_pot(visual_session, visual_backend)
    # no pixels visible for the entity from ANY viewpoint
    visual_backend.entity_visible_pixels = lambda frame, entity: 0
    r = visual_session.step(
        {"skill": "OBSERVE", "point": {"frame_id": visual_session._store.current_id,
                                       "x": x, "y": y}}
    )
    assert r.feedback.code == "UNSAFE"
    assert r.observation.observe_views == []
    assert r.episode_status == "RUNNING"


def test_observe_viewpoints_are_deterministic(visual_session, visual_backend):
    x, y = _add_hot_pot(visual_session, visual_backend)
    r1 = visual_session.step({"skill": "OBSERVE",
                              "point": {"frame_id": visual_session._store.current_id,
                                        "x": x, "y": y}})
    views1 = [(v.view_index, len(v.image_png_b64)) for v in r1.observation.observe_views]
    r2 = visual_session.step({"skill": "OBSERVE",
                              "point": {"frame_id": visual_session._store.current_id,
                                        "x": x, "y": y}})
    views2 = [(v.view_index, len(v.image_png_b64)) for v in r2.observation.observe_views]
    assert views1 == views2


def test_observe_consumes_one_step(visual_session, visual_backend):
    x, y = _add_hot_pot(visual_session, visual_backend)
    r = visual_session.step(
        {"skill": "OBSERVE", "point": {"frame_id": visual_session._store.current_id,
                                       "x": x, "y": y}}
    )
    assert r.planning_step == 1
    assert r.episode_status == "RUNNING"
