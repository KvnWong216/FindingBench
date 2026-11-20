"""Reject decorations that quietly turn a search task into another layout."""
import importlib.util
from pathlib import Path

import pytest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('background', Path(__file__).parents[2]/'scripts/enrich_background.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SUPPORT = [[0., 0., .85], [2., 1., .90]]
PROP = dict(size=[.1, .1, .1], position_xy=[.5, .5])


def test_robot_without_room_membership_is_still_an_obstacle():
    robot = SimpleNamespace(name='robot', category='robot', n_joints=7)
    table = SimpleNamespace(name='table', category='desk', model='native', n_joints=0, in_rooms=['office_0'])
    box = [[0., 0., 0.], [1., 1., 1.]]
    backend = SimpleNamespace(_env=SimpleNamespace(scene=SimpleNamespace(objects=[robot, table])), entity_aabb=lambda _: box)
    rows = m.scene_records(backend)
    assert rows['robot']['rooms'] == [] and rows['robot']['aabb'] == box
    assert rows['table']['rooms'] == ['office_0']


def test_sparse_supported_props_pass_without_occupying_task_space():
    result = m.plan_props(SUPPORT, [PROP], [[[3., 0., 0.], [4., 1., 1.]]], [])
    assert result['footprint_fraction'] < .08


@pytest.mark.parametrize('xy', [[.02, .5], [2.01, .5]])
def test_surface_edges_and_overhangs_rejected(xy):
    with pytest.raises(ValueError, match='SUPPORT_EDGE'):
        m.plan_props(SUPPORT, [dict(PROP, position_xy=xy)], [], [])


def test_rejects_props_in_manipulation_space_even_without_direct_collision():
    with pytest.raises(ValueError, match='TASK_WORKSPACE'):
        m.plan_props(SUPPORT, [PROP], [[[.7, .4, 0.], [1., .8, 1.]]], [])


def test_rejects_actual_occupied_counter_and_low_overhead_cabinet():
    for box in ([[.45, .45, .9], [.6, .6, 1.1]], [[0., 0., 1.], [2., 1., 1.5]]):
        with pytest.raises(ValueError, match='NEIGHBOUR_COLLISION'):
            m.plan_props(SUPPORT, [PROP], [], [box])


def test_props_cannot_overlap_each_other():
    with pytest.raises(ValueError, match='NEIGHBOUR_COLLISION'):
        m.plan_props(SUPPORT, [PROP, PROP], [], [])


def test_crowding_and_size_limits():
    with pytest.raises(ValueError, match='SPARSE_OBJECT'):
        m.plan_props(SUPPORT, [PROP]*3, [], [])
    with pytest.raises(ValueError, match='SPARSE_COVERAGE'):
        m.plan_props([[0, 0, .85], [.3, .3, .9]], [PROP], [], [])
    with pytest.raises(ValueError, match='ASSET_SIZE'):
        m.plan_props(SUPPORT, [dict(PROP, size=[.1, float('nan'), .1])], [], [])


def test_original_articulation_and_world_drift_cannot_be_hidden():
    before = dict(cabinet=dict(links=dict(root=([0, 0, 0], [0, 0, 0, 1])), joints=[0.]))
    opened = dict(cabinet=dict(links=dict(root=([0, 0, 0], [0, 0, 0, 1])), joints=[.1]))
    moved = dict(cabinet=dict(links=dict(root=([.01, 0, 0], [0, 0, 0, 1])), joints=[0.]))
    for after in (opened, moved):
        with pytest.raises(ValueError, match='ORIGINAL_WORLD_MOVED'):
            m.compare_poses(before, after)
    assert m.compare_poses(before, before)['max_displacement_m'] == 0
