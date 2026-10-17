"""Agent-protocol navigation witness (authoring/tasks/navigation.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_collapsed_exported_wall_triangles_keep_conservative_obstacles():
    from rummagebench.sim.omnigibson.usd_collision import mesh_navigation_aabbs
    points=[[0,0,.2],[1,0,.3],[2,0,.4]]
    boxes=mesh_navigation_aabbs(points,[3],[0,1,2],np.eye(4),.1)
    assert len(boxes)==1 and np.allclose(boxes[0],[[0,0,.1],[2,0,.3]])
    assert mesh_navigation_aabbs([[0,0,2]]*3,[3],[0,1,2],np.eye(4),.1)==[]


def test_triangle_navigation_envelope_uses_clipped_edges_not_whole_face():
    from rummagebench.sim.omnigibson.usd_collision import mesh_navigation_aabbs
    boxes=mesh_navigation_aabbs([[0,0,0],[2,0,1],[0,2,1]],[3],[0,1,2],np.eye(4),0.)
    assert len(boxes)==1
    assert np.allclose(boxes[0],[[0,0,.05],[1,1,.5]])

from rummagebench.authoring.tasks.navigation import (
    FootprintChecker, ProtocolLimits, plan_route, route_through)
from rummagebench.skills.move import _aabb_overlaps_footprint, footprint_corners


def test_merged_wall_mesh_keeps_interior_free_and_wall_crossings_blocked():
    """Synthetic wall faces only; real mesh clipping, no simulator fixture."""
    from types import SimpleNamespace
    from rummagebench.sim.omnigibson.usd_collision import mesh_navigation_aabbs
    from rummagebench.skills.move import base_pose_collision_free
    points=[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],[-2,-2,3],[2,-2,3],[2,2,3],[-2,2,3]]
    faces=[0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
    boxes=mesh_navigation_aabbs(points,[4]*4,faces,np.eye(4),.1)
    backend=SimpleNamespace(entity_names=lambda:['unit_wall'],robot_entity_names=lambda:set(),
        describe_entity=lambda n:object(),is_holding=lambda n:False,
        entity_aabb=lambda n:([-2,-2,0],[2,2,3]),navigation_obstacle_aabbs=lambda n:boxes)
    checker=FootprintChecker.from_backend(backend,.4,.03)
    assert checker.free(0,0,0) and base_pose_collision_free(backend,0,0,0,.4,.03)
    for x,y in [(0,-1.8),(1.8,0),(0,1.8),(-1.8,0)]:
        assert not checker.free(x,y,0)
        assert not base_pose_collision_free(backend,x,y,0,.4,.03)
    assert not checker.move_free(0,0,0,3.)
    assert all(lo[2]>=.05-1e-12 and hi[2]<=.5+1e-12 for lo,hi in boxes)


@pytest.mark.parametrize('bad',['nonplanar','concave','indices','perspective'])
def test_uncertain_architecture_faces_cannot_be_silently_removed(bad):
    from rummagebench.sim.omnigibson.usd_collision import mesh_navigation_aabbs
    points=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]];counts=[4];indices=[0,1,2,3];matrix=np.eye(4)
    if bad=='nonplanar':points[-1][2]=.2
    if bad=='concave':points[2]=[.1,.1,0]
    if bad=='indices':indices[-1]=10
    if bad=='perspective':matrix[0,3]=.2
    with pytest.raises(ValueError):mesh_navigation_aabbs(points,counts,indices,matrix,0.)


def _random_boxes(rng, n):
    out = []
    for _ in range(n):
        c = rng.uniform(-3, 3, 2)
        e = rng.uniform(0.05, 0.8, 2)
        z0 = rng.choice([0.0, 0.3, 0.6])
        out.append(([c[0] - e[0], c[1] - e[1], z0], [c[0] + e[0], c[1] + e[1], z0 + 0.4]))
    return out


def test_vectorised_footprint_matches_move_skill():
    rng = np.random.default_rng(0)
    boxes = _random_boxes(rng, 40)
    chk = FootprintChecker(boxes, 0.40, 0.03)
    for _ in range(400):
        x, y = rng.uniform(-4, 4, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        corners = footprint_corners(x, y, yaw, 0.40)
        ref = not any(_aabb_overlaps_footprint(b, corners, 0.03) for b in boxes)
        assert chk.free(x, y, yaw) == ref


def test_route_counts_turns_and_moves():
    chk = FootprintChecker([], 0.4, 0.03)
    r = route_through(chk, (0.0, 0.0, 0.0), [(2.5, 0.0)], 0.0)
    assert [a["skill"] for a in r.actions] == ["MOVE"] * 3
    assert sum(a["distance_cm"] for a in r.actions) == pytest.approx(250.0, abs=0.5)
    # behind the robot: drive backward instead of turning twice
    r = route_through(chk, (0.0, 0.0, 0.0), [(-1.5, 0.0)], 0.0)
    assert [a["skill"] for a in r.actions] == ["MOVE", "MOVE"]
    assert all(a["distance_cm"] < 0 for a in r.actions)
    # sideways goal facing +y: turn, move
    r = route_through(chk, (0.0, 0.0, 0.0), [(0.0, 0.8)], math.pi / 2)
    assert [a["skill"] for a in r.actions] == ["TURN", "MOVE"]
    assert r.final[:2] == pytest.approx((0.0, 0.8), abs=1e-6)


def test_every_action_respects_protocol_bounds():
    rng = np.random.default_rng(1)
    chk = FootprintChecker([], 0.4, 0.03)
    lim = ProtocolLimits()
    for _ in range(50):
        g = rng.uniform(-5, 5, 2)
        r = route_through(chk, (0.0, 0.0, rng.uniform(-3, 3)), [tuple(g)], rng.uniform(-3, 3))
        for a in r.actions:
            if a["skill"] == "MOVE":
                assert lim.min_move_m * 100 <= abs(a["distance_cm"]) <= lim.max_move_m * 100
            else:
                assert lim.min_turn_deg <= abs(a["angle_deg"]) <= lim.max_turn_deg


def replay(chk, start, actions):
    """Execute public actions with the same swept checks as the session."""
    x, y, h = start
    for a in actions:
        if a["skill"] == "TURN":
            assert 5.0 <= abs(a["angle_deg"]) <= 180.0
            assert chk.turn_free(x, y, h, a["angle_deg"]), a
            h += math.radians(a["angle_deg"])
        else:
            d = a["distance_cm"] / 100.0
            assert 0.05 <= abs(d) <= 1.0
            assert chk.move_free(x, y, h, d), a
            x, y = x + d * math.cos(h), y + d * math.sin(h)
    return x, y, h


def test_detour_around_a_wall():
    wall = ([0.9, -1.0, 0.0], [1.1, 1.0, 1.0])  # blocks the straight line
    chk = FootprintChecker([wall], 0.4, 0.03)
    start, goal = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    assert route_through(chk, start, [goal[:2]], 0.0) is None
    wps = [(x * 0.25, y * 0.25) for x in range(-4, 13) for y in range(-12, 13)]
    r = plan_route(chk, start, goal, wps)
    assert r is not None and r.waypoints
    x, y, h = replay(chk, start, r.actions)
    assert math.hypot(x - 2.0, y) < 5e-3  # distance_cm is rounded to 0.1 cm
    assert math.cos(h - goal[2]) > math.cos(math.radians(0.6))


def test_tight_anchor_is_entered_straight():
    """An anchor free only at its own heading (a 0.9 m wide alcove) is
    reached by driving straight in from a pivot on its approach line."""
    walls = [([1.0, 0.45, 0.0], [3.0, 0.6, 1.0]), ([1.0, -0.6, 0.0], [3.0, -0.45, 1.0]),
             ([3.0, -0.6, 0.0], [3.2, 0.6, 1.0])]
    chk = FootprintChecker(walls, 0.4, 0.03)
    goal = (2.5, 0.0, 0.0)  # inside the alcove, facing its back wall
    assert chk.free(*goal) and not chk.disc_free(goal[0], goal[1])
    start = (-1.0, 1.5, -math.pi / 2)
    wps = [(x * 0.25, y * 0.25) for x in range(-8, 16) for y in range(-10, 11)]
    r = plan_route(chk, start, goal, wps)
    assert r is not None
    x, y, h = replay(chk, start, r.actions)
    assert math.hypot(x - goal[0], y - goal[1]) < 5e-3
    assert r.actions[-1]["skill"] == "MOVE"  # no turn inside the alcove



def test_disc_shortcut_and_batch_agree_with_exact_checks():
    rng = np.random.default_rng(2)
    boxes = _random_boxes(rng, 30)
    chk = FootprintChecker(boxes, 0.40, 0.03)
    for _ in range(200):
        x, y = rng.uniform(-4, 4, 2)
        yaws = rng.uniform(-math.pi, math.pi, 12)
        batch = chk.free_many(np.full(12, x), np.full(12, y), yaws)
        for yaw, b in zip(yaws, batch):
            corners = footprint_corners(x, y, yaw, 0.40)
            ref = not any(_aabb_overlaps_footprint(bb, corners, 0.03) for bb in boxes)
            assert b == ref
        if chk.disc_free(x, y):  # sufficient: every heading must be free
            assert batch.all()
