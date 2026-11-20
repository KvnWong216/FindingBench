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


def test_bev_astar_detours_and_preserves_exact_destination():
    from rummagebench.skills.move import plan_bev_path
    chk=FootprintChecker([([-.2,-.5,0],[.2,.5,.5])],.2,.03)
    floors=[([-3,-3,-.1],[3,3,0])]
    path,reason=plan_bev_path(chk,[-1,0],[1.037,0],0,floors)
    assert reason is None and path[-1]==[1.037,0]
    assert max(abs(p[1]) for p in path)>.7
    assert all(chk.free(*p,0) for p in path)
    assert plan_bev_path(chk,[-1,0],[0,0],0,floors)[1]=='NAV_GOAL_BLOCKED'
    assert plan_bev_path(chk,[-1,0],[4,0],0,floors)[1]=='NAV_GOAL_BLOCKED'


def test_bev_astar_disconnected_and_search_limit():
    from rummagebench.skills.move import plan_bev_path
    chk=FootprintChecker([([-.1,-2,0],[.1,2,.5])],.1,.03)
    floors=[([-2,-2,-.1],[2,2,0])]
    assert plan_bev_path(chk,[-1,0],[1,0],0,floors)[1]=='NAV_NO_PATH'
    assert plan_bev_path(chk,[-1,0],[1,0],0,floors,max_expansions=1)[1]=='NAV_SEARCH_LIMIT'
    assert plan_bev_path(chk,[-1,0],[1,0],0,[])[1]=='NAV_SUPPORT_UNAVAILABLE'


def test_move_to_point_requires_floor_and_rejects_stale_frame(visual_session):
    from rummagebench.core.public_types import PublicAction
    from pydantic import TypeAdapter
    point={'frame_id':visual_session.observe().frame_id,'x':.1,'y':.5}
    raw={'skill':'MOVE_TO','point':point}
    assert TypeAdapter(PublicAction).validate_python(raw).skill=='MOVE_TO'
    before=visual_session._backend.robot_pose()
    result=visual_session.step(raw)
    # 2026-10-26: a non-floor click is object-directed MOVE_TO (stand in front,
    # facing it); in this fixture it is either untargetable or has no stand pose
    assert result.feedback.code.value in ('INVALID_ACTION','OUT_OF_CAPABILITY')
    assert 'Move-to not executed' in result.feedback.message
    assert visual_session._backend.robot_pose()==before
    result=visual_session.step({'skill':'MOVE_TO','point':{**point,'frame_id':'stale'}})
    assert result.feedback.code.value=='INVALID_ACTION'


def test_move_to_oracle_commits_only_after_path_and_keeps_heading():
    from types import SimpleNamespace
    from rummagebench.skills.move import execute_move_to,quat_from_yaw
    pose=([0.,0.,.005],quat_from_yaw(.3)); calls=[]
    floor=SimpleNamespace(fixed_base=True,category='floors')
    def teleport(anchor):
        nonlocal pose
        calls.append(anchor);pose=(anchor.position,anchor.orientation)
    backend=SimpleNamespace(robot_pose=lambda:pose,entity_names=lambda:['floor'],
        describe_entity=lambda n:floor,entity_aabb=lambda n:([-3,-3,-.1],[3,3,0]),
        robot_entity_names=lambda:set(),is_holding=lambda n:False,
        navigation_obstacle_aabbs=lambda n:[],teleport_robot=teleport)
    success,reason,info=execute_move_to(backend,[1.03,.23,0],.4)
    assert success and reason is None and len(calls)==1
    assert pose[0]==[1.03,.23,.005] and pose[1]==quat_from_yaw(.3)
    assert info['distance_m']>1 and info['execution']=='oracle_endpoint'
    assert execute_move_to(backend,[4,0,0],.4)[1]=='NAV_GOAL_BLOCKED'
    assert execute_move_to(backend,[0,0,1],.4)[1]=='NAV_DIFFERENT_LEVEL'
    assert len(calls)==1


@pytest.mark.parametrize('height,allowed',[(0.237,True),(-0.237,True),(.299999,True),(.3,False),(-.3,False),(.301,False),(float('nan'),False)])
def test_floor_height_limit_strict(height,allowed):
    from rummagebench.skills.move import floor_height_allowed
    assert floor_height_allowed(height,0.) is allowed


def test_move_to_accepts_recorded_24cm_floor_difference():
    from types import SimpleNamespace
    from rummagebench.skills.move import execute_move_to,same_level_floors
    # Recorded failure: base z=.240234, clicked ground z=.0036.
    pose=([0.,0.,.2402343899011612],[0.,0.,0.,1.]);calls=[]
    def arrive(anchor):
        nonlocal pose
        calls.append(anchor);pose=(anchor.position,anchor.orientation)
    backend=SimpleNamespace(robot_pose=lambda:pose,entity_names=lambda:['floor'],
        describe_entity=lambda n:SimpleNamespace(fixed_base=True,category='floors'),
        entity_aabb=lambda n:([-3,-3,-.1],[3,3,.0036]),robot_entity_names=lambda:set(),
        is_holding=lambda n:False,navigation_obstacle_aabbs=lambda n:[],teleport_robot=arrive)
    assert same_level_floors(backend,pose[0][2])
    success,reason,_=execute_move_to(backend,[.3905,.628,.0036],.4)
    assert success and reason is None and len(calls)==1


def test_floor_laid_carpet_is_terrain_but_hanging_carpet_is_not():
    from types import SimpleNamespace
    from rummagebench.skills.move import is_navigation_floor, navigation_aabbs
    box = [[-2,-2,0], [2,2,.004]]
    held = False
    backend = SimpleNamespace(
        describe_entity=lambda _: SimpleNamespace(category='carpet', fixed_base=True),
        entity_aabb=lambda _: box, robot_pose=lambda: ([0,0,.005],[0,0,0,1]),
        is_holding=lambda _: held, navigation_obstacle_aabbs=lambda _: [box])
    assert is_navigation_floor(backend, 'carpet')
    assert navigation_aabbs(backend, 'carpet') == []
    box[:] = [[-2,-2,.8], [2,2,.804]]
    assert not is_navigation_floor(backend, 'carpet')
    assert navigation_aabbs(backend, 'carpet') == [box]
    box[:] = [[0,0,0], [.004,2,1]]
    assert not is_navigation_floor(backend, 'carpet')
    box[:] = [[-2,-2,0], [2,2,.004]]
    held = True
    assert not is_navigation_floor(backend, 'carpet')


def test_carpet_click_uses_exact_floor_destination_not_furniture_ring(visual_session, monkeypatch):
    from types import SimpleNamespace
    import rummagebench.skills.move as move
    vs = visual_session
    b = vs._backend
    target = np.array([.7,.2,.004])
    monkeypatch.setattr(vs, '_resolve_point', lambda *args: ('carpet', target))
    monkeypatch.setattr(b, 'describe_entity', lambda _: SimpleNamespace(category='carpet', fixed_base=True))
    monkeypatch.setattr(b, 'entity_aabb', lambda _: ([-2,-2,0],[2,2,.004]))
    calls = []
    def floor_move(backend, surface, half):
        calls.append(surface.copy())
        return True, None, {'planner':'bev_astar_fixed_heading_v1'}
    def furniture_move(*args, **kwargs):
        raise AssertionError('Floor carpet must not use furniture stand poses')
    monkeypatch.setattr(move, 'execute_move_to', floor_move)
    monkeypatch.setattr(move, 'execute_move_to_entity', furniture_move)
    r = vs.step({'skill':'MOVE_TO','point':{'frame_id':vs.observe().frame_id,'x':.5,'y':.5}})
    assert r.feedback.code == 'EXECUTED'
    np.testing.assert_allclose(calls, [target])


@pytest.mark.parametrize('category', ['carpet', 'cabinet', 'unknown_asset', 'paver'])
@pytest.mark.parametrize('height,passable', [(.04,True),(.29,True),(.3,False),(.89,False)])
def test_navigation_uses_height_not_furniture_label(category, height, passable):
    from types import SimpleNamespace
    from rummagebench.skills.move import is_navigation_floor, navigation_aabbs, base_pose_collision_free
    box=([-1,-1,0],[1,1,height])
    backend=SimpleNamespace(entity_names=lambda:['object'], robot_entity_names=lambda:set(),
        robot_pose=lambda:([0,0,0],[0,0,0,1]), is_holding=lambda _:False,
        describe_entity=lambda _:SimpleNamespace(category=category,fixed_base=True),
        entity_aabb=lambda _:box, navigation_obstacle_aabbs=lambda _:[box])
    assert is_navigation_floor(backend,'object') is passable
    assert FootprintChecker.from_backend(backend,.4,.03).free(0,0,0) is passable
    assert base_pose_collision_free(backend,0,0,0,.4,.03) is passable
