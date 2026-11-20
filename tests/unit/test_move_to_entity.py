"""Object-directed MOVE_TO (2026-10-26): stand in front of a clicked entity, facing it."""
import math
from types import SimpleNamespace

from rummagebench.skills.move import execute_move_to_entity, quat_from_yaw, select_stand_pose, yaw_from_quat

TABLE = ([1.0, -0.5, 0.0], [2.0, 0.5, 0.8])
FLOOR = ([-4, -4, -.1], [5, 4, 0])


def _backend(obstacles=(("table", TABLE),), start=(0.0, 0.0, 0.0), categories=None):
    pose = [[start[0], start[1], .005], quat_from_yaw(start[2])]
    calls = []
    boxes = dict(obstacles)
    infos = {"floor": SimpleNamespace(fixed_base=True, category="floors"),
             **{n: SimpleNamespace(fixed_base=True, category=(categories or {}).get(n, "table")) for n in boxes}}

    def teleport(anchor):
        calls.append(anchor)
        pose[0], pose[1] = list(anchor.position), list(anchor.orientation)

    backend = SimpleNamespace(
        robot_pose=lambda: (pose[0], pose[1]), entity_names=lambda: ["floor", *boxes],
        describe_entity=lambda n: infos.get(n),
        entity_aabb=lambda n: FLOOR if n == "floor" else boxes.get(n),
        robot_entity_names=lambda: set(), is_holding=lambda n: False,
        navigation_obstacle_aabbs=lambda n: [] if n == "floor" else [boxes[n]],
        teleport_robot=teleport)
    return backend, calls


def test_stand_pose_faces_the_clicked_side():
    backend, _ = _backend()
    near, reason, info = select_stand_pose(backend, "table", [1.0, 0.0, .8], .4)
    assert reason is None and near[0] < 1.0 and abs(near[1]) < .25
    assert abs(yaw_from_quat(quat_from_yaw(near[2]))) < .3          # facing +x, towards the table
    far, _, _ = select_stand_pose(backend, "table", [2.0, 0.0, .8], .4)
    assert far[0] > 2.0 and abs(abs(far[2]) - math.pi) < .3         # other side, facing -x
    assert info["planner"] == "stand_pose_v1"


def test_stand_pose_unreachable_and_move_commits_once():
    # robot boxed in by walls: nothing reachable, nothing moves
    walls = (("table", TABLE), ("w1", ([-.9, -1, 0], [-.7, 1, 2])), ("w2", ([.7, -1, 0], [.9, 1, 2])),
             ("w3", ([-.9, .7, 0], [.9, .9, 2])), ("w4", ([-.9, -.9, 0], [.9, -.7, 2])))
    backend, calls = _backend(walls)
    ok, reason, _ = execute_move_to_entity(backend, "table", [1.0, 0.0, .8], .4)
    assert not ok and reason == "NAV_NO_STAND_POSE" and not calls
    backend, calls = _backend()
    ok, reason, info = execute_move_to_entity(backend, "table", [1.0, 0.0, .8], .4)
    assert ok and reason is None and len(calls) == 1
    assert backend.robot_pose()[0][2] == .005 and info["execution"] == "oracle_endpoint"


def test_no_stand_pose_through_the_wall_a_shelf_leans_on():
    # shelf against the far side of a wall at x=0.9..1.0; robot on the near side
    shelf = ([1.0, -0.4, 0.0], [1.4, 0.4, 0.9])
    wall = ([0.9, -4.0, 0.0], [1.0, 4.0, 2.5])   # spans the whole floor: no way around
    backend, _ = _backend((("shelf", shelf), ("wall", wall)), categories={"wall": "walls"})
    pose, reason, _ = select_stand_pose(backend, "shelf", [1.0, 0.0, .9], .4)
    assert pose is None and reason == "NAV_NO_STAND_POSE"
