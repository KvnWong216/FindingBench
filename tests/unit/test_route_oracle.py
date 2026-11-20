"""Pure-logic tests for the privileged scripted driver (no simulator)."""
import math
from types import SimpleNamespace as NS

from rummagebench.evaluation.route_oracle import _ring_poses, derive_subgoals


def test_oracle_recognizes_carpet_pixels_and_uses_public_support_rule():
    import numpy as np
    from rummagebench.evaluation.route_oracle import _floor_pixels, _same_level_floors
    boxes = {'carpet': ([-2,-2,.23],[2,2,.234]),
             'floor': ([-2,-2,-.1],[2,2,0]),
             'hanging': ([0,0,0],[.004,1,2])}
    backend = NS(entity_names=lambda: list(boxes),
                 describe_entity=lambda n: NS(category='floors' if n=='floor' else 'carpet', fixed_base=True),
                 entity_aabb=lambda n: boxes[n], robot_pose=lambda: ([0,0,.005],[0,0,0,1]),
                 is_holding=lambda _: False)
    frame = NS(instance_segmentation=np.array([[1,2,3]]), depth=np.ones((1,3)),
               meta={'instance_labels':{'1':'carpet','2':'floor','3':'hanging'},
                     'pixel_rays':np.array([[[0,0,1],[0,0,1],[0,0,1]]])},
               camera_extrinsics=np.eye(4), depth_convention='euclidean_range')
    pixels, _ = _floor_pixels(frame, backend)
    assert pixels.tolist() == [[0,0],[1,0]]
    assert _same_level_floors(backend, .005) == [boxes['carpet'], boxes['floor']]


class FakeBackend:
    def __init__(self, open_=(), articulated=()):
        self._open = set(open_)
        self._art = set(articulated)

    def is_open(self, e):
        return e in self._open

    def articulation_info(self, e):
        return {"joints": 1} if e in self._art else None


def _scenario(placement, goal):
    return NS(target=NS(entity="t"), placements=[NS(entity="t", **placement)],
              goal=[NS(entity="t", **g) for g in goal])


def test_fetch_from_open_surface_is_grasp_only():
    s = _scenario({"relation": "on_top", "receptacle": "counter"}, [{"kind": "holding", "receptacle": None}])
    assert derive_subgoals(s, FakeBackend()) == [("GRASP", "t")]


def test_closed_container_is_opened_first():
    s = _scenario({"relation": "inside", "receptacle": "drawer"}, [{"kind": "holding", "receptacle": None}])
    assert derive_subgoals(s, FakeBackend()) == [("OPEN", "drawer"), ("GRASP", "t")]
    assert derive_subgoals(s, FakeBackend(open_={"drawer"})) == [("GRASP", "t")]


def test_place_goals():
    s = _scenario({"relation": "on_top", "receptacle": "counter"},
                  [{"kind": "on_top", "receptacle": "table"}])
    assert derive_subgoals(s, FakeBackend()) == [("GRASP", "t"), ("PLACE", "table")]
    s = _scenario({"relation": "on_top", "receptacle": "counter"},
                  [{"kind": "inside", "receptacle": "fridge"}])
    assert derive_subgoals(s, FakeBackend(articulated={"fridge"})) == [
        ("OPEN", "fridge"), ("GRASP", "t"), ("PLACE", "fridge")]


def test_goal_container_opened_before_source_container_and_grasp():
    """The hand must be free to open the goal drawer (held body collides)."""
    s = _scenario({"relation": "inside", "receptacle": "src"},
                  [{"kind": "inside", "receptacle": "dst"}])
    assert derive_subgoals(s, FakeBackend(articulated={"src", "dst"})) == [
        ("OPEN", "dst"), ("OPEN", "src"), ("GRASP", "t"), ("PLACE", "dst")]
    # already open goal container: nothing to prepare
    assert derive_subgoals(s, FakeBackend(open_={"dst"}, articulated={"src", "dst"})) == [
        ("OPEN", "src"), ("GRASP", "t"), ("PLACE", "dst")]


def test_ring_poses_face_the_box_from_outside():
    aabb = ([0.0, 0.0, 0.0], [1.0, 2.0, 1.0])
    poses = _ring_poses(aabb, offsets=(0.5,))
    assert poses
    for x, y, yaw, d in poses:
        inside = 0.0 <= x <= 1.0 and 0.0 <= y <= 2.0
        assert not inside
        # facing direction points toward the box
        nx, ny = min(max(x, 0.0), 1.0), min(max(y, 0.0), 2.0)
        assert abs(math.atan2(ny - y, nx - x) - yaw) < 1e-9


def test_anchor_hints_are_tried_before_the_ring(monkeypatch):
    """A '<entity>__<SKILL>*' anchor is the first counterfactual check."""
    import rummagebench.evaluation.route_oracle as ro
    from rummagebench.core.scenario import AnchorSpec

    checked = []

    class Backend:
        def entity_aabb(self, e):
            return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        def robot_pose(self):
            return ([5.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        def teleport_robot(self, anchor):
            checked.append(tuple(round(v, 3) for v in anchor.position[:2]))

    class Chk:
        lo = hi = []

        def free_many(self, xs, ys, yaws):
            import numpy as np
            return np.ones(len(xs), bool)

        def free(self, x, y, yaw):
            return True

    monkeypatch.setattr("rummagebench.authoring.tasks.navigation.FootprintChecker.from_backend",
                        classmethod(lambda cls, *a, **k: Chk()))
    monkeypatch.setattr("rummagebench.authoring.tasks.navigation.plan_route",
                        lambda chk, start, goal, wps, lim: NS(actions=[{"skill": "MOVE", "distance_cm": 10}], steps=1))
    scenario = NS(anchors={"table__PLACE_0": AnchorSpec(position=[2.0, 0.5, 0.0], orientation=[0, 0, 0, 1])},
                  agent_protocol=NS(min_move_cm=5, max_move_cm=100, min_turn_deg=5, max_turn_deg=180))
    agent = ro.RouteOracleAgent.__new__(ro.RouteOracleAgent)
    agent._backend, agent._scenario, agent._room_map, agent.log = Backend(), scenario, None, []
    agent.route_debug = None
    agent._verdict = lambda skill, entity: NS(feasible=True, reason="none")
    route = agent._route_to_interaction_pose("PLACE", "table")
    assert route and checked[0] == (2.0, 0.5)
