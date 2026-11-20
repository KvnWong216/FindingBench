"""The grasp offset must be measured on the arm the feasibility engine models.

R1Pro lists arm_names = ["left", "right"] while the benchmark kinematics
use right_eef_link. eef_pose() used arm_names[0] and symbolic_grasp attached
the object to the nearest hand, so held_offset could describe the left hand
while PLACE/OPEN checks moved the right one (2026-10-19: PLACE/OPEN became
infeasible depending on which side the object was grasped from).
"""
import types

import numpy as np

from rummagebench.sim.omnigibson import backend as module


class _Link:
    def __init__(self, pos):
        self._pos = np.asarray(pos, float)

    def get_position_orientation(self, frame=None):
        return self._pos, np.array([0.0, 0.0, 0.0, 1.0])


def _backend(eef_link):
    b = object.__new__(module.OmniGibsonBackend)
    b._robot = types.SimpleNamespace(
        arm_names=["left", "right"],
        eef_link_names={"left": "left_eef_link", "right": "right_eef_link"},
        eef_links={"left": _Link([0.0, 0.3, 1.0]), "right": _Link([0.0, -0.3, 1.0])},
    )
    kin = None if eef_link is None else types.SimpleNamespace(end_effector_link=eef_link)
    b._scenario = types.SimpleNamespace(robot=types.SimpleNamespace(kinematics=kin))
    return b


def test_modelled_arm_follows_kinematics_eef_link():
    assert _backend("right_eef_link")._modelled_arm() == "right"
    assert _backend("left_eef_link")._modelled_arm() == "left"
    assert _backend(None)._modelled_arm() is None
    assert _backend("no_such_link")._modelled_arm() is None


def test_eef_pose_reports_the_modelled_arm():
    pose = _backend("right_eef_link").eef_pose()
    assert np.allclose(pose.position, [0.0, -0.3, 1.0])
    # without kinematics the legacy default (first arm) is kept
    pose = _backend(None).eef_pose()
    assert np.allclose(pose.position, [0.0, 0.3, 1.0])


def test_symbolic_grasp_uses_modelled_arm_even_if_other_hand_is_closer(monkeypatch):
    b = _backend("right_eef_link")
    obj = types.SimpleNamespace(
        links={"base_link": None},
        get_position_orientation=lambda: (np.array([0.0, 0.35, 1.0]), None),  # next to LEFT hand
        set_position_orientation=lambda position: None,
        set_linear_velocity=lambda v: None,
        set_angular_velocity=lambda v: None,
    )
    used = []
    b._robot._ag_obj_in_hand = {"left": None, "right": None}
    b._robot._get_assisted_grasp_joint_type = lambda o, l: "FixedJoint"

    def establish(target_obj, target_link_name, arm, contact_pos_world, joint_type):
        used.append(arm)
        b._robot._ag_obj_in_hand[arm] = target_obj
    b._robot._establish_grasp = establish
    b.settle = lambda n=None: None
    monkeypatch.setattr(module, "resolve_object", lambda scene, name: obj)
    b._env = types.SimpleNamespace(scene=None)

    assert b.symbolic_grasp("apple")
    assert used == ["right"]


def test_place_half_extents_use_native_bbox_and_held_name(monkeypatch):
    b = _backend("right_eef_link")
    fork = types.SimpleNamespace(name="target_fork", native_bbox=np.array([0.2, 0.04, 0.03]),
                                 scale=np.array([1.0, 1.0, 2.0]))
    monkeypatch.setattr(module, "resolve_object", lambda scene, name: fork)
    b._env = types.SimpleNamespace(scene=None)
    assert np.allclose(b.place_half_extents("target_fork"), [0.1, 0.02, 0.03])
    b._robot._ag_obj_in_hand = {"left": None, "right": fork}
    assert b.held_entity_name() == "target_fork"
    b._robot._ag_obj_in_hand = {"left": None, "right": None}
    assert b.held_entity_name() is None


def test_held_object_robot_collisions_filtered_while_held():
    b = _backend("right_eef_link")

    class Link:
        def __init__(self):
            self.filtered = set()

        def add_filtered_collision_pair(self, other):
            self.filtered.add(id(other))

        def remove_filtered_collision_pair(self, other):
            self.filtered.discard(id(other))

    olinks = {"base_link": Link()}
    b._robot.links = {"torso": Link(), "right_arm_link5": Link()}
    obj = types.SimpleNamespace(name="fork", links=olinks)
    assert b._filter_held_vs_robot(obj, True) == 2
    assert len(olinks["base_link"].filtered) == 2
    assert b._filter_held_vs_robot(obj, True) == 0      # idempotent
    assert b._filter_held_vs_robot(obj, False) == 2
    assert not olinks["base_link"].filtered


def test_long_objects_are_held_ahead_of_the_fingertips(monkeypatch):
    """Fork-sized objects go ahead of the EEF along tool +z; a mug stays on it."""
    b = _backend("right_eef_link")
    b._env = types.SimpleNamespace(scene=None)
    set_to = {}

    def make(name, bbox):
        return types.SimpleNamespace(
            name=name, native_bbox=np.array(bbox), scale=np.array([1.0, 1.0, 1.0]),
            set_position_orientation=lambda position: set_to.__setitem__(name, np.asarray(position, float)),
            set_linear_velocity=lambda v: None, set_angular_velocity=lambda v: None)

    objs = {"fork": make("fork", [0.20, 0.03, 0.02]), "mug": make("mug", [0.10, 0.10, 0.10])}
    monkeypatch.setattr(module, "resolve_object", lambda scene, name: objs[name])
    # right EEF at (0, -0.3, 1.0), identity orientation -> tool +z = world +z
    b._bring_to_hand(objs["fork"], "right")
    assert np.allclose(set_to["fork"], [0.0, -0.3, 1.0 + 0.10 + 0.02])
    b._bring_to_hand(objs["mug"], "right")
    assert np.allclose(set_to["mug"], [0.0, -0.3, 1.0])
