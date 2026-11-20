"""PLACE into drawers of fixed furniture (2026-10-18): open-drawer selection,
drawer-floor targets, and inside goals satisfied in a re-closed drawer."""
from types import SimpleNamespace as NS

from rummagebench.evaluation.goal_checker import check_goal
from rummagebench.feasibility.interaction_target import (
    INSIDE_FLOOR_OFFSET_M,
    InteractionRegion,
    receptacle_place_targets,
    select_open_interior,
)

BODY = ([0.0, 0.0, 0.0], [1.0, 0.6, 0.9])


def test_stack_initialization_centres_an_eccentric_native_root():
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    class EccentricBody:
        name='book'
        def __init__(self):
            self.position=np.zeros(3);self.quaternion=np.array([0.,0.,0.,1.])
        @property
        def aabb(self):
            rotation=Rotation.from_quat(self.quaternion)
            centre=self.position-rotation.apply([.08,-.03,0.])
            size=np.abs(rotation.as_matrix())@np.array([.4,.3,.02])
            return centre-size/2,centre+size/2
        def set_bbox_center_position_orientation(self,position,orientation):
            # The actual SDK transform utility requires tensor inputs.
            assert isinstance(position,torch.Tensor) and isinstance(orientation,torch.Tensor)
            self.quaternion=orientation.numpy()
            self.position=position.numpy()+Rotation.from_quat(self.quaternion).apply([.08,-.03,0.])
        def set_position_orientation(self,position,orientation):
            self.position=np.array(position);self.quaternion=np.array(orientation)
        def get_position_orientation(self):return self.position,self.quaternion
        def set_linear_velocity(self,value):pass
        def set_angular_velocity(self,value):pass
    body=EccentricBody();support=NS(name='support',aabb=(np.array([.75,1.8,.5]),np.array([1.25,2.2,.6])),
        get_position_orientation=lambda:(None,np.array([0.,0.,0.,1.])))
    backend=NS(settle=lambda steps:None,_is_stacked_on=OmniGibsonBackend._is_stacked_on)
    assert OmniGibsonBackend._stack_on(backend,body,support)
    lo,hi=body.aabb
    assert np.allclose(((lo+hi)/2)[:2],[1.,2.])
    assert abs(lo[2]-.603)<1e-6


def test_oriented_surface_authoring_rotates_root_offset_and_rests_above_support(monkeypatch):
    import numpy as np
    import sys,types
    from scipy.spatial.transform import Rotation
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    states=types.ModuleType('omnigibson.object_states');states.Inside=object();states.OnTop=object()
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',states)
    class Body:
        fixed_base=False;root_link=NS();scaled_bbox_center_in_base_frame=np.array([.1,.02,.03])
        def __init__(self):self.xyz=np.array([0.,0.,2.]);self.quat=np.array([0.,0.,0.,1.])
        def get_position_orientation(self):return self.xyz,self.quat
        def set_position_orientation(self,position=None,orientation=None):
            if position is not None:self.xyz=np.asarray(position)
            if orientation is not None:self.quat=np.asarray(orientation)
        @property
        def aabb(self):
            rotation=Rotation.from_quat(self.quat);centre=self.xyz-rotation.apply(self.scaled_bbox_center_in_base_frame)
            half=np.abs(rotation.as_matrix())@np.array([.4,.2,.1])/2
            return centre-half,centre+half
    body=Body();rec=NS(name='table',fixed_base=True,links={})
    synchronized=[]
    backend=NS(receptacle_region=lambda name:NS(kind='top_surface',lo=[0.,0.,0.],hi=[2.,2.,1.]),
        _sim=NS(sync_physx_to_fabric=lambda:synchronized.append(True)),settle=lambda steps:None,
        _verify_relation=lambda *args:True)
    quat=Rotation.from_euler('y',90,degrees=True).as_quat().tolist()
    assert OmniGibsonBackend._place_at_xy(backend,body,rec,[.5,.4],orientation_hint=quat)
    lo,hi=body.aabb
    assert np.allclose(((lo+hi)/2)[:2],[.5,.4]) and abs(lo[2]-1.01)<1e-8
    assert np.allclose(body.quat,quat) and synchronized==[True]


def _drawer(y_lo, z=(0.6, 0.8)):
    return InteractionRegion(lo=[0.05, y_lo, z[0]], hi=[0.95, y_lo + 0.55, z[1]], kind="inside_volume")


def test_select_open_interior_picks_the_pulled_out_drawer():
    closed = ("link_1", _drawer(0.02, (0.2, 0.4)))
    opened = ("link_2", _drawer(-0.35))
    assert select_open_interior(*BODY, [closed, opened])[0] == "link_2"
    assert select_open_interior(*BODY, [closed]) is None


def test_select_open_interior_prefers_the_top_drawer_when_all_are_open():
    """OPEN opens every drawer; lower ones are covered by the one above."""
    lower = ("link_0", _drawer(-0.45, (0.2, 0.35)))   # pulled out further
    upper = ("link_2", _drawer(-0.35, (0.67, 0.84)))
    assert select_open_interior(*BODY, [lower, upper])[0] == "link_2"


def test_inside_targets_sit_on_the_floor_and_approach_from_above():
    region = _drawer(-0.35)
    backend = NS(receptacle_region=lambda e: region,
                 robot_pose=lambda: ([0.5, -1.2, 0.0], [0, 0, 0, 1]))
    targets = receptacle_place_targets("cabinet", backend)
    assert targets
    for t in targets:
        p = t.metadata["surface_point"]
        assert abs(p[2] - (region.lo[2] + INSIDE_FLOOR_OFFSET_M)) < 1e-9
        assert region.lo[0] <= p[0] <= region.hi[0] and region.lo[1] <= p[1] <= region.hi[1]
        assert t.metadata["approach_normal"] == [0.0, 0.0, -1.0]


def test_inside_targets_keep_the_held_footprint_off_the_walls():
    """An 18 cm fork must not be centred 6 cm from a drawer wall."""
    from rummagebench.feasibility.interaction_target import INSIDE_WALL_CLEARANCE_M
    region = _drawer(-0.35)
    half = [0.10, 0.02, 0.015]
    backend = NS(receptacle_region=lambda e: region,
                 robot_pose=lambda: ([0.5, -1.2, 0.0], [0, 0, 0, 1]),
                 held_entity_name=lambda: "fork", place_half_extents=lambda e: half)
    targets = receptacle_place_targets("cabinet", backend)
    assert targets
    for t in targets:
        x, y, _ = t.metadata["surface_point"]
        assert region.lo[0] + half[0] + INSIDE_WALL_CLEARANCE_M - 1e-9 <= x
        assert x <= region.hi[0] - half[0] - INSIDE_WALL_CLEARANCE_M + 1e-9
        assert region.lo[1] + 0.06 - 1e-9 <= y <= region.hi[1] - 0.06 + 1e-9


def test_inside_targets_collapse_to_centre_when_object_barely_fits():
    region = _drawer(-0.35)
    backend = NS(receptacle_region=lambda e: region,
                 robot_pose=lambda: ([0.5, -1.2, 0.0], [0, 0, 0, 1]),
                 held_entity_name=lambda: "tray", place_half_extents=lambda e: [0.44, 0.1, 0.02])
    xs = {round(t.metadata["surface_point"][0], 6) for t in receptacle_place_targets("cabinet", backend)}
    assert xs == {0.5}


class ClosedDrawerBackend:
    """Fork inside a drawer that was closed again: the region is the body top."""

    def __init__(self, fork_box):
        self.fork_box = fork_box

    def entity_aabb(self, e):
        return self.fork_box

    def receptacle_region(self, e):
        return InteractionRegion(lo=BODY[0], hi=BODY[1], kind="top_surface")

    def receptacle_interior_regions(self, e):
        return [("link_2", _drawer(0.02))]

    def is_holding(self, e):
        return False


def _goal(backend, held=None):
    scenario = NS(goal=[NS(kind="inside", entity="fork", receptacle="cabinet")])
    return check_goal(NS(held_object=held), scenario, backend)


def test_inside_goal_holds_in_a_reclosed_drawer():
    inside = ([0.3, 0.2, 0.62], [0.5, 0.25, 0.64])
    assert _goal(ClosedDrawerBackend(inside)).satisfied


def test_inside_goal_fails_outside_the_drawer_or_when_held():
    on_counter = ([0.3, 0.2, 0.92], [0.5, 0.25, 0.94])
    assert not _goal(ClosedDrawerBackend(on_counter)).satisfied
    inside = ([0.3, 0.2, 0.62], [0.5, 0.25, 0.64])
    assert not _goal(ClosedDrawerBackend(inside), held="fork").satisfied


def test_open_strip_is_the_part_pulled_out_beyond_the_body():
    from rummagebench.sim.omnigibson.backend import DRAWER_OVERHANG_MARGIN_M, _with_open_strip
    region = _with_open_strip(_drawer(-0.35), BODY)   # y from -0.35 to 0.20, body y >= 0
    assert region.reach_lo == [0.05, -0.35]
    assert abs(region.reach_hi[1] - (0.0 - DRAWER_OVERHANG_MARGIN_M)) < 1e-9
    # not pulled out: no strip
    assert _with_open_strip(_drawer(0.02), BODY).reach_lo is None


def test_inside_targets_stay_in_the_open_strip():
    """Top-down only (no pull direction known): points under the countertop
    (inside the body footprint) are unreachable from above."""
    from rummagebench.sim.omnigibson.backend import _with_open_strip
    region = _with_open_strip(_drawer(-0.35), BODY)
    region.open_dir = None
    backend = NS(receptacle_region=lambda e: region,
                 robot_pose=lambda: ([0.5, -1.2, 0.0], [0, 0, 0, 1]),
                 held_entity_name=lambda: "fork", place_half_extents=lambda e: [0.10, 0.02, 0.015])
    targets = receptacle_place_targets("cabinet", backend)
    assert targets
    for t in targets:
        x, y, _ = t.metadata["surface_point"]
        assert region.reach_lo[1] - 1e-9 <= y <= region.reach_hi[1] + 1e-9
        assert region.lo[1] + 0.06 - 1e-9 <= y


def test_open_drawer_targets_are_angled_toward_the_body():
    """Drawer under a countertop: tilted approaches from the pulled-out side."""
    import numpy as np
    from rummagebench.feasibility.interaction_target import INSIDE_APPROACH_TILTS_DEG
    from rummagebench.sim.omnigibson.backend import _with_open_strip
    region = _with_open_strip(_drawer(-0.35), BODY)   # pulled out toward -y
    assert region.open_dir == [0.0, -1.0]
    backend = NS(receptacle_region=lambda e: region,
                 robot_pose=lambda: ([0.5, -1.2, 0.0], [0, 0, 0, 1]),
                 held_entity_name=lambda: "fork", place_half_extents=lambda e: [0.10, 0.02, 0.015])
    targets = receptacle_place_targets("cabinet", backend)
    tilts = [t.metadata["approach_tilt_deg"] for t in targets]
    assert tilts[0] == INSIDE_APPROACH_TILTS_DEG[0] and set(tilts) == set(INSIDE_APPROACH_TILTS_DEG)
    for t in targets:
        a = np.asarray(t.metadata["approach_normal"])
        assert a[2] < 0 and a[1] >= -1e-9          # moving down and toward the body (+y)
        tilt = np.degrees(np.arccos(-a[2]))
        assert abs(tilt - t.metadata["approach_tilt_deg"]) < 1e-6
        # standoff on the outside / above of the release point
        p = np.asarray(t.metadata["surface_point"])
        assert np.allclose(np.asarray(t.pose.position), p - a * t.metadata["standoff_m"])
    # strip points first, then points deeper under the body
    first_tilt = [t for t in targets if t.metadata["approach_tilt_deg"] == tilts[0]]
    ys = [t.metadata["surface_point"][1] for t in first_tilt]
    assert max(ys) > region.reach_hi[1]          # cavity beyond the strip is offered too
    assert ys[0] <= region.reach_hi[1] + 1e-9
