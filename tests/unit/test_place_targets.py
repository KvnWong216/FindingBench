"""PLACE grounding on BEHAVIOR support furniture (A.12 Core fix).

- fixed countertops / tables / desks / base cabinets are support surfaces
  (BEHAVIOR abilities only mark fillable / `inside` containers);
- top-surface PLACE candidates are grid points within reach of the robot
  base, not the whole-surface center / edge midpoints;
- articulated receptacles (cabinets) offer the same top-surface targets;
- the realization receives the validated support point.
"""

from types import SimpleNamespace

import numpy as np

from rummagebench.core.types import TargetKind
from rummagebench.feasibility.interaction_target import (
    PLACE_BATCH_SIZE,
    PLACE_PREFERRED_REACH_M,
    STANDOFF_M,
    receptacle_place_targets,
)
from rummagebench.object_interface.articulated import ArticulatedObject
from rummagebench.sim.base import ResolvedTarget
from rummagebench.sim.omnigibson.entity_resolver import build_entity_info
from rummagebench.skills.place import PlaceSkill


def _obj(category, fixed_base, abilities=None):
    return SimpleNamespace(name=f"{category}_1", category=category, model="m",
                           fixed_base=fixed_base, abilities=abilities or {},
                           in_rooms=["kitchen_0"])


def test_support_furniture_is_receptacle():
    assert build_entity_info(_obj("countertop", True)).is_receptacle
    assert build_entity_info(_obj("bottom_cabinet", True, {"openable": {}})).is_receptacle
    assert build_entity_info(_obj("desk", True)).is_receptacle
    # round-5: support-surface categories are support surfaces regardless of
    # the (unreliable) fixed_base metadata; such furniture is never graspable
    movable_table = build_entity_info(_obj("coffee_table", False))
    assert movable_table.is_receptacle and not movable_table.graspable
    # small objects are not support surfaces by category
    assert not build_entity_info(_obj("mug", False)).is_receptacle
    assert build_entity_info(_obj("bowl", False, {"fillable": {}})).is_receptacle


def test_support_surface_room_furniture_is_receptacle_not_graspable():
    """Round-5 GPU smoke finding: console_table is fixed_base=False in the
    BEHAVIOR metadata but IS a support surface by category — receptacle,
    NOT graspable, dispatched to the Receptacle adapter with the PLACE
    interface."""
    from rummagebench.object_interface.base import build_object_interface
    from rummagebench.object_interface.receptacle import Receptacle

    info = build_entity_info(_obj("console_table", False))
    assert info.is_receptacle
    assert not info.graspable
    assert isinstance(build_object_interface(info, None), Receptacle)


def test_graspable_receptacle_container_keeps_rigid_adapter():
    """Graspable fillable containers (bowls) keep the RigidObject GRASP
    interface; the Receptacle dispatch is for non-graspable surfaces."""
    from rummagebench.object_interface.base import build_object_interface
    from rummagebench.object_interface.rigid import RigidObject

    info = build_entity_info(_obj("bowl", False, {"fillable": {}}))
    assert info.is_receptacle and info.graspable
    assert isinstance(build_object_interface(info, None), RigidObject)


def test_top_surface_targets_near_robot(fake_backend):
    fake_backend._last_anchor_position = [2.0, 1.0, 0.0]
    targets = receptacle_place_targets("countertop", fake_backend)
    assert len(targets) > 0  # full ordered grid, no cap
    preferences = []
    for t in targets:
        sp = t.metadata["surface_point"]
        # inside the surface (edge inset respected); NO metric-circle cap
        assert 2.4 + 0.06 - 1e-9 <= sp[0] <= 3.6 - 0.06 + 1e-9
        assert 0.4 + 0.06 - 1e-9 <= sp[1] <= 1.6 - 0.06 + 1e-9
        assert sp[2] == 0.7  # countertop top
        assert abs(t.pose.position[2] - (0.7 + STANDOFF_M)) < 1e-9
        assert t.source == "receptacle_top_surface_near"
        assert t.to_dict()["surface_point"] == [round(v, 4) for v in sp]
        preferences.append(abs(np.hypot(sp[0] - 2.0, sp[1] - 1.0)
                               - PLACE_PREFERRED_REACH_M))
    # ordered by preferred arm reach: a preference, not an exclusion
    assert preferences == sorted(preferences)


def test_full_grid_is_not_capped(fake_backend):
    """Round-6: receptacle_place_targets returns the FULL ordered grid (no
    8-point cap); the batch size lives in the feasibility validator."""
    fake_backend._last_anchor_position = [2.0, 1.0, 0.0]
    targets = receptacle_place_targets("countertop", fake_backend)
    assert len(targets) > PLACE_BATCH_SIZE  # (1.48/0.08+1)^2 grid points
    preferences = [abs(np.hypot(t.metadata["surface_point"][0] - 2.0,
                                t.metadata["surface_point"][1] - 1.0)
                       - PLACE_PREFERRED_REACH_M) for t in targets]
    assert preferences == sorted(preferences)  # global preference order


def test_far_surface_still_yields_candidates(fake_backend):
    """Round-5 regression (GPU smoke finding): a surface entirely beyond the
    old 1 m sampling circle still yields candidates. Reachability is
    decided by the production IK/collision check on each candidate, not by
    the sampler."""
    fake_backend._last_anchor_position = [10.0, 10.0, 0.0]
    targets = receptacle_place_targets("countertop", fake_backend)
    assert len(targets) > 0  # full ordered grid, no cap
    distances = [np.hypot(t.metadata["surface_point"][0] - 10.0,
                          t.metadata["surface_point"][1] - 10.0)
                 for t in targets]
    assert distances == sorted(distances)  # nearest-to-preferred first
    for t in targets:
        assert t.metadata["surface_point"][2] == 0.7


def test_surface_just_beyond_one_metre_yields_candidates(fake_backend):
    """The console-table smoke finding: the robot stands ~1.1 m from the
    table and GRASPs from there are certified, but the old 1 m circle
    returned zero PLACE candidates. The whole-surface grid does not."""
    fake_backend._last_anchor_position = [4.8, 1.0, 0.0]
    targets = receptacle_place_targets("countertop", fake_backend)
    assert len(targets) > 0
    for t in targets:
        sp = t.metadata["surface_point"]
        assert np.hypot(sp[0] - 4.8, sp[1] - 1.0) > 1.0  # beyond the old cap
        assert sp[2] == 0.7


def test_articulated_receptacle_offers_place(fake_backend):
    fake_backend._last_anchor_position = [2.2, 1.0, 0.0]
    cab = ArticulatedObject(fake_backend.entities["cabinet_B"], fake_backend,
                            is_receptacle=True)
    assert cab.interaction_targets("PLACE", world=None)
    plain = ArticulatedObject(fake_backend.entities["cabinet_B"], fake_backend,
                              is_receptacle=False)
    assert plain.interaction_targets("PLACE", world=None) == []


def test_place_skill_passes_validated_point(fake_backend):
    fake_backend.holding_entity = "target_knife"
    state = SimpleNamespace(held_object="target_knife", release=lambda: None)
    resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity="countertop",
                              place_point=[2.6, 1.0, 0.7])
    result = PlaceSkill().execute(fake_backend, resolved, state)
    assert result.executed
    assert fake_backend.place_point == [2.6, 1.0, 0.7]
