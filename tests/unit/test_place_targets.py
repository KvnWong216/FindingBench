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
    PLACE_MAX_CANDIDATES,
    PLACE_MAX_REACH_M,
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
    # movable tables and small objects are not support surfaces by category
    assert not build_entity_info(_obj("coffee_table", False)).is_receptacle
    assert not build_entity_info(_obj("mug", False)).is_receptacle
    assert build_entity_info(_obj("bowl", False, {"fillable": {}})).is_receptacle


def test_top_surface_targets_near_robot(fake_backend):
    fake_backend._last_anchor_position = [2.0, 1.0, 0.0]
    targets = receptacle_place_targets("countertop", fake_backend)
    assert 0 < len(targets) <= PLACE_MAX_CANDIDATES
    for t in targets:
        sp = t.metadata["surface_point"]
        assert np.hypot(sp[0] - 2.0, sp[1] - 1.0) <= PLACE_MAX_REACH_M + 1e-9
        assert sp[2] == 0.7  # countertop top
        assert abs(t.pose.position[2] - (0.7 + STANDOFF_M)) < 1e-9
        assert t.source == "receptacle_top_surface_near"
        assert t.to_dict()["surface_point"] == [round(v, 4) for v in sp]


def test_top_surface_targets_out_of_reach_is_empty(fake_backend):
    fake_backend._last_anchor_position = [10.0, 10.0, 0.0]
    assert receptacle_place_targets("countertop", fake_backend) == []


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
