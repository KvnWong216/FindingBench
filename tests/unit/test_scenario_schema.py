"""Scenario schema validation tests."""

from pathlib import Path

import pytest

from rummagebench.core.scenario import ScenarioSpec, load_scenario

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def test_load_valid_scenario():
    spec = load_scenario(FIXTURE)
    assert spec.id == "mini_search_000"
    assert spec.robot.init_anchor == "living_room"
    assert spec.target.entity == "target_knife"
    assert spec.termination.max_planning_steps == 6


def test_init_anchor_must_exist():
    raw = {
        "id": "bad",
        "instruction": "x",
        "scene": {"model": "s"},
        "robot": {"init_anchor": "nowhere"},
        "anchors": {},
        "target": {"entity": "k", "category": "knife"},
        "objects": [{"name": "k", "category": "knife"}],
    }
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(raw)


def test_target_must_be_spawned_object():
    raw = {
        "id": "bad",
        "instruction": "x",
        "scene": {"model": "s"},
        "robot": {"init_anchor": "a"},
        "anchors": {"a": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}},
        "target": {"entity": "ghost", "category": "knife"},
        "objects": [{"name": "k", "category": "knife"}],
    }
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(raw)


def test_placement_entity_must_be_spawned():
    raw = {
        "id": "bad",
        "instruction": "x",
        "scene": {"model": "s"},
        "robot": {"init_anchor": "a"},
        "anchors": {"a": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}},
        "target": {"entity": "k", "category": "knife"},
        "objects": [{"name": "k", "category": "knife"}],
        "placements": [{"entity": "ghost", "relation": "inside", "receptacle": "c"}],
    }
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(raw)


def test_knife_search_pins_knife_to_top_drawer():
    spec = load_scenario(Path("scenarios/knife_search_001/scenario.yaml"))
    knife = next(p for p in spec.placements if p.entity == "target_knife")
    assert (knife.relation, knife.link) == ("inside", "link_2")


def test_placement_link_requires_inside():
    from rummagebench.core.scenario import PlacementSpec

    assert PlacementSpec(entity="k", relation="inside", receptacle="c").link is None
    with pytest.raises(Exception):
        PlacementSpec(entity="k", relation="on_top", receptacle="c", link="link_0")


def _goal_raw(goal):
    return {
        "id": "goal_receptacle",
        "instruction": "x",
        "scene": {"model": "s"},
        "robot": {"init_anchor": "a"},
        "anchors": {"a": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}},
        "target": {"entity": "k", "category": "knife"},
        "objects": [{"name": "k", "category": "knife"}],
        "goal": goal,
    }


def test_goal_receptacle_may_be_native_scene_furniture():
    # native furniture is resolved by the backend at evaluation time, the
    # same way placement receptacles are
    spec = ScenarioSpec.model_validate(_goal_raw(
        [{"kind": "on_top", "entity": "k", "receptacle": "breakfast_table_uhrsex_0"}]))
    assert spec.goal[0].receptacle == "breakfast_table_uhrsex_0"


def test_goal_entity_must_still_be_spawned():
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(_goal_raw(
            [{"kind": "on_top", "entity": "ghost", "receptacle": "table_0"}]))


def test_goal_receptacle_cannot_be_its_own_entity():
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(_goal_raw(
            [{"kind": "on_top", "entity": "k", "receptacle": "k"}]))


def _placement_raw(placement):
    raw = _goal_raw([])
    raw["placements"] = [placement]
    return raw


def test_placement_position_xy_for_on_top():
    spec = ScenarioSpec.model_validate(_placement_raw(
        {"entity": "k", "relation": "on_top", "receptacle": "shelf", "position_xy": [4.4, -5.5]}))
    assert spec.placements[0].position_xy == [4.4, -5.5]
    # omitted -> sampler path (None)
    spec = ScenarioSpec.model_validate(_placement_raw(
        {"entity": "k", "relation": "on_top", "receptacle": "shelf"}))
    assert spec.placements[0].position_xy is None


def test_private_inside_orientation_hint_is_a_unit_quaternion():
    from math import sqrt
    from rummagebench.core.scenario import PlacementSpec
    base=dict(entity='tool',relation='inside',receptacle='drawer')
    assert PlacementSpec(**base,orientation_hint_xyzw=[0,sqrt(.5),0,sqrt(.5)]).orientation_hint_xyzw
    for q in ([0,0,0,0],[0,0,1], [float('nan'),0,0,1]):
        with pytest.raises(Exception):PlacementSpec(**base,orientation_hint_xyzw=q)
    with pytest.raises(Exception):PlacementSpec(**dict(base,relation='on_top'),orientation_hint_xyzw=[0,0,0,1])
    p=PlacementSpec(**dict(base,relation='on_top'),position_xy=[1.,2.],orientation_hint_xyzw=[0,sqrt(.5),0,sqrt(.5)])
    assert p.position_xy==[1.,2.]


def test_placement_position_xy_rejected_for_inside_or_bad_shape():
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(_placement_raw(
            {"entity": "k", "relation": "inside", "receptacle": "c", "position_xy": [0, 0]}))
    with pytest.raises(Exception):
        ScenarioSpec.model_validate(_placement_raw(
            {"entity": "k", "relation": "on_top", "receptacle": "c", "position_xy": [0, 0, 1]}))
