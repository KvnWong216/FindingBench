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
