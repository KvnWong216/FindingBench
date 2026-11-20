"""Canonical action schema tests."""

import pytest
from pydantic import ValidationError

from rummagebench.core.types import Action, TargetKind, TargetRef


def test_place_target():
    a = Action.from_dict({"skill": "NAV", "target": {"type": "place", "value": "kitchen"}})
    assert a.target.type == TargetKind.PLACE
    assert a.to_dict() == {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}


def test_entity_target():
    a = Action.from_dict(
        {"skill": "OPEN", "target": {"type": "entity", "value": "cabinet_B"}}
    )
    assert a.target.type == TargetKind.ENTITY


def test_pixel_target_requires_coordinates():
    with pytest.raises(ValidationError):
        TargetRef(type=TargetKind.PIXEL)
    t = TargetRef(type=TargetKind.PIXEL, camera="head_rgb", x=0.1, y=0.2)
    assert t.value == ""


def test_pixel_coordinates_bounded():
    with pytest.raises(ValidationError):
        TargetRef(type=TargetKind.PIXEL, camera="head_rgb", x=1.5, y=0.2)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Action.from_dict(
            {"skill": "NAV", "target": {"type": "place", "value": "k", "bogus": 1}}
        )
