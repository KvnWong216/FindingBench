"""Target validator tests (with the fake backend grounding)."""

import pytest

from rummagebench.core.errors import UnresolvableTargetError
from rummagebench.validation.target import TargetValidator


def test_unknown_entity_is_invalid(fake_backend):
    # entity resolution itself must raise a structured error, not crash
    with pytest.raises(UnresolvableTargetError):
        fake_backend.resolve_entity("ghost_object")


def test_open_non_openable_is_invalid_target(fake_backend):
    from rummagebench.validation.target import TargetValidator

    resolved = fake_backend.resolve_entity("countertop")
    verdict = TargetValidator().check("OPEN", resolved)
    assert not verdict.valid


def test_open_openable_is_valid(fake_backend):
    from rummagebench.validation.target import TargetValidator

    resolved = fake_backend.resolve_entity("cabinet_B")
    verdict = TargetValidator().check("OPEN", resolved)
    assert verdict.valid


def test_grasp_fixed_base_passes_target_validation_unsafe_is_separately_classified(fake_backend):
    # A fixed-base entity is a valid GRASP *target*; its danger is a safety
    # question and must terminate as FAIL_UNSAFE_ACTION (see test_horizon).
    from rummagebench.validation.target import TargetValidator

    resolved = fake_backend.resolve_entity("countertop")
    verdict = TargetValidator().check("GRASP", resolved)
    assert verdict.valid
