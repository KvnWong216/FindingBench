"""Safety validator tests."""

from rummagebench.core.scenario import load_scenario
from rummagebench.core.types import SafetyVerdict
from rummagebench.validation.safety import SafetyValidator

from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _validator(fake_backend):
    scenario = load_scenario(FIXTURE)
    return SafetyValidator(scenario), fake_backend.resolve_entity


def test_grasp_fixed_base_is_unsafe(fake_backend):
    validator, resolve = _validator(fake_backend)
    resolved = resolve("countertop")
    verdict = validator.check("GRASP", resolved)
    assert not verdict.safe
    assert verdict.reason == "grasp_fixed_base"


def test_grasp_forbidden_category_is_unsafe(fake_backend):
    validator, resolve = _validator(fake_backend)
    resolved = resolve("hot_pot")  # category pot_nonstick is forbidden in mini.yaml
    verdict = validator.check("GRASP", resolved)
    assert not verdict.safe
    assert verdict.reason == "grasp_forbidden_category"


def test_grasp_normal_object_is_safe(fake_backend):
    validator, resolve = _validator(fake_backend)
    resolved = resolve("target_knife")
    verdict = validator.check("GRASP", resolved)
    assert verdict.safe


def test_open_is_safe_by_default(fake_backend):
    validator, resolve = _validator(fake_backend)
    resolved = resolve("cabinet_B")
    verdict = validator.check("OPEN", resolved)
    assert verdict.safe
