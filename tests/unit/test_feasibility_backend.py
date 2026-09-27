"""§23: runtime fallback policy — fail loudly, never silently degrade.

- production default is backend: pinocchio; a missing URDF or missing
  robot.kinematics block raises FeasibilityBackendError
- backend: proxy is an explicit test-mode choice
- a pinocchio-configured session never shrugs into the reach-radius proxy
"""

from pathlib import Path

import pytest

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.scenario import KinematicsSpec, load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.robots.model_loader import (
    classify_robot_link,
    load_kinematics_backend,
    resolve_urdf_path,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"
FIXTURES = Path(__file__).parent / "fixtures"


def test_missing_kinematics_block_fails_loudly():
    scenario = load_scenario(FIXTURE)
    scenario.feasibility.backend = "pinocchio"
    scenario.robot.kinematics = None
    with pytest.raises(FeasibilityBackendError, match="robot.kinematics"):
        load_kinematics_backend(scenario.robot, scenario.feasibility)


def test_missing_urdf_fails_loudly():
    scenario = load_scenario(FIXTURE)
    scenario.feasibility.backend = "pinocchio"
    scenario.robot.kinematics = KinematicsSpec(
        urdf_path="tests/unit/fixtures/robots/does_not_exist.urdf",
        end_effector_link="gripper",
    )
    with pytest.raises(FeasibilityBackendError, match="not found"):
        load_kinematics_backend(scenario.robot, scenario.feasibility)


def test_pinocchio_session_requires_real_backend(fake_backend):
    """A session configured for pinocchio without a usable URDF must raise at
    construction — never fall back to the reach-radius proxy."""
    scenario = load_scenario(FIXTURE)
    scenario.feasibility.backend = "pinocchio"
    scenario.robot.kinematics = KinematicsSpec(
        urdf_path="tests/unit/fixtures/robots/nope.urdf",
        end_effector_link="gripper",
    )
    with pytest.raises(FeasibilityBackendError):
        BenchmarkSession(fake_backend, scenario)


def test_explicit_proxy_backend_is_allowed_for_tests(fake_backend):
    scenario = load_scenario(FIXTURE)
    assert scenario.feasibility.backend == "proxy"
    session = BenchmarkSession(fake_backend, scenario)
    assert session._feasibility.engine == "proxy"


def test_urdf_path_resolution(tmp_path):
    with pytest.raises(FeasibilityBackendError):
        resolve_urdf_path("no/such/file.urdf")
    resolved = resolve_urdf_path("tests/unit/fixtures/robots/short_arm.urdf")
    assert resolved.is_file()


def test_link_classification():
    assert classify_robot_link("gripper", "gripper", None) == "gripper"
    assert classify_robot_link("left_finger", "tool", None) == "gripper"
    assert classify_robot_link("forearm_link", "tool", None) == "arm"
    assert classify_robot_link("anything", "tool", ["anything"]) == "gripper"
