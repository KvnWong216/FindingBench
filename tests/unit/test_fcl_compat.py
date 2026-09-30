"""Binding-version compatibility preserves real geometry and frame semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from rummagebench.feasibility import fcl_compat
from rummagebench.feasibility.pinocchio_solver import _frame_parent_joint


@pytest.mark.parametrize("attribute", ["parent", "parentJoint"])
def test_frame_parent_joint_rename(attribute):
    assert _frame_parent_joint(SimpleNamespace(**{attribute: 7})) == 7


@pytest.mark.parametrize("attribute", ["Transform3f", "Transform3s"])
def test_transform_class_rename_preserves_rotation_translation(attribute):
    class Transform:
        def __init__(self, translation):
            self.translation = translation

        def setRotation(self, rotation):
            self.rotation = rotation

    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = fcl_compat.transform3(
        SimpleNamespace(**{attribute: Transform}), [1., 2., 3.], rotation,
    )
    np.testing.assert_array_equal(result.translation, [1., 2., 3.])
    np.testing.assert_array_equal(result.rotation, rotation)


def test_missing_collision_bindings_fail_loudly(monkeypatch):
    attempted = []

    def missing(name):
        attempted.append(name)
        raise ImportError(name)

    monkeypatch.setattr(fcl_compat.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"rummagebench\[kinematics\]"):
        fcl_compat.collision_backend()
    assert attempted == ["coal", "hppfcl"]


def test_hppfcl_fallback_is_explicit(monkeypatch):
    expected = object()

    def load(name):
        if name == "coal":
            raise ImportError(name)
        assert name == "hppfcl"
        return expected

    monkeypatch.setattr(fcl_compat.importlib, "import_module", load)
    assert fcl_compat.collision_backend() is expected


def test_installed_collision_transform_round_trip():
    pytest.importorskip("pinocchio")
    backend = fcl_compat.collision_backend()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transform = fcl_compat.transform3(backend, [1., 2., 3.], rotation)
    np.testing.assert_allclose(transform.getTranslation(), [1., 2., 3.])
    np.testing.assert_allclose(transform.getRotation(), rotation)
