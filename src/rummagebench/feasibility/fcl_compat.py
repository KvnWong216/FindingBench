"""Explicit compatibility for coal and its NumPy-1-compatible hpp-fcl API.

Both bindings perform the same configuration-space collision queries. This
adapter never substitutes the geometric proxy when either binding is absent.
"""

from __future__ import annotations

import importlib

import numpy as np


def collision_backend():
    """Load coal when available, otherwise the supported hppfcl binding."""
    try:
        return importlib.import_module("coal")
    except ImportError:
        try:
            return importlib.import_module("hppfcl")
        except ImportError as exc:
            raise ImportError(
                "coal/hpp-fcl is required for configuration-space collision "
                "checking; install rummagebench[kinematics]"
            ) from exc


def transform3(backend, translation, rotation=None):
    """Construct an FCL rigid transform across the Transform3s/3f rename."""
    cls = getattr(backend, "Transform3s", None)
    if cls is None:
        cls = backend.Transform3f
    result = cls(np.asarray(translation, dtype=float))
    if rotation is not None:
        result.setRotation(np.asarray(rotation, dtype=float))
    return result
