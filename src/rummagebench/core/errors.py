"""Benchmark error hierarchy. Simulator processes must never crash on bad agent actions:
those are reported as structured step results instead."""

from __future__ import annotations


class RummageBenchError(Exception):
    """Base class for all benchmark errors."""


class ScenarioValidationError(RummageBenchError):
    """Scenario YAML failed schema or semantic validation at load/build time."""


class SimBackendError(RummageBenchError):
    """The simulator backend could not fulfill a request (build-time or setup only)."""


class UnresolvableTargetError(RummageBenchError):
    """A target reference could not be grounded to any simulator entity/place.

    This is NOT a crash: BenchmarkSession converts it into an invalid-action
    step result. It is raised internally by the grounding layer.
    """


class FeasibilityBackendError(RummageBenchError):
    """The physical feasibility backend could not be initialized or used.

    Raised instead of silently degrading to the reach-radius proxy: production
    benchmark runs must be grounded in real kinematics/collision or fail
    loudly with an actionable message (export the URDF, install pinocchio...).
    """
