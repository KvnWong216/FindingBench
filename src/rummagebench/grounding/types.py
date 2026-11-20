"""Target grounding types.

TargetRef (what the agent returns) is resolved into a ResolvedTarget (what the
backend understands). MVP ships Oracle/Debug grounding over entity names; the
pixel mode is kept in the data model and this package is where a visual
grounding resolver plugs in later, mapping a normalized pixel through the
evaluator-private GT segmentation to a simulator instance.
"""

from __future__ import annotations

from rummagebench.core.types import TargetRef


class TargetGrounding:
    """Resolver protocol: TargetRef -> ResolvedTarget (raises UnresolvableTargetError)."""

    def resolve(self, target: TargetRef) -> "object":  # noqa: ANN001 - set by subclass
        raise NotImplementedError
