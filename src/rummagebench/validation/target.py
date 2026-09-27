"""Target validator: is the resolved target actionable?

Semantic validity is about the action schema; target validity is about the
world: entity must exist, OPEN targets must be openable, NAV places must be
defined anchors (resolved by grounding). Invalid targets are structured
step results, never crashes.
"""

from __future__ import annotations

from dataclasses import dataclass

from rummagebench.sim.base import ResolvedTarget


@dataclass
class TargetVerdict:
    valid: bool
    reason: str = "none"


class TargetValidator:
    def check(self, skill_name: str, resolved: ResolvedTarget) -> TargetVerdict:
        if skill_name == "NAV":
            if resolved.place is None or resolved.anchor is None:
                return TargetVerdict(valid=False, reason="unknown place")
            return TargetVerdict(valid=True)

        # OPEN / GRASP operate on entities
        if resolved.entity is None or resolved.info is None:
            return TargetVerdict(valid=False, reason="unknown entity")
        if skill_name == "OPEN" and not resolved.info.openable:
            return TargetVerdict(
                valid=False, reason=f"entity {resolved.entity!r} is not openable"
            )
        # GRASP does not reject fixed-base objects here: "grasping furniture" is
        # a SAFE category question, owned by the SafetyValidator (FAIL_UNSAFE_ACTION).
        return TargetVerdict(valid=True)
