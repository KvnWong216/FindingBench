"""NAV(place): semantic navigation by teleporting to a predefined anchor.

Navigation is a benchmark skill here, but low-level locomotion is not
evaluated: the executor looks up the anchor and directly changes the robot
pose, settles physics, and lets the session capture a new RGB observation.
A NAV consumes exactly one semantic planning step; physics settling frames
are not planning steps.

Postcondition: the robot is located at the requested semantic anchor.
"""

from __future__ import annotations

from rummagebench.core.types import SkillResult, TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend


class NavSkill:
    name = "NAV"
    target_kind = TargetKind.PLACE

    def execute(self, backend: SimBackend, resolved: ResolvedTarget, state) -> SkillResult:
        assert resolved.anchor is not None, "NAV requires a resolved anchor"
        backend.teleport_robot(resolved.anchor)
        backend.settle()
        pos, _quat = backend.robot_pose()
        at_anchor = _close(pos, resolved.anchor.position)
        return SkillResult(
            skill=self.name,
            target={"type": TargetKind.PLACE.value, "value": resolved.place},
            executed=True,
            postcondition_satisfied=at_anchor,
            events=[
                {
                    "event": "nav_teleport",
                    "place": resolved.place,
                    "position": [round(c, 4) for c in pos],
                }
            ],
            details={"anchor": resolved.place},
        )


def _close(a: list[float], b: list[float], tol: float = 0.5) -> bool:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 <= tol
