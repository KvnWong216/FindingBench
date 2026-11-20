"""Robot embodiment layer.

The benchmark evaluates embodiment-aware reasoning: what a robot can do
depends on what it IS. The MVP abstracts the embodiment to the parameters
that matter for feasibility checking (arm reach band, hand capacity); a
URDF-based layer (Pinocchio/TRAC-IK) can replace the reach proxy without
touching the benchmark core (see feasibility/ik_solver.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotEmbodiment:
    """Capability parameters used by the feasibility engine."""

    name: str
    reach_radius: float = 1.0  # max arm reach from base origin (m)
    z_min: float = 0.0  # interaction height band (m, world frame)
    z_max: float = 1.6
    hand_capacity: int = 1  # simultaneous grasped objects

    @classmethod
    def from_spec(cls, spec) -> "RobotEmbodiment":
        return cls(
            name=spec.name,
            reach_radius=float(spec.reach_radius),
            z_min=float(spec.z_min),
            z_max=float(spec.z_max),
            hand_capacity=int(spec.hand_capacity),
        )
