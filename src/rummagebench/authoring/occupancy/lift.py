"""§17 step 5: lift a voxel placement proposal into a continuous pose.

The continuous pose is what the simulator instantiates; small seeded XY/yaw
perturbations are added here (§20 step 9) BEFORE real meshes load, and the
proposed transform is then re-verified against real collision geometry
(revision §7: never reuse an unrotated voxel mask after rotating).
"""
from __future__ import annotations

import numpy as np


def lift_to_pose(origin_world, dims, voxel_size: float, rng,
                 jitter_xy_m: float = 0.015,
                 jitter_yaw_deg: float = 6.0) -> dict:
    """Cell-aligned continuous pose + seeded jitter.

    Returns {"position", "yaw_deg"} — position is the CENTER of the lifted
    footprint base; the caller instantiates the real asset so its (possibly
    off-center) local origin lands on this pose, then lets PhysX settle.
    """
    base = np.asarray(origin_world, dtype=float)
    footprint = np.asarray(dims[:2], dtype=float) * voxel_size
    center_xy = base[:2] + footprint / 2.0
    jitter = (rng.random(2) * 2.0 - 1.0) * jitter_xy_m
    yaw = (rng.random() * 2.0 - 1.0) * float(jitter_yaw_deg)
    return {
        "position": [float(center_xy[0] + jitter[0]),
                     float(center_xy[1] + jitter[1]),
                     float(base[2])],
        "yaw_deg": float(yaw),
        "footprint_m": [float(v) for v in footprint],
    }
