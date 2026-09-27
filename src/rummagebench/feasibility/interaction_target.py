"""Interaction interfaces: WHERE the tool must go for a skill to apply.

The feasibility engine never grounds a skill at the entity root pose:

- OPEN/CLOSE are grounded at the articulated interaction interface: the
  handle link when the asset annotates one, else the surface-facing anchor of
  the moving link (source='fallback_link_anchor'), auto-derived from the
  articulation — never hand-authored per asset.
- GRASP uses canonical rigid-object candidates (AABB center + face centers).
  No grasp-pose annotation, no grasp-quality model: candidates only answer
  'does at least one reachable, collision-free interaction configuration
  exist?'.
- PLACE is grounded on the receptacle region (top surface / inside volume),
  never the receptacle root origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rummagebench.feasibility.ik_solver import (
    Pose,
    aabb_face_candidates,
    aabb_surface_point,
    look_at_quaternion,
)

# gripper-frame standoff behind each interface surface point: the tool frame
# is placed this far back along -outward so the tool body approaches the
# surface without boundary-grazing contact (recorded in target metadata).
STANDOFF_M = 0.06


@dataclass
class InteractionRegion:
    """World-frame AABB of the admissible interaction region."""

    lo: list[float]
    hi: list[float]
    kind: str  # 'link_surface' | 'top_surface' | 'inside_volume'


@dataclass
class InteractionTarget:
    """One candidate interaction configuration target (world frame)."""

    pose: Pose | None
    source: str  # handle_link | fallback_link_anchor | fallback_link_frame |
    #              canonical_candidates | receptacle_top_surface | ...
    region: InteractionRegion | None = None
    # articulated link the tool may touch (allowed-collision matrix input)
    link: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source, "link": self.link}
        if self.pose is not None:
            out["position"] = [round(float(v), 4) for v in self.pose.position]
        if self.region is not None:
            out["region_kind"] = self.region.kind
        return out


# --------------------------------------------------------------------------
# articulated objects: OPEN / CLOSE at the interaction interface
# --------------------------------------------------------------------------


_MOVABLE = ("revolute", "prismatic", "continuous")


def articulated_interaction_targets(entity: str, backend: Any) -> list[InteractionTarget]:
    """Interaction anchors for an openable entity, from its articulation.

    Fallback chain (automatic, recorded in ``source``): explicit handle link
    -> moving-link surface-facing anchor derived from the joint's child link
    AABB and the parent->child direction. The entity root pose is never used.
    """
    art = backend.articulation_info(entity)
    if art is None:
        raise RuntimeError(
            f"backend provides no articulation_info for {entity!r}: OPEN/CLOSE "
            "grounding requires the articulated interaction interface "
            "(fallback to the entity root pose is forbidden)"
        )

    targets: list[InteractionTarget] = []
    seen: list[tuple[float, float, float]] = []

    movable = [j for j in art.joints if j.joint_type in _MOVABLE]
    if not movable:
        return targets

    for joint in movable:
        link = joint.child_link
        if art.handle_link:
            link = art.handle_link
        if link is None:
            continue

        anchor_pos, outward, used_aabb = _link_anchor(entity, backend, joint, link)
        if anchor_pos is None:
            continue
        if art.handle_link:
            source = "handle_link"
        elif used_aabb:
            source = "fallback_link_anchor"
        else:
            source = "fallback_link_frame"
        if any(
            abs(anchor_pos[0] - s[0]) < 1e-6
            and abs(anchor_pos[1] - s[1]) < 1e-6
            and abs(anchor_pos[2] - s[2]) < 1e-6
            for s in seen
        ):
            continue
        seen.append(tuple(float(v) for v in anchor_pos))

        pose_pos = np.asarray(anchor_pos, dtype=float) + np.asarray(outward, dtype=float) * STANDOFF_M
        # tool z-axis = approach direction (points INTO the surface)
        pose = Pose(
            pose_pos,
            look_at_quaternion(-np.asarray(outward, dtype=float)),
        )
        region_aabb = backend.link_aabb(entity, link)
        region = (
            InteractionRegion(
                [float(v) for v in region_aabb[0]],
                [float(v) for v in region_aabb[1]],
                "link_surface",
            )
            if region_aabb is not None
            else None
        )
        targets.append(
            InteractionTarget(
                pose=pose,
                source=source,
                region=region,
                link=link,
                metadata={
                    "joint": joint.name,
                    "joint_type": joint.joint_type,
                    "joint_axis": joint.axis,
                    "joint_limits": list(joint.limits) if joint.limits else None,
                    "parent_link": joint.parent_link,
                    "standoff_m": STANDOFF_M,
                },
            )
        )
    return targets


def _link_anchor(entity: str, backend: Any, joint, link: str):
    """Surface-facing anchor of the interaction link + outward direction.

    outward: horizontal parent-link -> child-link direction (the face the
    user approaches from), projected onto the link's world AABB surface.
    Returns (anchor | None, outward | None, used_aabb).
    """
    link_pose = backend.link_pose(entity, link)
    if link_pose is None:
        return None, None, False
    lo_hi = backend.link_aabb(entity, link)
    anchor = np.asarray(link_pose.position, dtype=float)

    parent_pose = (
        backend.link_pose(entity, joint.parent_link) if joint.parent_link else None
    )
    if parent_pose is not None:
        outward = anchor - np.asarray(parent_pose.position, dtype=float)
    else:
        entity_center = np.asarray(backend.entity_pose(entity), dtype=float)
        outward = anchor - entity_center
    outward[2] = 0.0  # horizontal approach
    if np.linalg.norm(outward) < 1e-6:
        outward = np.array([1.0, 0.0, 0.0])
    outward = outward / np.linalg.norm(outward)

    if lo_hi is not None:
        anchor = aabb_surface_point(lo_hi[0], lo_hi[1], outward)
        return anchor, outward, True
    return anchor, outward, False


# --------------------------------------------------------------------------
# rigid objects: canonical GRASP candidates (no grasp annotation required)
# --------------------------------------------------------------------------


def rigid_interaction_candidates(entity: str, backend: Any) -> list[InteractionTarget]:
    """Canonical grasp candidates: AABB center + six face centers, each with
    an outward-normal approach orientation."""
    aabb = backend.entity_aabb(entity)
    if aabb is None:
        raise RuntimeError(
            f"backend provides no AABB for {entity!r}: canonical GRASP "
            "candidates need the object's world-frame AABB"
        )
    lo, hi = aabb
    targets: list[InteractionTarget] = []
    for center, normal in aabb_face_candidates(lo, hi):
        stand_off = center + normal * STANDOFF_M
        targets.append(
            InteractionTarget(
                pose=Pose(stand_off, look_at_quaternion(-normal)),
                source="canonical_candidates",
                region=InteractionRegion([float(v) for v in lo], [float(v) for v in hi],
                                         "link_surface"),
                metadata={
                    "approach_normal": [float(v) for v in normal],
                    "standoff_m": STANDOFF_M,
                },
            )
        )
    return targets


# --------------------------------------------------------------------------
# receptacles: PLACE regions (top surface / inside volume)
# --------------------------------------------------------------------------


def receptacle_place_targets(entity: str, backend: Any) -> list[InteractionTarget]:
    """PLACE targets from the receptacle support region, never the root pose."""
    region = backend.receptacle_region(entity)
    if region is None:
        aabb = backend.entity_aabb(entity)
        if aabb is None:
            raise RuntimeError(
                f"backend provides no place region for {entity!r}"
            )
        lo, hi = aabb
        region = InteractionRegion([float(v) for v in lo], [float(v) for v in hi],
                                   "top_surface")

    lo = np.asarray(region.lo, dtype=float)
    hi = np.asarray(region.hi, dtype=float)
    center = (lo + hi) / 2.0
    if region.kind == "top_surface":
        approach = np.array([0.0, 0.0, -1.0])  # tool pointing down onto surface
        center[2] = hi[2]
    else:  # inside_volume
        approach = np.array([0.0, 0.0, 1.0])
    orientation = look_at_quaternion(approach)

    # region center + four in-plane edge midpoints (deterministic samples)
    half = (hi - lo) / 2.0
    offsets = [np.zeros(3)]
    for axis in (0, 1):
        for sign in (1.0, -1.0):
            off = np.zeros(3)
            off[axis] = sign * half[axis] * 0.5
            offsets.append(off)

    targets: list[InteractionTarget] = []
    for off in offsets:
        pose_pos = center + off - approach * STANDOFF_M
        targets.append(
            InteractionTarget(
                pose=Pose(pose_pos, orientation),
                source=f"receptacle_{region.kind}",
                region=region,
                metadata={
                    "approach_normal": [float(v) for v in approach],
                    "standoff_m": STANDOFF_M,
                },
            )
        )
    return targets
