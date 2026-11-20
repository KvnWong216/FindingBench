"""Interaction interfaces: WHERE the tool must go for a skill to apply.

The feasibility engine never grounds a skill at the entity root pose:

- Visual OPEN/CLOSE use native handle links when available, otherwise the
  exact clicked surface point. No alternative contact search is performed.
  Symbolic/internal callers without a pixel retain the legacy link anchor.
- GRASP uses canonical rigid-object candidates (AABB center + face centers).
  No grasp-pose annotation, no grasp-quality model: candidates only answer
  'does at least one reachable, collision-free interaction configuration
  exist?'.
- PLACE is grounded on the receptacle region (top surface / inside volume),
  never the receptacle root origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import logging

import numpy as np

from rummagebench.feasibility.ik_solver import (
    Pose,
    aabb_face_candidates,
    aabb_surface_point,
    look_at_quaternion,
)

logger = logging.getLogger(__name__)

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
    # optional XY sub-box (x, y) from which the region is open from above
    # (an open drawer: the part pulled out beyond the furniture body; the
    # rest lies under the body / its countertop). PLACE candidates are
    # restricted to it; containment checks still use lo/hi.
    reach_lo: Optional[list[float]] = None
    reach_hi: Optional[list[float]] = None
    # optional unit XY direction an open drawer is pulled out along (from the
    # body outward); enables angled PLACE approaches into the cavity
    open_dir: Optional[list[float]] = None


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
        if "surface_point" in self.metadata:
            out["surface_point"] = [round(float(v), 4)
                                    for v in self.metadata["surface_point"]]
        return out


# --------------------------------------------------------------------------
# articulated objects: OPEN / CLOSE at the interaction interface
# --------------------------------------------------------------------------


_MOVABLE = ("revolute", "prismatic", "continuous")


def articulated_interaction_targets(entity: str, backend: Any) -> list[InteractionTarget]:
    """Interaction anchors for an openable entity, from its articulation.

    Priority: native handle link, then the actual visual click when provided.
    Legacy symbolic callers without a click retain the moving-link anchor.
    The entity root pose is never used as the contact point.
    """
    art = backend.articulation_info(entity)
    if art is None:
        raise RuntimeError(
            f"backend provides no articulation_info for {entity!r}: OPEN/CLOSE "
            "grounding requires the articulated interaction interface "
            "(fallback to the entity root pose is forbidden)"
        )

    clicked = getattr(backend, "_clicked_articulation", None)
    if not art.handle_link and clicked is not None and clicked[0] == entity:
        point = np.asarray(clicked[1], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Invalid clicked articulation surface point")
        interfaces = []
        for joint in art.joints:
            if joint.joint_type not in _MOVABLE or not joint.child_link:
                continue
            bounds = backend.link_aabb(entity, joint.child_link)
            if bounds is None:
                continue
            lo, hi = (np.asarray(v, dtype=float) for v in bounds)
            distance = float(np.linalg.norm(point - np.clip(point, lo, hi)))
            interfaces.append((distance, joint, bounds))
        if not interfaces:
            return []
        _, joint, bounds = min(interfaces, key=lambda row: row[0])
        _, outward, _ = _link_anchor(entity, backend, joint, joint.child_link)
        if outward is None:
            return []
        outward = np.asarray(outward, dtype=float)
        return [InteractionTarget(
            pose=Pose(point + outward * STANDOFF_M, look_at_quaternion(-outward)),
            source="clicked_articulation_surface", link=joint.child_link,
            region=InteractionRegion(list(bounds[0]), list(bounds[1]), "link_surface"),
            metadata={"surface_point": point.tolist(), "standoff_m": STANDOFF_M,
                      "joint": joint.name, "affordance_source": "user_selected_pixel"},
        )]

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
    try:
        link_pose = backend.link_pose(entity, link)
        lo_hi = backend.link_aabb(entity, link)
    except Exception as e:
        # PhysX execution noise can corrupt individual links: this candidate
        # interface is skipped, the grounding pass continues
        logger.warning("link anchor query failed for %s/%s: %s", entity, link, e)
        return None, None, False
    if link_pose is None:
        return None, None, False
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


# top-surface PLACE sampling (round-5): a grid over the WHOLE support
# surface, ordered by |d - PLACE_PREFERRED_REACH_M| — a preference, not an
# exclusion. There is NO metric-circle pre-filter: every returned candidate
# goes through the production feasibility check (real IK + configuration-
# space collision + approach), which is the actual embodiment/collision
# verdict. A fixed 1 m sampling circle rejected points the arm can reach
# (certified GRASPs beyond 1 m) and yielded zero candidates on surfaces the
# robot was interacting with.
PLACE_GRID_STEP_M = 0.08
PLACE_EDGE_INSET_M = 0.06
PLACE_PREFERRED_REACH_M = 0.65
# round-6: batch size for the feasibility validator, NOT a total-candidate
# cap — the FULL ordered grid is handed to the validator, which walks it
# batch by batch until a complete feasible witness is found or the discrete
# candidate set is exhausted.
PLACE_BATCH_SIZE = 8


# inside-volume PLACE (2026-10-18): the object is set down on the region
# floor (lo z + offset), approached top-down, on the same reach-ordered grid
# as top surfaces. Used for open drawers of fixed furniture (backend
# receptacle_region) and open movable containers.
INSIDE_FLOOR_OFFSET_M = 0.02
# inside volumes are walled: the held object's footprint (at the identity
# orientation it is set down with) plus this clearance must stay inside the
# region (2026-10-19: an 18 cm fork centred 6 cm from a drawer wall was
# teleported into the wall and fell out of the drawer)
INSIDE_WALL_CLEARANCE_M = 0.02
# angled PLACE into an open drawer (2026-10-19, user-approved): the tool
# comes in from the pulled-out side, tilted from vertical toward the body by
# these angles (deg), so the fingers can reach under a countertop that covers
# the drawer. 0 = straight down. Ordered: tried in this order.
INSIDE_APPROACH_TILTS_DEG = (45.0, 30.0, 60.0, 0.0)
# a fillable drawer link counts as pulled out (the open one) when its XY
# AABB protrudes beyond the furniture body by more than this
DRAWER_OPEN_PROTRUSION_M = 0.05


def select_open_interior(body_lo, body_hi, interiors, min_protrusion=DRAWER_OPEN_PROTRUSION_M):
    """Pick the interior region (``(name, InteractionRegion)``) of a fixed
    piece of furniture that is pulled out beyond its body XY AABB by more
    than ``min_protrusion``; among several (OPEN opens every drawer of the
    piece) the TOP-most one, the only one not covered by another open
    drawer and so reachable from above (2026-10-19). None when none
    protrudes."""
    best = None
    for name, region in interiors:
        protrusion = max(body_lo[0] - region.lo[0], region.hi[0] - body_hi[0],
                         body_lo[1] - region.lo[1], region.hi[1] - body_hi[1])
        if protrusion > min_protrusion and (best is None or region.hi[2] > best[0]):
            best = (region.hi[2], name, region)
    return None if best is None else (best[1], best[2])


def _robot_xy(backend: Any):
    try:
        pos, _ = backend.robot_pose()
    except Exception:
        return None
    if pos is None or not np.all(np.isfinite(np.asarray(pos, dtype=float)[:2])):
        return None
    return float(pos[0]), float(pos[1])


def _near_top_surface_points(lo, hi, robot_xy, inset=(PLACE_EDGE_INSET_M, PLACE_EDGE_INSET_M),
                             reach_lo=None, reach_hi=None) -> list[np.ndarray]:
    """Grid the WHOLE support surface, order by |d - PLACE_PREFERRED_REACH_M|
    (comfortable reach first). Round-6: the FULL ordered set is returned —
    distance orders the check sequence and never excludes a candidate; the
    feasibility validator walks it in batches of PLACE_BATCH_SIZE.
    ``inset`` (x, y) keeps points away from the region edges; an axis too
    narrow for it collapses to the region centre on that axis."""
    def axis(i):
        a, b = lo[i] + inset[i], hi[i] - inset[i]
        if a > b:
            a = b = (lo[i] + hi[i]) / 2.0
        if reach_lo is not None and reach_hi is not None:
            a, b = max(a, float(reach_lo[i])), min(b, float(reach_hi[i]))
            if a > b:  # footprint and open strip disagree: nearest feasible centre
                a = b = min(max((lo[i] + hi[i]) / 2.0, float(reach_lo[i])), float(reach_hi[i]))
        if a == b:
            return np.array([a])
        return np.arange(a, b + 1e-9, PLACE_GRID_STEP_M)
    xs, ys = axis(0), axis(1)
    scored = []
    for x in xs:
        for y in ys:
            d = float(np.hypot(x - robot_xy[0], y - robot_xy[1]))
            scored.append((abs(d - PLACE_PREFERRED_REACH_M), float(x), float(y)))
    scored.sort()
    return [np.array([x, y, float(hi[2])]) for _, x, y in scored]


def _held_half_extents(backend: Any):
    """Placement half extents of the object currently in hand, or None."""
    held_name = getattr(backend, "held_entity_name", None)
    extents = getattr(backend, "place_half_extents", None)
    if not callable(held_name) or not callable(extents):
        return None
    try:
        name = held_name()
        return None if name is None else extents(name)
    except Exception:
        return None


def receptacle_place_targets(entity: str, backend: Any) -> list[InteractionTarget]:
    """PLACE targets from the receptacle support region, never the root pose.

    Top surfaces: a grid over the whole surface, ordered by preferred arm
    reach (tool pointing down) — reachability itself is decided by the
    production IK/collision check on each candidate, not by a sampling
    radius. Without a robot pose (or for an inside volume): the region
    center + four in-plane edge midpoints.
    """
    clicked = getattr(backend, '_clicked_place', None)
    if clicked is not None and clicked[0] == entity:
        point=np.asarray(clicked[1], dtype=float)
        approach=np.array([0.,0.,-1.])
        return [InteractionTarget(pose=Pose(point-approach*STANDOFF_M,look_at_quaternion(approach)),
            source='receptacle_clicked_surface',region=clicked[2],
            metadata={'surface_point':point.tolist(),'approach_normal':approach.tolist(),
                      'standoff_m':STANDOFF_M})]
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
    # both kinds are approached top-down; an inside volume is filled from its
    # floor (an open drawer / container is reachable only from above)
    approach = np.array([0.0, 0.0, -1.0])  # tool pointing down
    support_z = float(hi[2]) if region.kind == "top_surface" else float(lo[2]) + INSIDE_FLOOR_OFFSET_M
    center[2] = support_z
    orientation = look_at_quaternion(approach)

    inset = (PLACE_EDGE_INSET_M, PLACE_EDGE_INSET_M)
    if region.kind == "inside_volume":
        half = _held_half_extents(backend)
        if half is not None:
            inset = tuple(max(PLACE_EDGE_INSET_M, float(half[i]) + INSIDE_WALL_CLEARANCE_M)
                          for i in (0, 1))

    robot_xy = _robot_xy(backend)
    if robot_xy is not None:
        points = _near_top_surface_points(lo, hi, robot_xy, inset,
                                          getattr(region, "reach_lo", None),
                                          getattr(region, "reach_hi", None))
        for point in points:
            point[2] = support_z
        source = ("receptacle_top_surface_near" if region.kind == "top_surface"
                  else "receptacle_inside_floor_near")
    else:
        # region center + four in-plane edge midpoints (deterministic samples)
        half = (hi - lo) / 2.0
        points = [center.copy()]
        for axis in (0, 1):
            for sign in (1.0, -1.0):
                off = np.zeros(3)
                off[axis] = sign * half[axis] * 0.5
                points.append(center + off)
        source = f"receptacle_{region.kind}"

    if region.kind == "inside_volume" and robot_xy is not None \
            and getattr(region, "open_dir", None) is not None:
        return _angled_inside_targets(region, points, lo, hi, robot_xy, inset, support_z)

    targets: list[InteractionTarget] = []
    for point in points:
        pose_pos = point - approach * STANDOFF_M
        targets.append(
            InteractionTarget(
                pose=Pose(pose_pos, orientation),
                source=source,
                region=region,
                metadata={
                    "approach_normal": [float(v) for v in approach],
                    "standoff_m": STANDOFF_M,
                    "surface_point": [float(v) for v in point],
                },
            )
        )
    return targets


def _angled_inside_targets(region, strip_points, lo, hi, robot_xy, inset, support_z
                           ) -> list[InteractionTarget]:
    """PLACE targets into an open drawer with tilted approaches.

    Points in the pulled-out strip come first, then the rest of the cavity
    (reachable only at a tilt, fingers under the countertop); each point is
    offered at every INSIDE_APPROACH_TILTS_DEG, tilt-major so the first
    validator batches already mix points and tilts. The standoff lies along
    the approach direction, outside / above the drawer."""
    d = np.asarray(region.open_dir[:2], dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    inward = np.array([-d[0], -d[1], 0.0])  # toward the furniture body
    seen = {(round(float(p[0]), 4), round(float(p[1]), 4)) for p in strip_points}
    rest = [p for p in _near_top_surface_points(lo, hi, robot_xy, inset)
            if (round(float(p[0]), 4), round(float(p[1]), 4)) not in seen]
    points = list(strip_points) + rest
    for point in points:
        point[2] = support_z
    targets: list[InteractionTarget] = []
    for tilt in INSIDE_APPROACH_TILTS_DEG:
        t = np.deg2rad(tilt)
        approach = np.cos(t) * np.array([0.0, 0.0, -1.0]) + np.sin(t) * inward
        approach = approach / np.linalg.norm(approach)
        orientation = look_at_quaternion(approach)
        for point in points:
            targets.append(
                InteractionTarget(
                    pose=Pose(point - approach * STANDOFF_M, orientation),
                    source="receptacle_inside_floor_angled",
                    region=region,
                    metadata={
                        "approach_normal": [float(v) for v in approach],
                        "approach_tilt_deg": float(tilt),
                        "standoff_m": STANDOFF_M,
                        "surface_point": [float(v) for v in point],
                    },
                )
            )
    return targets
