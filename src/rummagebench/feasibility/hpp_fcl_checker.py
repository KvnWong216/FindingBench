"""Configuration-space collision checking (hpp-fcl / coal).

robot configuration q -> URDF <collision> links -> world collision geometry
-> pairwise coal queries, with self-collision and robot-world collision
separated and attributed. Contact admissibility comes from the
AllowedCollisionMatrix (finger<->target allowed, arm<->foreign forbidden).

This is the production collision engine; the point-vs-AABB CollisionChecker
in collision.py is the test-mode proxy and cannot substitute for it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rummagebench.feasibility.fcl_compat import transform3

from rummagebench.feasibility.collision import (
    AllowedCollisionMatrix,
    CollisionPair,
    CollisionResult,
    InteractionCollisionContext,
)
from rummagebench.feasibility.ik_solver import Pose
from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics, _coal
from rummagebench.sim.base import WorldCollisionObject

logger = logging.getLogger(__name__)

# Self-collision pairs that are collider-decomposition artefacts of the
# exported R1Pro (TASK_TIERS_PLAN A.8 F9, user-approved 2026-10-05). Sampled
# over the hardware joint limits they never interpenetrate by more than
# 14 mm (arm_link5<->arm_link7 meet at the wrist-pitch housing: 5.7 mm apart
# at neutral, colliding for every wrist pitch < -0.4 rad, i.e. ~45% of the
# arm's joint space; torso_link4<->arm_link2 <= 1 mm at the shoulder). Deep
# contacts (gripper / wrist camera vs torso: 5-8 cm) remain checked; names
# absent from another robot's model make this a no-op.
DECOMPOSITION_ARTIFACT_SELF_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(p) for side in ("left", "right") for p in (
        (f"{side}_arm_link5", f"{side}_arm_link7"),
        ("torso_link4", f"{side}_arm_link2"),
    )
)


def _coal_object_from_pose(coal, kin, pose: Pose, geometry: Any):
    """coal.CollisionObject for a geometry placed at a base-frame Pose."""
    R = np.asarray(kin.quat_to_rotation_matrix(pose.orientation), dtype=float)
    T = transform3(coal, pose.position, R)
    return coal.CollisionObject(geometry, T)


class PinocchioCollisionChecker:
    """q-based collision engine over a PinocchioKinematics model."""

    def __init__(self, kinematics: PinocchioKinematics, compute_min_distance: bool = True):
        self._kin = kinematics
        self._coal = _coal()
        self._acm = AllowedCollisionMatrix(kinematics.link_classes)
        self._compute_min_distance = bool(compute_min_distance)
        # coal objects reused across queries; geometry cached, transforms set
        # per check_configuration call
        self._link_classes = None
        self._robot_geoms: list[Any] = [go for go in kinematics.geom_model.geometryObjects]
        self._skip_self_pairs = self._self_collision_skip_pairs()
        # bounding-box approximation artefacts: pairs already in contact at
        # the neutral configuration can never be resolved by the arm and are
        # disabled (the standard "default collision matrix" rule), recorded
        self.rest_contact_pairs = self._rest_contact_pairs()
        self._skip_self_pairs |= set(self.rest_contact_pairs)
        self.artifact_pairs = self._artifact_pairs()
        self._skip_self_pairs |= set(self.artifact_pairs)
        # bodies whose pose does not depend on the controlled joints (mobile
        # base, wheels, casters: parentJoint 0 of the reduced model) are not
        # interaction bodies: base placement is validated by the footprint
        # checks of MOVE / TURN / NAV anchors, and they rest on the floor
        self._base_body_index = {
            i for i, go in enumerate(self._robot_geoms) if go.parentJoint == 0}
        self._world: list[tuple[WorldCollisionObject, Any]] = []
        self._world_warned = False
        self._req: Any = None
        self._dreq: Any = None

    # ------------------------------------------------------------- world set

    def refresh_world(self, backend: Any) -> None:
        """(Re)load world collision geometry from the backend.

        Called by the feasibility engine once per interaction check (not per
        IK candidate). Empty world geometry is allowed but warned about once:
        some scenes legitimately have no queryable colliders, yet production
        grounding quality depends on this surface.
        """
        geoms = backend.collision_geometries()
        self._world = [(g, g.geometry) for g in geoms if g.geometry is not None]
        if not self._world and not self._world_warned:
            self._world_warned = True
            logger.warning(
                "backend.collision_geometries() returned no world collision "
                "bodies: robot-world collision checks will only see "
                "self-collision. Check the backend's collision export."
            )

    # ------------------------------------------------------- self-collision

    def _self_collision_skip_pairs(self) -> set[tuple[int, int]]:
        """Geom index pairs excluded from self-collision queries: identical
        joint or parent-child adjacent joints (permanent contact). Distal
        links vs the base remain checked (real self-collision)."""
        pin_model = self._kin.model
        objs = self._kin.geom_model.geometryObjects
        parents = pin_model.parents

        skip: set[tuple[int, int]] = set()
        for i in range(len(objs)):
            for k in range(i + 1, len(objs)):
                ji, jk = objs[i].parentJoint, objs[k].parentJoint
                if ji == jk or parents[ji] == jk or parents[jk] == ji:
                    skip.add((i, k))
        return skip

    def _artifact_pairs(self) -> list[tuple[int, int]]:
        names = self._kin.collision_geometry_names()
        return [(i, k) for i in range(len(names)) for k in range(i + 1, len(names))
                if frozenset((names[i], names[k])) in DECOMPOSITION_ARTIFACT_SELF_PAIRS]

    def _rest_contact_pairs(self) -> list[tuple[int, int]]:
        coal = self._coal
        bodies = self._kin.collision_bodies(self._kin.q_seed_neutral())
        out = []
        for i in range(len(bodies)):
            for k in range(i + 1, len(bodies)):
                if (i, k) in self._skip_self_pairs:
                    continue
                res = coal.CollisionResult()
                coal.collide(bodies[i][1], bodies[k][1], coal.CollisionRequest(), res)
                if res.isCollision():
                    out.append((i, k))
        if out:
            names = sorted({(bodies[i][0], bodies[k][0]) for i, k in out})
            logger.info("disabled %d self-collision pairs in contact at the neutral "
                        "configuration: %s", len(out), names)
        return out

    # ----------------------------------------------------------------- check

    def check_configuration(
        self,
        q: np.ndarray,
        interaction_context: InteractionCollisionContext,
    ) -> CollisionResult:
        """Collision status of one robot configuration.

        ``q`` is the controlled-joint vector (kinematics model order);
        ``interaction_context.base_pose`` places the robot base in the world
        so world geometry can be expressed in the base frame (robot bodies
        stay in base frame).
        """
        if interaction_context.base_pose is None:
            raise ValueError(
                "check_configuration requires interaction_context.base_pose"
            )
        coal = self._coal
        q = np.asarray(q, dtype=float)
        base_pose = interaction_context.base_pose
        robot_bodies = self._kin.collision_bodies(q)  # [(link, CollisionObject)]
        T_base_world = base_pose.inverse()

        result = CollisionResult(
            collision_free=True, self_collision=False, world_collision=False
        )

        # --- self collision: all pairs EXCEPT the adjacency skip set ---
        n_bodies = len(robot_bodies)
        for i in range(n_bodies):
            for k in range(i + 1, n_bodies):
                if (i, k) in self._skip_self_pairs:
                    continue
                req = self._collision_request()
                res = coal.CollisionResult()
                coal.collide(robot_bodies[i][1], robot_bodies[k][1], req, res)
                if res.isCollision():
                    result.collision_free = False
                    result.self_collision = True
                    result.pairs.append(
                        CollisionPair(
                            robot_link=robot_bodies[i][0],
                            other=robot_bodies[k][0],
                            kind="self",
                        )
                    )

        # --- robot vs world ---
        min_distance = float("inf")
        ctx = interaction_context
        # exact AABB prefilter (pairs whose bounding boxes are disjoint cannot
        # collide); disabled when min-distance reporting needs every pair
        overlap = None if self._compute_min_distance else \
            self._aabb_overlaps(robot_bodies, base_pose)
        for wi, (wco, geometry) in enumerate(self._world):
            if wco.entity == ctx.held_entity:
                continue  # travels with the tool; handled as a tool body
            if overlap is not None:
                if not overlap[:, wi].any():
                    continue
            # broadphase: distance from base origin to the body's world AABB
            elif not self._broadphase_ok(wco, T_base_world):
                continue
            for bi, (link, body) in enumerate(robot_bodies):
                if bi in self._base_body_index:
                    continue
                if overlap is not None and not overlap[bi, wi]:
                    continue
                if self._acm.is_allowed(
                    link,
                    wco.entity,
                    ctx.skill,
                    ctx.target_entity,
                    interaction_link=ctx.interaction_link,
                    world_link=wco.link,
                    held_entity=ctx.held_entity,
                ):
                    continue
                req = self._collision_request()
                res = coal.CollisionResult()
                world_obj = self._coal_object(wco, geometry, T_base_world)
                coal.collide(body, world_obj, req, res)
                if res.isCollision():
                    result.collision_free = False
                    result.world_collision = True
                    result.pairs.append(
                        CollisionPair(
                            robot_link=link,
                            other=(
                                wco.entity if wco.link is None else f"{wco.entity}:{wco.link}"
                            ),
                            kind="world",
                        )
                    )
                elif self._compute_min_distance:
                    dreq = self._distance_request()
                    dres = self._coal.DistanceResult()
                    d = coal.distance(body, world_obj, dreq, dres)
                    min_distance = min(min_distance, float(d))

        # --- held object as a tool body (PLACE admissibility etc.) ---
        held_body = self._held_body(q, T_base_world, ctx)
        if held_body is not None:
            for wco, geometry in self._world:
                if wco.entity == ctx.held_entity:
                    continue
                if not self._broadphase_ok(wco, T_base_world):
                    continue
                if self._acm.is_allowed(
                    f"held:{ctx.held_entity}",
                    wco.entity,
                    ctx.skill,
                    ctx.target_entity,
                    interaction_link=ctx.interaction_link,
                    world_link=wco.link,
                    held_entity=ctx.held_entity,
                ):
                    continue
                req = self._collision_request()
                res = coal.CollisionResult()
                world_obj = self._coal_object(wco, geometry, T_base_world)
                coal.collide(held_body, world_obj, req, res)
                if res.isCollision():
                    result.collision_free = False
                    result.world_collision = True
                    result.pairs.append(
                        CollisionPair(
                            robot_link=f"held:{ctx.held_entity}",
                            other=(
                                wco.entity if wco.link is None else f"{wco.entity}:{wco.link}"
                            ),
                            kind="world",
                        )
                    )
                    break

        if min_distance != float("inf"):
            result.min_distance = round(min_distance, 6)
        return result

    # ---------------------------------------------------------------- helpers

    def _world_aabb_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame AABBs of the current world set (unbounded if unknown),
        cached until the next refresh_world."""
        cached = getattr(self, "_world_aabb_cache", None)
        if cached is not None and cached[0] is self._world:
            return cached[1], cached[2]
        n = len(self._world)
        lo = np.full((n, 3), -np.inf)
        hi = np.full((n, 3), np.inf)
        for i, (wco, _) in enumerate(self._world):
            if wco.aabb is not None:
                a, b = np.asarray(wco.aabb[0], float), np.asarray(wco.aabb[1], float)
                if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a <= b):
                    lo[i], hi[i] = a, b
        self._world_aabb_cache = (self._world, lo, hi)
        return lo, hi

    def _aabb_overlaps(self, robot_bodies, base_pose: Pose) -> np.ndarray:
        """(n_robot, n_world) bool: world-frame AABB overlap of every robot
        body (at this configuration) with every world body."""
        w_lo, w_hi = self._world_aabb_arrays()
        R = np.asarray(self._kin.quat_to_rotation_matrix(base_pose.orientation), dtype=float)
        t = np.asarray(base_pose.position, dtype=float)
        pad = float(getattr(self._kin, "collision_padding", 0.0) or 0.0) + 1e-6
        r_lo = np.empty((len(robot_bodies), 3))
        r_hi = np.empty((len(robot_bodies), 3))
        signs = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        for i, (_, body) in enumerate(robot_bodies):
            body.computeAABB()
            box = body.getAABB()
            lo_b, hi_b = np.asarray(box.min_, float), np.asarray(box.max_, float)
            corners = lo_b + signs * (hi_b - lo_b)          # base frame
            world = corners @ R.T + t
            r_lo[i], r_hi[i] = world.min(0) - pad, world.max(0) + pad
        return (np.all(r_lo[:, None, :] <= w_hi[None, :, :], axis=2)
                & np.all(w_lo[None, :, :] <= r_hi[:, None, :], axis=2))

    def _coal_object(self, wco: WorldCollisionObject, geometry: Any, T_base_world):
        return _coal_object_from_pose(self._coal, self._kin, T_base_world.compose(wco.pose), geometry)

    def _held_body(self, q, T_base_world, ctx):
        """Coal body for the held object at its grasp offset from the EEF."""
        if ctx.held_entity is None or ctx.held_offset is None:
            return None
        eef_base = self._kin.eef_pose(q)
        held_base = eef_base.compose(ctx.held_offset)
        geometry = self._held_geometry(ctx.held_entity)
        if geometry is None:
            return None
        return _coal_object_from_pose(self._coal, self._kin, held_base, geometry)

    def _held_geometry(self, entity: str):
        for wco, geometry in self._world:
            if wco.entity == entity:
                return geometry
        return None

    def _broadphase_ok(self, wco: WorldCollisionObject, T_base_world) -> bool:
        aabb = wco.aabb
        if aabb is None:
            return True
        lo, hi = np.asarray(aabb[0], dtype=float), np.asarray(aabb[1], dtype=float)
        center = T_base_world.transform_point((lo + hi) / 2.0)
        radius = float(np.linalg.norm(hi - lo)) / 2.0
        # conservative robot extent: farthest collision-body origin + slack
        robot_reach = getattr(self, "_robot_reach", None)
        if robot_reach is None:
            robot_reach = 0.0
            for go in self._kin.geom_model.geometryObjects:
                robot_reach = max(robot_reach, float(np.linalg.norm(go.placement.translation)))
            robot_reach += 2.0  # link lengths slack
            self._robot_reach = robot_reach
        return float(np.linalg.norm(center)) <= robot_reach + radius

    def _collision_request(self):
        if self._req is None:
            req = self._coal.CollisionRequest()
            if self._kin.collision_padding and hasattr(req, "security_margin"):
                req.security_margin = float(self._kin.collision_padding)
            self._req = req
        return self._req

    def _distance_request(self):
        if self._dreq is None:
            self._dreq = self._coal.DistanceRequest()
        return self._dreq
