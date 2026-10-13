"""Feasibility validator: the geometry/state gate between an agent decision
and its instant symbolic execution.

Pipeline (production, backend=pinocchio — real kinematics + collision):

    skill candidate
    -> state preconditions                      (INVALID_STATE)
    -> interaction target candidates
       (articulated links / canonical rigid candidates / receptacle regions)
    -> for each candidate:
           IK (URDF kinematics, joint limits, multi-seed)
           -> if no IK: continue
           -> configuration-space collision (self + world, ACM)
           -> GRASP: approach segment, standoff -> contact, collision-free
              at every 1 cm waypoint (the standoff is a pre-grasp pose only)
           -> if collision-free: FEASIBLE(q, interaction target)
    -> OPEN: the fully opened entity must not intrude the robot base
       footprint (+ MOVE's margin): COLLISION otherwise
    -> all candidates failed:
           UNREACHABLE (no IK anywhere) | COLLISION (IK ok, collision)

Remediation R02: a normal "no solution found" stays a structured model-facing
rejection, but numerical/data errors and solver/collision-backend crashes are
INFRASTRUCTURE faults raised as ``FeasibilityBackendError`` — they must never
be laundered into an UNREACHABLE/COLLISION verdict (which would enter the
PVR model-violation numerator). An unreachability verdict additionally
requires every candidate to have completed a valid evaluation: when candidates
raised evaluation exceptions and no candidate produced positive evidence, the
check raises instead of claiming a proof of no solution.

The benchmark-facing reason stays UNREACHABLE/COLLISION/INVALID_STATE; the
fine-grained IK reasons (NO_IK_SOLUTION / JOINT_LIMIT / NUMERICAL_FAILURE)
and per-candidate evidence are preserved in the verdict details and the
event log. NAV is a perfect executor (anchors verified at build time).

The proxy path (backend=proxy: reach-radius solver + point-vs-AABB checker)
is TEST MODE ONLY. It never runs in production sessions: model_loader raises
instead of silently degrading.
"""

from __future__ import annotations

import logging

import numpy as np

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.types import FeasibilityVerdict
from rummagebench.feasibility.interaction_target import PLACE_BATCH_SIZE
from rummagebench.feasibility.collision import CollisionChecker, InteractionCollisionContext
from rummagebench.feasibility.ik_solver import IKSolver, Pose
from rummagebench.object_interface.base import build_object_interface
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import ResolvedTarget, SimBackend

logger = logging.getLogger(__name__)


class FeasibilityValidator:
    """Two interchangeable engines behind one check() interface:

    - config mode: ``kinematics`` (RobotKinematicsBackend, URDF-backed) +
      ``config_collision`` (q-based checker). Production.
    - proxy mode: ``ik_solver`` (reach-radius) + ``collision_checker``
      (point-vs-AABB). Unit tests only.
    """

    # distinct IK solutions checked per interaction candidate before it is
    # declared in collision (existence of ONE collision-free configuration)
    MAX_IK_SOLUTIONS = 8
    # GRASP approach: canonical
    # candidates stand off 6 cm from the object face, which can put the
    # gripper on the far side of a thin barrier (a countertop over a no-top
    # drawer). The gripper must also close the gap to the face: waypoints
    # every APPROACH_STEP_M down to APPROACH_CONTACT_M (EEF-to-face distance
    # at which the R1Pro fingertips reach the face), each with some
    # collision-free IK solution (ACM unchanged: gripper may touch the target)
    APPROACH_STEP_M = 0.01
    APPROACH_CONTACT_M = 0.02
    # OPEN: a drawer must not open INTO the robot base (the base would then
    # overlap it and every later MOVE / TURN be UNSAFE). The opened
    # entity's AABB is tested against the base footprint with the same rule
    # and margin as MOVE / TURN (skills/move.py); VisualProtocolSession sets
    # base_half_extent to its own value.
    base_half_extent = 0.40
    base_margin = 0.03

    def __init__(
        self,
        backend: SimBackend,
        kinematics=None,
        config_collision=None,
        ik_solver: IKSolver | None = None,
        collision_checker: CollisionChecker | None = None,
        mode: str = "endpoint",
    ):
        self._backend = backend
        self._mode = mode
        if kinematics is not None:
            if config_collision is None:
                raise ValueError(
                    "config-mode FeasibilityValidator requires a configuration-"
                    "space collision checker"
                )
            self._kinematics = kinematics
            self._config_collision = config_collision
            self._ik = None
            self._collision = None
            self.engine = "pinocchio"
        elif ik_solver is not None:
            self._kinematics = None
            self._config_collision = None
            self._ik = ik_solver
            self._collision = collision_checker
            self.engine = "proxy"
        else:
            raise ValueError(
                "FeasibilityValidator needs either (kinematics, config_collision) "
                "for production or (ik_solver, collision_checker) for test proxy"
            )

    def with_backend(self, backend: SimBackend) -> "FeasibilityValidator":
        """Same engine (IK solver / collision checker / mode), bound to a
        different backend view. Used by the oracle planner to evaluate
        hypothetical states through the production engine.

        The legacy CollisionChecker holds its own backend reference (for
        open-state queries), so it must be REBOUND, not shared.
        """
        clone = FeasibilityValidator.__new__(FeasibilityValidator)
        clone._backend = backend
        clone._mode = self._mode
        clone._kinematics = self._kinematics
        clone._config_collision = self._config_collision
        clone._ik = self._ik
        if self._collision is not None:
            from rummagebench.feasibility.collision import CollisionChecker

            clone._collision = CollisionChecker(backend)
        else:
            clone._collision = None
        clone.engine = self.engine
        return clone

    # ------------------------------------------------------------------ main

    def check(
        self,
        skill_name: str,
        resolved: ResolvedTarget,
        robot: RobotEmbodiment,
        world,
    ) -> FeasibilityVerdict:
        if skill_name == "NAV":
            # Unchecked legacy anchors may corrupt the world when teleported into.
            if not self._backend.is_anchor_validated(resolved.place, resolved.anchor):
                return FeasibilityVerdict(feasible=False, reason="INVALID_STATE",
                                          details={"reason": "UNVALIDATED_ANCHOR"})
            return FeasibilityVerdict(feasible=True)

        assert resolved.entity is not None
        info = resolved.info

        # 1. state preconditions (FAIL_INVALID_STATE)
        invalid = self._state_preconditions(skill_name, resolved, info, robot, world)
        if invalid is not None:
            return invalid

        # 2+3. geometry: interaction interfaces -> IK -> collision
        if self.engine == "pinocchio":
            verdict = self._check_configuration_space(skill_name, resolved, world)
        else:
            verdict = self._check_proxy(skill_name, resolved, robot)
        # 4. OPEN: the opened articulation must leave the robot base clear
        if skill_name == "OPEN" and verdict.feasible:
            intrusion = self._opened_intrudes_base(resolved.entity)
            if intrusion is not None:
                return FeasibilityVerdict(feasible=False, reason="COLLISION",
                                          details={"entity": resolved.entity,
                                                   "note": "opened links intrude the "
                                                           "robot base footprint",
                                                   **intrusion})
        return verdict

    def _opened_intrudes_base(self, entity: str) -> dict | None:
        """Evidence dict when the FULLY opened entity overlaps the robot base
        footprint (+ margin) while the closed one does not; None otherwise
        (also when the backend cannot predict the opened geometry)."""
        from rummagebench.skills.move import (
            _aabb_overlaps_footprint, footprint_corners, yaw_from_quat)

        opened_aabb = getattr(self._backend, "opened_entity_aabb", None)
        if opened_aabb is None:
            return None
        opened = opened_aabb(entity)
        closed = self._backend.entity_aabb(entity)
        if opened is None:
            return None
        pos, quat = self._backend.robot_pose()
        corners = footprint_corners(pos[0], pos[1], yaw_from_quat(list(quat)),
                                    self.base_half_extent)
        if not _aabb_overlaps_footprint(opened, corners, self.base_margin):
            return None
        if closed is not None and _aabb_overlaps_footprint(closed, corners, self.base_margin):
            return None  # pre-existing contact, not caused by opening
        return {"opened_aabb": [list(map(float, opened[0])), list(map(float, opened[1]))],
                "base_pose": [float(pos[0]), float(pos[1])],
                "base_half_extent": self.base_half_extent, "margin": self.base_margin}

    # ------------------------------------------------------- state prechecks

    def _state_preconditions(
        self, skill_name, resolved, info, robot, world
    ) -> FeasibilityVerdict | None:
        entity = resolved.entity
        if skill_name == "OPEN" and (info is None or not info.openable or self._backend.is_open(entity)):
            return FeasibilityVerdict(
                feasible=False, reason="INVALID_STATE", details={"entity": entity}
            )
        if skill_name == "CLOSE" and (info is None or not info.openable or not self._backend.is_open(entity)):
            return FeasibilityVerdict(
                feasible=False, reason="INVALID_STATE", details={"entity": entity}
            )
        if skill_name == "GRASP":
            if info is None or info.fixed_base or world.held_count >= robot.hand_capacity:
                return FeasibilityVerdict(
                    feasible=False,
                    reason="INVALID_STATE",
                    details={"entity": entity, "held": world.held_count},
                )
        if skill_name == "PLACE" and world.held_count == 0:
            return FeasibilityVerdict(
                feasible=False, reason="INVALID_STATE", details={"hand": "empty"}
            )
        return None

    # ------------------------------------------------------- production path

    def _check_configuration_space(
        self, skill_name: str, resolved: ResolvedTarget, world
    ) -> FeasibilityVerdict:
        entity = resolved.entity
        assert entity is not None
        kinematics = self._kinematics
        collision = self._config_collision

        if skill_name == "PLACE":
            # Missing carry data invalidates the evaluation even if every
            # candidate would otherwise fail IK before the approach stage.
            self._place_carry_data(getattr(world, "held_object", None),
                                   getattr(world, "held_offset", None))

        # interaction interface candidates (handle links / canonical rigid
        # candidates / receptacle regions) — never the entity root pose
        info = resolved.info
        obj = build_object_interface(info, self._backend)
        try:
            targets = obj.interaction_targets(skill_name, world)
        except RuntimeError as e:
            raise FeasibilityBackendError(str(e)) from e
        if not targets:
            return FeasibilityVerdict(
                feasible=False,
                reason="UNREACHABLE",
                details={"entity": entity, "note": "no interaction interface"},
            )

        # world collision cache: refreshed per interaction check (not per IK);
        # a backend that cannot supply world geometry fails loudly instead of
        # silently checking robot-vs-nothing
        try:
            collision.refresh_world(self._backend)
        except Exception as e:
            raise FeasibilityBackendError(
                f"world collision geometry unavailable from backend: {e}"
            ) from e

        base_pose = self._robot_base_pose()
        if not (
            np.all(np.isfinite(base_pose.position))
            and np.all(np.isfinite(base_pose.orientation))
        ):
            # PhysX corruption poisoned the base pose: this is a data error,
            # not a reachability measurement (remediation R02). NaN inputs
            # would segfault pinocchio/coal, and an UNREACHABLE verdict here
            # would fabricate a proof of no solution.
            raise FeasibilityBackendError(
                "configuration-space check aborted: non-finite robot base pose"
            )
        seed_q = self._current_seed_q()
        held_entity = getattr(world, "held_object", None)

        saw_ik = False
        saw_collision = False
        candidates: list[dict] = []

        # round-6 search bookkeeping: the FULL ordered candidate set is
        # walked in batches of PLACE_BATCH_SIZE; every candidate completes a
        # valid evaluation or the verdict becomes an infrastructure fault
        # (R02). No budget is imposed: "search_complete" is therefore always
        # true on a returned verdict, and "the discrete candidate set has no
        # solution" is NEVER worded as a continuous-space infeasibility
        # proof.
        candidates_total = len(targets)
        candidates_checked = 0
        success_index = None
        stop_reason = "candidate_set_exhausted"

        evaluation_errors = 0
        # round-6: walk the FULL ordered candidate set in batches of
        # PLACE_BATCH_SIZE. A batch with no feasible member is followed by
        # the next batch; the batch size is never a total-candidate cap and
        # there is no search budget: a returned verdict always means the
        # complete discrete candidate set was checked (or an infrastructure
        # fault was raised).
        for batch_start in range(0, candidates_total, PLACE_BATCH_SIZE):
            batch = targets[batch_start:batch_start + PLACE_BATCH_SIZE]
            for batch_offset, target in enumerate(batch):
                candidate_index = batch_start + batch_offset
                assert target.pose is not None
                base_target = base_pose.inverse().compose(target.pose)
                if not (
                    np.all(np.isfinite(base_target.position))
                    and np.all(np.isfinite(base_target.orientation))
                ):
                    # backend-supplied target data is corrupted: an evaluation
                    # that never ran, not a rejection (R02)
                    evaluation_errors += 1
                    candidates.append({
                        "target": target.to_dict(),
                        "evaluation_error": "non-finite interaction target",
                    })
                    continue
                # per-candidate isolation: one corrupted object (PhysX execution
                # noise) degrades only itself, never the episode
                try:
                    ik = kinematics.solve_ik(base_target, seed_q=seed_q)
                    if not ik.success:
                        candidates_checked += 1  # a clean, completed evaluation
                        candidates.append(
                            {
                                "target": target.to_dict(),
                                "ik": {
                                    "success": False,
                                    "reason": ik.reason,
                                    "position_error": ik.position_error,
                                    "orientation_error": ik.orientation_error,
                                },
                            }
                        )
                        continue
                    saw_ik = True

                    ctx = InteractionCollisionContext(
                        skill=skill_name,
                        target_entity=entity,
                        interaction_link=target.link,
                        held_entity=held_entity,
                        held_offset=getattr(world, "held_offset", None),
                        base_pose=base_pose,
                    )
                    collision_result = collision.check_configuration(ik.q, ctx)
                    approach = (self._approach(skill_name, target, ik.q, base_pose,
                                               ctx, world)
                                if collision_result.collision_free else None)
                    # existence semantics: the first converging seed is one
                    # configuration of a redundant arm — try further distinct IK
                    # solutions before declaring the candidate in collision
                    n_solutions = 1
                    if not (approach is not None and approach["clear"]) and hasattr(
                            kinematics, "iter_ik_solutions"):
                        for alt in kinematics.iter_ik_solutions(
                                base_target, seed_q=seed_q,
                                max_solutions=self.MAX_IK_SOLUTIONS):
                            if np.max(np.abs(np.asarray(alt.q) - np.asarray(ik.q))) < 0.05:
                                continue
                            n_solutions += 1
                            alt_result = collision.check_configuration(alt.q, ctx)
                            if alt_result.collision_free:
                                alt_approach = self._approach(skill_name, target,
                                                              alt.q, base_pose,
                                                              ctx, world)
                                if alt_approach["clear"] or approach is None:
                                    ik, collision_result, approach = alt, alt_result, alt_approach
                                if alt_approach["clear"]:
                                    break
                except Exception as e:
                    evaluation_errors += 1
                    candidates.append(
                        {
                            "target": target.to_dict(),
                            "evaluation_error": str(e),
                        }
                    )
                    continue
                candidates_checked += 1
                if collision_result.collision_free and approach is not None and approach["clear"]:
                    # existence semantics: ONE fully-measured collision-free
                    # configuration (IK + collision + approach all completed) is
                    # a complete feasibility witness. Later candidates — errored
                    # or not — can never invalidate it (R02 round-2: success is
                    # allowed on a complete witness).
                    stop_reason = "witness_found"
                    success_index = candidate_index
                    return FeasibilityVerdict(
                        feasible=True,
                        reason="none",
                        details={
                            "engine": "pinocchio",
                            "mode": self._mode,
                            "q": [float(v) for v in ik.q],
                            "joints": list(kinematics.controlled_joint_names),
                            "interaction_target": target.to_dict(),
                            "ik": {
                                "position_error": ik.position_error,
                                "orientation_error": ik.orientation_error,
                            },
                            "collision": collision_result.to_dict(),
                            "approach": approach,
                            "candidates_evaluated": len(candidates) + 1,
                            "ik_solutions_checked": n_solutions,
                            "search": {
                                "candidates_total": candidates_total,
                                "candidates_checked": candidates_checked,
                                "search_complete": True,
                                "stop_reason": stop_reason,
                                "success_candidate_index": success_index,
                            },
                        },
                    )
                saw_collision = True
                candidates.append(
                    {
                        "target": target.to_dict(),
                        "ik": {
                            "success": True,
                            "position_error": ik.position_error,
                            "orientation_error": ik.orientation_error,
                        },
                        "collision": collision_result.to_dict(),
                        "approach": approach,
                        "ik_solutions_checked": n_solutions,
                    }
                )

        # R02 round-2: a REJECTION verdict may only rest on candidates that
        # ALL completed a valid evaluation. The feasible-witness path above
        # already returned; here any incomplete candidate means the evidence
        # is incomplete — an infrastructure fault, never laundered into
        # UNREACHABLE/COLLISION as an action failure.
        if evaluation_errors:
            first_error = next(
                (c.get("evaluation_error") for c in candidates
                 if c.get("evaluation_error")), "unknown")
            raise FeasibilityBackendError(
                f"{evaluation_errors} of {candidates_total} interaction "
                f"candidate(s) did not complete a valid evaluation "
                f"({candidates_checked} checked; solver/collision backend "
                f"errors; first: {first_error}); no rejection verdict can "
                "be returned from incomplete evidence"
            )
        if saw_collision:
            reason = "COLLISION"
        elif not saw_ik:
            reason = "UNREACHABLE"
        else:  # defensive: IK ok everywhere but nothing reported collision-free
            reason = "UNREACHABLE"
        details = {
            "engine": "pinocchio",
            "mode": self._mode,
            "entity": entity,
            "candidates": candidates,
            "candidates_evaluated": len(candidates),
            "search": {
                "candidates_total": candidates_total,
                "candidates_checked": candidates_checked,
                "search_complete": True,
                "stop_reason": stop_reason,
                "success_candidate_index": success_index,
                # wording guard: the DISCRETE candidate set has no feasible
                # member — this is not a continuous-space infeasibility proof
                "note": "full discrete candidate set checked; no feasible member",
            },
        }
        return FeasibilityVerdict(feasible=False, reason=reason, details=details)

    def _approach(self, skill_name: str, target, q0, base_pose: Pose, ctx,
                  world=None) -> dict:
        """Approach-segment checks per skill.

        GRASP: standoff-to-contact descent along the grasp normal (existing).
        PLACE (round-6): the validated virtual carry-descent — EEF from the
        standoff pose to the release pose with the held body in the
        collision context (see _place_approach). Other skills / candidates
        without approach metadata are NOT modelled: {'modelled': False} with
        an explicit note — an unmodelled check is recorded, never evidence
        of safety."""
        meta = target.metadata or {}
        normal, standoff = meta.get("approach_normal"), meta.get("standoff_m")
        if skill_name == "GRASP":
            if normal is None or standoff is None:
                return {"clear": True, "modelled": False}
            return self._grasp_approach(target, meta, q0, base_pose, ctx,
                                        normal, standoff)
        if skill_name == "PLACE":
            return self._place_approach(target, meta, q0, base_pose, ctx,
                                        world)
        return {"clear": True, "modelled": False,
                "note": f"approach not modelled for {skill_name}"}

    def _grasp_approach(self, target, meta, q0, base_pose, ctx,
                        normal, standoff) -> dict:
        kinematics, collision = self._kinematics, self._config_collision
        n = np.asarray(normal, dtype=float)
        q = np.asarray(q0, dtype=float)
        dists = np.arange(float(standoff) - self.APPROACH_STEP_M,
                          self.APPROACH_CONTACT_M - 1e-9, -self.APPROACH_STEP_M)
        for d in dists:
            pose = Pose(np.asarray(target.pose.position, dtype=float) - n * (float(standoff) - d),
                        target.pose.orientation)
            base_target = base_pose.inverse().compose(pose)
            found = None
            last = None
            ik = kinematics.solve_ik(base_target, seed_q=q)
            tried = [ik] if ik.success else []
            if ik.success:
                last = collision.check_configuration(ik.q, ctx)
                if last.collision_free:
                    found = ik
            if found is None and hasattr(kinematics, "iter_ik_solutions"):
                for alt in kinematics.iter_ik_solutions(
                        base_target, seed_q=q, max_solutions=self.MAX_IK_SOLUTIONS):
                    if any(np.max(np.abs(np.asarray(alt.q) - np.asarray(t.q))) < 0.05
                           for t in tried):
                        continue
                    tried.append(alt)
                    last = collision.check_configuration(alt.q, ctx)
                    if last.collision_free:
                        found = alt
                        break
            if found is None:
                return {"clear": False, "modelled": True, "blocked_at_m": round(float(d), 4),
                        "reason": "COLLISION" if last is not None else "NO_IK",
                        "collision": None if last is None else last.to_dict()}
            q = np.asarray(found.q, dtype=float)
        return {"clear": True, "modelled": True, "waypoints": int(len(dists))}

    # PLACE approach segment (round-6)
    PLACE_APPROACH_STEP_M = 0.01  # linear waypoint resolution (m)
    PLACE_APPROACH_ANGLE_STEP_RAD = np.deg2rad(5.0)

    def _place_carry_data(self, held_entity, held_offset):
        """Require finite carry transforms and geometry before production IK."""
        if held_entity is None or held_offset is None:
            raise FeasibilityBackendError("PLACE requires held entity and held offset")
        if not (np.all(np.isfinite(held_offset.position))
                and np.all(np.isfinite(held_offset.orientation))):
            raise FeasibilityBackendError("PLACE held offset is non-finite")
        box = self._backend.entity_aabb(held_entity)
        if box is None:
            raise FeasibilityBackendError("PLACE held-object AABB is unavailable")
        bounds = np.asarray(box, dtype=float)
        if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
                or np.any(bounds[1] <= bounds[0])):
            raise FeasibilityBackendError("PLACE held-object AABB is invalid")
        return held_offset, float((bounds[1, 2] - bounds[0, 2]) / 2.0)

    # matches sim/omnigibson/backend.symbolic_place: the object ORIGIN is
    # set to support_z + half_height + 1 cm before release
    PLACE_RELEASE_CLEARANCE_M = 0.01

    def _place_approach(self, target, meta, q0, base_pose: Pose, ctx,
                        world) -> dict:
        """PLACE approach segment: the VIRTUAL carry-descent that the
        symbolic execution stands for.

        Production chain (verified): the interaction target's EEF pose sits
        ``STANDOFF_M`` above/below the validated ``surface_point`` along the
        region approach normal; ``symbolic_place`` teleports the held
        object's ORIGIN to ``[px, py, pz + half_z + 0.01]`` and releases.
        The checked path is therefore the straight-line EEF segment from the
        validated standoff pose to the release pose
        ``obj_release . compose(held_offset.inverse())`` — with the held
        body carried in the collision context via its real grasp offset.
        Every waypoint requires a real IK solution and a collision-free
        configuration of the SAME production checker used for the endpoint.
        Waypoints are discrete (resolution below): this evidences the
        sampled descent only, never a continuous sweep or the whole-arm
        motion. Position is interpolated linearly and orientation follows
        the shortest quaternion arc, with both linear and angular sampling
        limits. Missing carry data is an infrastructure fault, never an
        identity-offset substitute for the production check.
        """
        surface_point = meta.get("surface_point")
        if surface_point is None:
            return {"clear": False, "modelled": True,
                    "reason": "no_surface_point"}
        held_entity = getattr(ctx, "held_entity", None)
        held_offset, half_z = self._place_carry_data(
            held_entity, getattr(ctx, "held_offset", None))
        px, py, pz = (float(v) for v in surface_point)
        release_common = {
            "waypoint_resolution_m": self.PLACE_APPROACH_STEP_M,
            "angular_resolution_rad": float(self.PLACE_APPROACH_ANGLE_STEP_RAD),
            "held_geometry": True,
        }
        obj_release = Pose(np.array([px, py, pz + half_z + self.PLACE_RELEASE_CLEARANCE_M]),
                           np.array([0.0, 0.0, 0.0, 1.0]))
        eef_release = obj_release.compose(held_offset.inverse())
        start = np.asarray(target.pose.position, dtype=float)
        end = np.asarray(eef_release.position, dtype=float)
        segment = end - start
        length = float(np.linalg.norm(segment))
        start_quat = np.asarray(target.pose.orientation, dtype=float)
        end_quat = np.asarray(eef_release.orientation, dtype=float)
        dot = float(np.dot(start_quat, end_quat))
        if dot < 0.0:  # q and -q encode the same rotation; take the short arc
            end_quat = -end_quat
            dot = -dot
        theta = float(np.arccos(np.clip(dot, 0.0, 1.0)))
        n_steps = max(1, int(np.ceil(length / self.PLACE_APPROACH_STEP_M)),
                      int(np.ceil(2.0 * theta / self.PLACE_APPROACH_ANGLE_STEP_RAD)))
        kinematics, collision = self._kinematics, self._config_collision
        q = np.asarray(q0, dtype=float)
        for i in range(1, n_steps + 1):
            t = i / n_steps
            if theta < 1e-8:
                orientation = (1.0 - t) * start_quat + t * end_quat
            else:
                orientation = (np.sin((1.0 - t) * theta) * start_quat
                               + np.sin(t * theta) * end_quat) / np.sin(theta)
            waypoint = Pose(start + segment * t, orientation)
            base_target = base_pose.inverse().compose(waypoint)
            ik = kinematics.solve_ik(base_target, seed_q=q)
            if not ik.success or ik.q is None:
                return {"clear": False, "modelled": True, "reason": "NO_IK",
                        "blocked_at_m": round(length * t, 4),
                        "segment_m": round(length, 4),
                        "waypoints": n_steps, **release_common}
            last = collision.check_configuration(ik.q, ctx)
            if not last.collision_free:
                return {"clear": False, "modelled": True, "reason": "COLLISION",
                        "blocked_at_m": round(length * t, 4),
                        "segment_m": round(length, 4),
                        "waypoints": n_steps,
                        "collision": last.to_dict(), **release_common}
            q = np.asarray(ik.q, dtype=float)
        return {"clear": True, "modelled": True, "segment_m": round(length, 4),
                "waypoints": n_steps,
                "path": ("linear-position, shortest-arc-orientation virtual EEF path to the release "
                         "pose; symbolic execution teleports the object "
                         "(no continuous-sweep claim)"),
                **release_common}

    def _robot_base_pose(self) -> Pose:
        pos, quat = self._backend.robot_pose()
        return Pose.from_lists(pos, quat)

    def _current_seed_q(self) -> np.ndarray | None:
        """Current robot joint state, mapped into kinematics model order."""
        positions = self._backend.robot_joint_positions()
        if not positions:
            return None
        kinematics = self._kinematics
        neutral = kinematics.q_seed_neutral()
        seed = np.zeros(kinematics.nq)
        used = False
        for i, name in enumerate(kinematics.controlled_joint_names):
            if name in positions:
                seed[i] = positions[name]
                used = True
            else:
                seed[i] = neutral[i]
        return seed if used else None

    # ----------------------------------------------------------- proxy path

    def _check_proxy(
        self, skill_name: str, resolved: ResolvedTarget, robot: RobotEmbodiment
    ) -> FeasibilityVerdict:
        """TEST MODE ONLY: reach-radius + point-vs-AABB on the entity pose."""
        entity = resolved.entity
        pose = self._backend.entity_pose(entity)
        reach = self._reach_proxy(pose, robot)
        if not reach.feasible:
            return reach
        assert self._collision is not None
        return self._collision.check(entity, pose)

    def _reach_proxy(self, pose: list[float], robot: RobotEmbodiment) -> FeasibilityVerdict:
        if self._ik is None:
            return FeasibilityVerdict(feasible=True)
        q = self._ik.solve(pose, self._backend.robot_pose(), robot)
        if q is None:
            return FeasibilityVerdict(
                feasible=False, reason="UNREACHABLE", details={"pose": pose}
            )
        return FeasibilityVerdict(feasible=True, details=q)
