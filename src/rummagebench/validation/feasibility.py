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

        evaluation_errors = 0
        for target in targets:
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
                approach = (self._approach(skill_name, target, ik.q, base_pose, ctx)
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
                            alt_approach = self._approach(skill_name, target, alt.q,
                                                          base_pose, ctx)
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
            if collision_result.collision_free and approach is not None and approach["clear"]:
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

        # structured failure attribution (§11): collision evidence wins over
        # unreachability; fine-grained IK reasons stay in the details
        if evaluation_errors and not saw_ik and not saw_collision:
            # every candidate with a completed evaluation failed cleanly AND
            # the rest never completed: the exceptions prove nothing about
            # reachability (R02) — fail as infrastructure instead of
            # laundering them into an UNREACHABLE proof
            raise FeasibilityBackendError(
                "configuration-space check could not complete a valid "
                f"evaluation for any candidate ({evaluation_errors} evaluation "
                "error(s)); unreachability was NOT established"
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
        }
        if evaluation_errors:
            # positive evidence exists (measured collision or successful IK on
            # other candidates), but the rejection is not a complete proof
            details["evaluation_errors"] = evaluation_errors
            details["incomplete_evaluation"] = True
            details["note"] = (
                "some candidates raised evaluation errors; this verdict rests "
                "on the measured candidates only and is not a proof of "
                "infeasibility"
            )
        return FeasibilityVerdict(feasible=False, reason=reason, details=details)

    def _approach(self, skill_name: str, target, q0, base_pose: Pose, ctx) -> dict:
        """GRASP approach segment from the standoff configuration ``q0`` to
        the contact distance: {'clear': bool, ...evidence}. Other skills and
        candidates without an approach normal are not modelled (clear)."""
        meta = target.metadata or {}
        normal, standoff = meta.get("approach_normal"), meta.get("standoff_m")
        if skill_name != "GRASP" or normal is None or standoff is None:
            return {"clear": True, "modelled": False}
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
