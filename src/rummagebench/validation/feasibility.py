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
           -> if collision-free: FEASIBLE(q, interaction target)
    -> all candidates failed:
           UNREACHABLE (no IK anywhere) | COLLISION (IK ok, collision)

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

    # ------------------------------------------------------------------ main

    def check(
        self,
        skill_name: str,
        resolved: ResolvedTarget,
        robot: RobotEmbodiment,
        world,
    ) -> FeasibilityVerdict:
        if skill_name == "NAV":
            # navigation executor is perfect; anchors are verified at build time
            return FeasibilityVerdict(feasible=True)

        assert resolved.entity is not None
        info = resolved.info

        # 1. state preconditions (FAIL_INVALID_STATE)
        invalid = self._state_preconditions(skill_name, resolved, info, robot, world)
        if invalid is not None:
            return invalid

        # 2+3. geometry: interaction interfaces -> IK -> collision
        if self.engine == "pinocchio":
            return self._check_configuration_space(skill_name, resolved, world)
        return self._check_proxy(skill_name, resolved, robot)

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
            from rummagebench.core.errors import FeasibilityBackendError

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
            from rummagebench.core.errors import FeasibilityBackendError

            raise FeasibilityBackendError(
                f"world collision geometry unavailable from backend: {e}"
            ) from e

        base_pose = self._robot_base_pose()
        if not (
            np.all(np.isfinite(base_pose.position))
            and np.all(np.isfinite(base_pose.orientation))
        ):
            # PhysX corruption poisoned the base pose: no finite interaction
            # configuration can be derived from it (structured failure, never
            # NaN into pinocchio/coal — those segfault on NaN inputs)
            return FeasibilityVerdict(
                feasible=False,
                reason="UNREACHABLE",
                details={"engine": "pinocchio", "mode": self._mode,
                         "entity": entity, "note": "non-finite base pose"},
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
            except Exception as e:
                evaluation_errors += 1
                candidates.append(
                    {
                        "target": target.to_dict(),
                        "evaluation_error": str(e),
                    }
                )
                continue
            if collision_result.collision_free:
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
                        "candidates_evaluated": len(candidates) + 1,
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
                }
            )

        # structured failure attribution (§11): collision evidence wins over
        # unreachability; fine-grained IK reasons stay in the details
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
            details["evaluation_errors"] = evaluation_errors
        return FeasibilityVerdict(feasible=False, reason=reason, details=details)

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
