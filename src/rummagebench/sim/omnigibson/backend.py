"""OmniGibson implementation of the abstract SimBackend.

Everything OmniGibson-specific lives here. Benchmark logic (skills,
validators, session) depends only on rummagebench.sim.base.SimBackend.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from rummagebench.core.errors import SimBackendError
from rummagebench.core.scenario import AnchorSpec, ScenarioSpec
from rummagebench.core.types import TargetKind
from rummagebench.sim.base import ResolvedTarget, SimBackend, WorldCollisionObject
from rummagebench.sim.omnigibson.env_factory import apply_runtime_env, build_env
from rummagebench.sim.omnigibson.entity_resolver import build_entity_info, resolve_object
from rummagebench.sim.omnigibson.observation import capture_head_rgb
from rummagebench.sim.omnigibson.state_io import (
    dump_state,
    load_state,
    seed_everything,
)

logger = logging.getLogger(__name__)


class OmniGibsonBackend(SimBackend):
    def __init__(self, seed: int = 0, settle_steps: int = 15, spawn_stopped: bool = True, validate_legacy_anchors: bool = False):
        self._seed = seed
        self._settle_steps = settle_steps
        self._spawn_stopped = spawn_stopped
        self._validate_legacy_anchors = validate_legacy_anchors
        self._env = None
        self._og = None
        self._sim = None
        self._robot = None
        self._initial_state: Any = None
        self._validated_anchor_poses = {}
        self._entity_infos: dict[str, Any] = {}
        self._build_report: dict[str, Any] | None = None
        self._commanded_pose: tuple[list[float], list[float]] | None = None
        # world collision export caches: geometry per collider prim (static),
        # body list until a state transition moves objects
        self._geom_cache: dict[str, Any] = {}
        self._collision_body_cache: list[WorldCollisionObject] | None = None

    # ------------------------------------------------------------------ setup

    def setup(self, scenario: ScenarioSpec) -> dict[str, Any]:
        self._validated_anchor_poses.clear()
        seed_everything(self._seed)
        apply_runtime_env()

        import omnigibson as og
        from omnigibson.objects import DatasetObject

        self._og = og
        self._env, _ = build_env(scenario)
        self._sim = og.sim
        if not self._sim.is_playing():
            self._sim.play()
        self._robot = self._env.scene.robots[0]
        self._scenario = scenario

        report: dict[str, Any] = {"scene": scenario.scene.model, "spawned": [], "placements": [], "initial_states": [], "anchors": []}

        # 1. spawn scenario objects
        from rummagebench.sim.omnigibson.dataset import pick_model_for_category

        # Import all task objects before resuming simulation. Repeated live
        # imports invalidate PhysX views while renderer semantic graphs are active.
        # The live-import sequence reproduced graph crashes on this host.
        if self._spawn_stopped:
            self._sim.stop()
        for spec in scenario.objects:
            model = spec.model or pick_model_for_category(spec.category)
            obj = DatasetObject(
                name=spec.name,
                category=spec.category,
                model=model,
                fixed_base=spec.fixed_base,
            )
            # add first (loads the prim), then park it above its receptacle
            self._env.scene.add_object(obj)
            seed_pos = self._seed_pose_for(spec)
            obj.set_position_orientation(
                position=seed_pos,
                orientation=[0, 0, 0, 1],
            )
            report["spawned"].append({"name": spec.name, "category": spec.category, "model": model})
            if not self._spawn_stopped:
                self._og.sim.step()
        if self._spawn_stopped:
            self._sim.play()
            self._sim.step()

        # 2. placements via semantic relations
        from omnigibson.object_states import Inside, OnTop, Open

        for p in scenario.placements:
            obj = resolve_object(self._env.scene, p.entity)
            rec = resolve_object(self._env.scene, p.receptacle)
            ok = self._place(obj, rec, p.relation, Open)
            verified = self._verify_relation(obj, rec, p.relation, Inside, OnTop)
            report["placements"].append(
                {
                    "entity": p.entity,
                    "relation": p.relation,
                    "receptacle": p.receptacle,
                    "placed": bool(ok),
                    "verified": bool(verified),
                }
            )
            if not verified:
                raise SimBackendError(
                    f"placement failed verification: {p.entity} {p.relation} {p.receptacle}"
                )

        # 3. forced initial states (containers closed etc.)
        for entity, st in scenario.initial_states.items():
            obj = resolve_object(self._env.scene, entity)
            if st.open is not None:
                if obj.states.get(Open) is None:
                    raise SimBackendError(f"{entity} has no Open state")
                obj.states[Open].set_value(st.open, fully=True)
                report["initial_states"].append({"entity": entity, "open": st.open})

        self.settle()

        # AGENT uses only the initial spawn; visiting legacy ORACLE navigation
        # anchors here can collide with furniture and poison the saved world.
        # Keep legacy diagnostics explicit, and never label untested anchors valid.
        init_anchor = scenario.anchors[scenario.robot.init_anchor]
        anchors = (scenario.anchors if self._validate_legacy_anchors
                   else {scenario.robot.init_anchor: init_anchor})
        report["anchor_validation_scope"] = "all_legacy" if self._validate_legacy_anchors else "initial_spawn"
        report["untested_anchors"] = [name for name in scenario.anchors if name not in anchors]
        for name, anchor in anchors.items():
            self.teleport_robot(anchor)
            self.settle(5)
            self.validate_physics_state()
            pos, _ = self.robot_pose()
            dist = float(np.linalg.norm(np.asarray(pos) - np.asarray(anchor.position)))
            report["anchors"].append({"anchor": name, "distance_to_anchor": dist})
            if dist > 0.5:
                raise SimBackendError(
                    f"anchor {name!r} verification failed: robot ended {dist:.2f}m away"
                )
        self.teleport_robot(init_anchor)
        self.settle()

        self.validate_physics_state()

        self._validate_spawn_clearance()

        # 5. capture deterministic snapshot
        self._initial_state = dump_state(self._sim)
        self._build_report = report
        self._validated_anchor_poses = {name: (tuple(anchor.position), tuple(anchor.orientation))
                                        for name, anchor in anchors.items()}
        return report

    def _seed_pose_for(self, spec) -> list[float]:
        """Spawn seed pose: above the receptacle if placed, else above scene center."""
        placement = next(
            (p for p in self._scenario.placements if p.entity == spec.name), None
        )
        base = [0.0, 0.0, 1.0]
        if placement is not None:
            try:
                rec = resolve_object(self._env.scene, placement.receptacle)
                pos, _ = rec.get_position_orientation()
                base = [pos[0], pos[1], pos[2] + 0.8]
            except Exception:
                pass
        return base

    def _place(self, obj, rec, relation: str, Open) -> bool:
        """Place obj into/on rec via the state sampler; returns sampler success.

        Closed containers can block fillable sampling, so for `inside` we
        temporarily open an openable receptacle, sample, and let the caller's
        initial_states restore the closed state afterwards. The sampler is
        probabilistic, so we retry several times and fall back to direct pose
        placement inside the receptacle's AABB.
        """
        from omnigibson.object_states import Inside, OnTop

        was_open = None
        if relation == "inside" and rec.states.get(Open) is not None:
            was_open = rec.states[Open].get_value()
            if not was_open:
                rec.states[Open].set_value(True, fully=True)

        state = Inside if relation == "inside" else OnTop
        ok = False
        if state in obj.states:
            for i in range(5):
                ok = bool(
                    obj.states[state].set_value(
                        rec, True, reset_before_sampling=i > 0
                    )
                )
                if ok and bool(obj.states[state].get_value(rec)):
                    break
                ok = False

        # deterministic fallback: park the object at the receptacle center and
        # let physics settle it into the container volume
        if not ok and relation == "inside":
            try:
                rec_pos = rec.get_position_orientation()[0]
                if hasattr(rec_pos, "detach"):
                    rec_pos = rec_pos.detach().cpu().numpy()
                aabb = rec.states.get(Inside)
                obj.set_position_orientation(
                    position=[float(rec_pos[0]), float(rec_pos[1]), float(rec_pos[2]) + 0.05],
                    orientation=[0, 0, 0, 1],
                )
                self.settle(10)
                ok = bool(obj.states[Inside].get_value(rec))
            except Exception as e:
                logger.warning("pose fallback placement failed for %s: %s", obj.name, e)

        if was_open is False and rec.states.get(Open) is not None:
            rec.states[Open].set_value(False, fully=True)
        return ok

    def _verify_relation(self, obj, rec, relation: str, Inside, OnTop) -> bool:
        state = Inside if relation == "inside" else OnTop
        if state not in obj.states:
            return False
        # Inside/OnTop are binary (relative) states: get_value(other)
        return bool(obj.states[state].get_value(rec))

    # ------------------------------------------------------------- episode IO

    def reapply_scenario(self, scenario: ScenarioSpec) -> dict:
        """Re-apply a generated episode's placements / initial states on the
        ALREADY built scene (same object set, different relations), then
        re-capture the deterministic snapshot.

        This is the cheap path that makes a 30-episode distribution runnable
        in one simulator process: no scene reload, no new Kit launch. The
        physics sampler is re-seeded per call so the same (seed, episode)
        pair reproduces the same placement.
        """
        from omnigibson.object_states import Open

        seed_everything(int(scenario.id.encode().hex()[-6:], 16) % (2**31))
        report: dict[str, Any] = {"placements": [], "initial_states": []}
        from omnigibson.object_states import Inside, OnTop

        # start from the clean snapshot (releases lingering assisted grasps)
        self.reset()
        for p in scenario.placements:
            obj = resolve_object(self._env.scene, p.entity)
            rec = resolve_object(self._env.scene, p.receptacle)
            verified = False
            for attempt in range(3):
                ok = self._place(obj, rec, p.relation, Open)
                verified = self._verify_relation(obj, rec, p.relation, Inside, OnTop)
                if verified:
                    break
            report["placements"].append({
                "entity": p.entity,
                "relation": p.relation,
                "receptacle": p.receptacle,
                "placed": bool(ok),
                "verified": bool(verified),
            })
            if not verified:
                raise SimBackendError(
                    f"reapply placement failed verification: {p.entity} "
                    f"{p.relation} {p.receptacle}"
                )
        for entity, st in scenario.initial_states.items():
            obj = resolve_object(self._env.scene, entity)
            if st.open is not None:
                if obj.states.get(Open) is None:
                    raise SimBackendError(f"{entity} has no Open state")
                obj.states[Open].set_value(st.open, fully=True)
                report["initial_states"].append({"entity": entity, "open": st.open})
        self.settle()
        # re-capture the deterministic snapshot for THIS episode
        self._collision_body_cache = None
        self._initial_state = dump_state(self._sim)
        return report

    def reset(self) -> None:
        assert self._initial_state is not None, "backend.setup() has not run"
        self._collision_body_cache = None
        # release any assisted grasp first: the snapshot predates it, and a
        # lingering joint would yank the held object out of its container
        ag = getattr(self._robot, "_ag_obj_in_hand", {})
        for arm in list(getattr(self._robot, "arm_names", []) or []):
            if ag.get(arm) is not None:
                try:
                    self._robot._release_grasp(arm=arm)
                except Exception as e:
                    logger.warning("release_grasp(%s) failed on reset: %s", arm, e)
        load_state(self._sim, self._initial_state)
        # re-assert the base pose explicitly: the snapshot restores joints, but
        # re-teleporting removes any residual base-controller drift. Settle for
        # the same duration as setup so both paths converge to the same pose.
        self.teleport_robot(self._scenario.anchors[self._scenario.robot.init_anchor])
        self.settle(15)
        self.validate_physics_state()
        self._validate_spawn_clearance()

    def _validate_spawn_clearance(self) -> None:
        from rummagebench.skills.move import base_pose_collision_free, yaw_from_quat
        position, orientation = self.robot_pose()
        if not base_pose_collision_free(self, position[0], position[1],
                                        yaw_from_quat(orientation), 0.40, 0.03):
            raise SimBackendError("Initial robot footprint overlaps an obstacle; invalid spawn")

    def get_observation(self) -> np.ndarray:
        return capture_head_rgb(self._env, self._robot.name)

    def resolve_entity(self, name: str) -> ResolvedTarget:
        obj = resolve_object(self._env.scene, name)
        info = build_entity_info(obj)
        self._entity_infos[name] = info
        return ResolvedTarget(kind=TargetKind.ENTITY, entity=name, info=info)

    # ------------------------------------------------------------------ NAV

    def is_anchor_validated(self, name: str, anchor) -> bool:
        return (anchor is not None and self._validated_anchor_poses.get(name)
                == (tuple(anchor.position), tuple(anchor.orientation)))

    def teleport_robot(self, anchor: AnchorSpec) -> None:
        # Never hide a corrupt articulation behind a commanded pose.
        position = np.asarray(anchor.position, dtype=float)
        orientation = np.asarray(anchor.orientation, dtype=float)
        self._robot.set_position_orientation(
            position=position, orientation=orientation, frame="world"
        )
        self._commanded_pose = (
            [float(v) for v in anchor.position],
            [float(v) for v in anchor.orientation],
        )

    def robot_pose(self) -> tuple[list[float], list[float]]:
        try:
            pos, quat = self._robot.get_position_orientation(frame="world")
        except (AssertionError, ValueError) as error:
            raise SimBackendError("Robot physics pose is invalid; run is invalid") from error
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        quat = quat.detach().cpu().numpy() if hasattr(quat, "detach") else np.asarray(quat)
        if (pos.shape != (3,) or quat.shape != (4,)
                or not np.isfinite(pos).all() or not np.isfinite(quat).all()
                or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3)):
            raise SimBackendError("Robot physics pose is nonfinite or invalid; run is invalid")
        return pos.astype(float).tolist(), quat.astype(float).tolist()

    # ---------------------------------------------------------- OPEN / GRASP

    def set_open(self, entity: str, open_value: bool) -> bool:
        from omnigibson.object_states import Open

        self._collision_body_cache = None  # articulated links moved
        obj = resolve_object(self._env.scene, entity)
        if obj.states.get(Open) is None:
            return False
        return bool(obj.states[Open].set_value(open_value, fully=True))

    def is_open(self, entity: str) -> bool:
        from omnigibson.object_states import Open

        obj = resolve_object(self._env.scene, entity)
        if obj.states.get(Open) is None:
            return False
        return bool(obj.states[Open].get_value())

    def symbolic_grasp(self, entity: str) -> bool:
        """Assisted-grasp realization for the (already feasibility-validated)
        grasp. Realization ONLY: it never moves the robot base (implicit NAV
        is forbidden) and never defines benchmark holding state — the
        benchmark-owned state is the semantic truth. A long-range assisted
        joint may fail physically; that is visualization noise, logged and
        ignored by the benchmark."""
        obj = resolve_object(self._env.scene, entity)
        arm = self._pick_arm(obj)
        if arm is None:
            return False
        robot = self._robot
        link_name = next(iter(obj.links.keys()), None) if obj.links else None
        if link_name is None:
            return False
        try:
            joint_type = robot._get_assisted_grasp_joint_type(obj, link_name) or "FixedJoint"
            # keep the torch tensor: _establish_grasp goes through torch-compiled
            # code paths and a numpy array breaks Dynamo tracing
            contact_pos = obj.get_position_orientation()[0]
            if not hasattr(contact_pos, "detach"):
                import torch as th

                contact_pos = th.as_tensor(contact_pos, dtype=th.float32)
            robot._establish_grasp(
                target_obj=obj,
                target_link_name=link_name,
                arm=arm,
                contact_pos_world=contact_pos,
                joint_type=joint_type,
            )
            self.settle(3)
            if self.is_holding(entity):
                return True
            # retry on the other arm / the base link
            other_arms = [a for a in robot.arm_names if a != arm]
            for retry_arm in other_arms:
                robot._establish_grasp(
                    target_obj=obj,
                    target_link_name=link_name,
                    arm=retry_arm,
                    contact_pos_world=contact_pos,
                    joint_type=joint_type,
                )
                self.settle(3)
                if self.is_holding(entity):
                    return True
        except Exception as e:
            logger.warning("symbolic_grasp(%s) realization failed: %s", entity, e)
        return self.is_holding(entity)

    def _pick_arm(self, obj) -> str | None:
        robot = self._robot
        arms = list(getattr(robot, "arm_names", []) or [])
        if not arms:
            return None
        # prefer the arm whose end-effector is closer to the object
        pos = obj.get_position_orientation()[0]
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        best, best_dist = None, float("inf")
        for arm in arms:
            try:
                eef = robot.eef_links[arm]
                eef_pos = eef.get_position_orientation()[0]
                if hasattr(eef_pos, "detach"):
                    eef_pos = eef_pos.detach().cpu().numpy()
                d = float(np.linalg.norm(np.asarray(pos) - np.asarray(eef_pos)))
                if d < best_dist:
                    best, best_dist = arm, d
            except Exception:
                continue
        return best if best is not None else arms[0]

    def is_holding(self, entity: str) -> bool:
        try:
            obj = resolve_object(self._env.scene, entity)
        except Exception:
            return False
        robot = self._robot
        ag = getattr(robot, "_ag_obj_in_hand", {})
        for arm in getattr(robot, "arm_names", []):
            if ag.get(arm) is obj:
                return True
        return False

    # ------------------------------------------------------------ simulation

    def settle(self, steps: int | None = None) -> None:
        n = steps if steps is not None else self._settle_steps
        for _ in range(n):
            self._sim.step()

    def validate_physics_state(self) -> None:
        """Fail an invalid run before corrupt physics reaches rendering/scoring."""
        def finite(value):
            value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            return bool(np.isfinite(value).all())
        try:
            for obj in self._env.scene.objects:
                position, quaternion = obj.get_position_orientation()
                if not finite(position) or not finite(quaternion):
                    raise ValueError("nonfinite object pose")
                if obj.n_joints and not finite(obj.get_joint_positions()):
                    raise ValueError("nonfinite joint positions")
                for link in obj.links.values():
                    position, quaternion = link.get_position_orientation()
                    if not finite(position) or not finite(quaternion):
                        raise ValueError("nonfinite link pose")
        except (AssertionError, ValueError) as error:
            raise SimBackendError("Physics state is invalid; run must not be scored") from error

    def dump_state(self) -> Any:
        return dump_state(self._sim)

    def load_state(self, state: Any) -> None:
        load_state(self._sim, state)

    def capture_observe_state(self):
        """Full state for counterfactual multi-view rendering (evaluator-only)."""
        from copy import deepcopy

        return {"sim": self.dump_state(), "commanded_pose": deepcopy(self._commanded_pose)}

    def restore_observe_state(self, snapshot):
        """Restore without physics steps that would mutate the saved world."""
        from copy import deepcopy

        load_state(self._sim, snapshot["sim"], settle_steps=0)
        self._commanded_pose = deepcopy(snapshot["commanded_pose"])
        self._collision_body_cache = None

    def render_snapshot(self, path: str) -> None:
        from PIL import Image

        rgb = self.get_observation()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)

    def build_report(self) -> dict[str, Any] | None:
        return self._build_report

    # ------------------------------------------------- embodiment grounding

    def robot_entity_names(self) -> set[str]:
        return {self._robot.name}

    def entity_names(self) -> list[str]:
        return [o.name for o in self._env.scene.objects]

    def entity_pose(self, name: str) -> list[float]:
        obj = resolve_object(self._env.scene, name)
        pos, _ = obj.get_position_orientation()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        return [float(v) for v in np.asarray(pos).reshape(-1)[:3]]

    def entity_aabb(self, name: str) -> tuple[list[float], list[float]] | None:
        obj = resolve_object(self._env.scene, name)
        try:
            from omnigibson.object_states import AABB

            if obj.states.get(AABB) is None:
                return None
            lo, hi = obj.states[AABB].get_value()
            if hasattr(lo, "detach"):
                lo = lo.detach().cpu().numpy()
                hi = hi.detach().cpu().numpy()
            return (
                [float(v) for v in np.asarray(lo).reshape(-1)[:3]],
                [float(v) for v in np.asarray(hi).reshape(-1)[:3]],
            )
        except Exception as e:
            logger.warning("entity_aabb(%s) unavailable: %s", name, e)
            return None

    def held_count(self) -> int:
        ag = getattr(self._robot, "_ag_obj_in_hand", {})
        return sum(1 for v in ag.values() if v is not None)

    def symbolic_place(self, entity: str, receptacle: str) -> bool:
        """Release the benchmark-held ``entity`` onto/into ``receptacle``.

        The held object is passed explicitly by the benchmark core (never
        re-derived from an internal grasp dict); this is realization only.
        """
        obj = resolve_object(self._env.scene, entity)
        rec = resolve_object(self._env.scene, receptacle)
        self._collision_body_cache = None  # the placed object moves
        # release whichever assisted-grasp joint holds this exact object
        ag = getattr(self._robot, "_ag_obj_in_hand", {})
        for arm in getattr(self._robot, "arm_names", []):
            if ag.get(arm) is obj:
                try:
                    self._robot._release_grasp(arm=arm)
                except Exception as e:
                    logger.warning("release on place failed for %s: %s", entity, e)
                break
        pos, _ = rec.get_position_orientation()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        try:
            obj.set_position_orientation(
                position=[float(pos[0]), float(pos[1]), float(pos[2]) + 0.15],
                orientation=[0, 0, 0, 1],
            )
        except Exception as e:
            logger.warning("place pose write failed for %s: %s", entity, e)
            return False
        self.settle(10)
        return True

    def describe_entity(self, name: str) -> EntityInfo | None:
        if name not in self._entity_infos:
            try:
                obj = resolve_object(self._env.scene, name)
            except Exception:
                return None
            self._entity_infos[name] = build_entity_info(obj)
        return self._entity_infos[name]

    # ------------------------------------------------- physical grounding API
    # Surfaces the geometry/kinematics data the pinocchio feasibility engine
    # needs. Approximation levels are recorded on every body (never silent).

    def articulation_info(self, entity: str):
        """Articulation decomposition from the USD physics schema."""
        from rummagebench.sim.base import ArticulationInfo, ArticulationJointInfo
        from rummagebench.sim.omnigibson.robot_export import collect_articulation

        obj = resolve_object(self._env.scene, entity)
        try:
            art = collect_articulation(obj.prim)
        except Exception as e:
            logger.warning("articulation_info(%s) failed: %s", entity, e)
            return None
        joints = [
            ArticulationJointInfo(
                name=j["name"],
                joint_type=j["type"],
                parent_link=j["parent"],
                child_link=j["child"],
                axis=j["axis"],
                limits=j["limits"],
                position=None,
            )
            for j in art["joints"]
        ]
        return ArticulationInfo(
            entity=entity,
            joints=joints,
            links=list(art["links"]),
            handle_link=art["handle_link"],
        )

    def collision_geometries(self) -> list[WorldCollisionObject]:
        """World collision bodies, link-granular where the asset has them.

        Cached until a state transition moves objects (OPEN/CLOSE, PLACE,
        reset). Geometry fidelity per body is recorded (approximation level):
        mesh colliders are exported as exact triangle-mesh BVHs when the
        triangle budget allows, else as world-aligned AABB boxes. The robot
        itself is excluded (its collision geometry comes from the URDF).
        """
        if self._env is None:
            return []
        if self._collision_body_cache is not None:
            return self._collision_body_cache
        robot_name = getattr(self._robot, "name", None)
        out: list[WorldCollisionObject] = []
        from rummagebench.sim.omnigibson.usd_collision import (
            collect_entity_collision,
        )

        for obj in self._env.scene.objects:
            if robot_name is not None and obj.name == robot_name:
                continue
            try:
                out.extend(collect_entity_collision(obj, self._geom_cache))
            except Exception as e:
                logger.warning(
                    "collision geometry export failed for %s: %s", obj.name, e
                )
        self._collision_body_cache = out
        return out

    def receptacle_region(self, entity: str):
        """Support region: top surface; inside volume for open containers."""
        from rummagebench.feasibility.interaction_target import InteractionRegion

        aabb = self.entity_aabb(entity)
        if aabb is None:
            return None
        lo, hi = aabb
        if self.is_open(entity):
            return InteractionRegion(lo=lo, hi=hi, kind="inside_volume")
        top = list(lo)
        top[2] = hi[2]
        return InteractionRegion(lo=lo, hi=[hi[0], hi[1], hi[2]], kind="top_surface")

    def link_pose(self, entity: str, link: str):
        from rummagebench.feasibility.ik_solver import Pose

        obj = resolve_object(self._env.scene, entity)
        link_prim = (obj.links or {}).get(link)
        if link_prim is None:
            return None
        try:
            pos, quat = link_prim.get_position_orientation(frame="world")
        except TypeError:
            pos, quat = link_prim.get_position_orientation()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        if hasattr(quat, "detach"):
            quat = quat.detach().cpu().numpy()
        pos = np.asarray(pos).reshape(-1)[:3]
        quat = np.asarray(quat).reshape(-1)[:4]
        # PhysX teleport corruption can yield NaN link poses: degrade this
        # single query to None instead of poisoning the whole grounding pass
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return None
        return Pose.from_lists([float(v) for v in pos], [float(v) for v in quat])

    def link_aabb(self, entity: str, link: str):
        from rummagebench.sim.omnigibson.usd_collision import compute_link_aabb

        obj = resolve_object(self._env.scene, entity)
        link_prim = (obj.links or {}).get(link)
        if link_prim is None:
            return None
        return compute_link_aabb(link_prim)

    def eef_pose(self):
        from rummagebench.feasibility.ik_solver import Pose

        arms = list(getattr(self._robot, "arm_names", []) or [])
        if not arms:
            return None
        arm = arms[0]
        eef = getattr(self._robot, "eef_links", {}).get(arm)
        if eef is None:
            return None
        try:
            pos, quat = eef.get_position_orientation(frame="world")
        except TypeError:
            pos, quat = eef.get_position_orientation()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        if hasattr(quat, "detach"):
            quat = quat.detach().cpu().numpy()
        return Pose.from_lists(
            [float(v) for v in np.asarray(pos).reshape(-1)[:3]],
            [float(v) for v in np.asarray(quat).reshape(-1)[:4]],
        )

    def entity_pose6d(self, entity: str):
        from rummagebench.feasibility.ik_solver import Pose

        obj = resolve_object(self._env.scene, entity)
        pos, quat = obj.get_position_orientation()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        if hasattr(quat, "detach"):
            quat = quat.detach().cpu().numpy()
        pos = np.asarray(pos).reshape(-1)[:3]
        quat = np.asarray(quat).reshape(-1)[:4]
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return None
        return Pose.from_lists([float(v) for v in pos], [float(v) for v in quat])

    def visible_entities(self) -> list[str]:
        """Objects observable by the agent right now (§20): everything except
        the contents of CLOSED openable containers. Scene furniture and
        open-container contents are visible. Oracle knowledge (evaluation,
        traces) keeps using entity_names(); this method is the agent-facing
        view only."""
        from omnigibson.object_states import Inside, Open

        scene_objects = self._env.scene.objects
        containers = [
            o for o in scene_objects
            if getattr(o, "fixed_base", False) and o.states.get(Open) is not None
        ]
        closed = [
            c for c in containers if not bool(c.states[Open].get_value())
        ]
        visible: list[str] = []
        for obj in scene_objects:
            if obj.fixed_base:
                visible.append(obj.name)  # furniture is part of the room
                continue
            hidden = False
            for container in closed:
                try:
                    inside = obj.states.get(Inside)
                    if inside is not None and bool(inside.get_value(container)):
                        hidden = True
                        break
                except Exception:
                    continue  # corrupted state during PhysX noise: skip query
            if not hidden:
                visible.append(obj.name)
        return visible

    def robot_joint_positions(self) -> dict[str, float] | None:
        """Current joint positions (URDF joint names -> radians/meters),
        used as the IK seed. USD joint names must match the exported URDF."""
        try:
            joints = getattr(self._robot, "joints", None) or {}
            positions: dict[str, float] = {}
            for jname, jprim in joints.items():
                try:
                    state = jprim.get_state()
                    positions[jname] = float(state[0]) if state is not None else 0.0
                except Exception:
                    continue
            return positions or None
        except Exception as e:
            logger.warning("robot_joint_positions unavailable: %s", e)
            return None

    def export_kinematics_urdf(self, out_path: str) -> dict:
        """Export this robot's articulation to URDF (+ manifest + FK
        cross-validation). Called automatically when a scenario configures
        kinematics whose urdf_path does not exist yet."""
        from rummagebench.sim.omnigibson.robot_export import export_robot_urdf

        return export_robot_urdf(self._robot, out_path)

    def close(self) -> None:
        try:
            if self._og is not None:
                self._og.shutdown()
        except Exception:
            pass
        self._env = None

    # ================================================== visual protocol (§7)

    def capture_visual_frame(self):
        """Capture one camera's synchronized modalities and immutable metadata.

        OG RAW_SENSOR_TYPES: depth=distance_to_camera (range),
        depth_linear=distance_to_image_plane (Z). Source:
        https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/sensors/vision_sensor.py
        Runtime geometry is obtained from that same sensor. The convention is
        API-derived, NOT an assertion that hardware/simulator calibration passed.
        RGB-only frames remain useful for diagnostics, but are not valid AGENT
        interaction frames: collider rays alone cannot certify RGB visibility.
        """
        from rummagebench.perception.frame_store import VisualFramePrivate
        from rummagebench.perception.camera_geometry import world_from_usd_camera

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        obs_list, infos = self._env.get_obs()
        robot_obs = obs_list[0].get(self._robot.name, {})
        from rummagebench.sim.omnigibson.observation import select_head_rgb_sensor

        sensor_name, data = select_head_rgb_sensor(robot_obs)
        sensor = getattr(self._robot, "sensors", {}).get(sensor_name)
        K = None
        try:
            # OG lazily attaches camera_params and renders on first access.
            # Finish that work BEFORE fetching any buffers for the frame.
            from rummagebench.sim.omnigibson.raw_instance import wait_for_intrinsics

            K = array(wait_for_intrinsics(sensor, self._sim)).astype(float)
        except Exception:
            pass
        if self._sim is not None:
            # The first render after a counterfactual pose restore can still
            # return the previous camera buffer. Drain that frame without
            # stepping physics, then collect all modalities together.
            self._sim.render()
            self._sim.render()
        obs_list, infos = self._env.get_obs()
        data = obs_list[0].get(self._robot.name, {}).get(sensor_name)
        if not isinstance(data, dict) or data.get("rgb") is None:
            raise RuntimeError("visual protocol: selected RGB sensor disappeared")
        rgb = array(data["rgb"])[..., :3].astype(np.uint8).copy()
        H, W = rgb.shape[:2]
        missing = []
        depth_key = "depth_linear" if data.get("depth_linear") is not None else "depth"
        convention = "z_depth" if depth_key == "depth_linear" else "euclidean_range"
        depth = array(data[depth_key]).astype(float) if data.get(depth_key) is not None else None
        if depth is not None and depth.shape == (H, W, 1):
            depth = depth[..., 0]
        if depth is None or depth.shape != (H, W):
            depth = np.full((H, W), np.nan)
            missing.append("same-camera depth")
        seg = array(data["seg_instance"]) if data.get("seg_instance") is not None else None
        if seg is not None and seg.shape == (H, W, 1):
            seg = seg[..., 0]
        if seg is None or seg.shape != (H, W):
            seg = np.zeros((H, W), dtype=np.int64)
            missing.append("same-camera instance segmentation")

        # get_obs returns per-environment, per-robot, per-sensor info in OG.
        env_info = infos[0] if isinstance(infos, (list, tuple)) and infos else infos
        sensor_info = (env_info.get(self._robot.name, {}).get(sensor_name, {})
                       if isinstance(env_info, dict) else {})
        mapping = sensor_info.get("seg_instance", {})
        labels = {str(key): str(value.get("class", "") if isinstance(value, dict) else value)
                  for key, value in mapping.items()} if isinstance(mapping, dict) else {}
        if not labels:
            missing.append("same-camera instance labels")
        self._instance_labels = dict(labels)  # legacy diagnostics only

        if K is None or K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            K = np.full((3, 3), np.nan)
            missing.append("sensor intrinsics")
        pose = self._camera_pose(sensor_name)
        try:
            T = world_from_usd_camera(*pose) if pose is not None else None
        except ValueError:
            T = None
        if T is None:
            T = np.full((4, 4), np.nan)
            missing.append("sensor world pose")
        return VisualFramePrivate(
            frame_id="", rgb=rgb, depth=depth.copy(), instance_segmentation=seg.copy(),
            camera_intrinsics=K.copy(), camera_extrinsics=T,
            image_width=W, image_height=H, depth_convention=convention,
            meta={"bridge": "segmentation" if not missing else "unsupported",
                  "sensor_name": sensor_name, "instance_labels": dict(labels),
                  "visual_grounding_supported": not missing,
                  "unsupported_reason": ", ".join(missing),
                  "depth_convention_source": "OmniGibson modality API",
                  "calibration_verified": False},
        )

    def instance_to_entity(self, instance, frame=None) -> str | None:
        labels = (frame.meta.get("instance_labels", {}) if frame is not None
                  else getattr(self, "_instance_labels", {}))
        label = labels.get(str(instance.item() if hasattr(instance, "item") else instance))
        if not label or label.lower() in ("background", "unlabelled", "groundplane"):
            return None
        try:
            if label in self.entity_names() or label in self._entity_infos:
                return label
        except Exception:
            pass
        return None

    def entity_visible_pixels(self, frame, entity: str) -> int:
        """Count all instance IDs of an entity in THIS immutable camera frame."""
        seg = np.asarray(frame.instance_segmentation)
        labels = frame.meta.get("instance_labels", {})
        targets = [key for key, label in labels.items() if label == entity]
        total = 0
        for target in targets:
            try:
                key = int(target) if seg.dtype.kind not in "USO" else target
                total += int(np.count_nonzero(seg == key))
            except (TypeError, ValueError):
                continue
        return total

    # -------------------------------------------- rgb-only raycast fallback

    def _camera_prim(self, sensor_name: str | None = None):
        """The VisionSensor's USD camera prim (for intrinsics/extrinsics)."""
        try:
            sensors = ([self._robot.sensors[sensor_name]]
                       if sensor_name and sensor_name in self._robot.sensors
                       else list(self._robot.sensors.values()))
            for sensor in sensors:
                pp = getattr(sensor, "prim_path", None)
                if not pp:
                    continue
                import omnigibson.lazy as lazy

                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(pp)
                if prim is not None:
                    return prim
        except Exception:
            pass
        return None

    def _camera_pose(self, sensor_name: str | None = None):
        """World (position, quat xyzw) of the rgb sensor, via the sensor
        object's own XFormPrim API — the same path the robot pose uses."""
        try:
            sensors = ([self._robot.sensors[sensor_name]]
                       if sensor_name and sensor_name in self._robot.sensors
                       else list(self._robot.sensors.values()))
            for sensor in sensors:
                if hasattr(sensor, "get_position_orientation"):
                    pos, quat = sensor.get_position_orientation()
                    if hasattr(pos, "detach"):
                        pos = pos.detach().cpu().numpy()
                    if hasattr(quat, "detach"):
                        quat = quat.detach().cpu().numpy()
                    return (np.asarray(pos, dtype=float)[:3],
                            np.asarray(quat, dtype=float))
        except Exception:
            pass
        return None

    def _camera_intrinsics_from_prim(self, prim, width: int, height: int):
        """Pinhole K from the USD camera focalLength / horizontalAperture
        (both mm); falls back to the 90-deg HFOV default."""
        import numpy as np
        from rummagebench.perception.camera_geometry import default_intrinsics

        if prim is None:
            return default_intrinsics(width, height)
        try:
            focal = prim.GetAttribute("focalLength").Get()
            aperture = prim.GetAttribute("horizontalAperture").Get()
            if not focal or not aperture:
                return default_intrinsics(width, height)
            fx = width * float(focal) / float(aperture)
            fy = fx  # square pixels; horizontal aperture determines pixel pitch
            cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        except Exception:
            return default_intrinsics(width, height)

    def pixel_ray_hit(self, frame, x: float, y: float):
        """Evaluator diagnostic ONLY (not accepted by the AGENT boundary): resolve a
        normalized image point through a PhysX raytest instead of the
        instance-segmentation annotator (which segfaults this Isaac build).

        Returns (entity_name | None, hit_position_world | None). The hit
        collision prim is canonicalized to the OWNING OmniGibson object by
        prim-path prefix, exactly mirroring §8.3."""
        from rummagebench.perception.camera_geometry import (
            normalized_to_pixel, unproject_z_depth,
        )

        T = np.asarray(frame.camera_extrinsics)
        K = np.asarray(frame.camera_intrinsics)
        u, v = normalized_to_pixel(x, y, frame.image_width, frame.image_height)
        dir_cam = unproject_z_depth(u, v, 1.0, K)
        dir_cam = dir_cam / np.linalg.norm(dir_cam)
        origin = T[:3, 3]
        direction = T[:3, :3] @ dir_cam
        end = origin + direction * 20.0

        from omnigibson.utils.sampling_utils import raytest

        # the camera rides on the robot: never let the ray hit the robot's
        # own body (self-occlusion of the mount is not scene occlusion)
        ignore = [self._robot.prim_path] + [
            link.prim_path for link in getattr(self._robot, "links", {}).values()
        ]
        hit = raytest(start_point=[float(v) for v in origin],
                      end_point=[float(v) for v in end],
                      ignore_bodies=ignore)
        if not hit or not hit.get("hit"):
            return None, None

        rigid_path = hit.get("rigidBody") or hit.get("collision") or ""
        obj = self._owning_object(rigid_path)
        if obj is None:
            return None, None
        pos = hit.get("position")
        return obj.name, ([float(v) for v in pos] if pos is not None else None)

    def _owning_object(self, prim_path: str):
        """Walk the prim-path ancestry to the registered OmniGibson object."""
        if not prim_path:
            return None
        try:
            scene = self._env.scene
            obj = scene.object_registry("prim_path", prim_path)
            if obj is not None:
                return obj
            parts = prim_path.split("/")
            for i in range(len(parts) - 1, 0, -1):
                candidate = "/".join(parts[:i])
                if not candidate:
                    continue
                obj = scene.object_registry("prim_path", candidate)
                if obj is not None:
                    return obj
        except Exception:
            return None
        return None


    def _camera_prim_extrinsics(self, cam_prim):
        """T_world_camera 4x4 from the camera prim's WORLD pose, or None.

        The raw xformOp on the prim is its LOCAL pose relative to the robot
        — the world pose must come from the OmniGibson XFormPrim API."""
        if cam_prim is None:
            return None
        try:
            import numpy as _np
            import omnigibson.lazy as lazy

            pos, quat = lazy.omni.isaac.core.utils.xforms.get_world_pose(
                str(cam_prim.GetPath())
            )
            quat = _np.asarray(quat)  # omni convention wxyz
            T = _np.eye(4)
            T[:3, 3] = _np.asarray(pos, dtype=float)[:3]
            from scipy.spatial.transform import Rotation as R

            T[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            return T
        except Exception:
            return None
        return None
