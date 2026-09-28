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
    def __init__(self, seed: int = 0, settle_steps: int = 15):
        self._seed = seed
        self._settle_steps = settle_steps
        self._env = None
        self._og = None
        self._sim = None
        self._robot = None
        self._initial_state: Any = None
        self._entity_infos: dict[str, Any] = {}
        self._build_report: dict[str, Any] | None = None
        self._commanded_pose: tuple[list[float], list[float]] | None = None
        # world collision export caches: geometry per collider prim (static),
        # body list until a state transition moves objects
        self._geom_cache: dict[str, Any] = {}
        self._collision_body_cache: list[WorldCollisionObject] | None = None

    # ------------------------------------------------------------------ setup

    def setup(self, scenario: ScenarioSpec) -> dict[str, Any]:
        seed_everything(self._seed)
        apply_runtime_env()

        import omnigibson as og
        from omnigibson.objects import DatasetObject

        from rummagebench.sim.omnigibson.env_factory import apply_sim_settings

        apply_sim_settings()
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
            self._og.sim.step()

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

        # 4. verify all anchors are reachable poses, then return to init anchor
        init_anchor = scenario.anchors[scenario.robot.init_anchor]
        for name, anchor in scenario.anchors.items():
            self.teleport_robot(anchor)
            self.settle(5)
            pos, _ = self.robot_pose()
            dist = float(np.linalg.norm(np.asarray(pos) - np.asarray(anchor.position)))
            report["anchors"].append({"anchor": name, "distance_to_anchor": dist})
            if dist > 0.5:
                raise SimBackendError(
                    f"anchor {name!r} verification failed: robot ended {dist:.2f}m away"
                )
        self.teleport_robot(init_anchor)
        self.settle()

        # 5. capture deterministic snapshot
        self._initial_state = dump_state(self._sim)
        self._build_report = report
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

    def get_observation(self) -> np.ndarray:
        return capture_head_rgb(self._env, self._robot.name)

    def resolve_entity(self, name: str) -> ResolvedTarget:
        obj = resolve_object(self._env.scene, name)
        info = build_entity_info(obj)
        self._entity_infos[name] = info
        return ResolvedTarget(kind=TargetKind.ENTITY, entity=name, info=info)

    # ------------------------------------------------------------------ NAV

    def teleport_robot(self, anchor: AnchorSpec) -> None:
        # NOTE: do not zero velocities / call keep_still() here. Both were
        # tried and both destabilize the position-controlled suspension
        # (keep_still zeroes joint effort targets -> chassis collapses).
        # Teleport is a pure semantic jump; residual physics noise is handled
        # by robot_pose()'s NaN fallback below, not by touching the physics.
        position = np.asarray(anchor.position, dtype=float)
        orientation = np.asarray(anchor.orientation, dtype=float)
        try:
            self._robot.set_position_orientation(
                position=position, orientation=orientation, frame="world"
            )
        except (AssertionError, ValueError):
            # physics already NaN-corrupted: the Robot-level override reads
            # EEF link poses (and asserts on NaN) to preserve arm poses. Fall
            # back to the EntityPrim-level setter, which writes the root pose
            # without reading link states.
            from omnigibson.prims.entity_prim import EntityPrim

            EntityPrim.set_position_orientation(
                self._robot, position=position, orientation=orientation, frame="world"
            )
        self._commanded_pose = (
            [float(v) for v in anchor.position],
            [float(v) for v in anchor.orientation],
        )

    def robot_pose(self) -> tuple[list[float], list[float]]:
        try:
            pos, quat = self._robot.get_position_orientation(frame="world")
        except (AssertionError, ValueError):
            # physics diverged (NaN base orientation after PhysX broadphase
            # corruption); OmniGibson asserts before returning. The semantic
            # truth is the last commanded anchor pose — a perfect executor
            # would be there. Execution noise must not kill the episode.
            commanded = getattr(self, "_commanded_pose", None)
            if commanded is not None:
                return [list(commanded[0]), list(commanded[1])]
            raise
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        quat = quat.detach().cpu().numpy() if hasattr(quat, "detach") else np.asarray(quat)
        pos = [float(v) for v in pos]
        quat = [float(v) for v in quat]
        if not (all(math.isfinite(v) for v in pos) and all(math.isfinite(v) for v in quat)):
            commanded = getattr(self, "_commanded_pose", None)
            if commanded is not None:
                return [list(commanded[0]), list(commanded[1])]
        return pos, quat

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

    def dump_state(self) -> Any:
        return dump_state(self._sim)

    def load_state(self, state: Any) -> None:
        load_state(self._sim, state)

    def render_snapshot(self, path: str) -> None:
        from PIL import Image

        rgb = self.get_observation()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)

    def build_report(self) -> dict[str, Any] | None:
        return self._build_report

    # ------------------------------------------------- embodiment grounding

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
