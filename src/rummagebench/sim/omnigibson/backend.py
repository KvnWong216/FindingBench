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
from rummagebench.sim.omnigibson.inside_volume import defer_authoring_rendering
from rummagebench.sim.omnigibson.state_io import (
    dump_state,
    load_state,
    seed_everything,
)

logger = logging.getLogger(__name__)


def declared_open_abilities(initial_state, category_abilities):
    """Request native Open for declared doors omitted by the taxonomy.

    OmniGibson still checks the real asset and its articulated joints before
    constructing Open; this does not manufacture joints or change predicates.
    Never mutate the shared knowledge-base ability dictionary.
    """
    if initial_state is None or initial_state.open is None or 'openable' in category_abilities:
        return None
    from copy import deepcopy
    return dict(deepcopy(category_abilities), openable={})


def _arr(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return [float(v) for v in np.asarray(value).reshape(-1)]


# symbolic grasp realization: objects whose largest native half extent
# exceeds LONG_OBJECT_HALF_M are held ahead of the fingertips (see
# _bring_to_hand); HELD_FORWARD_AUTO selects that rule, 0.0 = legacy
LONG_OBJECT_HALF_M = 0.08
HELD_FORWARD_AUTO = -1.0

# the furniture body's top (or a countertop resting on it) overhangs the
# body AABB by a few cm; the reachable strip of an open drawer starts beyond
DRAWER_OVERHANG_MARGIN_M = 0.03


def _with_open_strip(region, body_aabb):
    """Mark the XY strip of an open drawer cavity that lies outside the
    furniture body (open from above) on the region (reach_lo / reach_hi)."""
    lo, hi = list(region.lo), list(region.hi)
    blo, bhi = body_aabb
    m = DRAWER_OVERHANG_MARGIN_M
    sides = [(blo[0] - lo[0], 0, -1), (hi[0] - bhi[0], 0, +1),
             (blo[1] - lo[1], 1, -1), (hi[1] - bhi[1], 1, +1)]
    protrusion, axis, sign = max(sides)
    if protrusion <= m:
        return region
    rlo, rhi = [lo[0], lo[1]], [hi[0], hi[1]]
    if sign < 0:
        rhi[axis] = float(blo[axis]) - m
    else:
        rlo[axis] = float(bhi[axis]) + m
    region.reach_lo, region.reach_hi = rlo, rhi
    region.open_dir = [float(sign), 0.0] if axis == 0 else [0.0, float(sign)]
    return region


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

    def setup(self, scenario: ScenarioSpec, *, native_scene_capture=None,
              native_scene_only: bool = False, task_seed_pose=None,
              pre_placement_capture=None) -> dict[str, Any]:
        if native_scene_only and native_scene_capture is None:
            raise ValueError("native-only setup requires an occupancy capture")
        frozen_path = scenario.scene.frozen_member_manifest
        frozen_doc = None
        if frozen_path is not None:
            if native_scene_capture is not None or pre_placement_capture is not None or task_seed_pose is not None:
                raise ValueError("frozen member restore cannot be combined with authoring hooks")
            from rummagebench.authoring.tasks.household_state import load_frozen_member
            frozen_doc = load_frozen_member(Path(frozen_path), scenario)
        import sys as _s
        self._validated_anchor_poses.clear()
        seed_everything(self._seed)
        apply_runtime_env()

        if getattr(getattr(scenario, 'robot', None), 'kinematics', None) is not None:
            # Register Pinocchio's Boost.Python containers before Kit loads
            # its own bindings. Use the environment's C++ runtime first:
            # torch's compiled kernels require newer symbols than the host.
            import ctypes
            cpp_runtime = Path(_s.prefix) / 'lib' / 'libstdc++.so.6'
            if cpp_runtime.exists():
                ctypes.CDLL(str(cpp_runtime), mode=ctypes.RTLD_GLOBAL)
            import pinocchio  # noqa: F401

        from rummagebench.sim.omnigibson.inside_volume import configure_isaac_thread_limit
        configure_isaac_thread_limit()
        import omnigibson as og
        from omnigibson.objects import DatasetObject

        self._og = og
        import os
        if os.environ.get('YGBENCH_MESH_ACCELERATION')=='1':
            from rummagebench.sim.omnigibson.inside_volume import install_mesh_acceleration
            install_mesh_acceleration()
        import time as _setup_time
        _setup_start=_setup_time.perf_counter()
        self._env, _ = build_env(scenario)
        if os.environ.get('YGBENCH_MESH_ACCELERATION')=='1':
            import json as _setup_json
            from omnigibson.utils import usd_utils as _usd_utils
            print('[mesh-acceleration-stats] '+_setup_json.dumps(dict(
                environment_seconds=_setup_time.perf_counter()-_setup_start,
                **getattr(_usd_utils,'_findingbench_mesh_cache_stats',{}))),flush=True)
        self._sim = og.sim
        if not self._sim.is_playing():
            self._sim.play()
        self._robot = self._env.scene.robots[0]
        self._scenario = scenario
        self._clearance_report = None
        if scenario.scene.clearance_manifest and scenario.behavior_task is not None:
            self._clearance_report = self._apply_declared_clearance(scenario,
                preserve_recipe_identity=frozen_doc is not None)

        if getattr(scenario, "behavior_task", None) is not None:
            self.validate_physics_state()
            self._validate_spawn_clearance()
            self._initial_state = dump_state(self._sim)
            task = self._env.task
            goals = task.activity_natural_language_goal_conditions
            # Prepared tasks may supply a human-readable instruction for the
            # same official goal. Legacy placeholders use the official text.
            if not scenario.instruction.strip() or scenario.instruction.startswith("Official BEHAVIOR goal:"):
                scenario.instruction = "\n".join(goals) if isinstance(goals, (list, tuple)) else str(goals)
            if not scenario.instruction.strip() or scenario.instruction == "None":
                raise SimBackendError("official task instruction unavailable")
            self._build_report = {"source": "official_behavior_presampled", "task": scenario.behavior_task.model_dump()}
            if scenario.scene.task_scene_manifest:
                self._build_report['source']='derived_behavior_clearance_scene'
                self._build_report['derivation_manifest']=scenario.scene.task_scene_manifest
            if self._clearance_report:
                self._build_report['clearance'] = self._clearance_report
                self._build_report['source'] = 'derived_behavior_clearance_scene'
            return self._build_report

        report: dict[str, Any] = {"scene": scenario.scene.model, "spawned": [], "placements": [], "initial_states": [], "anchors": []}

        if self._clearance_report:
            report['clearance'] = self._clearance_report
            report['source'] = 'derived_behavior_clearance_scene'

        # Evaluator-only authoring hook on the very same loaded scene used
        # below. Capture BEFORE importing task objects or changing doors.
        # The hook must only read state; it never enters agent observations.
        if native_scene_capture is not None:
            report["native_occupancy"] = native_scene_capture(self)
        if native_scene_only:
            self._initial_state = dump_state(self._sim)
            report["stage"] = "native_loaded"
            self._build_report = report
            return report

        # 1. spawn scenario objects
        from rummagebench.sim.omnigibson.dataset import pick_model_for_category

        # Import all task objects before resuming simulation. Repeated live
        # imports invalidate PhysX views while renderer semantic graphs are active.
        # The live-import sequence reproduced graph crashes on this host.
        print("[setup] stopping", file=_s.stdout, flush=True)
        if self._spawn_stopped:
            self._sim.stop()
        print("[setup] stopped", file=_s.stdout, flush=True)
        # Isolate pending imports during authoring. They regain normal gravity
        # before their native placement; final snapshots contain no staged body.
        self._isolated_authoring_staging = frozen_doc is None and task_seed_pose is None
        self._staged_authoring_names = ({p.entity for p in scenario.placements}
                                        if self._isolated_authoring_staging else set())
        for import_index, spec in enumerate(scenario.objects):
            print(f"[setup] import {spec.category}::{spec.model}",
                  file=_s.stdout, flush=True)
            model = spec.model or pick_model_for_category(spec.category)
            abilities = None
            initial_state = scenario.initial_states.get(spec.name)
            if initial_state is not None and initial_state.open is not None:
                from omnigibson.utils.bddl_utils import get_knowledge_base
                category_info = get_knowledge_base().get_category(spec.category)
                defaults = (category_info.synset.abilities
                            if category_info is not None and category_info.synset is not None else {})
                abilities = declared_open_abilities(initial_state, defaults)
            obj = DatasetObject(
                name=spec.name,
                category=spec.category,
                model=model,
                fixed_base=spec.fixed_base,
                abilities=abilities,
            )
            # Load first, preserving explicit evaluator parking hooks.
            self._env.scene.add_object(obj)
            seed_pos = (frozen_doc["parking"] if frozen_doc is not None else
                        self._seed_pose_for(spec) if task_seed_pose is None else task_seed_pose(self, spec))
            if np.asarray(seed_pos).shape != (3,) or not np.isfinite(seed_pos).all():
                raise ValueError("invalid evaluator task seed pose")
            obj.set_position_orientation(
                position=seed_pos,
                orientation=[0, 0, 0, 1],
            )
            if spec.name in self._staged_authoring_names:
                obj.set_position_orientation(position=[100.0+10.0*import_index,100.0,5.0],
                                             orientation=[0,0,0,1])
                obj.disable_gravity()
            report["spawned"].append({"name": spec.name, "category": spec.category, "model": model})
            if not self._spawn_stopped:
                self._og.sim.step()
        if self._spawn_stopped:
            report["collision_mesh_repairs"] = self._repair_place_collision_meshes()
            self._sim.play()
            self._sim.step()

        if scenario.scene.clearance_manifest:
            # Apply after the legacy object-import stop/play cycle, before
            # placements sample the moved receptacles.
            self._clearance_report = self._apply_declared_clearance(scenario,
                preserve_recipe_identity=frozen_doc is not None)
            report['clearance'] = self._clearance_report
            report['source'] = 'derived_behavior_clearance_scene'

        # Native EntityPrim initialization resets these thresholds. Configure
        # relevant dynamic task links afterwards, including frozen reloads, so
        # valid supporting contacts remain available to native predicates.
        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
        relevant_names = {spec.name for spec in scenario.objects}
        relevant_names.update(p.receptacle for p in scenario.placements)
        relevant_names.update(scenario.initial_states)
        for object_name in sorted(relevant_names):
            task_object = self._env.scene.object_registry('name', object_name)
            if task_object.kinematic_only:
                continue
            if not task_object.initialized:
                raise SimBackendError('TASK_BODY_NOT_INITIALIZED_BEFORE_SLEEP_CONFIGURATION')
            task_object.sleep_threshold = 0.0
            for link in task_object.links.values():
                if isinstance(link, RigidDynamicPrim):
                    link.sleep_threshold = 0.0
            if float(task_object.sleep_threshold) != 0.0:
                raise SimBackendError('NATIVE_TASK_SLEEP_CONFIGURATION_NOT_APPLIED')
            task_object.wake()
        report['native_task_sleep_threshold'] = 0.0


        # Render products belong to scene setup. Adding them after loading a
        # settled checkpoint can invalidate native sleeping-contact state.
        # Build the fixed panorama before either sampling or frozen restore.
        if scenario.robot.panorama is not None:
            from rummagebench.sim.omnigibson.panorama import PanoramaCamera
            self._panorama_camera = PanoramaCamera(self, scenario.robot.panorama)

        if pre_placement_capture is not None:
            # The complete object topology now exists. An authoring caller
            # can freeze background + parked target before target sampling.
            report["frozen_background"] = pre_placement_capture(self)
        if frozen_doc is None:
            report.update(self._apply_placements(scenario, attempts=3))
        else:
            from rummagebench.authoring.tasks.household_state import restore_frozen_member
            restore_frozen_member(self, Path(frozen_path), scenario)
            report.update(frozen_doc["builder"])
            report["frozen_member_manifest_sha256"] = frozen_doc["manifest_sha256"]

        # AGENT uses only the initial spawn; visiting legacy ORACLE navigation
        # anchors here can collide with furniture and poison the saved world.
        # Keep legacy diagnostics explicit, and never label untested anchors valid.
        print("[setup] anchors", file=_s.stdout, flush=True)
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

        if frozen_doc is not None:
            # Anchor diagnostics must not alter the state being certified.
            restore_frozen_member(self, Path(frozen_path), scenario)
            self.validate_physics_state()
            self._validate_spawn_clearance()

        # 5. capture deterministic snapshot
        print("[setup] snapshot", file=_s.stdout, flush=True)
        self._initial_state = dump_state(self._sim)
        self._build_report = report
        self._validated_anchor_poses = {name: (tuple(anchor.position), tuple(anchor.orientation))
                                        for name, anchor in anchors.items()}
        return report

    def _seed_pose_for(self, spec) -> list[float]:
        """Spawn seed pose: above the receptacle if placed, else above scene center.
        PILOT-DEBUG 2026-10-03: seed poses are spread per object — a stack of
        10 coincident objects at one point segfaults the stopped-sim import
        loop on this host (the knife scenario never stacked more than 3)."""
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
        idx = getattr(self, "_seed_spread_i", 0)
        self._seed_spread_i = idx + 1
        base[0] += 0.14 * (idx % 6) - 0.35
        base[1] += 0.12 * (idx % 4) - 0.18
        return base

    @defer_authoring_rendering
    def _apply_placements(self, scenario: ScenarioSpec, attempts: int) -> dict[str, Any]:
    
        from omnigibson.object_states import Inside, OnTop, Open

        scene = self._env.scene
        try:
            last_failure = None
            import time
            clean_state = dump_state(self._sim)
            from rummagebench.sim.omnigibson.inside_volume import placement_order
            import os
            expert_layout=os.environ.get('YGBENCH_EXPERT_LAYOUT')=='1'
            if expert_layout and os.environ.get('YGBENCH_GPU_PROPOSALS')!='1':
                raise ValueError('EXPERT_LAYOUT_REQUIRES_GPU_NATIVE_PROPOSALS')
            if expert_layout:attempts=1
            volumes={p.entity:float(np.prod(_arr(resolve_object(scene,p.entity).native_bbox))
                                     * np.prod(_arr(resolve_object(scene,p.entity).scale)))
                     for p in scenario.placements}
            order=placement_order(scenario.placements,volumes)
            if expert_layout:
                # Defer only thin leaf contents: no other placement depends on
                # them, so this remains a valid topological initialization.
                parents={p.receptacle for p in scenario.placements}
                thin={i for i,p in enumerate(scenario.placements) if p.relation=='inside'
                      and p.entity not in parents and float(_arr(resolve_object(scene,p.entity).native_bbox*
                          resolve_object(scene,p.entity).scale)[2])<.004}
                order=[i for i in order if i not in thin]+[i for i in order if i in thin]
            for attempt in range(attempts):
                expert_plans={};expert_placed={};expert_surface_plans={};deferred_surface_leaves=set()
                self._findingbench_contents_contexts={}
                report: dict[str, Any] = {"placements": [None]*len(scenario.placements), "initial_states": [],
                    "placement_order":[scenario.placements[i].entity for i in order],
                    "authoring_policy":"expert_live_cuda_joint_plan_v1" if expert_layout else "dependency_layers_oriented_inset_settled_open_v2"}
                if attempt:
                    logger.warning("placement attempt %d failed (%s); resampling",
                                   attempt, last_failure)
                    load_state(self._sim, clean_state, settle_steps=0)
                    self._collision_body_cache = None
    
                if getattr(self, '_isolated_authoring_staging', False):
                    for name in self._staged_authoring_names:
                        resolve_object(scene, name).disable_gravity()
    
                # remember open states BEFORE opening, so receptacles without a
                # forced initial state get restored to their scene default
                originally_open: dict[str, bool] = {}
                for p in scenario.placements:
                    rec = resolve_object(scene, p.receptacle)
                    if (p.relation == "inside" and rec.states.get(Open) is not None
                            and p.receptacle not in originally_open):
                        originally_open[p.receptacle] = bool(rec.states[Open].get_value())
                        if not originally_open[p.receptacle]:
                            self._set_open_carrying(rec, True, authoring_open_fraction=.9 if expert_layout else None)

                # A teleported open joint can recoil before any object arrives.
                # Settle this motion before the native sampler captures its
                # container-drift baseline; never lock joints or relax checks.
                if originally_open:
                    self.settle(30)
    
                for placement_index in order:
                    p=scenario.placements[placement_index]
                    obj = resolve_object(scene, p.entity)
                    rec = resolve_object(scene, p.receptacle)
                    if getattr(self, '_isolated_authoring_staging', False):
                        obj.enable_gravity()
                    started = time.monotonic()
                    print(f"[placement] attempt={attempt+1} start {p.entity} {p.relation} {p.receptacle} link={p.link}", flush=True)
                    native_open_container=(expert_layout and p.relation=='inside'
                        and getattr(rec,'category',None) in ('shopping_basket','wicker_basket','bowl','fruit_bowl','tray')
                        and Open not in rec.states)
                    if native_open_container:
                        from omnigibson.object_states.contains import m as contains_macros
                        names=[name for name,link in rec.links.items() if link.is_meta_link
                            and link.meta_link_type in contains_macros.CONTAINER_META_LINK_TYPES
                            and (not p.link or name==p.link or name.startswith(f'meta__{p.link}_'))]
                        if not names:raise SimBackendError('NATIVE_OPEN_CONTAINER_VOLUME_UNAVAILABLE')
                        print('[native-open-container-placement]',p.entity,p.receptacle,flush=True)
                        ok=any(self._place_in_link(obj,rec,name) for name in names)
                        report['open_container_initialization']='native_Inside_sampling_all_original_checks_GPU_proposals'
                    elif expert_layout and p.relation=='inside':
                        from rummagebench.sim.omnigibson.inside_volume import plan_native_contents
                        if p.receptacle not in expert_plans:
                            from rummagebench.sim.omnigibson.inside_volume import settle_authoring_container
                            report.setdefault('container_settling',{})[p.receptacle]=settle_authoring_container(self,rec)
                            expert_plans[p.receptacle]=plan_native_contents(self,scenario,p.receptacle)
                            report.setdefault('expert_layouts',{})[p.receptacle]=expert_plans[p.receptacle]
                        else:
                            from rummagebench.sim.omnigibson.inside_volume import replan_native_contents
                            expert_plans[p.receptacle]=replan_native_contents(self,scenario,p.receptacle,
                                expert_placed.get(p.receptacle,set()))
                        if p.entity not in expert_plans[p.receptacle]:
                            if p.entity in deferred_surface_leaves:
                                raise SimBackendError('EXPERT_DEFERRED_LEAF_HAS_NO_MEASURED_SUPPORT: '+p.entity)
                            deferred_surface_leaves.add(p.entity);order.append(placement_index)
                            obj.disable_gravity()
                            report.setdefault('deferred_contained_leaves',[]).append(p.entity)
                            print('[expert-deferred-contained-leaf] '+p.entity,flush=True)
                            continue
                        planned=expert_plans[p.receptacle][p.entity]
                        obj._findingbench_layout_candidate=planned.get('candidates',planned['candidate'])
                        try:
                            ok=self._place_in_link(obj,rec,planned['native_link'])
                        finally:
                            del obj._findingbench_layout_candidate
                    elif p.relation=='on_top' and p.orientation_hint_xyzw is not None:
                        ok=self._place_at_xy(obj,rec,p.position_xy,orientation_hint=p.orientation_hint_xyzw)
                    elif (expert_layout and os.environ.get('YGBENCH_MEASURED_SURFACE_LAYOUT','0')=='1'
                          and p.relation=='on_top' and not getattr(obj,'fixed_base',False)
                          and (not self._is_task_object(rec) or getattr(rec,'fixed_base',False))):
                        from rummagebench.sim.omnigibson.inside_volume import native_surface_group,plan_native_surface_group
                        group=native_surface_group(scenario,p.receptacle)
                        if (len(group)>=2 and any(q.entity==p.entity for q in group)
                                and getattr(rec,'category','') not in ('floor','floors')):
                            if p.receptacle not in expert_surface_plans:
                                expert_surface_plans[p.receptacle]=plan_native_surface_group(self,scenario,p.receptacle)
                                report.setdefault('expert_surface_layouts',{})[p.receptacle]=expert_surface_plans[p.receptacle]
                            ok=self._place_on_measured_surface(obj,rec,expert_surface_plans[p.receptacle][p.entity])
                        elif p.position_xy is not None:ok=self._place_at_xy(obj,rec,p.position_xy)
                        else:ok=self._place(obj,rec,p.relation,p.link)
                    elif p.position_xy is not None:
                        ok = self._place_at_xy(obj, rec, p.position_xy)
                    else:
                        ok = self._place(obj, rec, p.relation, p.link)
                    print(f"[placement] end {p.entity} placed={bool(ok)} elapsed_s={time.monotonic()-started:.3f}", flush=True)
                    if expert_layout and ok and p.relation=='inside':
                        expert_placed.setdefault(p.receptacle,set()).add(p.entity)
                    if expert_layout:
                        from rummagebench.sim.omnigibson.inside_volume import synchronize_authoring_geometry
                        synchronize_authoring_geometry(self)
                        import json
                        print('[expert-pose-result] '+json.dumps(dict(entity=p.entity,placed=bool(ok),
                            position=[float(v) for v in obj.get_position_orientation()[0]],
                            orientation=[float(v) for v in obj.get_position_orientation()[1]],
                            aabb=[[float(v) for v in bound] for bound in obj.aabb])),flush=True)
                    report["placements"][placement_index]={
                        "entity": p.entity,
                        "relation": p.relation,
                        "receptacle": p.receptacle,
                        "placed": bool(ok),
                        "elapsed_s": round(time.monotonic()-started, 3),
                    }
                    if not ok:
                        report['first_failed_placement']=dict(entity=p.entity,receptacle=p.receptacle,
                            relation=p.relation,link=p.link,attempt=attempt+1)
                        import json
                        print('[first-placement-failure] '+json.dumps(report['first_failed_placement']),flush=True)
                        # A failed early object makes this full layout invalid.
                        # Restart from clean state instead of spending hundreds
                        # of extra samples on its dependent / remaining objects.
                        for pending_index in order:
                            if report['placements'][pending_index] is None:
                                pending=scenario.placements[pending_index]
                                report['placements'][pending_index]=dict(entity=pending.entity,
                                    relation=pending.relation,receptacle=pending.receptacle,
                                    placed=False,elapsed_s=0,skipped='earlier_placement_failed')
                        break
                if getattr(self, '_isolated_authoring_staging', False):
                    for name in self._staged_authoring_names:
                        resolve_object(scene, name).enable_gravity()
                self.settle(5)  # let contents come to rest on the open drawer floors
    
                targets = {name: was for name, was in originally_open.items()}
                for entity, st in scenario.initial_states.items():
                    if st.open is not None:
                        targets[entity] = st.open
                        report["initial_states"].append({"entity": entity, "open": st.open})
                for entity, open_value in targets.items():
                    obj = resolve_object(scene, entity)
                    if obj.states.get(Open) is None:
                        raise SimBackendError(f"{entity} has no Open state")
                    self._set_open_carrying(obj, open_value)
                self.settle()
                # placement opens receptacles temporarily; a swinging door or the
                # settling contents can push a NEIGHBOUR's door / drawer open
                # (easy pilot, 6 of 23 episodes): re-assert once, then verify
                # below — an episode whose declared open states do not hold is
                # resampled, never built (TASK_TIERS_PLAN A.8.3)
                wrong = [e for e, v in targets.items()
                         if bool(resolve_object(scene, e).states[Open].get_value()) != v]
                if wrong:
                    for entity in wrong:
                        self._set_open_carrying(resolve_object(scene, entity), targets[entity])
                    self.settle()
    
                failed = []
                for entity, open_value in targets.items():
                    if bool(resolve_object(scene, entity).states[Open].get_value()) != open_value:
                        failed.append(f"{entity} open={not open_value} (declared {open_value})")
                for p, entry in zip(scenario.placements, report["placements"]):
                    obj = resolve_object(scene, p.entity)
                    rec = resolve_object(scene, p.receptacle)
                    strict_native = p.relation == "inside" and sum(q.receptacle == p.receptacle for q in scenario.placements) > 1
                    entry["verified"] = ((entry["placed"] or not strict_native)
                                         and self._verify_relation(obj, rec, p.relation, Inside, OnTop)
                                         and self._in_link(obj, rec, p.link))
                    if not entry["verified"]:
                        failed.append(f"{p.entity} {p.relation} {p.receptacle}")
                if not failed:
                    return report
                last_failure = failed
            raise SimBackendError(
                f"placement failed verification after physics settled: {last_failure}"
            )
        finally:
            # Gravity flags are not part of simulator state snapshots.
            for name in getattr(self, '_staged_authoring_names', ()):
                resolve_object(scene, name).enable_gravity()

    def _set_open_carrying(self, obj, open_value: bool, *, authoring_open_fraction: float | None = None) -> bool:
        
        from omnigibson.object_states import Open
        import os,json
        opening_targets=None
        if authoring_open_fraction is not None:
            if not open_value:raise ValueError('AUTHORING_FRACTION_ONLY_APPLIES_TO_OPENING')
            from rummagebench.sim.omnigibson.inside_volume import native_authoring_open_targets
            opening_targets=native_authoring_open_targets(obj,authoring_open_fraction)
        def audit(stage):
            if os.environ.get('YGBENCH_AUTHORING_CONTACT_AUDIT')!='1':return
            try:
                from omnigibson.utils.usd_utils import RigidContactAPI
                pairs=RigidContactAPI.get_contact_pairs(scene_idx=obj.scene.idx,query_set=[obj],with_set=None,current_only=True)
                print('[open-state-audit] '+json.dumps(dict(stage=stage,entity=obj.name,
                    requested_open=open_value,actual_open=bool(obj.states[Open].get_value()),
                    joint_positions=[float(v) for v in obj.get_joint_positions()],
                    joint_velocities=[float(v) for v in obj.get_joint_velocities()],
                    contacts=[list(p) for p in sorted(pairs)])),flush=True)
            except Exception as exc:
                print('[open-state-audit] '+json.dumps(dict(stage=stage,entity=obj.name,diagnostic_error=str(exc))),flush=True)
        audit('before_set')

        carried = []
        for item in self._env.scene.objects:
            link = self._containing_moving_link(obj, item)
            if link is not None:
                carried.append((item, link, self._prim_pose(link).inverse()
                                .compose(self._prim_pose(item))))
        for i, (item, _link, _rel) in enumerate(carried):
            # far apart from the scene and from each other for one step
            item.set_position_orientation(position=[1000.0 + 10.0 * i, 1000.0, 1000.0],
                                          orientation=[0, 0, 0, 1])
        ok = bool(obj.states[Open].set_value(open_value, fully=True))
        if ok and opening_targets is not None:
            for joint,position in opening_targets:joint.set_pos(position)
            ok=bool(obj.states[Open].get_value())
            print('[authoring-opening] '+json.dumps(dict(entity=obj.name,fraction=authoring_open_fraction,
                native_open=ok,targets=[dict(joint=j.name,position=p) for j,p in opening_targets])),flush=True)
        if carried:
            import torch as th

            self._sim.step()
            zero = th.zeros(3)
            for item, link, rel in carried:
                pose = self._prim_pose(link).compose(rel)
                item.set_position_orientation(position=pose.position.tolist(),
                                              orientation=pose.orientation.tolist())
                item.set_linear_velocity(zero)
                item.set_angular_velocity(zero)
            logger.info("%s %s carrying %s", "opened" if open_value else "closed",
                        obj.name, [(item.name, link.name) for item, link, _ in carried])
        audit('after_set')
        return ok

    def _containing_moving_link(self, container, item):
        held = (getattr(self._robot, "_ag_obj_in_hand", None) or {}).values()
        if (item is container or getattr(item, "fixed_base", False)
                or item in self._env.scene.robots or item in held):
            return None
        try:
            lo, hi = item.aabb
            center = (np.asarray(_arr(lo)) + np.asarray(_arr(hi))) / 2.0
        except Exception:
            return None
        best, best_vol = None, math.inf
        for name, link in (container.links or {}).items():
            if name == container.root_link_name or name.startswith("meta__"):
                continue
            llo, lhi = (np.asarray(_arr(v)) for v in link.aabb)
            if np.all(center >= llo - 1e-3) and np.all(center <= lhi + 1e-3):
                vol = float(np.prod(lhi - llo))
                if vol < best_vol:
                    best, best_vol = link, vol
        return best

    @staticmethod
    def _prim_pose(prim):
        from rummagebench.feasibility.ik_solver import Pose

        pos, quat = prim.get_position_orientation()
        return Pose.from_lists(_arr(pos), _arr(quat))

    def _place_in_link(self, obj, rec, link: str) -> bool:
        
        import torch as th

        actual_link=(rec.links or {}).get(link)
        if actual_link is not None and getattr(actual_link,'is_meta_link',False):
            from rummagebench.sim.omnigibson.inside_volume import sample_selected_volume
            self._collision_body_cache=None
            ok,evidence=sample_selected_volume(obj,rec,link)
            self._last_volume_placement=evidence
            self._collision_body_cache=None
            return ok

        volumes = [l for n, l in (rec.links or {}).items()
                   if n.startswith(f"meta__{link}_") and "fillable" in n]
        if not volumes:
            raise SimBackendError(f"{rec.name}:{link} has no fillable volume; "
                                  f"links: {sorted(rec.links or {})}")
        if sum(p.receptacle == rec.name and p.link == link
               for p in self._scenario.placements) > 1:
            # A dense drawer cannot place every object at its center. Keep
            # the native collision-aware sampler and select its real volume.
            key = next(n for n, v in rec.links.items() if v is volumes[0])
            return self._place_in_link(obj, rec, key)
        center = _arr(volumes[0].get_position_orientation()[0])
        obj.set_position_orientation(position=center, orientation=[0, 0, 0, 1])
        obj.set_linear_velocity(th.zeros(3))
        obj.set_angular_velocity(th.zeros(3))
        self.settle(30)
        return self._is_inside(obj, rec) and self._in_link(obj, rec, link)

    def _in_link(self, obj, rec, link: str | None) -> bool:
        if link is None:
            return True
        if link not in (rec.links or {}):
            raise SimBackendError(f"{rec.name} has no link {link!r}; "
                                  f"links: {sorted(rec.links or {})}")
        if getattr(rec.links[link],'is_meta_link',False):
            from rummagebench.sim.omnigibson.inside_volume import selected_volume_contains
            return selected_volume_contains(obj,rec,link)
        return self._containing_moving_link(rec, obj) is rec.links[link]

    def _place(self, obj, rec, relation: str, link: str | None = None) -> bool:
        """Place obj into/on rec via the state sampler; returns sampler success.

        `inside` receptacles must already be open (_apply_placements). The
        sampler is probabilistic, so we retry several times and fall back to
        direct pose placement inside the receptacle's AABB. With `link`, the
        object is dropped into that drawer instead (_place_in_link).
        """
        from omnigibson.object_states import Inside, OnTop
        import os
        if (os.environ.get('YGBENCH_GPU_OCCUPANCY_RANKING')=='1'
                or os.environ.get('YGBENCH_GPU_LOCAL_PROPOSALS')=='1') and relation=='inside':
            boxes=[]
            for placement in self._scenario.placements:
                if placement.entity in (obj.name,rec.name):continue
                other=resolve_object(self._env.scene,placement.entity)
                # Cabinet/floor AABBs include their empty interiors. Scoring
                # those as solid obstacles would penalize every valid cavity
                # pose. Native collision checks still test all real colliders.
                if getattr(other,'fixed_base',False):continue
                lo,hi=(np.asarray(_arr(v), dtype=float) for v in other.aabb)
                if np.isfinite(lo).all() and np.isfinite(hi).all() and (hi>=lo).all():
                    boxes.append([lo.tolist(),hi.tolist()])
            obj._findingbench_proposal_obstacles=boxes
        elif hasattr(obj,'_findingbench_proposal_obstacles'):
            del obj._findingbench_proposal_obstacles

        if link is not None:
            return self._place_in_link(obj, rec, link)
        if relation == "inside":
            from rummagebench.sim.omnigibson.inside_volume import fitting_volume_names, sample_selected_volume
            candidates, audit = fitting_volume_names(obj, rec)
            print(f"[placement-volume] {obj.name} in {rec.name}: {audit}", flush=True)
            for selected in candidates:
                ok, evidence = sample_selected_volume(obj, rec, selected)
                self._last_volume_placement = evidence
                if ok:
                    return True
            # No center-drop fallback, and no repeated sampling of an
            # impossible first drawer. Only native verified placements count.
            return False
        if relation == "on_top" and self._is_task_object(rec) and not getattr(rec, "fixed_base", False):
            return self._stack_on(obj, rec)
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

        if not ok and relation == "inside" and sum(p.receptacle == rec.name for p in self._scenario.placements) > 1:
            return False  # Dense authoring must not bypass the native sampler.

        # deterministic fallback: park the object at the receptacle center and
        # let physics settle it into the container volume
        if not ok and relation == "inside" and link is None:
            try:
                rec_pos = rec.get_position_orientation()[0]
                if hasattr(rec_pos, "detach"):
                    rec_pos = rec_pos.detach().cpu().numpy()
                obj.set_position_orientation(
                    position=[float(rec_pos[0]), float(rec_pos[1]), float(rec_pos[2]) + 0.05],
                    orientation=[0, 0, 0, 1],
                )
                self.settle(10)
                ok = self._is_inside(obj, rec)
            except Exception as e:
                logger.warning("pose fallback placement failed for %s: %s", obj.name, e)

        return ok

    def _place_at_xy(self, obj, rec, xy, *, orientation_hint=None) -> bool:
        """Deterministic on_top placement at world ``xy``: rest the object on
        the receptacle's support top (receptacle_region) there, settle, and
        report whether OnTop holds. The XY must lie on the support region."""
        import torch as th

        region = self.receptacle_region(rec.name)
        if (region is not None and region.kind == "top_surface"
                and getattr(rec, "fixed_base", False) and len(getattr(rec, "links", {}) or {}) > 1):
            # This is an evaluator-only initial OnTop proposal. Interaction
            # targeting uses a cabinet's root link, which need not include a
            # separate tabletop link. Use the real whole-body bounds here;
            # native contact/OnTop and subsequent settling still decide fit.
            bounds = self.entity_aabb(rec.name)
            if bounds is not None:
                from rummagebench.feasibility.interaction_target import InteractionRegion
                region = InteractionRegion(lo=list(bounds[0]), hi=list(bounds[1]), kind="top_surface")
        if region is None or region.kind != "top_surface":
            raise SimBackendError(f"{rec.name} has no top surface for position_xy")
        x, y = float(xy[0]), float(xy[1])
        if not (region.lo[0] <= x <= region.hi[0] and region.lo[1] <= y <= region.hi[1]):
            raise SimBackendError(f"position_xy {xy} is outside the top surface of {rec.name}")
        # Authored XY denotes the closed native bbox centre, as used by the
        # constructed-furniture map. Roots may be far from this centre.
        # Do not use the live bbox's XY centre: opening drawers moves it.
        quat=[0,0,0,1] if orientation_hint is None else orientation_hint
        obj.set_position_orientation(orientation=quat)
        if orientation_hint is not None:
            from rummagebench.sim.omnigibson.inside_volume import synchronize_authoring_geometry
            synchronize_authoring_geometry(self)
        lo, hi = (np.asarray(_arr(v)) for v in obj.aabb)
        pos = _arr(obj.get_position_orientation()[0])
        bottom_offset = float(pos[2] - lo[2])  # origin height above the AABB bottom
        native_offset = np.asarray(_arr(obj.scaled_bbox_center_in_base_frame))
        if orientation_hint is not None:
            from scipy.spatial.transform import Rotation
            native_offset=Rotation.from_quat(quat).apply(native_offset)
        obj.set_position_orientation(
            position=[x + float(native_offset[0]), y + float(native_offset[1]),
                      float(region.hi[2]) + bottom_offset + (0.0 if getattr(obj, "fixed_base", False) else 0.01)],
            orientation=quat)
        # Fixed native furniture has a kinematic root, not a dynamic velocity API.
        if callable(getattr(obj.root_link, "set_linear_velocity", None)):
            obj.set_linear_velocity(th.zeros(3))
            obj.set_angular_velocity(th.zeros(3))
        self.settle(20)
        from omnigibson.object_states import Inside, OnTop
        return bool(self._verify_relation(obj, rec, "on_top", Inside, OnTop))

    def _place_on_measured_surface(self,obj,rec,plan) -> bool:
        """Authoring-only proposal on an observed collider; native OnTop gates it."""
        import torch as th
        from omnigibson.utils.sampling_utils import raytest
        from omnigibson.object_states import Inside,OnTop
        xy=plan['position_xy'];height=float(plan['support_height_m'])
        half=np.asarray(plan['footprint_m'])/2
        owners=[link.prim_path for link in rec.links.values()]
        for dx,dy in [(0.,0.),*[(a*half[0],b*half[1]) for a in (-1,1) for b in (-1,1)]]:
            hit=raytest([xy[0]+dx,xy[1]+dy,height+.04],[xy[0]+dx,xy[1]+dy,height-.04])
            body=str(hit.get('rigidBody',''))
            if (not hit.get('hit') or not any(body==p or body.startswith(p+'/') for p in owners)
                    or float(hit.get('normal',[0,0,0])[2])<.98
                    or abs(float(hit['position'][2])-height)>.005):
                raise SimBackendError('EXPERT_SURFACE_FOOTPRINT_UNSUPPORTED: '+obj.name)
        yaw=math.radians(plan['yaw_degrees']);quat=[0.,0.,math.sin(yaw/2),math.cos(yaw/2)]
        obj.set_position_orientation(orientation=quat)
        lo,hi=(np.asarray(_arr(v)) for v in obj.aabb);pos=np.asarray(_arr(obj.get_position_orientation()[0]))
        offset=pos-(lo+hi)/2;bottom_offset=float(pos[2]-lo[2])
        obj.set_position_orientation(position=[xy[0]+float(offset[0]),xy[1]+float(offset[1]),height+bottom_offset+.015],orientation=quat)
        if callable(getattr(obj.root_link,'set_linear_velocity',None)):
            obj.set_linear_velocity(th.zeros(3));obj.set_angular_velocity(th.zeros(3))
        self.settle(20)
        valid=bool(self._verify_relation(obj,rec,'on_top',Inside,OnTop))
        if not valid:
            import json
            from omnigibson.object_states import Touching,Adjacency
            from omnigibson.utils.usd_utils import RigidContactAPI
            audit=dict(entity=obj.name,receptacle=rec.name,scope='read-only native rejection attribution')
            for state in (OnTop,Touching,Adjacency):
                try:
                    i=state.OBJ_IDXS[obj.relative_prim_path];j=state.OBJ_IDXS[rec.relative_prim_path];s=obj.scene.idx
                    audit[state.__name__]=dict(cpu=state.VALUES_CPU[s,i,j].tolist(),gpu=state.VALUES[s,i,j].detach().cpu().tolist())
                except Exception as exc:audit[state.__name__]=dict(error=str(exc))
            audit['physical_contact_pairs']=[list(p) for p in sorted(RigidContactAPI.get_contact_pairs(
                scene_idx=obj.scene.idx,query_set=[obj],with_set=[rec],current_only=True))]
            audit['all_contact_pairs']=[list(p) for p in sorted(RigidContactAPI.get_contact_pairs(
                scene_idx=obj.scene.idx,query_set=[obj],with_set=None,current_only=True))]
            try:
                view=RigidContactAPI._CONTACT_VIEW[obj.scene.idx]
                sensors=list(view.sensor_paths);filters=list(RigidContactAPI._COL_IDX_TO_PATH[obj.scene.idx])
                impulses=view.get_contact_force_matrix(dt=1.0);net=view.get_net_contact_forces(dt=1.0)
                rows=[sensors.index(link.prim_path) for link in obj.links.values() if link.prim_path in sensors]
                raw=[]
                for i in rows:
                    contacts=th.nonzero(impulses[i].abs().sum(-1)>0).flatten().cpu().tolist()
                    raw.append(dict(sensor=sensors[i],net_impulse=net[i].detach().cpu().tolist(),
                        contacts=[dict(body=str(filters[j]),impulse=impulses[i,j].detach().cpu().tolist()) for j in contacts[:32]]))
                audit['direct_physx']=raw
            except Exception as exc:audit['direct_physx_error']=type(exc).__name__+': '+str(exc)
            audit['object_mass']=float(obj.mass)
            print('[native-surface-rejection] '+json.dumps(audit),flush=True)
        return valid

    def _apply_declared_clearance(self, scenario, *, preserve_recipe_identity=False):
        """Edit native geometry without mutating a frozen member's declaration.

        Legacy authoring remaps explicit coordinates after moving a fixture.
        Frozen reload already restores the saved world and validates the
        original recipe twice; only a private working copy may be remapped.
        All physical edits and original bundle/hash checks still execute.
        """
        from rummagebench.sim.omnigibson.clearance import apply_clearance
        from rummagebench.sim.omnigibson.env_factory import repo_root
        previous=self._scenario
        self._scenario=scenario.model_copy(deep=True) if preserve_recipe_identity else scenario
        try:
            return apply_clearance(self, repo_root() / scenario.scene.clearance_manifest)
        finally:
            self._scenario=previous

    def _is_task_object(self, obj) -> bool:
        """A scenario-spawned object (not scene furniture: BEHAVIOR tables
        are often not fixed-base)."""
        return any(spec.name == obj.name for spec in self._scenario.objects)

    def _stack_on(self, obj, rec) -> bool:
        """Pose-level stack of ``obj`` on a MOVABLE object ``rec`` (medium
        covers): centred on rec's footprint, rec's orientation, resting just
        on its top, then settled. OnTop sampling picks a random point on a
        small support and the cover slides off while settling (medium pilot
        2, A.10: 9/9 stacks failed)."""
        import torch as th

        from scipy.spatial.transform import Rotation

        rlo, rhi = (np.asarray(_arr(v)) for v in rec.aabb)
        _, quat = rec.get_position_orientation()
        above = [float((rlo[0] + rhi[0]) / 2), float((rlo[1] + rhi[1]) / 2),
                 float(rhi[2]) + 1.0]
        # rec's orientation, or turned 90 deg about z when that aligns the
        # long axes better (the planner compares sorted footprints): keep the
        # yaw with the larger minimum overhang
        best = None
        for turn in (0.0, 90.0):
            q = (Rotation.from_euler("z", turn, degrees=True)
                 * Rotation.from_quat(_arr(quat))).as_quat().tolist()
            obj.set_bbox_center_position_orientation(position=th.tensor(above,dtype=th.float32),orientation=th.tensor(q,dtype=th.float32))
            olo, ohi = (np.asarray(_arr(v)) for v in obj.aabb)
            overhang = min((ohi[0] - olo[0]) - (rhi[0] - rlo[0]),
                           (ohi[1] - olo[1]) - (rhi[1] - rlo[1]))
            if best is None or overhang > best[0] + 1e-4:
                best = (overhang, q)
        quat = best[1]
        obj.set_bbox_center_position_orientation(position=th.tensor(above,dtype=th.float32),orientation=th.tensor(quat,dtype=th.float32))
        olo, _ = (np.asarray(_arr(v)) for v in obj.aabb)
        pos = np.asarray(_arr(obj.get_position_orientation()[0]))
        lift = float(pos[2] - olo[2])  # object origin above its own bottom
        obj.set_position_orientation(position=[float(pos[0]), float(pos[1]),
                                               float(rhi[2]) + lift + 0.003],
                                     orientation=quat)
        obj.set_linear_velocity(th.zeros(3))
        obj.set_angular_velocity(th.zeros(3))
        self.settle(20)
        ok = self._is_stacked_on(obj, rec)
        if not ok:
            logger.warning("stack %s on %s failed: obj aabb %s, rec aabb %s", obj.name,
                           rec.name, [_arr(v) for v in obj.aabb], [_arr(v) for v in rec.aabb])
        return ok

    @staticmethod
    def _is_stacked_on(obj, rec, tol: float = 0.01) -> bool:
        """obj covers rec: rec's footprint centre lies under obj's footprint,
        obj reaches above rec's top and rests on / near it (a wide cover on a
        small item may tilt down to the support on one side)."""
        try:
            lo, hi = (np.asarray(_arr(v)) for v in obj.aabb)
            rlo, rhi = (np.asarray(_arr(v)) for v in rec.aabb)
        except Exception:
            return False
        c = (rlo + rhi) / 2.0
        return bool(hi[2] >= rhi[2] - tol and lo[2] <= rhi[2] + 0.05
                    and lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1])

    def _verify_relation(self, obj, rec, relation: str, Inside, OnTop) -> bool:
        if relation == "inside":
            return self._is_inside(obj, rec)
        if getattr(obj, "fixed_base", False) and getattr(rec, "category", None) == "floors":
            # Static furniture has no static-static solver contact pair.
            # Match the already validated production support check.
            lo, hi = (np.asarray(_arr(v)) for v in obj.aabb)
            rlo, rhi = (np.asarray(_arr(v)) for v in rec.aabb)
            return bool(abs(float(lo[2] - rhi[2])) <= .001
                        and np.all(lo[:2] >= rlo[:2] - .001)
                        and np.all(hi[:2] <= rhi[:2] + .001))
        if self._is_task_object(rec) and not getattr(rec, "fixed_base", False):
            return self._is_stacked_on(obj, rec)
        if OnTop not in obj.states:
            return False
        # Inside/OnTop are binary (relative) states: get_value(other)
        return bool(obj.states[OnTop].get_value(rec))

    @staticmethod
    def _is_inside(obj, rec, tol: float = 0.01) -> bool:
    
        from omnigibson.object_states import Inside

        try:
            if Inside in obj.states and bool(obj.states[Inside].get_value(rec)):
                return True
            lo, hi = (np.asarray(_arr(v)) for v in obj.aabb)
            rlo, rhi = (np.asarray(_arr(v)) for v in rec.aabb)
        except Exception:
            return False
        return bool(np.all(lo >= rlo - tol) and np.all(hi <= rhi + tol))

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
        if scenario.scene.frozen_member_manifest is not None or self._scenario.scene.frozen_member_manifest is not None:
            raise SimBackendError("frozen family members require their own verified setup; placement resampling is forbidden")
        seed_everything(int(scenario.id.encode().hex()[-6:], 16) % (2**31))

        # start from the clean snapshot (releases lingering assisted grasps)
        self.reset()
        report = self._apply_placements(scenario, attempts=3)
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
        if self._scenario.scene.frozen_member_manifest is not None:
            # The checked member checkpoint already includes the robot start.
            # Loading a newly dumped world, stepping twenty times and then
            # teleporting creates a different reconstruction recipe and can
            # disturb native props. Replay the original member input instead.
            from rummagebench.authoring.tasks.household_state import restore_frozen_member
            restore_frozen_member(self, Path(self._scenario.scene.frozen_member_manifest), self._scenario)
            self.validate_physics_state()
            self._validate_spawn_clearance()
            return
        load_state(self._sim, self._initial_state)
        if getattr(self._scenario, "behavior_task", None) is not None:
            self.validate_physics_state()
            return
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
        half = getattr(getattr(getattr(self,"_scenario",None),"robot",None),"base_half_extent",.40)
        if not base_pose_collision_free(self, position[0], position[1],
                                        yaw_from_quat(orientation), half, 0.03):
            from rummagebench.skills.move import footprint_corners, navigation_aabbs, _aabb_overlaps_footprint
            corners=footprint_corners(position[0],position[1],yaw_from_quat(orientation),half)
            try:
                controlled=self.robot_entity_names()
                self._last_spawn_failure={'position':position,'orientation':orientation,'blockers':[
                    {'entity':name,'navigation_aabb':box} for name in self.entity_names()
                    if name not in controlled and self.describe_entity(name) is not None and not self.is_holding(name)
                    for box in navigation_aabbs(self,name) if _aabb_overlaps_footprint(box,corners,.03)]}
            except Exception as exc:
                self._last_spawn_failure={'diagnostic_error':type(exc).__name__}
            logger.error("Spawn clearance diagnostics: %s", self._last_spawn_failure)
            raise SimBackendError("Initial robot footprint overlaps an obstacle; invalid spawn")

    def get_observation(self) -> np.ndarray:
        if getattr(self, "_panorama_camera", None) is not None:
            return self.capture_panorama_frame().rgb
        return capture_head_rgb(self._env, self._robot.name)

    def capture_view_frame(self, yaw, pitch, width, height, zoom=1):
        return self._panorama_camera.capture_view(yaw, pitch, width, height, zoom=zoom)

    def capture_check_image(self, entity):
        return self._panorama_camera.capture_check(entity)

    def check_appearance_supported(self, entity):
        obj = resolve_object(self._env.scene, entity)
        return getattr(self, '_panorama_camera', None) is not None and obj.n_joints == 0

    def capture_panorama_frame(self):
        return self._panorama_camera.capture()

    def behavior_goal_satisfied(self):
        task = self._env.task
        # Current pinned BDDL compiled API. Fail loudly on incompatible SDKs.
        if hasattr(task, "compiled_task") and hasattr(task, "_evaluate_predicate"):
            success, _ = task.compiled_task.check_goal(
                lambda predicate, *entities: task._evaluate_predicate(0, predicate, *entities))
            return bool(success)
        raise SimBackendError("native task evaluation requires the pinned compiled BDDL API")

    def native_inside_relation(self, entity: str, receptacle: str):
        """Evaluator-only containment for open basins and movable baskets.

        Use the simulator predicate, never the outer-AABB fallback used by
        legacy placement authoring. None means unsupported, not success.
        """
        from omnigibson.object_states import Inside
        obj = resolve_object(self._env.scene, entity)
        rec = resolve_object(self._env.scene, receptacle)
        state = obj.states.get(Inside)
        if state is None:
            return None
        return bool(state.get_value(rec))

    def resolve_entity(self, name: str) -> ResolvedTarget:
        obj = resolve_object(self._env.scene, name)
        info = build_entity_info(obj)
        self._entity_infos[name] = info
        return ResolvedTarget(kind=TargetKind.ENTITY, entity=name, info=info)

    # ------------------------------------------------------------------ NAV

    def is_anchor_validated(self, name: str, anchor) -> bool:
        return (anchor is not None and self._validated_anchor_poses.get(name)
                == (tuple(anchor.position), tuple(anchor.orientation)))

    def park_entities(self, names: list[str]) -> list[tuple[Any, Any]]:
        """Counterfactual (oracle realization only): move objects far out of
        the scene without stepping physics; returns the handles to restore
        with unpark_entities()."""
        import torch as th

        parked = []
        for i, name in enumerate(names):
            obj = resolve_object(self._env.scene, name)
            parked.append((obj, obj.get_position_orientation()))
            obj.set_position_orientation(position=[1000.0 + 10.0 * i, 1000.0, 1000.0],
                                         orientation=[0, 0, 0, 1])
            obj.set_linear_velocity(th.zeros(3))
            obj.set_angular_velocity(th.zeros(3))
        self._collision_body_cache = None
        return parked

    def unpark_entities(self, parked: list[tuple[Any, Any]]) -> None:
        import torch as th

        for obj, (pos, quat) in parked:
            obj.set_position_orientation(position=pos, orientation=quat)
            obj.set_linear_velocity(th.zeros(3))
            obj.set_angular_velocity(th.zeros(3))
        self._collision_body_cache = None

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

    def navigate_oracle_endpoint(self, anchor: AnchorSpec) -> None:
        """Fixed-heading navigation transports attached objects rigidly too."""
        before, orientation = self.robot_pose()
        if not np.allclose(orientation, anchor.orientation, atol=1e-6):
            raise ValueError("oracle navigation must preserve heading")
        delta = np.asarray(anchor.position) - np.asarray(before)
        held = []
        for obj in (getattr(self._robot, "_ag_obj_in_hand", {}) or {}).values():
            if obj is not None:
                pos, quat = obj.get_position_orientation()
                held.append((obj, _arr(pos) + delta, quat))
        self.teleport_robot(anchor)
        for obj, pos, quat in held:
            obj.set_position_orientation(position=pos, orientation=quat)

    def navigate_oracle_pose(self, anchor: AnchorSpec) -> None:
        """Object-directed MOVE_TO arrival (position + new heading): attached
        objects follow the base rigidly (planar rotation about the base)."""
        from scipy.spatial.transform import Rotation
        from rummagebench.skills.move import yaw_from_quat
        before, orientation = self.robot_pose()
        dyaw = yaw_from_quat(list(anchor.orientation)) - yaw_from_quat(orientation)
        rot = Rotation.from_euler("z", dyaw)
        held = []
        for obj in (getattr(self._robot, "_ag_obj_in_hand", {}) or {}).values():
            if obj is not None:
                pos, quat = obj.get_position_orientation()
                rel = _arr(pos) - np.asarray(before)
                new_pos = np.asarray(anchor.position) + rot.apply(rel)
                new_quat = (rot * Rotation.from_quat(_arr(quat))).as_quat()
                held.append((obj, new_pos, new_quat))
        self.teleport_robot(anchor)
        for obj, pos, quat in held:
            obj.set_position_orientation(position=pos, orientation=quat)

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
        return self._set_open_carrying(obj, open_value)

    def opened_entity_aabb(self, entity: str):
        """World AABB of ``entity`` FULLY open (as OPEN leaves it), measured
        counterfactually: snapshot, open, one physics step for the link poses,
        read, restore without physics steps (OBSERVE's capture/restore).
        None when the entity has no Open state or is already open."""
        from omnigibson.object_states import Open

        obj = resolve_object(self._env.scene, entity)
        if obj.states.get(Open) is None or bool(obj.states[Open].get_value()):
            return None
        snapshot = self.capture_observe_state()
        try:
            obj.states[Open].set_value(True, fully=True)
            self._sim.step()
            return self.entity_aabb(entity)
        finally:
            self.restore_observe_state(snapshot)

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
        # the feasibility engine models ONE arm (kinematics.end_effector_link);
        # the object must be attached to that hand, otherwise the grasp
        # offset (skills/grasp.py) describes a different hand than the one
        # later PLACE/OPEN checks move
        arm = self._modelled_arm() or self._pick_arm(obj)
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
            contact_pos = self._bring_to_hand(obj, arm)
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
                contact_pos = self._bring_to_hand(obj, retry_arm)
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

    def _filter_held_vs_robot(self, obj, enable: bool) -> int:
        """Add (enable) / remove collision filter pairs between every link of
        ``obj`` and every robot link; returns the number of pairs changed.
        NOT USED: editing USD filtered pairs while the sim plays invalidates
        the PhysX tensor views ("Failed to get rigid body transforms",
        T01 L1_S1 2026-10-19); kept for a stop/play-safe caller."""
        pairs = getattr(self, "_held_filter_pairs", None)
        if pairs is None:
            pairs = self._held_filter_pairs = {}
        name = getattr(obj, "name", None)
        changed = 0
        if enable:
            if name in pairs:
                return 0
            done = []
            for olink in (obj.links or {}).values():
                for rlink in (getattr(self._robot, "links", None) or {}).values():
                    try:
                        olink.add_filtered_collision_pair(rlink)
                        done.append((olink, rlink))
                    except Exception as e:
                        logger.debug("collision filter %s/%s failed: %s", name, rlink, e)
            pairs[name] = done
            changed = len(done)
        else:
            for olink, rlink in pairs.pop(name, []):
                try:
                    olink.remove_filtered_collision_pair(rlink)
                    changed += 1
                except Exception as e:
                    logger.debug("collision unfilter %s failed: %s", name, e)
        return changed

    def _bring_to_hand(self, obj, arm: str):
       
        import torch as th

        eef = self._robot.eef_links[arm]
        pos, quat = eef.get_position_orientation()
        pos = np.asarray(_arr(pos), dtype=float).reshape(-1)[:3]
        fwd = float(getattr(self, "held_forward_m", HELD_FORWARD_AUTO))
        if fwd == HELD_FORWARD_AUTO:
            # long objects (fork, knife) placed at the EEF origin overlap the
            # gripper / camera links and the contact lifted the base after
            # GRASP (T03 L1_S1, L2_S1, 2026-10-19; scripts/probe_grasp_lift.py):
            # hold them ahead of the fingertips. Compact objects keep the
            # legacy realization (T02/T04 PLACE certified with it).
            half = self.place_half_extents(getattr(obj, "name", "")) or [0.0, 0.0, 0.0]
            fwd = max(half) + 0.02 if max(half) > LONG_OBJECT_HALF_M else 0.0
        if fwd > 0.0:
            # put the object ahead of the fingertips (tool +z), clear of the
            # robot's own links
            from rummagebench.feasibility.ik_solver import quat_rotate
            q = np.asarray(_arr(quat), dtype=float).reshape(-1)[:4]
            pos = pos + quat_rotate(q, np.array([0.0, 0.0, fwd]))
        pos = th.as_tensor(pos, dtype=th.float32)
        obj.set_position_orientation(position=pos)
        obj.set_linear_velocity(th.zeros(3))
        obj.set_angular_velocity(th.zeros(3))
        return pos

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

    def _modelled_arm(self) -> str | None:
        """Arm whose EEF link is the scenario's kinematics end_effector_link
        (the arm the pinocchio feasibility engine solves IK for); None when
        the scenario has no kinematics block or no arm matches."""
        kin = getattr(getattr(getattr(self, "_scenario", None), "robot", None),
                      "kinematics", None)
        eef_link = getattr(kin, "end_effector_link", None)
        if not eef_link:
            return None
        try:
            names = dict(self._robot.eef_link_names)
        except Exception:
            return None
        for arm, link in names.items():
            if link == eef_link:
                return arm
        return None

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

    def flush_render(self, renders: int = 6) -> None:
        """A few no-physics renders so renderer buffers reflect a just-
        executed state change (drawer/door moved). The capture path re-renders
        and camera-sync-gates on its own, so a small fixed flush suffices;
        30 was a contended-GPU sledgehammer dominating per-action latency."""
        if self._sim is not None:
            for _ in range(renders):
                self._sim.render()

    def validate_physics_state(self) -> None:
        """Fail an invalid run before corrupt physics reaches rendering/scoring."""
        def finite(value):
            value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            return bool(np.isfinite(value).all())
        context = {}
        self._last_physics_failure = None
        try:
            for obj in self._env.scene.objects:
                context = {'entity':obj.name,'component':'object_pose'}
                position, quaternion = obj.get_position_orientation()
                if not finite(position) or not finite(quaternion):
                    raise ValueError("nonfinite object pose")
                context = {'entity':obj.name,'component':'joint_positions'}
                if obj.n_joints and not finite(obj.get_joint_positions()):
                    raise ValueError("nonfinite joint positions")
                for link in obj.links.values():
                    context = {'entity':obj.name,'component':'link_pose','link':getattr(link,'name',None)}
                    position, quaternion = link.get_position_orientation()
                    if not finite(position) or not finite(quaternion):
                        raise ValueError("nonfinite link pose")
        except (AssertionError, ValueError) as error:
            # Evaluator-only diagnostics; public error text stays generic.
            self._last_physics_failure = {**context,'reason':str(error)}
            raise SimBackendError("Physics state is invalid; run must not be scored") from error

    def dump_state(self) -> Any:
        return dump_state(self._sim)

    def load_state(self, state: Any) -> None:
        load_state(self._sim, state)

    def capture_observe_state(self):
        """Full state for counterfactual multi-view rendering (evaluator-only)."""
        from copy import deepcopy

        from rummagebench.sim.omnigibson import spatial_snapshot
        return {"sim": self.dump_state(), "commanded_pose": deepcopy(self._commanded_pose),
                "spatial_cache": spatial_snapshot.capture(self._env.scene, self._sim.current_time_step_index)}

    def restore_observe_state(self, snapshot):
        """Restore without physics steps that would mutate the saved world."""
        from copy import deepcopy

        load_state(self._sim, snapshot["sim"], settle_steps=0)
        if "spatial_cache" in snapshot:
            from rummagebench.sim.omnigibson import spatial_snapshot
            spatial_snapshot.restore(self._env.scene, self._sim.current_time_step_index, snapshot["spatial_cache"])
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

    def navigation_obstacle_aabbs(self, name):
        """Floor-relative, conservative obstacles for the common MOVE gate."""
        scenario=getattr(self,'_scenario',None)
        if (getattr(scenario, 'behavior_task', None) is not None
                or getattr(getattr(scenario, 'scene', None), 'clearance_manifest', None)):
            from rummagebench.sim.omnigibson.usd_collision import collider_navigation_aabbs
            obj = resolve_object(self._env.scene, name)
            signature = tuple(tuple(np.r_[_arr(link.get_position_orientation()[0]),
                                            _arr(link.get_position_orientation()[1]), _arr(link.scale)])
                              for link in obj.links.values())
            cache = getattr(self, '_navigation_native_cache', {})
            if name not in cache or cache[name][0] != signature:
                cache[name] = (signature, collider_navigation_aabbs(obj))
                self._navigation_native_cache = cache
            origin=np.array([0.,0.,float(self.robot_pose()[0][2])])
            return [((np.asarray(lo)-origin).tolist(),(np.asarray(hi)-origin).tolist()) for lo,hi in cache[name][1]]
        if getattr(scenario,'scene',None) is None or scenario.scene.source!='infinigen':
            box=self.entity_aabb(name)
            origin=np.array([0.,0.,float(self.robot_pose()[0][2])])
            return [] if box is None else [((np.asarray(box[0])-origin).tolist(),(np.asarray(box[1])-origin).tolist())]
        manifest=getattr(self._env,'_findingbench_assembly_manifest',None)
        floor_z=getattr(self._env,'_findingbench_floor_z',None)
        if manifest is None or floor_z is None or not np.isfinite(floor_z):
            raise SimBackendError('INFINIGEN_VERIFIED_NAVIGATION_DATUM_REQUIRED')
        item=next((o for o in manifest['objects'] if o['name']==name),None)
        if (item is not None and item.get('role')=='static_background'
                and (item.get('source_component') in ('wall','exterior') or item.get('source_id') is None)):
            from rummagebench.sim.omnigibson.usd_collision import static_architecture_navigation_aabbs
            obj=resolve_object(self._env.scene,name)
            # Source meshes are immutable kinematic assets. Reuse clipping
            # only while every live link frame and scale is exactly unchanged;
            # restores or diagnostic parking at another pose invalidate it.
            frames=[]
            for link in obj.links.values():
                pos,quat=link.get_position_orientation()
                values=np.r_[_arr(pos),_arr(quat),_arr(link.scale)]
                if not np.isfinite(values).all():raise SimBackendError('NAVIGATION_STATIC_FRAME_NONFINITE')
                frames.append(tuple(values))
            scale=np.asarray(_arr(obj.scale),float)
            if not np.isfinite(scale).all():raise SimBackendError('NAVIGATION_STATIC_SCALE_NONFINITE')
            key=(id(obj),float(floor_z),tuple(scale),tuple(frames))
            cache=getattr(self,'_navigation_static_cache',{})
            if name not in cache or cache[name][0]!=key:
                cache[name]=(key,static_architecture_navigation_aabbs(obj,floor_z))
                self._navigation_static_cache=cache
            return cache[name][1]
        box=self.entity_aabb(name)
        if box is None:raise SimBackendError(f'INFINIGEN_NAVIGATION_GEOMETRY_UNAVAILABLE:{name}')
        values=np.asarray(box,float)
        if values.shape!=(2,3) or not np.isfinite(values).all():
            raise SimBackendError(f'INFINIGEN_NAVIGATION_GEOMETRY_INVALID:{name}')
        if item is not None and item.get('source_component')=='floor' and abs(values[1,2]-floor_z)>.02:
            raise SimBackendError('LOADED_FLOOR_DATUM_DIFFERS_FROM_EXPORT')
        values[:,2]-=floor_z
        return [(values[0].tolist(),values[1].tolist())]

    def symbolic_place(self, entity: str, receptacle: str,
                       at: list[float] | None = None) -> bool:
        """Release the benchmark-held ``entity`` onto ``receptacle``.

        The held object is passed explicitly by the benchmark core (never
        re-derived from an internal grasp dict); this is realization only.
        ``at`` is the support-surface point the feasibility check validated
        (the object is set down centred on it); without it the object goes on
        the first reachable surface point whose footprint is free.
        """
        from rummagebench.feasibility.interaction_target import (
            receptacle_place_targets,
        )

        obj = resolve_object(self._env.scene, entity)
        self._collision_body_cache = None  # the placed object moves
        # release whichever assisted-grasp joint holds this exact object
        ag = getattr(self._robot, "_ag_obj_in_hand", {})
        for arm in getattr(self._robot, "arm_names", []):
            if ag.get(arm) is obj:
                try:
                    # _release_grasp starts a delayed release window and keeps
                    # _ag_obj_in_hand populated. Symbolic PLACE is immediate.
                    self._robot.release_grasp_immediately(arm=arm)
                except Exception as e:
                    logger.warning("release on place failed for %s: %s", entity, e)
                    return False
        if self.is_holding(entity):
            return False
        points = [list(at)] if at is not None else []
        try:
            points += [] if at is not None else [t.metadata["surface_point"]
                       for t in receptacle_place_targets(receptacle, self)
                       if "surface_point" in t.metadata]
        except Exception as e:
            logger.warning("place surface points unavailable for %s: %s", receptacle, e)
        # the object is set down at identity orientation: size it by its
        # native extents, not by the AABB of its rotated in-hand pose
        half = self.place_half_extents(entity) or [0.1, 0.1, 0.05]
        others = []
        for name in self.entity_names():
            info = self.describe_entity(name)
            if name in (entity, receptacle) or info is None or info.fixed_base:
                continue
            if name in self.robot_entity_names():
                continue
            box = self.entity_aabb(name)
            if box is not None:
                others.append(box)

        def free(x, y, z):
            return not any(
                b[0][0] < x + half[0] and b[1][0] > x - half[0]
                and b[0][1] < y + half[1] and b[1][1] > y - half[1]
                and b[1][2] > z - 0.02 and b[0][2] < z + 2 * half[2] + 0.05
                for b in others)

        for i, (x, y, z) in enumerate(points):
            if i > 0 and not free(x, y, z):
                continue
            try:
                obj.set_position_orientation(
                    position=self.place_release_position(entity, [x, y, z]),
                    orientation=[0, 0, 0, 1],
                )
                import torch as th

                obj.set_linear_velocity(th.zeros(3))
                obj.set_angular_velocity(th.zeros(3))
                if getattr(obj, 'n_joints', 0):
                    # Reset internal motion inherited from carrying as well
                    # as root motion; keep the object's joint configuration.
                    obj.set_joint_velocities(th.zeros_like(obj.get_joint_velocities()))
            except Exception as e:
                logger.warning("place pose write failed for %s: %s", entity, e)
                return False
            self.settle(10)
            return True
        logger.warning("no free support point on %s for %s", receptacle, entity)
        return False

    def _repair_place_collision_meshes(self):
        """Repair a verified asset cooking defect before physics resumes.

        xbfgjc's base has a planar bottom in its authored mesh, but PhysX's
        convex cooking produces a sloping underside: the upright lamp tips
        even on an empty floor. SDF preserves the authored surface without
        adding support geometry or freezing the placed object. Keep this
        override model/mesh-specific; other assets retain native collisions.
        """
        from pxr import PhysxSchema, UsdPhysics, UsdGeom, Vt, Gf

        repairs = []
        with self._sim.editing_usd():
            for obj in self._env.scene.objects:
                if (getattr(obj,'category',None),getattr(obj,'model',None))==('shopping_basket','nedrsh'):
                    from rummagebench.sim.omnigibson.inside_volume import regularize_native_box_floor
                    from omnigibson.utils.transform_utils import quat2mat
                    mesh=obj.links['base_link'].collision_meshes.get('nedrsh__base_link__collision__0')
                    if mesh is None:raise SimBackendError('Audited native basket floor mesh is missing')
                    points=np.asarray(_arr(mesh.points_in_parent_frame),float).reshape(-1,3)
                    previous_thickness=float(np.ptp(points[:,2]))
                    if previous_thickness<.00001:
                        fixed=regularize_native_box_floor(points,np.asarray(_arr(mesh.faces),int).reshape(-1,3))
                        # Preserve the native local transform. Public pose/scale
                        # setters open their own USD context and cannot be nested
                        # inside this one. Transform repaired parent-frame points
                        # back to mesh-local coordinates and edit only vertices.
                        pos,quat=mesh.get_position_orientation(frame='parent')
                        rotation=np.asarray(_arr(quat2mat(quat)),float).reshape(3,3)
                        # Bake the existing nearly singular local mesh scale into
                        # the vertices. Keep the actual repaired parent-frame
                        # footprint and 2-mm thickness identical, but give
                        # PhysX a well-conditioned unit-scale collision mesh.
                        local=(fixed-np.asarray(_arr(pos)))@rotation
                        mesh.prim.GetAttribute('xformOp:scale').Set(Gf.Vec3d(1.,1.,1.))
                        mesh._cached_scale=None
                        repaired_mesh=UsdGeom.Mesh(mesh.prim)
                        repaired_mesh.GetPointsAttr().Set(Vt.Vec3fArray([tuple(float(v) for v in p) for p in local]))
                        # Keep authored USD bounds consistent with changed
                        # vertices, so boundingCube cannot reuse a flat extent.
                        repaired_mesh.CreateExtentAttr().Set(Vt.Vec3fArray([
                            tuple(float(v) for v in local.min(0)),tuple(float(v) for v in local.max(0))]))
                        UsdPhysics.CollisionAPI.Apply(mesh.prim).CreateCollisionEnabledAttr(True)
                        for key in ('points_and_faces','delaunay_triangulation'):mesh.__dict__.pop(key,None)
                        for key in ('collision_boundary_points_local','collision_mesh_cpu_data'):
                            obj.links['base_link'].__dict__.pop(key,None)
                        UsdPhysics.MeshCollisionAPI(mesh.prim).GetApproximationAttr().Set('boundingCube')
                    elif not .0019<=previous_thickness<=.0021:
                        raise SimBackendError('Native basket floor geometry changed; audited repair no longer applies')
                    repairs.append(dict(entity=obj.name,mesh=mesh.name,previous_thickness_m=previous_thickness,
                        thickness_m=.002,approximation='boundingCube',reason='regularize_degenerate_native_floor_inside_unchanged_exterior_bbox'))
                if (getattr(obj,'category',None),getattr(obj,'model',None))==('bottom_cabinet','jrhgeu'):
                    for mesh in obj.links['base_link'].collision_meshes.values():
                        collision=UsdPhysics.MeshCollisionAPI(mesh.prim)
                        previous=collision.GetApproximationAttr().Get()
                        print('[cabinet-collision-audit]',obj.name,mesh.name,previous,flush=True)
                        collision.GetApproximationAttr().Set('sdf')
                        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.prim).CreateSdfResolutionAttr(128)
                        repairs.append(dict(entity=obj.name,mesh=mesh.name,previous_approximation=previous,
                            approximation='sdf',reason='experimental_cabinet_cavity_preservation'))
                if (getattr(obj, 'category', None), getattr(obj, 'model', None)) != ('table_lamp', 'xbfgjc'):
                    continue
                link = obj.links.get('base_link')
                mesh = None if link is None else link.collision_meshes.get('xbfgjc__base_link__collision__0')
                if mesh is None:
                    raise SimBackendError('Known lamp collision repair mesh is missing')
                collision = UsdPhysics.MeshCollisionAPI(mesh.prim)
                previous = collision.GetApproximationAttr().Get()
                collision.GetApproximationAttr().Set('sdf')
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.prim).CreateSdfResolutionAttr(128)
                repairs.append({'entity': obj.name, 'mesh': mesh.name,
                                'previous_approximation': previous,
                                'approximation': 'sdf', 'resolution': 128,
                                'reason': 'convex_cooking_distorts_flat_support_base'})
        return repairs

    def place_release_position(self, entity, surface):
        """Identity-oriented object origin for a bbox centred above the click.

        DatasetObject's native bbox centre is not necessarily its root origin.
        Use the same signed, scaled offset as its bbox-centre placement API.
        This is shared with the PLACE feasibility descent, not a second point
        search or a relaxation of the postcondition.
        """
        obj = resolve_object(self._env.scene, entity)
        half = self.place_half_extents(entity)
        if half is None:
            raise SimBackendError('PLACE object bounds unavailable')
        offset = getattr(obj, 'scaled_bbox_center_in_base_frame', None)
        offset = np.zeros(3) if offset is None else np.asarray(_arr(offset), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise SimBackendError('PLACE object origin offset invalid')
        center = np.asarray(surface, dtype=float) + np.array([0., 0., half[2] + .01])
        return (center + offset).tolist()

    def place_half_extents(self, entity: str) -> list[float] | None:
        """Half extents (x, y, z) of ``entity`` at the identity orientation
        symbolic_place sets it down with (native bbox x scale); falls back to
        the current world AABB."""
        try:
            obj = resolve_object(self._env.scene, entity)
            ext = np.asarray(_arr(obj.native_bbox), dtype=float).reshape(-1)[:3]
            ext = ext * np.asarray(_arr(obj.scale), dtype=float).reshape(-1)[:3]
            if ext.shape == (3,) and np.all(np.isfinite(ext)) and np.all(ext > 0):
                return [float(v) / 2.0 for v in ext]
        except Exception as e:
            logger.debug("native bbox unavailable for %s: %s", entity, e)
        oa = self.entity_aabb(entity)
        if oa is None:
            return None
        return [(float(oa[1][i]) - float(oa[0][i])) / 2.0 for i in range(3)]

    def held_entity_name(self) -> str | None:
        """Name of the object attached to any hand (assisted grasp), or None."""
        ag = getattr(self._robot, "_ag_obj_in_hand", {}) or {}
        for arm in getattr(self._robot, "arm_names", []) or []:
            obj = ag.get(arm)
            if obj is not None:
                return getattr(obj, "name", None)
        return None

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

    def clicked_place_region(self, entity: str, surface):
        """Validate support at the visible click, without searching another site.

        Short downward rays check the centre and held footprint corners. This
        rejects vertical faces / unsupported edges, and cannot select a hidden
        drawer floor far below the visible clicked surface.
        """
        from omnigibson.utils.sampling_utils import raytest
        from rummagebench.feasibility.interaction_target import InteractionRegion
        p = np.asarray(surface, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            return None
        held = self.held_entity_name()
        if held is None:
            return None
        half = self.place_half_extents(held)
        if half is None:
            return None
        held_obj = resolve_object(self._env.scene, held)
        ignored = [self._robot.prim_path, held_obj.prim_path]
        ignored += [link.prim_path for item in (self._robot, held_obj)
                    for link in getattr(item, 'links', {}).values()]
        heights = []
        for dx, dy in [(0., 0.), *[(sx*half[0], sy*half[1]) for sx in (-1,1) for sy in (-1,1)]]:
            x, y = float(p[0]+dx), float(p[1]+dy)
            hit = raytest(start_point=[x,y,float(p[2]+.04)],
                          end_point=[x,y,float(p[2]-.04)], ignore_bodies=ignored)
            if not hit or not hit.get('hit'):
                return None
            owner = self._owning_object(hit.get('rigidBody') or hit.get('collision') or '')
            if owner is None or owner.name != entity or float(hit.get('normal',[0,0,0])[2]) < .85:
                return None
            heights.append(float(hit['position'][2]))
        if max(heights)-min(heights) > .015:
            return None
        # Preserve the clicked XY; only reconcile depth/render mesh roundoff.
        point=[float(p[0]),float(p[1]),heights[0]]
        region=InteractionRegion([point[0]-half[0]-.02,point[1]-half[1]-.02,point[2]],
                                 [point[0]+half[0]+.02,point[1]+half[1]+.02,point[2]],'top_surface')
        return point, region

    def receptacle_region(self, entity: str):
        """Support region. Fixed furniture: the top of its body (root link,
        so an opened drawer / door never extends the surface); other
        receptacles: top surface, inside volume when open."""
        from rummagebench.feasibility.interaction_target import InteractionRegion
        from rummagebench.sim.omnigibson.usd_collision import compute_link_aabb

        clicked = getattr(self, '_clicked_place', None)
        if clicked is not None and clicked[0] == entity:
            return clicked[2]

        obj = resolve_object(self._env.scene, entity)
        aabb = None
        if getattr(obj, "fixed_base", False):
            root = getattr(obj, "root_link", None)
            if root is not None and len(obj.links or {}) > 1:
                try:
                    box = compute_link_aabb(root)
                    if box is not None:
                        aabb = ([float(v) for v in box[0]], [float(v) for v in box[1]])
                except Exception as e:
                    logger.warning("root link AABB unavailable for %s: %s", entity, e)
            if aabb is None:
                aabb = self.entity_aabb(entity)
            if aabb is None:
                return None
            if self.is_open(entity):
                # an OPEN drawer with a native fillable volume is an inside
                # target (PLACE into the drawer); everything else keeps the
                # body top surface
                from rummagebench.feasibility.interaction_target import select_open_interior
                picked = select_open_interior(aabb[0], aabb[1], self.receptacle_interior_regions(entity))
                if picked is not None:
                    return _with_open_strip(picked[1], aabb)
            return InteractionRegion(lo=list(aabb[0]), hi=list(aabb[1]), kind="top_surface")
        aabb = self.entity_aabb(entity)
        if aabb is None:
            return None
        lo, hi = aabb
        if self.is_open(entity):
            return InteractionRegion(lo=lo, hi=hi, kind="inside_volume")
        return InteractionRegion(lo=lo, hi=[hi[0], hi[1], hi[2]], kind="top_surface")

    def receptacle_interior_regions(self, entity: str):
        """Interior (inside_volume) regions of FIXED furniture: one per moving
        link that carries a native fillable meta volume (drawers), as the
        world AABB of its fillable volume(s) (the cavity: floor to rim,
        inside the walls) — open or closed. Falls back to the link's own
        AABB, whose bottom is the front panel's, below the drawer floor
        (2026-10-19: a fork set down 2 cm above it was embedded in the floor
        and fell through). [] for movable objects and furniture without
        fillable links."""
        from rummagebench.feasibility.interaction_target import InteractionRegion

        obj = resolve_object(self._env.scene, entity)
        if not getattr(obj, "fixed_base", False):
            return []
        links = obj.links or {}
        out = []
        for name, link in links.items():
            if name.startswith("meta__") or name == getattr(obj, "root_link_name", None):
                continue
            volumes = [l for n, l in links.items()
                       if n.startswith(f"meta__{name}_") and "fillable" in n]
            if not volumes:
                continue
            lo = hi = None
            try:
                boxes = [[np.asarray(_arr(v), dtype=float) for v in vol.visual_aabb]
                         for vol in volumes]
                lo = np.min([b[0] for b in boxes], axis=0)
                hi = np.max([b[1] for b in boxes], axis=0)
            except Exception as e:
                logger.warning("fillable volume AABB unavailable for %s:%s: %s", entity, name, e)
            if lo is None or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
                                  and np.all(hi > lo)):
                try:
                    lo, hi = (np.asarray(_arr(v), dtype=float) for v in link.aabb)
                except Exception as e:
                    logger.warning("interior AABB unavailable for %s:%s: %s", entity, name, e)
                    continue
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)):
                continue
            out.append((name, InteractionRegion(lo=lo.tolist(), hi=hi.tolist(), kind="inside_volume")))
        return out

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
        # same frame as the kinematics model (R1Pro: right, not arm_names[0])
        arm = self._modelled_arm() or arms[0]
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
        from omnigibson.object_states import Open

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
                if self._is_inside(obj, container):
                    hidden = True
                    break
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
        except (Exception, SystemExit):
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

    def _camera_sensor(self, sensor_name: str | None = None):
        """Resolve the same RGB sensor as capture, never enumeration order."""
        if sensor_name is None:
            from rummagebench.sim.omnigibson.observation import select_head_rgb_sensor
            observations,_=self._env.get_obs()
            sensor_name,_=select_head_rgb_sensor(observations[0].get(self._robot.name,{}))
        return self._robot.sensors.get(sensor_name)

    def _camera_prim(self, sensor_name: str | None = None):
        """The VisionSensor's USD camera prim (for intrinsics/extrinsics)."""
        try:
            sensor=self._camera_sensor(sensor_name)
            sensors=[] if sensor is None else [sensor]
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
            sensor=self._camera_sensor(sensor_name)
            sensors=[] if sensor is None else [sensor]
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
