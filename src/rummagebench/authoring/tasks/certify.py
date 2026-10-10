"""GPU certification of one compiled task.

Fills SimulatorEvidence for a TaskPlan's SearchStructureCertificate from a
live simulator, then applies gates.certify(). Every measurement goes
through production code paths:

  1. builder            OmniGibsonBackend.setup (+ verify_predicates); the
                        target must sit in its planned link (drawer) or in
                        no drawer (cabinet interior)
  2. protocol witness   a VisualProtocolSession episode driven ONLY by the
                        public 8-skill protocol: MOVE/TURN route (planned by
                        navigation.py, executed with the session's own swept
                        checks), OPEN / GRASP by clicking a point on the
                        current RGB frame, REPORT_DONE. Its step count is the
                        certified execution length (16-step gate).
     along the way:     start visibility; at the interaction pose: target
                        visibility + GRASP feasibility BEFORE the reveal and
                        AFTER it (the causal-reveal evidence)
     viewpoint:         after the reveal the witness, like an agent, moves to
                        a pose that sees and can grasp the REAL target: the
                        evaluator searches the furniture's interaction poses
                        counterfactually (OBSERVE's capture/restore, no steps
                        spent) and drives to the cheapest one by MOVE/TURN
  3. oracle             evaluation.certification.certify_episode (full-
                        information d*) + plan replay through the production
                        BenchmarkSession (certify_split.py's replay gate)

Visibility = renderer-instance pixels / isolated reference pixels from the
same pose (sim/omnigibson/visibility.py). A target outside the view counts
as 0 visible, never as "undefined = pass".
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from rummagebench.authoring.tasks.navigation import FootprintChecker, plan_route
from rummagebench.authoring.tasks.plan import TaskPlan
from rummagebench.authoring.tasks.search_certificate import SimulatorEvidence
from rummagebench.authoring.tasks.slots import SceneSlots

logger = logging.getLogger(__name__)

# v2: viewpoint search for the real target, route slack
# v3 (2026-10-06): the viewpoint search also walks up to the REAL target
#     (ring of poses facing its measured position, pre-filtered by projecting
#     its AABB through the head camera) — an agent approaches what it sees;
#     v2 only tried the furniture's front poses, so a target at the far end of
#     a long counter or the back of a drawer was "unreachable" (A.9: 61/135
#     rejections)
# v4 (2026-10-09, medium only; easy path unchanged): before any cover is
#     moved the target must be PHYSICALLY ungraspable from every tried pose
#     (visibility not required); the oracle plan's moved-object count is
#     recorded (oracle_rearrangement_depth) and gated (A.12)
CERTIFIER_VERSION = 4
BASE_HALF_EXTENT = 0.40
BASE_MARGIN = 0.03
# planning slack beyond the session's margin: the physics settle after every
# MOVE / TURN drifts the base by a few cm
ROUTE_SLACK = 0.02
MAX_VIEW_TRIES = 16
# ring around the target: base-centre distances and bearings
VIEW_RING_DISTANCES_M = tuple(round(0.5 + 0.1 * i, 2) for i in range(11))  # 0.5-1.5
VIEW_RING_BEARINGS_DEG = tuple(range(0, 360, 15))
VIEW_PREFERRED_DISTANCE_M = 1.0  # tie-break among equally cheap poses
# base-centre-to-target distances where GRASP is plausible (fb_701c96: the
# oracle grasped a counter-top notebook from 1.01 m; the furniture-front
# poses at 1.3 m all saw it but none could grasp it)
VIEW_REACH_BAND_M = (0.55, 1.25)
MAX_VIEW_ROUTE_PLANS = 80


def planar_T(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def box_corners(aabb) -> np.ndarray:
    lo, hi = (np.asarray(v, float) for v in aabb)
    return np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]]
                     for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def predicts_in_view(K, T_base_cam, pose, z: float, aabb, width: int, height: int,
                     margin_px: float = 4.0) -> bool:
    """Every corner of the target AABB projects inside the image from base
    pose (x, y, yaw) — a cheap geometric pre-filter (no occlusion: the
    rendered visibility stays the measurement)."""
    from rummagebench.authoring.tasks.probe import points_in_view

    T = planar_T(pose[0], pose[1], z, pose[2]) @ np.asarray(T_base_cam, float)
    return points_in_view(K, T, width, height, box_corners(aabb), margin_px)


def target_ring(target_xy, distances=VIEW_RING_DISTANCES_M,
                bearings_deg=VIEW_RING_BEARINGS_DEG) -> list[tuple[float, float, float]]:
    """Base poses at each distance / bearing from the target, facing it."""
    out = []
    for d in distances:
        for b in bearings_deg:
            a = math.radians(b)
            out.append((target_xy[0] - d * math.cos(a), target_xy[1] - d * math.sin(a),
                        math.atan2(math.sin(a), math.cos(a))))
    return out


def click_point(frame, backend, entity: str, patch: int = 11) -> Optional[tuple[float, float]]:
    """Normalized point on ``entity`` in the private frame: the pixel of its
    largest instance with the most own pixels in a patch x patch window
    (same snap rule as the human-play assist). None = not in view."""
    seg = np.asarray(frame.instance_segmentation)
    labels = frame.meta.get("instance_labels", {})
    insts = [int(k) for k, v in labels.items() if v == entity]
    insts = [i for i in insts if (seg == i).any()]
    if not insts:
        return None
    inst = max(insts, key=lambda i: int((seg == i).sum()))
    mask = seg == inst
    H, W = mask.shape
    r = patch // 2
    c = np.pad(np.pad(mask.astype(np.int32), r).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    win = c[patch:patch + H, patch:patch + W] - c[:H, patch:patch + W] \
        - c[patch:patch + H, :W] + c[:H, :W]
    win[~mask] = -1
    v, u = np.unravel_index(int(np.argmax(win)), win.shape)
    return (round((u + 0.5) / W, 5), round((v + 0.5) / H, 5))


def _vis_value(m: dict) -> float:
    """Visibility ratio with out-of-view (empty reference) = 0.0."""
    return 0.0 if m["ratio"] is None else float(m["ratio"])


@dataclass
class EpisodeCertification:
    evidence: SimulatorEvidence = field(default_factory=SimulatorEvidence)
    details: dict[str, Any] = field(default_factory=dict)
    oracle_certificate: Optional[dict[str, Any]] = None
    witness_trace: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class TaskCertifier:
    """Runs inside a simulator process; one episode per backend.setup()."""

    def __init__(self, scene_slots: SceneSlots, overlay, room_map,
                 log: Callable[[str], None] = print, reveal_min: float = 0.65):
        self.ss = scene_slots
        self.overlay = overlay
        self.room_map = room_map
        self.log = log
        # visibility a viewpoint must reach to count (the tier's generator
        # acceptance margin, gates.simulator_violations)
        self.reveal_min = float(reveal_min)

    # ------------------------------------------------------------------ run

    def certify(self, plan: TaskPlan, cert, scenario_path) -> EpisodeCertification:
        from rummagebench.adapters.python_api import (
            install_renderer_grounding, prepare_renderer_grounding)
        from rummagebench.authoring.validation import verify_predicates
        from rummagebench.core.scenario import load_scenario
        from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

        out = EpisodeCertification()
        ev = out.evidence
        self._view_pose = None  # witness viewpoint, offered to the oracle as an anchor
        self._target_slot_id = plan.target.slot_id
        ev.reachable_candidate_slot_count = self._reachable_candidates(cert.candidate_slots)
        scenario = load_scenario(scenario_path)
        prepare_renderer_grounding(scenario)
        # validate every anchor at build time: the oracle's NAV only uses
        # build-time-verified anchors
        backend = OmniGibsonBackend(seed=0, validate_legacy_anchors=True)
        t0 = time.time()
        try:
            try:
                report = backend.setup(scenario)
            except Exception as e:
                ev.builder_ok = False
                out.error = f"builder: {type(e).__name__}: {e}"
                return out
            ev.builder_ok = True
            verdict = verify_predicates(scenario, report)
            out.details["build_verification"] = verdict
            link_ok, where = self._target_link_ok(backend, plan)
            out.details["target_containing_link"] = where
            ev.placements_verified = bool(verdict.get("passed")) and link_ok
            install_renderer_grounding(backend)
            out.details["build_seconds"] = round(time.time() - t0, 1)

            self._witness(backend, scenario, plan, out)
            # medium: a target graspable under its cover is rejected whatever
            # the oracle says; skip the (long) oracle search
            premature = plan.mode in ("buried", "covered") and ev.pre_reveal_graspable
            if ev.settle_ok is not False and not premature:
                self._oracle(backend, scenario, out)
        except Exception as e:
            logger.exception("certification of %s failed", plan.task_id)
            out.error = f"{type(e).__name__}: {e}"
        finally:
            # no backend.close() here: OmniGibson's shutdown terminates the
            # process; the caller writes the record first, then exits
            out.details["seconds"] = round(time.time() - t0, 1)
        return out

    # ------------------------------------------------------------ helpers

    def _reachable_candidates(self, candidate_slots: list[str]) -> int:
        by_id = self.ss.by_id()
        n = 0
        for sid in candidate_slots:
            inter = self.overlay.slots.get(sid)
            if inter is not None and inter.usable(by_id[sid].requires_open):
                n += 1
        return n

    def _target_link_ok(self, backend, plan: TaskPlan) -> tuple[bool, Optional[str]]:
        from rummagebench.sim.omnigibson.probe_support import containing_link

        slot = self.ss.by_id()[plan.target.slot_id]
        if slot.relation != "inside":
            return True, None
        # the builder leaves containers in their initial (closed) state; the
        # link test works on the closed geometry too
        where = containing_link(backend, slot.parent_entity, plan.target.entity)
        if slot.link is not None:
            return where == slot.link, where
        drawers = {s.link for s in self.ss.slots
                   if s.parent_entity == slot.parent_entity and s.slot_type == "drawer"}
        return where not in drawers, where

    def _feasible(self, session, skill: str, entity: str) -> bool:
        resolved = session._backend.resolve_entity(entity)
        return bool(session._feasibility.check(skill, resolved, session.robot,
                                               session.world_state).feasible)

    # ------------------------------------------------------------ witness

    def _witness(self, backend, scenario, plan: TaskPlan, out: EpisodeCertification) -> None:
        from rummagebench.core.visual_session import VisualProtocolSession
        from rummagebench.sim.omnigibson.probe_support import relation_holds
        from rummagebench.sim.omnigibson.visibility import measure_visibility

        ev = out.evidence
        target = plan.target.entity
        slot = self.ss.by_id()[plan.target.slot_id]
        vs = VisualProtocolSession(backend, scenario, base_half_extent=BASE_HALF_EXTENT)
        vs.reset()
        try:
            backend.validate_physics_state()
            ev.settle_ok = relation_holds(backend, target, slot.parent_entity, slot.relation)
        except Exception as e:
            ev.settle_ok = False
            out.details["settle_error"] = str(e)
            return
        if not ev.settle_ok:
            out.details["settle_error"] = "target relation lost after reset"
            return
        # the compiler closes every openable furniture of the room; overfilled
        # drawers can be pushed open by settling, which changes the episode
        # (contents exposed, base paths blocked) -> not the planned task
        opened = sorted(f.entity for f in self.ss.furniture
                        if f.room == plan.room and f.openable and backend.is_open(f.entity))
        out.details["opened_after_reset"] = opened
        if opened:
            ev.settle_ok = False
            out.details["settle_error"] = f"furniture open after settling: {opened}"
            return
        inner = vs._session  # same backend / feasibility engine as the agent

        start = measure_visibility(backend, target).to_dict()
        out.details["start_visibility"] = start
        ev.start_visibility = _vis_value(start)
        on_surface = not slot.requires_open
        if on_surface:
            # on_surface: the reveal is the viewpoint change itself
            ev.pre_reveal_graspable = self._feasible(inner, "GRASP", target)

        def step(action: dict) -> str:
            res = vs.step(action)
            rec = {"action": action, "feedback": res.feedback.code.value,
                   "status": res.episode_status, "step": res.planning_step}
            out.witness_trace.append(rec)
            return rec["feedback"]

        if not self._drive(vs, backend, scenario, plan, slot.parent_entity, step, out):
            return
        if not on_surface:
            closed = measure_visibility(backend, target).to_dict()
            out.details["closed_visibility"] = closed
            ev.closed_visibility = 0.0 if closed["visible_px"] == 0 else _vis_value(closed)
            ev.pre_reveal_graspable = self._feasible(inner, "GRASP", target)
            if not self._click(vs, backend, "OPEN", slot.parent_entity, step, out):
                return
        if plan.mode in ("buried", "covered") and not self._uncover(vs, backend, inner, scenario, plan,
                                                       slot, target, step, out):
            return

        # find a viewpoint for the REAL target (the agent may step back, turn,
        # sidestep): cheapest MOVE/TURN-reachable pose that sees + grasps it
        view = self._find_view_pose(backend, inner, scenario, plan, slot, target, out)
        if view is None:
            out.details["witness_failure"] = ("no MOVE/TURN-reachable pose sees and can "
                                              "grasp the target")
        elif view != "here":
            if not self._drive_pose(vs, backend, plan, view, "view", step, out):
                return
            self._view_pose = view
        revealed = measure_visibility(backend, target).to_dict()
        out.details["revealed_visibility"] = revealed
        ev.revealed_visibility = _vis_value(revealed)
        if view is None:
            return
        feasible_now = self._feasible(inner, "GRASP", target)
        grasped = self._click(vs, backend, "GRASP", target, step, out)
        ev.post_reveal_graspable = feasible_now and grasped
        if not grasped:
            return
        # objects moved before the target became graspable (0: easy)
        ev.rearrangement_depth = int(out.details.get("rearrangement_removed", 0))
        if step({"skill": "REPORT_DONE"}) == "EXECUTED" and vs.status() == "SUCCESS":
            ev.certified_execution_steps = len(out.witness_trace)
        else:
            out.details["witness_failure"] = f"REPORT_DONE -> {vs.status()}"

    # ------------------------------------------------- medium: rearrangement

    def _uncover(self, vs, backend, inner, scenario, plan: TaskPlan, slot, target: str,
                 step, out: EpisodeCertification) -> bool:
        """Medium buried / covered (tiers/medium.py): after the reveal (OPEN,
        or reaching the furniture), the target must
        NOT be graspable from any reachable viewpoint (§11); the witness then
        removes the covers top-down like an agent — view + GRASP the cover,
        PLACE it on a reachable surface — until the target is graspable. The
        number moved is the measured rearrangement depth."""
        covers = [o.entity for o in plan.distractors if o.role == "cover"][::-1]
        pre = self._find_view_pose(backend, inner, scenario, plan, slot, target, out,
                                   need_visible=False)
        out.details["covered_view_search"] = out.details.pop("view_search", None)
        if pre is not None:
            out.evidence.pre_reveal_graspable = True
            out.details["witness_failure"] = "target graspable before any cover is moved"
            return False
        removed = 0
        for c in covers:
            v = self._find_view_pose(backend, inner, scenario, plan, slot, c, out)
            out.details[f"cover_{removed}_view_search"] = out.details.pop("view_search", None)
            if v is None:
                out.details["witness_failure"] = f"cover {c}: no pose sees and can grasp it"
                return False
            if v != "here" and not self._drive_pose(vs, backend, plan, v, f"cover_{removed}",
                                                    step, out):
                return False
            if not self._click(vs, backend, "GRASP", c, step, out):
                return False
            if not self._place_held(vs, backend, inner, scenario, plan, step, out, removed):
                return False
            removed += 1
            if removed < len(covers):  # enough already? (counterfactual, no steps)
                done = self._find_view_pose(backend, inner, scenario, plan, slot, target, out)
                out.details.pop("view_search", None)
                if done is not None:
                    break
        out.details["rearrangement_removed"] = removed
        return True

    def _place_held(self, vs, backend, inner, scenario, plan: TaskPlan, step,
                    out: EpisodeCertification, k: int) -> bool:
        """Cheapest (pose, receptacle) where PLACE of the held object is
        feasible and the receptacle is clickable: the current pose or any
        furniture anchor of the room; evaluated counterfactually, then driven
        to and clicked."""
        from copy import deepcopy

        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.skills.move import quat_from_yaw, yaw_from_quat
        from rummagebench.sim.omnigibson.visibility import visible_entities_in_view

        room_furn = sorted(f.entity for f in self.ss.furniture if f.room == plan.room)
        pos, quat = backend.robot_pose()
        here = (float(pos[0]), float(pos[1]), yaw_from_quat(quat))
        poses = [(0, "here", here)]
        lim = self._limits_from(scenario)
        for name in room_furn:
            a = scenario.anchors.get(name)
            if a is None:
                continue
            g = (a.position[0], a.position[1], yaw_from_quat(list(a.orientation)))
            route, _, _ = self._plan(backend, plan, here, g, lim)
            if route is not None:
                poses.append((route.steps, name, g))
        poses.sort(key=lambda p: p[0])
        snapshot = deepcopy(backend.capture_observe_state())
        tried, found = [], None
        try:
            for steps, key, (x, y, yaw) in poses:
                if key != "here":
                    backend.teleport_robot(AnchorSpec(position=[x, y, pos[2]],
                                                      orientation=quat_from_yaw(yaw)))
                    backend.settle(5)
                seen = visible_entities_in_view(backend, 300)
                for r in room_furn:
                    if r not in seen:
                        continue
                    ok = self._feasible(inner, "PLACE", r)
                    tried.append({"pose": key, "steps": steps, "receptacle": r, "place": ok})
                    if ok:
                        found = (steps, key, (x, y, yaw), r)
                        break
                backend.restore_observe_state(deepcopy(snapshot))
                if found:
                    break
        finally:
            backend.restore_observe_state(deepcopy(snapshot))
        out.details[f"place_search_{k}"] = tried
        if found is None:
            out.details["witness_failure"] = "no reachable pose can PLACE the held cover"
            return False
        _, key, goal, r = found
        if key != "here" and not self._drive_pose(vs, backend, plan, goal, f"place_{k}",
                                                  step, out):
            return False
        return self._click(vs, backend, "PLACE", r, step, out)

    def _find_view_pose(self, backend, inner, scenario, plan: TaskPlan, slot, target: str,
                        out: EpisodeCertification, need_visible: bool = True):
        """Cheapest MOVE/TURN-reachable pose (current pose, the probe's reveal
        pose, or any interaction-anchor candidate of the furniture) from which
        the REAL target is visible (>= reveal_min) and GRASP is feasible.
        'here' = the current pose; None = no such pose. need_visible=False
        asks only for PHYSICAL graspability (medium's covered target: an
        invisible target the full-information oracle can still side-grasp
        under its cover is not buried, A.12). Poses are evaluated
        counterfactually with OBSERVE's capture/restore (no physics steps,
        no agent steps); only the chosen route is then executed."""
        from copy import deepcopy

        from rummagebench.authoring.tasks.compile import reveal_anchor_name
        from rummagebench.authoring.tasks.probe import ProbeConfig, anchor_candidates
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.skills.move import quat_from_yaw, yaw_from_quat
        from rummagebench.sim.omnigibson.visibility import measure_visibility

        pos, quat = backend.robot_pose()
        here = (float(pos[0]), float(pos[1]), yaw_from_quat(quat))
        cands = []
        rv = scenario.anchors.get(reveal_anchor_name(slot.parent_entity))
        if rv is not None:
            cands.append((rv.position[0], rv.position[1], yaw_from_quat(list(rv.orientation))))
        furn = self.ss.furniture_by_entity().get(slot.parent_entity)
        if furn is not None:
            cands += anchor_candidates(furn, ProbeConfig())
        # stepping straight back along the current heading: what an agent does
        # when an opened drawer pins it (turning in place is then blocked)
        cands += [(here[0] - d * math.cos(here[2]), here[1] - d * math.sin(here[2]), here[2])
                  for d in np.arange(0.05, 1.0001, 0.05)]
        lim = self._limits_from(scenario)
        # walk up to the real target: a ring of poses facing it, kept when
        # the footprint is free and the target's AABB projects into the view
        taabb = backend.entity_aabb(target)
        sees = None
        if taabb is not None:
            tc = [(taabb[0][i] + taabb[1][i]) / 2.0 for i in range(3)]
            fr = backend.capture_visual_frame()
            T_base_cam = np.linalg.inv(planar_T(here[0], here[1], float(pos[2]), here[2])) \
                @ np.asarray(fr.camera_extrinsics, float)

            def sees(c):
                return predicts_in_view(fr.camera_intrinsics, T_base_cam, c, float(pos[2]),
                                        taabb, fr.image_width, fr.image_height)
            ring = target_ring(tc[:2])
            chk = FootprintChecker.from_backend(backend, BASE_HALF_EXTENT,
                                                BASE_MARGIN + ROUTE_SLACK)
            free = chk.free_many(*(np.array(v) for v in zip(*ring)))
            ring = [c for c, ok in zip(ring, free) if ok and sees(c)]
            ring.sort(key=lambda c: abs(math.hypot(c[0] - tc[0], c[1] - tc[1])
                                        - VIEW_PREFERRED_DISTANCE_M))
            cands += ring[:MAX_VIEW_ROUTE_PLANS]
            out.details["view_target_aabb"] = [list(map(float, taabb[0])),
                                               list(map(float, taabb[1]))]
        ranked = [(0, "here", here)]
        for c in cands:
            route, _, _ = self._plan(backend, plan, here, tuple(c), lim)
            if route is not None:
                ranked.append((route.steps, tuple(float(v) for v in c), tuple(c)))

        def plausible(c):  # target predicted in view AND within arm's reach band
            if taabb is None:
                return True
            d = math.hypot(c[0] - tc[0], c[1] - tc[1])
            return VIEW_REACH_BAND_M[0] <= d <= VIEW_REACH_BAND_M[1] and sees(c)

        def rank(r):  # "here" first, then plausible poses, cheapest, nearest 1 m
            steps, key, c = r
            dist = 0.0 if taabb is None else abs(
                math.hypot(c[0] - tc[0], c[1] - tc[1]) - VIEW_PREFERRED_DISTANCE_M)
            return (key != "here", not plausible(c), steps, round(dist, 2))
        ranked.sort(key=rank)
        seen_keys, uniq = set(), []
        for r in ranked:  # one try per distinct pose
            k = tuple(round(v, 3) for v in r[2])
            if k not in seen_keys:
                seen_keys.add(k)
                uniq.append(r)
        ranked = uniq
        if sees is not None:  # out-of-view poses cannot pass; keep "here" (measured)
            ranked = [r for r in ranked if r[1] == "here" or sees(r[2])] or ranked[:1]
        if len(ranked) == 1:  # replayable on CPU, like route_debug
            chk = FootprintChecker.from_backend(backend, BASE_HALF_EXTENT, BASE_MARGIN)
            out.details["view_debug"] = {
                "here": list(here), "candidates": [list(map(float, c)) for c in cands],
                "half_extent": BASE_HALF_EXTENT, "margin_applied": BASE_MARGIN,
                "boxes_xy_inflated": [[*map(float, a), *map(float, b)]
                                      for a, b in zip(chk.lo, chk.hi)],
                "waypoints": [list(w) for w in self.room_map.free_points(plan.room, 0.4, 0.25)]}
        snapshot = deepcopy(backend.capture_observe_state())
        tried = []
        found = None
        try:
            for steps, key, (x, y, yaw) in ranked[:MAX_VIEW_TRIES]:
                if key != "here":
                    backend.teleport_robot(AnchorSpec(position=[x, y, pos[2]],
                                                      orientation=quat_from_yaw(yaw)))
                    backend.settle(5)  # as after a real MOVE / TURN
                vis = _vis_value(measure_visibility(backend, target).to_dict())
                grasp = ((not need_visible or vis >= self.reveal_min)
                         and self._feasible(inner, "GRASP", target))
                tried.append({"pose": [round(x, 3), round(y, 3), round(yaw, 3)],
                              "route_steps": steps, "visibility": round(vis, 3),
                              "grasp": grasp})
                backend.restore_observe_state(deepcopy(snapshot))
                if grasp:
                    found = key
                    break
        finally:
            backend.restore_observe_state(deepcopy(snapshot))
        out.details["view_search"] = {"candidates_reachable": len(ranked), "tried": tried}
        return found

    def _limits_from(self, scenario):
        from rummagebench.authoring.tasks.navigation import ProtocolLimits

        proto = scenario.agent_protocol
        return ProtocolLimits(proto.min_move_cm / 100.0, proto.max_move_cm / 100.0,
                              proto.min_turn_deg, proto.max_turn_deg)

    def _drive(self, vs, backend, scenario, plan: TaskPlan, anchor_name: str, step,
               out: EpisodeCertification) -> bool:
        from rummagebench.skills.move import yaw_from_quat

        anchor = scenario.anchors[anchor_name]
        return self._drive_pose(vs, backend, plan,
                                (anchor.position[0], anchor.position[1],
                                 yaw_from_quat(list(anchor.orientation))),
                                anchor_name, step, out)


    def _plan(self, backend, plan: TaskPlan, start, goal, lim):
        """Route planned with drift slack first, the session's exact margin
        as the fallback (CURRENT obstacles, e.g. an opened drawer)."""
        wps = self.room_map.free_points(plan.room, 0.4, 0.25)
        chk = None
        for margin in (BASE_MARGIN + ROUTE_SLACK, BASE_MARGIN):
            chk = FootprintChecker.from_backend(backend, BASE_HALF_EXTENT, margin)
            route = plan_route(chk, start, goal, wps, lim)
            if route is not None:
                return route, chk, wps
        return None, chk, wps

    def _drive_pose(self, vs, backend, plan: TaskPlan, goal, label: str, step,
                    out: EpisodeCertification) -> bool:
        """Public MOVE / TURN route from the current pose to goal (x, y, yaw)."""
        from rummagebench.skills.move import yaw_from_quat

        anchor_name = label
        pos, quat = backend.robot_pose()
        route, chk, wps = self._plan(backend, plan, (pos[0], pos[1], yaw_from_quat(quat)),
                                     tuple(goal), self._limits_from(vs._scenario))
        if route is None:
            out.details["witness_failure"] = f"no MOVE/TURN route to {anchor_name}"
            # replayable on CPU: FootprintChecker(boxes, half, margin=0)
            out.details["route_debug"] = {
                "anchor": anchor_name, "start": [pos[0], pos[1], yaw_from_quat(quat)],
                "goal": [float(goal[0]), float(goal[1]), float(goal[2])],
                "half_extent": BASE_HALF_EXTENT, "margin_applied": BASE_MARGIN,
                "boxes_xy_inflated": [[*map(float, a), *map(float, b)]
                                      for a, b in zip(chk.lo, chk.hi)],
                "waypoints": [list(w) for w in wps]}
            return False
        out.details.setdefault("routes", {})[anchor_name] = {
            "steps": route.steps, "waypoints": route.waypoints, "final": route.final}
        for a in route.actions:
            if step(a) != "EXECUTED":
                out.details["witness_failure"] = f"route to {anchor_name}: {a} not executed"
                return False
        return True

    def _click(self, vs, backend, skill: str, entity: str, step, out) -> bool:
        frame = vs._store.get(vs._store.current_id)
        pt = click_point(frame, backend, entity)
        if pt is None:
            out.details["witness_failure"] = f"{skill}: {entity} not in the current frame"
            return False
        fb = step({"skill": skill, "point": {"frame_id": frame.frame_id,
                                              "x": pt[0], "y": pt[1]}})
        if fb != "EXECUTED":
            out.details["witness_failure"] = f"{skill}({entity}) -> {fb}"
            return False
        return True

    # ------------------------------------------------------------- oracle

    def _register_view_anchor(self, backend, scenario, out: EpisodeCertification) -> None:
        """The full-information oracle navigates over the scenario's
        evaluator-private anchors only (§23). When the witness had to walk to
        a viewpoint that no anchor covers (e.g. the far end of a long
        counter), that pose becomes the anchor ``<furniture>__view`` —
        validated exactly like the builder validates anchors (teleport into
        the reset world, settle, stay within 0.5 m) — so d* is computed with
        the same reachable viewpoint. The worker writes it into the candidate
        scenario, so replaying the certified file reproduces the certificate."""
        from rummagebench.authoring.tasks.compile import VIEW_ANCHOR_SUFFIX
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.skills.move import quat_from_yaw, yaw_from_quat

        if self._view_pose is None:
            return
        x, y, yaw = (float(v) for v in self._view_pose)
        for a in scenario.anchors.values():  # already an anchor (e.g. the reveal pose)
            if (abs(a.position[0] - x) < 1e-3 and abs(a.position[1] - y) < 1e-3
                    and abs(math.remainder(yaw_from_quat(list(a.orientation)) - yaw,
                                           2 * math.pi)) < 1e-3):
                return
        z = float(scenario.anchors[scenario.robot.init_anchor].position[2])
        anchor = AnchorSpec(position=[round(x, 4), round(y, 4), round(z, 4)],
                            orientation=[round(v, 6) for v in quat_from_yaw(yaw)])
        slot = self.ss.by_id()[self._target_slot_id]
        name = f"{slot.parent_entity}{VIEW_ANCHOR_SUFFIX}"
        backend.reset()
        backend.teleport_robot(anchor)
        backend.settle(5)
        backend.validate_physics_state()
        pos, _ = backend.robot_pose()
        dist = float(np.linalg.norm(np.asarray(pos) - np.asarray(anchor.position)))
        backend.reset()
        rec = {"name": name, "position": anchor.position,
               "orientation": anchor.orientation, "validation_distance": round(dist, 4)}
        out.details["view_anchor"] = rec
        if dist > 0.5:
            rec["rejected"] = True
            return
        scenario.anchors[name] = anchor
        backend._validated_anchor_poses[name] = (tuple(anchor.position),
                                                 tuple(anchor.orientation))

    def _oracle(self, backend, scenario, out: EpisodeCertification) -> None:
        from rummagebench.core.session import BenchmarkSession
        from rummagebench.core.types import Action
        from rummagebench.evaluation.certification import certify_episode

        ev = out.evidence
        self._register_view_anchor(backend, scenario, out)
        session = BenchmarkSession(backend, scenario)
        session.reset()
        t0 = time.time()
        c = certify_episode(scenario, session)
        if c.solvable and c.oracle_plan:
            session.reset()
            for a in c.oracle_plan:
                session.act(Action.from_dict(a))
            c.plan_replay_status = session.status().value
            c.plan_replay_steps = session._planning_step
            if c.plan_replay_status != "SUCCESS":
                c.solvable = False
                c.reason = "PLAN_REPLAY_FAILED"
        ev.oracle_solvable = bool(c.solvable)
        ev.oracle_depth = c.oracle_depth
        if c.solvable and c.oracle_plan:
            ev.oracle_rearrangement_depth = sum(
                1 for a in c.oracle_plan
                if a.get("skill") == "GRASP"
                and (a.get("target") or {}).get("value") != scenario.target.entity)
        ev.replay_success = c.plan_replay_status == "SUCCESS"
        out.oracle_certificate = c.to_dict()
        out.details["oracle_seconds"] = round(time.time() - t0, 1)
