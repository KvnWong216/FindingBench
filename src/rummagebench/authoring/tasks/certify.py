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

CERTIFIER_VERSION = 2
BASE_HALF_EXTENT = 0.40
BASE_MARGIN = 0.03
# planning slack beyond the session's margin: the physics settle after every
# MOVE / TURN drifts the base by a few cm
ROUTE_SLACK = 0.02
MAX_VIEW_TRIES = 8


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
            if ev.settle_ok is not False:
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
            if inter is not None and inter.usable(by_id[sid].requires_open, False):
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

        # find a viewpoint for the REAL target (the agent may step back, turn,
        # sidestep): cheapest MOVE/TURN-reachable pose that sees + grasps it
        view = self._find_view_pose(backend, inner, scenario, plan, slot, target, out)
        if view is None:
            out.details["witness_failure"] = ("no MOVE/TURN-reachable pose sees and can "
                                              "grasp the target")
        elif view != "here":
            if not self._drive_pose(vs, backend, plan, view, "view", step, out):
                return
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
        ev.rearrangement_depth = 0  # reached without moving any other object
        if step({"skill": "REPORT_DONE"}) == "EXECUTED" and vs.status() == "SUCCESS":
            ev.certified_execution_steps = len(out.witness_trace)
        else:
            out.details["witness_failure"] = f"REPORT_DONE -> {vs.status()}"

    def _find_view_pose(self, backend, inner, scenario, plan: TaskPlan, slot, target: str,
                        out: EpisodeCertification):
        """Cheapest MOVE/TURN-reachable pose (current pose, the probe's reveal
        pose, or any interaction-anchor candidate of the furniture) from which
        the REAL target is visible (>= reveal_min) and GRASP is feasible.
        'here' = the current pose; None = no such pose. Poses are evaluated
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
        ranked = [(0, "here", here)]
        for c in cands:
            route, _, _ = self._plan(backend, plan, here, tuple(c), lim)
            if route is not None:
                ranked.append((route.steps, tuple(float(v) for v in c), tuple(c)))
        ranked.sort(key=lambda r: r[0])
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
                grasp = vis >= self.reveal_min and self._feasible(inner, "GRASP", target)
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

    def _oracle(self, backend, scenario, out: EpisodeCertification) -> None:
        from rummagebench.core.session import BenchmarkSession
        from rummagebench.core.types import Action
        from rummagebench.evaluation.certification import certify_episode

        ev = out.evidence
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
        ev.replay_success = c.plan_replay_status == "SUCCESS"
        out.oracle_certificate = c.to_dict()
        out.details["oracle_seconds"] = round(time.time() - t0, 1)
