"""Privileged scripted driver for data-driven fetch / place scenarios.

Evaluator-side ONLY (like episode_log.ScriptedPixelAgent): it reads the
backend (object poses, feasibility, segmentation) to produce PUBLIC actions
(MOVE / TURN, or MOVE_TO under the 2026-10-26 protocol / point-targeted
OPEN / GRASP / PLACE / REPORT_DONE) that go
through the real VisualProtocolSession. It establishes "Scripted Execution
Success" for a scenario — that the task is solvable through the public
protocol within the step budget — and is never a model baseline.

Subgoals are derived from the scenario itself:
    goal inside(t, R) with R closed        -> OPEN(R)   (first: hand still free)
    target inside a closed container      -> OPEN(container)
    always                                -> GRASP(target)
    goal on_top / inside (t, R)            -> PLACE(R)
    then                                  -> REPORT_DONE

For each subgoal the driver searches base poses around the entity (rings
offset from its AABB, facing it), keeps footprint-free poses, plans a
MOVE / TURN route with authoring.tasks.navigation.plan_route (the same
planner the easy_v1 certifier uses), checks skill feasibility at the pose
counterfactually (teleport -> feasibility engine -> teleport back, no
physics step), and executes the cheapest feasible route. The skill is then
clicked at the entity's pixel in the CURRENT public frame.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from rummagebench.skills.move import is_navigation_floor, same_level_floors, floor_height_allowed

logger = logging.getLogger(__name__)

BASE_HALF_EXTENT = 0.40   # VisualProtocolSession default footprint
BASE_MARGIN = 0.03        # skills/move.py execution margin
ROUTE_SLACK = 0.02        # extra clearance when planning (drift)
RING_OFFSETS_M = tuple(round(0.45 + 0.05 * i, 2) for i in range(14))  # from the AABB edge
RING_SPACING_M = 0.20
MAX_ROUTE_PLANS = 12       # feasible poses kept for route planning
MAX_FEASIBILITY_CHECKS = 40  # counterfactual checks per subgoal and margin
MAX_SEARCH_S = 240.0        # wall-clock cap per subgoal and margin
MAX_SEARCH_S_PLACE = 600.0  # one PLACE check takes ~100 s on this host (2026-10-18)


MOVE_TO_HOP_TOL_M = 0.15    # clicked floor point vs. an intermediate path point
MOVE_TO_PATH_STEP_M = 0.10
APPROACH_OFFSETS_M = tuple(round(0.5 + 0.1 * i, 1) for i in range(21))   # 0.5-2.5 m from the entity


def _move_to_protocol() -> bool:
    """Public protocol without MOVE / TURN (2026-10-26): navigation is MOVE_TO
    on a visible floor point, heading preserved."""
    from rummagebench.core.public_types import PUBLIC_SKILLS
    return "MOVE_TO" in PUBLIC_SKILLS and "MOVE" not in PUBLIC_SKILLS


def _same_level_floors(backend, z: float) -> list:
    """Floor support envelopes exactly as skills.move.execute_move_to collects them."""
    return same_level_floors(backend, z)


def _floor_pixels(frame, backend):
    """(pixels (N,2) as (u, v), world xy (N,2)) of every floor pixel in a frame."""
    seg = np.asarray(frame.instance_segmentation)
    labels = frame.meta.get("instance_labels", {})
    floor_ids = []
    for key, name in labels.items():
        info = backend.describe_entity(name) if name else None
        if info is not None and is_navigation_floor(backend, name, info):
            floor_ids.append(int(key))
    mask = np.isin(seg, floor_ids)
    depth = np.asarray(frame.depth, float)
    mask &= np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    if not len(u) or "pixel_rays" not in frame.meta or frame.depth_convention != "euclidean_range":
        return np.zeros((0, 2), int), np.zeros((0, 2))
    rays = np.asarray(frame.meta["pixel_rays"])[v, u]
    T = np.asarray(frame.camera_extrinsics, float)
    world = (depth[v, u][:, None] * rays) @ T[:3, :3].T + T[:3, 3]
    return np.stack([u, v], 1), world[:, :2]


def _densify(path, step=MOVE_TO_PATH_STEP_M):
    out = [np.asarray(path[0], float)]
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step))
        out += [a + (b - a) * i / n for i in range(1, n + 1)]
    return out


def _geodesic_field(chk, goal, yaw, floors, res=0.1):
    """Planned-distance field to `goal` (8-connected Dijkstra on a res grid,
    fixed-heading footprint free and centre on a same-level floor envelope).
    Returns lookup(xy array) -> distances (inf where unreachable / off grid)."""
    import heapq
    lo = np.min([b[0][:2] for b in floors], axis=0)
    hi = np.max([b[1][:2] for b in floors], axis=0)
    nx, ny = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    gx, gy = np.meshgrid(lo[0] + np.arange(nx) * res, lo[1] + np.arange(ny) * res, indexing="ij")
    on_floor = np.zeros(gx.shape, bool)
    for b in floors:
        on_floor |= (gx >= b[0][0]) & (gx <= b[1][0]) & (gy >= b[0][1]) & (gy <= b[1][1])
    free = on_floor.ravel() & chk.free_many(gx.ravel(), gy.ravel(), np.full(gx.size, yaw))
    free = free.reshape(gx.shape)
    dist = np.full(gx.shape, np.inf)
    gi, gj = int(round((goal[0] - lo[0]) / res)), int(round((goal[1] - lo[1]) / res))
    if not (0 <= gi < nx and 0 <= gj < ny):
        return lambda q: np.full(len(np.atleast_2d(q)), np.inf)
    dist[gi, gj] = 0.0
    heap = [(0.0, gi, gj)]
    steps = [(di, dj, res * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, c in steps:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and d + c < dist[a, b] \
                    and (not (di and dj) or (free[i + di, j] and free[i, j + dj])):
                dist[a, b] = d + c
                heapq.heappush(heap, (d + c, a, b))

    def lookup(q):
        q = np.atleast_2d(np.asarray(q, float))
        i = np.rint((q[:, 0] - lo[0]) / res).astype(int)
        j = np.rint((q[:, 1] - lo[1]) / res).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out = np.full(len(q), np.inf)
        out[ok] = dist[i[ok], j[ok]]
        return out
    return lookup


class OracleFailure(RuntimeError):
    """The scripted driver could not realise a subgoal (recorded, not raised
    out of the episode loop: the agent then abstains)."""


class _NeedReorient(OracleFailure):
    """Floor approach impossible at the current heading: turn first by an
    object-directed MOVE_TO on some visible furniture."""


def derive_subgoals(scenario, backend) -> list[tuple[str, str]]:
    target = scenario.target.entity
    plan: list[tuple[str, str]] = []
    # a closed goal container is opened FIRST, while the hand is free: the
    # modelled hand cannot pull a handle while holding the target (the held
    # body collides with the drawer front; T03 diagnosis 2026-10-19)
    for g in scenario.goal:
        if g.kind != "inside" or g.entity != target:
            continue
        info = backend.articulation_info(g.receptacle) if hasattr(backend, "articulation_info") else None
        if info is not None and not backend.is_open(g.receptacle):
            plan.append(("OPEN", g.receptacle))
    for p in scenario.placements:
        if p.entity == target and p.relation == "inside" and not backend.is_open(p.receptacle) \
                and ("OPEN", p.receptacle) not in plan:
            plan.append(("OPEN", p.receptacle))
    plan.append(("GRASP", target))
    for g in scenario.goal:
        if g.kind not in ("on_top", "inside") or g.entity != target:
            continue
        plan.append(("PLACE", g.receptacle))
    return plan


def _ring_poses(aabb, offsets=RING_OFFSETS_M, spacing=RING_SPACING_M):
    """Base poses on rectangles offset from the entity's XY AABB, each facing
    the nearest point of the AABB (so the arm works on the near edge)."""
    (lx, ly), (hx, hy) = aabb[0][:2], aabb[1][:2]
    out = []
    for d in offsets:
        x0, y0, x1, y1 = lx - d, ly - d, hx + d, hy + d
        per = 2 * ((x1 - x0) + (y1 - y0))
        n = max(8, int(per / spacing))
        for i in range(n):
            s = per * i / n
            if s < x1 - x0:
                x, y = x0 + s, y0
            elif s < (x1 - x0) + (y1 - y0):
                x, y = x1, y0 + s - (x1 - x0)
            elif s < 2 * (x1 - x0) + (y1 - y0):
                x, y = x1 - (s - (x1 - x0) - (y1 - y0)), y1
            else:
                x, y = x0, y1 - (s - 2 * (x1 - x0) - (y1 - y0))
            nx, ny = min(max(x, lx), hx), min(max(y, ly), hy)
            out.append((x, y, math.atan2(ny - y, nx - x), d))
    return out


class RouteOracleAgent:
    """Agent contract of run_visual_episode: reset(instruction), act(obs)."""

    name = "route-oracle"

    def __init__(self, session, room_map=None, waypoint_clearance: float = 0.4):
        self._vs = session                       # VisualProtocolSession
        self._backend = session._backend
        self._scenario = session._scenario
        self._room_map = room_map
        self._clearance = waypoint_clearance
        self._queue: deque[dict] = deque()
        self._subgoals: list[tuple[str, str]] = []
        self._phase = 0
        self._pending_click: Optional[tuple[str, str]] = None
        self._last_route_action: Optional[dict] = None
        self._done = False
        self.log: list[dict[str, Any]] = []
        self.failure: Optional[str] = None
        # CPU-replayable planning inputs of the last failed route search
        self.route_debug: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------ contract

    def reset(self, instruction: str) -> None:
        self._queue.clear()
        self._subgoals = derive_subgoals(self._scenario, self._backend)
        self._phase = 0
        self._pending_click = None
        self._last_route_action = None
        self._nav_key = None
        self._done = False
        self.log = [{"subgoals": [list(s) for s in self._subgoals]}]
        self.failure = None

    def act(self, observation) -> Optional[dict]:
        try:
            return self._act(observation)
        except OracleFailure as e:
            self.failure = str(e)
            self.log.append({"failure": str(e)})
            logger.warning("route oracle: %s", e)
            return None

    # ------------------------------------------------------------ internals

    def _act(self, observation) -> Optional[dict]:
        fb = getattr(getattr(observation, "feedback", None), "code", None)
        if self._last_route_action is not None:
            if fb is not None and getattr(fb, "value", fb) != "EXECUTED":
                raise OracleFailure(f"route step {self._last_route_action} -> "
                                    f"{getattr(fb, 'value', fb)}")
            self._last_route_action = None
        if self._queue:
            self._last_route_action = self._queue.popleft()
            return self._last_route_action
        if self._pending_click is not None:
            skill, entity = self._pending_click
            self._pending_click = None
            if fb is not None and getattr(fb, "value", fb) != "EXECUTED":
                raise OracleFailure(f"{skill}({entity}) -> {getattr(fb, 'value', fb)}")
            self._phase += 1
        if self._phase >= len(self._subgoals):
            if self._done:
                return None
            self._done = True
            return {"skill": "REPORT_DONE"}
        skill, entity = self._subgoals[self._phase]
        if self._ready(skill, entity):
            return self._click(skill, entity, observation)
        if _move_to_protocol():
            # one MOVE_TO per step: the next click depends on the new frame
            self._last_route_action = self._move_to_step(skill, entity)
            return self._last_route_action
        route = self._route_to_interaction_pose(skill, entity)
        self._queue.extend(route)
        self.log.append({"subgoal": [skill, entity], "route_actions": len(route)})
        if not self._queue:
            raise OracleFailure(f"empty route for {skill}({entity})")
        self._last_route_action = self._queue.popleft()
        return self._last_route_action

    def _verdict(self, skill: str, entity: str):
        """Counterfactual feasibility for the pose SEARCH only. For PLACE the
        candidate surface points are limited to the first production batch
        (ordered by comfortable reach from the base): the production check
        walks the whole surface grid when a pose is infeasible (minutes on a
        dining table). The executed PLACE click is still judged by the
        untouched production check inside the session."""
        inner = self._vs._session
        resolved = self._backend.resolve_entity(entity)
        if skill != "PLACE":
            return inner._feasibility.check(skill, resolved, inner.robot, inner.world_state)
        import rummagebench.validation.feasibility as feas_mod
        from rummagebench.feasibility.interaction_target import PLACE_BATCH_SIZE

        original = feas_mod.build_object_interface

        def limited(info, backend):
            obj = original(info, backend)
            full = obj.interaction_targets

            def first_batch(skill_name, world):
                targets = list(full(skill_name, world))
                # angled drawer targets: several tilts per point -> 3 batches
                inside = any(getattr(t.region, "kind", None) == "inside_volume" for t in targets[:1])
                return targets[:PLACE_BATCH_SIZE * (3 if inside else 1)]
            obj.interaction_targets = first_batch
            return obj

        feas_mod.build_object_interface = limited
        try:
            return inner._feasibility.check(skill, resolved, inner.robot, inner.world_state)
        finally:
            feas_mod.build_object_interface = original

    def _feasible(self, skill: str, entity: str) -> bool:
        return bool(self._verdict(skill, entity).feasible)

    def _visible_point(self, entity: str):
        from rummagebench.authoring.tasks.certify import click_point

        frame = self._vs._store.get(self._vs._store.current_id)
        return frame, click_point(frame, self._backend, entity)

    def _ready(self, skill: str, entity: str) -> bool:
        return self._visible_point(entity)[1] is not None and self._feasible(skill, entity)

    def _click(self, skill: str, entity: str, observation) -> dict:
        frame, pt = self._visible_point(entity)
        if pt is None:
            raise OracleFailure(f"{skill}: {entity} not visible in the current frame")
        self._pending_click = (skill, entity)
        self.log.append({"click": [skill, entity], "point": list(pt)})
        return {"skill": skill, "point": {"frame_id": frame.frame_id, "x": pt[0], "y": pt[1]}}

    def _waypoints(self) -> list[tuple[float, float]]:
        if self._room_map is None:
            return []
        pts: list[tuple[float, float]] = []
        for room in sorted(set(self._room_map.names.values())):
            pts += self._room_map.free_points(room, self._clearance, 0.25)
        return pts

    def _route_to_interaction_pose(self, skill: str, entity: str) -> list[dict]:
        from rummagebench.authoring.tasks.navigation import (
            FootprintChecker, ProtocolLimits, plan_route)
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.skills.move import quat_from_yaw, yaw_from_quat

        backend = self._backend
        aabb = backend.entity_aabb(entity)
        if aabb is None:
            raise OracleFailure(f"{entity} has no AABB")
        proto = self._scenario.agent_protocol
        lim = ProtocolLimits(proto.min_move_cm / 100.0, proto.max_move_cm / 100.0,
                             proto.min_turn_deg, proto.max_turn_deg)
        pos, quat = backend.robot_pose()
        here = (float(pos[0]), float(pos[1]), yaw_from_quat(quat))
        z = float(pos[2])
        wps = self._waypoints()
        for margin in (BASE_MARGIN + ROUTE_SLACK, BASE_MARGIN):
            chk = FootprintChecker.from_backend(backend, BASE_HALF_EXTENT, margin)
            ring = _ring_poses(aabb)
            free = chk.free_many(np.array([r[0] for r in ring]), np.array([r[1] for r in ring]),
                                 np.array([r[2] for r in ring]))
            cands = [r for r, ok in zip(ring, free) if ok]
            # closest to the entity first, then closest to the robot
            cands.sort(key=lambda r: (r[3], math.hypot(r[0] - here[0], r[1] - here[1])))
            # spread the bounded budget over the ring: every k-th candidate
            # (nearest offsets first) instead of exhausting one side
            if len(cands) > MAX_FEASIBILITY_CHECKS:
                k = len(cands) / MAX_FEASIBILITY_CHECKS
                cands = [cands[int(i * k)] for i in range(MAX_FEASIBILITY_CHECKS)]
            # evaluator-private interaction anchors "<entity>__<SKILL>*" (poses
            # where the skill succeeded before) are tried first
            hints = [(float(a.position[0]), float(a.position[1]),
                      yaw_from_quat(list(a.orientation)), 0.0)
                     for name, a in sorted(self._scenario.anchors.items())
                     if name.startswith(f"{entity}__{skill}")]
            hints = [h for h in hints if chk.free(h[0], h[1], h[2])]
            cands = hints + cands
            feasible, reasons, checked = [], {}, 0
            t0 = time.time()
            try:
                for c in cands:
                    if time.time() - t0 > (MAX_SEARCH_S_PLACE if skill == "PLACE" else MAX_SEARCH_S):
                        break
                    backend.teleport_robot(AnchorSpec(position=[c[0], c[1], z],
                                                      orientation=quat_from_yaw(c[2])))
                    verdict = self._verdict(skill, entity)
                    checked += 1
                    if verdict.feasible:
                        feasible.append(c)
                    else:
                        reasons[str(verdict.reason)] = reasons.get(str(verdict.reason), 0) + 1
                    if len(feasible) >= MAX_ROUTE_PLANS:
                        break
            finally:
                backend.teleport_robot(AnchorSpec(position=list(pos), orientation=list(quat)))
            best = None
            for c in feasible:
                route = plan_route(chk, here, (c[0], c[1], c[2]), wps, lim)
                if route is not None and (best is None or route.steps < best[0].steps):
                    best = (route, c)
            self.log.append({"pose_search": [skill, entity], "margin": margin,
                             "anchor_hints": len(hints),
                             "free_candidates": len(free) and int(free.sum()), "checked": checked,
                             "search_s": round(time.time() - t0, 1), "infeasible_reasons": reasons,
                             "feasible_candidates": len(feasible),
                             "feasible_poses": [[round(v, 3) for v in c[:3]] for c in feasible],
                             "chosen": None if best is None else
                             {"pose": [round(v, 3) for v in best[1][:3]], "steps": best[0].steps}})
            if best is not None:
                return list(best[0].actions)
            self.route_debug = {
                "skill": skill, "entity": entity, "start": list(here), "margin": margin,
                "half_extent": BASE_HALF_EXTENT,
                "boxes_xy_inflated": [[*map(float, lo), *map(float, hi)]
                                      for lo, hi in zip(chk.lo, chk.hi)],
                "goals": [list(c[:3]) for c in feasible], "waypoints": [list(w) for w in wps],
                "limits": [lim.min_move_m, lim.max_move_m, lim.min_turn_deg, lim.max_turn_deg]}
        raise OracleFailure(f"no reachable pose with feasible {skill}({entity})")

    # ------------------------------------------------- MOVE_TO protocol

    def _move_to_step(self, skill: str, entity: str) -> dict:
        """Entity visible in the current frame: click a pixel of it whose
        object-directed MOVE_TO stand pose (the session's own selection) makes
        the skill feasible. Otherwise walk on the floor towards it."""
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.perception.visual_bridge import VisualBridge, VisualGroundingError
        from rummagebench.skills.move import quat_from_yaw, select_stand_pose

        backend, vs = self._backend, self._vs
        frame = vs._panorama_frame if vs._panorama_frame is not None else vs._store.get(vs._store.current_id)
        seg = np.asarray(frame.instance_segmentation)
        H, W = seg.shape[:2]
        labels = frame.meta.get("instance_labels", {})
        # click the entity itself first; for GRASP also the furniture it rests
        # on / in (a small object yields one stand pose; its support offers others)
        click_targets = [entity]
        if skill == "GRASP":
            click_targets += [p.receptacle for p in self._scenario.placements
                              if p.entity == entity and p.receptacle not in click_targets]
        bridge = VisualBridge()
        pos, quat = backend.robot_pose()
        seen, checks, visible = set(), 0, False
        log = {"pixels": {}, "stand_poses": 0, "infeasible": 0}
        limit = 6 if skill == "PLACE" else 12
        for target in click_targets:
            insts = [int(k) for k, v in labels.items() if v == target]
            mask = np.isin(seg, insts) if insts else np.zeros(seg.shape, bool)
            log["pixels"][target] = int(mask.sum())
            if mask.sum() < (20 if target == entity else 200):
                continue
            visible = True
            v, u = np.nonzero(mask)
            order = np.argsort((u - u.mean()) ** 2 + (v - v.mean()) ** 2)   # centre first, then spread
            for i in order[:: max(1, len(order) // 24)][:24]:
                if checks >= limit:
                    break
                x, y = (u[i] + 0.5) / W, (v[i] + 0.5) / H
                try:
                    ent, surface, _ = bridge.resolve(frame, x, y, lambda inst: backend.instance_to_entity(inst, frame))
                except VisualGroundingError:
                    continue
                if ent != target:
                    continue
                pose, _, _ = select_stand_pose(backend, target, surface, BASE_HALF_EXTENT, BASE_MARGIN)
                key = None if pose is None else (round(pose[0], 2), round(pose[1], 2), round(pose[2], 2))
                if key is None or key in seen:
                    continue
                seen.add(key)
                log["stand_poses"] += 1
                try:
                    backend.teleport_robot(AnchorSpec(position=[pose[0], pose[1], float(pos[2])],
                                                      orientation=quat_from_yaw(pose[2])))
                    ok = self._feasible(skill, entity)
                finally:
                    backend.teleport_robot(AnchorSpec(position=list(pos), orientation=list(quat)))
                checks += 1
                if ok:
                    point = {"frame_id": frame.frame_id, "x": round(float(x), 5), "y": round(float(y), 5)}
                    self.log.append({"move_to": "entity", "subgoal": [skill, entity], "clicked": target,
                                     "point": point, "stand_pose": [round(c, 3) for c in pose], **log})
                    return {"skill": "MOVE_TO", "point": point}
                log["infeasible"] += 1
        if visible:
            self.log.append({"move_to_entity_failed": [skill, entity], **log})
            # walk closer and try again from there (bounded by the hop limit)
            try:
                return self._move_to_hop(skill, entity)
            except _NeedReorient:
                return self._reorient_or_fallback(skill, entity)
            except OracleFailure:
                raise OracleFailure(f"MOVE_TO: no feasible stand pose for {skill}({entity}) from visible pixels")
        try:
            return self._move_to_hop(skill, entity)
        except _NeedReorient:
            return self._reorient_or_fallback(skill, entity)

    def _reorient_or_fallback(self, skill: str, entity: str) -> dict:
        """Reorienting click first; when none helps, walk towards fallback
        approach goals (interaction anchors / loose ring) at this heading."""
        try:
            return self._reorient_click(entity)
        except OracleFailure:
            if getattr(self, "_nav_fallback", None) == (self._phase, entity):
                raise
            self._nav_fallback, self._nav_key = (self._phase, entity), None
            try:
                return self._move_to_hop(skill, entity)
            except _NeedReorient:
                raise OracleFailure(f"MOVE_TO: {entity} not approachable on the floor, "
                                    f"no reorienting click or fallback goal found")

    def _approach_goals(self, entity: str, yaw: Optional[float] = None, pos=None) -> list[tuple[float, float, None]]:
        """Floor goals for walking towards an entity until it can be clicked:
        0.5-2.5 m from it (or from the furniture it rests on), free at the
        heading, nothing but the entity / its support in between (never behind
        a wall), nearest by planned distance. Raises _NeedReorient when none is
        reachable at this heading."""
        from rummagebench.skills.move import stand_candidates, stand_ring, yaw_from_quat

        backend = self._backend
        pos_given = pos is not None
        if pos is None or yaw is None:
            p, quat = backend.robot_pose()
            pos, yaw = p if pos is None else pos, yaw_from_quat(quat) if yaw is None else yaw
        floors = _same_level_floors(backend, float(pos[2]))
        targets = [entity] + [p.receptacle for p in self._scenario.placements if p.entity == entity]
        chk, ring = None, []
        for target in targets:
            chk, cands = stand_candidates(backend, target, BASE_HALF_EXTENT, BASE_MARGIN, yaw=yaw,
                                          offsets=APPROACH_OFFSETS_M)
            ring += [(x, y) for x, y, _, _ in (cands or [])]
        if chk is None:
            raise OracleFailure(f"{entity} has no AABB")
        field = None
        # fallbacks when no ring pose is free with a clear reach at this heading
        # (e.g. a shelf reachable only diagonally from a bathroom corner):
        # evaluator-private interaction anchors ("<entity>__<SKILL>*", poses
        # where a skill succeeded), then ring points without the reach test.
        # Walking there only brings the entity into view; the final stand pose
        # is still the session's own, checked by _move_to_step.
        hints = [(float(a.position[0]), float(a.position[1]))
                 for name, a in sorted(self._scenario.anchors.items())
                 if any(name.startswith(f"{t}__") for t in targets)]
        loose = []
        for target in targets:
            aabb = backend.entity_aabb(target)
            if aabb is not None:
                loose += [(c[0], c[1]) for c in stand_ring(aabb, APPROACH_OFFSETS_M)]
        fallback = pos_given is False and getattr(self, "_nav_fallback", None) == (self._phase, entity)
        for kind, cand in (("ring", ring), ("anchors", hints), ("loose", loose))[:3 if fallback else 1]:
            cand = [c for c in cand if chk.free(c[0], c[1], yaw)]
            if not cand:
                continue
            if field is None:
                field = _geodesic_field(chk, pos[:2], yaw, floors)     # distance FROM the robot
            d = field(np.asarray(cand))
            order = [i for i in np.argsort(d) if np.isfinite(d[i])][:3]
            if order:
                if kind != "ring":
                    self.log.append({"approach_fallback": kind,
                                     "entity": entity, "goals": [[round(v, 2) for v in cand[i]] for i in order]})
                return [(cand[i][0], cand[i][1], None) for i in order]
        raise _NeedReorient(f"MOVE_TO: {entity} not approachable on the floor at heading {yaw:.2f}")

    def _reorient_click(self, entity: str) -> dict:
        """Object-directed MOVE_TO on some visible furniture whose stand-pose
        heading makes `entity` approachable on the floor (public action; the
        session computes the same stand pose)."""
        from rummagebench.perception.visual_bridge import VisualBridge, VisualGroundingError
        from rummagebench.skills.move import NOT_STAND_TARGETS, select_stand_pose

        backend, vs = self._backend, self._vs
        frame = vs._panorama_frame if vs._panorama_frame is not None else vs._store.get(vs._store.current_id)
        seg = np.asarray(frame.instance_segmentation)
        H, W = seg.shape[:2]
        labels = frame.meta.get("instance_labels", {})
        z = float(backend.robot_pose()[0][2])
        counts = {}
        for k, name in labels.items():
            info = backend.describe_entity(name) if name else None
            if info is None or not info.fixed_base or info.category in NOT_STAND_TARGETS | {"door"}:
                continue
            counts.setdefault(name, []).append(int(k))
        ranked = sorted(counts, key=lambda n: -int(np.isin(seg, counts[n]).sum()))[:10]
        bridge = VisualBridge()
        for name in ranked:
            mask = np.isin(seg, counts[name])
            if mask.sum() < 200:
                break
            v, u = np.nonzero(mask)
            i = int(np.argmin((u - u.mean()) ** 2 + (v - v.mean()) ** 2))
            x, y = (u[i] + 0.5) / W, (v[i] + 0.5) / H
            try:
                ent, surface, _ = bridge.resolve(frame, x, y, lambda inst: backend.instance_to_entity(inst, frame))
            except VisualGroundingError:
                continue
            if ent != name:
                continue
            pose, _, _ = select_stand_pose(backend, name, surface, BASE_HALF_EXTENT, BASE_MARGIN)
            if pose is None:
                continue
            try:
                self._approach_goals(entity, yaw=pose[2], pos=[pose[0], pose[1], z])
            except _NeedReorient:
                continue
            point = {"frame_id": frame.frame_id, "x": round(float(x), 5), "y": round(float(y), 5)}
            self.log.append({"move_to": "reorient", "clicked": name, "point": point,
                             "stand_pose": [round(c, 3) for c in pose]})
            self._nav_key = None   # re-plan the floor approach from the new pose
            return {"skill": "MOVE_TO", "point": point}
        raise OracleFailure(f"MOVE_TO: {entity} not approachable on the floor and no reorienting click found")

    def _move_to_hop(self, skill: str, entity: str) -> dict:
        """Next floor MOVE_TO towards an entity that is not yet in view: the
        farthest visible point along the planned path, else a visible point
        with a shorter planned distance, else back off to see the goal area."""
        from rummagebench.authoring.tasks.navigation import FootprintChecker
        from rummagebench.core.scenario import AnchorSpec
        from rummagebench.perception.visual_bridge import VisualBridge, VisualGroundingError
        from rummagebench.skills.move import plan_bev_path, yaw_from_quat

        key = (self._phase, skill, entity)
        if getattr(self, "_nav_key", None) != key:
            self._nav_key, self._nav_goals, self._nav_hops = key, self._approach_goals(entity), 0
        backend, vs = self._backend, self._vs
        frame = vs._panorama_frame if vs._panorama_frame is not None else vs._store.get(vs._store.current_id)
        pix, xy = _floor_pixels(frame, backend)
        if not len(pix):
            raise OracleFailure("MOVE_TO: no floor pixels in the current frame")
        H, W = np.asarray(frame.instance_segmentation).shape[:2]
        bridge = VisualBridge()
        pos, quat = backend.robot_pose()
        yaw = yaw_from_quat(quat)
        here = np.asarray(pos[:2], float)
        chk = FootprintChecker.from_backend(backend, BASE_HALF_EXTENT, BASE_MARGIN)
        floors = _same_level_floors(backend, float(pos[2]))

        diag = {"floor_pixels": int(len(xy)), "resolve_fail": 0, "not_floor": 0, "off_target": 0,
                "no_path": 0, "infeasible_at_point": 0}

        def click_candidates(q, tol, max_tries=6):
            """Floor pixels nearest to q whose resolution (same bridge as the
            session) lands within tol and is reachable by the session planner."""
            d = np.hypot(*(xy - np.asarray(q)).T)
            near = np.nonzero(d <= tol)[0]
            # spread the tries over the disc instead of one pixel cluster
            near = near[np.argsort(d[near])][::max(1, len(near) // (4 * max_tries))]
            found = 0
            for i in near:
                u, v = int(pix[i][0]), int(pix[i][1])
                x, y = (u + 0.5) / W, (v + 0.5) / H
                try:
                    ent, surface, _ = bridge.resolve(frame, x, y, lambda inst: backend.instance_to_entity(inst, frame))
                except VisualGroundingError:
                    diag["resolve_fail"] += 1
                    continue
                info = backend.describe_entity(ent)
                if not is_navigation_floor(backend, ent, info):
                    diag["not_floor"] += 1
                    continue
                if math.dist(surface[:2], q) > tol or not floor_height_allowed(surface[2], pos[2]):
                    diag["off_target"] += 1
                    continue
                if not chk.free(float(surface[0]), float(surface[1]), yaw) or \
                        plan_bev_path(chk, here, surface[:2], yaw, floors)[0] is None:
                    diag["no_path"] += 1
                    continue
                yield {"frame_id": frame.frame_id, "x": round(x, 5), "y": round(y, 5)}, surface
                found += 1
                if found >= max_tries:
                    return

        def click_near(q, tol, max_tries=3):
            return next(click_candidates(q, tol, max_tries), (None, None))

        # intermediate hop along the best path: farthest visible progress
        self._nav_hops = getattr(self, "_nav_hops", 0) + 1
        if self._nav_hops > 8:
            raise OracleFailure(f"MOVE_TO: more than 8 hops towards {skill}({entity})")
        for gx, gy, _ in self._nav_goals[:3]:
            path, _why = plan_bev_path(chk, here, (gx, gy), yaw, floors)
            if path is None:
                continue
            pts = _densify(path)
            for q in reversed(pts[1:]):
                if math.dist(q, here) < 0.3:
                    break
                point, surface = click_near(q, MOVE_TO_HOP_TOL_M)
                if point is not None:
                    self.log.append({"move_to": "hop", "subgoal": [skill, entity], "point": point,
                                     "target_xy": [round(float(v), 3) for v in surface[:2]]})
                    return {"skill": "MOVE_TO", "point": point}
        # path hidden from here (furniture): the visible, standable floor point
        # with the shortest planned distance to the best goal
        gx, gy, _ = self._nav_goals[0]
        field = _geodesic_field(chk, (gx, gy), yaw, floors)
        remaining = float(field(here)[0])
        step = max(1, len(xy) // 20000)
        sub = xy[::step]
        dist = field(sub)
        dist[np.hypot(*(sub - here).T) < 0.3] = np.inf
        diag["detour_remaining_m"] = round(remaining, 2)
        diag["detour_best_m"] = round(float(dist.min()), 2) if len(dist) else None
        tried = set()
        for i in np.argsort(dist)[:200]:
            if not np.isfinite(dist[i]) or dist[i] >= remaining - 0.3 or len(tried) >= 10:
                break
            cell = (round(float(sub[i][0]), 1), round(float(sub[i][1]), 1))
            if cell in tried:
                continue
            tried.add(cell)
            point, surface = click_near(tuple(sub[i]), MOVE_TO_HOP_TOL_M, max_tries=2)
            if point is not None:
                self.log.append({"move_to": "detour", "subgoal": [skill, entity], "point": point,
                                 "target_xy": [round(float(v), 3) for v in surface[:2]],
                                 "remaining_m": round(float(field(surface[:2])[0]), 2)})
                return {"skill": "MOVE_TO", "point": point}
        # close to the goal but its floor is hidden (robot body / furniture):
        # back off to a visible point 1-2.5 m (planned) from the goal first
        if remaining < 1.6:
            back = np.where((dist >= 1.0) & (dist <= 2.5), dist, np.inf)
            back[np.hypot(*(sub - here).T) < 0.5] = np.inf
            tried = set()
            for i in np.argsort(back)[:200]:
                if not np.isfinite(back[i]) or len(tried) >= 10:
                    break
                cell = (round(float(sub[i][0]), 1), round(float(sub[i][1]), 1))
                if cell in tried:
                    continue
                tried.add(cell)
                point, surface = click_near(tuple(sub[i]), MOVE_TO_HOP_TOL_M, max_tries=2)
                if point is not None:
                    self.log.append({"move_to": "back_off", "subgoal": [skill, entity], "point": point,
                                     "target_xy": [round(float(v), 3) for v in surface[:2]],
                                     "remaining_m": round(float(field(surface[:2])[0]), 2)})
                    return {"skill": "MOVE_TO", "point": point}
        self.log.append({"move_to_failure": [skill, entity], **diag})
        raise OracleFailure(f"MOVE_TO: no visible floor point towards {skill}({entity})")
