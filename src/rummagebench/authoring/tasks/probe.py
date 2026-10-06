"""Embodiment probing: SceneSlots x robot -> SlotEmbodimentOverlay.

Runs inside a live simulator session (OmniGibsonBackend + BenchmarkSession
with the production pinocchio feasibility engine). Nothing here is a proxy:

  * interaction anchors — poses in front of each slot furniture (standoff x
    lateral grid, facing the furniture) that pass the SAME base-footprint
    check the agent's MOVE/TURN use;
  * OPEN feasibility — FeasibilityValidator.check("OPEN") from the anchor
    (URDF IK + configuration-space collision at the handle interface);
  * GRASP-region feasibility — a probe object is physically placed in the
    slot (the builder's own placement routine) and GRASP is checked from the
    anchor, with the furniture open;
  * reveal evidence — the probe object's visibility ratio from the anchor
    with the furniture open vs closed, and GRASP with it closed (OPEN must
    be causally necessary for in_container slots);
  * start poses — free room poses (footprint-free at every heading, so the
    agent can turn in place) with the set of slot furniture they see.

The probe is a search-space PRE-FILTER: every episode is still certified
individually (certify.py) with its real objects.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rummagebench.authoring.tasks.embodiment_slots import Pose2, SlotInteraction, StartPose
from rummagebench.authoring.tasks.room_map import RoomMap
from rummagebench.authoring.tasks.slots import Furniture, SceneSlots, Slot

logger = logging.getLogger(__name__)

PROBE_VERSION = 5


@dataclass
class ProbeConfig:
    # robot base centre distance beyond the furniture front face (m)
    # the R1Pro head camera (1.62 m, 20 deg down, 58.7 deg vertical FOV —
    # measured) sees an object at drawer height only from >= ~1.1 m, while
    # GRASP reaches ~1.0-1.4 m: probe out to the far end of that band
    standoffs_m: tuple[float, ...] = (0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35)
    # lateral offsets as a fraction of the furniture width
    lateral_fracs: tuple[float, ...] = (0.0, -0.3, 0.3)
    footprint_half_m: float = 0.40
    footprint_margin_m: float = 0.03
    start_clearance_m: float = 0.5
    start_spacing_m: float = 0.5
    starts_per_room: int = 4
    start_yaws_deg: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    # furniture counts as "seen" from a start pose with this many pixels
    seen_min_px: int = 50
    settle_steps: int = 30
    # prefilter on the PROBE object (the benchmark threshold, not the
    # stricter generator margin: the real target is certified separately)
    reveal_visibility_min: float = 0.60
    # the agent must SEE the furniture to click OPEN on it
    click_min_px: int = 300
    # anchors tried (in preference order) for MOVE/TURN reachability
    reach_tries: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# pure geometry (CPU, unit-tested)
# ---------------------------------------------------------------------------

def wrap_yaw(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


FLOOR_MAX_THICKNESS_M = 0.03
FLOOR_MIN_AREA_M2 = 0.01


def drawer_floor_slab(link_aabbs) -> Optional[tuple[list[float], list[float]]]:
    """The drawer's floor: the largest thin horizontal collision body of the
    drawer link (world AABBs [(lo, hi)]); None when there is none."""
    slabs = [(lo, hi) for lo, hi in link_aabbs
             if hi[2] - lo[2] <= FLOOR_MAX_THICKNESS_M
             and (hi[0] - lo[0]) * (hi[1] - lo[1]) >= FLOOR_MIN_AREA_M2]
    if not slabs:
        return None
    lo, hi = max(slabs, key=lambda b: (b[1][0] - b[0][0]) * (b[1][1] - b[0][1]))
    return [float(v) for v in lo], [float(v) for v in hi]


def floor_region_corners(slab, inset_m: float = 0.04, lift_m: float = 0.03):
    """Four points spanning where an object can rest on the slab: an object
    centre stays >= inset_m from the walls; lift_m above the floor top."""
    lo, hi = slab
    z = hi[2] + lift_m
    x0, x1 = lo[0] + inset_m, hi[0] - inset_m
    y0, y1 = lo[1] + inset_m, hi[1] - inset_m
    if x0 > x1:
        x0 = x1 = (lo[0] + hi[0]) / 2.0
    if y0 > y1:
        y0 = y1 = (lo[1] + hi[1]) / 2.0
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def points_in_view(K, T_world_camera, width: int, height: int, points,
                   margin_px: float = 4.0) -> bool:
    """Every world point projects inside the image (pinhole K; T maps the
    optical camera frame — +Z forward, +Y down — to world)."""
    import numpy as np

    T = np.asarray(T_world_camera, dtype=float)
    K = np.asarray(K, dtype=float)
    if not (np.isfinite(T).all() and np.isfinite(K).all()):
        return False
    P = np.c_[np.asarray(points, dtype=float), np.ones(len(points))]
    cam = (np.linalg.inv(T) @ P.T).T[:, :3]
    if np.any(cam[:, 2] <= 1e-6):
        return False
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return bool(np.all((u >= margin_px) & (u <= width - margin_px)
                       & (v >= margin_px) & (v <= height - margin_px)))


def anchor_candidates(furn: Furniture, cfg: ProbeConfig) -> list[tuple[float, float, float]]:
    """(x, y, yaw) poses in front of the furniture, facing it.

    BEHAVIOR furniture faces its local +x (verified against the GPU-picked
    knife_search_001 anchors); size is the scaled local AABB (x = depth,
    y = width). Ordered nearest standoff first, centred first.
    """
    yaw = furn.yaw
    fx, fy = math.cos(yaw), math.sin(yaw)
    lx, ly = -fy, fx
    depth, width = furn.size[0], furn.size[1]
    out = []
    for d in cfg.standoffs_m:
        c = depth / 2.0 + d
        for lf in cfg.lateral_fracs:
            lat = lf * width
            out.append((furn.position[0] + fx * c + lx * lat,
                        furn.position[1] + fy * c + ly * lat,
                        wrap_yaw(yaw + math.pi)))
    return out


def farthest_points(points: list[tuple[float, float]], k: int) -> list[tuple[float, float]]:
    """Deterministic farthest-point subset, seeded at the point nearest the
    centroid (central, well-spread start poses)."""
    if len(points) <= k:
        return list(points)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    first = min(points, key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2, p))
    chosen = [first]
    while len(chosen) < k:
        nxt = max(points, key=lambda p: (min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                                             for q in chosen), p))
        chosen.append(nxt)
    return chosen


def pose_from_xyyaw(x: float, y: float, z: float, yaw: float) -> Pose2:
    return Pose2((round(x, 4), round(y, 4), round(z, 4)),
                 (0.0, 0.0, round(math.sin(yaw / 2.0), 6), round(math.cos(yaw / 2.0), 6)))


# ---------------------------------------------------------------------------
# simulator probing
# ---------------------------------------------------------------------------

@dataclass
class FurnitureProbe:
    entity: str
    anchors_tested: int = 0
    anchors_footprint_free: int = 0
    chosen_anchor: Optional[Pose2] = None
    open_feasible_from: list[int] = field(default_factory=list)
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    fillable_links: list[str] = field(default_factory=list)
    error: Optional[str] = None  # exceptions only (an incomplete probe)
    note: Optional[str] = None
    furniture_px_by_anchor: dict[str, int] = field(default_factory=dict)
    open_route_steps: Optional[int] = None  # MOVE/TURN steps from a room start
    # anchors whose footprint is still free with the furniture open
    free_after_open_from: list[int] = field(default_factory=list)
    seconds: float = 0.0


class SceneProber:
    """One live session; probes the slot furniture of the given rooms."""

    def __init__(self, backend, session, scene_slots: SceneSlots, room_map: RoomMap,
                 probe_entity: str, cfg: ProbeConfig,
                 log: Callable[[str], None] = print, views_dir=None):
        self.backend = backend
        self.session = session
        self.ss = scene_slots
        self.room_map = room_map
        self.probe = probe_entity
        self.cfg = cfg
        self.log = log
        self.z = backend.robot_pose()[0][2]
        if abs(self.z - 0.0053) > 0.05:  # robot not standing on a floor at z=0
            raise RuntimeError(f"probe base height {self.z:.3f} m: the robot is not "
                               "standing on the floor (fell off the scene?)")
        self._chk = None  # FootprintChecker of the reset (all closed) world
        self.room_starts: dict[str, list[StartPose]] = {}
        self.views_dir = views_dir

    # ------------------------------------------------------------ helpers

    def _teleport(self, x: float, y: float, yaw: float) -> None:
        from rummagebench.core.scenario import AnchorSpec

        p = pose_from_xyyaw(x, y, self.z, yaw)
        self.backend.teleport_robot(AnchorSpec(position=list(p.position),
                                               orientation=list(p.orientation)))
        self.backend._collision_body_cache = None

    def _footprint_free(self, x: float, y: float, yaw: float) -> bool:
        """skills/move.py footprint semantics over the CLOSED reset world
        (vectorised; obstacle AABBs read once — every caller probes from a
        freshly reset state, the probe object excluded)."""
        if self._chk is None:
            from rummagebench.authoring.tasks.navigation import FootprintChecker

            self._chk = FootprintChecker.from_backend(
                self.backend, self.cfg.footprint_half_m, self.cfg.footprint_margin_m,
                skip=frozenset({self.probe}))
        return self._chk.free(x, y, yaw)

    def _route_steps(self, chk, start, goal, room: str) -> Optional[int]:
        """MOVE/TURN steps from ``start`` (None: the room's start poses) to
        ``goal`` (x, y, yaw) under ``chk``; None when unreachable."""
        from rummagebench.authoring.tasks.navigation import plan_route

        wps = self.room_map.free_points(room, 0.4, 0.25)
        if start is not None:
            r = plan_route(chk, start, goal, wps)
            return None if r is None else r.steps
        seen_xy, best = set(), None
        for sp in self.room_starts.get(room, []):
            xy = (round(sp.pose.xy[0], 2), round(sp.pose.xy[1], 2))
            if xy in seen_xy:
                continue  # one heading per start position is enough
            seen_xy.add(xy)
            r = plan_route(chk, (sp.pose.xy[0], sp.pose.xy[1], sp.pose.yaw), goal, wps)
            if r is not None and (best is None or r.steps < best):
                best = r.steps
            if len(seen_xy) >= 2 and best is not None:
                break
        return best

    def _check(self, skill: str, entity: str):
        resolved = self.backend.resolve_entity(entity)
        return self.session._feasibility.check(skill, resolved, self.session.robot,
                                               self.session.world_state)

    def _save_view(self, slot_id: str, tag: str) -> None:
        """Head-camera RGB for human review of the reveal (build/ only)."""
        if self.views_dir is None:
            return
        try:
            from pathlib import Path

            name = slot_id.split("/", 1)[-1].replace("/", "__")
            path = Path(self.views_dir) / f"{name}__{tag}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.backend.render_snapshot(str(path))
        except Exception as e:  # diagnostics only
            logger.warning("view snapshot failed: %s", e)

    def _visibility(self, entity: str) -> dict[str, Any]:
        from rummagebench.sim.omnigibson.visibility import measure_visibility

        return measure_visibility(self.backend, entity).to_dict()

    def _place_probe(self, slot: Slot) -> dict[str, Any]:
        """Builder placement routine (an inside receptacle must be open).

        Verified like the builder does (relation + link membership); for a
        cabinet_interior slot the object must NOT have landed in a drawer.
        """
        from rummagebench.sim.omnigibson import probe_support as ps

        rec = slot.parent_entity
        try:
            sampled = ps.place_entity(self.backend, self.probe, rec, slot.relation, slot.link)
        except Exception as e:  # e.g. no fillable volume for the link
            return {"placed": False, "place_error": str(e)}
        self.backend.settle(self.cfg.settle_steps)
        self.backend._collision_body_cache = None
        relation = ps.relation_holds(self.backend, self.probe, rec, slot.relation)
        link_name = ps.containing_link(self.backend, rec, self.probe)
        if slot.link is not None:
            link_ok = link_name == slot.link
        elif slot.relation == "inside":
            link_ok = link_name is None or not self._is_drawer_link(slot, link_name)
        else:
            link_ok = True
        return {"placed": bool(relation and link_ok), "sampler_ok": sampled,
                "relation_ok": bool(relation), "containing_link": link_name}

    def _is_drawer_link(self, slot: Slot, link_name: str) -> bool:
        return any(s.link == link_name for s in self.ss.slots
                   if s.parent_entity == slot.parent_entity and s.slot_type == "drawer")

    # ------------------------------------------------------------ starts

    def probe_start_poses(self, room: str) -> list[StartPose]:
        cfg = self.cfg
        pts = self.room_map.free_points(room, cfg.start_clearance_m, cfg.start_spacing_m)
        free = [p for p in pts
                if all(self._footprint_free(p[0], p[1], math.radians(a))
                       for a in cfg.start_yaws_deg)]
        chosen = farthest_points(free, cfg.starts_per_room)
        slot_furn: dict[str, list[str]] = {}
        for s in self.ss.slots:
            slot_furn.setdefault(s.parent_entity, []).append(s.slot_id)
        out = []
        for i, (x, y) in enumerate(chosen):
            for a in cfg.start_yaws_deg:
                yaw = math.radians(a)
                self._teleport(x, y, yaw)
                self.backend.settle(3)
                from rummagebench.sim.omnigibson.visibility import visible_entities_in_view

                seen = visible_entities_in_view(self.backend, cfg.seen_min_px)
                vis = sorted(sid for e in seen for sid in slot_furn.get(e, []))
                out.append(StartPose(f"{room}_p{i}_y{a:03d}",
                                     pose_from_xyyaw(x, y, self.z, yaw), vis))
        self.log(f"[probe] {room}: {len(pts)} map points, {len(free)} footprint-free at "
                 f"all headings, {len(chosen)} chosen -> {len(out)} start poses")
        self.room_starts[room] = out
        return out

    # ------------------------------------------------------------ furniture

    def probe_furniture(self, furn: Furniture, slots: list[Slot]) -> FurnitureProbe:
        """Anchor + OPEN + per-slot GRASP / reveal evidence for one furniture.

        Pass 1 (furniture closed): footprint-free anchors, OPEN from each.
        Pass 2 (open): probe object in each slot, GRASP + visibility from
        each anchor. The furniture's OPEN pose is an OPEN-feasible anchor;
        each slot gets a REVEAL pose where the probe is visible and GRASP is
        feasible (the agent may OPEN near and step back: the head camera
        only sees drawer contents from ~1 m).
        Pass 3: per slot, visibility open vs closed from the reveal pose and
        GRASP while closed from both poses — the causal-reveal evidence.
        """
        from rummagebench.sim.omnigibson import probe_support as ps

        t0 = time.time()
        fp = FurnitureProbe(furn.entity)
        try:
            fp.fillable_links = [n for n in ps.link_names(self.backend, furn.entity)
                                 if "fillable" in n]
            self.backend.reset()
            cands = anchor_candidates(furn, self.cfg)
            fp.anchors_tested = len(cands)
            free = [c for c in cands if self._footprint_free(*c)]
            fp.anchors_footprint_free = len(free)
            if not free:  # a measurement, not a failure
                fp.note = "no footprint-free interaction anchor"
                return fp
            needs_open = furn.openable and any(s.requires_open for s in slots)
            open_ok = []
            from rummagebench.sim.omnigibson.visibility import visible_entities_in_view

            px = []
            for i, c in enumerate(free):
                self._teleport(*c)
                ok = True if not needs_open else bool(self._check("OPEN", furn.entity).feasible)
                open_ok.append(ok)
                n = 0
                if ok and needs_open:  # the agent clicks the furniture to OPEN it
                    self.backend.settle(2)
                    n = visible_entities_in_view(self.backend, 1).get(furn.entity, 0)
                    fp.furniture_px_by_anchor[str(i)] = int(n)
                px.append(n)
            fp.open_feasible_from = [i for i, ok in enumerate(open_ok) if ok]
            clickable = [(not needs_open) or (open_ok[i] and px[i] >= self.cfg.click_min_px)
                         for i in range(len(free))]

            if furn.openable:
                self.backend.set_open(furn.entity, True)
                self.backend.settle(10)
            from rummagebench.authoring.tasks.navigation import FootprintChecker

            chk_open = FootprintChecker.from_backend(  # the agent steps back after OPEN
                self.backend, self.cfg.footprint_half_m, self.cfg.footprint_margin_m,
                skip=frozenset({self.probe}))
            # pass 2: the agent must CLICK the target and GRASP it from the
            # same pose, so an anchor counts for a slot only when GRASP is
            # feasible there AND the revealed probe object is visible
            grasp: dict[str, list[bool]] = {}
            vis: dict[str, list[Optional[float]]] = {}
            for s in slots:
                rec = self._place_probe(s)
                fp.slots[s.slot_id] = rec
                if not rec.get("placed"):
                    continue
                # data only (not a filter): does this anchor see the whole
                # opened drawer floor, i.e. any target position in it?
                region = None
                if s.slot_type == "drawer" and s.link:
                    self.backend._collision_body_cache = None
                    slab = drawer_floor_slab(
                        [(g.aabb[0], g.aabb[1]) for g in self.backend.collision_geometries()
                         if g.entity == s.parent_entity and g.link == s.link
                         and g.aabb is not None])
                    rec["open_floor_slab"] = slab
                    region = floor_region_corners(slab) if slab else []
                g, v, r = [], [], []
                for c in free:
                    self._teleport(*c)
                    ok = bool(self._check("GRASP", self.probe).feasible)
                    g.append(ok)
                    if ok:
                        self.backend.settle(3)
                        v.append(self._visibility(self.probe)["ratio"] or 0.0)
                    else:
                        v.append(None)
                    if region is None:
                        r.append(True)
                    elif ok and region:
                        fr = self.backend.capture_visual_frame()
                        r.append(points_in_view(fr.camera_intrinsics, fr.camera_extrinsics,
                                                fr.image_width, fr.image_height, region))
                    else:
                        r.append(False)
                grasp[s.slot_id], vis[s.slot_id] = g, v
                rec["grasp_from"] = [i for i, ok in enumerate(g) if ok]
                rec["visibility_by_anchor"] = {str(i): round(x, 3)
                                               for i, x in enumerate(v) if x is not None}
                if region is not None:
                    rec["floor_in_view_from"] = [i for i, ok in enumerate(r) if ok]

            thr = self.cfg.reveal_visibility_min
            pose_of = [pose_from_xyyaw(c[0], c[1], self.z, c[2]) for c in free]

            def serves(k, i):  # slot k can be revealed + grasped at anchor i
                return grasp[k][i] and (vis[k][i] or 0.0) >= thr

            # OPEN pose (furniture level): OPEN-feasible, clickable, and its
            # footprint still free once the furniture is open — Core's OPEN
            # does not check the opened links against the base, and a drawer
            # swung into the footprint (+ margin) blocks every later MOVE and
            # TURN (the agent is trapped); then one that also serves slots directly, then nearest
            room_after = [chk_open.free(*c) for c in free]
            fp.free_after_open_from = [i for i, ok in enumerate(room_after) if ok]

            def usable_open(i):
                return open_ok[i] and clickable[i] and room_after[i]

            def open_score(i):
                return (usable_open(i), sum(serves(k, i) for k in grasp), -i)
            order = sorted(range(len(free)), key=open_score, reverse=True)
            best = None
            for i in order[:self.cfg.reach_tries]:
                if not usable_open(i):
                    break
                steps = self._route_steps(self._chk, None, free[i], furn.room)
                if steps is not None:
                    best, fp.open_route_steps = i, steps
                    break
            if best is None:
                fp.note = ("no OPEN-feasible, clickable anchor that stays free once open "
                           "and is reachable by MOVE/TURN from the room's start poses")
                for s in slots:
                    fp.slots[s.slot_id]["unreachable"] = True
                return fp
            fp.chosen_anchor = pose_of[best]
            bx, by = free[best][0], free[best][1]
            for s in slots:
                rec = fp.slots[s.slot_id]
                rec["anchor_index"] = best
                rec["open_feasible"] = open_ok[best] if s.requires_open else None
                g = grasp.get(s.slot_id)
                rec["grasp_feasible"] = bool(g and any(g))
                # REVEAL pose (slot level): the agent may OPEN near and step
                # back to see + GRASP; prefer the OPEN pose itself, then the
                # most visible, then the closest to the OPEN pose
                reveal = None
                if g:
                    ok = [i for i in range(len(free)) if serves(s.slot_id, i)]
                    ok.sort(key=lambda i: (
                        i == best, round(vis[s.slot_id][i], 2),
                        -((free[i][0] - bx) ** 2 + (free[i][1] - by) ** 2)), reverse=True)
                    for i in ok[:self.cfg.reach_tries]:
                        # reachable from the OPEN pose in the OPENED world
                        steps = 0 if i == best else self._route_steps(
                            chk_open, free[best], free[i], furn.room)
                        if steps is not None:
                            reveal = i
                            rec["reveal_route_steps"] = steps
                            break
                rec["reveal_anchor_index"] = reveal
                rec["reveal_anchor"] = None if reveal is None else pose_of[reveal].to_anchor()
                if not rec.get("placed"):
                    continue
                # pass 3: re-place (the probe moved on); evidence from the
                # reveal pose (or the OPEN pose when nothing reveals the slot)
                view = free[reveal if reveal is not None else best]
                if furn.openable and not self.backend.is_open(furn.entity):
                    self.backend.set_open(furn.entity, True)
                    self.backend.settle(10)
                rec3 = self._place_probe(s)
                if not rec3.get("placed"):
                    rec["reveal_error"] = f"re-placement failed: {rec3}"
                    continue
                self._teleport(*view)
                self.backend.settle(3)
                rec["open_visibility"] = self._visibility(self.probe)
                self._save_view(s.slot_id, "open")
                reject_unconfirmed_reveal(rec, thr)
                if s.requires_open and furn.openable:
                    self.backend.set_open(furn.entity, False)
                    self.backend.settle(10)
                    rec["closed_relation_ok"] = ps.relation_holds(
                        self.backend, self.probe, s.parent_entity, s.relation)
                    rec["closed_containing_link"] = ps.containing_link(
                        self.backend, s.parent_entity, self.probe)
                    # closed: invisible from the reveal pose AND ungraspable
                    # from both poses the agent would use
                    closed_grasp = []
                    for i in sorted({best, reveal if reveal is not None else best}):
                        self._teleport(*free[i])
                        self.backend.settle(3)
                        verdict = self._check("GRASP", self.probe)
                        closed_grasp.append(bool(verdict.feasible))
                        if verdict.feasible:  # audit trail: how was it reachable?
                            rec["closed_grasp_details"] = _brief(verdict.details)
                    rec["closed_graspable"] = any(closed_grasp)
                    self._teleport(*view)
                    self.backend.settle(3)
                    rec["closed_visibility"] = self._visibility(self.probe)
                    self._save_view(s.slot_id, "closed")
        except Exception as e:
            logger.exception("probe of %s failed", furn.entity)
            fp.error = f"{type(e).__name__}: {e}"
        finally:
            fp.seconds = round(time.time() - t0, 1)
        return fp


def reject_unconfirmed_reveal(rec: dict, threshold: float) -> bool:
    """Pass 3 is the final word on a slot's reveal pose: when the re-measured
    visibility there is below the threshold (pass-2 single readings can be
    wrong), the reveal pose is withdrawn. Returns True if withdrawn."""
    if rec.get("reveal_anchor") is None:
        return False
    m = rec.get("open_visibility") or {}
    seen = 0.0 if m.get("ratio") is None else float(m["ratio"])
    if seen >= threshold:
        return False
    rec["reveal_rejected"] = {"anchor_index": rec.get("reveal_anchor_index"),
                              "pass3_visibility": m}
    rec["reveal_anchor"] = None
    rec["reveal_anchor_index"] = None
    return True


def _brief(details: Any, limit: int = 1500) -> Any:
    """JSON-safe, size-capped copy of a feasibility verdict's details."""
    import json

    text = json.dumps(details, default=str)
    return json.loads(text) if len(text) <= limit else text[:limit]


def slot_interaction(slot: Slot, fp: FurnitureProbe) -> Optional[SlotInteraction]:
    """Overlay entry for one slot from its furniture probe (None if the
    furniture has no usable anchor at all)."""
    if fp.chosen_anchor is None:
        return None
    rec = fp.slots.get(slot.slot_id, {})
    placed = bool(rec.get("placed"))
    reveal = rec.get("reveal_anchor")
    return SlotInteraction(
        reveal_anchor=None if reveal is None else Pose2.from_dict(reveal),
        navigation_anchor=fp.chosen_anchor,
        open_feasible=rec.get("open_feasible") if slot.requires_open else None,
        grasp_region_feasible=placed and bool(rec.get("grasp_feasible")),
        probed=True,
        note="" if placed else f"probe placement failed: "
                               f"{rec.get('place_error') or rec.get('containing_link')}",
        evidence={k: v for k, v in rec.items()},
    )
