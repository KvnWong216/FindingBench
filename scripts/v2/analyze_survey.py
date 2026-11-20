"""vla_difficulty_v2: CPU analysis of a scene survey -> candidate locations.

Flood-fills the floor the 0.8 m R1Pro footprint can reach from the survey
start (0.1 m grid; a cell is reachable when the footprint is collision-free
there at some yaw in 0/45/90/135 deg and it is 4-connected to the start under
that same yaw set), then lists, per room, furniture with a reachable
interaction pose (a ring pose 0.45-1.1 m off its AABB, facing it, footprint
free at that heading) and its candidate relations:

  on_top  -- fixed furniture with a top_surface support region between 0.3 m
             and 1.5 m high (robot arm workspace)
  inside  -- each native drawer interior (fillable volume) between 0.25 and
             1.2 m high

This is a prefilter. Placement, visibility and GRASP / PLACE feasibility are
certified in simulation by certify_locations.py.

    python scripts/v2/analyze_survey.py Beechwood_0_int Rs_int ...
Writes runs/vla_difficulty_v2/_survey/<scene>.candidates.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

from rummagebench.authoring.tasks.navigation import FootprintChecker
from rummagebench.evaluation.route_oracle import _ring_poses

ROOT = Path(__file__).resolve().parents[2]
SURVEY = ROOT / "runs/vla_difficulty_v2/_survey"
GRID = 0.1
YAWS = np.radians([0, 45, 90, 135])
SURFACE_Z = (0.30, 1.50)
DRAWER_Z = (0.25, 1.20)
SKIP = ("wall", "floor", "ceiling", "door", "window", "carpet", "rug", "light", "lamp",
        "picture", "mirror", "curtain", "blind", "radiator", "vent", "outlet", "switch",
        "fan", "stair", "railing", "toilet", "bathtub", "shower", "bed", "sofa", "chair",
        "stool", "towel_rack", "heater", "electric", "plant", "tree", "range_hood")


def checker_from(survey: dict) -> FootprintChecker:
    chk = FootprintChecker([], 0.40, 0.03)
    boxes = np.asarray(survey["footprint_boxes"], float)
    chk.lo, chk.hi = boxes[:, :2], boxes[:, 2:]
    lo, hi = chk.lo, chk.hi
    chk.corners = np.stack([np.stack([lo[:, 0], lo[:, 1]], 1), np.stack([hi[:, 0], lo[:, 1]], 1),
                            np.stack([hi[:, 0], hi[:, 1]], 1), np.stack([lo[:, 0], hi[:, 1]], 1)], 1)
    return chk


def reachable(chk: FootprintChecker, start, bounds):
    (x0, y0), (x1, y1) = bounds
    xs = np.arange(x0, x1 + GRID, GRID)
    ys = np.arange(y0, y1 + GRID, GRID)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    free = np.zeros(gx.shape + (len(YAWS),), bool)
    for k, yaw in enumerate(YAWS):
        free[..., k] = chk.free_many(gx.ravel(), gy.ravel(), np.full(gx.size, yaw)).reshape(gx.shape)
    si, sj = int(round((start[0] - x0) / GRID)), int(round((start[1] - y0) / GRID))
    seen = np.zeros(free.shape, bool)
    q = deque()
    for k in range(len(YAWS)):
        if free[si, sj, k]:
            seen[si, sj, k] = True
            q.append((si, sj, k))
    while q:
        i, j, k = q.popleft()
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, (k + dk) % len(YAWS)
            if 0 <= ni < free.shape[0] and 0 <= nj < free.shape[1] and free[ni, nj, nk] \
                    and not seen[ni, nj, nk]:
                seen[ni, nj, nk] = True
                q.append((ni, nj, nk))
    return xs, ys, seen.any(-1)


# the widened layouts keep their validated (official / portal) survey start
KEEP_SURVEY_START = {"Beechwood_0_int", "Rs_int", "house_double_floor_lower", "house_double_floor_upper"}


def largest_component_seed(chk: FootprintChecker, bounds, fallback, scene: str):
    """A seed in the largest floor region the footprint can reach: the
    automatically picked survey start may sit in a pocket enclosed by
    furniture. Components of 'free at some yaw' cells (4-connected, slightly
    optimistic); reachable() then re-floods exactly from the seed."""
    from scipy import ndimage
    (x0, y0), (x1, y1) = bounds
    xs = np.arange(x0, x1 + GRID, GRID)
    ys = np.arange(y0, y1 + GRID, GRID)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    import os
    from rummagebench.authoring.tasks.room_map import RoomMap
    root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
    rm = RoomMap.load(root / "scenes" / scene, root / "metadata" / "room_categories.txt")
    free = np.zeros(gx.shape, bool)
    for yaw in YAWS:
        free |= chk.free_many(gx.ravel(), gy.ravel(), np.full(gx.size, yaw)).reshape(gx.shape)
    # indoor cells only: outside the house the floor is empty and always wins
    indoor = np.array([[rm.room_at(float(x), float(y)) is not None for y in ys] for x in xs])
    free &= indoor
    lab, n = ndimage.label(free)
    if n == 0:
        return fallback
    sizes = ndimage.sum(free, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    cells = np.argwhere(lab == k)
    disc = [(i, j) for i, j in cells if chk.disc_free(float(xs[i]), float(ys[j]))]
    i, j = (disc or cells.tolist())[len(disc or cells) // 2]
    return [float(xs[i]), float(ys[j])]


def pick_starts(scene: str, chk: FootprintChecker, xs, ys, reach, per_room: int = 2,
                min_sep_m: float = 1.5) -> list[dict]:
    """Up to `per_room` start poses per reachable room: turn-in-place safe
    (footprint disc free), maximal clearance to obstacles, >= min_sep apart,
    facing the centroid of the room's reachable floor."""
    import os
    from rummagebench.authoring.tasks.room_map import RoomMap
    root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
    rm = RoomMap.load(root / "scenes" / scene, root / "metadata" / "room_categories.txt")
    cells: dict[str, list] = {}
    for i, j in zip(*np.nonzero(reach)):
        x, y = float(xs[i]), float(ys[j])
        room = rm.room_at(x, y)
        if room is None or not chk.disc_free(x, y):
            continue
        dx = np.maximum(np.maximum(chk.lo[:, 0] - x, x - chk.hi[:, 0]), 0)
        dy = np.maximum(np.maximum(chk.lo[:, 1] - y, y - chk.hi[:, 1]), 0)
        cells.setdefault(room, []).append((float(np.hypot(dx, dy).min()), x, y))
    starts = []
    for room, pts in sorted(cells.items()):
        cx = float(np.mean([p[1] for p in pts]))
        cy = float(np.mean([p[2] for p in pts]))
        chosen: list = []
        for clear, x, y in sorted(pts, reverse=True):
            if all(math.hypot(x - a, y - b) >= min_sep_m for _, a, b in chosen):
                chosen.append((clear, x, y))
            if len(chosen) == per_room:
                break
        for k, (clear, x, y) in enumerate(chosen):
            yaw = math.atan2(cy - y, cx - x) if math.hypot(cy - y, cx - x) > 0.3 else 0.0
            starts.append({"name": f"start_{room}_{k}", "room": room, "x": round(x, 2), "y": round(y, 2),
                           "yaw": round(yaw, 3), "clearance_m": round(clear, 2), "room_cells": len(pts)})
    return starts


def main(scenes: list[str]) -> int:
    for scene in scenes:
        survey = json.loads((SURVEY / f"{scene}.json").read_text())
        chk = checker_from(survey)
        ents = survey["entities"]
        boxes = [e["aabb"] for e in ents.values() if e.get("aabb")]
        lo = np.min([b[0][:2] for b in boxes], 0) - 0.5
        hi = np.max([b[1][:2] for b in boxes], 0) + 0.5
        seed = survey["start"][:2]
        if scene not in KEEP_SURVEY_START:
            seed = largest_component_seed(chk, (lo, hi), seed, scene)
        xs, ys, reach = reachable(chk, seed, (lo, hi))

        def reach_at(x, y):
            i, j = int(round((x - xs[0]) / GRID)), int(round((y - ys[0]) / GRID))
            return 0 <= i < len(xs) and 0 <= j < len(ys) and bool(reach[i, j])

        out = {"scene": scene, "start": survey["start"], "reachable_m2": round(float(reach.sum()) * GRID * GRID, 1),
               "candidates": []}
        rooms_reached: dict[str, int] = {}
        for name, e in sorted(ents.items()):
            if not e.get("aabb") or any(s in e["category"] for s in SKIP):
                continue
            box = e["aabb"]
            poses = [p for p in _ring_poses(box, offsets=tuple(np.arange(0.45, 1.11, 0.05)))
                     if reach_at(p[0], p[1]) and chk.free(p[0], p[1], p[2])]
            if not poses:
                continue
            room = e.get("room_at_center") or (e.get("in_rooms") or ["?"])[0]
            rooms_reached[room] = rooms_reached.get(room, 0) + 1
            near = min(poses, key=lambda p: p[3])
            rel = []
            sup = e.get("support")
            if sup and sup["kind"] == "top_surface" and SURFACE_Z[0] <= sup["hi"][2] <= SURFACE_Z[1]:
                area = (sup["hi"][0] - sup["lo"][0]) * (sup["hi"][1] - sup["lo"][1])
                if area >= 0.04:
                    rel.append({"relation": "on_top", "z": round(sup["hi"][2], 2), "area_m2": round(area, 2)})
            for it in e.get("interiors") or []:
                if DRAWER_Z[0] <= it["hi"][2] <= DRAWER_Z[1]:
                    rel.append({"relation": "inside", "link": it["link"], "z": round(it["hi"][2], 2)})
            if rel:
                out["candidates"].append({"entity": name, "category": e["category"], "room": room,
                                          "aabb": box, "relations": rel, "n_poses": len(poses),
                                          "nearest_pose_offset_m": round(near[3], 2)})
        out["rooms_with_reachable_furniture"] = rooms_reached
        out["starts"] = pick_starts(scene, chk, xs, ys, reach)
        (SURVEY / f"{scene}.candidates.json").write_text(json.dumps(out, indent=1) + "\n")
        n_top = sum(any(r["relation"] == "on_top" for r in c["relations"]) for c in out["candidates"])
        n_in = sum(sum(r["relation"] == "inside" for r in c["relations"]) for c in out["candidates"])
        print(f"== {scene}: reachable {out['reachable_m2']} m2, start room "
              f"{next((c for c in [survey.get('start_room')] if c), '?')}; surfaces {n_top}, drawers {n_in}")
        for c in out["candidates"]:
            print(f"   {c['room']:16s} {c['entity']:36s} "
                  + ", ".join(f"{r['relation']}{':' + r['link'] if 'link' in r else ''}@{r['z']}" for r in c["relations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
