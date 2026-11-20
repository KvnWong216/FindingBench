"""vla_difficulty_v2: CPU doorway-widening proposals for more BEHAVIOR scenes.

Same edit vocabulary as the teammate's clearance manifests
(src/rummagebench/sim/omnigibson/clearance.py: wall_warps + removed_objects),
but the native wall / door collider inventory is read on the CPU from the
official scene json and the (decrypted) object USDs, so no GPU scene load is
needed for authoring. The GPU still has the last word:
scripts/environment/check_clearance.py replays every opening with the public
MOVE / TURN skills, and the v2 survey re-measures reachability.

Differences to scripts/prepare_clearance.py (which stays unchanged):
  * a doorway next to a corner is widened towards the side with more wall
    (the symmetric version sends it to manual review);
  * 1.4 m preferred, 1.2 m accepted (build/behavior/clearance_audit.json policy);
  * a warp that would also move vertices of another wall run of the same
    entity (different wall plane) goes to manual review instead;
  * door leaves inside an interior opening are listed in removed_objects.

Before / after reachability uses the survey's furniture obstacle boxes and
the 0.8 m footprint flood fill of analyze_survey.py (an estimate; the GPU
survey after enabling the manifest is authoritative).

    python scripts/v2/propose_clearance.py Wainscott_1_int Beechwood_1_int ...
    python scripts/v2/propose_clearance.py --plot Wainscott_1_int

Writes runs/vla_difficulty_v2/_clearance/<scene>.proposal.json (+ .png).
Enabling = copying a reviewed proposal to build/vla_difficulty_v2/clearance/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_survey as A  # noqa: E402
from rummagebench.authoring.tasks.navigation import Z_HIGH, Z_LOW  # noqa: E402
from rummagebench.sim.omnigibson.clearance import warp_points  # noqa: E402

OUT = ROOT / "runs/vla_difficulty_v2/_clearance"
CACHE = ROOT / "build/vla_difficulty_v2/_decrypted"   # local only, not distributable
DATA = Path(os.environ["OMNIGIBSON_DATA_PATH"])
ASSETS = DATA / "behavior-1k-assets"
PREFERRED, MINIMUM = 1.4, 1.2      # clearance_audit.json policy
PASSABLE = 0.86                    # 0.8 m footprint + 2 x 0.03 m margin
KEEP = 0.15                        # wall stub kept at each corner


# ---------------------------------------------------------------- inventory
def _colliders(category: str, model: str) -> list[np.ndarray]:
    from cryptography.fernet import Fernet
    from pxr import Usd, UsdGeom, UsdPhysics
    path = CACHE / f"{category}_{model}.usd"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        enc = ASSETS / "objects" / category / model / "usd" / f"{model}.encrypted.usd"
        path.write_bytes(Fernet((DATA / "omnigibson.key").read_bytes()).decrypt(enc.read_bytes()))
    stage = Usd.Stage.Open(str(path))
    hulls = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            m = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), float)
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            hulls.append(pts @ m[:3, :3] + m[3, :3])
    return hulls


def inventory(scene: str) -> list[dict]:
    """Walls and doors with their native colliders in world coordinates."""
    from scipy.spatial.transform import Rotation
    doc = json.loads((ASSETS / "scenes" / scene / "json" / f"{scene}_best.json").read_text())
    reg = doc["state"]["registry"]["object_registry"]
    rows = []
    for name, info in doc["objects_info"]["init_info"].items():
        args = info["args"]
        if args["category"] not in ("walls", "door"):
            continue
        root = reg[name]["root_link"]
        rot, pos = Rotation.from_quat(root["ori"]), np.asarray(root["pos"], float)
        scale = np.asarray(args.get("scale") or [1, 1, 1], float)
        rows.append({"name": name, "category": args["category"],
                     "hulls": [rot.apply(h * scale) + pos for h in _colliders(args["category"], args["model"])]})
    return rows


# ---------------------------------------------------------------- proposals
def _openings(hulls: list[np.ndarray]):
    """Axis-aligned wall runs of one entity -> (axis, plane, lo, hi, [(a, b, left, right)])."""
    boxes = np.array([[h.min(0), h.max(0)] for h in hulls])
    floor = boxes[:, 0, 2].min()
    runs = []
    for axis in (0, 1):
        normal = 1 - axis
        planes: dict[float, list] = {}
        for lo, hi in boxes:
            if hi[normal] - lo[normal] > 0.5 or hi[axis] - lo[axis] < hi[normal] - lo[normal]:
                continue   # not a wall piece along this axis
            if not (lo[2] < floor + 0.5 and hi[2] > floor + 0.05):
                continue   # lintel / window head: no floor contact
            c = round(float((lo[normal] + hi[normal]) / 2) / 0.1) * 0.1
            planes.setdefault(c, []).append((lo, hi))
        for _, pieces in planes.items():
            iv = sorted((float(lo[axis]), float(hi[axis])) for lo, hi in pieces)
            merged: list[list[float]] = []
            for a, b in iv:
                if merged and a <= merged[-1][1] + 0.02:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            plane = float(np.mean([(lo[normal] + hi[normal]) / 2 for lo, hi in pieces]))
            gaps = [(l[1], r[0], l, r) for l, r in zip(merged, merged[1:])]
            runs.append((axis, plane, gaps))
    return runs


def _widen(a: float, b: float, left, right):
    """New [lo, hi] for opening (a, b) inside walls left=[L0,a], right=[b,R1];
    symmetric when possible, else shifted towards the longer wall."""
    for width in (PREFERRED, MINIMUM):
        lo_lim, hi_lim = left[0] + KEEP, right[1] - KEEP
        if hi_lim - lo_lim < width:
            continue
        c = (a + b) / 2
        na, nb = c - width / 2, c + width / 2
        if na < lo_lim:
            na, nb = lo_lim, lo_lim + width
        if nb > hi_lim:
            na, nb = hi_lim - width, hi_lim
        na, nb = min(na, a), max(nb, b)       # never narrow an opening
        # teammate's local anchors (mid-wall) when they fit, else the wall ends
        anchor_lo = (left[0] + a) / 2 if na > (left[0] + a) / 2 + 0.05 else left[0] + 0.02
        anchor_hi = (b + right[1]) / 2 if nb < (b + right[1]) / 2 - 0.05 else right[1] - 0.02
        if not (anchor_lo < na and nb < anchor_hi):
            continue
        return width, [anchor_lo, a, b, anchor_hi], [anchor_lo, na, nb, anchor_hi]
    return None


def _floor_walls(inv) -> np.ndarray:
    """2D boxes of every wall collider touching the floor band (all entities)."""
    out = []
    for obj in inv:
        if obj["category"] != "walls":
            continue
        for h in obj["hulls"]:
            lo, hi = h.min(0), h.max(0)
            if lo[2] < 0.5 and hi[2] > 0.05:
                out.append([lo[0], lo[1], hi[0], hi[1]])
    return np.asarray(out, float)


def _is_interior_doorway(axis, plane, a, b, walls, room_map) -> str | None:
    """None when (a, b) on this plane is a free, interior doorway; else why not."""
    normal = 1 - axis
    lo = np.empty(2); hi = np.empty(2)
    lo[axis], hi[axis] = a + 0.05, b - 0.05
    lo[normal], hi[normal] = plane - 0.02, plane + 0.02
    covered = (walls[:, 0] < hi[0]) & (walls[:, 2] > lo[0]) & (walls[:, 1] < hi[1]) & (walls[:, 3] > lo[1])
    if covered.any():
        return "gap filled by another wall entity"
    sides = []
    for sign in (-1, 1):
        q = [0.0, 0.0]
        q[axis], q[normal] = (a + b) / 2, plane + sign * 0.5
        sides.append(room_map.room_at(*q))
    if None in sides:
        return "exterior opening"
    return None


def propose(scene: str, inv: list[dict], room_map) -> dict:
    warps, review, passable = [], [], []
    walls = _floor_walls(inv)
    for obj in inv:
        if obj["category"] != "walls" or not obj["hulls"]:
            continue
        for axis, plane, gaps in _openings(obj["hulls"]):
            taken = -np.inf   # end of the previous warp on this wall run: ranges must not overlap
            for a, b, left, right in gaps:
                left = [max(left[0], taken), left[1]]
                gap = b - a
                if not 0.55 < gap < 2.5 or _is_interior_doorway(axis, plane, a, b, walls, room_map):
                    continue
                item = {"entity": obj["name"], "axis": axis, "wall_plane": plane, "opening": [a, b],
                        "measured_width_m": gap}
                if gap >= PREFERRED:
                    passable.append(item)
                    continue
                w = _widen(a, b, left, right) if left[0] < a else None
                if w is None:
                    review.append({**item, "reason": "wall too short on both sides for >= 1.2 m"})
                    continue
                width, source, target = w
                # the warp acts on every vertex of the entity inside [source0, source3]
                normal = 1 - axis
                touched = []
                for h in obj["hulls"]:
                    moved = warp_points(h, axis, source, target)
                    if not np.allclose(moved, h) and abs((h[:, normal].min() + h[:, normal].max()) / 2 - plane) > 0.3:
                        touched.append([float(h[:, normal].min()), float(h[:, normal].max())])
                if touched:
                    review.append({**item, "reason": "warp would also move another wall run of this entity",
                                   "other_runs_normal_extent": touched})
                    continue
                taken = source[-1]
                warps.append({"entity": obj["name"], "axis": axis, "source": source, "target": target,
                              "measured_width_m": gap, "requested_width_m": width, "wall_plane": plane})
    # door leaves standing in an interior opening (widened or already wide)
    openings = [(w["axis"], w["wall_plane"], w["target"][1], w["target"][2]) for w in warps] + \
               [(p["axis"], p["wall_plane"], *p["opening"]) for p in passable] + \
               [(r["axis"], r["wall_plane"], *r["opening"]) for r in review]
    removed = []
    for obj in inv:
        if obj["category"] != "door":
            continue
        pts = np.vstack(obj["hulls"]) if obj["hulls"] else None
        if pts is None:
            continue
        c = (pts.min(0) + pts.max(0)) / 2
        for axis, plane, a, b in openings:
            if a - 0.1 <= c[axis] <= b + 0.1 and abs(c[1 - axis] - plane) < 0.6:
                removed.append(obj["name"])
                break
    return {"scene_model": scene, "status": "cpu_proposal_not_runtime_validated",
            "wall_warps": warps, "translations": [], "removed_objects": sorted(set(removed)),
            "manual_review": review, "already_wide_openings": passable,
            "note": "Generated by scripts/v2/propose_clearance.py from native colliders (CPU). "
                    "Interior openings only (both sides in a room of the official room map). "
                    "Validate with scripts/environment/check_clearance.py before use."}


# ---------------------------------------------------------------- estimate
def _nav_boxes(inv, warps=(), removed=()):
    out = []
    for obj in inv:
        if obj["name"] in removed:
            continue
        for h in obj["hulls"]:
            for w in warps:
                if w["entity"] == obj["name"]:
                    h = warp_points(h, w["axis"], w["source"], w["target"])
            lo, hi = h.min(0), h.max(0)
            if lo[2] > Z_HIGH or hi[2] < Z_LOW:
                continue
            out.append([lo[0], lo[1], hi[0], hi[1]])
    return out


def _furniture_boxes(survey: dict) -> list:
    """Survey obstacle boxes that belong to a non-architecture entity."""
    ents = [e for e in survey["entities"].values()
            if e.get("aabb") and e["category"] not in ("walls", "floors", "ceilings", "door")]
    keep = []
    for b in np.asarray(survey["footprint_boxes"], float):
        c = (b[:2] + b[2:]) / 2
        for e in ents:
            lo, hi = np.array(e["aabb"][0][:2]), np.array(e["aabb"][1][:2])
            if (c >= lo - .05).all() and (c <= hi + .05).all() and (b[:2] >= lo - .1).all() and (b[2:] <= hi + .1).all():
                keep.append(b.tolist())
                break
    return keep


def estimate(scene: str, survey: dict, boxes: list, room_map) -> dict:
    s = dict(survey, footprint_boxes=boxes)
    chk = A.checker_from(s)
    arr = np.asarray(boxes, float)
    bounds = ((arr[:, 0].min(), arr[:, 1].min()), (arr[:, 2].max(), arr[:, 3].max()))
    seed = A.largest_component_seed(chk, bounds, survey["start"][:2], scene)
    xs, ys, reach = A.reachable(chk, seed, bounds)
    rooms: dict[str, int] = {}
    for i, j in np.argwhere(reach):
        r = room_map.room_at(float(xs[i]), float(ys[j]))
        if r:
            rooms[r] = rooms.get(r, 0) + 1
    rooms_m2 = {r: round(c * A.GRID ** 2, 1) for r, c in sorted(rooms.items()) if c * A.GRID ** 2 >= 1.5}
    return {"indoor_m2": round(sum(rooms.values()) * A.GRID ** 2, 1), "rooms_m2": rooms_m2,
            "_grid": (xs, ys, reach)}


def plot(scene, inv, prop, furniture, est, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 13))
    xs, ys, reach = est["_grid"]
    ax.pcolormesh(xs, ys, reach.T, cmap="Greens", alpha=.35, shading="nearest")
    for b in furniture:
        ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fc="tab:blue", alpha=.4))
    warped = {w["entity"] for w in prop["wall_warps"]}
    for obj in inv:
        for h in obj["hulls"]:
            lo, hi = h.min(0), h.max(0)
            if lo[2] > Z_HIGH or hi[2] < Z_LOW:
                continue
            if obj["category"] == "door":
                col = "red" if obj["name"] in prop["removed_objects"] else "orange"
            else:
                col = "purple" if obj["name"] in warped else "black"
            ax.add_patch(plt.Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], fc=col, alpha=.6))
    for w in prop["wall_warps"]:
        ax.plot(*([[w["target"][1], w["target"][2]], [w["wall_plane"]] * 2][::1 if w["axis"] == 0 else -1]),
                color="lime", lw=5)
    for r in prop["manual_review"]:
        if "opening" in r:
            ax.plot(*([r["opening"], [r["wall_plane"]] * 2][::1 if r["axis"] == 0 else -1]), color="m", lw=3, ls=":")
    ax.set_aspect("equal")
    ax.grid(alpha=.3)
    ax.set_title(f"{scene}: green=reachable after edits, lime=widened, magenta dotted=review, "
                 f"red=removed door, purple=warped wall")
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenes", nargs="+")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    from rummagebench.authoring.tasks.room_map import RoomMap
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in a.scenes:
        inv = inventory(scene)
        rm = RoomMap.load(ASSETS / "scenes" / scene, ASSETS / "metadata" / "room_categories.txt")
        prop = propose(scene, inv, rm)
        survey_path = ROOT / "runs/vla_difficulty_v2/_survey" / f"{scene}.json"
        survey = json.loads(survey_path.read_text()) if survey_path.exists() else {}
        if survey.get("footprint_boxes"):
            furniture = _furniture_boxes(survey)
            before = estimate(scene, survey, furniture + _nav_boxes(inv), rm)
            after = estimate(scene, survey, furniture + _nav_boxes(inv, prop["wall_warps"], prop["removed_objects"]), rm)
            prop["estimate"] = {"note": "CPU flood fill, survey furniture boxes + native wall colliders",
                                "before": {k: v for k, v in before.items() if k != "_grid"},
                                "after": {k: v for k, v in after.items() if k != "_grid"}}
            if a.plot:
                plot(scene, inv, prop, furniture, after, OUT / f"{scene}.png")
        else:
            prop["estimate"] = {"note": "no survey furniture boxes for this scene; run survey_scene.py first"}
        (OUT / f"{scene}.proposal.json").write_text(json.dumps(prop, indent=1) + "\n")
        print(json.dumps({"scene": scene, "warps": len(prop["wall_warps"]), "review": len(prop["manual_review"]),
                          "removed": len(prop["removed_objects"]), **prop["estimate"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
