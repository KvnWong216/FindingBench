"""Plot the R1Pro base reachability map of a scene from a scene probe
(scripts/probe_vla_scene.py): MOVE obstacle boxes, the region reachable from
the start pose for the 0.40 m half-extent footprint (axis-aligned headings),
and labelled furniture. CPU only.

    python scripts/plot_vla_reachability.py runs/vla_difficulty_v1/_probe/scene_probe.json \
        --start 0.85 -7.70 --out docs/evidence/vla_difficulty_v1/beechwood_reachability.png
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rummagebench.authoring.tasks.navigation import FootprintChecker


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("probe", type=Path)
    p.add_argument("--start", nargs=2, type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", nargs="*", default=None, help="entity substrings to label")
    a = p.parse_args()
    d = json.loads(a.probe.read_text())
    obs = [o for o in d["obstacles"] if not (o["lo"][2] > 0.5 or o["hi"][2] < 0.05)]
    chk = FootprintChecker([(o["lo"], o["hi"]) for o in obs], 0.40, 0.03)
    res = 0.05
    xs = np.arange(-7, 8, res); ys = np.arange(-12, 4, res)
    X, Y = np.meshgrid(xs, ys)
    free = (chk.free_many(X.ravel(), Y.ravel(), np.zeros(X.size))
            | chk.free_many(X.ravel(), Y.ravel(), np.full(X.size, np.pi / 2))).reshape(X.shape)
    si, sj = int(round((a.start[1] - ys[0]) / res)), int(round((a.start[0] - xs[0]) / res))
    seen = np.zeros_like(free)
    q = collections.deque([(si, sj)]); seen[si, sj] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = i + di, j + dj
            if 0 <= u < free.shape[0] and 0 <= v < free.shape[1] and free[u, v] and not seen[u, v]:
                seen[u, v] = True; q.append((u, v))
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.imshow(np.where(seen, 1.0, np.where(free, 0.5, 0.0)), origin="lower", cmap="Greens",
              extent=[xs[0], xs[-1], ys[0], ys[-1]], alpha=0.5, vmin=0, vmax=1.2)
    keys = a.label or ["table", "counter", "cabinet", "desk", "door", "fridge", "sofa",
                       "dishwasher", "bookcase", "chair", "stove", "oven"]
    for o in obs:
        lo, hi = o["lo"], o["hi"]
        if hi[0] - lo[0] > 8 or hi[1] - lo[1] > 8:
            continue  # whole-house shells (walls / floors) would hide everything
        door = "door" in o["entity"] and "cabinet" not in o["entity"]
        ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False,
                                   ec="red" if door else "k", lw=1.2 if door else 0.5))
        if any(k in o["entity"] for k in keys):
            ax.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, o["entity"], fontsize=6,
                    ha="center", va="center", color="red" if door else "navy")
    ax.plot(*a.start, "b*", ms=14)
    ax.set_title("R1Pro base reachability (0.40 m half extent + 0.03 m margin) from the start pose\n"
                 "dark green = reachable, light = footprint-free but disconnected, red = doors")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"reachable area {seen.sum() * res * res:.1f} m^2 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
