"""One-shot scene probe for vla_difficulty_v1 location choices (GPU, one load).

Loads a compiled variant, resets, and dumps to <out>/scene_probe.json:
  * links (+ world AABB, meta/fillable flags) of candidate containers, closed
    and after set_open(True)
  * the MOVE obstacle set as FootprintChecker sees it (inflated XY boxes),
    so routes / reachable rooms can be analysed on CPU
  * AABB, fixed_base and room of candidate furniture

    python scripts/probe_vla_scene.py build/vla_difficulty_v1/T01_knife_fetch__L2_S1/scenario.yaml \
        --containers bottom_cabinet_rvpunw_0 top_cabinet_dmwxyl_0 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _aabb(backend, prim):
    from rummagebench.sim.omnigibson.usd_collision import compute_link_aabb
    try:
        box = compute_link_aabb(prim)
        return None if box is None else [[round(float(v), 3) for v in box[0]],
                                         [round(float(v), 3) for v in box[1]]]
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", type=Path)
    p.add_argument("--containers", nargs="*", default=[])
    p.add_argument("--furniture", nargs="*", default=[])
    p.add_argument("--out", type=Path, default=Path("runs/vla_difficulty_v1/_probe"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"scenario": str(a.scenario)}
    try:
        from rummagebench.authoring.tasks.navigation import FootprintChecker
        from rummagebench.environment.env import InteractiveSearchEnv
        from rummagebench.sim.omnigibson.backend import resolve_object

        env = InteractiveSearchEnv(a.scenario, mode="agent")
        env.reset()
        b = env._backend
        chk = FootprintChecker.from_backend(b, 0.40, 0.03)
        names = []
        for name in b.entity_names():
            if name in b.robot_entity_names():
                continue
            from rummagebench.skills.move import navigation_aabbs
            for box in navigation_aabbs(b, name):
                names.append({"entity": name, "lo": [round(float(v), 3) for v in box[0]],
                              "hi": [round(float(v), 3) for v in box[1]]})
        rep["obstacles"] = names
        rep["footprint_boxes_xy_inflated_0.03"] = [[*map(float, lo), *map(float, hi)]
                                                    for lo, hi in zip(chk.lo, chk.hi)]
        rep["containers"] = {}
        for c in a.containers:
            obj = resolve_object(b._env.scene, c)
            info = {"fixed_base": bool(getattr(obj, "fixed_base", False)),
                    "entity_aabb": b.entity_aabb(c), "is_open": b.is_open(c),
                    "links_closed": {n: _aabb(b, l) for n, l in (obj.links or {}).items()}}
            try:
                info["set_open"] = bool(b.set_open(c, True))
                b.settle(10)
                info["links_open"] = {n: _aabb(b, l) for n, l in (obj.links or {}).items()}
                info["entity_aabb_open"] = b.entity_aabb(c)
                b.set_open(c, False)
                b.settle(10)
            except Exception as e:  # noqa: BLE001
                info["open_error"] = f"{type(e).__name__}: {e}"
            rep["containers"][c] = info
        rep["furniture"] = {}
        for f in a.furniture:
            obj = resolve_object(b._env.scene, f)
            reg = b.receptacle_region(f)
            rep["furniture"][f] = {"fixed_base": bool(getattr(obj, "fixed_base", False)),
                                   "entity_aabb": b.entity_aabb(f),
                                   "region": None if reg is None else
                                   {"kind": reg.kind, "lo": list(reg.lo), "hi": list(reg.hi)}}
        rep["ok"] = True
    except BaseException as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
    (a.out / "scene_probe.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k in ("ok", "error")}), flush=True)
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
