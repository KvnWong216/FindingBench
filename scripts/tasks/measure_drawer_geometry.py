#!/usr/bin/env python3
"""Measure drawer slot geometry and static footprint obstacles from the
simulator's collision world (GPU) -> <build_root>/slot_geometry/<scene>.json,
which build_scene_slots.py merges into SceneSlots.

Per drawer slot, with every slot furniture closed:

  floor_top_z   top of the drawer link's largest horizontal slab
  inner_xy      world AABB of that slab (inner footprint)
  clearance_m   lowest underside of any OTHER collision body over the central
                footprint (slab shrunk by --edge-m), minus floor_top_z;
                null = nothing above (open top)

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python scripts/tasks/measure_drawer_geometry.py \
        --scene Benevolence_1_int
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import probe_embodiment as probe
import yaml
from _common import REPO, host_config, load_yaml

from rummagebench.authoring.tasks.probe import drawer_floor_slab
from rummagebench.authoring.tasks.slots import SceneSlots

MEASURE_VERSION = 1


def measure_drawer(bodies, entity: str, link: str, edge_m: float) -> dict:
    """bodies: [(entity, link, lo, hi)] world AABBs of every collision body."""
    own = [(lo, hi) for e, l, lo, hi in bodies if e == entity and l == link]
    slab = drawer_floor_slab(own)
    if slab is None:
        return {"ok": False, "reason": "no horizontal floor slab in the link's colliders",
                "n_link_bodies": len(own)}
    lo, hi = slab
    floor = float(hi[2])
    cx = (lo[0] + edge_m, hi[0] - edge_m)
    cy = (lo[1] + edge_m, hi[1] - edge_m)
    above = None
    blocker = None
    for e, l, blo, bhi in bodies:
        if e == entity and l == link:
            continue
        if blo[2] <= floor + 0.005:
            continue
        if bhi[0] <= cx[0] or blo[0] >= cx[1] or bhi[1] <= cy[0] or blo[1] >= cy[1]:
            continue
        if above is None or blo[2] < above:
            above, blocker = float(blo[2]), f"{e}:{l}"
    walls_top = max(float(h[2]) for _, h in own)
    return {"ok": True, "floor_top_z": round(floor, 4),
            "inner_xy": [[round(float(lo[0]), 4), round(float(lo[1]), 4)],
                         [round(float(hi[0]), 4), round(float(hi[1]), 4)]],
            "walls_top_z": round(walls_top, 4),
            "clearance_m": None if above is None else round(above - floor, 4),
            "blocker": blocker}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--robot", default="r1pro")
    ap.add_argument("--edge-m", type=float, default=0.03,
                    help="ignore rim bars within this distance of the slab edge")
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)

    host = host_config(args.host_config)
    ss = SceneSlots.load(host["build_root"] / "scenes" / f"{args.scene}.yaml")
    drawers = [s for s in ss.slots if s.slot_type == "drawer"]
    out_dir = host["build_root"] / "slot_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.scene}.json"
    from rummagebench.authoring.tasks.room_map import RoomMap
    room_map = RoomMap.load(host["behavior_assets_root"] / "scenes" / args.scene,
                            host["behavior_assets_root"] / "metadata" / "room_categories.txt")
    rooms = sorted({s.room for s in ss.slots})
    robot_doc = load_yaml(REPO / "configs" / "tasks" / "robots" / f"{args.robot}.yaml")
    cat, model = probe.DEFAULT_PROBE_OBJECT.split("/")
    doc = probe.probe_scenario(ss, rooms, room_map, robot_doc, cat, model)
    scen_path = out_dir / f"{args.scene}.scenario.yaml"
    scen_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    os.chdir(REPO)
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    scenario = load_scenario(scen_path)
    backend = OmniGibsonBackend(seed=0)
    backend._validate_spawn_clearance = lambda: None  # geometry only, robot pose irrelevant
    backend.setup(scenario)
    bodies = []
    for g in backend.collision_geometries():
        if g.aabb is None or g.entity in ("probe_object",) or g.entity.startswith("robot"):
            continue
        bodies.append((g.entity, g.link, list(g.aabb[0]), list(g.aabb[1])))
    # static footprint obstacles (every slot furniture closed), the same set
    # skills/move.py checks: lets the planner choose start poses that can
    # really reach the interaction poses (CPU plan_route)
    controlled = backend.robot_entity_names()
    entity_aabbs = {}
    for name in backend.entity_names():
        if name in controlled or name == "probe_object" or backend.describe_entity(name) is None:
            continue
        aabb = backend.entity_aabb(name)
        if aabb is not None:
            entity_aabbs[name] = [[round(float(v), 4) for v in aabb[0]],
                                  [round(float(v), 4) for v in aabb[1]]]
    rec = {"scene": args.scene, "version": MEASURE_VERSION, "edge_m": args.edge_m,
           "date": dt.date.today().isoformat(), "n_bodies": len(bodies), "slots": {},
           "entity_aabbs": entity_aabbs}
    for s in drawers:
        m = measure_drawer(bodies, s.parent_entity, s.link, args.edge_m)
        m["closed"] = not backend.is_open(s.parent_entity)
        m["metadata_floor_z"] = s.geometry.floor_z
        m["metadata_height"] = s.geometry.usable_extent[2]
        rec["slots"][s.slot_id] = m
        print(f"{s.slot_id}: {m}", flush=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"wrote {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # Kit teardown may segfault after the record is written


if __name__ == "__main__":
    raise SystemExit(main())
