"""Build SceneSlots from the BEHAVIOR scene JSON -> <build_root>/scenes/<scene>.yaml.

    PYTHONPATH=src python scripts/tasks/build_scene_slots.py [--scenes A B ...]

Drawer geometry measured by measure_drawer_geometry.py
(<build_root>/slot_geometry/<scene>.json) replaces the metadata AABB when present.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _common import ASSETS, host_config, load_yaml

from rummagebench.authoring.tasks.slots import apply_measured_geometry, build_scene_slots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--rules", default=str(ASSETS / "slot_rules_v1.yaml"))
    ap.add_argument("--host-config")
    ap.add_argument("--slot-geometry-dir",
                    help="measured drawer geometry (default: <build_root>/slot_geometry)")
    args = ap.parse_args(argv)

    host = host_config(args.host_config)
    rules = load_yaml(args.rules)
    scenes = args.scenes or rules["scenes"]
    out_dir = host["build_root"] / "scenes"
    for scene in scenes:
        ss = build_scene_slots(scene, host["behavior_assets_root"], rules)
        geo_dir = Path(args.slot_geometry_dir or host["build_root"] / "slot_geometry")
        geo = geo_dir / f"{scene}.json"
        if geo.exists():
            doc = json.loads(geo.read_text(encoding="utf-8"))
            counts = apply_measured_geometry(ss, doc["slots"], float(rules["wall_margin_m"]))
            ss.footprint_obstacles = dict(doc.get("entity_aabbs") or {})
            counts["footprint_obstacles"] = len(ss.footprint_obstacles)
            ss.source["slot_geometry"] = {"path": str(geo), "version": doc.get("version"),
                                          **counts}
            print(f"{scene}: measured drawer geometry {counts}")
        else:
            print(f"{scene}: no measured drawer geometry ({geo}) -> metadata AABB")
        ss.save(out_dir / f"{scene}.yaml")
        per_room = Counter((s.room, s.slot_type) for s in ss.slots)
        print(f"{scene}: {len(ss.slots)} slots, {len(ss.furniture)} furniture, "
              f"{len(ss.skipped)} skipped")
        for (room, st), n in sorted(per_room.items()):
            print(f"    {room:22s} {st:18s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
