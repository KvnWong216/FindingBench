#!/usr/bin/env python3
"""Offline scene layout dump from the dataset scene JSON (no simulator)."""
import json
import sys

d = json.load(open(sys.argv[1]))
info = d["objects_info"]["init_info"]
rows = []
for name, entry in info.items():
    args = entry.get("args", {})
    cat = args.get("category", "?")
    rooms = args.get("in_rooms", [])
    pos = args.get("position")
    rows.append((name, cat, rooms, pos))

print(f"total objects: {len(rows)}")
print("\n== kitchen objects ==")
for name, cat, rooms, pos in sorted(rows):
    if any("kitchen" in (r or "") for r in rooms):
        p = [round(float(v), 2) for v in pos[:3]] if pos else None
        print(f"{name:35s} cat={cat:25s} pos={p} rooms={rooms}")

print("\n== living_room objects ==")
for name, cat, rooms, pos in sorted(rows):
    if any("living" in (r or "") for r in rooms):
        p = [round(float(v), 2) for v in pos[:3]] if pos else None
        print(f"{name:35s} cat={cat:25s} pos={p} rooms={rooms}")
