#!/usr/bin/env python3
"""Phase 2 authoring helper: pick robot anchors for a scenario.

Loads the scene with an R1Pro, samples traversable points in the requested
rooms, teleports the robot to each candidate, settles physics, and records
the stable pose. Also lists kitchen openable containers and receptacles with
poses. Output: JSON on stdout (or --out file).

    python scripts/pick_anchors.py --scene Beechwood_0_int \
        --rooms living_room kitchen --out runs/anchors_Beechwood_0_int.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import os

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")

    import numpy as np
    import omnigibson as og
    from omnigibson.envs import Environment
    from omnigibson.object_states import Open

    env = Environment(
        configs={
            "env": {"action_frequency": 30, "physics_frequency": 120},
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": args.scene,
                "load_room_types": args.rooms,
                "include_robots": False,
            },
            "robots": [
                {
                    "model": "r1pro",
                    "name": "robot_0",
                    "obs_modalities": [],
                    "action_type": "continuous",
                    "grasping_mode": "sticky",
                }
            ],
            "objects": [],
            "task": {"type": "DummyTask"},
        }
    )
    scene = env.scene
    robot = scene.robots[0]
    seg_map = scene.seg_map
    ins_to_sem = {
        ins: sem
        for sem, instances in seg_map.room_sem_name_to_ins_name.items()
        for ins in (instances if isinstance(instances, list) else [instances])
    }

    def npy(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    def to_list(t):
        return [round(float(v), 4) for v in npy(t).reshape(-1)]

    # spawn the robot at the center of the first room to get a stable base height
    import torch as th

    def sample_room_point(room: str):
        """Sample a traversable world-frame (x, y) in a room.

        Bypasses seg_map.get_random_point_by_room_type, which is broken in
        3.9.3 (th.tensor over a th.where tuple), by sampling room_sem_map
        directly with the same map_to_world convention.
        """
        sem_id = seg_map.room_sem_name_to_sem_id[room]
        rows, cols = th.where(seg_map.room_sem_map == sem_id)
        k = int(th.randint(rows.shape[0], (1,)).item())
        point_map = th.stack([rows[k].float(), cols[k].float()])
        world = seg_map.map_to_world(point_map)
        z = seg_map.floor_heights[0]
        return [float(world[0]), float(world[1]), float(z)]

    first_room = args.rooms[0]
    spawn = sample_room_point(first_room)
    robot.set_position_orientation(position=spawn, orientation=[0, 0, 0, 1], frame="world")
    for _ in range(30):
        og.sim.step()
    pos, quat = robot.get_position_orientation(frame="world")
    stable_z = float(npy(pos)[2])
    print(f"stable base z after settle: {stable_z:.3f}", file=sys.stderr)

    # candidate anchors: sampled points per room with the stable z
    anchors = {}
    for room in args.rooms:
        cands = []
        for _ in range(6):
            p = sample_room_point(room)
            cands.append({"position": [p[0], p[1], stable_z], "orientation": [0, 0, 0, 1]})
        anchors[room] = dict(cands[0])
        anchors[room]["candidates"] = cands

    # verify each anchor: teleport, settle, check it holds
    for name, a in anchors.items():
        robot.set_position_orientation(
            position=a["position"], orientation=a["orientation"], frame="world"
        )
        for _ in range(30):
            og.sim.step()
        pos, quat = robot.get_position_orientation(frame="world")
        a["verified_position"] = to_list(pos)
        a["verified_orientation"] = to_list(quat)
        a["drift"] = round(
            float(np.linalg.norm(npy(pos) - np.asarray(a["position"]))), 3
        )

    # kitchen storage furniture: openable containers with poses and rooms
    containers = []
    receptacles = []
    for obj in scene.objects:
        rooms = list(getattr(obj, "in_rooms", None) or [])
        room_types = sorted({ins_to_sem.get(r, "?") for r in rooms})
        pos, quat = obj.get_position_orientation()
        entry = {
            "name": obj.name,
            "category": obj.category,
            "model": getattr(obj, "model", None),
            "rooms": rooms,
            "room_types": room_types,
            "position": to_list(pos),
            "openable": "openable" in (obj.abilities or {}),
            "fixed_base": bool(getattr(obj, "fixed_base", False)),
        }
        if not entry["openable"]:
            continue
        has_open = obj.states.get(Open) is not None
        if has_open:
            containers.append(entry)
        abilities = obj.abilities or {}
        if "fillable" in abilities or "inside" in abilities:
            receptacles.append(entry)

    result = {
        "scene": args.scene,
        "stable_base_z": round(stable_z, 4),
        "anchors": anchors,
        "openable_containers": containers,
        "receptacles": receptacles,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"saved -> {args.out}", file=sys.stderr)
    print(text)

    og.shutdown()
    # Kit teardown may segfault after everything completed; force a clean exit
    os._exit(0)


if __name__ == "__main__":
    main()
